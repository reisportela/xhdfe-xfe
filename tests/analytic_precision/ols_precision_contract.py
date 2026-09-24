"""Analytical guards for OLS offsets, redundant FEs, VCE, IV and failed-fit state."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module',type=Path,required=True)
parser.add_argument('--backend',choices=('cpu','cuda'),required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
os.environ['XHDFE_GPU_BACKEND']=args.backend
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
cpp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpp)
rows=[]
def check(label,model,beta,**extra):
    error=float(np.max(np.abs(np.asarray(model.coef_)[:len(beta)]-beta)))
    assert error<1e-10,(label,error,np.asarray(model.coef_))
    assert model.converged_ and model.precision_certified_,label
    assert bool(model.gpu_used_)==(args.backend=='cuda'),label
    rows.append(dict(case=label,status='PASS',beta_error=error,**extra))
def fit(y,X,fes,**kwargs):
    opts=dict(num_threads=2,drop_singletons=False,tol=1e-12,fit_intercept=False)
    opts.update(kwargs.pop('options',{}))
    model=cpp.HdfeRegressor(**opts);model.fit(y,X,fes=fes,**kwargs)
    return model
n=8192;i=np.arange(n);a=2*(i%2)-1;b=2*((i//2)%2)-1;e=2*((i//512)%2)-1
delta=3.*2.**-21
for offset in (0.,17.,1000.,1e6):
    X=np.column_stack((offset+a,2*offset+a+delta*b))
    y=.75*X[:,0]-.5*X[:,1]+2+.125*e
    for has_fe in ((False,True) if args.backend=='cpu' else (True,)):
        # This FE contains full 1024-row sign cycles, preserving the oracle.
        model=fit(y,X,[i//1024] if has_fe else [],options=dict(fit_intercept=True))
        check(f'offset_{offset}_{has_fe}',model,[.75,-.5,2.])

# Age is exactly a sum of two FE functions on an unbalanced connected graph.
rng=np.random.default_rng(472026)
worker=rng.integers(0,64,n);year=rng.integers(0,17,n)
age=year-worker%40;X=rng.normal(size=(n,2));y=X@np.array([.75,-.5])+year*.125-worker*.25
for tolerance in (1e-8,1e-10,1e-12):
    model=fit(y,np.column_stack((X,age)),[worker,year],options=dict(tol=tolerance))
    check(f'additive_age_{tolerance}',model,[.75,-.5,0.])

for p in (2,9):
    Z=np.column_stack([2*((i//2**bit)%2)-1 for bit in range(p)]).astype(float)
    X=Z.copy();X[:,1]=Z[:,0]+delta*Z[:,1]
    target=np.full(p,.25);target[:2]=[.75,-.5];u=.125*e;y=X@target+u
    transform=np.eye(p);transform[:2,:2]=[[1.,-1/delta],[0.,1/delta]]
    for weight in ('none','aw','fw','pw'):
      w=np.ones(n) if weight=='none' else 1.+2*((i//1024)%2)
      total=w.sum();N=total if weight=='fw' else n
      for vce in ('unadjusted','robust','cluster','multiway'):
        if weight=='pw' and vce=='unadjusted':continue
        kw={} if weight=='none' else dict(weights=w,fweights=weight=='fw',pweights=weight=='pw')
        # Constant FE also makes these same cases executable on CUDA.
        fes=[np.zeros(n,dtype=np.int64)];K=p+1
        if vce in ('cluster','multiway'):
            labels=i//2 if vce=='cluster' else (i%1024)//2
            kw['clusters']=i//2 if vce=='cluster' else np.column_stack((i//2,labels))
            scores=np.zeros((int(labels.max())+1,p))
            np.add.at(scores,labels,Z*(w*u)[:,None])
            G=len(scores);meat=scores.T@scores
            V=transform@meat@transform.T/total**2*((N-1)/(N-K))*G/(G-1)
        else:
            factor=total/(N-K) if vce=='unadjusted' else N/(N-K)*np.sum(w if weight=='fw' else w*w)/total
            V=(transform@transform.T)*(factor/(64*total))
        model=fit(y,X,fes,options=dict(se_type='cluster' if vce=='multiway' else vce),**kw)
        err=float(np.max(np.abs(np.asarray(model.covariance_)-V))/np.max(np.abs(V)))
        assert err<1e-8,(p,weight,vce,err,model.df_resid_)
        check(f'vce_p{p}_{weight}_{vce}',model,target,covariance_relative_error=err)

for has_fe in ((False,True) if args.backend=='cpu' else (True,)):
  X=np.column_stack((a+.25*e,a+delta*b+.5*e));Z=np.column_stack((a,b))
  y=.75*X[:,0]-.5*X[:,1]+.125*e
  for vce in ('unadjusted','robust','cluster'):
    kw=dict(instruments=Z,endogenous_idx=[0,1])
    if vce=='cluster':kw['clusters']=i//2
    model=fit(y,X,[np.zeros(n,dtype=np.int64)] if has_fe else [],options=dict(se_type=vce),**kw)
    assert abs(model.rss_-n/64)<1e-9
    check(f'iv_{has_fe}_{vce}',model,[.75,-.5])

X=np.column_stack((a,a+delta*b));fes=[np.zeros(n,dtype=np.int64)]
model=fit(.75*X[:,0]-.5*X[:,1]+.125*e,X,fes)
try:model.fit(1e300*e,X,fes=fes)
except RuntimeError as error:
    assert 'no estimates' in str(error).lower(),str(error)
    assert all(not np.asarray(getattr(model,key)).size for key in ('coef_','covariance_','residuals_','sample_index_'))
    assert not model.converged_
    rows.append(dict(case='late_failure',status='PASS',error=str(error)))
else:raise AssertionError('Numerical-range failure returned estimates')
with args.out.open('x') as h:json.dump(rows,h,indent=2)
print(json.dumps(dict(status='PASS',rows=len(rows),backend=args.backend)))
