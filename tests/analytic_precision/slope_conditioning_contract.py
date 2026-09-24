"""Exact dyadic OLS oracle with two nearly collinear, identified regressors."""
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
parser.add_argument('--legacy',action='store_true')
args=parser.parse_args()
os.environ['XHDFE_GPU_BACKEND']=args.backend
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
cpp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpp)
records=[]
for n in (1024,8192):
  i=np.arange(n);a=2*(i%2)-1;b=2*((i//2)%2)-1;e=2*((i//4)%2)-1
  for k in (16,18,19,20,21):
    delta=3.*2.**(-k);X=np.column_stack((a,a+delta*b))
    y=.75*X[:,0]-.5*X[:,1]+.125*e
    # The sign columns and their products balance exactly in each 8-row block.
    assert np.dot(a,e)==np.dot(b,e)==np.dot(a,b)==0
    bread=np.array([[1+delta*delta,-1.],[-1.,1.]])/(n*delta*delta)
    for kind in (('ols',) if args.backend=='cpu' else ())+('ordinary_fe','grouped_sum','grouped_mean'):
      K=2+(n//16 if kind=='ordinary_fe' else 1 if kind.startswith('grouped') else 0)
      for vce in ('unadjusted','robust','cluster'):
        row=dict(n=n,k=k,kind=kind,vce=vce,backend=args.backend)
        options=dict(num_threads=2,drop_singletons=False,fit_intercept=False,se_type=vce,tol=1e-12)
        if kind.startswith('grouped'):options['dofadjustments']='exact'
        model=cpp.HdfeRegressor(**options)
        try:
          if kind.startswith('grouped'):
            index=np.repeat(i,2);individual=np.tile([0,1],n)
            kw=dict(fes=[individual],group=index,individual=individual,aggregation=kind.split('_')[1])
            if vce=='cluster':kw['clusters']=(i//2)[index]
            model.fit(y[index],X[index],**kw)
          else:
            kw=dict(fes=[] if kind=='ols' else [i//16])
            if vce=='cluster':kw['clusters']=i//2
            model.fit(y,X,**kw)
          beta=np.asarray(model.coef_)[:2];V=np.asarray(model.covariance_)[:2,:2]
          if vce=='cluster':
            q=((n-1)/(n-K))*((n//2)/(n//2-1))
            reference=q/(32*n*delta*delta)*np.array([[1.,-1.],[-1.,1.]])
          else:reference=(n/(64*(n-K)))*bread
          db=float(np.max(np.abs(beta-[.75,-.5])))
          dv=float(np.max(np.abs(V-reference))/np.max(np.abs(reference)))
          rss=abs(model.rss_-n/64)
          valid=db<1e-10 and dv<1e-8 and rss<1e-9 and model.converged_ and model.precision_certified_
          valid=valid and bool(model.gpu_used_)==(args.backend=='cuda')
          row.update(status='PASS' if valid else 'FAIL',beta=beta.tolist(),beta_error=db,
                     covariance_relative_error=dv,rss_error=rss)
        except Exception as error:row.update(status='ERROR',error=str(error))
        records.append(row)
        if row['status']!='PASS':print(json.dumps(row),flush=True)
with args.out.open('x') as out:json.dump(records,out,indent=2)
print(json.dumps({'passed':sum(r['status']=='PASS' for r in records),'rows':len(records)}))
if not args.legacy:assert all(r['status']=='PASS' for r in records)
