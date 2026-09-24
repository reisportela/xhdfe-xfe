"""2x2 WLS with individual variation and exact orthogonal dyadic noise."""
import argparse
from fractions import Fraction
import hashlib
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
records=[]
cell=np.repeat(np.arange(4),5);row=cell//2;col=cell%2
methods=('auto','gauss-seidel','symmetric-gauss-seidel','jacobi')
if args.backend=='cpu':methods+=('lsmr','mlsmr')

for heterogeneous in (False,True):
    if heterogeneous:
        x=np.tile([0.,1.,1.,-1.,.125],4)
        noise=np.tile([0.,.125,-.0625,0.,0.],4)
        low_weights=np.array([1.,1.,2.,4.,8.])
    else:
        x=np.tile([0.,1.,1.,-1.,-1.],4)
        noise=np.tile([0.,.125,-.125,.125,-.125],4)
        low_weights=np.ones(5)
    y=.75*x+noise+3*row-2*col
    for exponent in (0,20,80,120):
        weights=np.tile(low_weights,4)*2.**-exponent
        weights[::5]=np.arange(1.,5.) if heterogeneous else 1.
        # Exact WLS: FE means and the weighted x/noise product are zero.
        for c in range(4):
            indices=np.flatnonzero(cell==c)
            assert sum(Fraction(float(weights[i]))*Fraction(float(x[i])) for i in indices)==0
            assert sum(Fraction(float(weights[i]))*Fraction(float(noise[i])) for i in indices)==0
        gram=sum(Fraction(float(w))*Fraction(float(a))**2 for w,a in zip(weights,x))
        rss=sum(Fraction(float(w))*Fraction(float(e))**2 for w,e in zip(weights,noise))
        assert sum(Fraction(float(w))*Fraction(float(a))*Fraction(float(e)) for w,a,e in zip(weights,x,noise))==0
        meat=sum((Fraction(float(w))*Fraction(float(a))*Fraction(float(e)))**2 for w,a,e in zip(weights,x,noise))
        # Rank([FE,x])=4, with 20 actual rows; never four pseudo-observations.
        expected={'unadjusted':float(rss/(16*gram)), 'robust':float(Fraction(20,16)*meat/(gram*gram))}
        expected_rss=float(rss*20/sum(Fraction(float(w)) for w in weights))
        for mode in ('xhdfe-fast','reghdfe-comparable'):
            for method in methods:
                recoveries=('none','map','hybrid') if method=='auto' else ('none',)
                for recovery in recoveries:
                    for vce in ('unadjusted','robust'):
                        record=dict(heterogeneous=heterogeneous,exponent=exponent,mode=mode,
                                    method=method,recovery=recovery,vce=vce)
                        model=cpp.HdfeRegressor(num_threads=2,drop_singletons=False,
                            fit_intercept=False,retain_fes=recovery!='none',
                            fe_recovery_method='map' if recovery=='map' else 'hybrid',
                            absorption_method=method,se_type=vce,tolerance_mode=mode,tol=1e-8,max_iter=1000)
                        try:
                            model.fit(y,x[:,None],fes=[np.where(row==0,11,29),np.where(col==0,7,900001)],weights=weights)
                            b=float(model.coef_[0]);v=float(model.covariance_[0,0]);u=np.asarray(model.residuals_)
                            record.update(beta_error=abs(b-.75),covariance_relative_error=abs(v/expected[vce]-1),
                                residual_error=float(np.max(np.abs(u-noise))),rss_relative_error=abs(float(model.rss_)/expected_rss-1))
                            if recovery!='none':
                                fitted=b*x+sum(np.asarray(a) for a in model.fe_effects_)
                                record['reconstruction_error']=float(np.max(np.abs(y-fitted-noise)))
                            passed=(record['beta_error']<=1e-9 and record['covariance_relative_error']<=1e-8 and
                                record['residual_error']<=1e-9 and record['rss_relative_error']<=1e-8 and
                                record.get('reconstruction_error',0)<=1e-9 and model.nobs_==20 and
                                model.converged_ and model.precision_certified_ and
                                bool(model.gpu_used_)==(args.backend=='cuda'))
                            record['status']='PASS' if passed else 'FAIL'
                        except RuntimeError as error:
                            record.update(status='FAIL_VALID_REFUSAL',error=str(error))
                        records.append(record)

counts={s:sum(r['status']==s for r in records) for s in ('PASS','FAIL','FAIL_VALID_REFUSAL')}
with args.out.open('x') as f:
    json.dump(dict(backend=args.backend,module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
        counts=counts,rows=records,reference='exact dyadic WLS on all 20 observations'),f,indent=2)
print(json.dumps(counts))
raise SystemExit(counts['FAIL']!=0 or counts['FAIL_VALID_REFUSAL']!=0)
