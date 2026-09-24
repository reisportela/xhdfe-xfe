"""Adding a redundant FE must not bypass the same analytic projection check."""
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
os.environ.update(XHDFE_GPU_BACKEND=args.backend,XHDFE_ABSORPTION_CACHE_MODE='off',XHDFE_MOBILITY_MODE='off')
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
cpp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpp)
i=np.arange(64);cell=i//16;a=cell//2;b=cell%2
x=(cell==1).astype(float);noise=.125*(2*(i%2)-1);y=.75*x+b-a+noise
rows=[]
for family,fes in [('two',[a,b]),('constant',[a,b,np.zeros(64,dtype=int)]),
                   ('duplicate',[a,b,a]),('two_duplicates',[a,b,a,b])]:
    for exponent in (0,40,80,120):
        epsilon=2.**-exponent;w=np.where(a==b,1.,epsilon)
        variance=(1+epsilon)**2/(960*epsilon)
        model=cpp.HdfeRegressor(num_threads=2,fit_intercept=False,drop_singletons=False,max_iter=1000)
        row=dict(family=family,exponent=exponent)
        try:model.fit(y,x[:,None],fes=fes,weights=w)
        except RuntimeError as error:
            empty=all(np.asarray(getattr(model,key)).size==0 for key in ('coef_','covariance_','residuals_'))
            informative=any(word in str(error).lower() for word in ('precision','converg','numerical','iteration','range'))
            safe=empty and informative and model.lifecycle_state_=='failed' and not model.converged_
            row.update(verdict='FAIL_VALID_REFUSAL' if safe else 'FAIL',error=str(error))
        else:
            db=abs(float(model.coef_[0])-.75)
            du=float(np.max(np.abs(np.asarray(model.residuals_)-noise)))
            dv=abs(float(model.covariance_[0,0])/variance-1)
            passed=db<1e-9 and du<1e-9 and dv<1e-8 and model.converged_ and model.precision_certified_
            passed=passed and bool(model.gpu_used_)==(args.backend=='cuda')
            row.update(verdict='PASS' if passed else 'FAIL',beta_error=db,residual_error=du,variance_error=dv)
        rows.append(row)
counts={v:sum(row['verdict']==v for row in rows)
        for v in ('PASS','FAIL_VALID_REFUSAL','FAIL')}
with args.out.open('x') as handle:json.dump(dict(counts=counts,rows=rows),handle,indent=2)
print(json.dumps(counts));raise SystemExit(counts['FAIL']!=0 or counts['FAIL_VALID_REFUSAL']!=0)
