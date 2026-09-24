"""One well-conditioned regressor; HC1 depends on low-noise observations."""
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
i=np.arange(4096);cell=i//256;a=cell//4;b=cell%4
A=2*(i%2)-1;B=2*((i//2)%2)-1;lever=(a==0)&(b==1)
x=A*(1+B)/2*lever
w=(1+(3*a+b)%7).astype(float);n=len(i);df=n-8
rows=[]
for exponent in (0,10,20,30,40):
    small=2.**-exponent;noise=np.where(lever,small,1.)*B
    for offset in (1.,16.,64.):
        y=.75*x+offset*(b-a)+noise
        variance=32*small**2/df
        model=cpp.HdfeRegressor(num_threads=2,drop_singletons=False,fit_intercept=False,max_iter=1000,se_type='robust')
        row=dict(exponent=exponent,offset=offset)
        try:model.fit(y,x[:,None],fes=[a,b],weights=w)
        except RuntimeError as error:
            empty=all(np.asarray(getattr(model,key)).size==0 for key in ('coef_','covariance_','residuals_'))
            informative=any(word in str(error).lower() for word in ('precision','converg','numerical','iteration','range'))
            safe=empty and informative and model.lifecycle_state_=='failed' and not model.converged_
            row.update(verdict='FAIL_VALID_REFUSAL' if safe else 'FAIL',error=str(error))
        else:
            row.update(beta_error=abs(float(model.coef_[0])-.75),
                variance_error=abs(float(model.covariance_[0,0])/variance-1),
                residual_error=float(np.max(np.abs(np.asarray(model.residuals_)-noise))))
            passed=row['beta_error']<=1e-9 and row['variance_error']<=1e-8 and row['residual_error']<=1e-9
            row['verdict']='PASS' if passed and model.converged_ and model.precision_certified_ and bool(model.gpu_used_)==(args.backend=='cuda') else 'FAIL'
        rows.append(row)
counts={verdict:sum(row['verdict']==verdict for row in rows)
        for verdict in ('PASS','FAIL_VALID_REFUSAL','FAIL')}
with args.out.open('x') as f:json.dump(dict(counts=counts,rows=rows),f,indent=2)
print(json.dumps(counts));raise SystemExit(counts['FAIL']!=0 or counts['FAIL_VALID_REFUSAL']!=0)
