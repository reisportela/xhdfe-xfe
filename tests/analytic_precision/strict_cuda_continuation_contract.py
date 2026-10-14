"""Synthetic laplacian strict CUDA continuation, budget and lifecycle checks."""
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import assess

ap=argparse.ArgumentParser()
ap.add_argument('--module',type=Path,required=True)
ap.add_argument('--fixture',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
assert not args.output.exists()
os.environ['XHDFE_GPU_BACKEND']='cuda'
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
core=importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
meta=json.loads((args.fixture/'case.json').read_text())
assert meta['kind']=='group_individual' and meta['aggregation']=='sum'
frame=pd.read_stata(args.fixture/'long.dta')
y=frame.y.to_numpy()
X=frame[['x1','x2','x3']].to_numpy()
kwargs=dict(fes=[frame[c].to_numpy(dtype=np.int64) for c in meta['fes']],
    group=frame.group.to_numpy(dtype=np.int64),
    individual=frame.individual.to_numpy(dtype=np.int64),aggregation='sum')
rows=[]
for budget in [100000,1,1128,1500]:
    reg=core.HdfeRegressor(num_threads=2,max_iter=budget,tolerance_mode='strict-residual',
        tol=1e-12,se_type=meta['vce'],fit_intercept=meta['intercept'])
    if 1<budget<100000:
        # A successful state must not survive the failed grouped fit.
        i=np.arange(32,dtype=float)
        reg.fit(1+2*i,np.column_stack([i]),fes=[np.zeros(len(i),dtype=np.int32)])
        assert reg.converged_
    try:
        reg.fit(y,X,**kwargs)
        assert budget==100000, 'Iteration-limited case unexpectedly returned estimates'
        ix=np.asarray(reg.sample_index_,dtype=int)
        raw=dict(rc=0,beta=np.asarray(reg.coef_)[:3].tolist(),
            cons=float(reg.coef_[3]) if len(reg.coef_)>3 else 0.,
            V=np.asarray(reg.covariance_)[:3,:3].tolist(),N=reg.nobs_,rss=reg.rss_,
            df_r=reg.df_resid_,df_a=reg.df_a_,singletons=reg.num_singletons_,
            iterations=reg.num_iterations_,converged=int(reg.converged_),
            certified=int(reg.precision_certified_),gpu_used=int(reg.gpu_used_),
            groups=frame.group.to_numpy()[ix].astype(int).tolist(),
            residuals=np.asarray(reg.residuals_).tolist())
        job=dict(case='laplacian_sum',interface='native',engine='xhdfe',
            method='auto',precision='strict',backend='cuda')
        verdict=assess(job,raw,args.fixture)
        assert verdict['verdict']=='PASS',verdict
        assert reg.gpu_used_ and reg.num_iterations_<=budget
        rows.append(dict(budget=budget,status='PASS',outcome='strict_oracle_pass',
                         iterations=int(reg.num_iterations_),gpu_used=True))
    except RuntimeError as error:
        assert budget<100000,error
        empty=(reg.lifecycle_state_=='failed' and not reg.converged_ and
               np.asarray(reg.coef_).size==0 and np.asarray(reg.covariance_).size==0)
        assert empty and reg.num_iterations_<=budget
        consumed=re.search(r'iterations=(\d+)/(\d+)',str(error))
        assert consumed and int(consumed[2])==budget and int(consumed[1])<=budget
        if budget==1500:
            # The unaccepted auxiliary state must not replace the complete
            # original-model diagnostics in the refusal message.
            mean=re.search(r'strict max weighted FE mean=([0-9.eE+-]+)',str(error))
            assert mean and float(mean[1])>1e-12
        rows.append(dict(budget=budget,status='PASS',outcome='expected_budget_refusal',
                         error=str(error),empty_state=empty,iterations=int(reg.num_iterations_)))
        if budget>1:
            reg.fit(1+2*i,np.column_stack([i]),fes=[np.zeros(len(i),dtype=np.int32)])
            assert reg.converged_ and np.allclose(reg.coef_,[2,1],atol=1e-12,rtol=0)
            rows[-1]['subsequent_fit_pass']=True
with args.output.open('x') as f:
    json.dump(dict(status='PASS',rows=rows,module=str(args.module.resolve()),
        module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
        scope='One numerical strict CUDA pass and three negative budget/lifecycle controls'),f,indent=2)
print('STRICT_CUDA_CONTINUATION_PASS',len(rows))
