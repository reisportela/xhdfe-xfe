"""Exact weighted two-way-FE oracle: four cell means identify four parameters."""
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
os.environ.update(XHDFE_GPU_BACKEND=args.backend,
                  XHDFE_ABSORPTION_CACHE_MODE='off',XHDFE_MOBILITY_MODE='off')
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
cpp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpp)
cell=np.repeat(np.arange(4),16);worker=cell//2;firm=cell%2
X=(cell==1).astype(float).reshape(-1,1)
noise=.125*(2*(np.arange(64)%2)-1)
y=.75*X[:,0]+firm-worker+noise
# Positive weights constant within each cell do not change its fitted mean.
# The coefficient is the cell-mean contrast; all regression residuals are noise.
rows=[]
for exponent in (0,20,40,60,80,120):
    epsilon=2.**-exponent
    for scale in (2.**-200,1.,2.**200):
        weights=scale*np.where(worker==firm,1.,epsilon)
        for reverse in (False,True):
            order=np.arange(63,-1,-1) if reverse else np.arange(64)
            for method in (('auto','mlsmr') if args.backend=='cpu' else ('auto',)):
                for vce in ('unadjusted','robust'):
                    record=dict(exponent=exponent,scale=scale,reverse=reverse,method=method,vce=vce)
                    model=cpp.HdfeRegressor(num_threads=1,max_iter=1000,tol=1e-8,
                        drop_singletons=False,fit_intercept=False,absorption_method=method,se_type=vce)
                    try:
                        model.fit(y[order],X[order],fes=[worker[order],firm[order]],weights=weights[order])
                    except RuntimeError as error:
                        empty=all(np.asarray(getattr(model,key)).size==0 for key in ('coef_','covariance_','residuals_'))
                        informative=any(token in str(error).lower() for token in ('converg','precision','iteration','numerical','range'))
                        safe=empty and informative and model.lifecycle_state_=='failed' and not model.converged_
                        record.update(verdict='FAIL_VALID_REFUSAL' if safe else 'FAIL',error=str(error))
                    else:
                        coefficient=float(model.coef_[0])
                        variance=float(model.covariance_[0,0])
                        expected_variance=(1.+epsilon)*(1.+1./epsilon)/960. if vce=='unadjusted' else 1./240.
                        beta_error=abs(coefficient-.75)
                        covariance_error=abs(variance/expected_variance-1.)
                        residual_error=float(np.max(np.abs(np.asarray(model.residuals_)-noise[order])))
                        passed=(beta_error<=1e-9 and covariance_error<=1e-8 and residual_error<=1e-9
                                and model.converged_ and model.precision_certified_
                                and bool(model.gpu_used_)==(args.backend=='cuda')
                                and model.num_iterations_<=1000)
                        record.update(verdict='PASS' if passed else 'FAIL',beta=coefficient,beta_error=beta_error,
                                      covariance_relative_error=covariance_error,residual_error=residual_error,
                                      converged=model.converged_,certified=model.precision_certified_,iterations=model.num_iterations_)
                    rows.append(record)
counts={verdict:sum(row['verdict']==verdict for row in rows)
        for verdict in ('PASS','FAIL_VALID_REFUSAL','FAIL')}
with args.out.open('x') as handle:json.dump(dict(backend=args.backend,counts=counts,rows=rows),handle,indent=2)
print(json.dumps(counts));raise SystemExit(counts['FAIL']!=0 or counts['FAIL_VALID_REFUSAL']!=0)
