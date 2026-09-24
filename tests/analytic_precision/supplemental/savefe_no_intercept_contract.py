"""Saving FEs must preserve the dummy-variable model, including its constant span."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from exact_wls_reference import fit

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module',type=Path,required=True)
parser.add_argument('--backend',choices=('cpu','cuda'),required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
os.environ.update(XHDFE_GPU_BACKEND=args.backend,XHDFE_ABSORPTION_CACHE_MODE='off',XHDFE_MOBILITY_MODE='off')
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
cpp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpp)
i=np.arange(96);worker=i//24;firm=(i//8)%3
x=(2*(i%2)-1)+.5*worker;noise=.125*(2*((i//2)%2)-1)
records=[]
for dims in (1,2):
    fes=[worker] if dims==1 else [worker,firm]
    y=3.+.75*x+2*worker-(firm if dims==2 else 0)+noise
    dummies=np.column_stack([worker==g for g in range(4)]+([firm==g for g in range(1,3)] if dims==2 else []))
    for weighted in (False,True):
        weights=1.+worker if weighted else np.ones(len(i))
        exact=fit(np.column_stack((dummies,x)),y,weights)
        reference=np.array([float(v) for v in exact['residuals']])
        for style in ('component','reghdfe'):
            os.environ['XHDFE_FE_NORMALIZE']=style
            for intercept in (False,True):
                for retain in (False,True):
                    model=cpp.HdfeRegressor(num_threads=2,max_iter=1000,tol=1e-8,
                        drop_singletons=False,fit_intercept=intercept,retain_fes=retain)
                    row=dict(dims=dims,weighted=weighted,style=style,intercept=intercept,retain=retain)
                    try:model.fit(y,x[:,None],fes=fes,weights=weights if weighted else None)
                    except RuntimeError as error:row.update(verdict='FAIL',error=str(error))
                    else:
                        residual=np.asarray(model.residuals_)
                        beta=float(model.coef_[0])
                        row.update(beta_error=abs(beta-float(exact['coefficients'][-1])),
                            residual_error=float(np.max(np.abs(residual-reference))),
                            residual_mean=float(np.average(residual,weights=weights)))
                        if retain:
                            fitted=beta*x+sum(np.asarray(v) for v in model.fe_effects_)
                            if intercept:fitted+=float(model.coef_[-1])
                            row['reconstruction_error']=float(np.max(np.abs(y-fitted-reference)))
                        passed=all(value<=1e-9 for key,value in row.items() if key.endswith('error'))
                        row['verdict']='PASS' if passed and model.converged_ and model.precision_certified_ and model.fe_recovery_converged_ and bool(model.gpu_used_)==(args.backend=='cuda') else 'FAIL'
                    records.append(row)
counts={verdict:sum(row['verdict']==verdict for row in records) for verdict in ('PASS','FAIL')}
with args.out.open('x') as handle:json.dump(dict(backend=args.backend,counts=counts,rows=records),handle,indent=2)
print(json.dumps(counts));raise SystemExit(counts['FAIL']!=0)
