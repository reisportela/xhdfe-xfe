"""Exact WLS and reconstruction across categorical and heterogeneous saved FEs."""
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
i=np.arange(192);worker=i//48;firm=(i//16)%3;year=(i//8)%2
x=(2*(i%2)-1)+.5*worker;z=2*((i//2)%2)-1;noise=.125*(2*((i//4)%2)-1)
records=[]
for family in ('one_fe','two_fe','three_fe','heterogeneous','pure_plus_fe','pure_only'):
    fes=[worker] if family in ('one_fe','pure_only') else [worker,firm]
    if family=='three_fe':fes.append(year)
    mixed=family in ('heterogeneous','pure_plus_fe','pure_only')
    include_worker=family not in ('pure_plus_fe','pure_only')
    columns=[(worker==g).astype(float) for g in range(4)] if include_worker else []
    if len(fes)>1:columns.extend((firm==g).astype(float) for g in range(1 if include_worker else 0,3))
    if family=='three_fe':columns.append((year==1).astype(float))
    if mixed:columns.extend((worker==g)*z for g in range(4))
    dummies=np.column_stack(columns)
    slopes=[(0,z,include_worker)] if mixed else None
    y=3+.75*x+noise+(2*worker if include_worker else 0)
    if len(fes)>1:y=y-firm
    if family=='three_fe':y=y+.5*year
    if mixed:y=y+(1.+worker)*z
    for weighted in (False,True):
        weights=1.+worker if weighted else np.ones(len(i))
        for intercept in (False,True):
            design=dummies
            if family=='pure_only' and intercept:design=np.column_stack((design,np.ones(len(i))))
            exact=fit(np.column_stack((design,x)),y,weights)
            reference=np.array([float(v) for v in exact['residuals']])
            for style in ('component','reghdfe'):
                os.environ['XHDFE_FE_NORMALIZE']=style
                for retain in (False,True):
                    model=cpp.HdfeRegressor(num_threads=2,max_iter=1000,tol=1e-8,
                        drop_singletons=False,fit_intercept=intercept,retain_fes=retain)
                    row=dict(family=family,weighted=weighted,style=style,intercept=intercept,retain=retain)
                    try:model.fit(y,x[:,None],fes=fes,slopes=slopes,weights=weights if weighted else None)
                    except RuntimeError as error:row.update(verdict='FAIL',error=str(error))
                    else:
                        beta=float(model.coef_[0]);residual=np.asarray(model.residuals_)
                        row.update(beta_error=abs(beta-float(exact['coefficients'][-1])),
                            residual_error=float(np.max(np.abs(residual-reference))),
                            variance_relative_error=abs(float(model.covariance_[0,0])/float(exact['variance'])-1))
                        if retain:
                            predicted=beta*x+sum(np.asarray(v) for v in model.fe_effects_)
                            if intercept:predicted+=float(model.coef_[-1])
                            row['reconstruction_error']=float(np.max(np.abs(y-predicted-reference)))
                        passed=all(value<=1e-8 for key,value in row.items() if key.endswith('error'))
                        passed=passed and model.converged_ and model.precision_certified_ and model.fe_recovery_converged_
                        row['verdict']='PASS' if passed and bool(model.gpu_used_)==(args.backend=='cuda') else 'FAIL'
                    records.append(row)
counts={verdict:sum(row['verdict']==verdict for row in records) for verdict in ('PASS','FAIL')}
with args.out.open('x') as handle:json.dump(dict(backend=args.backend,counts=counts,rows=records),handle,indent=2)
print(json.dumps(counts));raise SystemExit(counts['FAIL']!=0)
