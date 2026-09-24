"""Group-only sample mapping against explicit OLS after weighted pruning."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--module',type=Path,required=True)
parser.add_argument('--backend',choices=['cpu','cuda'],required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
os.environ.update(XHDFE_GPU_BACKEND=args.backend,XHDFE_ABSORPTION_CACHE_MODE='off',XHDFE_MOBILITY_MODE='off')
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
cpp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpp)
rng=np.random.default_rng(20260906)
a=np.r_[np.repeat([0,0,1,1],3),0,2,2]
b=np.r_[np.repeat([0,1,0,1],3),2,2,3]
X=rng.normal(size=(len(a),2))
y=X@np.array([.7,-.2])+.3*a-.1*b+rng.normal(size=len(a))*.05
records=[]
for weight in ('none','aw','fw'):
    w=np.ones(len(a)) if weight=='none' else 1.+np.arange(len(a))%3
    w[-3:]=[1,1,2]
    for drop in (False,True):
        retained=np.arange(len(a)) if not drop or weight=='fw' else np.arange(len(a)-3)
        for sparse in (False,True):
            group=np.arange(len(a))*(100003 if sparse else 1)+(100000003 if sparse else 0)
            repeats=np.repeat(np.arange(len(a)),2+np.arange(len(a))%4)
            model=cpp.HdfeRegressor(num_threads=2,drop_singletons=drop,fit_intercept=True,
                                   tolerance_mode='strict-residual',tol=1e-12,retain_fes=True)
            # Refit the same object in a different order to expose stale mappings.
            for order in (np.arange(len(repeats)),rng.permutation(len(repeats))):
                ids=repeats[order]
                kw={'fes':[a[ids],b[ids]],'group':group[ids]}
                if weight!='none':kw.update(weights=w[ids],fweights=weight=='fw')
                model.fit(y[ids],X[ids],**kw)
                selected=np.asarray(model.sample_index_,dtype=int)
                assert len(selected)==len(retained) and np.all((0<=selected)&(selected<len(ids)))
                assert set(ids[selected])==set(retained)
                assert all(row==np.flatnonzero(ids==g)[0] for row,g in zip(selected,ids[selected]))
                chosen=ids[selected]
                D=np.c_[np.eye(a.max()+1)[a[chosen]],np.eye(b.max()+1)[b[chosen]]]
                design=np.c_[X[chosen],D]
                rootw=np.sqrt(w[chosen] if weight!='none' else np.ones(len(chosen)))
                gold=np.linalg.lstsq(rootw[:,None]*design,rootw*y[chosen],rcond=1e-12)[0]
                residual=y[chosen]-design@gold
                np.testing.assert_allclose(np.asarray(model.coef_)[:2],gold[:2],rtol=0,atol=1e-10)
                np.testing.assert_allclose(model.residuals_,residual,rtol=0,atol=1e-9)
                assert model.converged_ and model.precision_certified_ and model.fe_recovery_converged_
                if args.backend=='cuda':assert model.gpu_used_ and model.gpu_status_code_==1
                record={'weight':weight,'drop_singletons':drop,'sparse_ids':sparse,'retained_groups':len(selected),'input_rows':len(ids),'dropped_groups':len(a)-len(retained),'beta_error':float(np.max(np.abs(np.asarray(model.coef_)[:2]-gold[:2]))),'residual_error':float(np.max(np.abs(np.asarray(model.residuals_)-residual)))}
                records.append(record)
with args.out.open('x') as f:json.dump({'backend':args.backend,'verdict':'PASS','checks':records},f,indent=2)
print('GROUP_SAMPLE_CONTRACT_PASS',args.backend,len(records),flush=True)
