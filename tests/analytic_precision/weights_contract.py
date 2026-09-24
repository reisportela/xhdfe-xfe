"""Native/formula Python inference parity with the explicit Stata OLS oracle."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--module',type=Path,required=True)
parser.add_argument('--formula-source',type=Path,required=True)
parser.add_argument('--backend',choices=['cpu','cuda'],required=True)
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--fixture',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
os.environ.update(XHDFE_GPU_BACKEND=args.backend,XHDFE_ABSORPTION_CACHE_MODE='off',XHDFE_MOBILITY_MODE='off')
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
cpp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpp)
sys.path.insert(0,str(ROOT));sys.modules['xhdfe.py_hdfe_v11']=cpp
import xhdfe
spec=importlib.util.spec_from_file_location('xhdfe._formula',args.formula_source)
formula=importlib.util.module_from_spec(spec);sys.modules['xhdfe._formula']=formula;spec.loader.exec_module(formula)
data=pd.read_stata(args.fixture)
data['audit_cluster']=(data.group*13+np.floor(data.group/5)).astype(int)%23
reference=pd.read_stata(args.reference)
rows=[]
for _,ref in reference[reference.engine=='regress'].iterrows():
    kind,vce=ref.weight_case,ref.vce
    w=data.weight.to_numpy().copy()
    if kind=='iw_fractional':w=.73+((data.g1.to_numpy()+2*data.g2.to_numpy())%7)/13
    cluster=data.audit_cluster.to_numpy() if vce=='cluster' else None
    flags=dict(fweights=kind=='fw',iweights=kind.startswith('iw_'),pweights=kind=='pw')
    for interface in ('native','formula'):
        options=dict(se_type=vce,tol=1e-12,tolerance_mode='strict-residual',num_threads=2)
        if interface=='native':
            model=cpp.HdfeRegressor(**options)
            model.fit(data.y.to_numpy(),data[['x1','x2','x3']].to_numpy(),
                      fes=[data.g1.to_numpy(dtype=np.int64),data.g2.to_numpy(dtype=np.int64)],
                      weights=w,clusters=None if cluster is None else cluster.reshape(-1,1),**flags)
        else:
            model=formula.feols('y ~ x1 + x2 + x3 | g1 + g2',data,weights=w,
                                clusters=cluster,**flags,**options)
        values={'N':model.nobs_,'df_r':model.df_resid_,'rss':model.rss_}
        for prefix,array in [('b',model.coef_),('se',model.stderr_),('p',model.pvalues_),
                             ('lo',np.asarray(model.conf_int_)[:,0]),('hi',np.asarray(model.conf_int_)[:,1])]:
            values.update({f'{prefix}{i+1}':float(array[i]) for i in range(3)})
        differences={k:abs(float(v)-float(ref[k])) for k,v in values.items()}
        assert np.isfinite(list(values.values())).all(),(kind,vce,interface,values)
        assert differences['N']==differences['df_r']==0,(kind,vce,interface,differences)
        assert max(differences.values())<1e-8,(kind,vce,interface,differences)
        assert model.converged_ and model.precision_certified_
        if args.backend=='cuda':assert model.gpu_used_ and model.gpu_status_code_==1
        rows.append(dict(weight_case=kind,vce=vce,interface=interface,max_error=max(differences.values()),differences=differences))
with args.out.open('x') as f:json.dump({'backend':args.backend,'PASS':True,'rows':rows},f,indent=2)
print('PYTHON_WEIGHTS_CONTRACT_PASS',args.backend,len(rows),flush=True)
