"""Independent blockwise analytic projector challenges residual-only refinement."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module',type=Path,required=True)
parser.add_argument('--backend',choices=('cpu','cuda'),required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
os.environ['XHDFE_GPU_BACKEND']=args.backend
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
cpp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpp)
here=Path(__file__).resolve().parent
base=pd.read_csv(here/'fixtures/patent_chain64.csv',float_precision='round_trip')
records=[]
for copies in (2,3,9):
  parts=[]
  for component in range(copies):
    part=base.copy()
    part.citations += component*.03*part.funding
    for variable in ('patent_id','inventor_id','year'):
      part[variable]=part[variable].astype(np.int64)+component*100000
    parts.append(part)
  data=pd.concat(parts,ignore_index=True)
  group=data.groupby('patent_id',sort=True).first()
  for aggregation in ('sum','mean'):
    for weight in ('','aw','fw'):
      projected=[]; weights=[]
      for part in parts:
        f=part.groupby('patent_id',sort=True).first()
        w=f.w2.to_numpy() if weight=='fw' else f.w1.to_numpy() if weight else np.ones(len(f))
        w=w.astype(np.longdouble)
        V=f[['citations','funding','lab_size']].to_numpy().astype(np.longdouble)
        z=np.tile([1,-1,-1,1],len(f)//8).repeat(2).astype(np.longdouble)
        if aggregation=='mean':z*=part.groupby('patent_id',sort=True).size().to_numpy()
        total=w.reshape(-1,2).sum(axis=1).repeat(2)
        means=(w[:,None]*V).reshape(-1,2,3).sum(axis=1).repeat(2,axis=0)/total[:,None]
        direction=z/total
        projected.append(V-means+direction[:,None]*((w*direction)@V)[None,:]/((w*direction)@direction))
        weights.append(w)
      v=np.vstack(projected);w=np.concatenate(weights);x=v[:,1:];y=v[:,0]
      h=x.T@(w[:,None]*x);rhs=x.T@(w*y);det=h[0,0]*h[1,1]-h[0,1]*h[1,0]
      beta=np.array([(rhs[0]*h[1,1]-rhs[1]*h[0,1])/det,(h[0,0]*rhs[1]-h[1,0]*rhs[0])/det])
      residual=y-x@beta
      for mode,tol in (('reghdfe-comparable',1e-8),('strict-residual',1e-12)):
        row=dict(copies=copies,aggregation=aggregation,weight=weight,mode=mode,backend=args.backend)
        try:
          model=cpp.HdfeRegressor(num_threads=2,drop_singletons=False,max_iter=100000,tolerance_mode=mode,tol=tol)
          kw=dict(group=data.patent_id.to_numpy(),individual=data.inventor_id.to_numpy(),aggregation=aggregation,
                  fes=[data.inventor_id.to_numpy(),data.year.to_numpy()])
          if weight:kw.update(weights=data.w2.to_numpy() if weight=='fw' else data.w1.to_numpy(),fweights=weight=='fw')
          model.fit(data.citations.to_numpy(),data[['funding','lab_size']].to_numpy(),**kw)
          selected=np.asarray(model.sample_index_,dtype=int);order=np.argsort(data.patent_id.to_numpy()[selected])
          db=float(np.max(np.abs(np.asarray(model.coef_)[:2]-beta)))
          du=float(np.max(np.abs(np.asarray(model.residuals_)[order]-residual))/np.std(residual,ddof=1))
          assert db<1e-8 and du<1e-7,(db,du)
          assert model.converged_ and model.precision_certified_
          assert bool(model.gpu_used_)==(args.backend=='cuda')
          row.update(status='PASS',beta_error=db,residual_error_sd=du)
        except Exception as error:row.update(status='FAIL',error=str(error))
        records.append(row)
with args.out.open('x') as f:json.dump(records,f,indent=2)
print(json.dumps(dict(passed=sum(r['status']=='PASS' for r in records),rows=len(records))))
for row in records:
  if row['status']!='PASS':print(json.dumps(row))
raise SystemExit(any(r['status']!='PASS' for r in records))
