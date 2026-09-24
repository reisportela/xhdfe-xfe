"""Distinguish exact FE redundancy from identified variation at a large level."""
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
os.environ['XHDFE_GPU_BACKEND']=args.backend
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
cpp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpp)
records=[]
def check(name,y,X,fes,beta,**kw):
    model=cpp.HdfeRegressor(num_threads=2,drop_singletons=False,fit_intercept=False,tol=1e-12)
    try:
        model.fit(y,X,fes=fes,**kw)
        error=float(np.max(np.abs(np.asarray(model.coef_)-beta)))
        assert error<1e-10,(name,error,np.asarray(model.coef_))
        assert model.converged_ and model.precision_certified_
        assert bool(model.gpu_used_)==(args.backend=='cuda')
        records.append(dict(case=name,status='PASS',beta_error=error))
    except Exception as error:
        records.append(dict(case=name,status='FAIL',error=str(error)))
        print(json.dumps(records[-1]),flush=True)
n=1024;i=np.arange(n);a=2*(i%2)-1;b=2*((i//2)%2)-1;e=2*((i//4)%2)-1
for offset in (1e6,1e10,1e12,1e14,1e15):
  for weight in ('none','aw','fw'):
    kw={} if weight=='none' else dict(weights=1.+2*((i//8)%2),fweights=weight=='fw')
    check(f'identified_{offset}_{weight}',.75*a+.125*e,(offset+a).reshape(-1,1),[np.zeros(n,dtype=int)],np.array([.75]),**kw)

n=4096;i=np.arange(n);block=i//8
a=2*(i%2)-1;b=2*((i//2)%2)-1;e=2*((i//4)%2)-1
fes=[block%7,(block//7)%11,(block//77)%5]
age=3*fes[0]-2*fes[1]+fes[2]
y=.75*a-.5*b+.5*fes[0]-.25*fes[1]+.125*fes[2]+.125*e
for offset in (0.,1e6,1e14):
    check(f'redundant_three_fe_{offset}',y,np.column_stack((a,b,offset+age)),fes,np.array([.75,-.5,0.]))

rng=np.random.default_rng(482026)
teams=[np.sort(rng.choice(12,size=2+2*(p%2),replace=False)) for p in range(40)]
pattern=np.repeat(np.arange(len(teams)),8);i=np.arange(len(pattern))
a=2*(i%2)-1;b=2*((i//2)%2)-1;e=2*((i//4)%2)-1
groups=np.concatenate([np.full(len(teams[p]),g) for g,p in enumerate(pattern)])
individual=np.concatenate([teams[p] for p in pattern])
for aggregation in ('sum','mean'):
    D=np.zeros((len(pattern),12))
    for g,p in enumerate(pattern):D[g,teams[p]]=1 if aggregation=='sum' else 1/len(teams[p])
    component=D@np.arange(12)
    y=.75*a-.5*b+component+.125*e
    for offset in (0.,1e6,1e14):
        X=np.column_stack((a,b,component+offset))
        check(f'redundant_group_{aggregation}_{offset}',y[groups],X[groups],
              [individual,np.zeros(len(groups),dtype=int)],np.array([.75,-.5,0.]),
              group=groups,individual=individual,aggregation=aggregation)

with args.out.open('x') as h:json.dump(records,h,indent=2)
print(json.dumps(dict(passed=sum(r['status']=='PASS' for r in records),rows=len(records),backend=args.backend)))
assert all(r['status']=='PASS' for r in records)
