"""Synthetic independent check for the offline three-FE Schur oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from schur_projection_oracle import SchurProjection

ap=argparse.ArgumentParser()
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
assert not args.output.exists()
worker=np.repeat(np.arange(24),12)
firm=np.tile(np.repeat(np.arange(6),2),24)
year=(firm//3)*2+np.tile([0,1],len(firm)//2)
fes=[worker,firm,year]
counts=[np.bincount(f) for f in fes]
def components(a,b,ga,gb):
    graph=sparse.coo_matrix((np.ones(2*len(a)),
        (np.r_[a,b+ga],np.r_[b+ga,a])),shape=(ga+gb,ga+gb)).tocsr()
    return connected_components(graph,directed=False)
projection=SchurProjection(fes,counts,components)
assert projection.metadata['known_null_rank']==3
assert projection.metadata['old_unconstrained_exact_nulls']==1
D=np.column_stack([np.eye(len(c))[f] for f,c in zip(fes,counts)])
rng=np.random.default_rng(204)
rows=[]
for scale in [1.,2.**-20,2.**20]:
    v=(rng.normal(size=len(worker))+D@rng.normal(size=D.shape[1]))*scale
    result,checks=projection.project(v)
    reference=v-D@np.linalg.lstsq(D,v,rcond=None)[0]
    error=float(np.linalg.norm(result-reference)/np.linalg.norm(v))
    assert checks['pass_oracle'] and error<=1e-11
    assert np.linalg.norm(reference)>0.1*np.linalg.norm(v)
    rows.append(dict(scale=scale,error=error,checks=checks))
receipt=dict(status='PASS',structure=projection.metadata,rows=rows,
    helper_sha256=hashlib.sha256(Path(__file__).with_name('schur_projection_oracle.py').read_bytes()).hexdigest())
with args.output.open('x') as f:json.dump(receipt,f,indent=2)
print('SCHUR_GAUGE_CONTRACT_PASS',len(rows))
