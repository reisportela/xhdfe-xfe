"""Validate the large public synthetic Walsh fixture and its exact oracle."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
df=pd.read_stata(a.fixture,convert_categoricals=False)
G=262144;N=12*G
assert len(df)==N
g1=df.g1.to_numpy(dtype=np.int64).reshape(G,3,4)
g2=df.g2.to_numpy(dtype=np.int64).reshape(G,3,4)
ids=np.arange(G)
assert np.all(g1==ids[:,None,None])
assert np.all(g2==g2[:,:,:1])
assert np.array_equal(g2[:,0,0],ids)
assert np.array_equal(g2[:,1,0],(ids+1)%G)
assert np.array_equal(g2[:,2,0],np.random.default_rng(20261008).permutation(ids))
x1=df.x1.to_numpy().reshape(-1,4);x2=df.x2.to_numpy().reshape(-1,4)
assert np.all(x1==np.array([1.,-1.,1.,-1.]))
assert np.all(x2==np.array([1.,1.,-1.,-1.]))
noise=df.noise.to_numpy()
assert np.array_equal(noise,df.x1.to_numpy()*df.x2.to_numpy())
expected=.5*df.x1.to_numpy()-1.25*df.x2.to_numpy()+(df.g1.to_numpy(dtype=float)+df.g2.to_numpy(dtype=float))*2.**-20+noise
assert np.array_equal(df.y.to_numpy(),expected)
rank=2*G-1;degrees=N-rank-2
receipt=dict(status='PASS',observations=N,groups_per_FE=G,rank_FE=rank,df_resid=degrees,
    beta=[.5,-1.25],rss=N,XtX_diagonal=[N,N],V_diagonal_exact=['1/'+str(degrees)]*2,
    proof='Each four-row edge block has zero sums of x1, x2, noise, x1*noise and x2*noise. Identity and successor edges connect the bipartite FE graph, proving rank 2G-1. All input arithmetic is exact dyadic arithmetic.',
    fixture_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest())
with a.output.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
