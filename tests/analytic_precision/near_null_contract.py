"""Analytic guard against false convergence along almost-null FE directions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.linalg import block_diag

parser = argparse.ArgumentParser()
parser.add_argument('--module',type=Path,required=True)
parser.add_argument('--backend',choices=['cpu','cuda'],required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--legacy',action='store_true')
parser.add_argument('--nodes',type=int,default=96)
args = parser.parse_args()
if args.nodes < 4: parser.error('--nodes must be at least 4')
os.environ['XHDFE_GPU_BACKEND'] = args.backend
spec = importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
n = args.nodes
sequence = [1,-1,1]
for i in range(3,n): sequence.append(-sequence[i-1]-sequence[i-3])
T = np.eye(n)+np.eye(n,k=1)+np.eye(n,k=3)
M = block_diag(T,T)
base = np.repeat(np.arange(2*n),16)
rep = np.tile(np.arange(16),2*n)
a,b = 2*(rep%2)-1,2*((rep>>1)%2)-1
edges = [(g,i) for g in range(len(base)) for i in np.flatnonzero(M[base[g]])]
g,individual = np.asarray(edges,dtype=np.int64).T
rows = []
for aggregation in ('sum','mean'):
    scale = np.ones(2*n) if aggregation=='sum' else M.sum(axis=1)
    design = M/scale[:,None]
    for weight in ('','aw','fw'):
        w = np.ones(len(base)) if not weight else 1.+base%n%3
        wg = 1.+np.arange(n)%3 if weight else np.ones(n)
        weak = np.asarray(sequence,dtype=float)/max(abs(v) for v in sequence)
        weak = np.r_[weak,-weak]*scale/np.r_[wg,wg]
        weak /= np.linalg.norm(weak)/np.sqrt(2.)
        good_x = .2*a
        good_y = .7*good_x+(design@(.2*np.cos(np.arange(2*n))))[base]+.113*b
        bad_x = weak[base]+.2*a
        bad_y = .7*bad_x+3*weak[base]+.113*b
        reference_rss = float(np.dot(w,(.113*b)**2))
        if weight=='aw': reference_rss *= len(base)/w.sum()
        for mode,tol in [('reghdfe-comparable',1e-8),('strict-residual',1e-12)]:
            reg = module.HdfeRegressor(num_threads=2,max_iter=100000,drop_singletons=False,
                                       tolerance_mode=mode,tol=tol,se_type='unadjusted')
            kwargs = dict(fes=[individual,np.zeros(len(g),dtype=np.int64)],group=g,
                          individual=individual,aggregation=aggregation)
            if weight: kwargs.update(weights=w[g],fweights=weight=='fw')
            row = dict(aggregation=aggregation,weight=weight,mode=mode,backend=args.backend,
                       nodes=n,groups=len(base),patterns=2*n)
            start = time.monotonic()
            try:
                reg.fit(good_y[g],good_x[g,None],**kwargs)
                assert reg.converged_ and abs(reg.coef_[0]-.7)<1e-8
                if args.backend == 'cuda': assert reg.gpu_used_ and reg.gpu_status_code_ == 1
                row['prior_good_fit'] = True
                try:
                    reg.fit(bad_y[g],bad_x[g,None],**kwargs)
                    correct = abs(reg.coef_[0]-.7)<1e-8 and abs(reg.rss_-reference_rss)<1e-7
                    row.update(status='CORRECT_RESULT' if correct else 'WRONG_RESULT',
                               beta=float(reg.coef_[0]),rss=reg.rss_,reference_rss=reference_rss,
                               converged=bool(reg.converged_),certified=bool(reg.precision_certified_))
                except Exception as error:
                    empty = not np.asarray(reg.coef_).size and not np.asarray(reg.sample_index_).size and not reg.converged_
                    row.update(status='SAFE_ERROR' if empty and 'no estimates' in str(error).lower() else 'UNSAFE_ERROR',
                               error=str(error),state_empty=bool(empty))
            except Exception as error:
                row.update(status='GOOD_FIT_REGRESSION',error=str(error))
            row['seconds'] = time.monotonic()-start
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='error'}),flush=True)
with args.out.open('x') as f:json.dump(rows,f,indent=2)
if not args.legacy:assert all(r['status'] in ('SAFE_ERROR','CORRECT_RESULT') for r in rows)
