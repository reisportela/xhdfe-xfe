"""Independent sparse-dummy OLS oracle for the 48-to-47 partition boundary."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--module',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--threads',type=int,default=16)
ap.add_argument('--worker',action='store_true')
args=ap.parse_args();out=args.output.resolve();module=args.module.resolve(strict=True)
assert out.is_relative_to(ROOT) and module.is_relative_to(ROOT)
if not args.worker:
    assert not out.exists();out.mkdir(parents=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('XHDFE_') and k!='LD_LIBRARY_PATH'}
    env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_DYNAMIC='FALSE',
               XHDFE_GPU_BACKEND='cpu',XHDFE_MLSMR_TRACE='1',XHDFE_MOBILITY_MODE='off',
               XHDFE_FE_STRUCTURE_MODE='off',XHDFE_ABSORPTION_CACHE_MODE='off')
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--module',str(module),
             '--output',str(out),'--threads',str(args.threads)]
    with (out/'stdout.log').open('x') as so,(out/'stderr.log').open('x') as se:
        try:rc=subprocess.run(command,cwd=ROOT,env=env,stdout=so,stderr=se,timeout=420).returncode
        except subprocess.TimeoutExpired:rc=124
    raw=json.loads((out/'oracle_raw.json').read_text()) if (out/'oracle_raw.json').exists() else {}
    trace=(out/'stderr.log').read_text()
    def route_evidence(value):
        dispatch=re.search(r'lockstep_memory_dispatch rhs_count=250 row_chunks=(\d+) legacy_decision_chunks=(\d+)[^\n]*lockstep=(\d+)',value)
        batch=re.search(r'batch_rhs_start rhs_count=250 team=(\d+)[^\n]*lockstep=(\d+)',value)
        phase=re.search(r'phase_workers phase=batch_rhs units=250 team=(\d+) active_workers=(\d+)',value)
        ok=(dispatch is not None and dispatch.groups()==('47','48','0') and
            batch is not None and batch.group(2)=='0' and phase is not None and
            1<=int(phase.group(2))<=int(phase.group(1))==int(batch.group(1)))
        return bool(ok),dict(dispatch_present=dispatch is not None,batch_present=batch is not None,
            phase_present=phase is not None,phase_team=int(phase.group(1)) if phase else None,
            phase_useful_workers=int(phase.group(2)) if phase else None)
    route_ok,evidence=route_evidence(trace)
    negative_routes={}
    for tag,token in [('missing_dispatch','lockstep_memory_dispatch'),
                      ('missing_batch','batch_rhs_start'),('missing_phase','phase_workers phase=batch_rhs')]:
        altered='\n'.join(line for line in trace.splitlines() if token not in line)
        negative_routes[tag]=not route_evidence(altered)[0]
    assert all(negative_routes.values())
    actual_lockstep=bool(re.search(r'batch_rhs_start[^\n]*lockstep=1',trace))
    passed=rc==0 and raw.get('numerical_pass') and route_ok
    with (out/'receipt.json').open('x') as f:json.dump(dict(status='PASS' if passed else 'FAIL',
        returncode=rc,actual_lockstep=actual_lockstep,expected_lockstep=False,route_evidence=evidence,
        missing_trace_negatives_rejected=negative_routes,oracle=raw,
        module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
        scope='Synthetic boundary; actual diversion and independent sparse-dummy OLS; no performance certification'),f,indent=2)
    print('PARTITION_BOUNDARY_ORACLE','PASS' if passed else 'FAIL',flush=True)
    raise SystemExit(not passed)

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.sparse.csgraph import connected_components

spec=importlib.util.spec_from_file_location('py_hdfe_v11',module)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
n=266000;p=249;rng=np.random.default_rng(20261009)
f1=(np.arange(n,dtype=np.int64)*10000//n).astype(np.int32)
f2=rng.integers(0,11184,size=n,dtype=np.int32)
assert len(np.unique(f2))==11184 and np.bincount(f2).min()>1
X=np.asfortranarray(rng.standard_normal((n,p)))
true_b=np.zeros(p);true_b[:3]=[.5,-1.25,2.]
y=X@true_b+np.sin(f1*.01)+np.cos(f2*.01)+rng.standard_normal(n)*.125
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
inputs=hashlib.sha256('|'.join(digest(a) for a in [y,X,f1,f2]).encode()).hexdigest()
model=core.HdfeRegressor(num_threads=args.threads,absorption_method='mlsmr',
    tolerance_mode='reghdfe-comparable',tol=1e-8,max_iter=5000,se_type='unadjusted')
model.fit(y,X,fes=[f1,f2])
assert model.converged_ and model.nobs_==n
assert np.array_equal(np.asarray(model.sample_index_),np.arange(n))
rows=np.arange(n);g1=10000;g2=11184
D=sparse.hstack([sparse.csr_matrix((np.ones(n),(rows,f1)),shape=(n,g1)),
                 sparse.csr_matrix((np.ones(n),(rows,f2)),shape=(n,g2))],format='csr')
A=sparse.coo_matrix((np.ones(2*n),(np.r_[f1,g1+f2],np.r_[g1+f2,f1])),shape=(g1+g2,g1+g2)).tocsr()
components=connected_components(A,directed=False,return_labels=False)
rankD=g1+g2-components;df=n-rankD-p
assert model.df_resid_==df
stops=[]
def project(v):
    result=v.copy()
    for _ in range(2):
        fit=lsmr(D,result,atol=1e-14,btol=1e-14,maxiter=200000)
        stops.append(int(fit[1]));assert fit[1] in (0,1,2)
        result-=D@fit[0]
    return result
yt=project(y);xt=np.column_stack([project(X[:,j]) for j in range(p)])
G=xt.T@xt;condition=float(np.linalg.cond(G));assert condition<100
beta=np.linalg.solve(G,xt.T@yt)
beta+=np.linalg.solve(G,xt.T@(yt-xt@beta));residual=yt-xt@beta
sigma2=float(residual@residual)/df
Vb=np.linalg.solve(G,np.eye(p))*sigma2;means=X.mean(axis=0)
b=np.r_[beta,np.mean(y)-means@beta]
V=np.empty((p+1,p+1));V[:p,:p]=Vb
V[:p,p]=-Vb@means;V[p,:p]=V[:p,p];V[p,p]=sigma2/n+means@Vb@means
actual_b=np.asarray(model.coef_);actual_V=np.asarray(model.covariance_)
actual_r=np.asarray(model.residuals_)
scale=np.sqrt(np.outer(np.diag(V),np.diag(V)))
be=float(np.max(np.abs(actual_b-b)/np.maximum(1,np.abs(b))))
ve=float(np.max(np.abs(actual_V-V)/scale))
re_error=float(np.linalg.norm(actual_r-residual)/np.linalg.norm(y))
norms=np.sqrt(np.r_[np.bincount(f1),np.bincount(f2)])
oracle_defect=float(np.max(np.abs(D.T@residual)/norms)/np.linalg.norm(residual))
actual_defect=float(np.max(np.abs(D.T@actual_r)/norms)/np.linalg.norm(actual_r))
passed=be<=1e-9 and ve<=1e-8 and re_error<=8e-9 and oracle_defect<=1e-11 and actual_defect<=1e-9
injected=actual_b.copy();injected[0]+=1e-4
negative_rejected=bool(np.max(np.abs(injected-b)/np.maximum(1,np.abs(b)))>1e-9)
assert negative_rejected
np.savez(out/'oracle_estimates.npz',coef=b,covariance=V)
np.savez(out/'candidate_estimates.npz',coef=actual_b,covariance=actual_V)
raw=dict(numerical_pass=bool(passed),synthetic=True,nobs=n,rhs=p+1,seed=20261009,
    rank_fe=int(rankD),components=int(components),df=int(df),iterations=int(model.num_iterations_),
    b_error=be,V_diagonal_scaled_error=ve,residual_relative_error=re_error,
    oracle_normal_defect=oracle_defect,actual_normal_defect=actual_defect,gram_condition=condition,
    input_sha256=inputs,coef_sha256=digest(actual_b),covariance_sha256=digest(actual_V),
    residual_sha256=digest(actual_r),oracle_stop=max(stops),negative_coefficient_rejected=negative_rejected,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Explicit sparse dummies and OLS; identified slopes, normalised intercept, full conventional covariance, residuals and independently determined sample/rank/df')
with (out/'oracle_raw.json').open('x') as f:json.dump(raw,f,indent=2)
print('ORACLE_NUMERICAL',passed,be,ve,re_error,flush=True)
