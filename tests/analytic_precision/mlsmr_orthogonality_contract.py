"""Finite analytical MLSMR contract for exactly orthogonal and nonzero FE inputs.

Walsh signs make the within design and residual orthogonal. On the active
half, y=3*x/4; on the other half, x=0 and residual=+/-1. Hence beta=(3/4,0),
RSS=128, slope variance=0 and normalized-constant CRV1=17/[2*(256-rank)].
The exponent200 input retains the unit noise exactly: its rows have x=0.
An additional unit-scale FE signal checks that nonzero FE are still removed.
"""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module',type=Path,required=True)
parser.add_argument('--module-sha256',required=True)
parser.add_argument('--final-inference',choices=('0','1'),required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(args.module)==args.module_sha256 and not args.out.exists()
for name in tuple(os.environ):
    if name.startswith('XHDFE_'):del os.environ[name]
os.environ.update(XHDFE_GPU_BACKEND='cpu',XHDFE_ABSORPTION_CACHE_MODE='off',
    XHDFE_FE_STRUCTURE_MODE='off',XHDFE_MOBILITY_MODE='off',
    XHDFE_ORDINARY_FINAL_INFERENCE=args.final_inference,OMP_DYNAMIC='FALSE',
    OPENBLAS_NUM_THREADS='1')
import numpy as np
spec=importlib.util.spec_from_file_location('py_hdfe_v11',args.module)
cpp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpp)
i=np.arange(256,dtype=np.int64)
sign=lambda bit:(2*((i>>bit)&1)-1).astype(float)
fes=[i&3,(i>>2)&1,(i>>3)&1];cluster=i>>4
error=np.where(i&128,0.,sign(6));rows=[]
for exponent,nonzero_fe in ((0,False),(200,False),(0,True)):
    x=np.where(i&128,np.ldexp(sign(4),exponent),0.)
    y=.75*x+error+(.125*sign(0) if nonzero_fe else 0.)
    exact_y=[Fraction(3,4)*Fraction(float(a))+Fraction(float(b))+
             (Fraction(1,8)*int(s) if nonzero_fe else 0)
             for a,b,s in zip(x,error,sign(0))]
    assert all(Fraction(float(a))==b for a,b in zip(y,exact_y))
    for fe_count,rank in ((1,5),(2,6),(3,7)):
        target=np.zeros((2,2));target[1,1]=float(Fraction(17,2*(256-rank)))
        # Prespecified FP64 scale for exact-zero score directions. Positive
        # variance retains the usual relative target; no global-y floor.
        ssc=16/15*255/(256-rank)
        quantum=64*np.finfo(float).eps*np.sqrt(128/(128*np.ldexp(1.,2*exponent))*ssc)
        root=np.sqrt(np.diag(target));q=np.array([quantum,0.])
        allowed=1e-8*root[:,None]*root[None,:]+q[:,None]*root[None,:]+root[:,None]*q[None,:]+q[:,None]*q[None,:]
        for threads,batch,arena in ((1,'0','0'),(2,'1','0'),(8,'1','1')):
            os.environ.update(XHDFE_MLSMR_BATCH_RHS=batch,XHDFE_MLSMR_TASK_ARENA=arena)
            model=cpp.HdfeRegressor(num_threads=threads,fit_intercept=True,se_type='cluster',
                absorption_method='mlsmr',tol=1e-8,max_iter=1000)
            row=dict(exponent=exponent,nonzero_fe=nonzero_fe,fes=fe_count,threads=threads,
                batch=batch,arena=arena,expected_V=target.tolist(),V_allowance=allowed.tolist())
            try:
                model.fit(y,np.asfortranarray(x[:,None]),fes=fes[:fe_count],clusters=[cluster])
                b=np.asarray(model.coef_);V=np.asarray(model.covariance_);residual=np.asarray(model.residuals_)
                good=(b.shape==(2,) and V.shape==(2,2) and residual.shape==(256,) and
                    np.isfinite(b).all() and np.isfinite(V).all() and np.isfinite(residual).all() and
                    np.max(np.abs(b-[.75,0.]))<=1e-9 and np.max(np.abs(residual-error))<=1e-8 and
                    abs(model.rss_-128)<=128e-8 and np.all(np.abs(V-target)<=allowed) and
                    model.nobs_==256 and model.df_a_==rank-1 and model.df_resid_==15 and
                    model.converged_ and model.precision_certified_ and not model.gpu_used_)
                row.update(passed=bool(good),returned=True,beta=b.tolist(),V=V.tolist(),
                    rss=float(model.rss_),residual_max_error=float(np.max(np.abs(residual-error))),
                    iterations=int(model.num_iterations_))
            except Exception as exc:
                row.update(passed=False,returned=False,error=str(exc),arrays_empty=all(
                    np.asarray(getattr(model,k)).size==0 for k in ('coef_','covariance_','residuals_','sample_index_')))
            rows.append(row)
report=dict(passed=all(r['passed'] for r in rows),rows=rows,module_sha256=args.module_sha256,
    worker_sha256=sha(Path(__file__)),final_inference=args.final_inference,
    scope='27 finite analytical CPU cases; scalar, batch and task arena; no timing or universal certification claim')
assert sha(args.module)==args.module_sha256
with args.out.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'passed':report['passed'],'rows':len(rows),'failures':[r for r in rows if not r['passed']]}))
raise SystemExit(0 if report['passed'] else 1)
