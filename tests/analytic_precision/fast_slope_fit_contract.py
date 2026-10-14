"""Real weighted/unweighted slope fits, with and without retained effects.

Run each module/backend in a separate process. The JSON records permit paired
baseline/candidate comparisons, including the full covariance and FE projection.
These small validation cases are separate from the timed AKM experiment.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module', required=True, type=Path)
parser.add_argument('--backend', required=True, choices=('cpu', 'cuda'))
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
os.environ['XHDFE_GPU_BACKEND'] = args.backend
spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
rng = np.random.default_rng(9013)
n = 1200
firm = np.repeat(np.arange(40), n//40).astype('int32')
worker = rng.integers(0, 50, n).astype('int32')
z = rng.normal(size=n)
spline = np.maximum(z, 0.0)
X = np.asfortranarray(rng.normal(size=(n, 2)))
y = X @ np.array([.7, -.2]) + rng.normal(size=50)[worker]
y += rng.normal(size=40)[firm] + rng.normal(size=40)[firm]*z
y += rng.normal(size=40)[firm]*spline + .2*rng.normal(size=n)
D = np.column_stack((np.eye(50)[worker], np.eye(40)[firm],
                     np.eye(40)[firm]*z[:, None], np.eye(40)[firm]*spline[:, None]))
design = np.column_stack((X, D))
rows = []
for weighted in (False, True):
    w = .5 + rng.random(n) if weighted else np.ones(n)
    sw = np.sqrt(w)
    oracle = np.linalg.lstsq(design*sw[:, None], y*sw, rcond=None)[0]
    reference_residual = y - design @ oracle
    for retain in (False, True):
        for mode in ('xhdfe-fast', 'reghdfe-comparable'):
            row = dict(weighted=weighted, retain=retain, mode=mode, backend=args.backend)
            model = core.HdfeRegressor(num_threads=2, tol=1e-8, max_iter=10000,
                                      tolerance_mode=mode, retain_fes=retain,
                                      drop_singletons=False, se_type='cluster')
            try:
                model.fit(y, X, [worker, firm, firm.copy()],
                          weights=w if weighted else None,
                          clusters=np.column_stack((worker, firm)),
                          slopes=[(1, z, True), (2, spline, False)])
                beta = np.asarray(model.coef_)
                residual = np.asarray(model.residuals_)
                row.update(beta=beta.tolist(), covariance=np.asarray(model.covariance_).tolist(),
                           residual=residual.tolist(), N=int(model.nobs_),
                           df_a=int(model.df_a_), df_r=float(model.df_resid_),
                           iterations=int(model.num_iterations_),
                           converged=bool(model.converged_), certified=bool(model.precision_certified_),
                           gpu_used=bool(model.gpu_used_), retries=int(model.slope_accuracy_retry_stages_),
                           block=float(model.slope_block_residual_rel_),
                           checked=bool(model.slope_block_checked_) if hasattr(model, 'slope_block_checked_') else None,
                           beta_oracle_error=float(np.max(np.abs(beta[:2]-oracle[:2]))),
                           residual_oracle_error=float(np.max(np.abs(residual-reference_residual))))
                assert row['N'] == n and row['converged'] and row['certified']
                assert row['gpu_used'] == (args.backend == 'cuda')
                # This independent well-conditioned fixture should agree at
                # the documented fast/comparable offline coefficient limits.
                assert row['beta_oracle_error'] <= (1e-8 if mode=='xhdfe-fast' else 1e-9)
                if row['checked'] is not None:
                    assert row['checked'] == (mode=='reghdfe-comparable')
                if retain:
                    effects=np.asarray(model.fe_effects_)
                    total=effects.sum(axis=0)
                    assert total.shape == (n,)
                    row['recovery_converged']=bool(model.fe_recovery_converged_)
                    row['reconstruction_error']=float(np.max(np.abs(y-X@beta[:2]-beta[-1]-total-residual)))
                    assert row['recovery_converged'] and row['reconstruction_error'] <= 1e-7
                row['status']='PASS'
            except Exception as error:
                row.update(status='FAIL', error=str(error))
            rows.append(row)
result=dict(module=str(args.module), sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
            backend=args.backend, rows=rows, passed=sum(r['status']=='PASS' for r in rows), total=len(rows))
args.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)
raise SystemExit(0 if result['passed']==result['total'] else 1)
