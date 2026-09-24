"""Analytic IV scores, saved-FE reconstruction and exact cache-key controls."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module', type=Path, required=True)
parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
os.environ.update(XHDFE_GPU_BACKEND=args.backend, XHDFE_ABSORPTION_CACHE_MODE='off',
                  XHDFE_MOBILITY_MODE='off', XHDFE_FE_NORMALIZE='component')
spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
cpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp)

i = np.arange(3072)
worker, firm = i // 768, (i // 256) % 3
a, b, c, d = (2 * ((i // (2 ** k)) % 2) - 1 for k in range(4))
X = np.column_stack((a + .5 * b + .25 * worker, c - .5 * firm))
Z = (a + .25 * firm)[:, None]
u = .25 * b + .125 * d
beta = np.array([.75, -.5])
y = X @ beta + 3 + 2 * worker - firm + u
score = np.column_stack((a, c)).astype(float)
actual = np.column_stack((a + .5 * b, c)).astype(float)
dummy = np.column_stack([worker == g for g in range(4)] + [firm == g for g in (1, 2)])
rank = int(np.linalg.matrix_rank(np.column_stack((dummy, X))))
assert rank == 8
rows = []

for weighted in (False, True):
    w = 1. + worker if weighted else np.ones(i.size)
    wn = w * (i.size / w.sum())
    assert np.max(np.abs(score.T @ (w * u))) == 0
    assert abs(actual[:, 0] @ (w * u)) > .1 * w.sum()
    gram = score.T @ (wn[:, None] * actual)
    bread = np.linalg.inv(gram)
    rss = float(wn @ (u * u))
    for mode in ('xhdfe-fast', 'reghdfe-comparable'):
        for vce in ('unadjusted', 'robust'):
            if vce == 'unadjusted':
                reference_v = rss / (i.size - rank) * bread
            else:
                meat = score.T @ (((wn * u) ** 2)[:, None] * score)
                reference_v = (i.size / (i.size - rank)) * bread @ meat @ bread
            for retain in (False, True):
                row = dict(kind='iv', weighted=weighted, mode=mode, vce=vce, retain=retain)
                model = cpp.HdfeRegressor(num_threads=2, max_iter=1000, tol=1e-8,
                    tolerance_mode=mode, drop_singletons=False, retain_fes=retain, se_type=vce)
                try:
                    model.fit(y, X, fes=[worker, firm], weights=w if weighted else None,
                              instruments=Z, endogenous_idx=[0])
                    residual = np.asarray(model.residuals_)
                    covariance = np.asarray(model.covariance_)[:2, :2]
                    vscale = np.sqrt(np.outer(np.diag(reference_v), np.diag(reference_v)))
                    row.update(beta_error=float(np.max(np.abs(np.asarray(model.coef_)[:2] - beta))),
                               residual_error=float(np.max(np.abs(residual - u))),
                               scaled_v_error=float(np.max(np.abs(covariance - reference_v) / vscale)),
                               rss_error=abs(float(model.rss_) - rss),
                               gpu_used=bool(model.gpu_used_), iterations=int(model.num_iterations_))
                    if retain:
                        fitted = X @ np.asarray(model.coef_)[:2] + float(model.coef_[-1])
                        fitted += sum(np.asarray(effect) for effect in model.fe_effects_)
                        row['reconstruction_error'] = float(np.max(np.abs(y - fitted - u)))
                    passed = (row['beta_error'] <= 1e-9 and row['residual_error'] <= 1e-8
                              and row['scaled_v_error'] <= 1e-8 and row['rss_error'] <= 1e-8
                              and row.get('reconstruction_error', 0.) <= 1e-6
                              and model.converged_ and model.precision_certified_
                              and model.df_resid_ == i.size - rank
                              and row['gpu_used'] == (args.backend == 'cuda'))
                    row['status'] = 'PASS' if passed else 'FAIL'
                except Exception as error:
                    row.update(status='FAIL', error=str(error))
                rows.append(row)

if args.backend == 'cpu':
    cache_dir = args.out.parent / (args.out.stem + '_caches')
    cache_dir.mkdir(exist_ok=False)
    X_ols = score + np.column_stack((.25 * worker, -.5 * firm))
    y_ols = X_ols @ beta + 3 + 2 * worker - firm + u
    for mode in ('xhdfe-fast', 'reghdfe-comparable'):
        cache = cache_dir / (mode + '.bin')
        os.environ['XHDFE_ABSORPTION_CACHE'] = str(cache.resolve())
        for operation, shift in (('write', 0.), ('read', 0.), ('read_changed_y', .125)):
            os.environ['XHDFE_ABSORPTION_CACHE_MODE'] = 'write' if operation == 'write' else 'read'
            row = dict(kind='cache', mode=mode, operation=operation)
            model = cpp.HdfeRegressor(num_threads=2, max_iter=1000, tol=1e-8,
                                      tolerance_mode=mode, drop_singletons=False)
            try:
                model.fit(y_ols + shift * X_ols[:, 0], X_ols, fes=[worker, firm])
                row.update(beta_error=float(np.max(np.abs(np.asarray(model.coef_)[:2] -
                                                           (beta + [shift, 0.])))),
                           residual_error=float(np.max(np.abs(np.asarray(model.residuals_) - u))))
                current_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
                if operation == 'write':
                    cache_sha = current_sha
                row['cache_unchanged'] = current_sha == cache_sha
                row['status'] = 'PASS' if (row['beta_error'] <= 1e-9 and row['residual_error'] <= 1e-8
                    and row['cache_unchanged'] and model.converged_ and model.precision_certified_) else 'FAIL'
            except Exception as error:
                row.update(status='FAIL', error=str(error))
            rows.append(row)

counts = {status: sum(row['status'] == status for row in rows) for status in ('PASS', 'FAIL')}
with args.out.open('x') as handle:
    json.dump(dict(module=str(args.module.resolve()), module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
                   backend=args.backend, reference='balanced dyadic IV, explicit rank and independent sandwich',
                   cache_scope='CPU exact read and changed-response signature; hit count is not surfaced',
                   counts=counts, rows=rows), handle, indent=2)
    handle.write('\n')
print(json.dumps(counts))
raise SystemExit(counts['FAIL'] != 0)
