#!/usr/bin/env python3
"""IV absorption cache: exact-key reuse, one-ULP input changes and sample changes.

The absorption cache key hashes y, the combined [X Z] design, weights, FE ids
and the solver options (hash_absorption_signature); reads are refused with
heterogeneous slopes or under a GPU request, so this contract is CPU only.
The reference is explicit 2SLS with explicit FE dummies on the represented
inputs (numpy least squares), independent of the cache and of the dyadic
exactness of the design. The hit flag is the XHDFE_DEBUG_SOLVER stderr line
"cache_hit=<0|1>" written by the fit; no hit count is otherwise surfaced.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
for key in [k for k in os.environ if k.startswith('XHDFE_')]:
    del os.environ[key]
os.environ.update(XHDFE_GPU_BACKEND='cpu', XHDFE_DEBUG_SOLVER='1', XHDFE_MOBILITY_MODE='off',
                  XHDFE_FE_NORMALIZE='component')
spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
cpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp)


def design(n=3072):
    i = np.arange(n)
    worker, firm = i // 768, (i // 256) % 3
    a, b, c, d = (2. * ((i // 2 ** k) % 2) - 1 for k in range(4))
    X = np.column_stack((a + .5 * b + .25 * worker, c - .5 * firm))
    Z = (a + .25 * firm)[:, None].astype(float)
    y = X @ np.array([.75, -.5]) + 3 + 2 * worker - firm + .25 * b + .125 * d
    return dict(y=y.copy(), X=X.copy(), Z=Z.copy(),
                fes=[worker.astype(np.int32), firm.astype(np.int32)])


def reference(case):
    """Explicit 2SLS on the represented inputs with explicit FE dummies."""
    y, X, Z = case['y'], case['X'], case['Z']
    w, f = case['fes']
    D = np.column_stack([w == g for g in np.unique(w)] + [f == g for g in np.unique(f)[1:]]).astype(float)
    W = np.column_stack((D, Z, X[:, 1:]))
    R = np.column_stack((D, X))
    R_hat = W @ np.linalg.lstsq(W, R, rcond=None)[0]
    b = np.linalg.lstsq(R_hat, y, rcond=None)[0]
    return b[-2:], y - R @ b


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit(case, mode, cache_mode):
    os.environ['XHDFE_ABSORPTION_CACHE_MODE'] = cache_mode
    saved = os.dup(2)
    capture = tempfile.TemporaryFile()
    os.dup2(capture.fileno(), 2)
    try:
        model = cpp.HdfeRegressor(num_threads=2, max_iter=1000, tol=1e-8, tolerance_mode=mode,
                                  drop_singletons=False)
        model.fit(case['y'], case['X'], fes=case['fes'], instruments=case['Z'], endogenous_idx=[0])
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    capture.seek(0)
    text = capture.read().decode(errors='replace')
    capture.close()
    hits = re.findall(r'cache_hit=(\d)', text)
    return model, (int(hits[-1]) if hits else None)


rows = []
cache_dir = args.out.parent / (args.out.stem + '_caches')
cache_dir.mkdir(exist_ok=False)


def record(scenario, mode, case, cache_mode, expected_hit, baseline=None):
    row = dict(scenario=scenario, mode=mode, cache_mode=cache_mode, expected_hit=expected_hit)
    try:
        model, hit = fit(case, mode, cache_mode)
        slopes, residual = reference(case)
        coef = np.asarray(model.coef_)
        resid = np.asarray(model.residuals_)
        cov = np.asarray(model.covariance_)
        row.update(cache_hit=hit, beta_error=float(np.max(np.abs(coef[:2] - slopes))),
                   residual_error=float(np.max(np.abs(resid - residual))),
                   converged=bool(model.converged_), certified=bool(model.precision_certified_),
                   nobs=int(model.nobs_), iterations=int(model.num_iterations_))
        if baseline is not None:
            row['bit_identical_to_write'] = bool(np.array_equal(coef, baseline['coef']) and
                                                 np.array_equal(cov, baseline['cov']) and
                                                 np.array_equal(resid, baseline['resid']))
        ok = (hit == expected_hit and row['beta_error'] <= 1e-9 and row['residual_error'] <= 1e-8
              and row['converged'] and row['certified'] and row.get('bit_identical_to_write', True))
        row['status'] = 'PASS' if ok else 'FAIL'
        rows.append(row)
        return dict(coef=coef, cov=cov, resid=resid)
    except Exception as error:
        row.update(status='FAIL', error=str(error))
        rows.append(row)
        return None


for mode in ('xhdfe-fast', 'reghdfe-comparable'):
    cache = cache_dir / (mode + '.bin')
    os.environ['XHDFE_ABSORPTION_CACHE'] = str(cache.resolve())
    written = record('write', mode, design(), 'write', 0)
    sha_written = sha(cache) if cache.exists() else None
    record('read_identical', mode, design(), 'read', 1, written)
    case = design()
    case['Z'][0, 0] = np.nextafter(case['Z'][0, 0], np.inf)
    record('read_instrument_one_ulp', mode, case, 'read', 0)
    case = design()
    case['y'][5] = np.nextafter(case['y'][5], np.inf)
    record('read_response_one_ulp', mode, case, 'read', 0)
    case = design()
    case['X'][7, 1] = np.nextafter(case['X'][7, 1], -np.inf)
    record('read_regressor_one_ulp', mode, case, 'read', 0)
    case = design()
    for key in ('y', 'X', 'Z'):
        case[key] = case[key][:-256]
    case['fes'] = [fe[:-256] for fe in case['fes']]
    record('read_sample_change', mode, case, 'read', 0)
    other = 'reghdfe-comparable' if mode == 'xhdfe-fast' else 'xhdfe-fast'
    record('read_other_mode', other, design(), 'read', 0)
    rows.append(dict(scenario='cache_unchanged_after_reads', mode=mode, cache_sha256=sha_written,
                     status='PASS' if sha_written and sha(cache) == sha_written else 'FAIL'))

counts = {status: sum(row['status'] == status for row in rows) for status in ('PASS', 'FAIL')}
with args.out.open('x') as handle:
    json.dump(dict(module=str(args.module.resolve()), module_sha256=sha(args.module),
                   reference='explicit 2SLS with FE dummies on the represented inputs',
                   scope='CPU only: cache reads are refused under a GPU request and with slopes; '
                         'hit flag from XHDFE_DEBUG_SOLVER stderr; one-ULP changes in Z, y and X, '
                         'a 256-row sample change and the other tolerance mode must miss; '
                         'the identical read must hit and be bit-identical to the writing fit',
                   counts=counts, rows=rows), handle, indent=2)
    handle.write('\n')
print(json.dumps(counts))
raise SystemExit(counts['FAIL'] != 0)
