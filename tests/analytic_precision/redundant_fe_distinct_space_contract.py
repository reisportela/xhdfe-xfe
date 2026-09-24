"""New FE interactions or within-cell splits must retain their identified space."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from precision_contract import RESIDUAL_ABS

sys.path.insert(0, str(Path(__file__).resolve().parent / 'supplemental'))
from exact_wls_reference import fit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module', type=Path, required=True)
parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
module_sha = hashlib.sha256(args.module.read_bytes()).hexdigest()
os.environ.update(XHDFE_GPU_BACKEND=args.backend, XHDFE_ABSORPTION_CACHE_MODE='off',
                  XHDFE_MOBILITY_MODE='off', XHDFE_FE_NORMALIZE='component')
spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
cpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp)
i = np.arange(256)
cell = i // 64
a, b = cell // 2, cell % 2
A = 2 * (i % 2) - 1
B = 2 * ((i // 2) % 2) - 1
noise = .125 * (2 * ((i // 4) % 2) - 1)
rows = []
for family in ('xor', 'cell_id', 'within_cell_split'):
    effect = B if family == 'within_cell_split' else a ^ b
    extra = (B > 0).astype(int) if family == 'within_cell_split' else (cell if family == 'cell_id' else effect)
    x = A + .5 * effect
    y = .75 * x + 3 * effect + 2 * a - b + noise
    design = np.column_stack((np.ones(256), a, b, effect, x))
    for weighted in (False, True):
        weights = 1. + cell if weighted else np.ones(256)
        reference = fit(design, y, weights)
        expected_v = float(reference['robust_variance']) * 251 / 256
        assert reference['coefficients'][-1] == .75
        assert np.max(np.abs(np.array([float(v) for v in reference['residuals']]) - noise)) == 0
        for mode in ('xhdfe-fast', 'reghdfe-comparable'):
            for retain in (False, True):
                row = dict(family=family, weighted=weighted, mode=mode, retain=retain)
                model = cpp.HdfeRegressor(num_threads=2, max_iter=1000, tol=1e-8,
                    fit_intercept=False, drop_singletons=False, tolerance_mode=mode,
                    se_type='robust', ssc_k_adj=False, retain_fes=retain)
                try:
                    model.fit(y, x[:, None], fes=[a, b, extra], weights=weights if weighted else None)
                    coefficient = float(model.coef_[0])
                    residuals = np.asarray(model.residuals_)
                    row.update(beta_error=abs(coefficient - .75),
                        variance_relative_error=abs(float(model.covariance_[0, 0]) / expected_v - 1),
                        residual_error=float(np.max(np.abs(residuals - noise))),
                        df_resid=float(model.df_resid_), gpu_used=bool(model.gpu_used_))
                    if retain:
                        effects = [np.asarray(v) for v in model.fe_effects_]
                        row['effect_dimensions'] = len(effects)
                        row['reconstruction_error'] = float(np.max(np.abs(y - coefficient * x - sum(effects) - noise)))
                    good = (row['beta_error'] <= 1e-9 and row['variance_relative_error'] <= 1e-8
                        and row['residual_error'] <= RESIDUAL_ABS
                        and row['gpu_used'] == (args.backend == 'cuda')
                        and model.converged_ and model.precision_certified_
                        and (not retain or (row['effect_dimensions'] == 3 and row['reconstruction_error'] <= 1e-6)))
                    row['verdict'] = 'PASS' if good else 'FAIL'
                except Exception as error:
                    row.update(verdict='FAIL_VALID_REFUSAL', error=str(error))
                rows.append(row)
counts = {v: sum(row['verdict'] == v for row in rows) for v in ('PASS', 'FAIL', 'FAIL_VALID_REFUSAL')}
assert hashlib.sha256(args.module.read_bytes()).hexdigest() == module_sha
with args.out.open('x') as handle:
    json.dump(dict(module_sha256=module_sha, backend=args.backend, counts=counts,
                   reference='Exact Fraction WLS and HC0 on the expanded FE space; rank5 including X', rows=rows), handle, indent=2)
    handle.write('\n')
print(json.dumps(counts))
raise SystemExit(counts['FAIL'] != 0 or counts['FAIL_VALID_REFUSAL'] != 0)
