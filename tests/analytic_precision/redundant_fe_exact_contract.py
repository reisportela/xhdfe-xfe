"""Exact WLS invariance to constant, duplicate and recoded redundant FE."""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'supplemental'))
from exact_wls_reference import fit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module', type=Path, required=True)
parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--savefe', action='store_true')
args = parser.parse_args()
module_sha = hashlib.sha256(args.module.read_bytes()).hexdigest()
os.environ.update(XHDFE_GPU_BACKEND=args.backend, XHDFE_ABSORPTION_CACHE_MODE='off',
                  XHDFE_MOBILITY_MODE='off', XHDFE_FE_NORMALIZE='component')
spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
cpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp)
i = np.arange(64)
cell = i // 16
a, b = cell // 2, cell % 2
x = (cell == 1).astype(float)
noise = .125 * (2 * (i % 2) - 1)
y = .75 * x + b - a + noise
design = np.column_stack((np.ones(64), a, b, x))
families = {
    'two': [a, b],
    'constant': [a, b, np.zeros(64, dtype=int)],
    'duplicate': [a, b, a],
    'two_duplicates': [a, b, a, b],
    'leading_constant': [np.zeros(64, dtype=int), b, a],
    'recoded_duplicates': [a, b, np.where(a == 0, 29, 11), np.where(b == 0, 7, 900001)],
}
rows, oracles = [], []
for exponent in (0, 40, 80, 120):
    weights = np.where(a == b, 1., 2. ** -exponent)
    reference = fit(design, y, weights)
    assert reference['coefficients'][-1] == Fraction(3, 4)
    assert reference['residuals'] == [Fraction(float(v)) for v in noise]
    expected_v = float(reference['variance'])
    expected_rss = float(sum(Fraction(float(w)) * Fraction(float(u)) ** 2
                             for w, u in zip(weights, noise)) * 64 /
                         sum(Fraction(float(w)) for w in weights))
    oracles.append(dict(exponent=exponent, beta=str(reference['coefficients'][-1]),
                        variance=str(reference['variance']), rank=4, df=60))
    for family, fes in families.items():
        for mode in ('xhdfe-fast', 'reghdfe-comparable'):
            for retain in ((False, True) if args.savefe else (False,)):
                row = dict(family=family, exponent=exponent, mode=mode, retain=retain)
                model = cpp.HdfeRegressor(num_threads=2, fit_intercept=False,
                    drop_singletons=False, max_iter=1000, tol=1e-8,
                    tolerance_mode=mode, retain_fes=retain, se_type='unadjusted')
                try:
                    model.fit(y, x[:, None], fes=fes, weights=weights)
                    coefficients = np.asarray(model.coef_)
                    residuals = np.asarray(model.residuals_)
                    variance = float(model.covariance_[0, 0])
                    row.update(beta=float(coefficients[0]), beta_error=abs(float(coefficients[0]) - .75),
                        variance=variance, variance_relative_error=abs(variance / expected_v - 1.),
                        residual_error=float(np.max(np.abs(residuals - noise))),
                        rss_error=abs(float(model.rss_) - expected_rss),
                        iterations=int(model.num_iterations_), df_resid=float(model.df_resid_),
                        method=int(model.absorption_method_used), gpu_used=bool(model.gpu_used_))
                    if retain:
                        effects = [np.asarray(v) for v in model.fe_effects_]
                        row['effect_dimensions'] = len(effects)
                        row['reconstruction_error'] = float(np.max(np.abs(y - coefficients[0] * x - sum(effects) - noise)))
                    beta_limit = 1e-8 if mode == 'xhdfe-fast' else 1e-9
                    variance_limit = 1e-12 + 1e-7 * abs(expected_v) if mode == 'xhdfe-fast' else 1e-8 * abs(expected_v)
                    passed = (row['beta_error'] <= beta_limit and abs(variance - expected_v) <= variance_limit
                        and row['residual_error'] <= 1e-8 and row['rss_error'] <= 1e-9
                        and row['df_resid'] == 60 and model.converged_ and model.precision_certified_
                        and row['gpu_used'] == (args.backend == 'cuda')
                        and (not retain or (row['effect_dimensions'] == len(fes) and row['reconstruction_error'] <= 1e-6)))
                    row['verdict'] = 'PASS' if passed else 'FAIL'
                except Exception as error:
                    row.update(verdict='FAIL_VALID_REFUSAL', error=str(error))
                rows.append(row)
counts = {v: sum(row['verdict'] == v for row in rows) for v in ('PASS', 'FAIL', 'FAIL_VALID_REFUSAL')}
assert hashlib.sha256(args.module.read_bytes()).hexdigest() == module_sha
with args.out.open('x') as handle:
    json.dump(dict(module=str(args.module.resolve()), module_sha256=module_sha,
                   backend=args.backend, exact_oracles=oracles, counts=counts, rows=rows), handle, indent=2)
    handle.write('\n')
print(json.dumps(counts))
raise SystemExit(counts['FAIL'] != 0 or counts['FAIL_VALID_REFUSAL'] != 0)
