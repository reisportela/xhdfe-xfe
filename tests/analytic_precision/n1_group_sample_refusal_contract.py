"""Exact regression for the undue N1 Strict/group-only CUDA refusal.

No diagnostic bypass: the ordinary product must return the contract-correct fit.
"""
import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import os
from pathlib import Path

BASE_Y = [0.43266510274897235, 0.7110058158405258, -0.25109036013826286, 0.2679479879139899, -0.3031235313682709, -1.0414322579394106, 0.03063190021181643, -0.14600581786767539, 0.6122594944195144, -0.4119218303095315, 0.5188426024099575, 1.3488793007921493, 1.0342165010531845, 0.4173997879162797, 0.695803836058226]
BASE_X = [[0.5024980092008311, -0.38379260833719925], [0.8120291791803402, -0.9150496019115383], [-0.043620139542810174, 0.8627051380319435], [0.9951054362025179, 1.4241782692569482], [-0.16488423183650885, 0.48318391275850686], [-1.3465709421659326, -0.04779589496958562], [-0.5967047883623221, -0.8684016463776124], [-0.4777469238346593, 0.35780239985349577], [1.022038940825072, 2.2932057546721962], [-0.7815433584423291, 0.36838302335827255], [0.7470410156360834, 0.8949714661540591], [1.884151674572773, 0.7789948449298696], [1.9033124394533338, 0.7438491361885807], [-0.03168184508281835, -0.14135490943170145], [0.10408891642253093, -0.9701480852751176]]
FIRST = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 2]
SECOND = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 3]

def transpose(A):
    return list(map(list, zip(*A)))

def multiply(A, B):
    return [[sum(x*y for x, y in zip(row, column)) for column in zip(*B)] for row in A]

def inverse(A):
    n = len(A)
    augmented = [row.copy() + [F(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if augmented[i][j])
        augmented[j], augmented[pivot] = augmented[pivot], augmented[j]
        diagonal = augmented[j][j]
        augmented[j] = [value/diagonal for value in augmented[j]]
        for i in range(n):
            if i != j:
                factor = augmented[i][j]
                augmented[i] = [a-factor*b for a, b in zip(augmented[i], augmented[j])]
    return [row[n:] for row in augmented]

def reference():
    y = list(map(F.from_float, BASE_Y))
    X = [[F.from_float(value) for value in row] for row in BASE_X]
    n = len(y)
    D = [[F(int(a == level)) for level in sorted(set(FIRST))[1:]] +
         [F(int(b == level)) for level in sorted(set(SECOND))] for a, b in zip(FIRST, SECOND)]
    projection = multiply(multiply(D, inverse(multiply(transpose(D), D))), transpose(D))
    raw = [[value] + row for value, row in zip(y, X)]
    removed = multiply(projection, raw)
    within = [[a-b for a, b in zip(row, fitted)] for row, fitted in zip(raw, removed)]
    x = [row[1:] for row in within]
    bread = inverse(multiply(transpose(x), x))
    beta = [row[0] for row in multiply(multiply(bread, transpose(x)), [[row[0]] for row in within])]
    residual = [row[0]-sum(a*b for a, b in zip(row[1:], beta)) for row in within]
    mean_x = [sum(row[j] for row in X)/n for j in range(2)]
    intercept = sum(y)/n-sum(a*b for a, b in zip(mean_x, beta))
    side = [-sum(bread[j][k]*mean_x[k] for k in range(2)) for j in range(2)]
    augmented = [bread[j] + [side[j]] for j in range(2)] + [side +
        [F(1, n) + sum(mean_x[j]*bread[j][k]*mean_x[k] for j in range(2) for k in range(2))]]
    rss = sum(value*value for value in residual)
    df = n-len(D[0])-len(beta)
    covariance = [[rss*value/df for value in row] for row in augmented]
    return beta+[intercept], covariance, residual, rss, df

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--module-sha256', required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    module = args.module.resolve()
    assert sha(module) == args.module_sha256 and not args.out.exists()
    for name in tuple(os.environ):
        if name.startswith('XHDFE_'):
            del os.environ[name]
    os.environ.update(XHDFE_GPU_BACKEND=args.backend, XHDFE_CERTIFY='0',
        XHDFE_ABSORPTION_CACHE_MODE='off', XHDFE_MOBILITY_MODE='off',
        XHDFE_FE_STRUCTURE_MODE='off', OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1')
    import numpy as np
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', module)
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    assert Path(cpp.__file__).resolve() == module
    b, V, residual, rss, df = reference()
    b = np.asarray(list(map(float, b)))
    V = np.asarray([[float(value) for value in row] for row in V])
    residual = np.asarray(list(map(float, residual)))
    ids = np.repeat(np.arange(15), 2+np.arange(15)%4)
    y, X = np.asarray(BASE_Y), np.asarray(BASE_X)
    first, second = np.asarray(FIRST), np.asarray(SECOND)
    model = cpp.HdfeRegressor(num_threads=2, drop_singletons=False, fit_intercept=True,
        tolerance_mode='strict-residual', tol=1e-12, retain_fes=True, se_type='unadjusted')
    row = dict(status='FAIL', backend=args.backend, module_sha256=args.module_sha256,
        worker_sha256=sha(Path(__file__)), oracle='exact Fraction FE projector and homoskedastic OLS; rank8, df7',
        reference_beta=b.tolist(), reference_V=V.tolist(), reference_rss=float(rss),
        acceptance_scope='Identified slopes/covariance and reconstruction; normalized intercept is diagnostic only')
    try:
        model.fit(y[ids], X[ids], fes=[first[ids], second[ids]], group=ids)
        selected = np.asarray(model.sample_index_, dtype=int)
        expected = np.array([np.flatnonzero(ids == g)[0] for g in range(15)])
        np.testing.assert_array_equal(selected, expected)
        actual_b, actual_V = np.asarray(model.coef_), np.asarray(model.covariance_)
        actual_u = np.asarray(model.residuals_)
        beta_error = float(np.max(np.abs(actual_b[:2]-b[:2])/np.maximum(1, np.abs(b[:2]))))
        covariance_error = float(np.max(np.abs(actual_V[:2, :2]-V[:2, :2])/np.sqrt(np.diag(V)[:2])[:, None]/np.sqrt(np.diag(V)[:2])[None, :]))
        residual_error = float(np.max(np.abs(actual_u-residual)))
        effects = [np.asarray(value) for value in model.fe_effects_]
        assert len(effects) == 2 and all(value.shape == y.shape for value in effects)
        reconstruction_error = float(np.max(np.abs(y-X@actual_b[:2]-actual_b[-1]-sum(effects)-actual_u)))
        assert beta_error <= 1e-9 and covariance_error <= 1e-8
        # This is stored saveFE output; retain its existing1e-6 recovery contract.
        assert residual_error <= 1e-6 and abs(model.rss_-float(rss)) <= 1e-10
        assert reconstruction_error <= 1e-6
        assert model.df_a_ == 6 and model.df_resid_ == df
        assert model.converged_ and model.precision_certified_ and model.fe_recovery_converged_
        assert bool(model.gpu_used_) == (args.backend == 'cuda')
        if args.backend == 'cuda':
            assert model.gpu_status_code_ == 1
        row.update(status='PASS', beta_error=beta_error, covariance_scaled_error=covariance_error,
                   residual_error=residual_error, reconstruction_error=reconstruction_error,
                   normalized_intercept_error_diagnostic=float(abs(actual_b[-1]-b[-1])),
                   iterations=int(model.num_iterations_))
    except Exception as error:
        row.update(error=str(error), arrays_empty=all(not np.asarray(getattr(model, name)).size
            for name in ('coef_', 'covariance_', 'residuals_', 'sample_index_')))
    row['custody_unchanged'] = sha(module) == args.module_sha256
    with args.out.open('x') as output:
        json.dump(row, output, indent=2)
        output.write('\n')
    print(json.dumps(row))
    return int(row['status'] != 'PASS' or not row['custody_unchanged'])

if __name__ == '__main__':
    raise SystemExit(main())
