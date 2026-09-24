"""Fixed cycle incidence: exercise N1 continuation/refusal at the unchanged Fast tolerance."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def fixture():
    row = np.arange(4096)
    first = row // 16
    second = (first + (row // 8) % 2) % 256
    X = (2 * (row % 2) - 1).astype(float)[:, None]
    noise = (2 * ((row // 2) % 2) - 1).astype(float)
    angle = 2 * np.pi * np.arange(256) / 256
    a = 64 * (1 + np.sin(4 * angle))
    b = 64 * (1 + np.cos(4 * angle))
    y = .75 * X[:, 0] + a[first] + b[second] + noise
    D = np.column_stack([first == g for g in range(256)] +
                        [second == g for g in range(1, 256)]).astype(float)
    Q, _ = np.linalg.qr(D, mode='reduced')
    within_X = X - Q @ (Q.T @ X)
    within_y = y - Q @ (Q.T @ y)
    bread = np.linalg.inv(within_X.T @ within_X)
    beta = bread @ (within_X.T @ within_y)
    residual = within_y - within_X @ beta
    covariance = (residual @ residual) / (y.size - D.shape[1] - 1) * bread
    assert abs(beta[0] - .75) < 1e-11
    assert np.max(np.abs(residual - noise)) < 1e-9
    identity = hashlib.sha256(b''.join(v.tobytes() for v in (y, X, first, second))).hexdigest()
    return y, X, [first, second], D, within_X, beta, residual, covariance, identity


def worker(args):
    os.environ.update(XHDFE_GPU_BACKEND='cpu', XHDFE_ABSORPTION_CACHE_MODE='off',
                      XHDFE_MOBILITY_MODE='off', XHDFE_FE_STRUCTURE_MODE='off')
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    y, X, fes, D, within_X, beta, residual, covariance, identity = fixture()
    model = cpp.HdfeRegressor(num_threads=2, max_iter=args.max_iter,
        absorption_method='jacobi', tol=1e-8, tolerance_mode='xhdfe-fast',
        drop_singletons=False, fit_intercept=True, se_type='unadjusted')
    row = dict(module=str(args.module.resolve()), module_sha256=sha(args.module),
        input_sha256=identity, maximum_iterations=args.max_iter,
        method_requested='jacobi', tolerance=1e-8, mode='xhdfe-fast')
    model.fit(.75 * X[:, 0] + residual, X)
    row['prior_good'] = bool(model.converged_ and
        abs(np.asarray(model.coef_)[0] - .75) < 1e-9)
    try:
        model.fit(y, X, fes=fes)
        actual_u = np.asarray(model.residuals_)
        actual_b = np.asarray(model.coef_)[0]
        actual_V = np.asarray(model.covariance_)[0, 0]
        row.update(returned=True, converged=bool(model.converged_),
            certified=bool(model.precision_certified_),
            method_used=int(model.absorption_method_used), gpu_used=bool(model.gpu_used_),
            iterations=int(model.num_iterations_), rho=float(model.abs_residual_rel_),
            beta_error=float(abs(actual_b - beta[0])),
            scaled_V_error=float(abs(actual_V - covariance[0, 0]) / covariance[0, 0]),
            residual_error=float(np.max(np.abs(actual_u - residual))),
            fe_cosine=float(np.max(np.abs(D.T @ actual_u) /
                (np.linalg.norm(D, axis=0) * np.linalg.norm(actual_u)))),
            normal_cosine=float(np.max(np.abs(within_X.T @ actual_u) /
                (np.linalg.norm(within_X, axis=0) * np.linalg.norm(actual_u)))))
    except Exception as error:
        row.update(returned=False, error=str(error), converged=bool(model.converged_),
            arrays_empty=all(np.asarray(getattr(model, name)).size == 0
                for name in ('coef_', 'covariance_', 'residuals_', 'sample_index_')))
    save(args.out, row)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', type=Path)
parser.add_argument('--candidate', type=Path)
parser.add_argument('--module', type=Path)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--max-iter', type=int, default=100000)
args = parser.parse_args()
if args.module:
    worker(args)
    raise SystemExit(0)

root = Path(__file__).resolve().parents[2]
env = {k: v for k, v in os.environ.items() if not k.startswith('XHDFE_')}
env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OMP_NUM_THREADS='2', OMP_DYNAMIC='FALSE')
args.out.mkdir(exist_ok=True)


def run(name, module, maximum=100000):
    output = args.out / (name + '.json')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--module',
        str(module.resolve()), '--out', str(output.resolve()), '--max-iter', str(maximum)]
    process = subprocess.run(command, cwd=root, env=env, capture_output=True,
                             text=True, timeout=420)
    save(args.out / (name + '.process.json'), dict(command=command,
        returncode=process.returncode, stdout=process.stdout, stderr=process.stderr))
    process.check_returncode()
    return json.loads(output.read_text())


rows = dict(baseline=run('baseline', args.baseline), candidate=run('candidate', args.candidate))
old, new = rows['baseline'], rows['candidate']
status = 'MISSING_COVERAGE'
coverage = 'No executed N1 continuation/refusal was established.'
same_input = old['input_sha256'] == new['input_sha256']
continued = (old['returned'] and new['returned'] and
             new['iterations'] > old['iterations'])
refused_after_continuation = (old['returned'] and not new['returned'] and
    new.get('prior_good') and new.get('arrays_empty') and
    'N1: FE moment failed' in new.get('error', '') and
    'the single N1 refinement was exhausted' in new.get('error', ''))
if continued or refused_after_continuation:
    bad = rows['exhausted_budget'] = run('exhausted_budget', args.candidate, old['iterations'])
    clean_budget_refusal = (bad.get('prior_good') and not bad['returned'] and
        bad.get('arrays_empty') and 'N1: FE moment failed' in bad.get('error', '') and
        'no absorption iteration budget remains' in bad.get('error', ''))
    positive = refused_after_continuation or (new['converged'] and new['certified'] and
        not new['gpu_used'] and new['method_used'] == old['method_used'] and
        new['iterations'] <= 100000 and new['beta_error'] <= max(old['beta_error'], 1e-9) and
        new['scaled_V_error'] <= max(old['scaled_V_error'], 1e-8))
    status = 'PASS' if (same_input and positive and clean_budget_refusal) else 'FAIL'
    coverage = ('Single same-contract continuation then necessary FE refusal; prior estimates cleared.'
        if refused_after_continuation else 'Single same-contract continuation succeeded; exhausted-budget refusal cleared prior estimates.')
elif old['returned'] and not new['returned']:
    status = 'FAIL'
save(args.out / 'RECEIPT.json', dict(status=status, coverage=coverage, rows=rows,
    worker_sha256=sha(Path(__file__)),
    reference='explicit FE QR and OLS on represented inputs; Walsh exact beta=.75 and u=+/-1',
    scope='one fixed cycle; functional branch coverage, not runtime acceptance or universal precision'))
print(json.dumps(dict(status=status, coverage=coverage, rows=rows)), flush=True)
raise SystemExit(status == 'FAIL')
