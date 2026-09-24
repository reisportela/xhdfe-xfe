"""One Jacobi fixture: unchanged tolerance, bounded N1 continuation and clean refusal."""
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
    rng = np.random.default_rng(20260912)
    n = 2048
    f0, f1 = rng.integers(0, 31, n), rng.integers(0, 41, n)
    X = rng.normal(size=(n, 2))
    D = np.column_stack([f0 == g for g in range(31)] +
                        [f1 == g for g in range(1, 41)]).astype(float)
    full = np.column_stack((X, D))
    noise = rng.normal(size=n)
    noise -= full @ np.linalg.lstsq(full, noise, rcond=None)[0]
    y = X @ np.array([.75, -.5]) + 64 * (rng.normal(size=31)[f0] +
                                              rng.normal(size=41)[f1]) + noise
    Q, _ = np.linalg.qr(D, mode='reduced')
    within_X = X - Q @ (Q.T @ X)
    within_y = y - Q @ (Q.T @ y)
    bread = np.linalg.inv(within_X.T @ within_X)
    beta = bread @ (within_X.T @ within_y)
    residual = within_y - within_X @ beta
    covariance = (residual @ residual) / (n - full.shape[1]) * bread
    identity = hashlib.sha256(b''.join(a.tobytes() for a in (y, X, f0, f1))).hexdigest()
    return y, X, [f0, f1], D, within_X, beta, residual, covariance, identity


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
    if args.prior_good:
        good = X @ np.array([.75, -.5]) + residual
        model.fit(good, X)
        row['prior_good'] = bool(model.converged_ and
            np.max(np.abs(np.asarray(model.coef_)[:2] - beta)) <= 1e-9)
    try:
        model.fit(y, X, fes=fes)
        actual_beta = np.asarray(model.coef_)[:2]
        actual_V = np.asarray(model.covariance_)[:2, :2]
        actual_u = np.asarray(model.residuals_)
        scale_V = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        row.update(returned=True, converged=bool(model.converged_),
            precision_certified=bool(model.precision_certified_),
            method_used=int(model.absorption_method_used), gpu_used=bool(model.gpu_used_),
            iterations=int(model.num_iterations_),
            beta_error=float(np.max(np.abs(actual_beta - beta))),
            scaled_V_error=float(np.max(np.abs(actual_V - covariance) / scale_V)),
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
parser.add_argument('--max-iter', type=int, default=1000)
parser.add_argument('--prior-good', action='store_true')
args = parser.parse_args()
if args.module:
    worker(args)
    raise SystemExit(0)

root = Path(__file__).resolve().parents[2]
env = {k: v for k, v in os.environ.items() if not k.startswith('XHDFE_')}
env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OMP_NUM_THREADS='2', OMP_DYNAMIC='FALSE')
args.out.mkdir(exist_ok=True)
rows = {}
for name, module in (('baseline', args.baseline), ('candidate', args.candidate)):
    output = args.out / (name + '.json')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--module',
               str(module.resolve()), '--out', str(output.resolve())]
    process = subprocess.run(command, cwd=root, env=env, capture_output=True,
                             text=True, timeout=420)
    save(args.out / (name + '.process.json'), dict(command=command,
        returncode=process.returncode, stdout=process.stdout, stderr=process.stderr))
    process.check_returncode()
    rows[name] = json.loads(output.read_text())

old, new = rows['baseline'], rows['candidate']
status = 'MISSING_COVERAGE'
if old['returned'] and new['returned']:
    same_contract = (old['input_sha256'] == new['input_sha256'] and
        new['method_used'] == old['method_used'] and not new['gpu_used'] and
        new['converged'] and new['precision_certified'] and new['iterations'] <= 1000)
    no_loss = (new['beta_error'] <= max(old['beta_error'], 1e-9) and
               new['scaled_V_error'] <= max(old['scaled_V_error'], 1e-8))
    if not same_contract or not no_loss:
        status = 'FAIL'
    elif new['iterations'] > old['iterations']:
        negative = args.out / 'exhausted_budget.json'
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--module',
            str(args.candidate.resolve()), '--out', str(negative.resolve()),
            '--max-iter', str(old['iterations']), '--prior-good']
        process = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                 text=True, timeout=420)
        save(args.out / 'exhausted_budget.process.json', dict(command=command,
            returncode=process.returncode, stdout=process.stdout, stderr=process.stderr))
        process.check_returncode()
        rows['exhausted_budget'] = bad = json.loads(negative.read_text())
        status = 'PASS' if (bad.get('prior_good') and not bad['returned'] and
            bad.get('arrays_empty') and 'N1:' in bad.get('error', '') and
            'no absorption iteration budget remains' in bad.get('error', '')) else 'FAIL'
elif old['returned'] and not new['returned']:
    status = 'FAIL'
save(args.out / 'RECEIPT.json', dict(status=status, rows=rows,
    worker_sha256=sha(Path(__file__)), reference='explicit FE QR plus independent OLS and covariance',
    scope='one fixed fixture; no runtime claim; missing trigger remains missing coverage'))
print(json.dumps(dict(status=status, rows=rows)), flush=True)
raise SystemExit(status == 'FAIL')
