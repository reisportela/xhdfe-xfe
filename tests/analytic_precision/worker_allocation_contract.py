"""Linux synthetic worker bad_alloc regression and process recovery contract."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys


def worker(args):
    import numpy as np
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(7)
    n, g1, g2, p = (1200000, 200000, 20000, 2) if args.case == 'dense' else (300000, 100000, 60000, 40)
    f1 = np.sort(rng.integers(0, g1, n)).astype(np.int32)
    f2 = rng.integers(0, g2, n).astype(np.int32)
    X = np.asfortranarray(rng.normal(size=(n, p)))
    y = X[:, :2]@np.array([.5, -1.25])+rng.normal(size=n)
    reg = core.HdfeRegressor(num_threads=4, absorption_method='mlsmr',
        tolerance_mode='reghdfe-comparable', tol=1e-8, max_iter=5000)
    # Populate a previous successful result, then require failure to clear it.
    reg.fit(y[:100], X[:100, :2], fes=[])
    lib = ctypes.CDLL(None)
    lib.shim_hits.restype = ctypes.c_long
    mode = 2 if args.fail_at >= 0 else 0
    minimum = 4096 if args.case == 'dense' else 150000
    lib.shim_arm(ctypes.c_int(mode), ctypes.c_long(args.fail_at), ctypes.c_long(minimum))
    failed = False
    try:
        reg.fit(y, X, fes=[f1, f2])
    except MemoryError:
        failed = True
        assert not reg.converged_, 'Failed fit retained a converged result'
    finally:
        lib.shim_arm(ctypes.c_int(0), ctypes.c_long(-1), ctypes.c_long(minimum))
    assert failed == (args.fail_at >= 0), 'Expected worker allocation failure was not observed'
    if failed:
        reg.fit(y, X, fes=[f1, f2])
    assert reg.converged_
    def digest(value):
        return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
    print('ALLOCATION_CONTRACT '+json.dumps(dict(case=args.case, failed_as_expected=failed,
        recovery_converged=bool(reg.converged_), iterations=int(reg.num_iterations_),
        coef_sha256=digest(reg.coef_), covariance_sha256=digest(reg.covariance_),
        residual_sha256=digest(reg.residuals_))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', type=Path, required=True)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--case', choices=['dense', 'wide'])
    ap.add_argument('--fail-at', type=int, default=-1)
    args = ap.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.case:
        worker(args)
        return
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    source = Path(__file__).with_name('worker_allocation_shim.c')
    shim = out/'worker_malloc.so'
    subprocess.run(['cc', '-shared', '-fPIC', '-O2', str(source), '-o', str(shim), '-ldl'], check=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('XHDFE_') and k != 'LD_LIBRARY_PATH'}
    env.update(LD_PRELOAD=str(shim), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_DYNAMIC='FALSE', XHDFE_GPU_BACKEND='cpu')
    rows = []
    baselines = {}
    for case, at in [('dense', -1), ('dense', 1), ('dense', 3),
                     ('wide', -1), ('wide', 1000), ('wide', 2500)]:
        label = case+'_'+str(at)
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--module', str(args.module.resolve()),
                   '--case', case, '--fail-at', str(at)]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=420)
        for suffix, content in [('stdout', result.stdout), ('stderr', result.stderr)]:
            with (out/(label+'.'+suffix)).open('x') as f:
                f.write(content)
        markers = [line for line in result.stdout.splitlines() if line.startswith('ALLOCATION_CONTRACT ')]
        row = dict(case=case, fail_at=at, returncode=result.returncode, status='FAIL')
        if result.returncode == 0 and len(markers) == 1:
            payload = json.loads(markers[0].split(' ', 1)[1])
            row.update(payload)
            if at < 0:
                baselines[case] = payload
                row['status'] = 'PASS'
            else:
                offsets = re.findall(r'py_hdfe_v11[^ ]*\+(0x[0-9a-f]+)', result.stderr)
                names = subprocess.run(['addr2line', '-f', '-C', '-e', str(args.module.resolve()), *offsets],
                                       capture_output=True, text=True, check=True).stdout
                with (out/(label+'.site')).open('x') as f:
                    f.write(names)
                expected = 'compute_keep_mask_drop_singletons' if case == 'dense' else 'absorb_fixed_effects_mlsmr'
                equal = all(payload[k] == baselines[case][k] for k in
                            ['iterations', 'coef_sha256', 'covariance_sha256', 'residual_sha256'])
                row['correct_failure_site'] = expected in names
                row['recovery_matches_uninjected'] = equal
                if expected in names and equal:
                    row['status'] = 'PASS'
        rows.append(row)
        print('WORKER_ALLOCATION', label, row['status'], result.returncode, flush=True)
    report = dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL', rows=rows,
        module=str(args.module.resolve()), module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shim_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    with (out/'receipt.json').open('x') as f:
        json.dump(report, f, indent=2)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
