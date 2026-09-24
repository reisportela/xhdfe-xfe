"""Deliberate N1 failure verifies one continuation and transactional cleanup."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module', type=Path, required=True)
parser.add_argument('--hook', type=Path, required=True)
parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
module, hook_path = args.module.resolve(), args.hook.resolve()
before = dict(module=sha(module), hook=sha(hook_path))
os.environ.update(XHDFE_GPU_BACKEND=args.backend, XHDFE_ABSORPTION_CACHE_MODE='off',
                  XHDFE_MOBILITY_MODE='off', XHDFE_FE_STRUCTURE_MODE='off')
hook = ctypes.CDLL(str(hook_path), mode=ctypes.RTLD_GLOBAL)
hook.n1_test_set_original.argtypes = [ctypes.c_void_p]
hook.n1_test_set_original.restype = ctypes.c_int
hook.n1_test_original_path.restype = ctypes.c_char_p
hook.n1_test_hook_address.restype = ctypes.c_void_p
hook.n1_test_arm.argtypes = [ctypes.c_int]
hook.n1_test_calls.restype = ctypes.c_int
hook.n1_test_forwarded.restype = ctypes.c_int
hook.n1_test_tolerance.argtypes = [ctypes.c_int]
hook.n1_test_tolerance.restype = ctypes.c_double

# dlsym on the actual module handle must resolve the module's own definition.
native_handle = ctypes.CDLL(str(module), mode=ctypes.RTLD_LOCAL)
symbol = '_ZNK4hdfe6detail16N1NormalEvidence7requireEd'
original = ctypes.cast(getattr(native_handle, symbol), ctypes.c_void_p).value
assert original != hook.n1_test_hook_address(), 'dlsym resolved the interposer, not the real function'
assert hook.n1_test_set_original(original) == 1
original_path = Path(hook.n1_test_original_path().decode()).resolve()
assert original_path == module, 'real function does not belong to the identified module'
spec = importlib.util.spec_from_file_location('py_hdfe_v11', module)
cpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp)

i = np.arange(3072)
worker, firm = i // 768, (i // 256) % 3
x = (2 * (i % 2) - 1).astype(float)
u = .25 * (2 * ((i // 2) % 2) - 1).astype(float)
X = x[:, None]
y = .75 * x + 3 + 2 * worker - firm + u
rows = []


def evidence():
    count = hook.n1_test_calls()
    return dict(check_calls=count, forwarded_calls=hook.n1_test_forwarded(),
                tolerances=[hook.n1_test_tolerance(k) for k in range(count)])


def snapshot(model):
    return dict(iterations=int(model.num_iterations_), method=int(model.absorption_method_used),
        gpu_used=bool(model.gpu_used_), gpu_status=int(model.gpu_status_code_),
        state=str(model.lifecycle_state_), generation=int(model.generation_),
        beta_error=float(abs(np.asarray(model.coef_)[0] - .75)),
        residual_error=float(np.max(np.abs(np.asarray(model.residuals_) - u))),
        converged=bool(model.converged_), certified=bool(model.precision_certified_))


for mode, tolerance in (('xhdfe-fast', 1e-8), ('reghdfe-comparable', 1e-9)):
    model = cpp.HdfeRegressor(num_threads=2, max_iter=1000, tol=1e-8,
        tolerance_mode=mode, absorption_method='gauss-seidel', drop_singletons=False)
    hook.n1_test_arm(0)
    model.fit(y, X, fes=[worker, firm])
    good = snapshot(model)
    good.update(evidence(), mode=mode, armed=0)
    good['passed'] = (good['check_calls'] == good['forwarded_calls'] == 1 and
        good['tolerances'] == [tolerance] and good['beta_error'] <= 1e-9 and
        good['residual_error'] <= 1e-8 and good['gpu_used'] == (args.backend == 'cuda'))
    rows.append(good)

    hook.n1_test_arm(1)
    row = dict(mode=mode, armed=1)
    try:
        model.fit(y, X, fes=[worker, firm])
        row.update(snapshot(model), **evidence())
        row['passed'] = (row['check_calls'] == 2 and row['forwarded_calls'] == 1 and
            row['tolerances'] == [tolerance, tolerance] and row['method'] == good['method'] and
            row['gpu_used'] == good['gpu_used'] and row['gpu_status'] == good['gpu_status'] and
            row['iterations'] > good['iterations'] and row['iterations'] <= 1000 and
            row['converged'] and row['certified'] and row['beta_error'] <= 1e-9 and
            row['residual_error'] <= 1e-8)
    except Exception as error:
        row.update(passed=False, error=str(error), **evidence())
    rows.append(row)

    hook.n1_test_arm(2)
    row = dict(mode=mode, armed=2)
    try:
        model.fit(y, X, fes=[worker, firm])
        row.update(passed=False, error='two injected failures returned estimates', **evidence())
    except Exception as error:
        row.update(error=str(error), state=str(model.lifecycle_state_),
            arrays_empty=all(np.asarray(getattr(model, name)).size == 0
                for name in ('coef_', 'covariance_', 'residuals_', 'sample_index_')),
            converged=bool(model.converged_), **evidence())
        row['passed'] = (row['check_calls'] == 2 and row['forwarded_calls'] == 0 and
            row['tolerances'] == [tolerance, tolerance] and row['arrays_empty'] and
            row['state'] == 'failed' and not row['converged'] and
            'the single N1 refinement was exhausted' in row['error'])
    rows.append(row)

after = dict(module=sha(module), hook=sha(hook_path))
passed = before == after and all(row['passed'] for row in rows)
report = dict(classification='FAULT_INJECTION_NEGATIVE_CONTROL', passed=passed,
    backend=args.backend, module=str(module), original_function_library=str(original_path),
    original_pointer=hex(original), hook_pointer=hex(hook.n1_test_hook_address()),
    hashes_before=before, hashes_after=after, worker_sha256=sha(Path(__file__)), rows=rows,
    scope='test-process-only ELF interception; not a natural estimator error; real check forwarded on accepted fit')
with args.out.open('x') as handle:
    json.dump(report, handle, indent=2)
    handle.write('\n')
print(json.dumps(dict(passed=passed, rows=rows)), flush=True)
raise SystemExit(not passed)
