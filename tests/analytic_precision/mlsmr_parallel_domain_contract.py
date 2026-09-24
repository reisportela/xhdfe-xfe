#!/usr/bin/env python3
"""MLSMR single-RHS OpenMP safety and explicit-dummy projection oracle.

Run with env -u LD_LIBRARY_PATH and --module pointing to the native extension.
Workers isolate a native crash; every generated file stays in --scratch.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr


def worker(args):
    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(5)
    n = 40000
    component = rng.integers(0, 4, n)
    w = (component * (n // 16) + rng.integers(0, n // 16, n)).astype("int32")
    f = (component * 50 + rng.integers(0, 50, n)).astype("int32")
    y = rng.normal(size=int(w.max()) + 1)[w]
    y += rng.normal(size=int(f.max()) + 1)[f] + rng.normal(size=n)
    model = core.HdfeRegressor(num_threads=args.threads, absorption_method="mlsmr",
                               tolerance_mode=args.mode)
    for repeat in range(2):
        model.fit(y, np.empty((n, 0)), [w, f])
        assert model.converged_ and model.precision_certified_
        sample = np.asarray(model.sample_index_, dtype=int)
        # The FE incidence matrix is independent of xhdfe's indexers/solver.
        rows = np.arange(sample.size)
        d = sparse.hstack([sparse.csr_matrix((np.ones(sample.size), (rows, v[sample])))
                           for v in (w, f)], format="csr")
        oracle = y[sample].copy()
        for _ in range(2):
            oracle -= d @ lsmr(d, oracle, atol=1e-14, btol=1e-14,
                               maxiter=100000)[0]
        residual = np.asarray(model.residuals_)
        if residual.size == n and sample.size != n:
            residual = residual[sample]
        error = float(np.linalg.norm(residual - oracle) / np.linalg.norm(y[sample]))
        # Fast retains the public 1e-8 target; Comparable has the 1e-9 floor.
        limit = 8e-8 if args.mode == "xhdfe-fast" else 8e-9
        assert error <= limit, (args.threads, args.mode, error)
        assert np.max(np.abs(d.T @ oracle)) <= 1e-9 * np.linalg.norm(y[sample])
    print(json.dumps(dict(status="PASS", threads=args.threads, mode=args.mode,
                          threads_used=int(model.threads_used_), iterations=int(model.num_iterations_),
                          projection_error=error, b=np.asarray(model.coef_).tolist(),
                          V=np.asarray(model.covariance_).tolist(),
                          residual_sha256=hashlib.sha256(residual.tobytes()).hexdigest())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mode", default="reghdfe-comparable")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.scratch.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in ("xhdfe-fast", "reghdfe-comparable"):
        for threads in (1, 2, 4):
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       XHDFE_MLSMR_PARALLEL_APPLY="1", XHDFE_MLSMR_BATCH_RHS="0")
            run = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                  "--worker", "--module", str(args.module.resolve()),
                                  "--scratch", str(args.scratch.resolve()),
                                  "--threads", str(threads), "--mode", mode],
                                 cwd=args.scratch, env=env, capture_output=True,
                                 text=True, timeout=420)
            row = dict(mode=mode, threads=threads, returncode=run.returncode)
            if run.returncode:
                row.update(status="FAIL", stdout=run.stdout, stderr=run.stderr)
            else:
                row.update(json.loads(run.stdout.strip().splitlines()[-1]))
            results.append(row)
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=results)
    (args.scratch / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    assert all(row["status"] == "PASS" for row in results), report
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
