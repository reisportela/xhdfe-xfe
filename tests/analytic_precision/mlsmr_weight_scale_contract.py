#!/usr/bin/env python3
"""Uniform weight scales must preserve projection, inference and convergence."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    os.environ["XHDFE_GPU_BACKEND"] = "cpu"
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(11)
    nw, nf, nc, n = 6000, 1200, 40, 48000
    worker = np.repeat(np.arange(nw), 8)
    origin = rng.integers(0, nc, nw)[worker]
    component = np.where(rng.random(n) < .03, rng.integers(0, nc, n), origin)
    firm = component * (nf // nc) + rng.integers(0, nf // nc, n)
    year = rng.integers(0, 6, n)
    aw, af = rng.normal(size=nw), rng.normal(size=nf)
    x = aw[worker] + af[firm] + rng.normal(size=n)
    y = x + aw[worker] + af[firm] + .2 * year + rng.normal(size=n)
    w = rng.uniform(.5, 2., n)
    fes = [v.astype("int32") for v in (worker, firm, year)]
    blocks = [sp.csr_matrix((np.ones(n), (np.arange(n), f))) for f in fes]
    d = sp.hstack([blocks[0], blocks[1][:, 1:], blocks[2][:, 1:]], format="csc")
    gram = (d.T @ sp.diags(w) @ d).tocsc()
    solver = sl.splu(gram)
    def within(v):
        rhs = d.T @ (w * v)
        coefficients = solver.solve(rhs)
        for _ in range(3):
            coefficients += solver.solve(rhs - gram @ coefficients)
        return v - d @ coefficients
    yt, xt = within(y), within(x)
    xx = np.dot(w * xt, xt)
    b = np.dot(w * xt, yt) / xx
    residual = yt - xt * b
    sigma2 = np.dot(w * residual, residual) / (n - d.shape[1] - 1)
    xb, yb = np.dot(w, x) / w.sum(), np.dot(w, y) / w.sum()
    bref = np.array([b, yb - xb * b])
    vref = sigma2 * np.array([[1/xx, -xb/xx], [-xb/xx, 1/w.sum() + xb*xb/xx]])
    rows = []
    for method in ("mlsmr", "lsmr"):
        for mode in ("reghdfe-comparable", "xhdfe-fast"):
            first = None
            for scale in (1., 2.**-23, 2.**20):
                model = core.HdfeRegressor(num_threads=2, max_iter=1000,
                    absorption_method=method, tolerance_mode=mode, se_type="unadjusted")
                model.fit(y, x[:, None], fes, weights=w * scale)
                bb, vv = np.asarray(model.coef_), np.asarray(model.covariance_)
                be = float(np.max(np.abs(bb - bref) / np.maximum(1, np.abs(bref))))
                ve = float(np.max(np.abs(vv - vref) / np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
                if first is None:
                    first = (bb.copy(), vv.copy())
                invariant = (np.max(np.abs(bb-first[0])) <= 1e-10 and
                    np.max(np.abs(vv-first[1]) / np.sqrt(np.outer(np.diag(vref), np.diag(vref)))) <= 1e-8)
                valid = model.converged_ and model.precision_certified_ and invariant
                if mode == "reghdfe-comparable":
                    valid = valid and be <= 1e-9 and ve <= 1e-8
                rows.append(dict(method=method, mode=mode, scale=scale,
                    b_error=be, V_error=ve, scale_invariant=bool(invariant),
                    iterations=int(model.num_iterations_), status="PASS" if valid else "FAIL"))
    report = dict(module=str(args.module.resolve()),
        sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows,
        fast_scope="Weight-scale invariance; oracle discrepancies reported without adopting the Comparable work target")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" for row in rows), report
    print("PASS: 12 weight-scale/solver/mode cases")


if __name__ == "__main__":
    main()
