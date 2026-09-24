#!/usr/bin/env python3
"""A nuisance constant must not turn a defective projection into a certificate."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr, lsqr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-known-cuda-refusals", action="store_true",
                        help="grade the represented offset1e9 cases as containment if they refuse cleanly")
    args = parser.parse_args()
    os.environ.update(XHDFE_MOBILITY_MODE="off", XHDFE_FE_STRUCTURE_MODE="off",
                      XHDFE_ABSORPTION_CACHE_MODE="off", XHDFE_GPU_BACKEND=args.backend)
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(1)
    n, workers, firms = 20000, 300, 60
    f1 = rng.integers(0, workers, n).astype(np.int32)
    f2 = (f1*firms//workers).astype(np.int32)
    movers = rng.random(n) < .01
    f2[movers] = rng.integers(0, firms, movers.sum())
    alpha, gamma = rng.normal(size=workers), rng.normal(size=firms)
    x0 = rng.normal(size=n) + .3*alpha[f1]
    y0 = .5*x0 + alpha[f1] + gamma[f2] + rng.normal(size=n)
    d = csr_matrix((np.ones(2*n), (np.repeat(np.arange(n), 2),
                    np.column_stack((f1, workers+f2)).ravel())), shape=(n, workers+firms))
    graph = csr_matrix((np.ones(n), (f1, workers+f2)), shape=(workers+firms, workers+firms))
    rank = workers + firms - connected_components(graph + graph.T, directed=False)[0]

    def reference(y, x, alternate=False):
        def project(v):
            shifted = v-v[0]  # The represented shifted inputs, not the original fixture.
            coefficients = (lsqr(d, shifted, atol=1e-14, btol=1e-14, iter_lim=5000)[0]
                            if alternate else lsmr(d, shifted, atol=1e-14, btol=1e-14, maxiter=5000)[0])
            return shifted-d @ coefficients
        yt, xt = project(y), project(x)
        b = xt @ yt/(xt @ xt)
        residual = yt-xt*b
        sigma2 = (residual @ residual)/(n-rank-1)
        v = sigma2/(xt @ xt)
        mx = float(np.mean(x, dtype=np.longdouble))
        my = float(np.mean(y, dtype=np.longdouble))
        return np.array([b, my-mx*b]), np.array([[v, -mx*v], [-mx*v, sigma2/n+mx*mx*v]])

    rows = []
    model = core.HdfeRegressor(num_threads=2)
    for offset in (0., 1e3, 1e6, 1e9):
        for which in (("both",) if offset == 0 else ("y", "x", "both")):
            y = y0 + (offset if which in ("y", "both") else 0)
            x = x0 + (offset if which in ("x", "both") else 0)
            b, v = reference(y, x)
            b2, _ = reference(y, x, alternate=True)
            assert abs(b[0]-b2[0]) <= 1e-12
            try:
                model.fit(y, x[:, None], [f1, f2])
            except RuntimeError as error:
                try:
                    assert np.asarray(model.coef_).size == 0
                except RuntimeError:
                    pass
                contained = args.allow_known_cuda_refusals and args.backend == "cuda" and offset == 1e9
                rows.append(dict(offset=offset, which=which,
                    status="CONTAINED" if contained else "REFUSED", error=str(error)))
                continue
            assert bool(model.gpu_used_) == (args.backend == "cuda")
            be = float(np.max(abs(model.coef_-b)/np.maximum(1, abs(b))))
            ve = float(np.max(abs(model.covariance_-v)/np.sqrt(np.outer(v.diagonal(), v.diagonal()))))
            rows.append(dict(offset=offset, which=which, b_error=be, V_error=ve,
                iterations=model.num_iterations_, certified=bool(model.precision_certified_),
                status="PASS" if be <= 1e-9 and ve <= 1e-8 else "FAIL"))
    args.output.write_text(json.dumps(rows, indent=2)+"\n")
    assert all(row["status"] == "CONTAINED" or
               (row["status"] == "PASS" and row["certified"]) for row in rows), rows
    model.fit(y0, x0[:, None], [f1, f2])
    assert model.converged_ and bool(model.gpu_used_) == (args.backend == "cuda")
    contained = sum(row["status"] == "CONTAINED" for row in rows)
    print(f"PASS: {len(rows)-contained} numerical origin cases; {contained} known extreme refusals contained; refit valid")


if __name__ == "__main__":
    main()
