#!/usr/bin/env python3
"""Saved-FE units, b/full-V and RSS against an independent sparse projection.

Forward FE errors are diagnostics. This test does not invent a forward-error
interpretation of the legacy max-update fe_tolerance.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as graph
import scipy.sparse.linalg as sl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    assert not args.output.exists()
    os.environ["XHDFE_GPU_BACKEND"] = args.backend
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(1)
    n, ga, gb = 20000, 300, 60
    a = rng.integers(0, ga, n)
    b = a * gb // ga
    move = rng.random(n) < .01
    b[move] = rng.integers(0, gb, move.sum())
    af, bf = rng.normal(size=ga), rng.normal(size=gb)
    x = rng.normal(size=n) + .3*af[a]
    y = .5*x + af[a] + bf[b] + rng.normal(size=n)
    d = sp.hstack((sp.csr_matrix((np.ones(n), (np.arange(n), a)), shape=(n, ga)),
                   sp.csr_matrix((np.ones(n), (np.arange(n), b)), shape=(n, gb))), format="csr")
    components = graph.connected_components(d.T @ d, directed=False, return_labels=False)
    df = n - (ga + gb - components) - 1
    residualized = np.column_stack((y, x))
    for col in range(2):
        for _ in range(3):
            residualized[:, col] -= d @ sl.lsmr(
                d, residualized[:, col], atol=1e-15, btol=1e-15, maxiter=50000)[0]
    yr, xr = residualized.T
    beta = float(xr @ yr / (xr @ xr))
    u = yr - xr*beta
    sigma2 = float(u @ u / df)
    vb = sigma2/(xr @ xr)
    base_b = np.array([beta, np.mean(y - x*beta)])
    base_v = np.array([[vb, -x.mean()*vb],
                       [-x.mean()*vb, sigma2/n + x.mean()**2*vb]])
    rows = []
    for scale in (1., 1e-8, 1e3):
        transform = np.diag([1., scale])
        bref, vref = transform @ base_b, transform @ base_v @ transform
        for method in ("hybrid", "map"):
            model = core.HdfeRegressor(num_threads=2, retain_fes=True,
                                       fe_recovery_method=method, max_iter=100000)
            model.fit(y*scale, (x*scale)[:, None], [a, b])
            assert np.array_equal(model.sample_index_, np.arange(n))
            berror = float(np.max(np.abs(model.coef_ - bref)/np.maximum(1, np.abs(bref))))
            verror = float(np.max(np.abs(model.covariance_ - vref)/
                                 np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
            rss_error = float(abs(model.residuals_ @ model.residuals_ - model.rss_)/model.rss_)
            component = sum(np.asarray(v) for v in model.fe_effects_) + model.coef_[-1]
            reference_fe = scale*(y - x*beta - u)
            fe_error = float(np.linalg.norm(component-reference_fe)/np.linalg.norm(reference_fe))
            assert berror <= 1e-9 and verror <= 1e-8 and rss_error <= 1e-8
            assert model.fe_recovery_converged_ and model.precision_certified_
            assert bool(model.gpu_used_) == (args.backend == "cuda")
            rows.append(dict(scale=scale, method=method, b_error=berror, V_error=verror,
                             RSS_inconsistency=rss_error, forward_FE_relative_L2=fe_error,
                             recovery_iterations=int(model.fe_recovery_iterations_),
                             max_delta=float(model.fe_recovery_max_delta_), status="PASS"))
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("PASS: six FE recovery scale/method cases; independent b and full covariance")


if __name__ == "__main__":
    main()
