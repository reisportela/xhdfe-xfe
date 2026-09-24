#!/usr/bin/env python3
"""Consecutive same-shape CUDA recoveries must refresh FE IDs and weights."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import qr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(XHDFE_GPU_BACKEND="cuda", XHDFE_MOBILITY_MODE="off",
                      XHDFE_FE_STRUCTURE_MODE="off", XHDFE_ABSORPTION_CACHE_MODE="off")
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(921)
    n = 4096
    model = core.HdfeRegressor(num_threads=2, retain_fes=True)
    rows = []
    for case in range(6):
        a = rng.integers(0, 17, n).astype(np.int32)
        b = rng.integers(0, 13, n).astype(np.int32)
        weights = np.ones(n) if case % 2 == 0 else rng.uniform(.5, 2., n)
        x = rng.normal(size=(n, 2))
        y = x @ np.array([.7, -.2]) + rng.normal(size=17)[a]
        y += rng.normal(size=13)[b] + rng.normal(size=n)
        scale = 1e-8 if case >= 4 else 1.
        y *= scale
        d = np.column_stack((np.eye(17)[a], np.eye(13)[b]))
        sw = np.sqrt(weights)
        q, r, _ = qr(d*sw[:, None], mode="economic", pivoting=True)
        rank = int(np.sum(abs(np.diag(r)) > 1e-10))
        q = q[:, :rank]
        xt, yt = x*sw[:, None], y*sw
        xt -= q @ (q.T @ xt)
        yt -= q @ (q.T @ yt)
        beta = np.linalg.lstsq(xt, yt, rcond=None)[0]
        u = yt-xt @ beta
        covariance = (u @ u)/(n-rank-2)*np.linalg.inv(xt.T @ xt)
        model.fit(y, x, [a, b], weights=weights)
        be = float(np.max(abs(model.coef_[:2]-beta)/np.maximum(1, abs(beta))))
        ve = float(np.max(abs(model.covariance_[:2, :2]-covariance)/
            np.sqrt(np.outer(covariance.diagonal(), covariance.diagonal()))))
        total = sum(np.asarray(v) for v in model.fe_effects_) + model.coef_[-1]
        reconstruction = np.max(abs(y-x @ model.coef_[:2]-total-model.residuals_))
        assert be <= 1e-9 and ve <= 1e-8 and reconstruction <= 1e-10*scale
        assert model.gpu_used_ and model.fe_recovery_converged_ and model.precision_certified_
        rows.append(dict(case=case, scale=scale, b_error=be, V_error=ve,
                         reconstruction_error=float(reconstruction), status="PASS"))
    args.output.write_text(json.dumps(rows, indent=2)+"\n")
    print("PASS: six consecutive CUDA recovery graphs/weights, independent QR")


if __name__ == "__main__":
    main()
