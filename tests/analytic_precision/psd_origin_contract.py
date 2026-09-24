#!/usr/bin/env python3
"""Multiway PSD metric against WLS/CGM/eigenvalue-clamp on exact translations."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


def fixture(seed):
    rng = np.random.default_rng(seed)
    n = 400
    fe = rng.integers(0, 10, n).astype(np.int32)
    clusters = [rng.integers(0, g, n).astype(np.int32) for g in (3, 4)]
    c1, c2 = clusters
    x = rng.normal(size=(n, 3)) + np.column_stack((
        rng.normal(size=3)[c1], rng.normal(size=4)[c2], rng.normal(size=3)[c1]))
    x = np.rint(x*256)/256
    y = x @ np.array([1., .5, -.3]) + rng.normal(size=3)[c1]
    y += rng.normal(size=4)[c2] + rng.normal(size=n)
    return y, x, fe, clusters, rng.integers(1, 4, n).astype(float)


def oracle(y, x, fe, clusters, weights):
    sw = np.sqrt(weights)
    q, _ = np.linalg.qr(sw[:, None]*np.eye(10)[fe], mode="reduced")
    xt = sw[:, None]*x
    yt = sw*y
    xt -= q @ (q.T @ xt)
    yt -= q @ (q.T @ yt)
    b = np.linalg.lstsq(xt, yt, rcond=None)[0]
    residual = (yt - xt @ b)/sw
    score = (xt @ np.linalg.inv(xt.T @ xt))*sw[:, None]*residual[:, None]
    covariance = np.zeros((3, 3))
    labels = (clusters[0][:, None], clusters[1][:, None], np.column_stack(clusters))
    for group, sign in zip(labels, (1, 1, -1)):
        _, ids = np.unique(group, axis=0, return_inverse=True)
        sums = np.zeros((int(ids.max())+1, 3))
        np.add.at(sums, ids, score)
        covariance += sign*sums.T @ sums
    mean = weights @ x/weights.sum()
    sd = np.sqrt((weights[:, None]*(x-mean)**2).sum(axis=0)/(weights.sum()-1))
    sd = np.maximum(sd, .001)
    eigenvalues, vectors = np.linalg.eigh(sd[:, None]*covariance*sd[None, :])
    covariance = (vectors*np.maximum(eigenvalues, 0)) @ vectors.T/sd[:, None]/sd[None, :]
    return b, covariance, float(eigenvalues.min())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    os.environ.update(XHDFE_MOBILITY_MODE="off", XHDFE_FE_STRUCTURE_MODE="off",
                      XHDFE_ABSORPTION_CACHE_MODE="off", XHDFE_GPU_BACKEND=args.backend)
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rows = []
    for seed in (0, 1, 2, 3, 7, 11):
        y, x, fe, clusters, frequencies = fixture(seed)
        for frequency in (False, True):
            weights = frequencies if frequency else np.ones(y.size)
            b, v, negative = oracle(y, x, fe, clusters, weights)
            for offset in (0., 2.**30):
                translated = x.copy()
                translated[:, 0] += offset
                assert np.array_equal(translated[:, 0]-offset, x[:, 0])
                for intercept in (False, True):
                    model = core.HdfeRegressor(num_threads=1, se_type="cluster",
                        fit_intercept=intercept, ssc_g_adj=False, ssc_k_adj=False)
                    model.fit(y, translated, [fe], clusters=clusters,
                              weights=weights if frequency else None, fweights=frequency)
                    assert bool(model.gpu_used_) == (args.backend == "cuda")
                    be = float(np.max(abs(model.coef_[:3]-b)/np.maximum(1, abs(b))))
                    ve = float(np.max(abs(model.covariance_[:3, :3]-v)/
                        np.sqrt(np.outer(v.diagonal(), v.diagonal()))))
                    rows.append(dict(seed=seed, frequency=frequency, offset=offset,
                        report_intercept=intercept, b_error=be, V_error=ve,
                        raw_min_eigenvalue=negative,
                        status="PASS" if be <= 1e-9 and ve <= 1e-8 else "FAIL"))
    args.output.write_text(json.dumps(rows, indent=2)+"\n")
    assert all(row["status"] == "PASS" for row in rows), rows
    print("PASS: 48 PSD/translation/frequency/intercept cases against WLS/CGM/clamp")


if __name__ == "__main__":
    main()
