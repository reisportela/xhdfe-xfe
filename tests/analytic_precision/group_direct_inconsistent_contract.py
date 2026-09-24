#!/usr/bin/env python3
"""Ill-conditioned incidence, exact rational oracle from an equivalent FE span.

Each triangular T block has diagonal one, so the membership columns span all
pattern indicators. Adding year gives the balanced pattern/year projection,
without solving the ill-conditioned matrix or using xhdfe as a reference.
"""
import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy.linalg import block_diag


def fixture(size, aggregation, seed, consistent):
    rng = np.random.default_rng(seed)
    t = np.eye(size) + np.eye(size, k=1) + np.eye(size, k=3)
    incidence = block_diag(t, t)
    patterns = 2 * size
    seq = [1, -1, 1]
    for j in range(3, size):
        seq.append(-seq[j-1] - seq[j-3])
    weak = np.asarray(seq, dtype=float) / max(abs(v) for v in seq)
    weak = np.r_[weak, -weak]
    if aggregation == "mean":
        weak *= incidence.sum(axis=1)
    weak /= np.linalg.norm(weak) / np.sqrt(2.)
    p = np.repeat(np.arange(patterns), 8)
    year = np.tile(np.repeat([0, 1], 4), patterns)
    copy = np.tile(np.arange(4), patterns * 2)
    u = np.round(rng.normal(size=(patterns, 2)) * 2**10) / 2**10
    v = np.round(rng.normal(size=(patterns, 2)) * 2**10) / 2**10
    if consistent:
        u.fill(0)
        v.fill(0)
    x = weak[p] + .3 * v[p, year] + .2 * np.where(copy < 2, 1., -1.)
    y = .7 * x + 3 * weak[p] + .5 * year + .05 * u[p, year]
    y += np.where(copy % 2 == 0, .125, -.125)
    edges = [(row, individual) for row, pattern in enumerate(p)
             for individual in np.flatnonzero(incidence[pattern])]
    g, individual = np.asarray(edges, dtype=np.int64).T
    return y, x, p, year, g, individual


def rational_reference(y, x, p, year):
    n = y.size
    patterns = int(p.max()) + 1
    def project(values):
        q = [F(float(v)) for v in values]
        total = sum(q) / n
        means_p = [sum(q[row] for row in range(8 * k, 8 * k + 8)) / 8
                   for k in range(patterns)]
        means_year = [sum(q[row] for row in range(n) if year[row] == k) / (n // 2)
                      for k in range(2)]
        return [q[row] - means_p[p[row]] - means_year[year[row]] + total
                for row in range(n)], total
    ry, ymean = project(y)
    rx, xmean = project(x)
    xx = sum(v * v for v in rx)
    beta = sum(a * b for a, b in zip(rx, ry)) / xx
    residual = [a - beta * b for a, b in zip(ry, rx)]
    rss = sum(v * v for v in residual)
    # Rank(pattern FE + year FE)=patterns+1; plus one identified regressor.
    sigma2 = rss / (n - patterns - 2)
    var = sigma2 / xx
    covariance = [[var, -xmean * var],
                  [-xmean * var, sigma2 / n + xmean * xmean * var]]
    return [beta, ymean - xmean * beta], rss, covariance


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
    rows = []
    for size in (24, 42, 54):
        for seed in (1, 2):
            for aggregation in ("sum", "mean"):
                for consistent in (False, True):
                    y, x, p, year, group, individual = fixture(size, aggregation, seed, consistent)
                    bref, rssref, vv = rational_reference(y, x, p, year)
                    vref = np.array([[float(v) for v in line] for line in vv])
                    for exact in (False, True):
                        # The documented default counts all membership columns
                        # for nonuniform sum teams. Exact DoF is a distinct option.
                        expected_v = vref
                        if not exact and aggregation == "sum":
                            expected_v = vref * (y.size - 2 * size - 2) / (y.size - 2 * size - 3)
                        for mode in ("xhdfe-fast", "reghdfe-comparable"):
                            model = core.HdfeRegressor(num_threads=2, drop_singletons=False,
                                tolerance_mode=mode, se_type="unadjusted",
                                dofadjustments="exact" if exact else None)
                            try:
                                model.fit(y[group], x[group, None], [individual, year[group]],
                                          group=group, individual=individual, aggregation=aggregation)
                            except RuntimeError as error:
                                rows.append(dict(size=size, seed=seed, aggregation=aggregation,
                                    exact_dof=exact, consistent=consistent, mode=mode,
                                    status="REFUSAL", certified=False, error=str(error)))
                                continue
                            b = np.asarray(model.coef_)
                            berror = max(float(abs(F(float(b[j])) - bref[j]) / max(1, abs(bref[j])))
                                         for j in range(2))
                            rss_error = float(abs(F(float(model.rss_)) - rssref) / rssref)
                            verror = float(np.max(np.abs(np.asarray(model.covariance_) - expected_v) /
                                np.sqrt(np.outer(np.diag(expected_v), np.diag(expected_v)))))
                            rows.append(dict(size=size, seed=seed, aggregation=aggregation, exact_dof=exact,
                                consistent=consistent, mode=mode, b_error=berror, RSS_error=rss_error,
                                V_error=verror, certified=bool(model.precision_certified_),
                                gpu_used=bool(model.gpu_used_),
                                status="PASS" if berror <= 1e-9 and verror <= 1e-8 and rss_error <= 1e-8
                                and bool(model.gpu_used_) == (args.backend == "cuda") else "FAIL"))
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" and row["certified"] for row in rows), report
    print("PASS: 96 group/individual cases with separate default/exact DoF contracts")


if __name__ == "__main__":
    main()
