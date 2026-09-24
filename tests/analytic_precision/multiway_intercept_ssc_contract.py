#!/usr/bin/env python3
"""Full covariance from an independent dummy-WLS influence matrix.

Tests Conventional/Min, per-component corrections, omitted columns and
frequency weights. No xhdfe output enters the mathematical reference.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np


def fixture():
    rng = np.random.default_rng(11)
    n = 3000
    fe = rng.integers(0, 60, n).astype("int32")
    c1 = rng.integers(0, 8, n).astype("int32")
    c2 = rng.integers(0, 150, n).astype("int32")
    x = rng.normal(size=(n, 2)) + np.column_stack(
        [rng.normal(size=8)[c1], rng.normal(size=150)[c2]])
    y = x @ np.array([1., -2.]) + rng.normal(size=60)[fe]
    y += rng.normal(size=8)[c1] + rng.normal(size=150)[c2] + rng.normal(size=n)
    weights = rng.integers(1, 4, n).astype(float)
    return y, x, fe, [c1, c2], weights


def oracle(y, x, fe, clusters, weights, conventional, adjusted):
    w = np.ones(y.size) if weights is None else weights
    d = np.eye(60)[fe]
    design = np.column_stack((x, d))
    # Full rank: every FE has observations and the two regressors vary within FE.
    sw = np.sqrt(w)
    q, r = np.linalg.qr(design * sw[:, None], mode="reduced")
    beta = np.linalg.solve(r, q.T @ (sw * y))
    residual = y - design @ beta
    transform = np.zeros((3, design.shape[1]))
    transform[:2, :2] = np.eye(2)
    transform[2, 2:] = (w @ d) / w.sum()
    influence = transform @ np.linalg.solve(r, q.T) * sw[None, :]
    score = influence.T * residual[:, None]
    parts = []
    for labels in (clusters[0][:, None], clusters[1][:, None],
                   np.column_stack(clusters)):
        _, inverse = np.unique(labels, axis=0, return_inverse=True)
        g = int(inverse.max()) + 1
        totals = np.zeros((g, 3))
        np.add.at(totals, inverse, score)
        parts.append((g, totals.T @ totals))
    covariance = np.zeros((3, 3))
    for sign, (g, meat) in zip((1, 1, -1), parts):
        covariance += sign * (g / (g - 1) if conventional and adjusted else 1) * meat
    if not conventional and adjusted:
        g = min(parts[0][0], parts[1][0])
        covariance *= g / (g - 1)
    assert np.linalg.eigvalsh(covariance).min() > 0, "fixture needs no PSD correction"
    return transform @ beta, covariance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    y, x, fe, clusters, weights = fixture()
    rows = []
    for frequency in (False, True):
        w = weights if frequency else None
        for gdf in ("min", "conventional"):
            for adjusted in (False, True):
                bref, vref = oracle(y, x, fe, clusters, w, gdf == "conventional", adjusted)
                for omitted in (False, True):
                    xx = np.column_stack((x, x[:, 0] + x[:, 1])) if omitted else x
                    model = core.HdfeRegressor(num_threads=2, se_type="cluster",
                                              ssc_g_df=gdf, ssc_g_adj=adjusted,
                                              ssc_k_adj=False)
                    model.fit(y, xx, [fe], weights=w, fweights=frequency, clusters=clusters)
                    selected = [0, 1, xx.shape[1]]
                    b = np.asarray(model.coef_)[selected]
                    v = np.asarray(model.covariance_)[np.ix_(selected, selected)]
                    berror = float(np.max(np.abs(b - bref) / np.maximum(1, np.abs(bref))))
                    verror = float(np.max(np.abs(v - vref) /
                                         np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
                    row = dict(frequency=frequency, gdf=gdf, g_adj=adjusted, omitted=omitted,
                               b_error=berror, V_error=verror, certified=bool(model.precision_certified_),
                               status="PASS" if berror <= 1e-9 and verror <= 1e-8 else "FAIL")
                    if frequency:
                        index = np.repeat(np.arange(y.size), weights.astype(int))
                        replicated = core.HdfeRegressor(num_threads=2, se_type="cluster",
                                                       ssc_g_df=gdf, ssc_g_adj=adjusted,
                                                       ssc_k_adj=False)
                        replicated.fit(y[index], xx[index], [fe[index]],
                                       clusters=[c[index] for c in clusters])
                        vv = np.asarray(replicated.covariance_)[np.ix_(selected, selected)]
                        row["replication_error"] = float(np.max(np.abs(v - vv) /
                            np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
                        if row["replication_error"] > 1e-8:
                            row["status"] = "FAIL"
                    rows.append(row)
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" and row["certified"] for row in rows), report
    print("PASS: 16 full-covariance cases, including 8 fweight replication checks")


if __name__ == "__main__":
    main()
