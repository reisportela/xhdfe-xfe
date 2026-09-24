#!/usr/bin/env python3
"""Zero aweights/pweights leave the sample before FE, cluster and singleton counts."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np


def reference(y, x, fe, w, cluster, vce):
    _, code = np.unique(fe, return_inverse=True)
    d = np.eye(int(code.max()) + 1)[code]
    design = np.column_stack((x, d))
    sw = np.sqrt(w)
    q, r = np.linalg.qr(design * sw[:, None], mode="reduced")
    beta = np.linalg.solve(r, q.T @ (sw * y))
    residual = y - design @ beta
    transform = np.zeros((x.shape[1] + 1, design.shape[1]))
    transform[:-1, :x.shape[1]] = np.eye(x.shape[1])
    transform[-1, x.shape[1]:] = (w @ d) / w.sum()
    influence = np.linalg.solve(r.T, transform.T).T @ q.T * sw[None, :]
    df = y.size - design.shape[1]
    if vce == "unadjusted":
        # The homoskedastic WLS covariance has one, not two, factors of W.
        covariance = (influence / sw[None, :]) @ (influence / sw[None, :]).T
        covariance *= np.sum(w * residual**2) / df
    elif vce == "robust":
        scores = influence * residual[None, :]
        covariance = scores @ scores.T * y.size / df
    else:
        _, codes = np.unique(cluster, return_inverse=True)
        g = int(codes.max()) + 1
        scores = np.zeros((g, x.shape[1] + 1))
        np.add.at(scores, codes, (influence * residual[None, :]).T)
        covariance = scores.T @ scores * g / (g - 1) * (y.size - 1) / df
    return transform @ beta, covariance


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
    rng = np.random.default_rng(120923)
    n = 1200
    fe = rng.integers(0, 38, n).astype("int32")
    fe[-80:] = 39
    fe[:2] = 38
    cluster = rng.integers(0, 9, n).astype("int32")
    cluster[-80:] = 9 + np.arange(80) % 3
    x = rng.normal(size=(n, 2))
    y = x @ np.array([1.5, -.7]) + rng.normal(size=40)[fe] + rng.normal(size=12)[cluster] + rng.normal(size=n)
    w = .5 + rng.random(n)
    w[-80:] = 0
    w[1] = -0.
    positive = w > 0
    rows = []
    for drop in (False, True):
        selected = positive & (fe != 38) if drop else positive
        index = np.flatnonzero(selected)
        for probability in (False, True):
            for vce in ("unadjusted", "robust", "cluster"):
                if probability and vce == "unadjusted":
                    continue
                kw = dict(weights=w, pweights=probability)
                if vce == "cluster":
                    kw["clusters"] = cluster
                model = core.HdfeRegressor(num_threads=2, drop_singletons=drop, se_type=vce)
                model.fit(y, x, [fe], **kw)
                bref, vref = reference(y[selected], x[selected], fe[selected], w[selected], cluster[selected], vce)
                berror = float(np.max(np.abs(np.asarray(model.coef_) - bref) / np.maximum(1, np.abs(bref))))
                verror = float(np.max(np.abs(np.asarray(model.covariance_) - vref) /
                    np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
                sample_ok = np.array_equal(np.asarray(model.sample_index_), index)
                row = dict(kind="WLS_oracle", drop_singletons=drop, pweights=probability, vce=vce,
                           b_error=berror, V_error=verror, sample_ok=sample_ok,
                           nobs=int(model.nobs_), nobs_full=int(model.nobs_full_),
                           clusters=int(model.num_clusters_), gpu_used=bool(model.gpu_used_))
                row["status"] = "PASS" if (berror <= 1e-9 and verror <= 1e-8 and sample_ok and
                    row["nobs"] == index.size and row["nobs_full"] == int(positive.sum()) and
                    (vce != "cluster" or row["clusters"] == 9)) else "FAIL"
                rows.append(row)
    # Other affected inputs must be filtered in the same row order as y/X.
    z = x[:, 1] + .3 * rng.normal(size=n)
    slope = rng.normal(size=n)
    pos = np.flatnonzero(positive)
    for kind in ("IV", "slopes", "savefe"):
        opts = dict(num_threads=2, drop_singletons=False, se_type="robust", retain_fes=kind == "savefe")
        full_kw, kept_kw = dict(weights=w), dict(weights=w[pos])
        if kind == "IV":
            full_kw.update(instruments=z[:, None], endogenous_idx=[1])
            kept_kw.update(instruments=z[pos, None], endogenous_idx=[1])
        if kind == "slopes":
            full_kw["slopes"] = [(0, slope, True)]
            kept_kw["slopes"] = [(0, slope[pos], True)]
        full, kept = core.HdfeRegressor(**opts), core.HdfeRegressor(**opts)
        full.fit(y, x, [fe], **full_kw)
        kept.fit(y[pos], x[pos], [fe[pos]], **kept_kw)
        error = float(np.max(np.abs(np.asarray(full.coef_) - np.asarray(kept.coef_))))
        vref = np.asarray(kept.covariance_)
        verror = float(np.max(np.abs(np.asarray(full.covariance_) - vref) /
            np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
        rows.append(dict(kind=kind, b_error=error, V_error=verror,
            gpu_used=bool(full.gpu_used_), status="PASS" if error <= 1e-9 and verror <= 1e-8 and
            np.array_equal(np.asarray(full.sample_index_), pos) else "FAIL"))
    # Whole zero-weight groups are removed after within-group consistency checks.
    groups, individuals = 240, 40
    members = np.array([rng.choice(individuals, 3, replace=False) for _ in range(groups)])
    members[-12:, 0] = 40
    group = np.repeat(np.arange(groups), 3)
    ind = members.ravel()
    xg = rng.normal(size=(groups, 1))
    yg = .8 * xg[:, 0] + rng.normal(size=groups)
    wg = np.ones(groups)
    wg[-12:] = 0
    keep = wg[group] > 0
    for aggregation in ("mean", "sum"):
        opts = dict(num_threads=2, drop_singletons=False, dofadjustments="exact")
        full, kept = core.HdfeRegressor(**opts), core.HdfeRegressor(**opts)
        full.fit(yg[group], xg[group], [ind], weights=wg[group], group=group, individual=ind, aggregation=aggregation)
        kept.fit(yg[group][keep], xg[group][keep], [ind[keep]], weights=wg[group][keep],
                 group=group[keep], individual=ind[keep], aggregation=aggregation)
        error = float(np.max(np.abs(np.asarray(full.coef_) - np.asarray(kept.coef_))))
        vref = np.asarray(kept.covariance_)
        verror = float(np.max(np.abs(np.asarray(full.covariance_) - vref) /
            np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
        rows.append(dict(kind="group_" + aggregation, b_error=error, V_error=verror,
            nobs=int(full.nobs_), nobs_full=int(full.nobs_full_), gpu_used=bool(full.gpu_used_),
            status="PASS" if error <= 1e-9 and verror <= 1e-8 and full.nobs_ == groups-12 and
            full.nobs_full_ == groups-12 and np.all(wg[group[np.asarray(full.sample_index_)]] > 0) else "FAIL"))
    if args.backend == "cuda":
        for row in rows:
            if not row["gpu_used"]:
                row["status"] = "FAIL"
    report = dict(module=str(args.module.resolve()), backend=args.backend,
        sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" for row in rows), report
    print("PASS: zero-weight sample, WLS inference, IV, slopes, savefe and grouped inputs")


if __name__ == "__main__":
    main()
