#!/usr/bin/env python3
"""Singleton fixed point and diagnostic-only groupvar against explicit QR."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.linalg import qr


def oracle(y, x, fes, weights):
    d = np.column_stack([np.eye(len(np.unique(f)))[np.unique(f, return_inverse=True)[1]]
                         for f in fes])
    root = np.sqrt(weights)
    q, r, _ = qr(root[:, None] * d, mode="economic", pivoting=True)
    rank = int(np.sum(abs(np.diag(r)) > 1e-10))
    q = q[:, :rank]
    xt = root[:, None] * x
    yt = root * y
    xt -= q @ (q.T @ xt)
    yt -= q @ (q.T @ yt)
    b = np.linalg.lstsq(xt, yt, rcond=None)[0]
    u = yt - xt @ b
    v = (u @ u) / (weights.sum() - rank - x.shape[1]) * np.linalg.inv(xt.T @ xt)
    return b, v, rank


def peel(fes, weights):
    keep = np.ones(len(weights), dtype=bool)
    while True:
        drop = np.zeros(len(weights), dtype=bool)
        for f in fes:
            counts = np.bincount(f[keep], weights=weights[keep], minlength=int(f.max()) + 1)
            drop |= keep & (counts[f] == 1)
        if not drop.any():
            return keep
        keep &= ~drop


def first_pair_components(fes):
    left, right = (np.unique(f, return_inverse=True)[1] for f in fes[:2])
    offset = int(left.max()) + 1
    parent = list(range(offset + int(right.max()) + 1))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(left, right):
        parent[root(int(a))] = root(offset + int(b))
    return np.asarray([root(int(a)) for a in left])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(7)
    rows = []
    for length in (49, 150, 300):
        w = np.repeat(np.arange(60), 5).tolist()
        f = rng.integers(0, 8, 300).tolist()
        previous = 0
        for k in range(length):
            w.extend((60+k, 60+k))
            f.extend((previous, 8+k))
            previous = 8+k
        fes = [np.asarray(w, dtype=np.int32), np.asarray(f, dtype=np.int32)]
        x = rng.normal(size=(len(w), 2))
        y = x @ np.array([1., -.5]) + rng.normal(size=len(w))
        for frequency in (False, True):
            weights = np.ones(len(w))
            keep = peel(fes, weights)
            b, v, rank = oracle(y[keep], x[keep], [f[keep] for f in fes], weights[keep])
            for threads in (1, 2):
                model = core.HdfeRegressor(num_threads=threads)
                model.fit(y, x, fes, **({"weights": weights, "fweights": True} if frequency else {}))
                assert np.array_equal(model.sample_index_, np.flatnonzero(keep))
                assert model.nobs_ == 300 and model.num_singletons_ == 2*length
                np.testing.assert_allclose(model.coef_[:2], b, atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(model.covariance_[:2, :2], v, atol=1e-12, rtol=1e-8)
                rows.append(dict(case="singleton", length=length, frequency=frequency,
                                 threads=threads, n=model.nobs_, rank=rank, status="PASS"))
        weights[-1] = 2
        keep = peel(fes, weights)
        model = core.HdfeRegressor(num_threads=1)
        model.fit(y, x, fes, weights=weights, fweights=True)
        assert np.array_equal(model.sample_index_, np.flatnonzero(keep)) and keep.all()
        rows.append(dict(case="weighted_tip", length=length, status="PASS"))

    w = np.repeat(np.arange(80), 4).astype(np.int32)
    f = np.tile([0, 1, 0, 1], 80).astype(np.int32) + 2*(w >= 40)
    region = (f % 2).astype(np.int32)
    x = rng.normal(size=(len(w), 1))
    y = .5*x[:, 0] + rng.normal(size=80)[w] + rng.normal(size=4)[f] + rng.normal(size=len(w))
    for fes in ([region, w, f], [w, f, region], [f, region, w]):
        b, v, rank = oracle(y, x, fes, np.ones(len(w)))
        models = []
        for groupvar in (False, True):
            model = core.HdfeRegressor(num_threads=1, groupvar=groupvar)
            model.fit(y, x, fes)
            assert model.df_a_ == rank == 82
            np.testing.assert_allclose(model.coef_[:1], b, atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(model.covariance_[:1, :1], v, atol=1e-12, rtol=1e-8)
            if groupvar:
                assert len(model.groupvar_) == len(y)
                pairs = set(zip(first_pair_components(fes), np.asarray(model.groupvar_)))
                assert len(pairs) == len({a for a, _ in pairs}) == len({b for _, b in pairs})
            models.append(model)
        np.testing.assert_array_equal(models[0].coef_, models[1].coef_)
        np.testing.assert_array_equal(models[0].covariance_, models[1].covariance_)
        rows.append(dict(case="groupvar", rank=rank, status="PASS"))
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print("PASS: 18 sample/DoF cases, independent fixed point and QR")


if __name__ == "__main__":
    main()
