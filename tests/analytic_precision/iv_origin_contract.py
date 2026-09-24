#!/usr/bin/env python3
"""IV instrument/regressor translations against an independent QR sandwich."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np


def fixture(n):
    rng = np.random.default_rng(99)
    z = rng.integers(0, 2000, n).astype(float)
    v = rng.integers(-200, 200, n)
    x = np.column_stack((z // 2 + v, rng.integers(-50, 50, n))).astype(float)
    y = (3 * x[:, 0] * 1024 + 7 * x[:, 1] * 1024 + 512 * v +
         rng.integers(-2**20, 2**20, n)) / 1024
    return y, x, z


def oracle(y, x, z, weights, robust, offset):
    w = np.ones(y.size) if weights is None else weights
    sw = np.sqrt(w)
    instruments = np.column_stack((x[:, 1], np.ones(y.size), z))
    actual = np.column_stack((x, np.ones(y.size)))
    qz, _ = np.linalg.qr(instruments * sw[:, None], mode="reduced")
    fitted = qz @ (qz.T @ (actual * sw[:, None]))
    q, r = np.linalg.qr(fitted, mode="reduced")
    beta = np.linalg.solve(r, q.T @ (y * sw))
    inverse_r = np.linalg.solve(r, np.eye(3))
    bread = inverse_r @ inverse_r.T
    residual = y - actual @ beta
    if robust:
        scores = fitted * (sw * residual)[:, None]
        covariance = (bread @ (scores.T @ scores) @ bread) * y.size / (y.size - 3)
    else:
        covariance = bread * np.sum(w * residual**2) / (y.size - 3)
    transform = np.eye(3)
    transform[2, 0] = -offset
    return transform @ beta, transform @ covariance @ transform.T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rows = []
    cases = [(8192, weighted, xoff, zoff, robust) for weighted in (False, True)
             for xoff in (0., 1e6) for zoff in (0., 1e6) for robust in (False, True)]
    cases += [(1000000, False, 1e6, 1e6, robust) for robust in (False, True)]
    for n, weighted, xoff, zoff, robust in cases:
        y, x, z = fixture(n)
        w = 1. + np.arange(n) % 3 if weighted else None
        bref, vref = oracle(y, x, z, w, robust, xoff)
        xx = x + np.array([xoff, 0.])
        zz = z + zoff
        assert np.array_equal(xx[:, 0] - xoff, x[:, 0]) and np.array_equal(zz - zoff, z)
        model = core.HdfeRegressor(num_threads=4, se_type="robust" if robust else "unadjusted")
        model.fit(y, xx, weights=w, instruments=zz[:, None], endogenous_idx=[0])
        berror = float(np.max(np.abs(np.asarray(model.coef_) - bref) / np.maximum(1, np.abs(bref))))
        verror = float(np.max(np.abs(np.asarray(model.covariance_) - vref) /
            np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
        rows.append(dict(n=n, weighted=weighted, x_offset=xoff, z_offset=zoff, robust=robust,
                         b_error=berror, V_error=verror, certified=bool(model.precision_certified_),
                         status="PASS" if berror <= 1e-9 and verror <= 1e-8 else "FAIL"))
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" and row["certified"] for row in rows), report
    print("PASS: 18 IV translation cases with independent b and full covariance")


if __name__ == "__main__":
    main()
