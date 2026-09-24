#!/usr/bin/env python3
"""OLS offsets: rational coefficients and independent full sandwich covariance."""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np


def reference(u, v, y_integer, offset, robust):
    columns = [u, v, np.ones(u.size, dtype=np.int64)]
    # Bounds of this fixture keep these integer products and sums below 2^63.
    gram = [[Fraction(int(a @ b)) for b in columns] for a in columns]
    rhs = [Fraction(int(a @ y_integer), 1024) for a in columns]
    augmented = [gram[i] + [Fraction(int(i == j)) for j in range(3)] + [rhs[i]]
                 for i in range(3)]
    for k in range(3):
        pivot = augmented[k][k]
        augmented[k] = [value / pivot for value in augmented[k]]
        for i in range(3):
            if i != k:
                factor = augmented[i][k]
                augmented[i] = [a - factor * b for a, b in zip(augmented[i], augmented[k])]
    beta = [row[-1] for row in augmented]
    ld = np.longdouble
    convert = lambda x: ld(x.numerator) / ld(x.denominator)
    bread = np.array([[convert(value) for value in row[3:6]] for row in augmented])
    design = np.column_stack(columns).astype(ld)
    residual = y_integer.astype(ld) / 1024 - design @ np.array([convert(b) for b in beta])
    n = u.size
    if robust:
        scores = design * residual[:, None]
        covariance = (bread @ (scores.T @ scores) @ bread) * ld(n) / (n - 3)
    else:
        covariance = bread * (residual @ residual) / (n - 3)
    transform = np.eye(3, dtype=ld)
    transform[2, 0] = -offset
    covariance = transform @ covariance @ transform.T
    beta[2] -= offset * beta[0]
    return beta, np.asarray(covariance, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rows = []
    for n, offset in ((200000, 1000000), (1000000, 1000000), (200000, -1000000)):
        rng = np.random.default_rng(12345)
        u = rng.integers(0, 720, n)
        v = rng.integers(-50, 50, n)
        y_integer = 3 * u * 1024 + 7 * v * 1024 + rng.integers(-2**20, 2**20, n)
        y = y_integer.astype(float) / 1024
        x = np.column_stack((u + offset, v)).astype(float)
        for robust in (False, True):
            bref, vref = reference(u, v, y_integer, offset, robust)
            for mode in ("xhdfe-fast", "reghdfe-comparable"):
                model = core.HdfeRegressor(num_threads=4, tolerance_mode=mode,
                                          se_type="robust" if robust else "unadjusted")
                model.fit(y, x)
                b = np.asarray(model.coef_)
                vout = np.asarray(model.covariance_)
                berror = max(float(abs(Fraction(float(b[j])) - bref[j]) / max(1, abs(bref[j])))
                             for j in range(3))
                verror = float(np.max(np.abs(vout - vref) /
                    np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
                row = dict(n=n, offset=offset, robust=robust, mode=mode,
                           b_error=berror, V_error=verror,
                           certified=bool(model.precision_certified_),
                           status="PASS" if berror <= 1e-9 and verror <= 1e-8 else "FAIL")
                rows.append(row)
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" and row["certified"] for row in rows), report
    print("PASS: 12 offset cases with rational b and full covariance")


if __name__ == "__main__":
    main()
