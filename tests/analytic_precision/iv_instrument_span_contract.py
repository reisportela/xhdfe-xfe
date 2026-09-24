#!/usr/bin/env python3
"""Identified 2SLS is invariant to redundant instrument columns."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(111)
    n = 4096
    z1, z2, exo, error = rng.integers(-8, 9, size=(4, n)).astype(float)
    endogenous = z1 / 2 + z2 / 4 + error
    x = np.column_stack((endogenous, exo))
    y = 1.25 * endogenous + .375 * exo + 2 + .25 * error + rng.normal(size=n)
    z = np.column_stack((z1, z2))
    variants = dict(base=z, duplicate=np.column_stack((z, 2*z1, -z2)),
                    combination=np.column_stack((z, z1+z2, z1-2*z2)),
                    zero=np.column_stack((z, np.zeros(n))),
                    exogenous=np.column_stack((z, exo, np.ones(n))),
                    all=np.column_stack((z, z1+z2, exo, np.ones(n), np.zeros(n))))
    rows = []
    for weighted in (False, True):
        w = 1. + np.arange(n) % 3 if weighted else np.ones(n)
        sw = np.sqrt(w)
        actual = np.column_stack((x, np.ones(n)))
        instruments = np.column_stack((exo, np.ones(n), z))
        qz, _ = np.linalg.qr(instruments * sw[:, None], mode="reduced")
        projected = qz @ (qz.T @ (actual * sw[:, None]))
        q, r = np.linalg.qr(projected, mode="reduced")
        bref = np.linalg.solve(r, q.T @ (sw*y))
        invr = np.linalg.solve(r, np.eye(3))
        bread = invr @ invr.T
        residual = y - actual @ bref
        for robust in (False, True):
            if robust:
                scores = projected * (sw*residual)[:, None]
                vref = bread @ (scores.T @ scores) @ bread * n/(n-3)
            else:
                vref = bread * np.sum(w*residual**2)/(n-3)
            for name, zz in variants.items():
                model = core.HdfeRegressor(num_threads=2,
                    se_type="robust" if robust else "unadjusted")
                record = dict(case=name, weighted=weighted, robust=robust)
                try:
                    model.fit(y, x, instruments=zz, endogenous_idx=[0],
                              weights=w if weighted else None)
                    berror = float(np.max(np.abs(model.coef_ - bref)/np.maximum(1, np.abs(bref))))
                    verror = float(np.max(np.abs(model.covariance_ - vref)/
                                         np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
                    record.update(b_error=berror, V_error=verror,
                                  status="PASS" if berror <= 1e-9 and verror <= 1e-8 else "FAIL")
                except RuntimeError as exc:
                    record.update(status="REFUSAL", error=str(exc))
                rows.append(record)
    # Redundant instruments wholly in the exogenous span do not identify q.
    model = core.HdfeRegressor(num_threads=2)
    try:
        model.fit(y, x, instruments=np.column_stack((exo, 2*exo, np.ones(n))), endogenous_idx=[0])
    except RuntimeError:
        rows.append(dict(case="underidentified", status="PASS"))
    else:
        rows.append(dict(case="underidentified", status="FAIL"))
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" for row in rows), report
    print("PASS: 24 instrument-span cases and underidentification control")


if __name__ == "__main__":
    main()
