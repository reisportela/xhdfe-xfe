#!/usr/bin/env python3
"""Identified large-mean regressor with uniform sum teams; dummy QR oracle."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(7)
    groups, individuals = 4000, 800
    membership = np.array([rng.choice(individuals, 2, replace=False) for _ in range(groups)])
    g = np.repeat(np.arange(groups), 2)
    i = membership.ravel()
    alpha = rng.normal(size=individuals)
    x = np.column_stack((1e5 + rng.normal(size=groups), rng.normal(size=groups)))
    d = np.zeros((groups, individuals))
    np.add.at(d, (g, i), 1.)
    y = .5 * (x[:, 0] - 1e5) + .25 * x[:, 1] + d @ alpha + rng.normal(size=groups)
    # Uniform teams put the constant in the incidence span. Remove its origin
    # only in this independent oracle, to keep its QR well conditioned.
    xx = x - np.array([1e5, 0.])
    design = np.column_stack((xx, d))
    scale = np.linalg.norm(design, axis=0)
    beta = np.linalg.lstsq(design / scale, y, rcond=None)[0] / scale
    rows = []
    for mode in ("xhdfe-fast", "reghdfe-comparable"):
        for aggregation in ("sum", "mean"):
            model = core.HdfeRegressor(num_threads=2, tolerance_mode=mode)
            model.fit(y[g], x[g], [i], group=g, individual=i, aggregation=aggregation)
            error = float(np.max(np.abs(np.asarray(model.coef_)[:2] - beta[:2])))
            row = dict(mode=mode, aggregation=aggregation, error=error,
                       omitted=list(model.omitted_reason_), certified=bool(model.precision_certified_),
                       constant=bool(model.model_has_constant_))
            row["status"] = "PASS" if error <= 1e-9 and not any(row["omitted"]) and row["certified"] else "FAIL"
            rows.append(row)
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" for row in rows), report
    print("PASS: sum/mean large-mean regressor, both tolerance modes")


if __name__ == "__main__":
    main()
