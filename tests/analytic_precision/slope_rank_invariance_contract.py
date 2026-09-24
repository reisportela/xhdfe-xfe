#!/usr/bin/env python3
"""Explicit dummy/slopes OLS for units, weights and transformed intercept rank."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np


def solve_reference(y, design, w):
    sw = np.ones(y.size) if w is None else np.sqrt(w)
    a = design * sw[:, None]
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0] = 1
    coef = np.linalg.lstsq(a / scale, y * sw, rcond=None)[0] / scale
    return coef, y - design @ coef


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    assert not args.output.exists()
    os.environ["XHDFE_GPU_BACKEND"] = args.backend
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(12345)
    g = np.repeat(np.arange(200), 6).astype("int32")
    n = g.size
    h = rng.integers(0, 30, n).astype("int32")
    u = rng.normal(size=n)
    x = rng.normal(size=n) + .5 * u
    y = 1.5 * x + rng.normal(size=200)[g] + rng.normal(size=200)[g] * u
    y += rng.normal(size=30)[h] + .1 * rng.normal(size=n)
    dg, dh = np.eye(200)[g], np.eye(30)[h]
    rows = []
    for intercept in (False, True):
        for scale in (1., 1e-7, -2.**-30):
            z = u * scale
            columns = [x[:, None], dg * z[:, None], dh]
            if intercept:
                columns.append(dg)
            design = np.column_stack(columns)
            for weight_scale in (1., 2.**-23):
                w = None if weight_scale == 1 else np.full(n, weight_scale)
                bref, rref = solve_reference(y, design, w)
                for mode in ("xhdfe-fast", "reghdfe-comparable"):
                    model = core.HdfeRegressor(num_threads=2, tolerance_mode=mode,
                                              drop_singletons=False)
                    model.fit(y, x[:, None], [g, h], weights=w, slopes=[(0, z, intercept)])
                    error = float(abs(np.asarray(model.coef_)[0] - bref[0]))
                    # Comparable uses the published coefficient metric. Fast
                    # has a separate, approximate 1e-8 offline fixture check.
                    limit = (1e-9 if mode == "reghdfe-comparable" else 1e-8) * max(1, abs(bref[0]))
                    row = dict(kind="slope_units", intercept=intercept, scale=scale,
                               weight_scale=weight_scale, mode=mode, b_error=error,
                               certified=bool(model.precision_certified_),
                               gpu_used=bool(model.gpu_used_),
                               status="PASS" if error <= limit else "FAIL")
                    if args.backend == "cuda" and not row["gpu_used"]:
                        row["status"] = "FAIL"
                    rows.append(row)
    # A raw intercept is transformed by slope-only absorption. Both affine
    # slope-span duplicates and raw-constant duplicates must be omitted.
    rng = np.random.default_rng(3)
    n = 3000
    g = rng.integers(0, 60, n).astype("int32")
    z = rng.normal(size=n) + .5
    x = rng.normal(size=n)
    y = 2 + .7 * x + rng.normal(size=60)[g] * z + rng.normal(size=n)
    slope_design = np.eye(60)[g] * z[:, None]
    design = np.column_stack((x, np.ones(n), slope_design))
    bref, rref = solve_reference(y, design, None)
    for duplicate in (None, 1 + 2 * z, np.full(n, 5.)):
        xx = x[:, None] if duplicate is None else np.column_stack((x, duplicate))
        for se in ("unadjusted", "robust", "cluster"):
            model = core.HdfeRegressor(num_threads=2, drop_singletons=False, se_type=se)
            model.fit(y, xx, [g], slopes=[(0, z, False)],
                      clusters=(np.arange(n) % 19).astype("int32") if se == "cluster" else None)
            b = np.asarray(model.coef_)
            error = float(np.max(np.abs(b[[0, -1]] - bref[:2]) / np.maximum(1, np.abs(bref[:2]))))
            omitted = list(model.omitted_reason_)
            right_rank = duplicate is None or omitted[1] != 0
            row = dict(kind="transformed_intercept", se=se, duplicate=duplicate is not None,
                       b_error=error, omitted=omitted, certified=bool(model.precision_certified_),
                       gpu_used=bool(model.gpu_used_),
                       status="PASS" if error <= 1e-9 and right_rank else "FAIL")
            if args.backend == "cuda" and not row["gpu_used"]:
                row["status"] = "FAIL"
            rows.append(row)
    report = dict(module=str(args.module.resolve()), backend=args.backend,
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" and row["certified"] for row in rows), report
    print("PASS: 24 slope unit/weight cases and 9 transformed-intercept cases")


if __name__ == "__main__":
    main()
