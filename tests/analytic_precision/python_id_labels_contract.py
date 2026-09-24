#!/usr/bin/env python3
"""Numeric ID equality classes, with explicit dummy-OLS and grouping controls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np


def labels(code):
    return code // 2 + np.where(code % 2 == 0, .25, .75)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(12345)
    n = 2000
    g = rng.integers(0, 40, n)
    t = rng.integers(0, 10, n)
    alpha, gamma = rng.normal(0, 3, 40), rng.normal(size=10)
    x = .8 * alpha[g] + rng.normal(size=n)
    y = 2 * x + alpha[g] + gamma[t] + rng.normal(size=n)
    design = np.column_stack((x, np.eye(40)[g], np.eye(10)[t, 1:]))
    bref = np.linalg.lstsq(design, y, rcond=None)[0][0]
    rows = []
    def fit(fes, clusters=None):
        model = core.HdfeRegressor(num_threads=2, se_type="cluster" if clusters is not None else "unadjusted")
        model.fit(y, x[:, None], fes, clusters=clusters)
        return model
    representations = dict(int64=g, float_integer=g.astype(float),
        float32=labels(g).astype("float32"), float64=labels(g), float_list=labels(g).tolist(),
        uint64=np.uint64(2**63 + 9) + g.astype("uint64"),
        integer_object=g.astype(object), integer_strings=g.astype(str).astype(object))
    for name, values in representations.items():
        model = fit([values, t])
        error = float(abs(model.coef_[0] - bref))
        rows.append(dict(case=name, b_error=error, levels=list(model.fe_num_levels_),
                         status="PASS" if error <= 1e-9 and list(model.fe_num_levels_) == [40, 10] else "FAIL"))
    # The covariance control is invariant to recoding cluster labels. FE/b have
    # already been independently checked above; this isolates parser equality.
    for matrix in (False, True):
        codes = np.column_stack((g, t)) if matrix else g
        recoded = labels(codes)
        control = fit([g, t], clusters=codes)
        model = fit([g, t], clusters=np.asfortranarray(recoded) if matrix else recoded)
        v = np.asarray(control.covariance_)
        error = float(np.max(np.abs(np.asarray(model.covariance_) - v) /
                            np.maximum(1e-30, np.sqrt(np.outer(np.diag(v), np.diag(v))))))
        rows.append(dict(case="cluster_matrix" if matrix else "cluster_vector", V_error=error,
                         status="PASS" if error <= 1e-8 else "FAIL"))
    # Distinct group/individual labels must survive the two independent parsers.
    group = np.repeat(np.arange(120), 2)
    individual = np.column_stack((np.arange(120) % 12, (np.arange(120) + 1) % 12)).ravel()
    xx = rng.normal(size=120)
    yy = 2 * xx + rng.normal(size=120)
    control = core.HdfeRegressor(num_threads=2)
    control.fit(yy[group], xx[group, None], [individual], group=group,
                individual=individual, aggregation="mean")
    model = core.HdfeRegressor(num_threads=2)
    model.fit(yy[group], xx[group, None], [labels(individual)], group=labels(group),
              individual=labels(individual), aggregation="mean")
    error = float(np.max(np.abs(np.asarray(model.coef_) - np.asarray(control.coef_))))
    rows.append(dict(case="group_individual", b_error=error, status="PASS" if error <= 1e-9 else "FAIL"))
    rejections = []
    for value in (np.nan, np.inf, -.5):
        invalid = labels(g)
        invalid[0] = value
        try:
            fit([invalid, t])
        except RuntimeError as error:
            rejections.append(dict(value=str(value), status="EXPECTED_INPUT_ERROR", message=str(error)))
        else:
            rejections.append(dict(value=str(value), status="FAIL"))
    report = dict(module=str(args.module.resolve()),
        sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows, input_errors=rejections)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" for row in rows), report
    assert all(row["status"] == "EXPECTED_INPUT_ERROR" for row in rejections), report
    print("PASS: 11 label cases; 3 expected invalid-input errors")


if __name__ == "__main__":
    main()
