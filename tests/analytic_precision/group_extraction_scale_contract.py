#!/usr/bin/env python3
"""Group FE extraction: explicit dummy projection, scale and zero-weight rows."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import warnings

sys.dont_write_bytecode = True
import numpy as np


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
    rng = np.random.default_rng(3)
    groups, individuals = 600, 150
    rg, ri = [], []
    for g in range(groups):
        for i in rng.choice(individuals, size=int(rng.integers(1, 4)), replace=False):
            rg.append(g)
            ri.append(i)
    rg, ri = np.array(rg), np.array(ri)
    incidence = np.zeros((groups, individuals))
    incidence[rg, ri] = 1.
    incidence /= incidence.sum(axis=1, keepdims=True)
    fg = rng.integers(0, 15, size=groups)
    x = rng.normal(size=groups)
    base = .7*x + incidence @ rng.normal(size=individuals) + rng.normal(size=15)[fg] + .3*rng.normal(size=groups)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for scale in (1e-6, 1e-3, 1., 1e3):
            for extra_fe in (False, True):
                for zero_weight in (False, True):
                    y = scale * base
                    wg = np.ones(groups)
                    if zero_weight:
                        wg[:20] = 0.
                    weights = wg[rg] if zero_weight else None
                    fes = [fg[rg], ri] if extra_fe else [ri]
                    model = core.HdfeRegressor(num_threads=2)
                    model.fit(y[rg], x[rg, None], fes, weights=weights,
                              group=rg, individual=ri, aggregation="mean")
                    retained = rg[np.asarray(model.sample_index_)]
                    assert np.all(wg[retained] > 0)
                    fe = incidence if not extra_fe else np.column_stack((incidence, np.eye(15)[fg]))
                    design = np.column_stack((x[retained], fe[retained]))
                    beta = np.linalg.lstsq(design, y[retained], rcond=None)[0][0]
                    assert abs(model.coef_[0] - beta) <= 1e-9 * max(1., abs(beta))
                    target = fe[retained] @ np.linalg.lstsq(
                        fe[retained], y[retained] - model.coef_[0]*x[retained], rcond=None)[0]
                    out = model.extract_group_individual_fes(
                        y[rg], x[rg, None], fes, rg, ri, weights, "mean")
                    alpha = np.zeros(individuals)
                    alpha[np.asarray(out["individual_ids"])]=out["individual_effects"]
                    component = incidence[retained] @ alpha
                    if extra_fe:
                        gamma = np.zeros(15)
                        gamma[np.asarray(out["fe_level_ids"][0])] = out["fe_level_effects"][0]
                        component += gamma[fg[retained]]
                    else:
                        component += model.coef_[-1]
                    mse = float(np.mean((component - target)**2))
                    rms = float(np.linalg.norm(target)/math.sqrt(retained.size))
                    exponent = min(0, math.frexp(rms)[1]-1) if rms > 0 else 0
                    # Existing extraction tol_main=1e-9, in normalized coordinates.
                    # The explicit QR oracle has its own floating-point allowance.
                    limit = 1e-9 * math.ldexp(1., 2*exponent)
                    allowance = (128*np.finfo(float).eps*max(np.max(np.abs(target)), scale))**2
                    assert mse <= limit + allowance, (scale, extra_fe, zero_weight, mse, limit)
                    rows.append(dict(scale=scale, extra_fe=extra_fe, zero_weight=zero_weight,
                                     mse=mse, limit=limit, iterations=int(out["iterations"]),
                                     gpu_used=bool(model.gpu_used_), status="PASS"))
                    assert bool(model.gpu_used_) == (args.backend == "cuda")
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("PASS: 16 scale/FE/zero-weight extraction cases with explicit projection")


if __name__ == "__main__":
    main()
