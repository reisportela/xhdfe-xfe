#!/usr/bin/env python3
"""Run the public xhdfe core-23 replication benchmarks (Python front-end).

Reads registry.json, loads each dataset (.dta or .parquet), fits the spec with
xhdfe.HdfeRegressor, and appends timings + coefficients to output/python_runs.csv
and output/python_coefficients.csv.

Environment knobs:
    SPECS=credit,credit2      comma list of spec names (default: all available)
    GROUPS=sergio,pyfixest    filter by group (sergio|simulated|pyfixest)
    REPS=3                    timed repetitions per spec (default 1)
    MODE=comparable|fast      tolerance mode (default comparable, the xhdfe default)
    THREADS=8                 OpenMP threads (default: library default)
    GPU=1                     request the CUDA backend (fail-closed)
    OTHERS=1                  also time pyfixest.feols on each spec, if installed

Datasets whose file is missing are skipped with a note — run get_sergio_data.sh
/ simulate_panel.py / generate_pyfixest_data.py first.

Usage:  python3 run_python.py
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def load(path: Path):
    import pandas as pd
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_stata(path)


def main() -> None:
    import xhdfe

    registry = json.loads((HERE / "registry.json").read_text())
    want_specs = {s for s in os.environ.get("SPECS", "").split(",") if s}
    want_groups = {g for g in os.environ.get("GROUPS", "").split(",") if g}
    reps = int(os.environ.get("REPS", "1"))
    mode = os.environ.get("MODE", "comparable")
    threads = int(os.environ.get("THREADS", "0"))
    use_gpu = os.environ.get("GPU", "") not in ("", "0")
    others = os.environ.get("OTHERS", "") not in ("", "0")

    if use_gpu:
        os.environ["XHDFE_GPU_BACKEND"] = "cuda"
    tolerance_mode = "reghdfe-comparable" if mode == "comparable" else "xhdfe-fast"

    runs_path = OUT / "python_runs.csv"
    coef_path = OUT / "python_coefficients.csv"
    new_runs = not runs_path.exists()
    new_coefs = not coef_path.exists()
    runs_f = open(runs_path, "a", newline="")
    coef_f = open(coef_path, "a", newline="")
    runs_w = csv.writer(runs_f)
    coef_w = csv.writer(coef_f)
    if new_runs:
        runs_w.writerow(["spec", "engine", "mode", "gpu", "rep",
                         "elapsed_seconds", "n_obs", "iterations", "converged"])
    if new_coefs:
        coef_w.writerow(["spec", "engine", "mode", "variable", "coef", "se"])

    for spec in registry["specs"]:
        name = spec["name"]
        if want_specs and name not in want_specs:
            continue
        if want_groups and spec["group"] not in want_groups:
            continue
        path = HERE / spec["file"]
        if not path.exists():
            print(f"skip {name}: {spec['file']} not found (generate/download it first)")
            continue

        print(f"=== {name} ===", flush=True)
        df = load(path)
        y = np.ascontiguousarray(df[spec["depvar"]], dtype=np.float64)
        X = np.ascontiguousarray(df[spec["regressors"]], dtype=np.float64)
        fes = [np.asarray(df[c]) for c in spec["absorb"]]
        cl = np.asarray(df[spec["cluster"]])

        for rep in range(1, reps + 1):
            reg = xhdfe.HdfeRegressor(se_type="cluster",
                                      tolerance_mode=tolerance_mode,
                                      num_threads=threads)
            t0 = time.perf_counter()
            reg.fit(y, X, fes=fes, clusters=[cl])
            dt = time.perf_counter() - t0
            runs_w.writerow([name, "xhdfe_py", mode, int(use_gpu), rep,
                             f"{dt:.4f}", getattr(reg, "nobs_", len(y)),
                             getattr(reg, "num_iterations_", ""),
                             getattr(reg, "converged_", "")])
            print(f"  xhdfe rep{rep}: {dt:.2f}s", flush=True)
            if rep == 1:
                se = np.asarray(reg.stderr_, dtype=float)
                for v, b, s in zip(spec["regressors"], reg.coef_, se):
                    coef_w.writerow([name, "xhdfe_py", mode, v, repr(float(b)), repr(float(s))])

        if others:
            try:
                import pyfixest as pf
                fml = f"{spec['depvar']} ~ {' + '.join(spec['regressors'])} | {' + '.join(spec['absorb'])}"
                for rep in range(1, reps + 1):
                    t0 = time.perf_counter()
                    m = pf.feols(fml, data=df, vcov={"CRV1": spec["cluster"]},
                                 fixef_rm="singleton")
                    dt = time.perf_counter() - t0
                    runs_w.writerow([name, "pyfixest", mode, 0, rep,
                                     f"{dt:.4f}", int(m._N), "", ""])
                    print(f"  pyfixest rep{rep}: {dt:.2f}s", flush=True)
                    if rep == 1:
                        for v, b in m.coef().items():
                            coef_w.writerow([name, "pyfixest", mode, v, repr(float(b)), ""])
            except ImportError:
                print("  (pyfixest not installed; skipping OTHERS)")

        runs_f.flush()
        coef_f.flush()

    runs_f.close()
    coef_f.close()
    print(f"done -> {runs_path} / {coef_path}")


if __name__ == "__main__":
    main()
