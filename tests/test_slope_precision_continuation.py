#!/usr/bin/env python3
"""Focused Option-A negative/success gates for ordinary heterogeneous slopes."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

sys.path.extend((
    "/home/mangelo/.local/lib/python3.12/site-packages",
    "/home/mangelo/miniconda3/lib/python3.12/site-packages",
))

import numpy as np
import pandas as pd


def factorize(series: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(series, sort=True)
    return codes.astype(np.int32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()

    os.environ.update({
        "XHDFE_GPU_BACKEND": "cpu",
        "XHDFE_ABSORPTION_CACHE_MODE": "off",
        "XHDFE_FE_STRUCTURE_MODE": "off",
        "XHDFE_MOBILITY_MODE": "off",
        "OMP_NUM_THREADS": "12",
        "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    sys.path.insert(0, str(args.module_dir.resolve()))
    mod = importlib.import_module("py_hdfe_v11")

    x_names = [
        "exp_akm", "exp_akm_sq_100", "exp_akm_cu_1000",
        "exp_akm_qt_10000",
    ]
    frame = pd.read_parquet(args.dataset, columns=[
        "ln_wgain_hour_wz", *x_names, "idtrab", "ano", "NPC_FIC",
        "firm_seniority_obs", "firm_seniority_spline10",
    ])
    y = frame["ln_wgain_hour_wz"].to_numpy(np.float64, copy=False)
    X = np.asfortranarray(frame[x_names].to_numpy(np.float64, copy=False))
    idtrab = factorize(frame["idtrab"])
    ano = factorize(frame["ano"])
    npc = factorize(frame["NPC_FIC"])
    fes = [idtrab, ano, npc, npc.copy()]
    clusters = np.column_stack((idtrab, npc))
    slopes = [
        (2, frame["firm_seniority_obs"].to_numpy(np.float64, copy=False), True),
        (3, frame["firm_seniority_spline10"].to_numpy(np.float64, copy=False), False),
    ]

    def fit(convergence: str):
        reg = mod.HdfeRegressor(
            se_type="cluster", tol=1e-8, max_iter=100_000,
            convergence=convergence, fit_intercept=True, num_threads=12,
            drop_singletons=True, retain_fes=False,
            absorption_method="auto", tolerance_mode="reghdfe-comparable",
        )
        error = ""
        try:
            reg.fit(y, X, fes=fes, clusters=clusters, slopes=slopes)
        except RuntimeError as exc:
            # Since 2.27.0 an absorption that converges by the solver's rule
            # but fails the precision certificate returns no estimates
            # (fail-closed); 2.26.2 returned it with precision_certified=0.
            error = str(exc)
            if "No estimates returned" not in error:
                raise
        return reg, error

    with np.testing.suppress_warnings() as warning_state:
        warning_state.filter(RuntimeWarning)
        explicit, explicit_error = fit("reghdfe")
    # Explicit-criterion gate: the reghdfe stopping rule leaves the slope
    # block inaccurate at the public tolerance. 2.26.2 returned that result
    # uncertified; 2.27.0 first refused it, and since 22sep2026 it continues
    # the same rule at tighter internal tolerances once and returns the first
    # certified stage (refusing only if no stage certifies). An uncertified
    # result must never be returned.
    assert not explicit_error, explicit_error
    assert explicit.converged_ and explicit.precision_certified_
    assert explicit.slope_accuracy_retry_stages_ > 0
    assert explicit.slope_accuracy_retry_iterations_ > 0
    assert explicit.slope_internal_tolerance_ < 1e-8
    assert explicit.slope_block_residual_rel_ <= 8e-9

    repaired, repaired_error = fit("auto")
    assert not repaired_error, repaired_error
    assert repaired.converged_ and repaired.precision_certified_
    assert repaired.slope_accuracy_retry_stages_ > 0
    assert repaired.slope_accuracy_retry_iterations_ > 0
    assert repaired.slope_internal_tolerance_ < 1e-8
    assert repaired.slope_block_residual_rel_ <= 1e-9
    assert repaired.absorption_method_used == explicit.absorption_method_used
    # Both certified: the two routes agree within the comparable contract.
    assert np.max(np.abs(np.asarray(repaired.coef_) - np.asarray(explicit.coef_))
                  / np.maximum(1.0, np.abs(np.asarray(repaired.coef_)))) <= 1e-8
    # Structural pins of the 57,466-worker subsample the gate was designed on;
    # on any other sample (e.g. the full AKM data) they are reported, not asserted.
    pinned = list(repaired.cluster_counts_) == [57466, 52732]
    if pinned:
        assert list(repaired.fe_num_levels_) == [57466, 36, 105464, 52732]
        assert list(repaired.fe_redundant_) == [57466, 1, 55813, 46450]
        assert list(repaired.fe_num_coefs_) == [0, 35, 49651, 6282]
    else:
        print(f"SLOPE_OPTION_A_NOTE structural pins skipped: cluster_counts={list(repaired.cluster_counts_)} "
              f"fe_num_levels={list(repaired.fe_num_levels_)}")

    negative = explicit
    print(
        "SLOPE_OPTION_A_PASS "
        f"explicit_iterations={negative.num_iterations_} "
        f"explicit_stages={negative.slope_accuracy_retry_stages_} "
        f"explicit_block_rel={negative.slope_block_residual_rel_:.17g} "
        f"repaired_iterations={repaired.num_iterations_} "
        f"retry_stages={repaired.slope_accuracy_retry_stages_} "
        f"repaired_block_rel={repaired.slope_block_residual_rel_:.17g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
