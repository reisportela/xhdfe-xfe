#!/usr/bin/env python3
"""Cross-front-end Gelbach parity gates.

Preserves the weighted/clustered standard-Gelbach fixture and adds a second
fixture for the opt-in absorbed-target estimand. Generated files stay in a
temporary directory under build/.

The shipped-example gate executes the standard and absorbed-target examples
in the three frontends (two designs x three frontends).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def read_matrix(path):
    return pd.read_csv(path, na_values=["."]).to_numpy(dtype=float)


def check(name, got, expected, tol=1e-11):
    got = np.asarray(got, dtype=float)
    expected = np.asarray(expected, dtype=float)
    same_shape = got.shape == expected.shape
    same_special = False
    diff = np.inf
    if same_shape:
        same_special = (
            np.array_equal(np.isnan(got), np.isnan(expected))
            and np.array_equal(np.isposinf(got), np.isposinf(expected))
            and np.array_equal(np.isneginf(got), np.isneginf(expected))
        )
        finite = np.isfinite(got) & np.isfinite(expected)
        diff = (float(np.max(np.abs(got[finite] - expected[finite])))
                if np.any(finite) else 0.0)
    ok = same_shape and same_special and diff <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
          f"shape={got.shape}, max|diff|={diff:.2e}")
    return ok


def check_text(name, got, expected):
    got = str(got).strip()
    expected = str(expected).strip()
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
          f"got={got!r}, expected={expected!r}")
    return ok


def padded_gamma(result):
    """Return Python's per-block gamma dictionary as the shared padded matrix."""
    observed = [
        np.asarray(result["gamma"][name], dtype=float)
        for name in result["names"]
        if result["group_kinds"][name] == "x2"
    ]
    if not observed:
        return np.empty((0, 0), dtype=float)
    out = np.full((max(v.size for v in observed), len(observed)), np.nan)
    for g, values in enumerate(observed):
        out[:values.size, g] = values
    return out


def gpu_contract(result, requested, require_used, prefix):
    """Check that public GPU labels cannot disguise a CPU result as CUDA."""
    ok = bool(result["gpu_requested"]) == bool(requested)
    if not requested:
        ok &= not bool(result["gpu_used"])
        ok &= result["gpu_backend"] == "cpu"
        ok &= result["gpu_status"] == "not_requested"
        ok &= int(result["gpu_status_code"]) == 0
        ok &= not bool(result["gpu_attempted"])
        ok &= not bool(result["gpu_absorption_converged"])
        ok &= int(result["gpu_absorption_iterations"]) == 0
    elif bool(result["gpu_used"]):
        ok &= result["gpu_backend"] == "cuda"
        ok &= result["gpu_status"] == "used"
        ok &= int(result["gpu_status_code"]) == 1
        ok &= bool(result["gpu_attempted"])
        ok &= bool(result["gpu_absorption_converged"])
        ok &= int(result["gpu_absorption_iterations"]) >= 0
    else:
        ok &= result["gpu_backend"] == "cpu"
        ok &= result["gpu_status"] != "used"
        ok &= int(result["gpu_status_code"]) != 1
    if require_used:
        ok &= requested and bool(result["gpu_used"])
    print(f"[{'PASS' if ok else 'FAIL'}] {prefix}:gpu_contract: "
          f"requested={result['gpu_requested']}, used={result['gpu_used']}, "
          f"backend={result['gpu_backend']!r}, status={result['gpu_status']!r}, "
          f"code={result['gpu_status_code']}")
    return ok


STATA_STANDARD_DO = r'''
clear all
set more off
adopath ++ "{stata_ado}"
import delimited using "{data}", clear asdouble
xhdfegelbach y [aweight=wgt], x1(x11 x12) ///
    x2groups("A = a1 a2 : B = b1") fes(firm) ///
    vce(cluster) cluster(cl) tol(1e-10) focal(x11) shares(movement) {gpu_option}
assert r(converged) == 1
assert "`r(estimand)'" == "coefficient_movement"
assert "`r(causal_interpretation)'" == "no"
assert "`r(absorbed_targets)'" == ""
assert "`r(absorbed_target_names)'" == ""
assert "`r(b_full_status)'" == "estimated estimated"
assert "`r(inference_status)'" == "not_applicable"
assert "`r(focal_indices)'" == "0"
assert "`r(focal_names)'" == "x11"
assert "`r(share_denominator)'" == "movement"
assert "`r(share_se_type)'" == "joint_covariance_delta_method"
assert r(gpu_requested) == {gpu_requested}
local gel_gpu_backend "`r(gpu_backend)'"
local gel_gpu_status "`r(gpu_status)'"
assert (r(gpu_used) != 1 | "`gel_gpu_backend'" == "cuda")
assert (r(gpu_used) != 1 | "`gel_gpu_status'" == "used")
assert (r(gpu_used) != 1 | r(gpu_status_code) == 1)
assert (r(gpu_used) != 0 | "`gel_gpu_backend'" == "cpu")
assert (r(gpu_used) != 0 | "`gel_gpu_status'" != "used")
assert (r(gpu_used) != 0 | r(gpu_status_code) != 1)
assert ({gpu_requested} != 0 | "`gel_gpu_status'" == "not_requested")
matrix D = r(delta)
matrix S = r(se)
matrix T = r(total)
matrix C = r(cov)
matrix TC = r(total_cov)
matrix BC = r(base_cov)
matrix CDB = r(cov_delta_bbase)
matrix CTB = r(cov_total_bbase)
matrix BB = r(b_base)
matrix BF = r(b_full)
matrix AM = r(absorbed_mask)
matrix FR = r(x1_fe_collinear_ratio)
matrix NM = r(x1_near_collinear_mask)
matrix GM = r(gamma)
matrix FT = r(fe_total)
matrix SH0 = r(share)
matrix SS0 = r(share_se)
matrix SH = SH0[1, 1..3]
matrix SS = SS0[1, 1..3]
matrix M = (r(identity_gap), r(n_obs_input), r(n_obs), r(n_obs_effective), ///
            r(n_singletons_dropped), r(df_full), r(converged), r(tol), ///
            r(fe_collinear_ss_ratio_tol), ///
            r(absorbed_target_inference_valid), r(absorbing_fe_index), ///
            r(df_base), r(n_clusters), ///
            r(near_fe_warn_upper), ///
            r(few_cluster_warning_threshold), r(threads_used), ///
            r(gpu_requested), r(gpu_used), r(gpu_status_code), ///
            r(gpu_attempted), r(gpu_absorption_converged), ///
            r(gpu_absorption_iterations))

tempname GPUF
file open `GPUF' using "{td}/stata_gpu_contract.txt", write replace text
file write `GPUF' "`gel_gpu_backend'" _n "`gel_gpu_status'" _n
file close `GPUF'

capture program drop dump_matrix
program define dump_matrix
    syntax name, using(string)
    preserve
    clear
    svmat double `namelist', names(c)
    format c* %21.17g
    export delimited using "`using'", replace datafmt
    restore
end
dump_matrix D, using("{td}/stata_delta.csv")
dump_matrix S, using("{td}/stata_se.csv")
dump_matrix T, using("{td}/stata_total.csv")
dump_matrix C, using("{td}/stata_cov.csv")
dump_matrix TC, using("{td}/stata_total_cov.csv")
dump_matrix BC, using("{td}/stata_base_cov.csv")
dump_matrix CDB, using("{td}/stata_cov_delta_bbase.csv")
dump_matrix CTB, using("{td}/stata_cov_total_bbase.csv")
dump_matrix BB, using("{td}/stata_b_base.csv")
dump_matrix BF, using("{td}/stata_b_full.csv")
dump_matrix AM, using("{td}/stata_absorbed_mask.csv")
dump_matrix FR, using("{td}/stata_x1_fe_collinear_ratio.csv")
dump_matrix NM, using("{td}/stata_x1_near_collinear_mask.csv")
dump_matrix GM, using("{td}/stata_gamma.csv")
dump_matrix FT, using("{td}/stata_fe_total.csv")
dump_matrix SH, using("{td}/stata_share_focal.csv")
dump_matrix SS, using("{td}/stata_share_se_focal.csv")
dump_matrix M, using("{td}/stata_meta.csv")
'''


R_STANDARD_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], args[2], .libPaths()))
library(xhdfe)
options(digits = 17)
d <- read.csv(args[3], check.names = FALSE)
gpu_requested <- identical(args[5], "1")
r <- xhdfe_gelbach(
  d$y, cbind(x11 = d$x11, x12 = d$x12),
  x2_groups = list(A = cbind(d$a1, d$a2), B = d$b1),
  fes = list(FIRM = d$firm),
  vce = "cluster", cluster = d$cl, weights = d$wgt, tol = 1e-10,
  focal = "x11", gpu = gpu_requested
)
stopifnot(r$converged, identical(r$estimand, "coefficient_movement"),
          identical(r$causal_interpretation, FALSE),
          identical(r$absorbed_targets, integer(0)),
          identical(r$absorbed_target_names, character(0)),
          identical(unname(r$b_full_status), c("estimated", "estimated")),
          identical(r$inference_status, "not_applicable"),
          identical(r$focal_indices, 0L),
          identical(r$focal_names, "x11"))
stopifnot(identical(isTRUE(r$gpu_requested), gpu_requested))
if (isTRUE(r$gpu_used)) {
  stopifnot(identical(r$gpu_backend, "cuda"),
            identical(r$gpu_status, "used"),
            as.integer(r$gpu_status_code) == 1L)
} else {
  stopifnot(identical(r$gpu_backend, "cpu"),
            !identical(r$gpu_status, "used"),
            as.integer(r$gpu_status_code) != 1L)
}
if (!gpu_requested) {
  stopifnot(identical(r$gpu_status, "not_requested"),
            as.integer(r$gpu_status_code) == 0L)
}
tab <- xhdfe_gelbach_tidy(r, share = "movement", include_total = FALSE,
                          include_full = FALSE)
stopifnot(all(tab$share_defined),
          identical(unique(tab$share_se_type),
                    "joint_covariance_delta_method"))
write.csv(r$delta, file.path(args[4], "r_delta.csv"), row.names = FALSE)
write.csv(r$se, file.path(args[4], "r_se.csv"), row.names = FALSE)
write.csv(cbind(r$total, r$total_se),
          file.path(args[4], "r_total.csv"), row.names = FALSE)
write.csv(r$cov, file.path(args[4], "r_cov.csv"), row.names = FALSE)
write.csv(r$total_cov, file.path(args[4], "r_total_cov.csv"), row.names = FALSE)
write.csv(r$base_cov, file.path(args[4], "r_base_cov.csv"),
          row.names = FALSE)
write.csv(r$cov_delta_bbase,
          file.path(args[4], "r_cov_delta_bbase.csv"), row.names = FALSE)
write.csv(r$cov_total_bbase,
          file.path(args[4], "r_cov_total_bbase.csv"), row.names = FALSE)
write.csv(t(r$b_base), file.path(args[4], "r_b_base.csv"), row.names = FALSE)
write.csv(t(r$b_full), file.path(args[4], "r_b_full.csv"), row.names = FALSE)
write.csv(t(as.integer(r$absorbed_mask)),
          file.path(args[4], "r_absorbed_mask.csv"), row.names = FALSE)
write.csv(t(r$x1_fe_collinear_ratio),
          file.path(args[4], "r_x1_fe_collinear_ratio.csv"),
          row.names = FALSE)
write.csv(t(as.integer(r$x1_near_collinear_mask)),
          file.path(args[4], "r_x1_near_collinear_mask.csv"),
          row.names = FALSE)
write.csv(r$gamma, file.path(args[4], "r_gamma.csv"), row.names = FALSE,
          na = "")
write.csv(cbind(r$fe_total$coef, r$fe_total$se),
          file.path(args[4], "r_fe_total.csv"), row.names = FALSE)
write.csv(matrix(tab$share, nrow = 1),
          file.path(args[4], "r_share_focal.csv"), row.names = FALSE)
write.csv(matrix(tab$share_std_error, nrow = 1),
          file.path(args[4], "r_share_se_focal.csv"), row.names = FALSE)
write.csv(matrix(c(r$identity_gap, r$n_obs_input, r$n_obs, r$n_obs_effective,
                   r$n_singletons_dropped, r$df_full,
                   as.numeric(r$converged), r$tol,
                   r$fe_collinear_ss_ratio_tol,
                   as.numeric(r$absorbed_target_inference_valid),
                   r$absorbing_fe_index, r$df_base, r$n_clusters,
                   r$near_fe_collinear_ss_ratio_warn_upper,
                   r$few_cluster_warning_threshold, r$threads_used,
                   as.numeric(r$gpu_requested), as.numeric(r$gpu_used),
                   r$gpu_status_code, as.numeric(r$gpu_attempted),
                   as.numeric(r$gpu_absorption_converged),
                   r$gpu_absorption_iterations), nrow = 1),
          file.path(args[4], "r_meta.csv"), row.names = FALSE)
writeLines(c(r$gpu_backend, r$gpu_status),
           file.path(args[4], "r_gpu_contract.txt"))
'''


STATA_ABSORBED_DO = r'''
clear all
set more off
adopath ++ "{stata_ado}"
import delimited using "{data}", clear asdouble
xhdfegelbach y, x1(focal experience) x2groups("observed = observed") ///
    fes(worker) absorbedtargets(focal) ///
    vce(cluster) cluster(worker) tol(1e-10) focal(focal) shares(base) {gpu_option}
assert r(converged) == 1
assert "`r(estimand)'" == "absorbed_target_allocation"
assert "`r(identity_status)'" == "exact_ols_constrained"
assert "`r(absorbed_targets)'" == "0"
assert "`r(absorbed_target_names)'" == "focal"
assert "`r(b_full_status)'" == "imposed_zero estimated"
assert "`r(focal_status)'" == "absorbed identified"
assert "`r(total_se_type)'" == "target_exact_base_vce_mixed_components"
assert "`r(inference_status)'" == "clustered_at_absorbing_fe"
assert r(absorbed_target_inference_valid) == 1
assert r(absorbing_fe_index) == 0
assert "`r(focal_indices)'" == "0"
assert "`r(share_se_type)'" == "joint_base_covariance_delta_method"
matrix ABS_SHARE_SE = r(share_se)
assert !missing(ABS_SHARE_SE[1, 1])
assert !missing(ABS_SHARE_SE[1, 2])
assert r(gpu_requested) == {gpu_requested}
local gel_gpu_backend "`r(gpu_backend)'"
local gel_gpu_status "`r(gpu_status)'"
assert (r(gpu_used) != 1 | "`gel_gpu_backend'" == "cuda")
assert (r(gpu_used) != 1 | "`gel_gpu_status'" == "used")
assert (r(gpu_used) != 1 | r(gpu_status_code) == 1)
assert (r(gpu_used) != 0 | "`gel_gpu_backend'" == "cpu")
assert (r(gpu_used) != 0 | "`gel_gpu_status'" != "used")
assert (r(gpu_used) != 0 | r(gpu_status_code) != 1)
assert ({gpu_requested} != 0 | "`gel_gpu_status'" == "not_requested")
matrix D = r(delta)
matrix S = r(se)
matrix T = r(total)
matrix C = r(cov)
matrix TC = r(total_cov)
matrix BC = r(base_cov)
matrix CDB = r(cov_delta_bbase)
matrix CTB = r(cov_total_bbase)
matrix BB = r(b_base)
matrix BF = r(b_full)
matrix AM = r(absorbed_mask)
matrix FR = r(x1_fe_collinear_ratio)
matrix NM = r(x1_near_collinear_mask)
matrix GM = r(gamma)
matrix FT = r(fe_total)
matrix SH0 = r(share)
matrix SS0 = r(share_se)
matrix SH = SH0[1, 1..2]
matrix SS = SS0[1, 1..2]
matrix M = (r(identity_gap), r(n_obs_input), r(n_obs), r(n_obs_effective), ///
            r(n_singletons_dropped), r(df_full), r(converged), r(tol), ///
            r(fe_collinear_ss_ratio_tol), ///
            r(absorbed_target_inference_valid), r(absorbing_fe_index), ///
            r(df_base), r(n_clusters), ///
            r(near_fe_warn_upper), ///
            r(few_cluster_warning_threshold), r(threads_used), ///
            r(gpu_requested), r(gpu_used), r(gpu_status_code), ///
            r(gpu_attempted), r(gpu_absorption_converged), ///
            r(gpu_absorption_iterations))

tempname GPUF
file open `GPUF' using "{td}/stata_gpu_contract.txt", write replace text
file write `GPUF' "`gel_gpu_backend'" _n "`gel_gpu_status'" _n
file close `GPUF'

capture program drop dump_matrix
program define dump_matrix
    syntax name, using(string)
    preserve
    clear
    svmat double `namelist', names(c)
    format c* %21.17g
    export delimited using "`using'", replace datafmt
    restore
end
dump_matrix D, using("{td}/stata_delta.csv")
dump_matrix S, using("{td}/stata_se.csv")
dump_matrix T, using("{td}/stata_total.csv")
dump_matrix C, using("{td}/stata_cov.csv")
dump_matrix TC, using("{td}/stata_total_cov.csv")
dump_matrix BC, using("{td}/stata_base_cov.csv")
dump_matrix CDB, using("{td}/stata_cov_delta_bbase.csv")
dump_matrix CTB, using("{td}/stata_cov_total_bbase.csv")
dump_matrix BB, using("{td}/stata_b_base.csv")
dump_matrix BF, using("{td}/stata_b_full.csv")
dump_matrix AM, using("{td}/stata_absorbed_mask.csv")
dump_matrix FR, using("{td}/stata_x1_fe_collinear_ratio.csv")
dump_matrix NM, using("{td}/stata_x1_near_collinear_mask.csv")
dump_matrix GM, using("{td}/stata_gamma.csv")
dump_matrix FT, using("{td}/stata_fe_total.csv")
dump_matrix SH, using("{td}/stata_share_focal.csv")
dump_matrix SS, using("{td}/stata_share_se_focal.csv")
dump_matrix M, using("{td}/stata_meta.csv")
'''


R_ABSORBED_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], args[2], .libPaths()))
library(xhdfe)
options(digits = 17)
d <- read.csv(args[3], check.names = FALSE)
gpu_requested <- identical(args[5], "1")
x1 <- cbind(focal = d$focal, experience = d$experience)
r <- xhdfe_gelbach(
  d$y, x1,
  x2_groups = list(observed = d$observed),
  fes = list(worker = d$worker),
  vce = "cluster", cluster = d$worker, tol = 1e-10,
  absorbed_targets = "focal", focal = "focal", gpu = gpu_requested
)
stopifnot(r$converged,
          identical(r$estimand, "absorbed_target_allocation"),
          identical(r$identity_status, "exact_ols_constrained"),
          identical(unname(r$b_full_status), c("imposed_zero", "estimated")),
          identical(unname(r$focal_status), c("absorbed", "identified")),
          identical(r$absorbed_targets, 0L),
          identical(r$absorbed_target_names, "focal"),
          identical(r$total_se_type,
                    "target_exact_base_vce_mixed_components"),
          identical(r$inference_status, "clustered_at_absorbing_fe"),
          isTRUE(r$absorbed_target_inference_valid),
          identical(r$absorbing_fe_index, 0L),
          identical(r$focal_indices, 0L),
          identical(r$focal_names, "focal"))
stopifnot(identical(isTRUE(r$gpu_requested), gpu_requested))
if (isTRUE(r$gpu_used)) {
  stopifnot(identical(r$gpu_backend, "cuda"),
            identical(r$gpu_status, "used"),
            as.integer(r$gpu_status_code) == 1L)
} else {
  stopifnot(identical(r$gpu_backend, "cpu"),
            !identical(r$gpu_status, "used"),
            as.integer(r$gpu_status_code) != 1L)
}
if (!gpu_requested) {
  stopifnot(identical(r$gpu_status, "not_requested"),
            as.integer(r$gpu_status_code) == 0L)
}
tab <- xhdfe_gelbach_tidy(r, share = "base", include_total = FALSE,
                          include_full = FALSE)
stopifnot(all(is.finite(tab$share_std_error)),
          identical(unique(tab$share_se_type),
                    "joint_base_covariance_delta_method"))
write.csv(r$delta, file.path(args[4], "r_delta.csv"), row.names = FALSE)
write.csv(r$se, file.path(args[4], "r_se.csv"), row.names = FALSE)
write.csv(cbind(r$total, r$total_se),
          file.path(args[4], "r_total.csv"), row.names = FALSE)
write.csv(r$cov, file.path(args[4], "r_cov.csv"), row.names = FALSE)
write.csv(r$total_cov, file.path(args[4], "r_total_cov.csv"), row.names = FALSE)
write.csv(r$base_cov, file.path(args[4], "r_base_cov.csv"),
          row.names = FALSE)
write.csv(r$cov_delta_bbase,
          file.path(args[4], "r_cov_delta_bbase.csv"), row.names = FALSE)
write.csv(r$cov_total_bbase,
          file.path(args[4], "r_cov_total_bbase.csv"), row.names = FALSE)
write.csv(t(r$b_base), file.path(args[4], "r_b_base.csv"), row.names = FALSE)
write.csv(t(r$b_full), file.path(args[4], "r_b_full.csv"), row.names = FALSE)
write.csv(t(as.integer(r$absorbed_mask)),
          file.path(args[4], "r_absorbed_mask.csv"), row.names = FALSE)
write.csv(t(r$x1_fe_collinear_ratio),
          file.path(args[4], "r_x1_fe_collinear_ratio.csv"),
          row.names = FALSE)
write.csv(t(as.integer(r$x1_near_collinear_mask)),
          file.path(args[4], "r_x1_near_collinear_mask.csv"),
          row.names = FALSE)
write.csv(r$gamma, file.path(args[4], "r_gamma.csv"), row.names = FALSE,
          na = "")
write.csv(cbind(r$fe_total$coef, r$fe_total$se),
          file.path(args[4], "r_fe_total.csv"), row.names = FALSE)
write.csv(matrix(tab$share, nrow = 1),
          file.path(args[4], "r_share_focal.csv"), row.names = FALSE)
write.csv(matrix(tab$share_std_error, nrow = 1),
          file.path(args[4], "r_share_se_focal.csv"), row.names = FALSE)
write.csv(matrix(c(r$identity_gap, r$n_obs_input, r$n_obs, r$n_obs_effective,
                   r$n_singletons_dropped, r$df_full,
                   as.numeric(r$converged), r$tol,
                   r$fe_collinear_ss_ratio_tol,
                   as.numeric(r$absorbed_target_inference_valid),
                   r$absorbing_fe_index, r$df_base, r$n_clusters,
                   r$near_fe_collinear_ss_ratio_warn_upper,
                   r$few_cluster_warning_threshold, r$threads_used,
                   as.numeric(r$gpu_requested), as.numeric(r$gpu_used),
                   r$gpu_status_code, as.numeric(r$gpu_attempted),
                   as.numeric(r$gpu_absorption_converged),
                   r$gpu_absorption_iterations), nrow = 1),
          file.path(args[4], "r_meta.csv"), row.names = FALSE)
writeLines(c(r$gpu_backend, r$gpu_status),
           file.path(args[4], "r_gpu_contract.txt"))
'''


def run_frontends(args, td, data, stata_template, r_template):
    td.mkdir()
    data_path = td / "fixture.csv"
    data.to_csv(data_path, index=False, float_format="%.17g")

    do_path = td / "frontends.do"
    do_path.write_text(
        stata_template.format(
            stata_ado=os.path.abspath(args.stata_ado),
            data=data_path,
            td=td,
            gpu_option=("gpu" if args.gpu else ""),
            gpu_requested=int(args.gpu),
        ),
        encoding="utf-8",
    )
    subprocess.run([args.stata, "-q", "-b", "do", str(do_path)],
                   cwd=td, check=True, timeout=420)

    r_path = td / "frontends.R"
    r_path.write_text(r_template, encoding="utf-8")
    subprocess.run(
        [args.rscript, str(r_path), os.path.abspath(args.r_lib),
         os.path.abspath(args.rcpp_lib), str(data_path), str(td),
         str(int(args.gpu))],
        cwd=td, check=True, timeout=420,
    )


def compare_frontends(td, expected, prefix):
    ok = True
    for frontend in ("stata", "r"):
        for key, value in expected.items():
            if key in ("gpu_backend", "gpu_status"):
                continue
            ok &= check(
                f"{prefix}:{frontend}:{key}",
                read_matrix(td / f"{frontend}_{key}.csv"),
                value,
            )
        gpu_contract_path = td / f"{frontend}_gpu_contract.txt"
        gpu_contract = gpu_contract_path.read_text(
            encoding="utf-8", errors="strict").splitlines()
        ok &= check_text(
            f"{prefix}:{frontend}:gpu_backend",
            gpu_contract[0] if gpu_contract else "",
            expected["gpu_backend"],
        )
        ok &= check_text(
            f"{prefix}:{frontend}:gpu_status",
            gpu_contract[1] if len(gpu_contract) > 1 else "",
            expected["gpu_status"],
        )
    return ok


def run_shipped_examples(args, repo, td):
    """Execute both shipped designs in three frontends (2 x 3 executions)."""
    td.mkdir()
    ok = True
    module_dir = os.path.abspath(args.module_dir or os.path.join(repo, "xhdfe"))
    py_env = os.environ.copy()
    py_env["PYTHONPATH"] = os.pathsep.join(
        [module_dir, repo, py_env.get("PYTHONPATH", "")])
    for stem in ("gelbach_example", "gelbach_absorbed_target"):
        script = os.path.join(repo, "examples", f"{stem}.py")
        code = ("import py_hdfe_v11, runpy; "
                f"runpy.run_path({script!r}, run_name='__main__')")
        try:
            subprocess.run([sys.executable, "-c", code], cwd=td,
                           env=py_env, check=True, timeout=420,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
            print(f"[PASS] examples:python:{stem}")
        except subprocess.SubprocessError as exc:
            print(f"[FAIL] examples:python:{stem}: {exc}")
            ok = False

        r_script = os.path.join(repo, "examples", f"{stem}.R")
        r_expr = (
            ".libPaths(c(" + json.dumps(os.path.abspath(args.r_lib)) + "," +
            json.dumps(os.path.abspath(args.rcpp_lib)) + ",.libPaths()));" +
            "source(" + json.dumps(r_script) + ", chdir=TRUE)"
        )
        try:
            subprocess.run([args.rscript, "-e", r_expr], cwd=td,
                           check=True, timeout=420, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
            print(f"[PASS] examples:r:{stem}")
        except subprocess.SubprocessError as exc:
            print(f"[FAIL] examples:r:{stem}: {exc}")
            ok = False

        stata_do = td / f"{stem}_example_gate.do"
        stata_do.write_text(
            "clear all\nset more off\n"
            f'adopath ++ "{os.path.abspath(args.stata_ado)}"\n'
            f'do "{os.path.join(repo, "examples", stem + ".do")}"\n'
            f'display as result "EXAMPLE_GATE_PASS_{stem}"\n',
            encoding="utf-8",
        )
        try:
            subprocess.run([args.stata, "-q", "-b", "do", str(stata_do)],
                           cwd=td, check=True, timeout=420)
            log = (td / f"{stem}_example_gate.log").read_text(
                encoding="utf-8", errors="replace")
            marker = f"EXAMPLE_GATE_PASS_{stem}"
            if marker not in log:
                raise RuntimeError("success marker missing from Stata log")
            if stem == "gelbach_absorbed_target" and "0 (imposed)" not in log:
                raise RuntimeError("imposed-zero row marker missing from Stata log")
            print(f"[PASS] examples:stata:{stem}")
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            print(f"[FAIL] examples:stata:{stem}: {exc}")
            ok = False
    return ok


def standard_fixture(gb, args, td):
    rng = np.random.default_rng(20260710)
    n = 700
    firm = rng.integers(0, 35, n)
    cl = np.arange(n) % 40
    x1 = rng.normal(size=(n, 2))
    a = np.column_stack([
        0.4 * x1[:, 0] + rng.normal(size=n),
        rng.normal(size=n),
    ])
    b = -0.25 * x1[:, 1] + rng.normal(size=n)
    psi = rng.normal(scale=0.6, size=35)
    y = (x1 @ np.array([1.1, -0.4]) + a @ np.array([0.7, 0.2]) +
         0.5 * b + psi[firm] + rng.normal(size=n))
    wgt = rng.uniform(0.25, 3.0, n)
    data = pd.DataFrame({
        "y": y, "x11": x1[:, 0], "x12": x1[:, 1],
        "a1": a[:, 0], "a2": a[:, 1], "b1": b,
        "firm": firm, "cl": cl, "wgt": wgt,
    })
    py = gb.decompose(
        y, x1, {"A": a, "B": b}, {"FIRM": firm},
        vce="cluster", cluster=cl, weights=wgt, tol=1e-10,
        x1_names=["x11", "x12"], focal="x11", gpu=args.gpu,
    )
    ok_gpu = gpu_contract(
        py, args.gpu, args.require_gpu_used, "standard:python")
    tab = gb.tidy(py, share="movement", include_total=False,
                  include_full=False)
    expected = {
        "delta": np.column_stack(
            [py["delta"][name]["coef"] for name in py["names"]]),
        "se": np.column_stack(
            [py["delta"][name]["se"] for name in py["names"]]),
        "total": np.column_stack([py["total"]["coef"], py["total"]["se"]]),
        "cov": py["cov"],
        "total_cov": py["total_cov"],
        "base_cov": py["base_cov"],
        "cov_delta_bbase": py["cov_delta_bbase"],
        "cov_total_bbase": py["cov_total_bbase"],
        "b_base": py["b_base"][None, :],
        "b_full": py["b_full"][None, :],
        "absorbed_mask": np.asarray(py["absorbed_mask"], dtype=int)[None, :],
        "x1_fe_collinear_ratio":
            np.asarray(py["x1_fe_collinear_ratio"], dtype=float)[None, :],
        "x1_near_collinear_mask":
            np.asarray(py["x1_near_collinear_mask"], dtype=int)[None, :],
        "gamma": padded_gamma(py),
        "fe_total": np.column_stack(
            [py["fe_total"]["coef"], py["fe_total"]["se"]]),
        "share_focal": np.array([[row["share"] for row in tab]]),
        "share_se_focal": np.array(
            [[row["share_std_error"] for row in tab]]),
        "meta": np.array([[
            py["identity_gap"], py["n_obs_input"], py["n_obs"],
            py["n_obs_effective"],
            py["n_singletons_dropped"], py["df_full"],
            float(py["converged"]), py["tol"],
            py["fe_collinear_ss_ratio_tol"],
            float(py["absorbed_target_inference_valid"]),
            py["absorbing_fe_index"],
            py["df_base"], py["n_clusters"],
            py["near_fe_collinear_ss_ratio_warn_upper"],
            py["few_cluster_warning_threshold"], py["threads_used"],
            float(py["gpu_requested"]), float(py["gpu_used"]),
            py["gpu_status_code"], float(py["gpu_attempted"]),
            float(py["gpu_absorption_converged"]),
            py["gpu_absorption_iterations"],
        ]]),
        "gpu_backend": py["gpu_backend"],
        "gpu_status": py["gpu_status"],
    }
    run_frontends(args, td, data, STATA_STANDARD_DO, R_STANDARD_SCRIPT)
    return ok_gpu and compare_frontends(td, expected, "standard")


def absorbed_fixture(gb, args, td):
    rng = np.random.default_rng(20260719)
    n_workers, periods = 90, 6
    worker = np.repeat(np.arange(1, n_workers + 1), periods)
    n = worker.size
    focal = rng.integers(0, 2, size=n_workers)[worker - 1].astype(float)
    experience = (np.tile(np.arange(periods), n_workers) +
                  rng.normal(0, 0.15, n))
    observed = 0.35 * focal + 0.18 * experience + rng.normal(size=n)
    worker_pay = rng.normal(size=n_workers)[worker - 1]
    y = (0.22 * focal + 0.07 * experience + 0.55 * observed + worker_pay +
         rng.normal(0, 0.45, n))
    x1 = np.column_stack([focal, experience])
    data = pd.DataFrame({
        "y": y, "focal": focal, "experience": experience,
        "observed": observed, "worker": worker,
    })
    py = gb.decompose(
        y, x1, {"observed": observed}, {"worker": worker},
        vce="cluster", cluster=worker, tol=1e-10, absorbed_targets=[0],
        x1_names=["focal", "experience"], focal="focal", gpu=args.gpu,
    )
    ok_gpu = gpu_contract(
        py, args.gpu, args.require_gpu_used, "absorbed:python")
    tab = gb.tidy(py, share="base", include_total=False, include_full=False)
    if not all(np.isfinite(row["share_std_error"]) for row in tab):
        raise AssertionError("joint base-share inference must be finite")
    if {row["share_se_type"] for row in tab} != {
            "joint_base_covariance_delta_method"}:
        raise AssertionError("base-share inference label is not the joint VCE")
    k1 = len(py["labels"])
    denom = float(py["b_base"][0])
    formula_se = []
    for g, row in enumerate(tab):
        estimate = float(row["estimate"])
        variance = (
            float(py["cov"][g * k1, g * k1]) / (denom ** 2)
            + estimate * estimate * float(py["base_cov"][0, 0])
            / (denom ** 4)
            - 2.0 * estimate
            * float(py["cov_delta_bbase"][g * k1, 0])
            / (denom ** 3)
        )
        formula_se.append(np.sqrt(max(0.0, variance)))
    ok_share_formula = check(
        "absorbed:python:share_base_joint_formula",
        [row["share_std_error"] for row in tab],
        formula_se,
    )
    expected = {
        "delta": np.column_stack(
            [py["delta"][name]["coef"] for name in py["names"]]),
        "se": np.column_stack(
            [py["delta"][name]["se"] for name in py["names"]]),
        "total": np.column_stack([py["total"]["coef"], py["total"]["se"]]),
        "cov": py["cov"],
        "total_cov": py["total_cov"],
        "base_cov": py["base_cov"],
        "cov_delta_bbase": py["cov_delta_bbase"],
        "cov_total_bbase": py["cov_total_bbase"],
        "b_base": py["b_base"][None, :],
        "b_full": py["b_full"][None, :],
        "absorbed_mask": np.asarray(py["absorbed_mask"], dtype=int)[None, :],
        "x1_fe_collinear_ratio":
            np.asarray(py["x1_fe_collinear_ratio"], dtype=float)[None, :],
        "x1_near_collinear_mask":
            np.asarray(py["x1_near_collinear_mask"], dtype=int)[None, :],
        "gamma": padded_gamma(py),
        "fe_total": np.column_stack(
            [py["fe_total"]["coef"], py["fe_total"]["se"]]),
        "share_focal": np.array([[row["share"] for row in tab]]),
        "share_se_focal": np.array(
            [[row["share_std_error"] for row in tab]]),
        "meta": np.array([[
            py["identity_gap"], py["n_obs_input"], py["n_obs"],
            py["n_obs_effective"],
            py["n_singletons_dropped"], py["df_full"],
            float(py["converged"]), py["tol"],
            py["fe_collinear_ss_ratio_tol"],
            float(py["absorbed_target_inference_valid"]),
            py["absorbing_fe_index"],
            py["df_base"], py["n_clusters"],
            py["near_fe_collinear_ss_ratio_warn_upper"],
            py["few_cluster_warning_threshold"], py["threads_used"],
            float(py["gpu_requested"]), float(py["gpu_used"]),
            py["gpu_status_code"], float(py["gpu_attempted"]),
            float(py["gpu_absorption_converged"]),
            py["gpu_absorption_iterations"],
        ]]),
        "gpu_backend": py["gpu_backend"],
        "gpu_status": py["gpu_status"],
    }
    run_frontends(args, td, data, STATA_ABSORBED_DO, R_ABSORBED_SCRIPT)
    return (ok_gpu and ok_share_formula
            and compare_frontends(td, expected, "absorbed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stata", default="stata-mp")
    ap.add_argument("--stata-ado", default="stata")
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--r-lib", required=True)
    ap.add_argument(
        "--rcpp-lib",
        default="/home/mangelo/R/x86_64-pc-linux-gnu-library/4.3",
    )
    ap.add_argument(
        "--module-dir", default=None,
        help="directory containing the py_hdfe_v11 extension to validate",
    )
    ap.add_argument(
        "--gpu", action="store_true",
        help=("request CUDA through all three public frontends; a non-used "
              "result must report a truthful CPU fallback status"),
    )
    ap.add_argument(
        "--require-gpu-used", action="store_true",
        help=("with --gpu, require affirmative CUDA use instead of accepting "
              "a truthfully labelled fallback"),
    )
    args = ap.parse_args()
    if args.require_gpu_used and not args.gpu:
        ap.error("--require-gpu-used requires --gpu")
    if not args.gpu:
        # Make the default parity gate deterministic even if the caller's
        # shell has a persistent backend override. Explicit CUDA validation
        # remains available through --gpu.
        os.environ["XHDFE_GPU_BACKEND"] = "cpu"

    repo = os.path.dirname(os.path.abspath(__file__))
    if args.module_dir:
        sys.path.insert(0, os.path.abspath(args.module_dir))
        __import__("py_hdfe_v11")
    sys.path.insert(0, repo)
    import xhdfe.gelbach as gb

    build_dir = Path(repo) / "build"
    build_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="gelbach_frontends_", dir=build_dir) as tmp:
        root = Path(tmp)
        ok_standard = standard_fixture(gb, args, root / "standard")
        ok_absorbed = absorbed_fixture(gb, args, root / "absorbed")
        ok_examples = run_shipped_examples(args, repo, root / "examples")
    if not (ok_standard and ok_absorbed and ok_examples):
        raise SystemExit(1)
    print("ALL FRONT-END PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
