#!/usr/bin/env python3
"""Cross-front-end Gelbach parity gates.

Validates weighted/clustered standard Gelbach, the opt-in absorbed-target
estimand, selectable added-FE connectivity, and the conditional common-FE
estimand. Generated files stay in a temporary directory under build/.

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
import warnings

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


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
local gel_fe_split_status "`r(fe_split_status)'"
local gel_connectivity_pair_status "`r(connectivity_pair_status)'"
local gel_connected_mode "`r(connected_mode)'"
local gel_mobility_scope "`r(mobility_component_scope)'"
local gel_share_status_all "`r(share_interval_status)'"
local gel_share_status : word 1 of `gel_share_status_all'
local gel_fe_variance_status "`r(fe_variance_status)'"
local gel_share_se_type "`r(share_se_type)'"
local gel_total_se_type "`r(total_se_type)'"
assert (r(gpu_used) != 1 | "`gel_gpu_backend'" == "cuda")
assert (r(gpu_used) != 1 | "`gel_gpu_status'" == "used")
assert (r(gpu_used) != 1 | r(gpu_status_code) == 1)
assert (r(gpu_used) != 0 | "`gel_gpu_backend'" == "cpu")
assert (r(gpu_used) != 0 | "`gel_gpu_status'" != "used")
assert (r(gpu_used) != 0 | r(gpu_status_code) != 1)
assert ({gpu_requested} != 0 | "`gel_gpu_status'" == "not_requested")
assert r(n_mobility_components) == 0
assert r(fe_split_identified) == 0
assert r(connectivity_fe1_index) == -1
assert r(connectivity_fe2_index) == -1
assert r(connectivity_pair_explicit) == 0
assert "`gel_fe_split_status'" == "single_fe_dimension"
assert "`gel_connectivity_pair_status'" == "not_applicable"
assert "`gel_connected_mode'" == "diagnose"
assert "`gel_mobility_scope'" == "not_applicable"
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
matrix B2 = r(beta2)
matrix B2C = r(beta2_cov)
matrix AL = r(auxiliary_loadings)
matrix ALD = r(auxiliary_loading_diagnostics)
matrix ALZ = r(auxiliary_loading_max_abs_z)
matrix ALP = r(auxiliary_loading_pvalue)
matrix ALE = r(auxiliary_loading_test_evaluated)
matrix B2W = r(beta2_wald)
matrix GN = r(contribution_gradient_norm)
matrix RIV = r(regular_inference_valid)
matrix RIS = r(regular_inference_status_code)
matrix RM = (r(regular_inference_all_valid), r(regularity_test_alpha))
assert "`r(regular_inference_status)'" != ""
matrix FT = r(fe_total)
matrix SH0 = r(share)
matrix SS0 = r(share_se)
matrix SHT0 = r(share_denominator_t)
matrix SHV0 = r(share_interval_status_code)
matrix SH = SH0[1, 1..3]
matrix SS = SS0[1, 1..3]
matrix SHT = SHT0[1, 1]
matrix SHV = SHV0[1, 1]
matrix GATE = (r(share_t_min), r(fe_variance_ratio_min))
matrix M = (r(identity_gap), r(n_obs_input), r(n_obs), r(n_obs_effective), ///
            r(n_singletons_dropped), r(df_full), r(converged), r(tol), ///
            r(fe_collinear_ss_ratio_tol), ///
            r(absorbed_target_inference_valid), r(absorbing_fe_index), ///
            r(df_base), r(n_clusters), ///
            r(near_fe_warn_upper), ///
            r(few_cluster_warning_threshold), r(threads_used), ///
            r(gpu_requested), r(gpu_used), r(gpu_status_code), ///
            r(gpu_attempted), r(gpu_absorption_converged), ///
            r(gpu_absorption_iterations), ///
            r(n_mobility_components), ///
            r(largest_mobility_component_n_obs), ///
            r(largest_mobility_component_share), ///
            r(largest_mobility_weight_share), ///
            r(fe_split_identified), ///
            r(connectivity_fe1_index), ///
            r(connectivity_fe2_index), ///
            r(connectivity_pair_explicit))

tempname GPUF
file open `GPUF' using "{td}/stata_gpu_contract.txt", write replace text
file write `GPUF' "`gel_gpu_backend'" _n "`gel_gpu_status'" _n ///
    "`gel_fe_split_status'" _n "`gel_connectivity_pair_status'" _n ///
    "`gel_connected_mode'" _n "`gel_mobility_scope'" _n ///
    "`gel_share_status'" _n "`gel_fe_variance_status'" _n ///
    "`gel_share_se_type'" _n "`gel_total_se_type'" _n
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
dump_matrix B2, using("{td}/stata_beta2.csv")
dump_matrix B2C, using("{td}/stata_beta2_cov.csv")
dump_matrix AL, using("{td}/stata_auxiliary_loadings.csv")
dump_matrix ALD, using("{td}/stata_auxiliary_loading_diagnostics.csv")
dump_matrix ALZ, using("{td}/stata_auxiliary_loading_max_abs_z.csv")
dump_matrix ALP, using("{td}/stata_auxiliary_loading_pvalue.csv")
dump_matrix ALE, using("{td}/stata_auxiliary_loading_test_evaluated.csv")
dump_matrix B2W, using("{td}/stata_beta2_wald.csv")
dump_matrix GN, using("{td}/stata_contribution_gradient_norm.csv")
dump_matrix RIV, using("{td}/stata_regular_inference_valid.csv")
dump_matrix RIS, using("{td}/stata_regular_inference_status_code.csv")
dump_matrix RM, using("{td}/stata_regular_meta.csv")
dump_matrix FT, using("{td}/stata_fe_total.csv")
dump_matrix SH, using("{td}/stata_share_focal.csv")
dump_matrix SS, using("{td}/stata_share_se_focal.csv")
dump_matrix SHT, using("{td}/stata_share_t_focal.csv")
dump_matrix SHV, using("{td}/stata_share_status_code_focal.csv")
dump_matrix GATE, using("{td}/stata_gate_meta.csv")
dump_matrix M, using("{td}/stata_meta.csv")
'''


R_STANDARD_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], args[2], .libPaths()))
library(xhdfe)
assign(".xhdfe_cpp_gelbach",
       get(".xhdfe_cpp_gelbach", envir = asNamespace("xhdfe")),
       envir = .GlobalEnv)
assign(".akm_id_codes",
       get(".akm_id_codes", envir = asNamespace("xhdfe")),
       envir = .GlobalEnv)
source(file.path(args[6], "r/xhdfe/R/gelbach.R"), local = .GlobalEnv)
source(file.path(args[6], "r/xhdfe/R/gelbach_features.R"),
       local = .GlobalEnv)
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
stopifnot(r$n_mobility_components == 0L,
          !isTRUE(r$fe_split_identified),
          identical(r$connectivity_fe_indices, integer(0)),
          identical(r$connectivity_fe_names, character(0)),
          !isTRUE(r$connectivity_pair_explicit),
          identical(r$fe_split_status, "single_fe_dimension"),
          identical(r$connectivity_pair_status, "not_applicable"),
          identical(r$connected_mode, "diagnose"),
          identical(r$mobility_component_scope, "not_applicable"))
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
write.csv(t(r$beta2), file.path(args[4], "r_beta2.csv"),
          row.names = FALSE)
write.csv(r$beta2_cov, file.path(args[4], "r_beta2_cov.csv"),
          row.names = FALSE)
write.csv(r$auxiliary_loadings,
          file.path(args[4], "r_auxiliary_loadings.csv"),
          row.names = FALSE)
write.csv(cbind(r$auxiliary_loading_ss_ratio,
                r$auxiliary_loading_rank,
                r$auxiliary_loading_condition_number),
          file.path(args[4], "r_auxiliary_loading_diagnostics.csv"),
          row.names = FALSE)
write.csv(r$auxiliary_loading_max_abs_z,
          file.path(args[4], "r_auxiliary_loading_max_abs_z.csv"),
          row.names = FALSE)
write.csv(r$auxiliary_loading_pvalue,
          file.path(args[4], "r_auxiliary_loading_pvalue.csv"),
          row.names = FALSE)
write.csv(r$auxiliary_loading_test_evaluated * 1,
          file.path(args[4], "r_auxiliary_loading_test_evaluated.csv"),
          row.names = FALSE)
write.csv(cbind(r$beta2_wald_stat, r$beta2_wald_df,
                r$beta2_wald_pvalue),
          file.path(args[4], "r_beta2_wald.csv"), row.names = FALSE)
write.csv(r$contribution_gradient_norm,
          file.path(args[4], "r_contribution_gradient_norm.csv"),
          row.names = FALSE)
write.csv(r$regular_inference_valid * 1,
          file.path(args[4], "r_regular_inference_valid.csv"),
          row.names = FALSE)
status_code <- matrix(-1, nrow(r$regular_inference_status),
                      ncol(r$regular_inference_status))
status_code[r$regular_inference_status == "nonregular_not_ruled_out"] <- 0
status_code[r$regular_inference_status == "regular_beta_nonzero"] <- 1
status_code[r$regular_inference_status == "regular_loading_nonzero"] <- 2
write.csv(status_code,
          file.path(args[4], "r_regular_inference_status_code.csv"),
          row.names = FALSE)
write.csv(matrix(c(as.numeric(r$regular_inference_all_valid),
                   r$regularity_test_alpha), nrow = 1),
          file.path(args[4], "r_regular_meta.csv"), row.names = FALSE)
write.csv(cbind(r$fe_total$coef, r$fe_total$se),
          file.path(args[4], "r_fe_total.csv"), row.names = FALSE)
write.csv(matrix(tab$share, nrow = 1),
          file.path(args[4], "r_share_focal.csv"), row.names = FALSE)
write.csv(matrix(tab$share_std_error, nrow = 1),
          file.path(args[4], "r_share_se_focal.csv"), row.names = FALSE)
write.csv(matrix(tab$share_denominator_t[1], nrow = 1),
          file.path(args[4], "r_share_t_focal.csv"), row.names = FALSE)
write.csv(matrix(as.numeric(
  tab$share_interval_status[1] == "valid_first_order"
), nrow = 1), file.path(args[4], "r_share_status_code_focal.csv"),
row.names = FALSE)
write.csv(matrix(c(tab$share_t_min[1], r$fe_variance_ratio_min), nrow = 1),
          file.path(args[4], "r_gate_meta.csv"), row.names = FALSE)
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
                   r$gpu_absorption_iterations,
                   r$n_mobility_components,
                   r$largest_mobility_component_n_obs,
                   r$largest_mobility_component_share,
                   r$largest_mobility_component_weight_share,
                   as.numeric(r$fe_split_identified),
                   r$connectivity_fe_index1,
                   r$connectivity_fe_index2,
                   as.numeric(r$connectivity_pair_explicit)), nrow = 1),
          file.path(args[4], "r_meta.csv"), row.names = FALSE)
writeLines(c(r$gpu_backend, r$gpu_status, r$fe_split_status,
             r$connectivity_pair_status, r$connected_mode,
             r$mobility_component_scope,
             tab$share_interval_status[1],
             paste(unname(r$fe_variance_status), collapse = " "),
             tab$share_se_type[1], r$total_se_type),
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
assert "`r(total_se_type)'" == "target_exact_base_vce_mixed_components_conditional_only_diagnostic"
assert "`r(inference_status)'" == "clustered_at_absorbing_fe"
assert r(absorbed_target_inference_valid) == 1
assert r(absorbing_fe_index) == 0
assert "`r(focal_indices)'" == "0"
assert "`r(share_se_type)'" == "joint_base_covariance_delta_method_weak_denominator_diagnostic_only"
matrix ABS_SHARE_SE = r(share_se)
assert !missing(ABS_SHARE_SE[1, 1])
assert !missing(ABS_SHARE_SE[1, 2])
assert r(gpu_requested) == {gpu_requested}
local gel_gpu_backend "`r(gpu_backend)'"
local gel_gpu_status "`r(gpu_status)'"
local gel_fe_split_status "`r(fe_split_status)'"
local gel_connectivity_pair_status "`r(connectivity_pair_status)'"
local gel_connected_mode "`r(connected_mode)'"
local gel_mobility_scope "`r(mobility_component_scope)'"
local gel_share_status_all "`r(share_interval_status)'"
local gel_share_status : word 1 of `gel_share_status_all'
local gel_fe_variance_status "`r(fe_variance_status)'"
local gel_share_se_type "`r(share_se_type)'"
local gel_total_se_type "`r(total_se_type)'"
assert (r(gpu_used) != 1 | "`gel_gpu_backend'" == "cuda")
assert (r(gpu_used) != 1 | "`gel_gpu_status'" == "used")
assert (r(gpu_used) != 1 | r(gpu_status_code) == 1)
assert (r(gpu_used) != 0 | "`gel_gpu_backend'" == "cpu")
assert (r(gpu_used) != 0 | "`gel_gpu_status'" != "used")
assert (r(gpu_used) != 0 | r(gpu_status_code) != 1)
assert ({gpu_requested} != 0 | "`gel_gpu_status'" == "not_requested")
assert r(n_mobility_components) == 0
assert r(fe_split_identified) == 0
assert r(connectivity_fe1_index) == -1
assert r(connectivity_fe2_index) == -1
assert r(connectivity_pair_explicit) == 0
assert "`gel_fe_split_status'" == "single_fe_dimension"
assert "`gel_connectivity_pair_status'" == "not_applicable"
assert "`gel_connected_mode'" == "diagnose"
assert "`gel_mobility_scope'" == "not_applicable"
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
matrix SHT0 = r(share_denominator_t)
matrix SHV0 = r(share_interval_status_code)
matrix SH = SH0[1, 1..2]
matrix SS = SS0[1, 1..2]
matrix SHT = SHT0[1, 1]
matrix SHV = SHV0[1, 1]
matrix GATE = (r(share_t_min), r(fe_variance_ratio_min))
matrix M = (r(identity_gap), r(n_obs_input), r(n_obs), r(n_obs_effective), ///
            r(n_singletons_dropped), r(df_full), r(converged), r(tol), ///
            r(fe_collinear_ss_ratio_tol), ///
            r(absorbed_target_inference_valid), r(absorbing_fe_index), ///
            r(df_base), r(n_clusters), ///
            r(near_fe_warn_upper), ///
            r(few_cluster_warning_threshold), r(threads_used), ///
            r(gpu_requested), r(gpu_used), r(gpu_status_code), ///
            r(gpu_attempted), r(gpu_absorption_converged), ///
            r(gpu_absorption_iterations), ///
            r(n_mobility_components), ///
            r(largest_mobility_component_n_obs), ///
            r(largest_mobility_component_share), ///
            r(largest_mobility_weight_share), ///
            r(fe_split_identified), ///
            r(connectivity_fe1_index), ///
            r(connectivity_fe2_index), ///
            r(connectivity_pair_explicit))

tempname GPUF
file open `GPUF' using "{td}/stata_gpu_contract.txt", write replace text
file write `GPUF' "`gel_gpu_backend'" _n "`gel_gpu_status'" _n ///
    "`gel_fe_split_status'" _n "`gel_connectivity_pair_status'" _n ///
    "`gel_connected_mode'" _n "`gel_mobility_scope'" _n ///
    "`gel_share_status'" _n "`gel_fe_variance_status'" _n ///
    "`gel_share_se_type'" _n "`gel_total_se_type'" _n
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
dump_matrix SHT, using("{td}/stata_share_t_focal.csv")
dump_matrix SHV, using("{td}/stata_share_status_code_focal.csv")
dump_matrix GATE, using("{td}/stata_gate_meta.csv")
dump_matrix M, using("{td}/stata_meta.csv")
'''


R_ABSORBED_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], args[2], .libPaths()))
library(xhdfe)
assign(".xhdfe_cpp_gelbach",
       get(".xhdfe_cpp_gelbach", envir = asNamespace("xhdfe")),
       envir = .GlobalEnv)
assign(".akm_id_codes",
       get(".akm_id_codes", envir = asNamespace("xhdfe")),
       envir = .GlobalEnv)
source(file.path(args[6], "r/xhdfe/R/gelbach.R"), local = .GlobalEnv)
source(file.path(args[6], "r/xhdfe/R/gelbach_features.R"),
       local = .GlobalEnv)
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
                    paste0("target_exact_base_vce_mixed_components",
                           "_conditional_only_diagnostic")),
          identical(r$inference_status, "clustered_at_absorbing_fe"),
          isTRUE(r$absorbed_target_inference_valid),
          identical(r$absorbing_fe_index, 0L),
          identical(r$focal_indices, 0L),
          identical(r$focal_names, "focal"))
stopifnot(identical(isTRUE(r$gpu_requested), gpu_requested))
stopifnot(r$n_mobility_components == 0L,
          !isTRUE(r$fe_split_identified),
          identical(r$connectivity_fe_indices, integer(0)),
          identical(r$connectivity_fe_names, character(0)),
          !isTRUE(r$connectivity_pair_explicit),
          identical(r$fe_split_status, "single_fe_dimension"),
          identical(r$connectivity_pair_status, "not_applicable"),
          identical(r$connected_mode, "diagnose"),
          identical(r$mobility_component_scope, "not_applicable"))
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
                    paste0("joint_base_covariance_delta_method",
                           "_weak_denominator_diagnostic_only")))
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
write.csv(matrix(tab$share_denominator_t[1], nrow = 1),
          file.path(args[4], "r_share_t_focal.csv"), row.names = FALSE)
write.csv(matrix(as.numeric(
  tab$share_interval_status[1] == "valid_first_order"
), nrow = 1), file.path(args[4], "r_share_status_code_focal.csv"),
row.names = FALSE)
write.csv(matrix(c(tab$share_t_min[1], r$fe_variance_ratio_min), nrow = 1),
          file.path(args[4], "r_gate_meta.csv"), row.names = FALSE)
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
                   r$gpu_absorption_iterations,
                   r$n_mobility_components,
                   r$largest_mobility_component_n_obs,
                   r$largest_mobility_component_share,
                   r$largest_mobility_component_weight_share,
                   as.numeric(r$fe_split_identified),
                   r$connectivity_fe_index1,
                   r$connectivity_fe_index2,
                   as.numeric(r$connectivity_pair_explicit)), nrow = 1),
          file.path(args[4], "r_meta.csv"), row.names = FALSE)
writeLines(c(r$gpu_backend, r$gpu_status, r$fe_split_status,
             r$connectivity_pair_status, r$connected_mode,
             r$mobility_component_scope,
             tab$share_interval_status[1],
             paste(unname(r$fe_variance_status), collapse = " "),
             tab$share_se_type[1], r$total_se_type),
           file.path(args[4], "r_gpu_contract.txt"))
'''


STATA_CONNECTIVITY_DO = r'''
clear all
set more off
adopath ++ "{stata_ado}"
import delimited using "{data}", clear asdouble
xhdfegelbach y [aweight=wgt], x1(x1) ///
    x2groups("observed = observed") fes(worker firm bridge) ///
    connectivityfes(worker bridge) {gpu_option}
assert r(converged) == 1
assert r(n_obs_input) == _N
assert r(n_obs) == _N
assert r(n_singletons_dropped) == 0
assert r(n_mobility_components) == 1
assert r(largest_mobility_component_n_obs) == _N
assert r(largest_mobility_component_share) == 1
assert r(largest_mobility_weight_share) == 1
assert r(fe_split_identified) == 0
assert "`r(fe_split_status)'" == "not_certified_multiway"
assert r(connectivity_fe1_index) == 0
assert r(connectivity_fe2_index) == 2
assert r(connectivity_pair_explicit) == 1
assert "`r(connectivity_fes)'" == "worker bridge"
assert "`r(connectivity_fe_indices)'" == "0 2"
assert "`r(connectivity_pair_status)'" == "connected"
assert "`r(connected_mode)'" == "diagnose"
assert "`r(mobility_component_scope)'" == "selected_fe_pair"
assert r(gpu_requested) == {gpu_requested}
local gel_gpu_backend "`r(gpu_backend)'"
local gel_gpu_status "`r(gpu_status)'"
local gel_fe_split_status "`r(fe_split_status)'"
local gel_connectivity_pair_status "`r(connectivity_pair_status)'"
local gel_connected_mode "`r(connected_mode)'"
local gel_mobility_scope "`r(mobility_component_scope)'"
assert (r(gpu_used) != 1 | "`gel_gpu_backend'" == "cuda")
assert (r(gpu_used) != 1 | "`gel_gpu_status'" == "used")
assert (r(gpu_used) != 1 | r(gpu_status_code) == 1)
assert (r(gpu_used) != 0 | "`gel_gpu_backend'" == "cpu")
assert (r(gpu_used) != 0 | "`gel_gpu_status'" != "used")
assert (r(gpu_used) != 0 | r(gpu_status_code) != 1)
matrix D = r(delta)
matrix C = r(cov)
matrix T = r(total)
matrix M = (r(identity_gap), r(n_obs_input), r(n_obs), ///
            r(n_singletons_dropped), r(n_mobility_components), ///
            r(largest_mobility_component_n_obs), ///
            r(largest_mobility_component_share), ///
            r(largest_mobility_weight_share), ///
            r(fe_split_identified), r(connectivity_fe1_index), ///
            r(connectivity_fe2_index), r(connectivity_pair_explicit), ///
            r(gpu_requested), r(gpu_used), r(gpu_status_code))

tempname GPUF
file open `GPUF' using "{td}/stata_gpu_contract.txt", write replace text
file write `GPUF' "`gel_gpu_backend'" _n "`gel_gpu_status'" _n ///
    "`gel_fe_split_status'" _n "`gel_connectivity_pair_status'" _n ///
    "`gel_connected_mode'" _n "`gel_mobility_scope'" _n
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
dump_matrix C, using("{td}/stata_cov.csv")
dump_matrix T, using("{td}/stata_total.csv")
dump_matrix M, using("{td}/stata_meta.csv")
'''


R_CONNECTIVITY_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], args[2], .libPaths()))
library(xhdfe)
assign(".xhdfe_cpp_gelbach",
       get(".xhdfe_cpp_gelbach", envir = asNamespace("xhdfe")),
       envir = .GlobalEnv)
assign(".akm_id_codes",
       get(".akm_id_codes", envir = asNamespace("xhdfe")),
       envir = .GlobalEnv)
source(file.path(args[6], "r/xhdfe/R/gelbach.R"), local = .GlobalEnv)
source(file.path(args[6], "r/xhdfe/R/gelbach_features.R"),
       local = .GlobalEnv)
options(digits = 17)
d <- read.csv(args[3], check.names = FALSE)
gpu_requested <- identical(args[5], "1")
r <- suppressWarnings(xhdfe_gelbach(
  d$y, d$x1, x2_groups = list(observed = d$observed),
  fes = list(worker = d$worker, firm = d$firm, bridge = d$bridge),
  weights = d$wgt, gpu = gpu_requested,
  connectivity_fes = c("worker", "bridge")
))
stopifnot(r$converged, r$n_obs_input == nrow(d), r$n_obs == nrow(d),
          r$n_singletons_dropped == 0, r$n_mobility_components == 1L,
          r$largest_mobility_component_n_obs == nrow(d),
          r$largest_mobility_component_share == 1,
          r$largest_mobility_component_weight_share == 1,
          !isTRUE(r$fe_split_identified),
          identical(r$fe_split_status, "not_certified_multiway"),
          identical(r$connectivity_fe_indices, c(0L, 2L)),
          identical(r$connectivity_fe_names, c("worker", "bridge")),
          isTRUE(r$connectivity_pair_explicit),
          identical(r$connectivity_pair_status, "connected"),
          identical(r$connected_mode, "diagnose"),
          identical(r$mobility_component_scope, "selected_fe_pair"),
          identical(isTRUE(r$gpu_requested), gpu_requested))
if (isTRUE(r$gpu_used)) {
  stopifnot(identical(r$gpu_backend, "cuda"),
            identical(r$gpu_status, "used"),
            as.integer(r$gpu_status_code) == 1L)
} else {
  stopifnot(identical(r$gpu_backend, "cpu"),
            !identical(r$gpu_status, "used"),
            as.integer(r$gpu_status_code) != 1L)
}
write.csv(r$delta, file.path(args[4], "r_delta.csv"), row.names = FALSE)
write.csv(r$cov, file.path(args[4], "r_cov.csv"), row.names = FALSE)
write.csv(cbind(r$total, r$total_se),
          file.path(args[4], "r_total.csv"), row.names = FALSE)
write.csv(matrix(c(r$identity_gap, r$n_obs_input, r$n_obs,
                   r$n_singletons_dropped, r$n_mobility_components,
                   r$largest_mobility_component_n_obs,
                   r$largest_mobility_component_share,
                   r$largest_mobility_component_weight_share,
                   as.numeric(r$fe_split_identified),
                   r$connectivity_fe_index1, r$connectivity_fe_index2,
                   as.numeric(r$connectivity_pair_explicit),
                   as.numeric(r$gpu_requested), as.numeric(r$gpu_used),
                   r$gpu_status_code), nrow = 1),
          file.path(args[4], "r_meta.csv"), row.names = FALSE)
writeLines(c(r$gpu_backend, r$gpu_status, r$fe_split_status,
             r$connectivity_pair_status, r$connected_mode,
             r$mobility_component_scope),
           file.path(args[4], "r_gpu_contract.txt"))
'''


STATA_COMMON_FE_DO = r'''
clear all
set more off
adopath ++ "{stata_ado}"
import delimited using "{data}", clear asdouble
xhdfegelbach y [aweight=wgt], x1(x11 x12) ///
    x2groups("observed = z1 z2") commonfes(common_fe) fes(added_fe) ///
    vce(cluster) cluster(cl) tol(1e-10) {gpu_option}
assert r(converged) == 1
assert r(n_common_fes) == 1
assert r(common_fes_applied) == 1
assert r(intercept_inference_available) == 0
assert "`r(common_fes)'" == "common_fe"
assert "`r(intercept_status)'" == "not_certified_common_fes"
assert "`r(identity_status)'" == "exact_ols_conditional_common_fes"
assert r(gpu_requested) == {gpu_requested}
matrix D = r(delta)
matrix C = r(cov)
matrix T = r(total)
matrix BC = r(base_cov)
matrix CDB = r(cov_delta_bbase)
matrix CTB = r(cov_total_bbase)
matrix M = (r(identity_gap), r(n_obs_input), r(n_obs), ///
            r(n_singletons_dropped), r(df_full), r(df_base), ///
            r(n_clusters), r(n_common_fes), r(common_fes_applied), ///
            r(intercept_inference_available), r(gpu_requested), ///
            r(gpu_used), r(gpu_status_code))
assert missing(T[3, 1]) & missing(T[3, 2])
assert missing(BC[3, 3])

local gel_gpu_backend "`r(gpu_backend)'"
local gel_gpu_status "`r(gpu_status)'"
local gel_fe_split_status "`r(fe_split_status)'"
local gel_connectivity_pair_status "`r(connectivity_pair_status)'"
local gel_connected_mode "`r(connected_mode)'"
local gel_mobility_scope "`r(mobility_component_scope)'"
tempname GPUF
file open `GPUF' using "{td}/stata_gpu_contract.txt", write replace text
file write `GPUF' "`gel_gpu_backend'" _n "`gel_gpu_status'" _n ///
    "`gel_fe_split_status'" _n "`gel_connectivity_pair_status'" _n ///
    "`gel_connected_mode'" _n "`gel_mobility_scope'" _n
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
dump_matrix C, using("{td}/stata_cov.csv")
dump_matrix T, using("{td}/stata_total.csv")
dump_matrix BC, using("{td}/stata_base_cov.csv")
dump_matrix CDB, using("{td}/stata_cov_delta_bbase.csv")
dump_matrix CTB, using("{td}/stata_cov_total_bbase.csv")
dump_matrix M, using("{td}/stata_meta.csv")
'''


R_COMMON_FE_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], args[2], .libPaths()))
library(xhdfe)
assign(".xhdfe_cpp_gelbach",
       get(".xhdfe_cpp_gelbach", envir = asNamespace("xhdfe")),
       envir = .GlobalEnv)
assign(".akm_id_codes",
       get(".akm_id_codes", envir = asNamespace("xhdfe")),
       envir = .GlobalEnv)
source(file.path(args[6], "r/xhdfe/R/gelbach.R"), local = .GlobalEnv)
source(file.path(args[6], "r/xhdfe/R/gelbach_features.R"),
       local = .GlobalEnv)
options(digits = 17)
d <- read.csv(args[3], check.names = FALSE)
gpu_requested <- identical(args[5], "1")
r <- xhdfe_gelbach(
  d$y, cbind(x11 = d$x11, x12 = d$x12),
  x2_groups = list(observed = cbind(z1 = d$z1, z2 = d$z2)),
  fes = list(added_fe = d$added_fe),
  common_fes = list(common_fe = d$common_fe),
  vce = "cluster", cluster = d$cl, weights = d$wgt,
  tol = 1e-10, gpu = gpu_requested
)
stopifnot(r$converged, r$n_common_fes == 1L,
          isTRUE(r$common_fes_applied),
          !isTRUE(r$intercept_inference_available),
          identical(r$common_fe_names, "common_fe"),
          identical(r$intercept_status, "not_certified_common_fes"),
          identical(r$identity_status, "exact_ols_conditional_common_fes"),
          is.na(r$total[3]), is.na(r$total_se[3]),
          identical(isTRUE(r$gpu_requested), gpu_requested))
write.csv(r$delta, file.path(args[4], "r_delta.csv"), row.names = FALSE,
          na = "")
write.csv(r$cov, file.path(args[4], "r_cov.csv"), row.names = FALSE,
          na = "")
write.csv(cbind(r$total, r$total_se),
          file.path(args[4], "r_total.csv"), row.names = FALSE, na = "")
write.csv(r$base_cov, file.path(args[4], "r_base_cov.csv"),
          row.names = FALSE, na = "")
write.csv(r$cov_delta_bbase,
          file.path(args[4], "r_cov_delta_bbase.csv"),
          row.names = FALSE, na = "")
write.csv(r$cov_total_bbase,
          file.path(args[4], "r_cov_total_bbase.csv"),
          row.names = FALSE, na = "")
write.csv(matrix(c(
  r$identity_gap, r$n_obs_input, r$n_obs, r$n_singletons_dropped,
  r$df_full, r$df_base, r$n_clusters, r$n_common_fes,
  as.numeric(r$common_fes_applied),
  as.numeric(r$intercept_inference_available),
  as.numeric(r$gpu_requested), as.numeric(r$gpu_used), r$gpu_status_code
), nrow = 1), file.path(args[4], "r_meta.csv"), row.names = FALSE)
writeLines(c(r$gpu_backend, r$gpu_status, r$fe_split_status,
             r$connectivity_pair_status, r$connected_mode,
             r$mobility_component_scope),
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
         str(int(args.gpu)), str(REPO_ROOT)],
        cwd=td, check=True, timeout=420,
    )


def compare_frontends(td, expected, prefix):
    ok = True
    for frontend in ("stata", "r"):
        for key, value in expected.items():
            if key in (
                    "gpu_backend", "gpu_status", "fe_split_status",
                    "connectivity_pair_status", "connected_mode",
                    "mobility_component_scope", "share_interval_status",
                    "fe_variance_status", "share_se_type",
                    "total_se_type"):
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
        ok &= check_text(
            f"{prefix}:{frontend}:fe_split_status",
            gpu_contract[2] if len(gpu_contract) > 2 else "",
            expected["fe_split_status"],
        )
        ok &= check_text(
            f"{prefix}:{frontend}:connectivity_pair_status",
            gpu_contract[3] if len(gpu_contract) > 3 else "",
            expected["connectivity_pair_status"],
        )
        ok &= check_text(
            f"{prefix}:{frontend}:connected_mode",
            gpu_contract[4] if len(gpu_contract) > 4 else "",
            expected["connected_mode"],
        )
        ok &= check_text(
            f"{prefix}:{frontend}:mobility_component_scope",
            gpu_contract[5] if len(gpu_contract) > 5 else "",
            expected["mobility_component_scope"],
        )
        for line, key in enumerate((
                "share_interval_status", "fe_variance_status",
                "share_se_type", "total_se_type"), start=6):
            if key in expected:
                ok &= check_text(
                    f"{prefix}:{frontend}:{key}",
                    gpu_contract[line] if len(gpu_contract) > line else "",
                    expected[key],
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
            "library(xhdfe);" +
            "assign('.xhdfe_cpp_gelbach',get('.xhdfe_cpp_gelbach'," +
            "envir=asNamespace('xhdfe')),envir=.GlobalEnv);" +
            "assign('.akm_id_codes',get('.akm_id_codes'," +
            "envir=asNamespace('xhdfe')),envir=.GlobalEnv);" +
            "source(" + json.dumps(os.path.join(
                repo, "r", "xhdfe", "R", "gelbach.R")) +
            ",local=.GlobalEnv);" +
            "source(" + json.dumps(os.path.join(
                repo, "r", "xhdfe", "R", "gelbach_features.R")) +
            ",local=.GlobalEnv);" +
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
    observed_names = [
        name for name in py["names"]
        if py["group_kinds"][name] == "x2"
    ]
    regular_valid = np.column_stack([
        py["regularity"][name]["regular_inference_valid"]
        for name in observed_names
    ]).astype(int)
    status_codes = {
        "not_certified": -1,
        "nonregular_not_ruled_out": 0,
        "regular_beta_nonzero": 1,
        "regular_loading_nonzero": 2,
    }
    regular_status_code = np.column_stack([
        [status_codes[value] for value in
         py["regularity"][name]["regular_inference_status"]]
        for name in observed_names
    ])
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
        "beta2": py["beta2"][None, :],
        "beta2_cov": py["beta2_cov"],
        "auxiliary_loadings": py["auxiliary_loadings"],
        "auxiliary_loading_diagnostics": np.asarray([
            [
                py["regularity"][name]["auxiliary_loading_ss_ratio"],
                py["regularity"][name]["auxiliary_loading_rank"],
                py["regularity"][name][
                    "auxiliary_loading_condition_number"
                ],
            ]
            for name in observed_names
        ]),
        "auxiliary_loading_max_abs_z": np.column_stack([
            py["regularity"][name]["auxiliary_loading_max_abs_z"]
            for name in observed_names
        ]),
        "auxiliary_loading_pvalue": np.column_stack([
            py["regularity"][name]["auxiliary_loading_pvalue"]
            for name in observed_names
        ]),
        "auxiliary_loading_test_evaluated": np.column_stack([
            py["regularity"][name]["auxiliary_loading_test_evaluated"]
            for name in observed_names
        ]).astype(int),
        "beta2_wald": np.asarray([
            [
                py["regularity"][name]["beta2_wald_stat"],
                py["regularity"][name]["beta2_wald_df"],
                py["regularity"][name]["beta2_wald_pvalue"],
            ]
            for name in observed_names
        ]),
        "contribution_gradient_norm": np.column_stack([
            py["regularity"][name]["contribution_gradient_norm"]
            for name in observed_names
        ]),
        "regular_inference_valid": regular_valid,
        "regular_inference_status_code": regular_status_code,
        "regular_meta": np.asarray([[
            float(py["regular_inference_all_valid"]),
            py["regularity_test_alpha"],
        ]]),
        "fe_total": np.column_stack(
            [py["fe_total"]["coef"], py["fe_total"]["se"]]),
        "share_focal": np.array([[row["share"] for row in tab]]),
        "share_se_focal": np.array(
            [[row["share_std_error"] for row in tab]]),
        "share_t_focal": np.asarray(
            [[tab[0]["share_denominator_t"]]], dtype=float),
        "share_status_code_focal": np.asarray([[
            float(tab[0]["share_interval_status"] == "valid_first_order")
        ]]),
        "gate_meta": np.asarray([[
            tab[0]["share_t_min"], py["fe_variance_ratio_min"]
        ]]),
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
            py["n_mobility_components"],
            py["largest_mobility_component_n_obs"],
            py["largest_mobility_component_share"],
            py["largest_mobility_component_weight_share"],
            float(py["fe_split_identified"]),
            py["connectivity_fe_index1"],
            py["connectivity_fe_index2"],
            float(py["connectivity_pair_explicit"]),
        ]]),
        "gpu_backend": py["gpu_backend"],
        "gpu_status": py["gpu_status"],
        "fe_split_status": py["fe_split_status"],
        "connectivity_pair_status": py["connectivity_pair_status"],
        "connected_mode": py["connected_mode"],
        "mobility_component_scope": py["mobility_component_scope"],
        "share_interval_status": tab[0]["share_interval_status"],
        "fe_variance_status":
            " ".join(py["fe_variance_status"]),
        "share_se_type": tab[0]["share_se_type"],
        "total_se_type": py["total"]["se_type"],
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
            ("joint_base_covariance_delta_method"
             "_weak_denominator_diagnostic_only")}:
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
        "share_t_focal": np.asarray(
            [[tab[0]["share_denominator_t"]]], dtype=float),
        "share_status_code_focal": np.asarray([[
            float(tab[0]["share_interval_status"] == "valid_first_order")
        ]]),
        "gate_meta": np.asarray([[
            tab[0]["share_t_min"], py["fe_variance_ratio_min"]
        ]]),
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
            py["n_mobility_components"],
            py["largest_mobility_component_n_obs"],
            py["largest_mobility_component_share"],
            py["largest_mobility_component_weight_share"],
            float(py["fe_split_identified"]),
            py["connectivity_fe_index1"],
            py["connectivity_fe_index2"],
            float(py["connectivity_pair_explicit"]),
        ]]),
        "gpu_backend": py["gpu_backend"],
        "gpu_status": py["gpu_status"],
        "fe_split_status": py["fe_split_status"],
        "connectivity_pair_status": py["connectivity_pair_status"],
        "connected_mode": py["connected_mode"],
        "mobility_component_scope": py["mobility_component_scope"],
        "share_interval_status": tab[0]["share_interval_status"],
        "fe_variance_status":
            " ".join(py["fe_variance_status"]),
        "share_se_type": tab[0]["share_se_type"],
        "total_se_type": py["total"]["se_type"],
    }
    run_frontends(args, td, data, STATA_ABSORBED_DO, R_ABSORBED_SCRIPT)
    return (ok_gpu and ok_share_formula
            and compare_frontends(td, expected, "absorbed"))


def connectivity_fixture(gb, args, td):
    """Cross-language parity for an explicit pair in a three-way FE design."""
    rng = np.random.default_rng(20260725)
    n = 1200
    row = np.arange(n, dtype=np.int64)
    worker = row % 120
    firm = (row // 5 + 7 * worker) % 24
    bridge = (row // 7 + 13 * worker) % 23
    x1 = rng.normal(size=n)
    observed = 0.3 * x1 + rng.normal(size=n)
    y = (
        0.8 * x1 + 0.5 * observed + 0.01 * worker
        - 0.02 * firm + 0.03 * bridge + rng.normal(size=n)
    )
    weights = 1.0 + row % 3
    data = pd.DataFrame({
        "y": y, "x1": x1, "observed": observed, "worker": worker,
        "firm": firm, "bridge": bridge, "wgt": weights,
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        default = gb.decompose(
            y, x1, {"observed": observed},
            {"worker": worker, "firm": firm, "bridge": bridge},
            weights=weights, gpu=args.gpu,
        )
        py = gb.decompose(
            y, x1, {"observed": observed},
            {"worker": worker, "firm": firm, "bridge": bridge},
            weights=weights, gpu=args.gpu,
            connectivity_fes=("worker", "bridge"),
        )
    ok_gpu = gpu_contract(
        py, args.gpu, args.require_gpu_used, "connectivity:python")
    ok_selector = (
        default["n_mobility_components"] == 10
        and default["connectivity_pair_status"] == "disconnected"
        and default["fe_split_status"] == "not_certified_multiway"
        and py["n_mobility_components"] == 1
        and py["connectivity_pair_status"] == "connected"
        and py["connectivity_fe_indices"] == [0, 2]
        and py["connectivity_fe_names"] == ["worker", "bridge"]
        and py["connectivity_pair_explicit"] is True
        and py["connected_mode"] == "diagnose"
        and py["mobility_component_scope"] == "selected_fe_pair"
        and py["fe_split_identified"] is False
        and py["fe_split_status"] == "not_certified_multiway"
    )
    print(f"[{'PASS' if ok_selector else 'FAIL'}] "
          "connectivity:python:selector-contract")
    ok_inert = check(
        "connectivity:python:selector-numerically-inert",
        py["cov"], default["cov"], tol=1e-11,
    )
    expected = {
        "delta": np.column_stack(
            [py["delta"][name]["coef"] for name in py["names"]]),
        "cov": py["cov"],
        "total": np.column_stack([py["total"]["coef"], py["total"]["se"]]),
        "meta": np.array([[
            py["identity_gap"], py["n_obs_input"], py["n_obs"],
            py["n_singletons_dropped"], py["n_mobility_components"],
            py["largest_mobility_component_n_obs"],
            py["largest_mobility_component_share"],
            py["largest_mobility_component_weight_share"],
            float(py["fe_split_identified"]),
            py["connectivity_fe_index1"], py["connectivity_fe_index2"],
            float(py["connectivity_pair_explicit"]),
            float(py["gpu_requested"]), float(py["gpu_used"]),
            py["gpu_status_code"],
        ]]),
        "gpu_backend": py["gpu_backend"],
        "gpu_status": py["gpu_status"],
        "fe_split_status": py["fe_split_status"],
        "connectivity_pair_status": py["connectivity_pair_status"],
        "connected_mode": py["connected_mode"],
        "mobility_component_scope": py["mobility_component_scope"],
    }
    run_frontends(
        args, td, data, STATA_CONNECTIVITY_DO, R_CONNECTIVITY_SCRIPT)
    return (ok_gpu and ok_selector and ok_inert
            and compare_frontends(td, expected, "connectivity"))


def common_fe_fixture(gb, args, td):
    """Cross-language parity for FEs conditioned out of base and full."""
    rng = np.random.default_rng(20260726)
    n = 1200
    row = np.arange(n, dtype=np.int64)
    common_fe = row % 40
    added_fe = (row // 3 + 7 * common_fe) % 31
    cluster = row % 60
    x1 = rng.normal(size=(n, 2))
    x2 = np.column_stack([
        0.35 * x1[:, 0] + rng.normal(size=n),
        -0.20 * x1[:, 1] + rng.normal(size=n),
    ])
    y = (
        x1 @ np.array([1.1, -0.6])
        + x2 @ np.array([0.7, -0.25])
        + rng.normal(size=40)[common_fe]
        + rng.normal(size=31)[added_fe]
        + rng.normal(scale=0.3, size=n)
    )
    weights = 1.0 + row % 3
    data = pd.DataFrame({
        "y": y, "x11": x1[:, 0], "x12": x1[:, 1],
        "z1": x2[:, 0], "z2": x2[:, 1],
        "common_fe": common_fe, "added_fe": added_fe,
        "cl": cluster, "wgt": weights,
    })
    py = gb.decompose(
        y, x1, {"observed": x2}, {"added_fe": added_fe},
        common_fes={"common_fe": common_fe},
        vce="cluster", cluster=cluster, weights=weights,
        tol=1e-10, gpu=args.gpu, x1_names=["x11", "x12"],
    )
    ok_gpu = gpu_contract(
        py, args.gpu, args.require_gpu_used, "common-fe:python")
    ok_contract = (
        py["converged"]
        and py["n_common_fes"] == 1
        and py["common_fes_applied"] is True
        and py["common_fe_names"] == ["common_fe"]
        and py["intercept_inference_available"] is False
        and py["intercept_status"] == "not_certified_common_fes"
        and py["identity_status"] == "exact_ols_conditional_common_fes"
        and np.isnan(py["total"]["coef"][2])
        and np.isnan(py["total"]["se"][2])
        and py["fe_split_status"] == "single_fe_dimension"
    )
    print(f"[{'PASS' if ok_contract else 'FAIL'}] "
          "common-fe:python:conditional-contract")
    expected = {
        "delta": np.column_stack(
            [py["delta"][name]["coef"] for name in py["names"]]),
        "cov": py["cov"],
        "total": np.column_stack([py["total"]["coef"], py["total"]["se"]]),
        "base_cov": py["base_cov"],
        "cov_delta_bbase": py["cov_delta_bbase"],
        "cov_total_bbase": py["cov_total_bbase"],
        "meta": np.array([[
            py["identity_gap"], py["n_obs_input"], py["n_obs"],
            py["n_singletons_dropped"], py["df_full"], py["df_base"],
            py["n_clusters"], py["n_common_fes"],
            float(py["common_fes_applied"]),
            float(py["intercept_inference_available"]),
            float(py["gpu_requested"]), float(py["gpu_used"]),
            py["gpu_status_code"],
        ]]),
        "gpu_backend": py["gpu_backend"],
        "gpu_status": py["gpu_status"],
        "fe_split_status": py["fe_split_status"],
        "connectivity_pair_status": py["connectivity_pair_status"],
        "connected_mode": py["connected_mode"],
        "mobility_component_scope": py["mobility_component_scope"],
    }
    run_frontends(
        args, td, data, STATA_COMMON_FE_DO, R_COMMON_FE_SCRIPT)
    return (
        ok_gpu and ok_contract
        and compare_frontends(td, expected, "common-fe")
    )


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

    repo = str(REPO_ROOT)
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
        ok_connectivity = connectivity_fixture(
            gb, args, root / "connectivity")
        ok_common = common_fe_fixture(gb, args, root / "common_fe")
        ok_examples = run_shipped_examples(args, repo, root / "examples")
    if not (
            ok_standard and ok_absorbed and ok_connectivity
            and ok_common and ok_examples):
        raise SystemExit(1)
    print("ALL FRONT-END PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
