#!/usr/bin/env python3
"""Cross-front-end Gelbach parity gates.

Preserves the weighted/clustered standard-Gelbach fixture and adds a second
fixture for the opt-in absorbed-target estimand. Generated files stay in a
temporary directory under build/.
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
    return pd.read_csv(path).to_numpy(dtype=float)


def check(name, got, expected, tol=1e-11):
    got = np.asarray(got, dtype=float)
    expected = np.asarray(expected, dtype=float)
    same_shape = got.shape == expected.shape
    diff = (float(np.max(np.abs(got - expected)))
            if same_shape and got.size else (0.0 if same_shape else np.inf))
    ok = same_shape and diff <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
          f"shape={got.shape}, max|diff|={diff:.2e}")
    return ok


STATA_STANDARD_DO = r'''
clear all
set more off
adopath ++ "{stata_ado}"
import delimited using "{data}", clear asdouble
xhdfegelbach y [aweight=wgt], x1(x11 x12) ///
    x2groups("A = a1 a2 : B = b1") fes(firm) ///
    vce(cluster) cluster(cl) tol(1e-10) focal(x11) shares(movement)
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
matrix D = r(delta)
matrix S = r(se)
matrix T = r(total)
matrix C = r(cov)
matrix TC = r(total_cov)
matrix BB = r(b_base)
matrix BF = r(b_full)
matrix AM = r(absorbed_mask)
matrix FT = r(fe_total)
matrix SH0 = r(share)
matrix SS0 = r(share_se)
matrix SH = SH0[1, 1..3]
matrix SS = SS0[1, 1..3]
matrix M = (r(identity_gap), r(n_obs_input), r(n_obs), r(n_obs_effective), ///
            r(n_singletons_dropped), r(df_full), r(converged), r(tol), ///
            r(fe_collinear_ss_ratio_tol), ///
            r(absorbed_target_inference_valid), r(absorbing_fe_index))

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
dump_matrix BB, using("{td}/stata_b_base.csv")
dump_matrix BF, using("{td}/stata_b_full.csv")
dump_matrix AM, using("{td}/stata_absorbed_mask.csv")
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
r <- xhdfe_gelbach(
  d$y, cbind(x11 = d$x11, x12 = d$x12),
  x2_groups = list(A = cbind(d$a1, d$a2), B = d$b1),
  fes = list(FIRM = d$firm),
  vce = "cluster", cluster = d$cl, weights = d$wgt, tol = 1e-10,
  focal = "x11"
)
stopifnot(r$converged, identical(r$estimand, "coefficient_movement"),
          identical(r$causal_interpretation, FALSE),
          identical(r$absorbed_targets, integer(0)),
          identical(r$absorbed_target_names, character(0)),
          identical(unname(r$b_full_status), c("estimated", "estimated")),
          identical(r$inference_status, "not_applicable"),
          identical(r$focal_indices, 0L),
          identical(r$focal_names, "x11"))
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
write.csv(t(r$b_base), file.path(args[4], "r_b_base.csv"), row.names = FALSE)
write.csv(t(r$b_full), file.path(args[4], "r_b_full.csv"), row.names = FALSE)
write.csv(t(as.integer(r$absorbed_mask)),
          file.path(args[4], "r_absorbed_mask.csv"), row.names = FALSE)
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
                   r$absorbing_fe_index), nrow = 1),
          file.path(args[4], "r_meta.csv"), row.names = FALSE)
'''


STATA_ABSORBED_DO = r'''
clear all
set more off
adopath ++ "{stata_ado}"
import delimited using "{data}", clear asdouble
xhdfegelbach y, x1(focal experience) x2groups("observed = observed") ///
    fes(worker) absorbedtargets(focal) ///
    vce(cluster) cluster(worker) tol(1e-10) focal(focal) shares(base)
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
assert "`r(share_se_type)'" == "not_available_joint_base_covariance"
matrix ABS_SHARE_SE = r(share_se)
assert missing(ABS_SHARE_SE[1, 1])
matrix D = r(delta)
matrix S = r(se)
matrix T = r(total)
matrix C = r(cov)
matrix TC = r(total_cov)
matrix BB = r(b_base)
matrix BF = r(b_full)
matrix AM = r(absorbed_mask)
matrix FT = r(fe_total)
matrix M = (r(identity_gap), r(n_obs_input), r(n_obs), r(n_obs_effective), ///
            r(n_singletons_dropped), r(df_full), r(converged), r(tol), ///
            r(fe_collinear_ss_ratio_tol), ///
            r(absorbed_target_inference_valid), r(absorbing_fe_index))

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
dump_matrix BB, using("{td}/stata_b_base.csv")
dump_matrix BF, using("{td}/stata_b_full.csv")
dump_matrix AM, using("{td}/stata_absorbed_mask.csv")
dump_matrix FT, using("{td}/stata_fe_total.csv")
dump_matrix M, using("{td}/stata_meta.csv")
'''


R_ABSORBED_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], args[2], .libPaths()))
library(xhdfe)
options(digits = 17)
d <- read.csv(args[3], check.names = FALSE)
x1 <- cbind(focal = d$focal, experience = d$experience)
r <- xhdfe_gelbach(
  d$y, x1,
  x2_groups = list(observed = d$observed),
  fes = list(worker = d$worker),
  vce = "cluster", cluster = d$worker, tol = 1e-10,
  absorbed_targets = "focal", focal = "focal"
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
tab <- xhdfe_gelbach_tidy(r, share = "base", include_total = FALSE,
                          include_full = FALSE)
stopifnot(all(is.na(tab$share_std_error)),
          identical(unique(tab$share_se_type),
                    "not_available_joint_base_covariance"))
write.csv(r$delta, file.path(args[4], "r_delta.csv"), row.names = FALSE)
write.csv(r$se, file.path(args[4], "r_se.csv"), row.names = FALSE)
write.csv(cbind(r$total, r$total_se),
          file.path(args[4], "r_total.csv"), row.names = FALSE)
write.csv(r$cov, file.path(args[4], "r_cov.csv"), row.names = FALSE)
write.csv(r$total_cov, file.path(args[4], "r_total_cov.csv"), row.names = FALSE)
write.csv(t(r$b_base), file.path(args[4], "r_b_base.csv"), row.names = FALSE)
write.csv(t(r$b_full), file.path(args[4], "r_b_full.csv"), row.names = FALSE)
write.csv(t(as.integer(r$absorbed_mask)),
          file.path(args[4], "r_absorbed_mask.csv"), row.names = FALSE)
write.csv(cbind(r$fe_total$coef, r$fe_total$se),
          file.path(args[4], "r_fe_total.csv"), row.names = FALSE)
write.csv(matrix(c(r$identity_gap, r$n_obs_input, r$n_obs, r$n_obs_effective,
                   r$n_singletons_dropped, r$df_full,
                   as.numeric(r$converged), r$tol,
                   r$fe_collinear_ss_ratio_tol,
                   as.numeric(r$absorbed_target_inference_valid),
                   r$absorbing_fe_index), nrow = 1),
          file.path(args[4], "r_meta.csv"), row.names = FALSE)
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
        ),
        encoding="utf-8",
    )
    subprocess.run([args.stata, "-q", "-b", "do", str(do_path)],
                   cwd=td, check=True, timeout=420)

    r_path = td / "frontends.R"
    r_path.write_text(r_template, encoding="utf-8")
    subprocess.run(
        [args.rscript, str(r_path), os.path.abspath(args.r_lib),
         os.path.abspath(args.rcpp_lib), str(data_path), str(td)],
        cwd=td, check=True, timeout=420,
    )


def compare_frontends(td, expected, prefix):
    ok = True
    for frontend in ("stata", "r"):
        for key, value in expected.items():
            ok &= check(
                f"{prefix}:{frontend}:{key}",
                read_matrix(td / f"{frontend}_{key}.csv"),
                value,
            )
    return ok


def run_shipped_examples(args, repo, td):
    """Execute the committed standard and absorbed examples in all frontends."""
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
        x1_names=["x11", "x12"], focal="x11",
    )
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
        "b_base": py["b_base"][None, :],
        "b_full": py["b_full"][None, :],
        "absorbed_mask": np.asarray(py["absorbed_mask"], dtype=int)[None, :],
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
        ]]),
    }
    run_frontends(args, td, data, STATA_STANDARD_DO, R_STANDARD_SCRIPT)
    return compare_frontends(td, expected, "standard")


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
        x1_names=["focal", "experience"], focal="focal",
    )
    tab = gb.tidy(py, share="base", include_total=False, include_full=False)
    if not all(np.isnan(row["share_std_error"]) for row in tab):
        raise AssertionError("base-share inference must remain unavailable")
    expected = {
        "delta": np.column_stack(
            [py["delta"][name]["coef"] for name in py["names"]]),
        "se": np.column_stack(
            [py["delta"][name]["se"] for name in py["names"]]),
        "total": np.column_stack([py["total"]["coef"], py["total"]["se"]]),
        "cov": py["cov"],
        "total_cov": py["total_cov"],
        "b_base": py["b_base"][None, :],
        "b_full": py["b_full"][None, :],
        "absorbed_mask": np.asarray(py["absorbed_mask"], dtype=int)[None, :],
        "fe_total": np.column_stack(
            [py["fe_total"]["coef"], py["fe_total"]["se"]]),
        "meta": np.array([[
            py["identity_gap"], py["n_obs_input"], py["n_obs"],
            py["n_obs_effective"],
            py["n_singletons_dropped"], py["df_full"],
            float(py["converged"]), py["tol"],
            py["fe_collinear_ss_ratio_tol"],
            float(py["absorbed_target_inference_valid"]),
            py["absorbing_fe_index"],
        ]]),
    }
    run_frontends(args, td, data, STATA_ABSORBED_DO, R_ABSORBED_SCRIPT)
    return compare_frontends(td, expected, "absorbed")


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
    args = ap.parse_args()

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
