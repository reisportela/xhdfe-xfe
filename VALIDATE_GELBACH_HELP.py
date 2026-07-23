#!/usr/bin/env python3
"""Static and runtime contract checks for the Gelbach help surfaces.

This validator deliberately does not fit a model.  It verifies that the
Python, Stata, and R documentation exposes the public arguments, stored-result
schema, inference qualifications, examples, and deliberate limitations that
the wrappers currently implement.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re
from pathlib import Path

import xhdfe
from xhdfe import gelbach
from xhdfe import _help


ROOT = Path(__file__).resolve().parent
CHECKS = 0


def require(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def require_tokens(text: str, tokens: list[str], label: str) -> None:
    normalized_text = " ".join(text.split())
    missing = [
        token for token in tokens
        if " ".join(token.split()) not in normalized_text
    ]
    require(not missing, f"{label} is missing: {', '.join(missing)}")


def parameter_names(function) -> list[str]:
    return list(inspect.signature(function).parameters)


def validate_python_help() -> None:
    default_help = xhdfe.help_text()
    topic_help = xhdfe.help_text("gelbach")
    require(default_help.startswith("# xhdfe Python help"),
            "the historical no-argument Python help topic changed")
    require(xhdfe.__version__ in default_help,
            "the packaged Python help does not identify the current version")
    require(Path(xhdfe.help_path("gelbach")).exists(),
            "the packaged Gelbach help resource is unavailable")

    require(parameter_names(gelbach.decompose) == [
        "y", "x1", "x2_groups", "fes", "vce", "cluster", "gamma0",
        "cov0", "tol", "num_threads", "weights", "fweights",
        "absorbed_targets", "x1_names", "focal", "gpu",
    ], "unexpected Python decompose signature")
    require(parameter_names(gelbach.tidy) == [
        "result", "focal", "include_intercept", "include_total",
        "include_full", "conf_level", "share", "share_tol",
    ], "unexpected Python tidy signature")
    require(parameter_names(gelbach.contrast) == [
        "result", "focal", "groups", "conf_level",
    ], "unexpected Python contrast signature")

    require_tokens(topic_help, [
        "absorbed_targets", "x1_names", "focal", "x2_groups", "fes",
        "gamma0", "cov0", "num_threads", "weights", "fweights", "gpu",
        "n_obs_input", "n_obs_effective", "n_singletons_dropped",
        "b_full_status", "gamma", "base_cov", "cov_delta_bbase",
        "cov_total_bbase", "absorbed_target_inference_valid",
        "absorbing_fe_index", "x1_fe_collinear_ratio",
        "x1_near_collinear_mask", "fe_collinear_ss_ratio_tol",
        "near_fe_collinear_ss_ratio_warn_upper",
        "few_cluster_warning_threshold", "df_base", "n_clusters",
        "threads_used", "gpu_requested", "gpu_used", "gpu_status",
        "identity_gap",
        "total_se_type", "conditional_gamma0", "group-major",
        "movement", "base_fixed", "full_model_residual", "contrast",
        "joint_base_covariance_delta_method", "linear probability model",
        "random-design", "Deliberate boundaries", "not causal mediation",
        "examples/gelbach_example.py", "examples/gelbach_absorbed_target.py",
    ], "Python Gelbach help")

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        require(_help.main(["--topics"]) == 0, "--topics returned nonzero")
    require(stream.getvalue().splitlines() == ["gelbach", "xhdfe"],
            "--topics did not expose both help topics")

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        require(_help.main(["gelbach"]) == 0,
                "Gelbach CLI help returned nonzero")
    require(stream.getvalue() == topic_help,
            "CLI Gelbach help differs from the packaged resource")


def validate_stata_help() -> None:
    ado = (ROOT / "stata" / "xhdfegelbach.ado").read_text(encoding="utf-8")
    help_text = (ROOT / "stata" / "xhdfegelbach.sthlp").read_text(
        encoding="utf-8"
    )
    ado_lower = ado.lower()
    help_lower = help_text.lower()
    ado_version = re.search(
        r"^\*!\s+version\s+(\S+)\s+(\S+)", ado, flags=re.MULTILINE
    )
    require(ado_version is not None, "Stata component version is unparseable")
    require(ado_version.group(1) in help_text,
            "Stata help omits the component version")
    require(ado_version.group(2) in help_text,
            "Stata help omits the component date")

    options = [
        "x1", "x2groups", "fes", "absorbedtargets", "focal", "shares",
        "sharetol", "level", "vce", "cluster", "gamma0", "cov0", "tol",
        "threads", "gpu", "verbose",
    ]
    for option in options:
        require(option in ado_lower, f"Stata implementation lacks {option}")
        require(option in help_lower, f"Stata help lacks {option}")

    stored_results = [
        "delta", "se", "total", "total_cov", "cov", "b_full", "b_base",
        "absorbed_mask", "x1_fe_collinear_ratio",
        "x1_near_collinear_mask", "gamma", "base_cov",
        "cov_delta_bbase", "cov_total_bbase", "fe_total",
        "share", "share_se", "share_ci_low",
        "share_ci_high", "share_defined", "residual_share", "identity_gap",
        "n_obs_input", "n_obs", "n_obs_effective", "n_singletons_dropped",
        "df_full", "df_base", "n_clusters", "fe_collinear_ss_ratio_tol",
        "near_fe_warn_upper",
        "few_cluster_warning_threshold",
        "absorbed_target_inference_valid", "absorbing_fe_index", "converged",
        "tol", "focal_selection_explicit", "conf_level", "share_tol",
        "threads_used", "gpu_requested", "gpu_used", "gpu_status_code",
        "gpu_attempted", "gpu_absorption_converged",
        "gpu_absorption_iterations", "vce", "groups", "x1_names",
        "focal_indices", "focal_names", "share_denominator",
        "share_se_type", "share_units", "notes", "estimand",
        "identity_status", "absorbed_targets", "absorbed_target_names",
        "b_full_status", "focal_status", "observed_se_type",
        "total_se_type", "inference_status", "causal_interpretation",
        "fe_se_type", "gpu_backend", "gpu_status",
    ]
    require_tokens(help_lower, stored_results, "Stata stored-results help")
    require_tokens(help_text, [
        "Specification and sample contract", "Displayed output",
        "Programmatic matrix layout", "Deliberate limits", "Stored results",
        "0 (imposed)", "random-design", "group-major", "one-way clustered",
        "linear probability model", "joint_base_covariance_delta_method",
        "examples/gelbach_example.do", "examples/gelbach_absorbed_target.do",
    ], "Stata Gelbach help")


def validate_r_help() -> None:
    main = (ROOT / "r" / "xhdfe" / "man" / "xhdfe_gelbach.Rd").read_text(
        encoding="utf-8"
    )
    tidy = (ROOT / "r" / "xhdfe" / "man" /
            "xhdfe_gelbach_tidy.Rd").read_text(encoding="utf-8")
    contrast = (ROOT / "r" / "xhdfe" / "man" /
                "xhdfe_gelbach_contrast.Rd").read_text(encoding="utf-8")

    require_tokens(main, [
        "x2_groups", "fes", "vce", "cluster", "gamma0", "cov0", "tol",
        "num_threads", "weights", "fweights", "absorbed_targets", "focal",
        "Specification and sample contract", "Inference and covariance",
        "Printing and reporting", "Deliberate limits", "b_base", "b_full",
        "b_full_status", "x1_absorbed", "delta",
        "total_se", "total_cov", "fe_total", "focal_indices",
        "absorbed_target_inference_valid", "absorbing_fe_index",
        "fe_collinear_ss_ratio_tol", "group-major", "zero-based",
        "random-design", "\\examples", "imposed_zero",
    ], "R main Gelbach help")
    require_tokens(tidy, [
        "include_intercept", "include_total", "include_full", "conf_level",
        "share_tol", "Share contract", "Added rows", "share_std_error",
        "share_defined", "movement", "base_fixed", "full_model_residual",
        "never truncated or renormalized", "\\examples",
    ], "R tidy help")
    require_tokens(contrast, [
        "focal", "groups", "conf_level", "named numeric vector",
        "std_error", "joint_covariance_including_conditional_fe",
        "\\examples",
    ], "R contrast help")


def validate_examples() -> None:
    for stem in ("gelbach_example", "gelbach_absorbed_target"):
        for suffix in (".py", ".R", ".do"):
            require((ROOT / "examples" / f"{stem}{suffix}").is_file(),
                    f"missing documented example: {stem}{suffix}")


def main() -> int:
    validate_python_help()
    validate_stata_help()
    validate_r_help()
    validate_examples()
    print(f"Gelbach help contract: PASS ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
