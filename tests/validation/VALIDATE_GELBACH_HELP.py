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


ROOT = Path(__file__).resolve().parents[2]
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
        "absorbed_targets", "x1_names", "focal", "gpu", "connected",
        "connectivity_fes", "common_fes", "sample_info",
        "fe_variance_ratio_min",
    ], "unexpected Python decompose signature")
    require(parameter_names(gelbach.tidy) == [
        "result", "focal", "include_intercept", "include_total",
        "include_full", "conf_level", "share", "share_tol", "share_t_min",
    ], "unexpected Python tidy signature")
    require(parameter_names(gelbach.contrast) == [
        "result", "focal", "groups", "conf_level",
    ], "unexpected Python contrast signature")
    require(parameter_names(gelbach.bootstrap) == [
        "y", "x1", "x2_groups", "fes", "method", "bootstrap_cluster",
        "reps", "seed", "conf_level", "ci_method", "min_valid_reps",
        "store_draws", "require_gpu_used", "share_tol",
        "decompose_kwargs",
    ], "unexpected Python bootstrap signature")
    require(parameter_names(gelbach.etable) == [
        "result", "panels", "format", "type", "focal", "keep", "drop",
        "exact_match", "labels", "include_other", "digits", "caption",
        "notes", "conf_level", "interval", "share_tol", "share_t_min",
    ], "unexpected Python etable signature")
    require(parameter_names(gelbach.waterfall_data) == [
        "result", "focal", "keep", "drop", "exact_match", "labels",
        "include_other", "share_tol",
    ], "unexpected Python waterfall_data signature")
    require(parameter_names(gelbach.coefplot) == [
        "result", "focal", "annotate_shares", "title", "figsize", "keep",
        "drop", "exact_match", "labels", "notes", "include_other",
        "share_tol", "ax", "save", "show",
    ], "unexpected Python coefplot signature")

    require_tokens(topic_help, [
        "absorbed_targets", "x1_names", "focal", "x2_groups", "fes",
        "common_fes", "common_fe_names", "n_common_fes",
        "common_fes_applied", "intercept_inference_available",
        "intercept_status", "not_certified_common_fes",
        "gamma0", "cov0", "num_threads", "weights", "fweights", "gpu",
        "n_obs_input", "n_obs_effective", "n_singletons_dropped",
        "sample_info", "sample_index", "sample_mask", "sample_hash",
        "sample_hash_algorithm", "sample_index_scope",
        "fnv1a64-le-v1", "non-cryptographic",
        "b_full_status", "gamma", "base_cov", "cov_delta_bbase",
        "cov_total_bbase", "absorbed_target_inference_valid",
        "beta2", "beta2_cov", "auxiliary_loadings", "regularity",
        "regular_inference_valid", "regular_inference_status",
        "regular_inference_all_valid", "regularity_test_alpha",
        "nonregular_not_ruled_out",
        "absorbing_fe_index", "x1_fe_collinear_ratio",
        "fe_variance_status", "fe_variance_ratio_min",
        "conditional_only_between_fe_dominant",
        "x1_near_collinear_mask", "fe_collinear_ss_ratio_tol",
        "near_fe_collinear_ss_ratio_warn_upper",
        "few_cluster_warning_threshold", "df_base", "n_clusters",
        "n_mobility_components", "largest_mobility_component_n_obs",
        "largest_mobility_component_share",
        "largest_mobility_component_weight_share",
        "fe_split_identified", "fe_split_status",
        "connectivity_fe_index1", "connectivity_fe_index2",
        "connectivity_fe_indices", "connectivity_fe_names",
        "connectivity_pair_explicit", "connectivity_pair_status",
        "connected_mode", "mobility_component_scope",
        "not_certified_multiway", "not_certified_with_common_fes",
        "threads_used", "gpu_requested", "gpu_used", "gpu_status",
        "identity_gap",
        "total_se_type", "conditional_gamma0", "group-major",
        "movement", "base_fixed", "full_model_residual", "contrast",
        "share_t_min", "share_denominator_t", "share_interval_status",
        "weak_denominator_delta_method_unreliable",
        "half the family-wise level",
        "Full-refit pairs bootstrap", "bootstrap_cluster", "cluster_pairs",
        "SeedSequence", "PCG64", "L'Ecuyer-CMRG", "min_valid_reps",
        "reps_valid", "reps_failed", "resampling_unit",
        "bootstrap_cluster_name", "draws_stored", "point_vce",
        "replicate_vce", "gpu_used_all_valid", "failure_counts",
        "percentile", "basic", "total_share_base", "require_gpu_used",
        "Tables and waterfall plots", "etable", "waterfall_data", "coefplot",
        "type=None", "include_other=True", "digits=3", "caption=None",
        "annotate_shares=True", "show=False",
        "share_full", "share_explained", "Other (filtered)",
        "great_tables",
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
        "x1", "x2groups", "fes", "commonfes", "absorbedtargets", "focal", "shares",
        "sharetol", "sharetmin", "fevarmin", "level", "vce", "cluster",
        "gamma0", "cov0", "tol",
        "threads", "gpu", "verbose", "connected", "connectivityfes",
        "sampleaudit", "generate",
    ]
    for option in options:
        require(option in ado_lower, f"Stata implementation lacks {option}")
        require(option in help_lower, f"Stata help lacks {option}")

    stored_results = [
        "delta", "se", "total", "total_cov", "cov", "b_full", "b_base",
        "absorbed_mask", "x1_fe_collinear_ratio",
        "x1_near_collinear_mask", "gamma", "base_cov",
        "beta2", "beta2_cov", "auxiliary_loadings",
        "auxiliary_loading_diagnostics",
        "auxiliary_loading_max_abs_z", "auxiliary_loading_pvalue",
        "auxiliary_loading_test_evaluated", "beta2_wald",
        "contribution_gradient_norm", "regular_inference_valid",
        "regular_inference_status_code", "regular_inference_all_valid",
        "regularity_test_alpha", "regular_inference_status",
        "regular_inference_status_order",
        "regular_inference_codebook",
        "cov_delta_bbase", "cov_total_bbase", "fe_total",
        "share", "share_se", "share_ci_low",
        "share_ci_high", "share_defined", "share_denominator_t",
        "share_interval_status_code", "residual_share", "identity_gap",
        "n_obs_input", "n_obs", "n_obs_effective", "n_singletons_dropped",
        "n_common_fes", "common_fes_applied",
        "intercept_inference_available",
        "n_mobility_components", "largest_mobility_component_n_obs",
        "largest_mobility_component_share",
        "largest_mobility_weight_share", "fe_split_identified",
        "connectivity_fe1_index", "connectivity_fe2_index",
        "connectivity_pair_explicit",
        "df_full", "df_base", "n_clusters", "fe_collinear_ss_ratio_tol",
        "near_fe_warn_upper",
        "few_cluster_warning_threshold",
        "absorbed_target_inference_valid", "absorbing_fe_index", "converged",
        "tol", "focal_selection_explicit", "conf_level", "share_tol",
        "share_t_min", "fe_variance_ratio_min",
        "sample_info_requested", "sample_hash", "sample_hash_algorithm",
        "sample_index_scope", "sample_variable",
        "threads_used", "gpu_requested", "gpu_used", "gpu_status_code",
        "gpu_attempted", "gpu_absorption_converged",
        "gpu_absorption_iterations", "vce", "groups", "common_fes",
        "intercept_status", "x1_names",
        "focal_indices", "focal_names", "share_denominator",
        "share_se_type", "share_units", "notes", "estimand",
        "identity_status", "absorbed_targets", "absorbed_target_names",
        "b_full_status", "focal_status", "observed_se_type",
        "total_se_type", "inference_status", "causal_interpretation",
        "fe_se_type", "fe_variance_status", "fe_split_status",
        "mobility_component_scope",
        "connected_mode", "connectivity_fes",
        "connectivity_fe_indices", "connectivity_pair_status",
        "gpu_backend", "gpu_status",
    ]
    require_tokens(help_lower, stored_results, "Stata stored-results help")
    require_tokens(help_text, [
        "Specification and sample contract", "Displayed output",
        "Programmatic matrix layout", "Deliberate limits", "Stored results",
        "0 (imposed)", "random-design", "group-major", "one-way clustered",
        "linear probability model", "joint_base_covariance_delta_method",
        "not_certified_with_common_fes",
        "not_applicable_common_fe_intercept",
        "examples/gelbach_example.do", "examples/gelbach_absorbed_target.do",
    ], "Stata Gelbach help")

    companions = {
        "xhdfegelbachbootstrap": [
            "method", "bootcluster", "reps", "seed", "minvalid",
            "bootci", "requiregpu", "bootstrap_ledger",
            "bootstrap_delta_draws", "bootstrap_total_share_base_draws",
            "gpu_used_point", "gpu_used_all_valid", "point_gpu_backend",
            "bootstrap_cluster", "min_valid_reps", "share_tol", "rng",
            "version", "identity_status", "groups", "x1_names",
            "focal_names", "notes", "cov_total_bbase",
            "percentile", "basic", "failed closed", "not causal mediation",
        ],
        "xhdfegelbachetable": [
            "panels", "format", "share_full", "share_explained", "markdown",
            "latex", "html", "csv", "keep", "drop", "labels",
            "noother", "interval", "sharetol", "sharetmin",
            "Other (filtered)", "joint covariance",
            "normal_delta_diagnostic_only_weak_denominator",
            "c(level)", "bootstrap_", "immediately follow",
            "not causal mediation",
        ],
        "xhdfegelbachcoefplot": [
            "focal", "keep", "drop", "exact", "labels", "noother",
            "Other (filtered)", "accounting identity", "saving", "nodraw",
            "volatile", "returns no stored results",
            "not evidence of causal mediation",
        ],
    }
    package_manifest = (ROOT / "stata" / "xhdfe.pkg").read_text(
        encoding="utf-8"
    )
    for command, tokens in companions.items():
        companion_ado = ROOT / "stata" / f"{command}.ado"
        companion_help = ROOT / "stata" / f"{command}.sthlp"
        require(companion_ado.is_file(), f"missing {command}.ado")
        require(companion_help.is_file(), f"missing {command}.sthlp")
        ado_text = companion_ado.read_text(encoding="utf-8")
        companion_text = companion_help.read_text(encoding="utf-8")
        version = re.search(
            r"^\*!\s+version\s+(\S+)\s+(\S+)",
            ado_text,
            flags=re.MULTILINE,
        )
        require(version is not None, f"{command} version is unparseable")
        require(version.group(1) in companion_text,
                f"{command} help omits its version")
        require(version.group(2) in companion_text,
                f"{command} help omits its date")
        require_tokens(
            f"{ado_text}\n{companion_text}", tokens, f"{command} contract"
        )
        require(f"f {command}.ado" in package_manifest,
                f"xhdfe.pkg omits {command}.ado")
        require(f"f {command}.sthlp" in package_manifest,
                f"xhdfe.pkg omits {command}.sthlp")


def validate_r_help() -> None:
    main = (ROOT / "r" / "xhdfe" / "man" / "xhdfe_gelbach.Rd").read_text(
        encoding="utf-8"
    )
    tidy = (ROOT / "r" / "xhdfe" / "man" /
            "xhdfe_gelbach_tidy.Rd").read_text(encoding="utf-8")
    contrast = (ROOT / "r" / "xhdfe" / "man" /
                "xhdfe_gelbach_contrast.Rd").read_text(encoding="utf-8")
    bootstrap = (ROOT / "r" / "xhdfe" / "man" /
                 "xhdfe_gelbach_bootstrap.Rd").read_text(encoding="utf-8")
    etable = (ROOT / "r" / "xhdfe" / "man" /
              "xhdfe_gelbach_etable.Rd").read_text(encoding="utf-8")
    waterfall = (ROOT / "r" / "xhdfe" / "man" /
                  "xhdfe_gelbach_waterfall_data.Rd").read_text(
                      encoding="utf-8"
                  )
    coefplot = (ROOT / "r" / "xhdfe" / "man" /
                "xhdfe_gelbach_coefplot.Rd").read_text(encoding="utf-8")

    require_tokens(main, [
        "x2_groups", "fes", "common_fes", "vce", "cluster", "gamma0", "cov0", "tol",
        "num_threads", "weights", "fweights", "absorbed_targets", "focal",
        "connected", "connectivity_fes", "fe_variance_ratio_min",
        "sample_info", "sample_index", "sample_mask", "sample_hash",
        "sample_hash_algorithm", "sample_index_scope",
        "non-cryptographic",
        "Specification and sample contract", "Inference and covariance",
        "Printing and reporting", "Deliberate limits", "b_base", "b_full",
        "b_full_status", "x1_absorbed", "delta",
        "beta2", "beta2_cov", "auxiliary_loadings", "regularity",
        "regular_inference_valid", "regular_inference_status",
        "regular_inference_all_valid", "regularity_test_alpha",
        "nonregular_not_ruled_out", "family-wise",
        "fe_variance_status", "conditional_only_between_fe_dominant",
        "total_se", "total_cov", "fe_total", "focal_indices",
        "absorbed_target_inference_valid", "absorbing_fe_index",
        "fe_collinear_ss_ratio_tol", "n_mobility_components",
        "largest_mobility_component_n_obs",
        "largest_mobility_component_share",
        "largest_mobility_component_weight_share",
        "fe_split_identified", "fe_split_status",
        "connectivity_fe_indices", "connectivity_fe_names",
        "connectivity_pair_explicit", "connectivity_pair_status",
        "connected_mode", "mobility_component_scope",
        "common_fe_names", "n_common_fes", "common_fes_applied",
        "intercept_inference_available", "intercept_status",
        "not_certified_common_fes", "not_certified_multiway",
        "not_certified_with_common_fes",
        "group-major", "zero-based",
        "random-design", "\\examples", "imposed_zero",
    ], "R main Gelbach help")
    require_tokens(tidy, [
        "include_intercept", "include_total", "include_full", "conf_level",
        "share_tol", "share_t_min", "share_denominator_t",
        "share_interval_status",
        "weak_denominator_delta_method_unreliable",
        "fe_variance_status", "Share contract", "Added rows",
        "share_std_error",
        "share_defined", "movement", "base_fixed", "full_model_residual",
        "regular_inference_valid", "regular_inference_status",
        "confidence_interval_status",
        "never truncated or renormalized", "\\examples",
    ], "R tidy help")
    require_tokens(contrast, [
        "focal", "groups", "conf_level", "named numeric vector",
        "std_error", "joint_covariance_including_conditional_fe",
        "regular_inference_valid", "regular_inference_status",
        "confidence_interval_status",
        "\\examples",
    ], "R contrast help")
    require_tokens(bootstrap, [
        "method", "pairs", "cluster_pairs", "bootstrap_cluster", "reps",
        "seed", "conf_level", "ci_method", "percentile", "basic",
        "min_valid_reps", "store_draws", "require_gpu_used", "share_tol",
        "common_fes", "ledger", "failure counts", "L'Ecuyer-CMRG",
        "PCG64", "Stata",
        "Frequency-weight", "not causal mediation", "\\examples",
    ], "R bootstrap help")
    require_tokens(etable, [
        "panels", "share_base", "share_full", "share_movement",
        "share_explained", "data.frame", "df", "markdown", "md",
        "latex", "tex", "html", "gt", "records",
        "keep", "drop", "exact_match", "labels", "include_other",
        "Other (filtered)", "joint covariance", "share_t_min",
        "bootstrap", "\\examples",
    ], "R etable help")
    require_tokens(waterfall, [
        "focal", "keep", "drop", "exact_match", "labels", "include_other",
        "Other (filtered)", "waterfall_residual", "identity", "\\examples",
    ], "R waterfall-data help")
    require_tokens(coefplot, [
        "annotate_shares", "keep", "drop", "exact_match", "include_other",
        "Other (filtered)", "identity", "main", "title", "dev.off",
        "\\examples",
    ], "R coefplot help")

    namespace = (ROOT / "r" / "xhdfe" / "NAMESPACE").read_text(
        encoding="utf-8"
    )
    for function in (
        "xhdfe_gelbach_bootstrap", "xhdfe_gelbach_etable",
        "xhdfe_gelbach_waterfall_data", "xhdfe_gelbach_coefplot",
    ):
        require(f"export({function})" in namespace,
                f"R NAMESPACE omits {function}")


def validate_examples() -> None:
    for stem in ("gelbach_example", "gelbach_absorbed_target"):
        for suffix in (".py", ".R", ".do"):
            require((ROOT / "examples" / f"{stem}{suffix}").is_file(),
                    f"missing documented example: {stem}{suffix}")


def validate_reporting_source() -> None:
    stata = (ROOT / "stata" / "xhdfegelbachetable.ado").read_text(
        encoding="utf-8"
    )
    r_source = (ROOT / "r" / "xhdfe" / "R" /
                "gelbach_features.R").read_text(encoding="utf-8")
    require(r"Method \\\\" not in stata,
            "Stata LaTeX header still emits a four-backslash terminator")
    require(r"method' \\\\" not in stata,
            "Stata LaTeX body still emits a four-backslash terminator")
    require(r'paste0("\\\\footnotesize "' not in r_source,
            "R LaTeX notes still emit a literal double backslash")
    require(r'paste0("\\footnotesize "' in r_source,
            "R LaTeX notes do not emit the footnotesize control sequence")


def main() -> int:
    validate_python_help()
    validate_stata_help()
    validate_r_help()
    validate_examples()
    validate_reporting_source()
    print(f"Gelbach help contract: PASS ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
