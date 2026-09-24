"""Additional frozen Fast/Comparable adjudication for T01.

The historical evaluator remains authoritative for its original verdict.  T01
adds a separate G1 verdict and never promotes a row that failed that evaluator.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from precision_contract import (
    COMPARABLE_BETA_REL,
    COMPARABLE_V_REL,
    CONTRACT_ID,
    FAST_BETA_ABS,
    H2_JOINT_TAU_HYPOTHESIS,
    HISTORICAL_V_ATOL,
    HISTORICAL_V_RTOL,
    contrast_covariance_margin,
    covariance_operation_count,
    mode_and_tolerance,
    oracle_covariance_arithmetic_margin,
)


PASS_STATES = {"PASS", "UNSUPPORTED_EXPECTED"}
VERIFIER_WORDS = re.compile(
    r"precision could not be established|independent check|final inference|certificate",
    re.I,
)


def _failure(verdict, reason, **extra):
    return dict(contract_id=CONTRACT_ID, applicable=True, g1_verdict=verdict,
                passed=False, reason=reason, **extra)


def _weighted_moments(matrix, residuals, weights):
    matrix = np.asarray(matrix, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != residuals.size or weights.size != residuals.size:
        raise ValueError("moment inputs have incompatible dimensions")
    if matrix.shape[1] == 0:
        return dict(max_normalized=0.0, rho=0.0, active_columns=0)
    score = matrix.T @ (weights * residuals)
    residual_w_norm = math.sqrt(max(0.0, float(np.dot(weights, residuals * residuals))))
    column_norm = np.sqrt(np.maximum(0.0, np.sum(weights[:, None] * matrix * matrix, axis=0)))
    active = column_norm > 0.0
    normalized = np.zeros(matrix.shape[1])
    if residual_w_norm > 0.0:
        normalized[active] = np.abs(score[active]) / (column_norm[active] * residual_w_norm)
    elif np.any(score != 0.0):
        normalized[:] = math.inf
    operator_norm = math.sqrt(max(0.0, float(np.sum((weights[:, None] * matrix) ** 2))))
    residual_norm = float(np.linalg.norm(residuals))
    denominator = operator_norm * residual_norm
    rho = float(np.linalg.norm(score)) / denominator if denominator > 0.0 else (0.0 if not np.any(score) else math.inf)
    return dict(max_normalized=float(np.max(normalized)), rho=float(rho),
                active_columns=int(np.count_nonzero(active)))


def _within_score_design(X, D, weights):
    """Independent weighted projection onto the fixture's explicit FE space."""
    if D.shape[1] == 0:
        return X
    sqrt_weights = np.sqrt(weights)[:, None]
    coefficients = np.linalg.lstsq(sqrt_weights * D, sqrt_weights * X, rcond=None)[0]
    return X - D @ coefficients


def _fallback_margin(oracle, reference_V):
    n = int(oracle.get("groups", oracle.get("N", 1)))
    p = int(np.asarray(reference_V).shape[0])
    condition_x = max(1.0, float(oracle.get("condition_X", 1.0)))
    operation_count = covariance_operation_count(n, p, "fallback")
    scale = np.abs(np.asarray(reference_V, dtype=float))
    margin, model = oracle_covariance_arithmetic_margin(
        scale, operation_count, condition_x * condition_x)
    model["scale_source"] = "absolute stored oracle V fallback"
    return margin, model


def _joint_diagnostic(oracle, delta):
    gram = oracle.get("within_gram")
    radius = oracle.get("within_outcome_norm")
    if gram is None or radius is None:
        return dict(available=False, diagnostic_only=True,
                    reason="within Gram/radius absent from legacy fixture")
    gram = np.asarray(gram, dtype=float)
    quadratic = float(np.asarray(delta) @ gram @ np.asarray(delta))
    numerator = math.sqrt(max(0.0, quadratic))
    eta = numerator / float(radius) if float(radius) > 0.0 else (0.0 if numerator == 0.0 else math.inf)
    return dict(available=True, diagnostic_only=True, numerator=numerator,
                within_outcome_norm=float(radius), eta_fit=eta,
                h2_tau_hypothesis=H2_JOINT_TAU_HYPOTHESIS, gate_applied=False)


def assess_t01(job, result, fixture, historical):
    metadata = json.loads((Path(fixture) / "case.json").read_text())
    if job.get("engine") != "xhdfe":
        return dict(contract_id=CONTRACT_ID, applicable=False,
                    g1_verdict="REFERENCE_ONLY", passed=None)
    mode, requested_tol = mode_and_tolerance(job, metadata)
    if mode is None:
        return dict(contract_id=CONTRACT_ID, applicable=False,
                    g1_verdict="OUTSIDE_FAST_COMPARABLE", passed=None)
    base = dict(mode=mode, requested_tolerance=requested_tol,
                historical_verdict=historical.get("verdict"))
    if historical.get("verdict") == "UNSUPPORTED_EXPECTED":
        return dict(contract_id=CONTRACT_ID, applicable=True,
                    g1_verdict="UNSUPPORTED_EXPECTED", passed=True, **base)
    if result.get("timeout"):
        return _failure("COVERAGE_MISSING", "timeout left no adjudicable result", **base)
    if result.get("harness_error") or historical.get("verdict") == "HARNESS_ERROR":
        return _failure("HARNESS_FAILURE", "evaluation or runner failure", **base)
    if result.get("rc", 0) != 0 or result.get("error"):
        error = str(result.get("error", "")) + " " + str(result.get("diagnostic", ""))
        kind = "VERIFIER_FAILURE" if VERIFIER_WORDS.search(error) else "REAL_ESTIMATOR_FAILURE"
        return _failure(kind, "valid requested row returned no estimate", error=error[-2000:], **base)
    if historical.get("verdict") != "PASS":
        return _failure("REAL_ESTIMATOR_FAILURE",
                        "row failed the preserved historical numerical gate", **base)

    fixture = Path(fixture)
    oracle = json.loads((fixture / "oracle.json").read_text())
    C = np.asarray(oracle["contrast"], dtype=float)
    beta = np.asarray(result.get("beta", []), dtype=float)
    beta_reference = np.asarray(oracle["beta"], dtype=float)
    V = np.asarray(result.get("V", []), dtype=float)
    V_reference = np.asarray(oracle["V_contract"], dtype=float)
    if beta.shape != beta_reference.shape or V.shape != V_reference.shape or not np.isfinite(beta).all() or not np.isfinite(V).all():
        return _failure("REAL_ESTIMATOR_FAILURE", "coefficient/covariance shape or finiteness", **base)

    delta = C @ (beta - beta_reference)
    identified_reference = C @ beta_reference
    candidate_V = C @ V @ C.T
    reference_V = C @ V_reference @ C.T
    historical_beta_ok = bool(np.max(np.abs(delta)) <= FAST_BETA_ABS)
    historical_V_ok = bool(np.allclose(candidate_V, reference_V,
                                       atol=HISTORICAL_V_ATOL,
                                       rtol=HISTORICAL_V_RTOL))

    full_margin_raw = oracle.get("V_contract_arithmetic_margin")
    arithmetic_model = oracle.get("V_arithmetic_model")
    if full_margin_raw is None and arithmetic_model is None:
        full_margin, arithmetic_model = _fallback_margin(oracle, V_reference)
    elif full_margin_raw is None:
        full_margin = None
    else:
        full_margin = np.asarray(full_margin_raw, dtype=float)
    contrast_margin = (contrast_covariance_margin(C, full_margin)
                       if full_margin is not None else
                       np.full((C.shape[0], C.shape[0]), math.inf))
    diag = np.diag(reference_V)
    valid_reference_diagonal = bool(np.isfinite(diag).all() and np.all(diag >= 0.0))
    relative_scale = np.sqrt(np.maximum(0.0, diag[:, None] * diag[None, :]))
    comparable_V_limit = COMPARABLE_V_REL * relative_scale + contrast_margin
    covariance_difference = np.abs(candidate_V - reference_V)
    finite_margin = bool(np.isfinite(comparable_V_limit).all())
    exact_covariance_match = bool(np.array_equal(candidate_V, reference_V))
    comparable_V_ok = valid_reference_diagonal and (
        exact_covariance_match or (finite_margin and np.all(covariance_difference <= comparable_V_limit)))
    comparable_beta_limit = COMPARABLE_BETA_REL * np.maximum(1.0, np.abs(identified_reference))
    comparable_beta_ok = bool(np.all(np.abs(delta) <= comparable_beta_limit))

    rows = pd.read_stata(fixture / "explicit.dta")
    wanted = rows.loc[rows.expected_sample == 1].set_index("group")
    ids = np.asarray(result.get("groups", []), dtype=int)
    residuals = np.asarray(result.get("residuals", []), dtype=float)
    if (ids.size != residuals.size or ids.size != len(wanted) or len(set(ids.tolist())) != len(wanted)
            or set(ids.tolist()) != set(wanted.index.tolist()) or not np.isfinite(residuals).all()):
        return _failure("REAL_ESTIMATOR_FAILURE", "final sample/residuals unavailable for N1", **base)
    truth = wanted.loc[ids]
    weights = truth.weight.to_numpy(dtype=float) if metadata.get("weight") else np.ones(ids.size)
    X = truth[["x1", "x2", "x3"]].to_numpy(dtype=float)
    D = np.load(fixture / "design.npz")["D"][ids]
    fe_moments = _weighted_moments(D, residuals, weights)
    raw_regressor_moments = _weighted_moments(X, residuals, weights)
    score_X = _within_score_design(X, D, weights) if metadata.get("fes") else X
    regressor_moments = _weighted_moments(score_X, residuals, weights)
    tau = requested_tol if mode == "fast" else min(requested_tol, 1.0e-9)
    n1_checks = {
        "fe_moments": fe_moments["max_normalized"] <= tau,
        "fe_rho": fe_moments["rho"] <= 8.0 * tau,
        "normal_defect": regressor_moments["max_normalized"] <= tau,
    }
    n1_ok = all(n1_checks.values())
    mode_checks = {
        "preserved_historical_beta": historical_beta_ok,
        "preserved_historical_covariance": historical_V_ok,
        "n1": n1_ok,
    }
    if mode == "comparable":
        mode_checks.update(comparable_beta=comparable_beta_ok,
                           comparable_covariance=comparable_V_ok)
    passed = all(mode_checks.values())
    reason = "all frozen mode gates passed" if passed else "one or more frozen mode gates failed"
    if not finite_margin and not exact_covariance_match and mode == "comparable":
        return _failure("COVERAGE_MISSING",
                        "first-order covariance arithmetic model is non-finite",
                        checks=mode_checks, arithmetic_model=arithmetic_model,
                        contract_status="INCOMPLETE", **base)
    return dict(
        contract_id=CONTRACT_ID,
        applicable=True,
        mode=mode,
        requested_tolerance=requested_tol,
        tau=tau,
        historical_verdict=historical.get("verdict"),
        g1_verdict="PASS" if passed else "REAL_ESTIMATOR_FAILURE",
        passed=passed,
        reason=reason,
        checks=mode_checks,
        metrics={
            "identified_beta_abs": np.abs(delta).tolist(),
            "comparable_beta_limit": comparable_beta_limit.tolist(),
            "covariance_abs_difference": covariance_difference.tolist(),
            "comparable_covariance_limit": comparable_V_limit.tolist(),
            "arithmetic_margin": contrast_margin.tolist(),
            "fe_moments": fe_moments,
            "regressor_moments": regressor_moments,
            "raw_regressor_moments": raw_regressor_moments,
            "normal_score_design": "weighted within X" if metadata.get("fes") else "X",
        },
        arithmetic_model=arithmetic_model,
        joint_fit_diagnostic=_joint_diagnostic(oracle, beta - beta_reference),
        reported_precision_certified_used_as_oracle=False,
        t_p_finiteness_gate=False,
    )


def selftest_t01(oracle, metadata, fixture, historical_assess):
    fixture = Path(fixture)
    truth = pd.read_stata(fixture / "explicit.dta")
    truth = truth.loc[truth.expected_sample == 1]
    good = dict(rc=0, beta=oracle["beta"], V=oracle["V_contract"], N=oracle["N"],
                rss=oracle["rss"], groups=truth.group.astype(int).tolist(),
                residuals=truth.exact_residual.tolist(), converged=1, gpu_used=0,
                singletons=oracle["dropped"])
    job = dict(engine="xhdfe", backend="cpu", case=metadata["name"],
               method="auto", precision="default")
    historical = historical_assess(job, good, fixture)
    assert assess_t01(job, good, fixture, historical)["g1_verdict"] == "PASS"
    bad = dict(good, beta=[oracle["beta"][0] + 0.01, *oracle["beta"][1:]])
    historical_bad = historical_assess(job, bad, fixture)
    assert assess_t01(job, bad, fixture, historical_bad)["g1_verdict"] == "REAL_ESTIMATOR_FAILURE"
