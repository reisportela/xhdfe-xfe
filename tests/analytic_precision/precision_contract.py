"""Frozen mode-specific T01 precision contract.

This module contains only grading formulas.  It does not load xhdfe, mutate
environment variables, or use an observed candidate/reference discrepancy to
set an allowance.
"""
from __future__ import annotations

import math

import numpy as np


CONTRACT_ID = "t01-fast-comparable-v1-20260911"
UNIT_ROUNDOFF = 2.0 ** -53
FAST_BETA_ABS = 1.0e-8
HISTORICAL_V_ATOL = 1.0e-12
HISTORICAL_V_RTOL = 1.0e-7
COMPARABLE_BETA_REL = 1.0e-9
COMPARABLE_V_REL = 1.0e-8
RSS_ABS = 1.0e-8
RSS_REL = 1.0e-10
RESIDUAL_ABS = 1.0e-7
FE_RECOVERY_ABS = 1.0e-6
H2_JOINT_TAU_HYPOTHESIS = 1.0e-8


def gamma(operation_count: int) -> float:
    """Higham gamma_m with u already included exactly once."""
    if operation_count < 0:
        raise ValueError("operation_count must be nonnegative")
    product = float(operation_count) * UNIT_ROUNDOFF
    return math.inf if product >= 1.0 else product / (1.0 - product)


def covariance_operation_count(n: int, p: int, vce: str, cluster_terms: int = 0) -> int:
    """Predeclared first-order operation count for the oracle V graph.

    The count covers weighted moments, the small bread/sandwich products and
    inclusion-exclusion components.  It deliberately does not use candidate
    errors or fitted differences.
    """
    if n <= 0 or p <= 0:
        raise ValueError("positive n and p are required")
    moment_passes = 2 if vce == "unadjusted" else 3
    return moment_passes * n + 6 * p * p * p + 8 * p * p + 16 + max(0, cluster_terms) * p * p


def oracle_covariance_arithmetic_margin(
    absolute_operation_scale,
    operation_count: int,
    gram_condition: float,
):
    """Return a prior first-order covariance rounding envelope.

    ``absolute_operation_scale`` is formed from absolute products in the
    oracle's bread/meat graph, before cancellation.  The Gram condition enters
    through theta=gamma_m*kappa(G).  If theta>=1 the first-order model cannot
    certify a finite margin; callers retain the historical gate and mark the
    arithmetic model unavailable for nonzero discrepancies.
    """
    scale = np.asarray(absolute_operation_scale, dtype=float)
    if scale.ndim != 2 or scale.shape[0] != scale.shape[1]:
        raise ValueError("covariance arithmetic scale must be square")
    if not np.isfinite(scale).all() or np.any(scale < 0.0):
        raise ValueError("covariance arithmetic scale must be finite and nonnegative")
    g = gamma(operation_count)
    condition = max(1.0, float(gram_condition))
    theta = g * condition
    valid = math.isfinite(theta) and theta < 1.0
    if valid:
        multiplier = theta / (1.0 - theta)
        margin = multiplier * scale
        margin += np.spacing(np.maximum(scale, np.finfo(float).tiny))
        margin = np.nextafter(margin, np.inf)
    else:
        margin = None
    return margin, {
        "model": "gamma_m_conditioned_absolute_product_v1",
        "operation_count": int(operation_count),
        "unit_roundoff": UNIT_ROUNDOFF,
        "gamma_m": g,
        "gram_condition": condition,
        "theta": theta,
        "finite_first_order_bound": bool(valid),
    }


def contrast_covariance_margin(contrast, full_margin):
    C = np.asarray(contrast, dtype=float)
    margin = np.asarray(full_margin, dtype=float)
    if C.ndim != 2 or margin.ndim != 2 or margin.shape[0] != margin.shape[1] or C.shape[1] != margin.shape[0]:
        raise ValueError("contrast and covariance margin dimensions differ")
    transformed = np.zeros((C.shape[0], C.shape[0]))
    for j in range(C.shape[0]):
        for k in range(C.shape[0]):
            terms = []
            for a in range(C.shape[1]):
                if C[j, a] == 0.0:
                    continue
                for b in range(C.shape[1]):
                    if C[k, b] != 0.0:
                        terms.append(abs(C[j, a]) * margin[a, b] * abs(C[k, b]))
            transformed[j, k] = math.fsum(terms)
    return np.nextafter(transformed, np.inf)


def mode_and_tolerance(job, metadata):
    precision = job.get("precision", "default")
    if precision == "fast":
        return "fast", 1.0e-8
    if precision == "default":
        return "comparable", 1.0e-8
    if precision == "strict" and metadata.get("slopes"):
        return "comparable", 1.0e-12
    return None, None


def contract_manifest():
    return {
        "contract_id": CONTRACT_ID,
        "fast": {
            "identified_coefficient_absolute": FAST_BETA_ABS,
            "covariance_atol": HISTORICAL_V_ATOL,
            "covariance_rtol": HISTORICAL_V_RTOL,
            "n1_tau": "requested tol",
        },
        "comparable": {
            "inherits_historical_gate": True,
            "identified_coefficient": "1e-9*max(1,abs(reference identified coefficient))",
            "covariance": "1e-8*sqrt(Vref_jj*Vref_kk)+A_jk",
            "n1_tau": "min(requested tol,1e-9)",
        },
        "joint_fit_metric": {
            "formula": "norm(X_tilde*(b-b_ref),W)/norm(y_tilde,W)",
            "h2_tau_hypothesis": H2_JOINT_TAU_HYPOTHESIS,
            "diagnostic_only": True,
        },
        "undefined_t_p_when_b_se_zero": "not a failure",
        "reported_precision_certified_is_not_an_oracle": True,
        "group_individual_near_null_protection_required_by_default": True,
        "required_downstream_performance_gate": "core24 x 8 xhdfe cells",
    }
