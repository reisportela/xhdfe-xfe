#!/usr/bin/env python3
"""R6A.3 IV-only adversarial certification, oracle, and parity snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


EPS = np.finfo(np.float64).eps
RANK_FACTOR = 16.0


def load_module(directory: Path):
    sys.path.insert(0, str(directory.resolve()))
    return importlib.import_module("py_hdfe_v11")


def digest(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def json_safe(value):
    """Keep legacy non-finite outputs inspectable in strict JSON evidence."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def max_differences(actual, expected) -> tuple[float, float]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        return math.inf, math.inf
    difference = np.abs(actual - expected)
    absolute = float(np.max(difference, initial=0.0))
    relative = float(
        np.max(difference / np.maximum(np.abs(expected), 1.0), initial=0.0)
    )
    return absolute, relative


def rank_tolerance(rows: int, cols: int) -> float:
    return RANK_FACTOR * EPS * float(max(1, rows, cols))


def normalized_singular_values(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("rank input must be two-dimensional")
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    scales = np.linalg.norm(matrix, axis=0)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        return np.empty(0, dtype=np.float64), scales
    singular = np.linalg.svd(matrix / scales, compute_uv=False)
    return singular, scales


def relative_svd_rank(matrix: np.ndarray, tolerance: float) -> tuple[int, list[float]]:
    singular, _ = normalized_singular_values(matrix)
    if singular.size == 0 or not np.all(np.isfinite(singular)):
        return 0, singular.tolist()
    cutoff = tolerance * float(singular[0])
    return int(np.count_nonzero(singular > cutoff)), singular.tolist()


def absolute_svd_rank(matrix: np.ndarray, tolerance: float) -> tuple[int, list[float]]:
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if singular.size == 0 or not np.all(np.isfinite(singular)):
        return 0, singular.tolist()
    return int(np.count_nonzero(singular > tolerance)), singular.tolist()


def dummy_matrix(fes: list[np.ndarray]) -> np.ndarray:
    blocks = []
    for fe in fes:
        _, codes = np.unique(np.asarray(fe), return_inverse=True)
        block = np.zeros((len(fe), int(codes.max()) + 1), dtype=np.float64)
        block[np.arange(len(fe)), codes] = 1.0
        blocks.append(block)
    if not blocks:
        return np.empty((0, 0), dtype=np.float64)
    return np.column_stack(blocks)


def residualize(matrix: np.ndarray, fes: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    was_vector = matrix.ndim == 1
    if was_vector:
        matrix = matrix[:, None]
    if not fes:
        return matrix[:, 0] if was_vector else matrix.copy()
    dummies = dummy_matrix(fes)
    root_w = np.sqrt(weights)[:, None]
    coef = np.linalg.lstsq(root_w * dummies, root_w * matrix, rcond=None)[0]
    result = matrix - dummies @ coef
    return result[:, 0] if was_vector else result


def orthogonalize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def oracle_diagnostics(case: dict) -> dict:
    y = np.asarray(case["y"], dtype=np.float64)
    x = np.asarray(case["x"], dtype=np.float64)
    excluded = np.asarray(case["z"], dtype=np.float64)
    endogenous = list(case["endogenous"])
    weights = case.get("weights")
    if weights is None:
        weights = np.ones(y.size, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    fes = case.get("fes", [])
    finite = (
        np.all(np.isfinite(y))
        and np.all(np.isfinite(x))
        and np.all(np.isfinite(excluded))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0)
    )
    base = {
        "finite": bool(finite),
        "observations": int(y.size),
        "excluded_columns": int(excluded.shape[1]),
        "endogenous_columns": int(len(endogenous)),
        "rank_factor": RANK_FACTOR,
    }
    if not finite:
        return base | {"identified": False, "reason": "non-finite input"}

    yr = residualize(y, fes, weights)
    xr = residualize(x, fes, weights)
    zr = residualize(excluded, fes, weights)
    exogenous = [j for j in range(xr.shape[1]) if j not in endogenous]
    instruments = np.column_stack((xr[:, exogenous], zr))
    root_w = np.sqrt(weights)[:, None]
    zw = root_w * instruments
    qw = root_w * xr[:, endogenous]
    exog_w = root_w * xr[:, exogenous]
    excluded_w = root_w * zr

    full_tol = rank_tolerance(zw.shape[0], zw.shape[1])
    full_rank, full_singular = relative_svd_rank(zw, full_tol)
    if exog_w.shape[1]:
        excluded_res = excluded_w - exog_w @ np.linalg.lstsq(
            exog_w, excluded_w, rcond=None
        )[0]
        endogenous_res = qw - exog_w @ np.linalg.lstsq(
            exog_w, qw, rcond=None
        )[0]
    else:
        excluded_res = excluded_w.copy()
        endogenous_res = qw.copy()
    excluded_tol = rank_tolerance(excluded_res.shape[0], excluded_res.shape[1])
    excluded_rank, excluded_singular = relative_svd_rank(
        excluded_res, excluded_tol
    )

    excluded_scales = np.linalg.norm(excluded_res, axis=0)
    endogenous_scales = np.linalg.norm(endogenous_res, axis=0)
    relation = np.empty((excluded_res.shape[1], endogenous_res.shape[1]))
    relation.fill(np.nan)
    if (
        np.all(np.isfinite(excluded_scales))
        and np.all(excluded_scales > 0.0)
        and np.all(np.isfinite(endogenous_scales))
        and np.all(endogenous_scales > 0.0)
    ):
        relation = (
            (excluded_res / excluded_scales).T
            @ (endogenous_res / endogenous_scales)
        )
    relation_tol = rank_tolerance(relation.shape[0], relation.shape[1])
    if np.all(np.isfinite(relation)):
        relation_rank, relation_singular = absolute_svd_rank(
            relation, relation_tol
        )
    else:
        relation_rank, relation_singular = 0, []

    identified = (
        excluded.shape[1] >= len(endogenous)
        and full_rank == instruments.shape[1]
        and excluded_rank == excluded.shape[1]
        and relation_rank >= len(endogenous)
    )
    result = base | {
        "identified": bool(identified),
        "instrument_rank": full_rank,
        "instrument_columns": int(instruments.shape[1]),
        "instrument_threshold": full_tol,
        "instrument_singular": full_singular,
        "excluded_fwl_rank": excluded_rank,
        "excluded_fwl_threshold": excluded_tol,
        "excluded_fwl_singular": excluded_singular,
        "first_stage_rank": relation_rank,
        "first_stage_threshold": relation_tol,
        "first_stage_singular": relation_singular,
    }
    if not identified or case.get("oracle", True) is False:
        return result

    gamma = np.linalg.lstsq(zw, qw, rcond=None)[0]
    projected = instruments @ gamma
    second = xr.copy()
    second[:, endogenous] = projected
    second_w = root_w * second
    beta = np.linalg.lstsq(second_w, np.sqrt(weights) * yr, rcond=None)[0]
    residual = yr - xr @ beta
    normal_residual = zw.T @ qw - (zw.T @ zw) @ gamma

    gram = second_w.T @ second_w
    bread = np.linalg.pinv(gram, rcond=EPS * max(gram.shape), hermitian=True)
    if fes:
        dummies = dummy_matrix(fes)
        dummy_rank = int(np.linalg.matrix_rank(root_w * dummies))
    else:
        dummy_rank = 0
    df_resid = max(0.0, float(y.size - dummy_rank - second.shape[1]))
    rss = float(np.dot(weights, residual * residual))
    sigma2 = rss / df_resid if df_resid > 0.0 else 0.0
    covariance = sigma2 * bread
    stderr = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    return result | {
        "coef": beta.tolist(),
        "stderr": stderr.tolist(),
        "covariance": covariance.tolist(),
        "projection_sha256": digest(projected),
        "projection_norm": float(np.linalg.norm(projected)),
        "projection_normal_residual_norm": float(np.linalg.norm(normal_residual)),
        "residual_sha256": digest(residual),
        "residual_norm": float(np.linalg.norm(residual)),
        "df_resid": df_resid,
    }


def expected_valid(case: dict, oracle: dict) -> bool | None:
    expected = case.get("expected", "invalid")
    if expected == "ambiguous":
        return None
    if expected == "oracle":
        return bool(oracle["identified"])
    return expected == "valid"


def fit_case(mod, case: dict, threads: int, backend_requested: str) -> dict:
    oracle = oracle_diagnostics(case)
    reg = mod.HdfeRegressor(
        se_type="unadjusted",
        fit_intercept=case.get("fit_intercept", False),
        drop_singletons=False,
        tol=1e-11,
        max_iter=100000,
        num_threads=threads,
        tolerance_mode="reghdfe-comparable",
    )
    reg.fit(
        case["y"],
        np.asfortranarray(case["x"]),
        fes=case.get("fes", []),
        weights=case.get("weights"),
        instruments=np.asfortranarray(case["z"]),
        endogenous_idx=case["endogenous"],
    )
    if not reg.converged_:
        raise AssertionError("estimator did not converge")
    coefficients = np.asarray(reg.coef_, dtype=np.float64)
    stderr = np.asarray(reg.stderr_, dtype=np.float64)
    covariance = np.asarray(reg.covariance_, dtype=np.float64)
    residual = np.asarray(reg.residuals_, dtype=np.float64)
    result = {
        "coef": coefficients.tolist(),
        "stderr": stderr.tolist(),
        "covariance": covariance.tolist(),
        "residual_norm": float(np.linalg.norm(residual)),
        "coef_sha256": digest(coefficients),
        "stderr_sha256": digest(stderr),
        "covariance_sha256": digest(covariance),
        "residual_sha256": digest(residual),
        "iterations": int(reg.num_iterations_),
        "converged": bool(reg.converged_),
        "method": str(reg.absorption_method_used),
        "threads_used": int(reg.threads_used_),
        "backend_requested": backend_requested,
        "gpu_used": bool(reg.gpu_used_),
        "gpu_status_code": int(reg.gpu_status_code_),
        "oracle": oracle,
    }
    if oracle.get("coef") is not None and coefficients.shape == np.asarray(
        oracle["coef"]
    ).shape:
        coef_abs, coef_rel = max_differences(coefficients, oracle["coef"])
        se_abs, se_rel = max_differences(stderr, oracle["stderr"])
        cov_abs, cov_rel = max_differences(covariance, oracle["covariance"])
        result |= {
            "oracle_coef_max_abs": coef_abs,
            "oracle_coef_max_rel": coef_rel,
            "oracle_stderr_max_abs": se_abs,
            "oracle_stderr_max_rel": se_rel,
            "oracle_covariance_max_abs": cov_abs,
            "oracle_covariance_max_rel": cov_rel,
        }
    return result


def outcome(mod, case: dict, threads: int, backend_requested: str) -> dict:
    oracle = oracle_diagnostics(case)
    expected = expected_valid(case, oracle)
    try:
        value = fit_case(mod, case, threads, backend_requested)
        return {
            "status": "ok",
            "expected_valid": expected,
            "cross_thread_bitwise": case.get("cross_thread_bitwise", True),
            "value": value,
        }
    except BaseException as exc:
        return {
            "status": "error",
            "expected_valid": expected,
            "cross_thread_bitwise": case.get("cross_thread_bitwise", True),
            "type": type(exc).__name__,
            "message": str(exc),
            "oracle": oracle,
        }


def clone_case(case: dict, **updates) -> dict:
    copied = dict(case)
    copied["y"] = np.asarray(case["y"]).copy()
    copied["x"] = np.asarray(case["x"]).copy()
    copied["z"] = np.asarray(case["z"]).copy()
    if case.get("weights") is not None:
        copied["weights"] = np.asarray(case["weights"]).copy()
    if case.get("fes"):
        copied["fes"] = [np.asarray(fe).copy() for fe in case["fes"]]
    copied.update(updates)
    return copied


def permute_rows(case: dict, permutation: np.ndarray) -> dict:
    updates = {
        "y": np.asarray(case["y"])[permutation],
        "x": np.asarray(case["x"])[permutation],
        "z": np.asarray(case["z"])[permutation],
    }
    if case.get("weights") is not None:
        updates["weights"] = np.asarray(case["weights"])[permutation]
    if case.get("fes"):
        updates["fes"] = [np.asarray(fe)[permutation] for fe in case["fes"]]
    return clone_case(case, **updates)


def make_cases() -> dict[str, dict]:
    rng = np.random.default_rng(20260714)
    n = 1600
    row = np.arange(n)
    exog = rng.normal(size=n)
    control = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    z3 = rng.normal(size=n)
    u1 = rng.normal(scale=0.45, size=n)
    u2 = rng.normal(scale=0.40, size=n)
    endog1 = 0.85 * z1 - 0.30 * z2 + 0.20 * exog + u1
    endog2 = 0.45 * z2 + 0.60 * z3 - 0.15 * control + u2
    y1 = 0.55 * exog + 1.25 * endog1 + rng.normal(scale=0.35, size=n)
    y2 = (
        0.45 * exog
        - 0.20 * control
        + 1.10 * endog1
        - 0.75 * endog2
        + rng.normal(scale=0.30, size=n)
    )
    weights = 0.25 + 2.0 * rng.random(n)
    integer_weights = (1 + row % 4).astype(np.float64)
    fe1 = (row % 43).astype(np.int64)
    fe2 = ((row * 7) % 29).astype(np.int64)
    alpha = rng.normal(scale=0.4, size=44)
    psi = rng.normal(scale=0.3, size=30)
    y_fe = y1 + alpha[fe1] + psi[fe2]

    cases: dict[str, dict] = {
        "exact": dict(
            y=y1,
            x=np.column_stack((exog, endog1)),
            z=z1[:, None],
            endogenous=[1],
            expected="valid",
        ),
        "overidentified": dict(
            y=y1,
            x=np.column_stack((exog, endog1)),
            z=np.column_stack((z1, z2)),
            endogenous=[1],
            expected="valid",
        ),
        "weighted": dict(
            y=y1,
            x=np.column_stack((exog, endog1)),
            z=np.column_stack((z1, z2)),
            endogenous=[1],
            weights=weights,
            expected="valid",
        ),
        "integer_weighted": dict(
            y=y1,
            x=np.column_stack((exog, endog1)),
            z=np.column_stack((z1, z2)),
            endogenous=[1],
            weights=integer_weights,
            expected="valid",
        ),
        "fe_controls": dict(
            y=y_fe,
            x=np.column_stack((exog, control, endog1)),
            z=np.column_stack((z1, z2)),
            endogenous=[2],
            weights=weights,
            fes=[fe1, fe2],
            expected="valid",
        ),
        "absorbed_intercept": dict(
            y=y_fe,
            x=np.column_stack((exog, endog1)),
            z=np.column_stack((z1, z2)),
            endogenous=[1],
            fes=[fe1, fe2],
            fit_intercept=True,
            expected="valid",
            oracle=False,
        ),
        "multiple": dict(
            y=y2,
            x=np.column_stack((exog, control, endog1, endog2)),
            z=np.column_stack((z1, z2, z3)),
            endogenous=[2, 3],
            expected="valid",
        ),
        "multiple_exact": dict(
            y=y2,
            x=np.column_stack((exog, control, endog1, endog2)),
            z=np.column_stack((z1, z3)),
            endogenous=[2, 3],
            expected="valid",
        ),
    }
    weak_endog = 1e-8 * z1 + rng.normal(size=n)
    cases["weak_full_rank"] = dict(
        y=0.4 * exog + 0.8 * weak_endog + rng.normal(size=n),
        x=np.column_stack((exog, weak_endog)),
        z=z1[:, None],
        endogenous=[1],
        expected="valid",
    )
    near_z = np.column_stack((z1, z1 + 1e-5 * z2))
    cases["near_full_rank"] = dict(
        y=y2,
        x=np.column_stack((exog, endog1, endog2)),
        z=near_z,
        endogenous=[1, 2],
        expected="valid",
    )
    cases |= {
        "zero": dict(
            y=y1,
            x=np.column_stack((exog, endog1)),
            z=np.zeros((n, 1)),
            endogenous=[1],
            expected="invalid",
        ),
        "duplicate": dict(
            y=y1,
            x=np.column_stack((exog, endog1)),
            z=np.column_stack((z1, z1)),
            endogenous=[1],
            expected="invalid",
        ),
        "collinear": dict(
            y=y1,
            x=np.column_stack((exog, endog1)),
            z=np.column_stack((z1, z2, z1 + z2)),
            endogenous=[1],
            expected="invalid",
        ),
        "collinear_signed_scaled": dict(
            y=y1,
            x=np.column_stack((exog, endog1)),
            z=np.column_stack((1e12 * z1, -1e-6 * z2, 2e12 * z1 + 3e-6 * z2)),
            endogenous=[1],
            expected="invalid",
        ),
        "underidentified": dict(
            y=y2,
            x=np.column_stack((exog, endog1, endog2)),
            z=z1[:, None],
            endogenous=[1, 2],
            expected="invalid",
        ),
        "residual_rank": dict(
            y=y_fe,
            x=np.column_stack((exog, endog1)),
            z=fe1.astype(np.float64)[:, None],
            endogenous=[1],
            fes=[fe1],
            expected="invalid",
        ),
        "fwl_duplicate": dict(
            y=y_fe,
            x=np.column_stack((exog, endog1)),
            z=np.column_stack((z1, z1 + 7.0 * alpha[fe1])),
            endogenous=[1],
            fes=[fe1],
            expected="invalid",
        ),
        "near_deficient": dict(
            y=y1,
            x=np.column_stack((exog, endog1)),
            z=np.column_stack((z1, z1 + 1e-14 * z2)),
            endogenous=[1],
            expected="invalid",
        ),
    }
    controls = np.column_stack((exog, z1))
    unrelated = rng.normal(size=n)
    orthogonal = orthogonalize(unrelated, controls)
    cases["first_stage_rank"] = dict(
        y=y1,
        x=np.column_stack((exog, orthogonal)),
        z=z1[:, None],
        endogenous=[1],
        expected="invalid",
    )
    design = np.column_stack((exog, z1, z2, z3))
    orthogonal_pair = orthogonalize(rng.normal(size=(n, 2)), design)
    rank1_e1 = 0.7 * z1 + orthogonal_pair[:, 0]
    rank1_e2 = -0.4 * z1 + orthogonal_pair[:, 1]
    cases["first_stage_multi_rank1"] = dict(
        y=0.3 * exog + rank1_e1 - 0.5 * rank1_e2 + rng.normal(size=n),
        x=np.column_stack((exog, rank1_e1, rank1_e2)),
        z=np.column_stack((z1, z2, z3)),
        endogenous=[1, 2],
        expected="invalid",
    )
    weak_pair = orthogonalize(rng.normal(size=(n, 2)), design)
    weak_e1 = 1e-8 * z1 + weak_pair[:, 0]
    weak_e2 = -2e-8 * z2 + weak_pair[:, 1]
    cases["weak_multiple_full_rank"] = dict(
        y=0.2 * exog + weak_e1 - 0.3 * weak_e2 + rng.normal(size=n),
        x=np.column_stack((exog, weak_e1, weak_e2)),
        z=np.column_stack((z1, z2, z3)),
        endogenous=[1, 2],
        expected="valid",
    )
    bad_nan = z1.copy()
    bad_nan[17] = np.nan
    bad_inf = z1.copy()
    bad_inf[31] = np.inf
    cases["nonfinite_nan"] = dict(
        y=y1,
        x=np.column_stack((exog, endog1)),
        z=bad_nan[:, None],
        endogenous=[1],
        expected="invalid",
    )
    cases["nonfinite_inf"] = dict(
        y=y1,
        x=np.column_stack((exog, endog1)),
        z=bad_inf[:, None],
        endogenous=[1],
        expected="invalid",
    )

    cases["perm_overidentified"] = clone_case(
        cases["overidentified"], z=np.column_stack((z2, z1))
    )
    cases["perm_multiple"] = clone_case(
        cases["multiple"], z=np.column_stack((z3, z1, z2))
    )
    for factor in (1e-12, 1e-6, 1.0, 1e6, 1e12):
        label = f"rescale_uniform_{factor:.0e}".replace("+", "")
        cases[label] = clone_case(
            cases["overidentified"],
            z=np.asarray(cases["overidentified"]["z"]) * factor,
        )
    cases["mixed_amplitudes"] = clone_case(
        cases["overidentified"], z=np.column_stack((1e-12 * z1, 1e12 * z2))
    )

    for delta in (1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-8):
        label = f"near_grid_{delta:.0e}".replace("+", "")
        if delta <= 1e-12:
            expectation = "invalid"
        elif delta >= 1e-10:
            expectation = "valid"
        else:
            # At this point the normalized minimum singular value and the
            # pivoted-QR boundary straddle the same O(eps*n) threshold.  The
            # transition is recorded, but neither side is mislabeled as an
            # algebraically definitive valid/invalid model.
            expectation = "ambiguous"
        near_case = dict(
            y=y2,
            x=np.column_stack((exog, endog1, endog2)),
            z=np.column_stack((z1, z1 + delta * z2)),
            endogenous=[1, 2],
            expected=expectation,
            invariance_group=label,
        )
        cases[label] = near_case
        cases[label + "_perm"] = clone_case(
            near_case,
            z=np.asarray(near_case["z"])[:, ::-1],
        )
        cases[label + "_scaled"] = clone_case(
            near_case,
            z=np.asarray(near_case["z"]) * np.array([1e-6, 1e6]),
        )

    permutation = rng.permutation(n)
    cases["row_permuted_valid"] = permute_rows(cases["multiple"], permutation)
    cases["row_permuted_invalid"] = permute_rows(cases["collinear"], permutation)

    large_rng = np.random.default_rng(20260715)
    large_n = 200_000
    large_x = large_rng.normal(size=large_n)
    large_z = large_rng.normal(size=large_n)
    large_u = large_rng.normal(size=large_n)
    large_design = np.column_stack((large_x, large_z))
    large_u = orthogonalize(large_u, large_design)
    large_e = 0.2 * large_x + 1e-10 * large_z + large_u
    large_y = 0.4 * large_x + 0.8 * large_e + large_rng.normal(size=large_n)
    cases["large_sample_weak_full_rank"] = dict(
        y=large_y,
        x=np.column_stack((large_x, large_e)),
        z=large_z[:, None],
        endogenous=[1],
        expected="valid",
    )

    wide_rng = np.random.default_rng(20260716)
    wide_n = 40_000
    wide_exog = wide_rng.normal(size=(wide_n, 3))
    wide_z = wide_rng.normal(size=(wide_n, 24))
    wide_u = wide_rng.normal(scale=0.5, size=(wide_n, 4))
    wide_endog = wide_z[:, :4] @ np.diag([0.8, -0.6, 0.5, 0.7]) + wide_u
    wide_y = (
        wide_exog @ np.array([0.3, -0.2, 0.1])
        + wide_endog @ np.array([1.0, -0.7, 0.4, 0.9])
        + wide_rng.normal(scale=0.4, size=wide_n)
    )
    cases["wide_large"] = dict(
        y=wide_y,
        x=np.column_stack((wide_exog, wide_endog)),
        z=wide_z,
        endogenous=[3, 4, 5, 6],
        expected="valid",
        # Large Eigen GEMMs use a thread-dependent reduction tree even on
        # origin/main. Candidate/main must be bit-identical at a fixed thread
        # count, while cross-thread comparison uses the inherited FP envelope.
        cross_thread_bitwise=False,
    )
    return cases


CORE_CASES = {
    "exact",
    "overidentified",
    "weighted",
    "fe_controls",
    "absorbed_intercept",
    "multiple",
    "weak_full_rank",
    "near_full_rank",
    "zero",
    "duplicate",
    "collinear",
    "underidentified",
    "residual_rank",
    "near_deficient",
    "first_stage_rank",
    "nonfinite_nan",
}


def compare_valid_results(results: dict, reference: dict, cross_thread: bool) -> dict:
    keys = (
        "coef_sha256",
        "stderr_sha256",
        "covariance_sha256",
        "residual_sha256",
        "iterations",
        "method",
        "converged",
        "gpu_used",
        "gpu_status_code",
    )
    checked = []
    mismatches = {}
    for name, current in results.items():
        if current.get("expected_valid") is not True:
            continue
        prior = reference.get(name)
        checked.append(name)
        if current.get("status") != "ok" or not prior or prior.get("status") != "ok":
            mismatches[name] = {"current": current.get("status"), "reference": None if not prior else prior.get("status")}
            continue
        if cross_thread and current.get("cross_thread_bitwise") is False:
            differences = [
                max_differences(current["value"][key], prior["value"][key])[0]
                for key in ("coef", "stderr", "covariance")
            ]
            differences.append(
                abs(current["value"]["residual_norm"] - prior["value"]["residual_norm"])
            )
            metadata = [
                key
                for key in ("iterations", "method", "converged", "gpu_used", "gpu_status_code")
                if current["value"].get(key) != prior["value"].get(key)
            ]
            if max(differences) > 2e-11 or metadata:
                mismatches[name] = {
                    "max_abs": max(differences),
                    "metadata": metadata,
                }
            continue
        changed = [
            key
            for key in keys
            if current["value"].get(key) != prior["value"].get(key)
        ]
        if changed:
            mismatches[name] = changed
    return {
        "checked_valid_cases": checked,
        "bit_identical": not mismatches,
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", required=True, type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--expect-preflight", action="store_true")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--require-bitwise-valid", action="store_true")
    parser.add_argument("--suite", choices=("core", "all"), default="all")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--backend-requested",
        default=os.environ.get("XHDFE_GPU_BACKEND", "cpu"),
    )
    args = parser.parse_args()
    mod = load_module(args.module_dir)
    cases = make_cases()
    if args.suite == "core":
        cases = {name: case for name, case in cases.items() if name in CORE_CASES}
    results = {
        name: outcome(mod, case, args.threads, args.backend_requested)
        for name, case in cases.items()
    }

    invariance = {}
    for name, case in cases.items():
        group = case.get("invariance_group")
        if group is not None:
            invariance.setdefault(group, {})[name] = results[name]["status"]
    invariance_failures = {
        group: statuses
        for group, statuses in invariance.items()
        if len(set(statuses.values())) != 1
    }

    failures = []
    if args.expect_preflight:
        for name, result in results.items():
            if result["expected_valid"] is None:
                continue
            if result["expected_valid"]:
                if result["status"] != "ok":
                    failures.append(f"{name} should pass: {result}")
            else:
                if result["status"] != "error":
                    failures.append(f"{name} should fail preflight")
                elif result.get("type") != "RuntimeError":
                    failures.append(f"{name} raised {result.get('type')}, not RuntimeError")
        if invariance_failures:
            failures.append(
                "rank classification changed under permutation/rescaling: "
                + json.dumps(invariance_failures, sort_keys=True)
            )
        if failures:
            raise AssertionError("\n".join(failures))

    comparison = None
    if args.compare:
        reference_payload = json.loads(args.compare.read_text(encoding="utf-8"))
        reference = reference_payload["results"]
        comparison = compare_valid_results(
            results, reference, reference_payload.get("threads") != args.threads
        )
        if args.require_bitwise_valid and not comparison["bit_identical"]:
            raise AssertionError(
                "valid IV snapshots differ: " + json.dumps(comparison["mismatches"], sort_keys=True)
            )

    payload = {
        "schema": "xhdfe-r6a3-iv-certification-v2",
        "module": str(Path(mod.__file__).resolve()),
        "numpy": np.__version__,
        "threads": args.threads,
        "backend_requested": args.backend_requested,
        "suite": args.suite,
        "counts": {
            "total": len(results),
            "expected_valid": sum(r["expected_valid"] is True for r in results.values()),
            "expected_invalid": sum(r["expected_valid"] is False for r in results.values()),
            "threshold_ambiguous": sum(r["expected_valid"] is None for r in results.values()),
            "observed_ok": sum(r["status"] == "ok" for r in results.values()),
            "observed_error": sum(r["status"] == "error" for r in results.values()),
        },
        "rank_invariance": {
            "groups": invariance,
            "failures": invariance_failures,
        },
        "comparison": comparison,
        "results": results,
    }
    rendered = json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
