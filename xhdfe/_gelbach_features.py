"""Resampling and reporting helpers for :mod:`xhdfe.gelbach`.

This module deliberately sits above the shared C++ estimator.  Every bootstrap
replication calls the same public decomposition used for the point estimate;
no approximate or reporting-only estimator is introduced here.
"""

from __future__ import annotations

from collections import Counter
import html
import math
import re
from statistics import NormalDist
import warnings

import numpy as np


_PANEL_ALIASES = {
    "levels": "levels",
    "share_base": "share_base",
    "share_full": "share_base",
    "share_movement": "share_movement",
    "share_explained": "share_movement",
}


def _positive_int(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _seed_value(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)):
        raise TypeError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0 or seed >= 2 ** 64:
        raise ValueError("seed must lie in [0, 2**64)")
    return seed


def _row_vector(values, n, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != n:
        raise ValueError(f"{name} must be one-dimensional with length n")
    return array


def _has_missing_ids(values):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number):
        return not np.all(np.isfinite(values))
    for value in values.tolist():
        if value is None:
            return True
        if type(value).__name__ in {"NAType", "NaTType"}:
            return True
        try:
            if bool(value != value):
                return True
        except Exception:
            pass
    return False


def _resample_mapping(mapping, indices):
    return {
        name: np.asarray(values)[indices]
        for name, values in dict(mapping or {}).items()
    }


def _point_arrays(result):
    names = list(result["names"])
    delta = np.column_stack([
        np.asarray(result["delta"][name]["coef"], dtype=float)
        for name in names
    ])
    return {
        "delta": delta,
        "total": np.asarray(result["total"]["coef"], dtype=float),
        "b_base": np.asarray(result["b_base"], dtype=float),
        "b_full": np.asarray(result["b_full"], dtype=float),
    }


def _finite_quantile_interval(draws, point, conf_level, method):
    draws = np.asarray(draws, dtype=float)
    point = np.asarray(point, dtype=float)
    if draws.ndim < 2:
        raise ValueError("bootstrap draws must have a replication dimension")
    if draws.shape[1:] != point.shape:
        raise ValueError("bootstrap draw and point-estimate shapes do not match")
    alpha = 1.0 - float(conf_level)
    low = np.full(point.shape, np.nan, dtype=float)
    high = np.full(point.shape, np.nan, dtype=float)
    std_error = np.full(point.shape, np.nan, dtype=float)
    valid_counts = np.zeros(point.shape, dtype=np.int64)
    flat_draws = draws.reshape(draws.shape[0], -1)
    flat_point = point.reshape(-1)
    flat_low = low.reshape(-1)
    flat_high = high.reshape(-1)
    flat_se = std_error.reshape(-1)
    flat_n = valid_counts.reshape(-1)
    for column in range(flat_draws.shape[1]):
        values = flat_draws[:, column]
        values = values[np.isfinite(values)]
        flat_n[column] = values.size
        if values.size == 0 or not np.isfinite(flat_point[column]):
            continue
        q_low, q_high = np.quantile(
            values, [alpha / 2.0, 1.0 - alpha / 2.0],
            method="linear",
        )
        if method == "percentile":
            flat_low[column], flat_high[column] = q_low, q_high
        else:
            flat_low[column] = 2.0 * flat_point[column] - q_high
            flat_high[column] = 2.0 * flat_point[column] - q_low
        if values.size > 1:
            flat_se[column] = np.std(values, ddof=1)
    return {
        "low": low,
        "high": high,
        "std_error": std_error,
        "valid_counts": valid_counts,
    }


def _bootstrap_share_draws(draws, share_tol):
    delta = draws["delta"]
    total = draws["total"]
    b_base = draws["b_base"]
    reps, k1, groups = delta.shape
    p = b_base.shape[1]
    share_base = np.full((reps, k1, groups), np.nan, dtype=float)
    share_movement = np.full((reps, k1, groups), np.nan, dtype=float)
    full_share_base = np.full((reps, p), np.nan, dtype=float)
    total_share_base = np.full((reps, p), np.nan, dtype=float)
    for replication in range(reps):
        for row in range(p):
            denominator = b_base[replication, row]
            if np.isfinite(denominator) and abs(denominator) > share_tol:
                share_base[replication, row, :] = (
                    delta[replication, row, :] / denominator
                )
                full_share_base[replication, row] = (
                    draws["b_full"][replication, row] / denominator
                )
                total_share_base[replication, row] = (
                    total[replication, row] / denominator
                )
        for row in range(k1):
            denominator = total[replication, row]
            if np.isfinite(denominator) and abs(denominator) > share_tol:
                share_movement[replication, row, :] = (
                    delta[replication, row, :] / denominator
                )
    return {
        "share_base": share_base,
        "share_movement": share_movement,
        "full_share_base": full_share_base,
        "total_share_base": total_share_base,
    }


def bootstrap(
        decompose,
        y,
        x1,
        x2_groups=None,
        fes=None,
        *,
        method="pairs",
        bootstrap_cluster=None,
        reps=999,
        seed=0,
        conf_level=0.95,
        ci_method="percentile",
        min_valid_reps=None,
        store_draws=True,
        require_gpu_used=False,
        share_tol=1e-12,
        **decompose_kwargs,
):
    """Fit and fully re-estimate a Gelbach pairs bootstrap.

    ``method='pairs'`` samples observations. ``method='cluster_pairs'``
    samples the explicitly supplied ``bootstrap_cluster`` blocks.  The VCE
    requested for the point estimate is retained there; replications use
    unadjusted VCE because only their re-estimated point functionals enter the
    empirical bootstrap distribution.
    """
    if not isinstance(method, str):
        raise TypeError("method must be 'pairs' or 'cluster_pairs'")
    method = method.strip().lower()
    if method not in ("pairs", "cluster_pairs"):
        raise ValueError("method must be 'pairs' or 'cluster_pairs'")
    reps = _positive_int(reps, "reps", minimum=2)
    seed = _seed_value(seed)
    if not np.isfinite(conf_level) or not 0.0 < conf_level < 1.0:
        raise ValueError("conf_level must lie strictly between zero and one")
    if not isinstance(ci_method, str):
        raise TypeError("ci_method must be 'percentile' or 'basic'")
    ci_method = ci_method.strip().lower()
    if ci_method not in ("percentile", "basic"):
        raise ValueError("ci_method must be 'percentile' or 'basic'")
    if not np.isfinite(share_tol) or share_tol < 0:
        raise ValueError("share_tol must be finite and nonnegative")
    if not isinstance(store_draws, (bool, np.bool_)):
        raise TypeError("store_draws must be True or False")
    if not isinstance(require_gpu_used, (bool, np.bool_)):
        raise TypeError("require_gpu_used must be True or False")
    if require_gpu_used and not bool(decompose_kwargs.get("gpu", False)):
        raise ValueError("require_gpu_used=True requires gpu=True")
    if min_valid_reps is None:
        min_valid_reps = int(math.ceil(0.9 * reps))
    else:
        min_valid_reps = _positive_int(
            min_valid_reps, "min_valid_reps", minimum=2
        )
    if min_valid_reps > reps:
        raise ValueError("min_valid_reps may not exceed reps")

    y_array = np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError("y must be one-dimensional")
    n = y_array.shape[0]
    x1_array = np.asarray(x1)
    if x1_array.ndim == 1:
        x1_array = x1_array[:, None]
    if x1_array.ndim != 2 or x1_array.shape[0] != n:
        raise ValueError("x1 must have n rows")

    x2_groups = dict(x2_groups or {})
    fes = dict(fes or {})
    common_fes = dict(decompose_kwargs.get("common_fes") or {})
    weights = decompose_kwargs.get("weights")
    fweights = bool(decompose_kwargs.get("fweights", False))
    if fweights:
        raise NotImplementedError(
            "frequency-weight pairs bootstrap is not yet exposed because "
            "record-level and expanded-sample resampling are different "
            "estimands; expand the data explicitly before bootstrapping"
        )
    if weights is not None:
        weights = _row_vector(weights, n, "weights")

    if method == "pairs":
        if bootstrap_cluster is not None:
            raise ValueError(
                "bootstrap_cluster is only valid with method='cluster_pairs'"
            )
        cluster_groups = None
    else:
        if bootstrap_cluster is None:
            raise ValueError(
                "method='cluster_pairs' requires bootstrap_cluster; the "
                "resampling unit is never inferred from the VCE"
            )
        bootstrap_cluster_name = getattr(bootstrap_cluster, "name", None)
        bootstrap_cluster = _row_vector(
            bootstrap_cluster, n, "bootstrap_cluster"
        )
        if _has_missing_ids(bootstrap_cluster):
            raise ValueError(
                "bootstrap_cluster must not contain missing values"
            )
        _, cluster_codes = np.unique(
            bootstrap_cluster, return_inverse=True
        )
        n_boot_clusters = int(cluster_codes.max()) + 1
        if n_boot_clusters < 2:
            raise ValueError(
                "cluster-pairs bootstrap requires at least two clusters"
            )
        cluster_groups = [
            np.flatnonzero(cluster_codes == code)
            for code in range(n_boot_clusters)
        ]
    if method == "pairs":
        bootstrap_cluster_name = None

    point = decompose(
        y, x1, x2_groups=x2_groups, fes=fes, **decompose_kwargs
    )
    if not point["converged"]:
        raise RuntimeError(
            "the point decomposition did not converge; bootstrap aborted"
        )
    if require_gpu_used and not point["gpu_used"]:
        raise RuntimeError(
            "GPU use was required but the point estimate used a fallback"
        )

    point_arrays = _point_arrays(point)
    master = np.random.SeedSequence(seed)
    child_sequences = master.spawn(reps)
    valid = []
    ledger = []
    failure_counter = Counter()

    for replication, child_sequence in enumerate(child_sequences, start=1):
        rng = np.random.Generator(np.random.PCG64(child_sequence))
        if method == "pairs":
            indices = rng.integers(0, n, size=n, dtype=np.int64)
        else:
            selected = rng.integers(
                0, len(cluster_groups), size=len(cluster_groups),
                dtype=np.int64,
            )
            indices = np.concatenate([
                cluster_groups[int(code)] for code in selected
            ])

        replicate_kwargs = dict(decompose_kwargs)
        replicate_kwargs["vce"] = "unadjusted"
        replicate_kwargs.pop("cluster", None)
        replicate_kwargs["common_fes"] = _resample_mapping(
            common_fes, indices
        )
        if weights is not None:
            replicate_kwargs["weights"] = weights[indices]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fit = decompose(
                    y_array[indices],
                    x1_array[indices, :],
                    x2_groups=_resample_mapping(x2_groups, indices),
                    fes=_resample_mapping(fes, indices),
                    **replicate_kwargs,
                )
            if not fit["converged"]:
                raise RuntimeError("decomposition_not_converged")
            if require_gpu_used and not fit["gpu_used"]:
                raise RuntimeError("gpu_fallback")
            arrays = _point_arrays(fit)
            p = point_arrays["b_base"].size
            if not (
                np.all(np.isfinite(arrays["delta"][:p, :]))
                and np.all(np.isfinite(arrays["total"][:p]))
                and np.all(np.isfinite(arrays["b_base"]))
                and np.all(np.isfinite(arrays["b_full"]))
            ):
                raise RuntimeError("nonfinite_slope_result")
            if arrays["delta"].shape != point_arrays["delta"].shape:
                raise RuntimeError("replicate_schema_changed")
            valid.append(arrays)
            ledger.append({
                "replication": replication,
                "status": "valid",
                "reason": "",
                "n_rows_drawn": int(indices.size),
                "n_obs_retained": int(fit["n_obs"]),
                "n_singletons_dropped": int(
                    fit["n_singletons_dropped"]
                ),
                "identity_gap": float(fit["identity_gap"]),
                "gpu_used": bool(fit["gpu_used"]),
                "spawn_key": tuple(child_sequence.spawn_key),
            })
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failure_counter[reason] += 1
            ledger.append({
                "replication": replication,
                "status": "failed",
                "reason": reason,
                "n_rows_drawn": int(indices.size),
                "n_obs_retained": None,
                "n_singletons_dropped": None,
                "identity_gap": None,
                "gpu_used": None,
                "spawn_key": tuple(child_sequence.spawn_key),
            })

    valid_count = len(valid)
    if valid_count < min_valid_reps:
        summary = "; ".join(
            f"{count} x {reason}"
            for reason, count in failure_counter.most_common(3)
        ) or "no individual exception recorded"
        raise RuntimeError(
            "Gelbach bootstrap failed closed: "
            f"{valid_count}/{reps} valid replications, fewer than "
            f"min_valid_reps={min_valid_reps}. Failures: {summary}"
        )
    if valid_count < reps:
        warnings.warn(
            "xhdfe.gelbach.bootstrap retained "
            f"{valid_count}/{reps} valid replications; inspect "
            "result['bootstrap']['ledger'] and failure_counts.",
            RuntimeWarning,
            stacklevel=2,
        )

    draws = {
        key: np.stack([replicate[key] for replicate in valid], axis=0)
        for key in ("delta", "total", "b_base", "b_full")
    }
    draws.update(_bootstrap_share_draws(draws, float(share_tol)))
    point_share_base = np.full_like(point_arrays["delta"], np.nan)
    point_share_movement = np.full_like(point_arrays["delta"], np.nan)
    p = point_arrays["b_base"].size
    for row in range(p):
        denominator = point_arrays["b_base"][row]
        if np.isfinite(denominator) and abs(denominator) > share_tol:
            point_share_base[row, :] = (
                point_arrays["delta"][row, :] / denominator
            )
    for row in range(point_arrays["total"].size):
        denominator = point_arrays["total"][row]
        if np.isfinite(denominator) and abs(denominator) > share_tol:
            point_share_movement[row, :] = (
                point_arrays["delta"][row, :] / denominator
            )
    point_full_share_base = np.full_like(point_arrays["b_full"], np.nan)
    point_total_share_base = np.full_like(point_arrays["b_full"], np.nan)
    valid_base = (
        np.isfinite(point_arrays["b_base"])
        & (np.abs(point_arrays["b_base"]) > share_tol)
    )
    point_full_share_base[valid_base] = (
        point_arrays["b_full"][valid_base]
        / point_arrays["b_base"][valid_base]
    )
    point_total_share_base[valid_base] = (
        point_arrays["total"][:p][valid_base]
        / point_arrays["b_base"][valid_base]
    )
    interval_points = {
        **point_arrays,
        "share_base": point_share_base,
        "share_movement": point_share_movement,
        "full_share_base": point_full_share_base,
        "total_share_base": point_total_share_base,
    }
    intervals = {
        key: _finite_quantile_interval(
            draws[key], interval_points[key], conf_level, ci_method
        )
        for key in interval_points
    }

    point["bootstrap"] = {
        "method": method,
        "resampling_unit": (
            "observation" if method == "pairs" else "declared_cluster"
        ),
        "bootstrap_cluster_name": bootstrap_cluster_name,
        "seed": seed,
        "rng": "numpy.SeedSequence+PCG64_per_replication",
        "reps_requested": reps,
        "reps_valid": valid_count,
        "reps_failed": reps - valid_count,
        "min_valid_reps": min_valid_reps,
        "conf_level": float(conf_level),
        "ci_method": ci_method,
        "interval_status":
            "resampling_based_not_a_nonregularity_cure",
        "share_tol": float(share_tol),
        "component_names": list(point["names"]),
        "coefficient_names": list(point["labels"]),
        "ledger": ledger,
        "failure_counts": dict(failure_counter),
        "intervals": intervals,
        "draws": draws if store_draws else None,
        "draws_stored": bool(store_draws),
        "point_vce": point["vce"],
        "replicate_vce": "unadjusted_point_functional_only",
        "gpu_required": bool(require_gpu_used),
        "gpu_used_all_valid": bool(
            all(entry["gpu_used"] for entry in ledger
                if entry["status"] == "valid")
        ) if bool(decompose_kwargs.get("gpu", False)) else False,
    }
    return point


def _patterns(value, name):
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a string or sequence") from exc
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} patterns must be non-empty strings")
    return values


def _selected_components(names, keep=None, drop=None, exact_match=False):
    keep = _patterns(keep, "keep")
    drop = _patterns(drop, "drop")

    def matches(name, patterns):
        if exact_match:
            return name in patterns
        try:
            return any(re.search(pattern, name) is not None
                       for pattern in patterns)
        except re.error as exc:
            raise ValueError(f"invalid keep/drop regular expression: {exc}") from exc

    selected = [
        name for name in names
        if (not keep or matches(name, keep))
        and not (drop and matches(name, drop))
    ]
    return selected


def _bootstrap_interval_for(result, key, index):
    bootstrap_result = result.get("bootstrap")
    if not bootstrap_result:
        return None
    interval = bootstrap_result["intervals"][key]
    return {
        "low": float(interval["low"][index]),
        "high": float(interval["high"][index]),
        "std_error": float(interval["std_error"][index]),
        "valid_count": int(interval["valid_counts"][index]),
        "method": bootstrap_result["ci_method"],
        "conf_level": float(bootstrap_result["conf_level"]),
    }


def _panel_names(panels):
    if panels == "all":
        return ["levels", "share_base", "share_movement"]
    if isinstance(panels, str):
        panels = [panels]
    try:
        normalized = [_PANEL_ALIASES[str(panel)] for panel in panels]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "panels must be 'all', 'levels', 'share_base'/'share_full', "
            "or 'share_movement'/'share_explained'"
        ) from exc
    if len(set(normalized)) != len(normalized):
        raise ValueError("panels may not contain duplicates")
    return normalized


def _filtered_aggregate_stats(
        result, coefficient_index, members, panel, share_tol
):
    """Joint-covariance statistics for one aggregate of filtered components."""
    names = list(result["names"])
    k1 = len(result["labels"])
    member_indices = [names.index(name) for name in members]
    weights = np.zeros(len(names), dtype=float)
    weights[member_indices] = 1.0
    positions = [group * k1 + coefficient_index for group in range(len(names))]
    covariance = np.asarray(result["cov"])[np.ix_(positions, positions)]
    contributions = np.asarray([
        result["delta"][name]["coef"][coefficient_index] for name in names
    ], dtype=float)
    numerator = float(weights @ contributions)
    numerator_variance = float(weights @ covariance @ weights)
    if panel == "levels":
        return numerator, np.sqrt(max(0.0, numerator_variance))

    if panel == "share_base":
        denominator = float(result["b_base"][coefficient_index])
        if not np.isfinite(denominator) or abs(denominator) <= share_tol:
            return np.nan, np.nan
        cross = np.asarray(result["cov_delta_bbase"], dtype=float)
        covariance_num_den = float(sum(
            cross[group * k1 + coefficient_index, coefficient_index]
            for group in member_indices
        ))
        denominator_variance = float(
            np.asarray(result["base_cov"])[
                coefficient_index, coefficient_index
            ]
        )
        variance = (
            numerator_variance / (denominator ** 2)
            + numerator ** 2 * denominator_variance / (denominator ** 4)
            - 2.0 * numerator * covariance_num_den / (denominator ** 3)
        )
        return (
            numerator / denominator,
            np.sqrt(max(0.0, variance))
            if np.isfinite(variance) else np.nan,
        )

    denominator = float(result["total"]["coef"][coefficient_index])
    if not np.isfinite(denominator) or abs(denominator) <= share_tol:
        return np.nan, np.nan
    gradient = (
        weights / denominator
        - numerator * np.ones(len(names)) / (denominator ** 2)
    )
    variance = float(gradient @ covariance @ gradient)
    return numerator / denominator, np.sqrt(max(0.0, variance))


def _table_records(
        result,
        tidy,
        *,
        focal,
        panels,
        keep,
        drop,
        exact_match,
        labels,
        include_other,
        conf_level,
        interval,
        share_tol,
        share_t_min,
):
    if interval not in ("auto", "normal", "bootstrap"):
        raise ValueError("interval must be 'auto', 'normal' or 'bootstrap'")
    use_bootstrap = (
        interval == "bootstrap"
        or (interval == "auto" and "bootstrap" in result)
    )
    if use_bootstrap and "bootstrap" not in result:
        raise ValueError("bootstrap intervals requested but result has no bootstrap")
    if labels is None:
        labels = {}
    if not isinstance(labels, dict):
        raise TypeError("labels must be a mapping from component names to labels")
    unknown_labels = [name for name in labels if name not in result["names"]]
    if unknown_labels:
        raise ValueError(f"labels contains unknown component(s): {unknown_labels}")
    selected_order = _selected_components(
        result["names"], keep, drop, exact_match
    )
    selected = set(selected_order)
    omitted = [
        name for name in result["names"] if name not in selected
    ]
    if omitted and not include_other:
        warnings.warn(
            "xhdfe.gelbach.etable: keep/drop filtered components without "
            "an Other (filtered) aggregate; displayed components do not "
            "preserve the Gelbach accounting identity.",
            RuntimeWarning,
            stacklevel=3,
        )
    panel_names = _panel_names(panels)
    records = []
    for panel in panel_names:
        share = None if panel == "levels" else (
            "base" if panel == "share_base" else "movement"
        )
        rows = tidy(
            result,
            focal=focal,
            include_intercept=False,
            include_total=True,
            include_full=True,
            conf_level=conf_level,
            share=share,
            share_tol=share_tol,
            share_t_min=share_t_min,
        )
        coefficient_names = []
        for row in rows:
            if row["coefficient"] not in coefficient_names:
                coefficient_names.append(row["coefficient"])
        if panel in ("levels", "share_base"):
            for coefficient in coefficient_names:
                coefficient_index = result["labels"].index(coefficient)
                if panel == "levels":
                    estimate = float(result["b_base"][coefficient_index])
                    std_error = float(np.sqrt(max(
                        0.0,
                        result["base_cov"][
                            coefficient_index, coefficient_index
                        ],
                    )))
                    zcrit = NormalDist().inv_cdf(
                        0.5 + conf_level / 2.0
                    )
                    low = estimate - zcrit * std_error
                    high = estimate + zcrit * std_error
                    boot = _bootstrap_interval_for(
                        result, "b_base", coefficient_index
                    ) if use_bootstrap else None
                else:
                    estimate, std_error = 1.0, 0.0
                    low = high = 1.0
                    boot = None
                if boot is not None:
                    low, high = boot["low"], boot["high"]
                    std_error = boot["std_error"]
                    ci_method = f"bootstrap_{boot['method']}"
                    ci_valid = boot["valid_count"]
                else:
                    ci_method = (
                        "identity" if panel == "share_base"
                        else ("normal_delta" if not use_bootstrap
                              else "not_available")
                    )
                    ci_valid = None
                records.append({
                    "panel": panel,
                    "coefficient": coefficient,
                    "component": "base_model",
                    "component_name": "base_model",
                    "component_kind": "base_model",
                    "estimate": estimate,
                    "std_error": std_error,
                    "conf_low": low,
                    "conf_high": high,
                    "conf_level": (
                        result["bootstrap"]["conf_level"]
                        if boot is not None else conf_level
                    ),
                    "confidence_method": ci_method,
                    "bootstrap_valid_count": ci_valid,
                })
        for row in rows:
            component = row["component"]
            if component in result["names"] and component not in selected:
                continue
            coefficient_index = result["labels"].index(row["coefficient"])
            if (
                component == "total_movement"
                and omitted
                and include_other
            ):
                estimate, std_error = _filtered_aggregate_stats(
                    result,
                    coefficient_index,
                    omitted,
                    panel,
                    share_tol,
                )
                zcrit = NormalDist().inv_cdf(0.5 + conf_level / 2.0)
                aggregate_method = "normal_delta"
                if (
                    panel != "levels"
                    and row.get("share_interval_status")
                    == "weak_denominator_delta_method_unreliable"
                ):
                    aggregate_method = (
                        "normal_delta_diagnostic_only_weak_denominator"
                    )
                records.append({
                    "panel": panel,
                    "coefficient": row["coefficient"],
                    "component": "Other (filtered)",
                    "component_name": "other_filtered",
                    "component_kind": "filtered_aggregate",
                    "estimate": estimate,
                    "std_error": std_error,
                    "conf_low": estimate - zcrit * std_error,
                    "conf_high": estimate + zcrit * std_error,
                    "conf_level": conf_level,
                    "confidence_method": aggregate_method,
                    "bootstrap_valid_count": None,
                })
            if panel == "levels":
                estimate = float(row["estimate"])
                std_error = float(row["std_error"])
                low = float(row["conf_low"])
                high = float(row["conf_high"])
                if component in result["names"]:
                    group_index = result["names"].index(component)
                    boot = _bootstrap_interval_for(
                        result, "delta", (coefficient_index, group_index)
                    ) if use_bootstrap else None
                elif component == "total_movement":
                    boot = _bootstrap_interval_for(
                        result, "total", coefficient_index
                    ) if use_bootstrap else None
                else:
                    boot = _bootstrap_interval_for(
                        result, "b_full", coefficient_index
                    ) if use_bootstrap else None
            else:
                estimate = float(row["share"])
                std_error = float(row["share_std_error"])
                low = float(row["share_conf_low"])
                high = float(row["share_conf_high"])
                if component in result["names"]:
                    group_index = result["names"].index(component)
                    key = (
                        "share_base" if panel == "share_base"
                        else "share_movement"
                    )
                    boot = _bootstrap_interval_for(
                        result, key, (coefficient_index, group_index)
                    ) if use_bootstrap else None
                elif component == "total_movement":
                    boot = (
                        _bootstrap_interval_for(
                            result, "total_share_base", coefficient_index
                        )
                        if use_bootstrap and panel == "share_base"
                        else None
                    )
                    if panel == "share_movement" and np.isfinite(estimate):
                        low = high = estimate
                        std_error = 0.0
                else:
                    boot = _bootstrap_interval_for(
                        result, "full_share_base", coefficient_index
                    ) if use_bootstrap and panel == "share_base" else None
            if boot is not None:
                low, high = boot["low"], boot["high"]
                std_error = boot["std_error"]
                ci_method = f"bootstrap_{boot['method']}"
                ci_valid = boot["valid_count"]
            else:
                if (
                    use_bootstrap
                    or not all(np.isfinite(value) for value in (
                        std_error, low, high
                    ))
                ):
                    ci_method = "not_available"
                elif (
                    panel != "levels"
                    and row.get("share_interval_status")
                    == "weak_denominator_delta_method_unreliable"
                ):
                    ci_method = (
                        "normal_delta_diagnostic_only_weak_denominator"
                    )
                elif str(row.get(
                    "confidence_interval_status", ""
                )).startswith("diagnostic_only"):
                    ci_method = "normal_delta_diagnostic_only"
                else:
                    ci_method = "normal_delta"
                ci_valid = None
            records.append({
                "panel": panel,
                "coefficient": row["coefficient"],
                "component": labels.get(component, component),
                "component_name": component,
                "component_kind": row["component_kind"],
                "estimate": estimate,
                "std_error": std_error,
                "conf_low": low,
                "conf_high": high,
                "conf_level": (
                    result["bootstrap"]["conf_level"]
                    if boot is not None else conf_level
                ),
                "confidence_method": ci_method,
                "bootstrap_valid_count": ci_valid,
            })
    return records


def _format_number(value, digits):
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def _render_markdown(records, digits, caption, notes):
    columns = [
        "Panel", "Coefficient", "Component", "Estimate", "Std. error",
        "CI low", "CI high", "CI method",
    ]
    lines = []
    if caption:
        safe_caption = str(caption).replace("|", "\\|")
        lines.extend([f"**{safe_caption}**", ""])
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in records:
        values = [
            row["panel"], row["coefficient"], row["component"],
            _format_number(row["estimate"], digits),
            _format_number(row["std_error"], digits),
            _format_number(row["conf_low"], digits),
            _format_number(row["conf_high"], digits),
            row["confidence_method"],
        ]
        values = [str(value).replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(values) + " |")
    if notes:
        lines.extend(["", f"Notes: {notes}"])
    return "\n".join(lines)


def _latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _render_latex(records, digits, caption, notes):
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        lines.append(r"\caption{" + _latex_escape(caption) + "}")
    lines.extend([
        r"\begin{tabular}{lllrrrrl}",
        r"\hline",
        r"Panel & Coefficient & Component & Estimate & SE & CI low & CI high & Method \\",
        r"\hline",
    ])
    for row in records:
        values = [
            _latex_escape(row["panel"]),
            _latex_escape(row["coefficient"]),
            _latex_escape(row["component"]),
            _format_number(row["estimate"], digits),
            _format_number(row["std_error"], digits),
            _format_number(row["conf_low"], digits),
            _format_number(row["conf_high"], digits),
            _latex_escape(row["confidence_method"]),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    if notes:
        lines.append(
            r"\begin{minipage}{0.95\linewidth}\footnotesize "
            + _latex_escape(notes) + r"\end{minipage}"
        )
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _render_html(records, digits, caption, notes):
    parts = ["<table class=\"xhdfe-gelbach\">"]
    if caption:
        parts.append(f"<caption>{html.escape(str(caption))}</caption>")
    parts.append(
        "<thead><tr><th>Panel</th><th>Coefficient</th><th>Component</th>"
        "<th>Estimate</th><th>Std. error</th><th>CI low</th>"
        "<th>CI high</th><th>CI method</th></tr></thead><tbody>"
    )
    for row in records:
        values = [
            row["panel"], row["coefficient"], row["component"],
            _format_number(row["estimate"], digits),
            _format_number(row["std_error"], digits),
            _format_number(row["conf_low"], digits),
            _format_number(row["conf_high"], digits),
            row["confidence_method"],
        ]
        parts.append(
            "<tr>" + "".join(
                f"<td>{html.escape(str(value))}</td>" for value in values
            ) + "</tr>"
        )
    parts.append("</tbody>")
    if notes:
        parts.append(
            f"<tfoot><tr><td colspan=\"8\">{html.escape(str(notes))}"
            "</td></tr></tfoot>"
        )
    parts.append("</table>")
    return "".join(parts)


def etable(
        result,
        tidy,
        *,
        panels="all",
        format="dataframe",
        type=None,
        focal=None,
        keep=None,
        drop=None,
        exact_match=False,
        labels=None,
        include_other=True,
        digits=3,
        caption=None,
        notes=None,
        conf_level=0.95,
        interval="auto",
        share_tol=1e-12,
        share_t_min=3.0,
):
    """Create a Gelbach table as records, DataFrame, Markdown, LaTeX, HTML or GT."""
    if type is not None:
        if format != "dataframe":
            raise ValueError("specify only one of format= or type=")
        format = type
    aliases = {
        "df": "dataframe",
        "records": "records",
        "md": "markdown",
        "tex": "latex",
        "gt": "gt",
        "html": "html",
        "markdown": "markdown",
        "latex": "latex",
        "dataframe": "dataframe",
    }
    if not isinstance(format, str) or format.lower() not in aliases:
        raise ValueError(
            "format must be records, dataframe/df, markdown/md, "
            "latex/tex, html or gt"
        )
    output_format = aliases[format.lower()]
    digits = _positive_int(digits, "digits", minimum=0)
    records = _table_records(
        result,
        tidy,
        focal=focal,
        panels=panels,
        keep=keep,
        drop=drop,
        exact_match=bool(exact_match),
        labels=labels,
        include_other=bool(include_other),
        conf_level=conf_level,
        interval=interval,
        share_tol=share_tol,
        share_t_min=share_t_min,
    )
    default_note = (
        "Specification accounting, not causal mediation. "
        f"VCE={result['vce']}; identity={result['identity_status']}."
    )
    point_notes = str(result.get("notes", "")).strip()
    if point_notes:
        default_note += f" {point_notes}"
    if "bootstrap" in result:
        boot = result["bootstrap"]
        default_note += (
            f" Bootstrap={boot['method']}, "
            f"valid reps={boot['reps_valid']}/{boot['reps_requested']}; "
            "resampling intervals do not by themselves cure nonregularity."
        )
    notes = default_note if notes is None else f"{default_note} {notes}"
    if output_format == "records":
        return records
    if output_format == "markdown":
        return _render_markdown(records, digits, caption, notes)
    if output_format == "latex":
        return _render_latex(records, digits, caption, notes)
    if output_format == "html":
        return _render_html(records, digits, caption, notes)
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "format='dataframe' and format='gt' require the optional pandas "
            "dependency; use format='records' for a dependency-free result"
        ) from exc
    frame = pd.DataFrame.from_records(records)
    if output_format == "dataframe":
        return frame
    try:
        from great_tables import GT
    except ImportError as exc:
        raise ImportError(
            "format='gt' requires the optional great_tables package"
        ) from exc
    table = GT(frame)
    if caption:
        table = table.tab_header(title=caption)
    return table


def waterfall_data(
        result,
        *,
        focal=None,
        keep=None,
        drop=None,
        exact_match=False,
        labels=None,
        include_other=True,
        share_tol=1e-12,
):
    """Return a dependency-free, identity-preserving waterfall specification."""
    from .gelbach import _result_focal_indices

    rows = _result_focal_indices(result, focal, False)
    if len(rows) != 1:
        raise ValueError("waterfall_data requires exactly one focal coefficient")
    row = rows[0]
    names = list(result["names"])
    selected = _selected_components(names, keep, drop, bool(exact_match))
    omitted = [name for name in names if name not in selected]
    if labels is None:
        labels = {}
    if not isinstance(labels, dict):
        raise TypeError("labels must be a mapping")
    unknown_labels = [name for name in labels if name not in names]
    if unknown_labels:
        raise ValueError(f"labels contains unknown component(s): {unknown_labels}")
    base = float(result["b_base"][row])
    full = float(result["b_full"][row])
    total = float(result["total"]["coef"][row])
    current = base
    records = [{
        "position": 0,
        "name": "base_model",
        "label": "Base model",
        "kind": "endpoint",
        "contribution": None,
        "change": None,
        "start": 0.0,
        "end": base,
        "value": base,
        "share_of_movement": None,
    }]
    plot_groups = [(name, [name]) for name in selected]
    if include_other and omitted:
        plot_groups.append(("other_filtered", omitted))
    for name, members in plot_groups:
        contribution = float(sum(
            result["delta"][member]["coef"][row] for member in members
        ))
        end = current - contribution
        share = (
            contribution / total
            if np.isfinite(total) and abs(total) > share_tol else np.nan
        )
        records.append({
            "position": len(records),
            "name": name,
            "label": (
                "Other (filtered)"
                if name == "other_filtered" else labels.get(name, name)
            ),
            "kind": (
                "filtered_aggregate" if name == "other_filtered"
                else result["group_kinds"][name]
            ),
            "members": members,
            "contribution": contribution,
            "change": -contribution,
            "start": current,
            "end": end,
            "value": end,
            "share_of_movement": share,
        })
        current = end
    records.append({
        "position": len(records),
        "name": "full_model",
        "label": "Full model",
        "kind": "endpoint",
        "contribution": None,
        "change": None,
        "start": 0.0,
        "end": full,
        "value": full,
        "share_of_movement": None,
        "waterfall_residual": current - full,
    })
    return {
        "coefficient": result["labels"][row],
        "base": base,
        "full": full,
        "movement": total,
        "identity_gap": float(result["identity_gap"]),
        "selected_components": selected,
        "omitted_components": omitted,
        "include_other": bool(include_other),
        "rows": records,
    }


def coefplot(
        result,
        *,
        focal=None,
        annotate_shares=True,
        title=None,
        figsize=None,
        keep=None,
        drop=None,
        exact_match=False,
        labels=None,
        notes=None,
        include_other=True,
        share_tol=1e-12,
        ax=None,
        save=None,
        show=False,
):
    """Draw an identity-preserving Gelbach waterfall chart with Matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "coefplot requires the optional matplotlib dependency; "
            "waterfall_data() remains dependency-free"
        ) from exc
    specification = waterfall_data(
        result,
        focal=focal,
        keep=keep,
        drop=drop,
        exact_match=exact_match,
        labels=labels,
        include_other=include_other,
        share_tol=share_tol,
    )
    if ax is None:
        figure, ax = plt.subplots(figsize=figsize or (12, 7))
    else:
        figure = ax.figure
    rows = specification["rows"]
    positions = np.arange(len(rows))
    endpoint_color = "#374151"
    positive_color = "#2563eb"
    negative_color = "#dc2626"
    other_color = "#6b7280"
    for entry in rows:
        x = entry["position"]
        if entry["kind"] == "endpoint":
            value = entry["value"]
            ax.bar(
                x, abs(value), bottom=min(0.0, value), width=0.72,
                color=endpoint_color, alpha=0.9,
            )
            continue
        start, end = entry["start"], entry["end"]
        color = (
            other_color if entry["kind"] == "filtered_aggregate"
            else (positive_color if entry["contribution"] >= 0
                  else negative_color)
        )
        ax.bar(
            x, abs(end - start), bottom=min(start, end), width=0.72,
            color=color, alpha=0.88,
        )
        if annotate_shares and np.isfinite(entry["share_of_movement"]):
            midpoint = 0.5 * (start + end)
            ax.text(
                x, midpoint,
                f"{100.0 * entry['share_of_movement']:.1f}%",
                ha="center", va="center", color="white",
                fontsize=9, fontweight="bold",
            )
    for left, right in zip(rows[:-1], rows[1:]):
        left_level = (
            left["value"] if left["kind"] == "endpoint" else left["end"]
        )
        right_level = (
            right["value"] if right["kind"] == "endpoint"
            else right["start"]
        )
        if np.isfinite(left_level) and np.isfinite(right_level):
            ax.plot(
                [left["position"] + 0.36, right["position"] - 0.36],
                [left_level, right_level],
                color="#9ca3af", linewidth=1.0, linestyle="--",
            )
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [entry["label"] for entry in rows], rotation=35, ha="right"
    )
    ax.set_ylabel(f"Coefficient: {specification['coefficient']}")
    ax.set_title(
        title or
        f"Gelbach decomposition of {specification['coefficient']}"
    )
    default_note = (
        "Movement = base - full. Filtered components are aggregated as "
        "'Other' so the waterfall remains an accounting identity."
    )
    ax.text(
        0.0, -0.22, default_note if notes is None else notes,
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
    )
    figure.tight_layout()
    if save is not None:
        figure.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    ax._xhdfe_gelbach_waterfall = specification
    return figure, ax
