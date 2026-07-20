"""Gelbach (2016) coefficient-movement decomposition, HDFE-aware (M9B).

This module provides specification accounting, not causal mediation. A causal
interpretation requires a separately justified research design; block names
and the decision to add a control or fixed effect are supplied by the user and
are not validated by the command.

Decomposes the change in the base-specification coefficients when covariate
groups (and/or absorbed fixed-effect blocks) are added:

    base:  y = X1*b_base + e_base            (always includes a constant)
    full:  y = X1*b_full + sum_g X2_g*G_g [+ absorbed FEs] + e

    b_base - b_full = sum_g delta_g,   delta_g = (X1~'X1~)^{-1} X1~'H_g

with H_g = X2_g @ G_g for observed groups and H_d = the observation-level
fixed-effect contribution for absorbed dimensions (recovered by the xhdfe
core). X1~ = [X1, 1]; the implicit constant makes each block's x1-row split
invariant to the fixed-effect normalization (only the constant row shifts),
exactly as in Gelbach's b1x2.

The default is the standard Gelbach estimand and remains fail-closed when an
X1 column is not identified in the full model. ``absorbed_targets=`` activates
a separate constrained estimand for a focal X1 column that belongs to the span
of an added FE (for example, a worker-invariant group indicator with worker
FEs). The backend must classify every declared target specifically as
FE-collinear; its full-model coefficient is imposed at zero and labelled
``imposed_zero``. It is not an estimated within-FE coefficient. Undeclared
omissions and generic rank dependencies still raise an error.

Inference reproduces b1x2 (validated to machine precision in
VALIDATE_GELBACH.py):

The reported covariance is the random-design/stacked-moment variance used by
``b1x2``: it includes sampling variation in the auxiliary projections. It is
not the smaller variance conditional on the realised covariate matrix. For an
absorbed target, the target-target block of ``total_cov`` is anchored to the
same base-model VCE because ``total_j`` and ``b_base_j`` are one estimator.

- vce='unadjusted' (default): V[d_g,d_h] = Omega[v_g,v_h]*(X1~'X1~)^{-1}
  (auxregV) + Gamma_g' V(G) Gamma_h (v2upart), Omega = centred residual
  cross-products / df_full. The stacked cross terms are identically zero
  (e is orthogonal to X2 in sample), so b1x2's cov0 coincides with its
  default here.
- vce='robust' / vce='cluster' (with `cluster` ids): b1x2's stacked-system
  sandwich. Bread = blockdiag(S, I_G x P) with S the full-design
  (X'X)^{-1} — in the absorbed case the within-transformed representation,
  obtained by absorbing the fixed effects out of each [x1, x2] column;
  scores = (X_i e_i ; X1~_i v_gi). Multipliers match b1x2/_robust:
  stacked n/(n-1) (G_c/(G_c-1) clustered), full-model n/df_full
  ((n-1)/df_full * G_c/(G_c-1) clustered). The total-change variance is the
  sum over all block pairs (equals b1x2's reported __TC).

Options: gamma0=True -> auxregV only (b1x2 gamma0); cov0=True -> drop the
robust cross terms (no-op for vce='unadjusted'). Absorbed FE blocks always
receive the gamma0 treatment (the sampling variance of absorbed
coefficients is unavailable by construction); observed groups get the full
formula, including robust cross terms via the within representation.
Accordingly, ``total_se_type`` is labelled ``mixed_*_conditional_fe`` when a
total combines fully modelled observed blocks with conditional FE blocks.

Per the brief, NO KSS-style correction is applied: the contributions are
linear functionals of the fixed effects and largely escape the quadratic
limited-mobility bias. Stata-style aweights and fweights are supported
(weights=, fweights=True), matching b1x2's weighted estimators exactly.

Fast/legacy paths (compiled core): the default fast path resolves the
per-FE-dimension split via an MLSMR exact-normal-equations solve with a
fail-closed convergence gate — it is the MORE-converged path. The env
kill-switches XHDFE_GELBACH_FAST_FIT=0 / XHDFE_GELBACH_WARM_RECOVERY=0 /
XHDFE_GELBACH_WITHIN_BATCH=0 exist for bitwise A/B reproduction of the
pre-2.14.1 legacy output only; on ill-conditioned FE graphs the legacy
retained path's per-dimension FE split can be materially wrong (its GS
recovery certificate is blind to slow graph modes — adversarial audit
09jul2026, finding G1) and is now cross-checked and flagged via
`converged`/`notes`. Do NOT treat FAST_FIT=0 as a safety fallback.

Severely near-collinear columns inside an observed x2 block are detected with
a bounded-cost normalized-Gram diagnostic. The decomposition values are left
unchanged, but ``notes`` and a ``RuntimeWarning`` flag that the block SE split
can be tolerance/rounding sensitive. For clustered VCE, the streamed meat may
use FMA and differ from the former materialized path by one last-place unit in
well-conditioned cells; coefficients and deltas remain bit-identical.

Note on interpretation: with two or more mobility components, the split of
the combined FE contribution into per-FE-dimension deltas depends on a
normalization convention (the component mean-shift documented above); the
total across FE dimensions and b_base - b_full are convention-invariant.
Within a single connected mobility component the x1-row split is identified.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


__all__ = ["decompose", "tidy", "contrast"]


def _core():
    import sys

    core = sys.modules.get("py_hdfe_v11")
    if core is None:
        core = sys.modules.get("xhdfe.py_hdfe_v11")
    if core is None:
        from . import py_hdfe_v11 as core
    return core


def _cluster_meat(Z, codes):
    if codes is None:
        return Z.T @ Z
    G = int(codes.max()) + 1
    sums = np.zeros((G, Z.shape[1]))
    np.add.at(sums, codes, Z)
    return sums.T @ sums


def _x1_labels(p, x1_names=None):
    if x1_names is None:
        return [f"x1_{v + 1}" for v in range(p)]
    labels = list(x1_names)
    if len(labels) != p:
        raise ValueError("x1_names must have one entry per x1 column")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("x1_names must contain non-empty strings")
    labels = [label.strip() for label in labels]
    if len(set(labels)) != len(labels):
        raise ValueError("x1_names must be unique")
    if "_cons" in labels:
        raise ValueError("x1_names may not use the reserved name '_cons'")
    return labels


def _focal_indices(focal, p, labels):
    """Normalize a reporting-only focal selector to zero-based indices."""
    if focal is None:
        return list(range(p))
    values = np.asarray(focal if np.ndim(focal) else [focal], dtype=object)
    if values.ndim != 1:
        raise ValueError("focal must be a one-dimensional name/index selector")
    if values.size == p and all(isinstance(v, (bool, np.bool_)) for v in values):
        indices = np.flatnonzero(values.astype(bool)).astype(int).tolist()
    elif all(isinstance(v, str) for v in values):
        unknown = [v for v in values if v not in labels]
        if unknown:
            raise ValueError(f"focal contains unknown x1 name(s): {unknown}")
        indices = [labels.index(v) for v in values]
    else:
        indices = []
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)):
                raise ValueError("focal must contain x1 names, zero-based indices, or a boolean mask")
            indices.append(int(value))
        if any(value < 0 or value >= p for value in indices):
            raise ValueError("focal index is outside the x1 column range")
    if not indices:
        raise ValueError("focal must select at least one x1 column")
    if len(set(indices)) != len(indices):
        raise ValueError("focal entries must be unique")
    return indices


def decompose(y, x1, x2_groups=None, fes=None, vce="unadjusted", cluster=None,
              gamma0=False, cov0=False, tol=1e-8, num_threads=0,
              weights=None, fweights=False, absorbed_targets=None,
              x1_names=None, focal=None):
    """Gelbach decomposition of the coefficient movement b_base - b_full.

    The result is an accounting identity for the declared base and full
    specifications. It does not identify a causal mechanism or mediated
    effect.

    Parameters
    ----------
    y : (n,) outcome.
    x1 : (n, p) base covariates (no constant column; one is implicit).
    x2_groups : dict name -> (n,) or (n, q_g) observed covariate group(s).
    fes : dict name -> (n,) integer ids of absorbed fixed-effect dimensions.
    vce : 'unadjusted' (b1x2 default), 'robust', or 'cluster' (requires
        `cluster` ids). Matches b1x2's estimators exactly (see module doc).
    cluster : optional length-n one-way cluster ids. Required exactly when
        ``vce='cluster'``; at least two clusters are required.
    gamma0 : bool — aux-regression variance only (b1x2's gamma0).
    cov0 : bool — drop the robust cross terms (b1x2's cov0; no-op for
        vce='unadjusted').
    tol : positive float — fixed-effect absorption tolerance. The default
        1e-8 preserves the historical effective tolerance of this wrapper.
        It controls iterative absorption, not the separate FE-collinearity
        classification. The latter uses the documented squared-norm rule
        ``||M_D x||^2 / ||x||^2 <= 1e-9`` (relative norm about 3.16e-5),
        returned as metadata.
    num_threads : nonnegative OpenMP thread request; zero uses the library
        default.
    weights : optional positive analytic or frequency weights.
    fweights : interpret ``weights`` as positive integer frequency weights.
    absorbed_targets : optional zero-based X1 column indices or a length-p
        boolean mask. This activates a distinct, constrained estimand for
        targets that are fully absorbed by ``fes``. Every declared target
        must be classified by the backend as collinear with the absorbed FEs;
        its full-model coefficient is imposed at zero, not estimated. The
        standard rank guard is unchanged when this argument is omitted.
    x1_names : optional sequence of unique names for the X1 columns. Names are
        reporting metadata only and do not alter the design matrix.
    focal : optional X1 names, zero-based indices, or length-p boolean mask.
        This selects the coefficients to report in :func:`tidy`; all X1
        columns remain in both the base and full specifications. It is useful
        when X1 contains one focal regressor plus common controls. The
        decomposition and returned full-precision matrices are unchanged.

    Returns
    -------
    dict with 'b_base', 'b_full', per-group 'delta' (contribution over
    [x1..., _cons] with 'se'), 'total' (with SE), 'cov', 'identity_gap',
    sample sizes and notes. The total covariance is available both as
    ``total['cov']`` and the cross-language top-level alias ``total_cov``.
    ``fe_total`` is the normalization-safe aggregate FE object. Always inspect
    ``converged``, ``notes``, ``total_se_type`` and, in absorbed-target mode,
    ``absorbed_target_inference_valid``.

    See Also
    --------
    ``xhdfe.help_text('gelbach')`` documents every field, share convention,
    inference qualifier, example and deliberate limitation.
    """
    y = np.ascontiguousarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    x1 = np.asarray(x1, dtype=float)
    if x1.ndim == 1:
        x1 = x1[:, None]
    n, p = x1.shape
    if p == 0:
        raise ValueError("x1 must contain at least one focal column")
    x1_labels = _x1_labels(p, x1_names)
    focal_x1 = _focal_indices(focal, p, x1_labels)
    if y.shape[0] != n:
        raise ValueError("y and x1 must have the same number of rows")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x1)):
        raise ValueError("y and x1 must contain only finite values")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and strictly positive")
    if vce not in ("unadjusted", "robust", "cluster"):
        raise ValueError("vce must be 'unadjusted', 'robust' or 'cluster'")
    if vce == "cluster":
        if cluster is None:
            raise ValueError("vce='cluster' requires cluster ids")
        cluster = np.asarray(cluster)
        if cluster.ndim != 1 or cluster.shape[0] != n:
            raise ValueError("cluster ids must be one-dimensional with length n")
        if np.issubdtype(cluster.dtype, np.number) and not np.all(np.isfinite(cluster)):
            raise ValueError("cluster ids must not contain missing or non-finite values")
        _, ccodes = np.unique(cluster, return_inverse=True)
        if np.unique(ccodes).size < 2:
            raise ValueError("vce='cluster' requires at least two clusters")
    else:
        # Single source of truth: cluster ids with a non-cluster vce would
        # otherwise be silently dropped and the user would get robust/
        # unadjusted SEs labelled as such — match the Stata and R front-ends,
        # which reject this combination rather than ignore it.
        if cluster is not None:
            raise ValueError("cluster ids supplied but vce != 'cluster'")
        ccodes = None
    x2_groups = dict(x2_groups or {})
    fes = dict(fes or {})
    if not x2_groups and not fes:
        raise ValueError("provide at least one x2 group or fixed-effect dimension")
    all_names = list(x2_groups) + list(fes)
    if any(not isinstance(name, str) or not name.strip() for name in all_names):
        raise ValueError("every x2/FE block must have a non-empty string name")
    if len(set(all_names)) != len(all_names):
        raise ValueError("x2 and FE block names must be unique")

    if absorbed_targets is None:
        absorbed_x1 = []
    else:
        target_arr = np.asarray(absorbed_targets)
        if target_arr.ndim == 0:
            target_arr = target_arr.reshape(1)
        if target_arr.ndim != 1:
            raise ValueError("absorbed_targets must be a one-dimensional mask or index list")
        if np.issubdtype(target_arr.dtype, np.bool_):
            if target_arr.size != p:
                raise ValueError("an absorbed_targets boolean mask must have length p")
            absorbed_x1 = np.flatnonzero(target_arr).astype(int).tolist()
        else:
            if not np.issubdtype(target_arr.dtype, np.integer):
                if (np.issubdtype(target_arr.dtype, np.floating) and
                        np.all(np.isfinite(target_arr)) and
                        np.all(target_arr == np.rint(target_arr))):
                    target_arr = np.rint(target_arr).astype(np.int64)
                else:
                    raise ValueError("absorbed_targets indices must be integers")
            absorbed_x1 = target_arr.astype(int).tolist()
            if any(c < 0 or c >= p for c in absorbed_x1):
                raise ValueError("absorbed_targets index is outside the x1 column range")
            if len(set(absorbed_x1)) != len(absorbed_x1):
                raise ValueError("absorbed_targets indices must be unique")
        if absorbed_x1 and not fes:
            raise ValueError("absorbed_targets requires at least one absorbed FE dimension")

    groups = []  # (name, kind, payload)
    x2_cols = []
    for name, arr in x2_groups.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] != n:
            raise ValueError(f"x2 group {name!r} has wrong length")
        if arr.shape[1] == 0:
            raise ValueError(f"x2 group {name!r} must contain at least one column")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"x2 group {name!r} must contain only finite values")
        groups.append((name, "x2", arr))
        x2_cols.append(arr)
    fe_lists = []
    for name, ids in fes.items():
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.shape[0] != n:
            raise ValueError(f"fe {name!r} must be one-dimensional with length n")
        if not np.issubdtype(ids.dtype, np.integer):
            if (np.issubdtype(ids.dtype, np.floating) and
                    np.all(np.isfinite(ids)) and np.all(ids == np.rint(ids))):
                ids = np.rint(ids).astype(np.int64)
            else:
                raise ValueError(f"fe {name!r} ids must be finite integers")
        groups.append((name, "fe", len(fe_lists)))
        fe_lists.append(ids)

    w = None
    if weights is not None:
        w = np.ascontiguousarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != n:
            raise ValueError("weights must be one-dimensional with length n")
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("weights must be finite and strictly positive")
        if fweights and not np.all(w == np.rint(w)):
            raise ValueError("frequency weights must be integers")
    elif fweights:
        raise ValueError("fweights=True requires weights")

    core = _core()
    if not hasattr(core, "gelbach_decompose"):
        raise ImportError("the compiled xhdfe extension predates the Gelbach "
                          "module; rebuild the package")

    x2_sizes = [np.asarray(d).shape[1] for _, k, d in groups if k == "x2"]
    X2 = (np.hstack([np.asarray(d, dtype=float) for _, k, d in groups
                     if k == "x2"]) if x2_sizes else None)
    r = core.gelbach_decompose(y, x1, x2=X2, x2_group_sizes=x2_sizes,
                               fes=fe_lists if fe_lists else None,
                               cluster=(np.ascontiguousarray(ccodes, dtype=np.int64)
                                        if ccodes is not None else None),
                               vce=vce, gamma0=bool(gamma0), cov0=bool(cov0),
                               tol=float(tol), num_threads=num_threads,
                               weights=w,
                               fweights=bool(fweights),
                               absorbed_x1=absorbed_x1)

    p_ = x1.shape[1]
    k1 = p_ + 1
    delta = np.asarray(r["delta"])
    fullcov = np.asarray(r["cov"])
    # group order in the compiled core: x2 groups first, then FE dims — match
    # the caller's insertion order of `groups`.
    order = [g for g, (_, kind, _d) in enumerate(groups) if kind == "x2"] + \
            [g for g, (_, kind, _d) in enumerate(groups) if kind == "fe"]
    names = [groups[g][0] for g in order]
    kinds = [groups[g][1] for g in order]
    labels = x1_labels + ["_cons"]
    observed_se_type = (
        "gamma0" if gamma0 else
        ("cov0" if cov0 and vce != "unadjusted" else "full")
    )
    se_type = {
        name: ("conditional_gamma0" if kind == "fe" else
               observed_se_type)
        for name, kind in zip(names, kinds)
    }
    absorbed_mask = np.asarray(r["x1_absorbed"], dtype=bool)
    absorbed_mode = bool(np.any(absorbed_mask))
    b_full_status = ["imposed_zero" if value else "estimated"
                     for value in absorbed_mask]
    focal_status = ["absorbed" if value else "identified"
                    for value in absorbed_mask]
    has_fe_groups = any(kind == "fe" for kind in kinds)
    has_observed_groups = any(kind == "x2" for kind in kinds)
    total_cov = np.asarray(r["total_cov"])
    if has_fe_groups:
        total_se_type = (
            "conditional_gamma0" if gamma0 or not has_observed_groups else
            ("mixed_cov0_observed_conditional_fe"
             if cov0 and vce != "unadjusted" else
             "mixed_full_observed_conditional_fe")
        )
    else:
        total_se_type = observed_se_type
    if absorbed_mode:
        total_se_type = "target_exact_base_vce_mixed_components"
    inference_status = "not_applicable"
    if absorbed_mode:
        inference_status = (
            "clustered_at_absorbing_fe"
            if bool(r["absorbed_target_inference_valid"])
            else "warning_unsupported_vce_or_cluster"
        )
    out = {
        "names": names,
        "group_kinds": dict(zip(names, kinds)),
        "labels": labels,
        "x1_names": x1_labels,
        "focal_indices": focal_x1,
        "focal_names": [x1_labels[c] for c in focal_x1],
        "b_base": np.asarray(r["b_base"]),
        "b_full": np.asarray(r["b_full"]),
        "b_full_status": b_full_status,
        "focal_status": focal_status,
        "absorbed_mask": absorbed_mask.astype(bool).tolist(),
        "absorbed_targets": np.flatnonzero(absorbed_mask).astype(int).tolist(),
        "absorbed_target_names": [labels[c] for c in np.flatnonzero(absorbed_mask)],
        "delta": {
            name: {"coef": delta[:, g],
                   "se": np.sqrt(np.diag(
                       fullcov[g * k1:(g + 1) * k1, g * k1:(g + 1) * k1])),
                   "se_type": se_type[name]}
            for g, name in enumerate(names)
        },
        "total": {"coef": np.asarray(r["total"]),
                  "cov": total_cov,
                  "se": np.sqrt(np.diag(total_cov)),
                  "se_type": total_se_type},
        "total_cov": total_cov,
        "cov": fullcov,
        "identity_gap": float(r["identity_gap"]),
        "n_obs_input": int(r["n_obs_input"]),
        "n_obs": int(r["n_obs"]),
        "n_obs_effective": int(r["n_obs_effective"]),
        "n_singletons_dropped": int(r["n_singletons_dropped"]),
        "df_full": float(r["df_full"]),
        "fe_collinear_ss_ratio_tol": float(r["fe_collinear_ss_ratio_tol"]),
        "fe_collinear_relative_norm_tol": float(
            np.sqrt(r["fe_collinear_ss_ratio_tol"])
        ),
        "absorbed_target_inference_valid": bool(
            r["absorbed_target_inference_valid"]
        ),
        "absorbing_fe_index": int(r["absorbing_fe_index"]),
        "inference_status": inference_status,
        "vce": vce,
        "gamma0": bool(gamma0),
        "cov0": bool(cov0),
        "tol": float(tol),
        "estimand": ("absorbed_target_allocation" if absorbed_mode else
                     "coefficient_movement"),
        "identity_status": ("exact_ols_constrained" if absorbed_mode else
                            "exact_ols"),
        "total_se_type": total_se_type,
        "causal_interpretation": False,
        "notes": r["notes"],
        "converged": bool(r["converged"]),
    }
    fe_groups = [g for g, kind in enumerate(kinds) if kind == "fe"]
    if fe_groups:
        fe_coef = delta[:, fe_groups].sum(axis=1)
        fe_cov = np.zeros((k1, k1), dtype=float)
        for g in fe_groups:
            for h in fe_groups:
                fe_cov += fullcov[g * k1:(g + 1) * k1,
                                  h * k1:(h + 1) * k1]
        out["fe_total"] = {
            "members": [names[g] for g in fe_groups],
            "coef": fe_coef,
            "cov": fe_cov,
            "se": np.sqrt(np.diag(fe_cov)),
            "se_type": "conditional_gamma0",
        }
    else:
        out["fe_total"] = None
    if not r["converged"]:
        out["notes"] += " (not converged)"
        import warnings

        warnings.warn(
            "xhdfe.gelbach.decompose: the decomposition did not converge or "
            "failed a convergence cross-check — results are unreliable. "
            f"notes: {out['notes'].strip()}",
            RuntimeWarning,
            stacklevel=2,
        )
    elif "warning:" in out["notes"].lower():
        import warnings

        warnings.warn(
            "xhdfe.gelbach.decompose: inferential diagnostic. "
            f"notes: {out['notes'].strip()}",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def _result_focal_indices(result, focal=None, include_intercept=False):
    labels = list(result["labels"])
    p = len(result["b_base"])
    selected = _focal_indices(
        result.get("focal_indices") if focal is None else focal,
        p,
        labels[:p],
    )
    if include_intercept:
        selected.append(p)
    return selected


def _row_group_covariance(result, row):
    names = list(result["names"])
    k1 = len(result["labels"])
    positions = [g * k1 + row for g in range(len(names))]
    return np.asarray(result["cov"])[np.ix_(positions, positions)]


def _share_rows(result, denominator, share_tol):
    if denominator not in ("base", "base_fixed", "movement"):
        raise ValueError("share must be None, 'base', 'base_fixed' or 'movement'")
    if not np.isfinite(share_tol) or share_tol < 0:
        raise ValueError("share_tol must be finite and nonnegative")

    names = list(result["names"])
    k1 = len(result["labels"])
    p = len(result["b_base"])
    G = len(names)
    coef = np.column_stack([result["delta"][name]["coef"] for name in names])
    share = np.full((k1, G), np.nan)
    se = np.full((k1, G), np.nan)
    defined = np.zeros(k1, dtype=bool)

    for row in range(k1):
        if denominator in ("base", "base_fixed"):
            if row >= p:
                continue
            denom = float(result["b_base"][row])
        else:
            denom = float(result["total"]["coef"][row])
        if not np.isfinite(denom) or abs(denom) <= share_tol:
            continue
        defined[row] = True
        d = coef[row, :]
        share[row, :] = d / denom
        V = _row_group_covariance(result, row)
        if denominator == "base_fixed":
            # This is the reporting convention used in several applications:
            # scale Gelbach's component SE by the reported base coefficient.
            # The denominator's sampling uncertainty is not available in the
            # current public covariance contract, so label it explicitly.
            se[row, :] = np.sqrt(np.maximum(0.0, np.diag(V))) / abs(denom)
        elif denominator == "movement":
            # The movement is exactly sum_g delta_g, so the joint covariance
            # already contains everything needed for the ratio delta method.
            for g in range(G):
                grad = -d[g] * np.ones(G) / (denom * denom)
                grad[g] += 1.0 / denom
                se[row, g] = np.sqrt(max(0.0, float(grad @ V @ grad)))
    return {
        "coef": share,
        "se": se,
        "defined": defined,
        "denominator": denominator,
        "se_type": ("not_available_joint_base_covariance" if denominator == "base"
                    else ("fixed_base_denominator_scaling"
                          if denominator == "base_fixed" else
                          "joint_covariance_delta_method")),
        "units": "fraction",
        "tol": float(share_tol),
    }


def tidy(result, *, focal=None, include_intercept=False, include_total=True,
         include_full=True, conf_level=0.95, share=None, share_tol=1e-12):
    """Return publication-ready Gelbach rows without changing the estimator.

    The result is a list of dictionaries, so it can be passed directly to
    pandas, Polars or a CSV writer without adding a package dependency.
    ``share='base'`` reports the common descriptive convention
    ``delta / b_base`` but leaves its SE missing because the public result does
    not yet contain covariance with ``b_base``. ``share='base_fixed'``
    additionally reproduces the widespread fixed-denominator scaling
    convention, labelled as such. ``share='movement'`` uses the full joint
    covariance and the delta method for ``delta / sum(delta)``.
    Signed shares are never truncated or renormalized.

    Parameters
    ----------
    result : dictionary returned by :func:`decompose`.
    focal : optional X1 names, zero-based indices, or Boolean mask. If omitted,
        use the selector stored by :func:`decompose`.
    include_intercept : include the implicit ``_cons`` row.
    include_total : add a ``total_movement`` row.
    include_full : add a ``full_model_residual`` row; its SE is unavailable.
    conf_level : normal-approximation confidence level in (0, 1).
    share : None, ``'movement'``, ``'base'``, or ``'base_fixed'``.
    share_tol : nonnegative absolute threshold below which ratios are missing.

    Returns
    -------
    list of dictionaries with coefficient/component labels, estimates, SEs,
    confidence intervals and inference type; share fields are added only when
    requested. See ``xhdfe.help_text('gelbach')`` for the complete schema.
    """
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must lie strictly between zero and one")
    rows = _result_focal_indices(result, focal, include_intercept)
    zcrit = NormalDist().inv_cdf(0.5 + conf_level / 2.0)
    share_stats = None if share is None else _share_rows(result, share, share_tol)
    output = []
    p = len(result["b_base"])

    for row in rows:
        label = result["labels"][row]
        for g, name in enumerate(result["names"]):
            estimate = float(result["delta"][name]["coef"][row])
            std_error = float(result["delta"][name]["se"][row])
            item = {
                "coefficient": label,
                "component": name,
                "component_kind": result["group_kinds"][name],
                "estimate": estimate,
                "std_error": std_error,
                "conf_low": estimate - zcrit * std_error,
                "conf_high": estimate + zcrit * std_error,
                "conf_level": float(conf_level),
                "se_type": result["delta"][name]["se_type"],
            }
            if share_stats is not None:
                s = float(share_stats["coef"][row, g])
                sse = float(share_stats["se"][row, g])
                item.update({
                    "share": s,
                    "share_std_error": sse,
                    "share_conf_low": s - zcrit * sse,
                    "share_conf_high": s + zcrit * sse,
                    "share_defined": bool(share_stats["defined"][row]),
                    "share_denominator": share_stats["denominator"],
                    "share_se_type": share_stats["se_type"],
                    "share_units": share_stats["units"],
                    "share_tol": share_stats["tol"],
                })
            output.append(item)

        if include_total:
            estimate = float(result["total"]["coef"][row])
            std_error = float(result["total"]["se"][row])
            item = {
                "coefficient": label,
                "component": "total_movement",
                "component_kind": "total",
                "estimate": estimate,
                "std_error": std_error,
                "conf_low": estimate - zcrit * std_error,
                "conf_high": estimate + zcrit * std_error,
                "conf_level": float(conf_level),
                "se_type": result["total"]["se_type"],
            }
            if share_stats is not None:
                if share == "movement" and share_stats["defined"][row]:
                    sval, sse = 1.0, 0.0
                elif share in ("base", "base_fixed") and row < p and share_stats["defined"][row]:
                    denom = float(result["b_base"][row])
                    sval = estimate / denom
                    sse = (std_error / abs(denom) if share == "base_fixed"
                           else float("nan"))
                else:
                    sval = sse = float("nan")
                item.update({
                    "share": sval,
                    "share_std_error": sse,
                    "share_conf_low": sval - zcrit * sse,
                    "share_conf_high": sval + zcrit * sse,
                    "share_defined": bool(share_stats["defined"][row]),
                    "share_denominator": share_stats["denominator"],
                    "share_se_type": share_stats["se_type"],
                    "share_units": share_stats["units"],
                    "share_tol": share_stats["tol"],
                })
            output.append(item)

        if include_full and row < p:
            item = {
                "coefficient": label,
                "component": "full_model_residual",
                "component_kind": "full_model",
                "estimate": float(result["b_full"][row]),
                "std_error": float("nan"),
                "conf_low": float("nan"),
                "conf_high": float("nan"),
                "conf_level": float(conf_level),
                "se_type": "not_available_in_public_contract",
            }
            if share_stats is not None:
                if share in ("base", "base_fixed") and share_stats["defined"][row]:
                    sval = float(result["b_full"][row] / result["b_base"][row])
                else:
                    sval = float("nan")
                item.update({
                    "share": sval,
                    "share_std_error": float("nan"),
                    "share_conf_low": float("nan"),
                    "share_conf_high": float("nan"),
                    "share_defined": bool(share_stats["defined"][row]),
                    "share_denominator": share_stats["denominator"],
                    "share_se_type": "not_available_for_full_model_residual",
                    "share_units": share_stats["units"],
                    "share_tol": share_stats["tol"],
                })
            output.append(item)
    return output


def contrast(result, focal, groups, *, conf_level=0.95):
    """Estimate a linear contrast of Gelbach blocks from the joint covariance.

    ``groups`` may be a mapping ``{group_name: weight}``, a sequence of group
    names (all weights one), or a numeric sequence in ``result['names']``
    order. No model is re-estimated.

    ``focal`` must select exactly one X1 coefficient. The returned dictionary
    contains the complete weight mapping, estimate, joint-covariance SE,
    confidence interval and inference qualifier. Contrasts containing an FE
    component retain the conditional-FE label. See
    ``xhdfe.help_text('gelbach')`` for examples and limitations.
    """
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must lie strictly between zero and one")
    rows = _result_focal_indices(result, focal, False)
    if len(rows) != 1:
        raise ValueError("contrast requires exactly one focal coefficient")
    names = list(result["names"])
    if isinstance(groups, dict):
        unknown = [name for name in groups if name not in names]
        if unknown:
            raise ValueError(f"contrast contains unknown group(s): {unknown}")
        weights = np.asarray([float(groups.get(name, 0.0)) for name in names])
    else:
        values = list(groups)
        if values and all(isinstance(value, str) for value in values):
            unknown = [name for name in values if name not in names]
            if unknown:
                raise ValueError(f"contrast contains unknown group(s): {unknown}")
            if len(set(values)) != len(values):
                raise ValueError("contrast group names must be unique")
            weights = np.asarray([1.0 if name in values else 0.0 for name in names])
        else:
            weights = np.asarray(values, dtype=float)
            if weights.ndim != 1 or weights.size != len(names):
                raise ValueError("numeric contrast must have one weight per group")
    if not np.all(np.isfinite(weights)) or not np.any(weights):
        raise ValueError("contrast weights must be finite and not all zero")
    row = rows[0]
    d = np.asarray([result["delta"][name]["coef"][row] for name in names])
    V = _row_group_covariance(result, row)
    estimate = float(weights @ d)
    variance = float(weights @ V @ weights)
    std_error = np.sqrt(max(0.0, variance))
    zcrit = NormalDist().inv_cdf(0.5 + conf_level / 2.0)
    included = [name for name, weight in zip(names, weights) if weight != 0]
    has_fe = any(result["group_kinds"][name] == "fe" for name in included)
    return {
        "coefficient": result["labels"][row],
        "weights": dict(zip(names, weights.tolist())),
        "estimate": estimate,
        "std_error": float(std_error),
        "conf_low": estimate - zcrit * std_error,
        "conf_high": estimate + zcrit * std_error,
        "conf_level": float(conf_level),
        "se_type": ("joint_covariance_including_conditional_fe" if has_fe else
                    "joint_covariance"),
    }
