"""Gelbach (2016) conditional decomposition, HDFE-aware (M9B).

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

Inference reproduces b1x2 (validated to machine precision in
VALIDATE_GELBACH.py):

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

Note on interpretation: with two or more mobility components, the split of
the combined FE contribution into per-FE-dimension deltas depends on a
normalization convention (the component mean-shift documented above); the
total across FE dimensions and b_base - b_full are convention-invariant.
Within a single connected mobility component the x1-row split is identified.
"""

from __future__ import annotations

import numpy as np


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


def decompose(y, x1, x2_groups=None, fes=None, vce="unadjusted", cluster=None,
              gamma0=False, cov0=False, tol=1e-10, num_threads=0,
              weights=None, fweights=False):
    """Gelbach conditional decomposition of b_base - b_full.

    Parameters
    ----------
    y : (n,) outcome.
    x1 : (n, p) base covariates (no constant column; one is implicit).
    x2_groups : dict name -> (n,) or (n, q_g) observed covariate group(s).
    fes : dict name -> (n,) integer ids of absorbed fixed-effect dimensions.
    vce : 'unadjusted' (b1x2 default), 'robust', or 'cluster' (requires
        `cluster` ids). Matches b1x2's estimators exactly (see module doc).
    gamma0 : bool — aux-regression variance only (b1x2's gamma0).
    cov0 : bool — drop the robust cross terms (b1x2's cov0; no-op for
        vce='unadjusted').

    Returns
    -------
    dict with 'b_base', 'b_full', per-group 'delta' (contribution over
    [x1..., _cons] with 'se'), 'total' (with SE), 'cov', 'identity_gap',
    sample sizes and notes.
    """
    y = np.ascontiguousarray(y, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    if x1.ndim == 1:
        x1 = x1[:, None]
    n, p = x1.shape
    if y.shape[0] != n:
        raise ValueError("y and x1 must have the same number of rows")
    if vce not in ("unadjusted", "robust", "cluster"):
        raise ValueError("vce must be 'unadjusted', 'robust' or 'cluster'")
    if vce == "cluster":
        if cluster is None:
            raise ValueError("vce='cluster' requires cluster ids")
        cluster = np.asarray(cluster)
        _, ccodes = np.unique(cluster, return_inverse=True)
    else:
        ccodes = None
    x2_groups = dict(x2_groups or {})
    fes = dict(fes or {})
    if not x2_groups and not fes:
        raise ValueError("provide at least one x2 group or fixed-effect dimension")

    groups = []  # (name, kind, payload)
    x2_cols = []
    for name, arr in x2_groups.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] != n:
            raise ValueError(f"x2 group {name!r} has wrong length")
        groups.append((name, "x2", arr))
        x2_cols.append(arr)
    fe_lists = []
    for name, ids in fes.items():
        ids = np.asarray(ids)
        if ids.shape[0] != n:
            raise ValueError(f"fe {name!r} has wrong length")
        groups.append((name, "fe", len(fe_lists)))
        fe_lists.append(ids)

    core = _core()
    if not hasattr(core, "gelbach_decompose"):
        raise ImportError("the compiled xhdfe extension predates the Gelbach "
                          "module; rebuild the package")

    x2_sizes = [np.asarray(d).shape[1] for _, k, d in groups if k == "x2"]
    X2 = (np.hstack([np.asarray(d, dtype=float) for _, k, d in groups
                     if k == "x2"]) if x2_sizes else None)
    r = core.gelbach_decompose(y, x1, x2=X2, x2_group_sizes=x2_sizes,
                               fes=fe_lists if fe_lists else None,
                               cluster=(np.asarray(cluster) if ccodes is not None
                                        else None),
                               vce=vce, gamma0=bool(gamma0), cov0=bool(cov0),
                               num_threads=num_threads,
                               weights=(np.ascontiguousarray(weights, dtype=float)
                                        if weights is not None else None),
                               fweights=bool(fweights))

    p_ = x1.shape[1]
    k1 = p_ + 1
    delta = np.asarray(r["delta"])
    fullcov = np.asarray(r["cov"])
    # group order in the compiled core: x2 groups first, then FE dims — match
    # the caller's insertion order of `groups`.
    order = [g for g, (_, kind, _d) in enumerate(groups) if kind == "x2"] + \
            [g for g, (_, kind, _d) in enumerate(groups) if kind == "fe"]
    names = [groups[g][0] for g in order]
    labels = [f"x1_{v + 1}" for v in range(p_)] + ["_cons"]
    out = {
        "names": names,
        "labels": labels,
        "b_base": np.asarray(r["b_base"]),
        "b_full": np.asarray(r["b_full"]),
        "delta": {
            name: {"coef": delta[:, g],
                   "se": np.sqrt(np.diag(
                       fullcov[g * k1:(g + 1) * k1, g * k1:(g + 1) * k1]))}
            for g, name in enumerate(names)
        },
        "total": {"coef": np.asarray(r["total"]),
                  "se": np.sqrt(np.diag(np.asarray(r["total_cov"])))},
        "cov": fullcov,
        "identity_gap": float(r["identity_gap"]),
        "n_obs": int(r["n_obs"]),
        "df_full": float(r["df_full"]),
        "vce": vce,
        "gamma0": bool(gamma0),
        "notes": r["notes"],
        "converged": bool(r["converged"]),
    }
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
    return out
