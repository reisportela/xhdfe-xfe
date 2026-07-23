#!/usr/bin/env python3
"""Validation suite for the HDFE-aware Gelbach decomposition (M9B).

Checks `xhdfe.gelbach.decompose` against:
  (c) a hand-computed omitted-variables-bias decomposition (always);
  (d) an independent dense stacked-score oracle for base-model covariance,
      Cov(delta, b_base), Cov(total, b_base), gamma, G and df_base across
      iid/robust/cluster VCEs, weights, and designs with/without one FE;
  (a) Gelbach's b1x2 (Stata) exactly, no absorbed effects — deltas, totals,
      default and gamma0 SEs;                       [needs --b1x2-dir]
  (b) the strong HDFE test: firm effects as explicit LSDV dummies fed to
      b1x2 as one X2 group vs the absorbed xhdfe version — x1-row
      contributions and gamma0 SEs exact; the constant-row difference is
      verified to be exactly the fixed-effect normalization convention
      (isolated to the FE block).                   [needs --b1x2-dir]

Stata reads the exchange CSVs with `asdouble` (float default costs 1e-9).

Usage:
  python VALIDATE_GELBACH.py [--b1x2-dir DIR_WITH_b1x2.ado] [--stata stata-mp]
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np

FAILURES = []


def check(name, a, b, tol):
    d = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
    ok = d <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max|diff|={d:.2e} (tol {tol:g})")
    if not ok:
        FAILURES.append(name)


def check_condition(name, condition, detail=""):
    ok = bool(condition)
    suffix = f": {detail}" if detail else ""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{suffix}")
    if not ok:
        FAILURES.append(name)


def check_raises(name, fn, text):
    try:
        fn()
    except Exception as exc:  # validation intentionally spans Python/C++ errors
        ok = text in str(exc)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {type(exc).__name__}: {exc}")
        if not ok:
            FAILURES.append(name)
    else:
        print(f"[FAIL] {name}: no error raised")
        FAILURES.append(name)


def sim_a(seed=5, n=400):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=(n, 2))
    x2a = 0.5 * x1[:, 0:1] + rng.normal(size=(n, 2))
    x2b = -0.3 * x1[:, 1:2] + rng.normal(size=(n, 1))
    y = x1 @ [1.0, -0.5] + x2a @ [0.8, 0.2] + x2b @ [0.6] + rng.normal(size=n)
    return y, x1, x2a, x2b


def sim_b(seed=9, n=600, nf=12):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, nf, n)
    x1 = rng.normal(size=(n, 2)) + 0.4 * rng.normal(size=nf)[firm][:, None]
    z = 0.3 * x1[:, 0] + rng.normal(size=n)
    psi = rng.normal(0, 0.7, nf)
    y = x1 @ [1.0, -0.5] + 0.8 * z + psi[firm] + rng.normal(size=n)
    return y, x1, z, firm


def absorbed_target_fixture(seed=123, n_workers=80, periods=5):
    """Worker-invariant focal plus a within-worker identified X1 column."""
    rng = np.random.default_rng(seed)
    worker = np.repeat(np.arange(n_workers), periods)
    n = worker.size
    focal = rng.integers(0, 2, n_workers)[worker].astype(float)
    within = np.tile(np.arange(periods), n_workers) + rng.normal(scale=0.1, size=n)
    z = 0.3 * focal + 0.2 * within + rng.normal(size=n)
    y = (0.4 * focal + 0.1 * within + 0.7 * z +
         rng.normal(size=n_workers)[worker] + rng.normal(scale=0.4, size=n))
    return y, np.column_stack([focal, within]), z, worker


def near_fe_fixture(target_ratio, seed=20260723, n_groups=60, periods=20):
    """Construct an X1 column with an exact pre-fit within/total SS ratio."""
    rng = np.random.default_rng(seed)
    fe = np.repeat(np.arange(n_groups), periods)
    signal = rng.normal(size=n_groups)[fe]
    within_noise = rng.normal(size=fe.size)
    within_noise -= np.bincount(
        fe, weights=within_noise, minlength=n_groups)[fe] / periods
    signal_ss = float(signal @ signal)
    noise_ss = float(within_noise @ within_noise)
    scale = np.sqrt(
        target_ratio * signal_ss / (noise_ss * (1.0 - target_ratio))
    )
    focal = signal + scale * within_noise
    z = rng.normal(size=fe.size)
    y = (
        0.8 * focal + 0.5 * z + rng.normal(size=n_groups)[fe]
        + rng.normal(scale=0.4, size=fe.size)
    )
    return y, focal, z, fe


def absorbed_target_oracle(y, x1, z, fe, df_full, vce="cluster",
                           cluster=None, weights=None, fweights=False):
    """Direct constrained-LSDV oracle for absorbed target column 0.

    The constrained full model omits X1[:, 0], includes the remaining X1,
    X2=z and a complete dummy basis for the FE.  This is independent of the
    HDFE recovery path.  The FE contribution includes the full-model constant;
    that changes only its auxiliary intercept, so slope rows and all residual-
    based covariance blocks are directly comparable to the normalized HDFE
    result.
    """
    n = y.size
    _, fe_codes = np.unique(fe, return_inverse=True)
    n_fe = int(fe_codes.max()) + 1
    D = np.eye(n_fe)[fe_codes]
    active = x1[:, 1:]
    x2 = np.asarray(z, dtype=float).reshape(n, 1)
    A = np.column_stack([active, x2])
    p_score = active.shape[1]
    Kx = A.shape[1]

    if weights is None:
        w_raw = np.ones(n)
        wq = np.ones(n)
        sf = np.ones(n)
        n_eff = float(n)
    else:
        w_raw = np.asarray(weights, dtype=float)
        if fweights:
            wq = w_raw
            # A clustered row represents w copies before the cluster sum;
            # the ordinary robust meat instead contains w outer products.
            sf = w_raw if vce == "cluster" else np.sqrt(w_raw)
            n_eff = float(w_raw.sum())
        else:
            wq = w_raw * (n / w_raw.sum())
            sf = wq
            n_eff = float(n)

    Q = np.column_stack([A, D])
    sw = np.sqrt(wq)
    coef = np.linalg.lstsq(Q * sw[:, None], y * sw, rcond=None)[0]
    e = y - Q @ coef
    b2 = coef[p_score:Kx]

    X1t = np.column_stack([x1, np.ones(n)])
    k1 = X1t.shape[1]
    P = np.linalg.inv(X1t.T @ (wq[:, None] * X1t))
    H = [x2 @ b2, D @ coef[Kx:]]
    delta = [P @ X1t.T @ (wq * h) for h in H]
    vres = [h - X1t @ d for h, d in zip(H, delta)]

    # Weighted within transform of the actually estimated [X1_K, X2].
    group_w = np.bincount(fe_codes, weights=wq, minlength=n_fe)
    group_sum = np.zeros((n_fe, Kx))
    np.add.at(group_sum, fe_codes, wq[:, None] * A)
    W = A - group_sum[fe_codes] / group_w[fe_codes, None]
    Sw = np.linalg.inv(W.T @ (wq[:, None] * W))

    if vce == "unadjusted":
        R = np.column_stack([e, vres[0], vres[1]])
        R -= (wq @ R / wq.sum())[None, :]
        omega = R.T @ (wq[:, None] * R) / df_full
        cov = np.zeros((2 * k1, 2 * k1))
        for g in range(2):
            for h in range(2):
                cov[g * k1:(g + 1) * k1,
                    h * k1:(h + 1) * k1] = omega[g + 1, h + 1] * P
        gam = P @ X1t.T @ (wq[:, None] * x2)
        s2u_w = np.sum(wq * e ** 2) / df_full
        cov[:k1, :k1] += (
            gam @ (s2u_w * Sw[p_score:p_score + 1,
                               p_score:p_score + 1]) @ gam.T
        )
        return {
            "b_active": coef[:p_score],
            "delta": delta,
            "cov": cov,
        }

    score = np.column_stack([
        W * (sf * e)[:, None],
        X1t * (sf * vres[0])[:, None],
        X1t * (sf * vres[1])[:, None],
    ])
    if vce == "cluster":
        if cluster is None:
            raise ValueError("cluster ids required for clustered oracle")
        _, cl_codes = np.unique(cluster, return_inverse=True)
        n_clusters = int(cl_codes.max()) + 1
        cluster_sum = np.zeros((n_clusters, score.shape[1]))
        np.add.at(cluster_sum, cl_codes, score)
        meat = cluster_sum.T @ cluster_sum
        q_big = n_clusters / (n_clusters - 1.0)
        q_vu = ((n_eff - 1.0) / df_full *
                n_clusters / (n_clusters - 1.0))
    elif vce == "robust":
        meat = score.T @ score
        q_big = n_eff / (n_eff - 1.0)
        q_vu = n_eff / df_full
    else:
        raise ValueError(f"unknown oracle vce: {vce}")

    bread = np.zeros((Kx + 2 * k1, Kx + 2 * k1))
    bread[:Kx, :Kx] = Sw
    bread[Kx:Kx + k1, Kx:Kx + k1] = P
    bread[Kx + k1:, Kx + k1:] = P
    C = q_big * (bread @ meat @ bread)
    vu = q_vu * (Sw @ meat[:Kx, :Kx] @ Sw)
    cov = C[Kx:, Kx:].copy()
    gam = P @ X1t.T @ (wq[:, None] * x2)
    obs = cov[:k1, :k1]
    x2_offset = p_score
    obs += gam @ vu[x2_offset:x2_offset + 1,
                    x2_offset:x2_offset + 1] @ gam.T
    cross = gam @ C[x2_offset:x2_offset + 1, Kx:Kx + k1]
    obs += cross + cross.T
    cov[:k1, :k1] = obs
    return {
        "b_active": coef[:p_score],
        "delta": delta,
        "cov": cov,
    }


ORACLE_DO = r"""
adopath ++ "{b1x2_dir}"
import delimited using "{tdir}/gelb_a.csv", clear asdouble
gen long cl = mod(_n, 40)
local i = 0
foreach opt in "" "gamma0" "robust" "cluster(cl)" {{
    local ++i
    b1x2 y, x1all(x11 x12) x2all(a1 a2 b1) x2delta("A = a1 a2 : B = b1") `opt'
    matrix D = e(b)
    matrix V = e(V)
    local suf : word `i' of def g0 rob cl
    svmat double D, names(d_`suf'_)
    matrix SE = vecdiag(V)
    svmat double SE, names(v_`suf'_)
}}
keep d_* v_*
keep in 1
format d_* v_* %21.0g
export delimited using "{tdir}/gelb_a_oracle.csv", replace datafmt

import delimited using "{tdir}/gelb_w.csv", clear asdouble
gen long cl = mod(_n, 40)
local i = 0
foreach spec in "aw= " "aw=gamma0" "aw=robust" "aw=cluster(cl)" "fw= " "fw=robust" "fw=cluster(cl)" {{
    local ++i
    gettoken wt opt : spec, parse("=")
    gettoken eq opt : opt, parse("=")
    local wexp = cond("`wt'"=="aw", "[aweight=aw]", "[fweight=fw]")
    b1x2 y `wexp', x1all(x11 x12) x2all(a1 a2 b1) x2delta("A = a1 a2 : B = b1") `opt'
    matrix D = e(b)
    matrix V = e(V)
    local suf : word `i' of awdef awg0 awrob awcl fwdef fwrob fwcl
    svmat double D, names(d_`suf'_)
    matrix SE = vecdiag(V)
    svmat double SE, names(v_`suf'_)
}}
keep d_* v_*
keep in 1
format d_* v_* %21.0g
export delimited using "{tdir}/gelb_w_oracle.csv", replace datafmt

import delimited using "{tdir}/gelb_b.csv", clear asdouble
quietly tab firm, gen(fd)
local dums
forvalues k=2/12 {{
    local dums "`dums' fd`k'"
}}
local i = 0
foreach opt in "" "gamma0" "gamma0 robust" {{
    local ++i
    b1x2 y, x1all(x11 x12) x2all(z `dums') x2delta("OBS = z : FIRM = `dums'") `opt'
    matrix D = e(b)
    matrix V = e(V)
    local suf : word `i' of def g0 g0rob
    svmat double D, names(d_`suf'_)
    matrix SE = vecdiag(V)
    svmat double SE, names(v_`suf'_)
}}
keep d_* v_*
keep in 1
format d_* v_* %21.0g
export delimited using "{tdir}/gelb_b_oracle.csv", replace datafmt
"""


EXTERNAL_ABSORBED_ORACLE_DO = r"""
clear all
set more off
import delimited using "{data}", clear asdouble

quietly regress y focal experience
scalar o_base_focal = _b[focal]
scalar o_base_experience = _b[experience]
scalar o_base_cons = _b[_cons]
scalar o_se_unadjusted = _se[focal]

quietly regress y focal experience, vce(robust)
scalar o_se_robust = _se[focal]

quietly regress y focal experience, vce(cluster worker)
scalar o_se_cluster = _se[focal]

* A genuinely external constrained LSDV full model.  The absorbed target is
* absent by construction; all worker indicators are materialized and no
* xhdfe plugin, Python binding, or shared C++ code is loaded.
quietly regress y experience observed ibn.worker, noconstant
scalar o_full_focal = 0
scalar o_full_experience = _b[experience]
scalar o_observed_coef = _b[observed]
predict double xb_full, xb
generate double h_observed = scalar(o_observed_coef) * observed
generate double h_worker = xb_full - scalar(o_full_experience) * experience - h_observed

quietly regress h_observed focal experience
scalar o_dobs_focal = _b[focal]
scalar o_dobs_experience = _b[experience]
scalar o_dobs_cons = _b[_cons]

quietly regress h_worker focal experience
scalar o_dfe_focal = _b[focal]
scalar o_dfe_experience = _b[experience]
scalar o_total_focal = scalar(o_base_focal)
scalar o_total_experience = scalar(o_base_experience) - scalar(o_full_experience)

tempname posth
postfile `posth' double (base_focal base_experience base_cons ///
    se_unadjusted se_robust se_cluster full_focal full_experience ///
    dobs_focal dobs_experience dobs_cons dfe_focal dfe_experience ///
    total_focal total_experience) using "{outdta}", replace
post `posth' (scalar(o_base_focal)) (scalar(o_base_experience)) ///
    (scalar(o_base_cons)) (scalar(o_se_unadjusted)) ///
    (scalar(o_se_robust)) (scalar(o_se_cluster)) ///
    (scalar(o_full_focal)) (scalar(o_full_experience)) ///
    (scalar(o_dobs_focal)) (scalar(o_dobs_experience)) ///
    (scalar(o_dobs_cons)) (scalar(o_dfe_focal)) ///
    (scalar(o_dfe_experience)) (scalar(o_total_focal)) ///
    (scalar(o_total_experience))
postclose `posth'
use "{outdta}", clear
format _all %21.17g
export delimited using "{output}", replace datafmt
display as result "EXTERNAL ABSORBED-TARGET LSDV ORACLE PASSED"
"""


def base_ols_cov(y, x1, vce="unadjusted", cluster=None,
                 weights=None, fweights=False):
    """Independent dense base-OLS covariance on an unchanged sample."""
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.asarray(x1, dtype=float), np.ones(y.size)])
    if weights is None:
        wq = np.ones(y.size)
        sf = np.ones(y.size)
        n_eff = float(y.size)
    else:
        w = np.asarray(weights, dtype=float)
        if fweights:
            wq = w
            sf = w if vce == "cluster" else np.sqrt(w)
            n_eff = float(w.sum())
        else:
            wq = w * (y.size / w.sum())
            sf = wq
            n_eff = float(y.size)
    p = np.linalg.inv(x.T @ (wq[:, None] * x))
    b = p @ x.T @ (wq * y)
    u = y - x @ b
    df = n_eff - x.shape[1]
    if vce == "unadjusted":
        return b, (np.sum(wq * u ** 2) / df) * p
    score = x * (sf * u)[:, None]
    if vce == "cluster":
        if cluster is None:
            raise ValueError("cluster ids are required")
        _, codes = np.unique(cluster, return_inverse=True)
        ng = int(codes.max()) + 1
        sums = np.zeros((ng, x.shape[1]))
        np.add.at(sums, codes, score)
        meat = sums.T @ sums
        correction = ((n_eff - 1.0) / df) * ng / (ng - 1.0)
    else:
        meat = score.T @ score
        correction = n_eff / df
    return b, correction * (p @ meat @ p)


def stacked_base_cross_oracle(y, x1, x2_groups, *, fe=None,
                              vce="unadjusted", cluster=None,
                              weights=None, fweights=False, df_full=None,
                              gamma0=False):
    """Independent dense oracle for Cov(delta, b_base).

    This deliberately assembles explicit weighted LSDV normal equations and
    the augmented [full score, auxiliary scores, base score] system in NumPy.
    For robust/cluster VCEs, the direct and full-coefficient cross terms both
    receive b1x2's ``q_big`` multiplier.  ``q_vu`` belongs only to the
    component-component full-coefficient covariance and therefore never
    enters this cross block.  Under ``gamma0`` the full-coefficient cross term
    is omitted, while the direct auxiliary/base term remains.
    """
    y = np.asarray(y, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    if x1.ndim == 1:
        x1 = x1[:, None]
    groups = []
    group_sizes = []
    for value in x2_groups.values():
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        groups.append(arr)
        group_sizes.append(arr.shape[1])
    x2 = np.column_stack(groups)
    n, p = x1.shape
    q = x2.shape[1]
    x1t = np.column_stack([x1, np.ones(n)])
    k1 = p + 1

    if weights is None:
        wq = np.ones(n)
        sf = np.ones(n)
        n_eff = float(n)
    else:
        raw = np.asarray(weights, dtype=float)
        if fweights:
            wq = raw
            sf = raw if vce == "cluster" else np.sqrt(raw)
            n_eff = float(raw.sum())
        else:
            wq = raw * (n / raw.sum())
            sf = wq
            n_eff = float(n)
    sw = np.sqrt(wq)

    base_gram = x1t.T @ (wq[:, None] * x1t)
    P = np.linalg.inv(base_gram)
    b_base = np.linalg.solve(base_gram, x1t.T @ (wq * y))
    u_base = y - x1t @ b_base
    df_base = n_eff - k1

    if fe is None:
        full_design = np.column_stack([x1, x2, np.ones(n)])
        full_coef = np.linalg.lstsq(
            full_design * sw[:, None], y * sw, rcond=None)[0]
        full_resid = y - full_design @ full_coef
        b2 = full_coef[p:p + q]
        W = full_design
        Kx = p + q + 1
        fe_effect = None
    else:
        _, fe_codes = np.unique(np.asarray(fe), return_inverse=True)
        n_fe = int(fe_codes.max()) + 1
        dummies = np.eye(n_fe)[fe_codes, 1:]
        full_design = np.column_stack([x1, x2, np.ones(n), dummies])
        full_coef = np.linalg.lstsq(
            full_design * sw[:, None], y * sw, rcond=None)[0]
        full_resid = y - full_design @ full_coef
        b2 = full_coef[p:p + q]
        # The constant is part of the recovered FE contribution.  Its
        # normalization affects only the auxiliary intercept, not residuals.
        fe_effect = (
            np.full(n, full_coef[p + q])
            + dummies @ full_coef[p + q + 1:]
        )
        explicit = np.column_stack([x1, x2])
        group_w = np.bincount(fe_codes, weights=wq, minlength=n_fe)
        group_sum = np.zeros((n_fe, explicit.shape[1]))
        np.add.at(group_sum, fe_codes, wq[:, None] * explicit)
        W = explicit - group_sum[fe_codes] / group_w[fe_codes, None]
        Kx = p + q

    if df_full is None:
        df_full = n_eff - np.linalg.matrix_rank(
            full_design * sw[:, None])
    Sw = np.linalg.inv(W.T @ (wq[:, None] * W))

    effects = []
    aux_gamma = []
    cursor = 0
    for width in group_sizes:
        x2g = x2[:, cursor:cursor + width]
        effects.append(x2g @ b2[cursor:cursor + width])
        aux_gamma.append(P @ (x1t.T @ (wq[:, None] * x2g)))
        cursor += width
    if fe_effect is not None:
        effects.append(fe_effect)
        aux_gamma.append(None)

    delta = [P @ (x1t.T @ (wq * effect)) for effect in effects]
    vres = [effect - x1t @ d for effect, d in zip(effects, delta)]
    G = len(effects)

    _, base_cov = base_ols_cov(
        y, x1, vce=vce, cluster=cluster, weights=weights,
        fweights=fweights)
    cross = np.zeros((G * k1, k1))
    n_clusters = 0
    if vce == "unadjusted":
        centered_base = u_base - (wq @ u_base) / wq.sum()
        for g in range(G):
            cov_v_base = (
                np.sum(wq * vres[g] * centered_base) / df_full
            )
            cross[g * k1:(g + 1) * k1, :] = cov_v_base * P
    else:
        full_score = W * (sf * full_resid)[:, None]
        aux_scores = [
            x1t * (sf * vres_g)[:, None] for vres_g in vres
        ]
        base_score = x1t * (sf * u_base)[:, None]
        system_score = np.column_stack([full_score] + aux_scores)
        if vce == "cluster":
            if cluster is None:
                raise ValueError("cluster ids are required")
            _, cluster_codes = np.unique(
                np.asarray(cluster), return_inverse=True)
            n_clusters = int(cluster_codes.max()) + 1
            system_sums = np.zeros((n_clusters, system_score.shape[1]))
            base_sums = np.zeros((n_clusters, k1))
            np.add.at(system_sums, cluster_codes, system_score)
            np.add.at(base_sums, cluster_codes, base_score)
            score_cross = system_sums.T @ base_sums
            q_big = n_clusters / (n_clusters - 1.0)
        elif vce == "robust":
            score_cross = system_score.T @ base_score
            q_big = n_eff / (n_eff - 1.0)
        else:
            raise ValueError(f"unknown vce: {vce}")

        bread = np.zeros((Kx + G * k1, Kx + G * k1))
        bread[:Kx, :Kx] = Sw
        for g in range(G):
            bread[Kx + g * k1:Kx + (g + 1) * k1,
                  Kx + g * k1:Kx + (g + 1) * k1] = P
        system_base_cov = q_big * bread @ score_cross @ P
        cursor = 0
        for g, width in enumerate(group_sizes):
            block = system_base_cov[
                Kx + g * k1:Kx + (g + 1) * k1, :
            ].copy()
            if not gamma0:
                block += (
                    aux_gamma[g]
                    @ system_base_cov[
                        p + cursor:p + cursor + width, :
                    ]
                )
            cross[g * k1:(g + 1) * k1, :] = block
            cursor += width
        if fe_effect is not None:
            g = G - 1
            cross[g * k1:(g + 1) * k1, :] = system_base_cov[
                Kx + g * k1:Kx + (g + 1) * k1, :
            ]

    return {
        "base_cov": base_cov,
        "cov_delta_bbase": cross,
        "cov_total_bbase": sum(
            (cross[g * k1:(g + 1) * k1, :] for g in range(G)),
            np.zeros((k1, k1)),
        ),
        "gamma": [
            b2[sum(group_sizes[:g]):sum(group_sizes[:g + 1])].copy()
            for g in range(len(group_sizes))
        ],
        "df_base": df_base,
        "n_clusters": n_clusters,
    }


def run_external_absorbed_oracle(args, gb, y, x1, z, worker):
    """Validate the new estimand against Stata's built-in constrained LSDV."""
    mode = args.external_absorbed_oracle
    if mode == "skip":
        print("[SKIP] external absorbed-target LSDV oracle (explicitly skipped)")
        return
    stata = shutil.which(args.stata)
    if stata is None and os.path.isfile(args.stata):
        stata = os.path.abspath(args.stata)
    if stata is None:
        msg = f"Stata executable not found: {args.stata}"
        if mode == "require":
            print(f"[FAIL] external absorbed-target LSDV oracle: {msg}")
            FAILURES.append("external-absorbed-target-lsdv-oracle")
        else:
            print(f"[SKIP] external absorbed-target LSDV oracle: {msg}")
        return

    with tempfile.TemporaryDirectory(prefix="gelbach_external_lsdv_") as td:
        data = os.path.join(td, "fixture.csv")
        output = os.path.join(td, "oracle.csv")
        outdta = os.path.join(td, "oracle.dta")
        do = os.path.join(td, "external_absorbed_oracle.do")
        with open(data, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["y", "focal", "experience", "observed", "worker"])
            writer.writerows(zip(y, x1[:, 0], x1[:, 1], z, worker))
        with open(do, "w", encoding="utf-8") as fh:
            fh.write(EXTERNAL_ABSORBED_ORACLE_DO.format(
                data=data, output=output, outdta=outdta))
        try:
            subprocess.run([stata, "-q", "-b", "do", do], cwd=td,
                           check=True, timeout=420)
            with open(output, newline="", encoding="utf-8") as fh:
                oracle = next(csv.DictReader(fh))
            oracle = {name: float(value) for name, value in oracle.items()}
        except (OSError, subprocess.SubprocessError, StopIteration,
                ValueError) as exc:
            print(f"[FAIL] external absorbed-target LSDV oracle: {exc}")
            FAILURES.append("external-absorbed-target-lsdv-oracle")
            return

    got_cluster = gb.decompose(
        y, x1, {"OBS": z}, {"WORKER": worker}, vce="cluster",
        cluster=worker, absorbed_targets=[0])
    check("external-lsdv:b-base", got_cluster["b_base"],
          [oracle["base_focal"], oracle["base_experience"]], 1e-11)
    check("external-lsdv:b-full", got_cluster["b_full"],
          [oracle["full_focal"], oracle["full_experience"]], 1e-11)
    check("external-lsdv:delta-observed", got_cluster["delta"]["OBS"]["coef"],
          [oracle["dobs_focal"], oracle["dobs_experience"],
           oracle["dobs_cons"]], 1e-11)
    check("external-lsdv:delta-fe-slope-rows",
          got_cluster["delta"]["WORKER"]["coef"][:2],
          [oracle["dfe_focal"], oracle["dfe_experience"]], 1e-11)
    check("external-lsdv:total-x1", got_cluster["total"]["coef"][:2],
          [oracle["total_focal"], oracle["total_experience"]], 1e-11)
    for vce, key in (("unadjusted", "se_unadjusted"),
                     ("robust", "se_robust"),
                     ("cluster", "se_cluster")):
        kw = {"cluster": worker} if vce == "cluster" else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            got = gb.decompose(y, x1, {"OBS": z}, {"WORKER": worker},
                               vce=vce, absorbed_targets=[0], **kw)
        check(f"external-lsdv:target-total-se:{vce}",
              got["total"]["se"][0], oracle[key], 2e-11)


def unpack(row, prefix):
    d = [row[f"{prefix}_{k}"] for k in range(1, 10)]
    # b1x2 order: (x11, x12, _cons) x (group1, group2, __TC)
    return (np.array([d[0], d[3], d[6]]), np.array([d[1], d[4], d[7]]),
            np.array([d[2], d[5], d[8]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1x2-dir", default=None,
                    help="directory containing b1x2.ado (SSC download)")
    ap.add_argument("--stata", default="stata-mp")
    ap.add_argument("--module-dir", default=None,
                    help="directory containing the py_hdfe_v11 extension to validate")
    ap.add_argument(
        "--external-absorbed-oracle",
        choices=("require", "auto", "skip"), default="auto",
        help=("run the independent Stata regress/LSDV oracle for the absorbed-"
              "target estimand; recertification must use 'require'"),
    )
    args = ap.parse_args()

    if args.module_dir:
        sys.path.insert(0, os.path.abspath(args.module_dir))
        __import__("py_hdfe_v11")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import xhdfe.gelbach as gb
    from xhdfe import gelbach as gb_from
    if gb_from is gb:
        print("[PASS] packaging:from-xhdfe-import-gelbach")
    else:
        print("[FAIL] packaging:from-xhdfe-import-gelbach")
        FAILURES.append("packaging:from-xhdfe-import-gelbach")

    # ---- (c) hand-computed OVB --------------------------------------------
    y, x1, x2a, x2b = sim_a()
    n = y.size
    r = gb.decompose(y, x1, x2_groups={"A": x2a, "B": x2b})
    X1t = np.hstack([x1, np.ones((n, 1))])
    Xf = np.hstack([x1, x2a, x2b, np.ones((n, 1))])
    bf = np.linalg.lstsq(Xf, y, rcond=None)[0]
    bb = np.linalg.lstsq(X1t, y, rcond=None)[0]
    P = np.linalg.inv(X1t.T @ X1t)
    check("hand-ovb:delta_A", r["delta"]["A"]["coef"],
          P @ X1t.T @ (x2a @ bf[2:4]), 1e-12)
    check("hand-ovb:delta_B", r["delta"]["B"]["coef"],
          P @ X1t.T @ (x2b @ bf[4:5]), 1e-12)
    check("hand-ovb:total==b_base-b_full", r["total"]["coef"],
          bb - np.append(bf[:2], bf[-1]), 1e-12)
    check("hand-ovb:total-cov-public", r["total"]["cov"],
          r["total_cov"], 0.0)
    check("hand-ovb:identity_gap", r["identity_gap"], 0.0, 1e-12)
    check("api:tol-default-is-effective-historical-default", r["tol"], 1e-8, 0.0)
    if r["estimand"] != "coefficient_movement" or r["causal_interpretation"]:
        FAILURES.append("api:interpretation-metadata")
        print("[FAIL] api:interpretation-metadata")
    else:
        print("[PASS] api:interpretation-metadata")

    # Reporting/application layer: focal selection keeps every X1 column in
    # both models and must be numerically inert. Shares and contrasts are pure
    # functions of the already-certified delta/covariance objects.
    named = gb.decompose(
        y, x1, x2_groups={"A": x2a, "B": x2b},
        x1_names=["target", "common_control"], focal="target",
    )
    check("reporting:focal-is-numerically-inert:cov", named["cov"],
          r["cov"], 0.0)
    check("reporting:focal-is-numerically-inert:total",
          named["total"]["coef"], r["total"]["coef"], 0.0)
    if (named["focal_indices"] == [0] and
            named["focal_names"] == ["target"] and
            named["labels"][:2] == ["target", "common_control"]):
        print("[PASS] reporting:focal-metadata")
    else:
        print(f"[FAIL] reporting:focal-metadata: {named}")
        FAILURES.append("reporting:focal-metadata")

    movement_rows = gb.tidy(
        named, share="movement", include_total=False, include_full=False)
    component_shares = np.asarray([row["share"] for row in movement_rows])
    component_share_se = np.asarray(
        [row["share_std_error"] for row in movement_rows])
    check("reporting:movement-shares-sum-one", component_shares.sum(),
          1.0, 2e-14)
    if (np.all(np.isfinite(component_share_se)) and all(
            row["share_se_type"] == "joint_covariance_delta_method"
            for row in movement_rows)):
        print("[PASS] reporting:movement-share-joint-inference")
    else:
        print("[FAIL] reporting:movement-share-joint-inference")
        FAILURES.append("reporting:movement-share-joint-inference")

    base_rows = gb.tidy(
        named, share="base", include_total=False, include_full=False)
    base_denom = float(named["b_base"][0])
    base_var = float(named["base_cov"][0, 0])
    manual_base_se = []
    for g, group in enumerate(named["names"]):
        delta_g = float(named["delta"][group]["coef"][0])
        var_delta = float(named["cov"][g * 3, g * 3])
        cov_delta_base = float(named["cov_delta_bbase"][g * 3, 0])
        manual_base_se.append(np.sqrt(max(
            0.0,
            var_delta / base_denom ** 2
            + delta_g ** 2 * base_var / base_denom ** 4
            - 2.0 * delta_g * cov_delta_base / base_denom ** 3,
        )))
    check("reporting:base-share-points-unchanged",
          [row["share"] for row in base_rows],
          [named["delta"][group]["coef"][0] / base_denom
           for group in named["names"]], 0.0)
    check("reporting:base-share-joint-delta-method",
          [row["share_std_error"] for row in base_rows],
          manual_base_se, 2e-15)
    check_condition(
        "reporting:base-share-joint-inference-labelled",
        all(np.isfinite(row["share_std_error"]) for row in base_rows)
        and all(row["share_se_type"] ==
                "joint_base_covariance_delta_method" for row in base_rows),
    )

    with warnings.catch_warnings(record=True) as caught_undefined:
        warnings.simplefilter("always", RuntimeWarning)
        undefined_rows = gb.tidy(
            named, share="base", share_tol=1e100,
            include_total=False, include_full=False)
    undefined_messages = [
        str(item.message) for item in caught_undefined
        if issubclass(item.category, RuntimeWarning)
    ]
    check_condition(
        "reporting:undefined-base-share-warns-once",
        len(undefined_messages) == 1
        and "share denominator is undefined" in undefined_messages[0]
        and all(not row["share_defined"] for row in undefined_rows)
        and all(np.isnan(row["share"]) for row in undefined_rows),
        f"warnings={undefined_messages}",
    )

    fixed_rows = gb.tidy(
        named, share="base_fixed", include_total=False, include_full=False)
    check("reporting:fixed-base-convention-unchanged",
          [row["share_std_error"] for row in fixed_rows],
          [named["delta"][group]["se"][0] / abs(base_denom)
           for group in named["names"]], 0.0)
    check_condition(
        "reporting:fixed-base-convention-labelled",
        all(np.isfinite(row["share_std_error"]) for row in fixed_rows)
        and all(row["share_se_type"] == "fixed_base_denominator_scaling"
                for row in fixed_rows),
    )

    all_groups = gb.contrast(named, "target", ["A", "B"])
    check("reporting:all-group-contrast-is-total:estimate",
          all_groups["estimate"], named["total"]["coef"][0], 2e-14)
    check("reporting:all-group-contrast-is-total:se",
          all_groups["std_error"], named["total"]["se"][0], 2e-14)
    check_raises(
        "reporting:unknown-focal-fails",
        lambda: gb.decompose(y, x1, x2_groups={"A": x2a},
                             x1_names=["target", "common_control"],
                             focal="not_a_column"),
        "unknown x1 name",
    )
    check_raises(
        "reporting:unknown-contrast-group-fails",
        lambda: gb.contrast(named, "target", ["not_a_group"]),
        "unknown group",
    )

    # A common low-dimensional FE is represented by explicit base-model
    # indicators. This is the exact FWL/LSDV route for papers whose short and
    # long regressions both contain year/region effects; focal remains a
    # reporting selector and the added HDFE still uses native absorption.
    rng_common = np.random.default_rng(20260720)
    nc = 900
    year = rng_common.integers(0, 6, nc)
    firm = rng_common.integers(0, 45, nc)
    target = rng_common.normal(size=nc)
    common = 0.25 * target + rng_common.normal(size=nc)
    year_dummies = np.eye(6)[year, 1:]
    x1_common = np.column_stack([target, common, year_dummies])
    z_common = np.column_stack([
        0.35 * target + rng_common.normal(size=nc),
        -0.20 * target + 0.15 * common + rng_common.normal(size=nc),
    ])
    year_effect = rng_common.normal(scale=0.3, size=6)[year]
    firm_effect = rng_common.normal(scale=0.6, size=45)[firm]
    y_common = (0.8 * target + 0.25 * common + year_effect +
                z_common @ np.array([0.7, -0.4]) + firm_effect +
                rng_common.normal(size=nc))
    common_names = ["target", "common_control"] + [
        f"year_{value}" for value in range(1, 6)]
    common_fit = gb.decompose(
        y_common, x1_common, {"observed": z_common}, {"firm": firm},
        x1_names=common_names, focal="target",
    )
    firm_dummies = np.eye(45)[firm, 1:]
    X1_common = np.column_stack([x1_common, np.ones(nc)])
    Xfull_common = np.column_stack(
        [x1_common, z_common, firm_dummies, np.ones(nc)])
    bbase_common = np.linalg.lstsq(X1_common, y_common, rcond=None)[0]
    bfull_common = np.linalg.lstsq(Xfull_common, y_common, rcond=None)[0]
    p_common = x1_common.shape[1]
    P_common = np.linalg.inv(X1_common.T @ X1_common)
    observed_oracle = P_common @ X1_common.T @ (
        z_common @ bfull_common[p_common:p_common + 2])
    fe_oracle = P_common @ X1_common.T @ (
        firm_dummies @ bfull_common[p_common + 2:-1])
    check("reporting:common-controls:base-target-lsdv",
          common_fit["b_base"][0], bbase_common[0], 2e-12)
    check("reporting:common-controls:full-target-lsdv",
          common_fit["b_full"][0], bfull_common[0], 2e-12)
    check("reporting:common-controls:observed-block-lsdv",
          common_fit["delta"]["observed"]["coef"][:-1],
          observed_oracle[:-1], 3e-12)
    check("reporting:common-controls:added-hdfe-lsdv",
          common_fit["delta"]["firm"]["coef"][:-1],
          fe_oracle[:-1], 3e-12)

    # Independent stacked-score oracle for the additive Gate-2 contract.
    # This covers every VCE, aweights/fweights, designs with/without one FE,
    # and gamma0's deliberate removal of only the full-coefficient cross term.
    rng_cross = np.random.default_rng(20260723)
    ng = 480
    fe_cross = np.repeat(np.arange(12), ng // 12)
    cluster_cross = np.arange(ng) % 40
    x1_cross = np.column_stack([
        rng_cross.normal(size=ng) + 0.2 * rng_cross.normal(size=12)[fe_cross],
        rng_cross.normal(size=ng),
    ])
    x2a_cross = np.column_stack([
        0.35 * x1_cross[:, 0] + rng_cross.normal(size=ng),
        -0.20 * x1_cross[:, 1] + rng_cross.normal(size=ng),
    ])
    x2b_cross = (
        0.15 * x1_cross[:, 0] - 0.10 * x1_cross[:, 1]
        + rng_cross.normal(size=ng)
    )
    y_cross = (
        x1_cross @ np.array([0.9, -0.4])
        + x2a_cross @ np.array([0.7, -0.3])
        + 0.5 * x2b_cross
        + rng_cross.normal(scale=0.6, size=12)[fe_cross]
        + rng_cross.normal(scale=0.5, size=ng)
    )
    aw_cross = rng_cross.uniform(0.25, 2.75, size=ng)
    fw_cross = rng_cross.integers(1, 5, size=ng)
    x2_cross = {"A": x2a_cross, "B": x2b_cross}
    for design, fe_arg in (("no-fe", None), ("one-fe", fe_cross)):
        fes_arg = None if fe_arg is None else {"firm": fe_arg}
        for vce_cross in ("unadjusted", "robust", "cluster"):
            vce_args = (
                {"cluster": cluster_cross}
                if vce_cross == "cluster" else {}
            )
            for weight_tag, weight_args in (
                    ("unweighted", {}),
                    ("aweight", {"weights": aw_cross}),
                    ("fweight", {"weights": fw_cross,
                                 "fweights": True})):
                gamma_modes = (False, True) if weight_tag == "unweighted" else (False,)
                for gamma0_cross in gamma_modes:
                    got_cross = gb.decompose(
                        y_cross, x1_cross, x2_cross, fes=fes_arg,
                        vce=vce_cross, gamma0=gamma0_cross,
                        **vce_args, **weight_args)
                    oracle_cross = stacked_base_cross_oracle(
                        y_cross, x1_cross, x2_cross, fe=fe_arg,
                        vce=vce_cross, gamma0=gamma0_cross,
                        df_full=got_cross["df_full"],
                        **vce_args, **weight_args)
                    stem = (
                        f"base-cross:{design}:{vce_cross}:{weight_tag}:"
                        f"{'gamma0' if gamma0_cross else 'default'}"
                    )
                    check(f"{stem}:base-cov", got_cross["base_cov"],
                          oracle_cross["base_cov"], 1e-14)
                    check(f"{stem}:cov-delta-base",
                          got_cross["cov_delta_bbase"],
                          oracle_cross["cov_delta_bbase"], 1e-14)
                    check(f"{stem}:cov-total-base",
                          got_cross["cov_total_bbase"],
                          oracle_cross["cov_total_bbase"], 1e-14)
                    for group_no, group_name in enumerate(("A", "B")):
                        check(f"{stem}:gamma:{group_name}",
                              got_cross["gamma"][group_name],
                              oracle_cross["gamma"][group_no], 1e-14)
                    check(f"{stem}:df-base", got_cross["df_base"],
                          oracle_cross["df_base"], 0.0)
                    check(f"{stem}:cluster-count",
                          got_cross["n_clusters"],
                          oracle_cross["n_clusters"], 0.0)
                    if (weight_tag == "unweighted"
                            and not gamma0_cross
                            and vce_cross != "unadjusted"):
                        cov0_cross = gb.decompose(
                            y_cross, x1_cross, x2_cross, fes=fes_arg,
                            vce=vce_cross, cov0=True, **vce_args)
                        check(
                            f"{stem}:cov0-does-not-change-base-cross",
                            cov0_cross["cov_delta_bbase"],
                            got_cross["cov_delta_bbase"], 0.0)

    check_raises(
        "guard:duplicate-block-name",
        lambda: gb.decompose(y, x1, x2_groups={"same": x2a},
                             fes={"same": np.arange(n) % 20}),
        "names must be unique",
    )
    check_raises(
        "guard:rank-deficient-duplicate-x2",
        lambda: gb.decompose(y, x1, x2_groups={"A": x2a[:, 0],
                                               "B": x2a[:, 0]}),
        "rank deficient",
    )
    check_raises(
        "guard:invalid-tol",
        lambda: gb.decompose(y, x1, x2_groups={"A": x2a}, tol=0),
        "strictly positive",
    )
    check_raises(
        "guard:one-cluster",
        lambda: gb.decompose(y, x1, x2_groups={"A": x2a}, vce="cluster",
                             cluster=np.ones(n, dtype=int)),
        "at least two clusters",
    )
    check_raises(
        "guard:empty-x1-python",
        lambda: gb.decompose(y, np.empty((n, 0)), x2_groups={"A": x2a}),
        "at least one focal column",
    )
    core = gb._core()
    check_raises(
        "guard:empty-x1-release-core",
        lambda: core.gelbach_decompose(
            y, np.empty((n, 0)), x2=x2a[:, :1], x2_group_sizes=[1],
            fes=[], cluster=None, vce="unadjusted", gamma0=False,
            cov0=False, tol=1e-8, num_threads=0, weights=None,
            fweights=False, absorbed_x1=[], gpu=False),
        "at least one focal column",
    )

    # Saturation is a catchable input error, never converged output carrying
    # NaN/Inf inference.  Exercise both the public wrapper and Release core,
    # then prove that the process remains usable.
    rng_sat = np.random.default_rng(1707)
    n_sat = 5
    x1_sat = rng_sat.normal(size=n_sat)
    x2_sat = rng_sat.normal(size=(n_sat, 3))
    y_sat = rng_sat.normal(size=n_sat)
    check_raises(
        "guard:saturated-full-model-python",
        lambda: gb.decompose(y_sat, x1_sat, {"saturated": x2_sat}),
        "df_full must be positive",
    )
    check_raises(
        "guard:saturated-full-model-release-core",
        lambda: core.gelbach_decompose(
            y_sat, x1_sat[:, None], x2=x2_sat, x2_group_sizes=[3],
            fes=[], cluster=None, vce="unadjusted", gamma0=False,
            cov0=False, tol=1e-8, num_threads=0, weights=None,
            fweights=False, absorbed_x1=[], gpu=False),
        "df_full must be positive",
    )
    post_sat = gb.decompose(y, x1, x2_groups={"A": x2a})
    check_condition(
        "guard:saturated-error-process-stays-alive",
        post_sat["converged"] and np.all(np.isfinite(post_sat["cov"])),
    )

    # F-01 warning-band straddle.  The construction targets the exact
    # within/total SS ratio, while the assertions use the ratio returned by
    # the backend classifier (the single source of truth).
    y_near, x_near, z_near, fe_near = near_fe_fixture(4e-9)
    with warnings.catch_warnings(record=True) as caught_near:
        warnings.simplefilter("always", RuntimeWarning)
        near = gb.decompose(
            y_near, x_near, {"observed": z_near}, {"firm": fe_near})
    near_messages = [str(item.message) for item in caught_near]
    near_note = "near-FE-collinear focal"
    check_condition(
        "diagnostic:near-fe-band-warns-and-marks",
        near["converged"]
        and near["fe_collinear_ss_ratio_tol"]
        < near["x1_fe_collinear_ratio"][0]
        <= near["near_fe_collinear_ss_ratio_warn_upper"]
        and near["x1_near_collinear_mask"] == [True]
        and near_note in near["notes"]
        and any(near_note in message for message in near_messages),
        (f"ratio={near['x1_fe_collinear_ratio'][0]:.3e}, "
         f"notes={near['notes']!r}"),
    )

    y_far, x_far, z_far, fe_far = near_fe_fixture(2e-2, seed=20260724)
    with warnings.catch_warnings(record=True) as caught_far:
        warnings.simplefilter("always", RuntimeWarning)
        far = gb.decompose(
            y_far, x_far, {"observed": z_far}, {"firm": fe_far})
    check_condition(
        "diagnostic:well-identified-focal-is-silent",
        far["converged"]
        and far["x1_fe_collinear_ratio"][0] >= 1e-2
        and far["x1_near_collinear_mask"] == [False]
        and near_note not in far["notes"]
        and not any(near_note in str(item.message) for item in caught_far),
        f"ratio={far['x1_fe_collinear_ratio'][0]:.3e}",
    )

    y_abs, x_abs, z_abs, fe_abs = near_fe_fixture(0.0, seed=20260725)
    absorbed_side = gb.decompose(
        y_abs, x_abs, {"observed": z_abs}, {"firm": fe_abs},
        vce="cluster", cluster=fe_abs, absorbed_targets=[0])
    check_condition(
        "diagnostic:absorbed-side-classification-unchanged",
        absorbed_side["absorbed_mask"] == [True]
        and absorbed_side["x1_fe_collinear_ratio"][0]
        <= absorbed_side["fe_collinear_ss_ratio_tol"]
        and absorbed_side["x1_near_collinear_mask"] == [False],
        f"ratio={absorbed_side['x1_fe_collinear_ratio'][0]:.3e}",
    )

    # G is computed on the retained sample; the warning threshold is
    # documented and additive.  A 3-cluster design warns, a 50-cluster design
    # remains silent.
    cluster3 = np.arange(n) % 3
    with warnings.catch_warnings(record=True) as caught_g3:
        warnings.simplefilter("always", RuntimeWarning)
        g3 = gb.decompose(
            y, x1, {"A": x2a}, vce="cluster", cluster=cluster3)
    few_note = "few clusters (G < 30)"
    check_condition(
        "diagnostic:few-cluster-warning-and-count",
        g3["n_clusters"] == np.unique(cluster3).size
        and g3["few_cluster_warning_threshold"] == 30
        and few_note in g3["notes"]
        and any(few_note in str(item.message) for item in caught_g3),
        f"G={g3['n_clusters']}, notes={g3['notes']!r}",
    )
    cluster50 = np.arange(n) % 50
    with warnings.catch_warnings(record=True) as caught_g50:
        warnings.simplefilter("always", RuntimeWarning)
        g50 = gb.decompose(
            y, x1, {"A": x2a}, vce="cluster", cluster=cluster50)
    check_condition(
        "diagnostic:many-cluster-design-is-silent",
        g50["n_clusters"] == np.unique(cluster50).size
        and few_note not in g50["notes"]
        and not any(few_note in str(item.message) for item in caught_g50),
        f"G={g50['n_clusters']}",
    )

    # Severe within-block collinearity is not necessarily rank deficiency:
    # the decomposition identity can converge while the split SE is highly
    # tolerance/ISA sensitive.  Keep the values, but require a fail-loud
    # diagnostic on the audited GEL_XBLOCK design.
    rngc = np.random.default_rng(20260714)
    nc = 800
    firmc = np.repeat(np.arange(40), 20)
    yearc = np.clip(firmc // 5 + rngc.choice([-1, 0, 0, 0, 1], nc), 0, 7)
    s1c = rngc.normal(size=nc)
    s2c = s1c + 1e-6 * rngc.normal(size=nc)
    tenc = rngc.normal(size=nc)
    expc = 0.7 * tenc + 0.7 * rngc.normal(size=nc)
    fec = 0.05 * np.arange(40) + rngc.normal(0, 0.2, 40)
    yec = rngc.normal(0, 0.3, 8)
    x1c = np.column_stack([rngc.normal(size=nc) + 0.3 * fec[firmc],
                           rngc.normal(size=nc)])
    clc = rngc.integers(1, 26, nc)
    yc = (1.0 + 0.8 * x1c[:, 0] - 0.5 * x1c[:, 1] + 0.3 * s1c +
          0.2 * s2c + 0.4 * tenc + 0.1 * expc + fec[firmc] + yec[yearc] +
          rngc.normal(0, 0.4, nc))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        ill = gb.decompose(
            yc, x1c,
            {"skill": np.column_stack([s1c, s2c]),
             "job": np.column_stack([tenc, expc])},
            {"firm": firmc, "year": yearc}, vce="cluster", cluster=clc,
            tol=1e-8, num_threads=8)
    cond_note = "x2 group 1 is severely ill-conditioned"
    if (ill["converged"] and cond_note in ill["notes"] and
            any(cond_note in str(w.message) for w in caught)):
        print("[PASS] diagnostic:near-collinear-x2-block-is-audible")
    else:
        print(f"[FAIL] diagnostic:near-collinear-x2-block-is-audible: "
              f"converged={ill['converged']} notes={ill['notes']!r}")
        FAILURES.append("diagnostic:near-collinear-x2-block-is-audible")

    # `tol` used to be accepted by the Python wrapper but silently omitted
    # from the compiled call. A slow-mode two-FE graph makes the setting
    # observably consequential: the deliberately loose solve is rejected by
    # the identity/convergence gate while the tight solve certifies.
    rngt = np.random.default_rng(8)
    mt = 200
    wt = np.repeat(np.arange(mt), 3)
    ft = np.column_stack([np.arange(mt), np.arange(mt) + 1,
                          rngt.integers(0, mt + 1, mt)]).ravel()
    nt = wt.size
    xt = rngt.normal(size=nt)
    zt = 0.3 * xt + rngt.normal(size=nt)
    yt = (0.7 * xt + 0.5 * zt + rngt.normal(size=mt)[wt] +
          rngt.normal(size=mt + 1)[ft] + rngt.normal(scale=0.1, size=nt))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        loose = gb.decompose(yt, xt, {"z": zt}, {"worker": wt, "firm": ft},
                             tol=1e-2)
        tight = gb.decompose(yt, xt, {"z": zt}, {"worker": wt, "firm": ft},
                             tol=1e-12)
    tol_effect = abs(float(loose["b_full"][0] - tight["b_full"][0]))
    if (not loose["converged"] and tight["converged"] and tol_effect > 1e-8):
        print(f"[PASS] api:tol-is-active: |b_loose-b_tight|={tol_effect:.2e}")
    else:
        print(f"[FAIL] api:tol-is-active: |b_loose-b_tight|={tol_effect:.2e}, "
              f"converged=({loose['converged']},{tight['converged']})")
        FAILURES.append("api:tol-is-active")

    # ---- absorbed-target allocation (distinct opt-in estimand) -----------
    ya, x1a, za, workera = absorbed_target_fixture()
    check_raises(
        "absorbed-target:standard-still-fails-rank",
        lambda: gb.decompose(ya, x1a, {"OBS": za}, {"WORKER": workera}),
        "rank deficient",
    )
    ra = gb.decompose(
        ya, x1a, {"OBS": za}, {"WORKER": workera},
        vce="cluster", cluster=workera, absorbed_targets=[0])
    if (ra["estimand"] == "absorbed_target_allocation" and
            ra["identity_status"] == "exact_ols_constrained" and
            ra["b_full_status"] == ["imposed_zero", "estimated"] and
            ra["focal_status"] == ["absorbed", "identified"] and
            ra["absorbed_targets"] == [0] and
            ra["absorbed_mask"] == [True, False] and
            ra["total_se_type"] == "target_exact_base_vce_mixed_components" and
            ra["inference_status"] == "clustered_at_absorbing_fe" and
            ra["absorbed_target_inference_valid"] and
            ra["absorbing_fe_index"] == 0 and
            ra["fe_collinear_ss_ratio_tol"] == 1e-9):
        print("[PASS] absorbed-target:identification-metadata")
    else:
        print(f"[FAIL] absorbed-target:identification-metadata: {ra}")
        FAILURES.append("absorbed-target:identification-metadata")
    check("absorbed-target:b_full-imposed-zero", ra["b_full"][0], 0.0, 0.0)
    check("absorbed-target:identity", ra["identity_gap"], 0.0, 1e-11)
    check("absorbed-target:target-total-is-base-minus-zero",
          ra["total"]["coef"][0], ra["b_base"][0], 1e-11)
    check("absorbed-target:total-base-cross-is-base-vce",
          ra["cov_total_bbase"][0, :], ra["base_cov"][0, :], 0.0)
    check("absorbed-target:n-input", ra["n_obs_input"], ya.size, 0.0)
    check("absorbed-target:n-singletons", ra["n_singletons_dropped"], 0, 0.0)
    absorbed_share_total = [
        row for row in gb.tidy(
            ra, focal=0, share="base", include_full=False)
        if row["component"] == "total_movement"
    ][0]
    check("absorbed-target:total-base-share-floating-identity",
          absorbed_share_total["share"], 1.0,
          64.0 * np.finfo(float).eps)
    check("absorbed-target:total-base-share-numerical-zero-se",
          absorbed_share_total["share_std_error"], 0.0,
          2.0 * np.sqrt(np.finfo(float).eps))
    for option, kwargs in (("gamma0", {"gamma0": True}),
                           ("cov0", {"cov0": True})):
        altered = gb.decompose(
            ya, x1a, {"OBS": za}, {"WORKER": workera},
            vce="cluster", cluster=workera, absorbed_targets=[0], **kwargs)
        check(f"absorbed-target:{option}:target-total-vce-invariant",
              altered["total_cov"][0, 0], ra["total_cov"][0, 0], 0.0)

    crossed = np.arange(ya.size) % 17
    with warnings.catch_warnings(record=True) as caught_abs:
        warnings.simplefilter("always", RuntimeWarning)
        r_crossed = gb.decompose(
            ya, x1a, {"OBS": za}, {"WORKER": workera},
            vce="cluster", cluster=crossed, absorbed_targets=[0])
    if (not r_crossed["absorbed_target_inference_valid"] and
            r_crossed["absorbing_fe_index"] == -1 and
            r_crossed["inference_status"] ==
            "warning_unsupported_vce_or_cluster" and
            "WARNING:" in r_crossed["notes"] and caught_abs):
        print("[PASS] absorbed-target:crossed-cluster-warning-is-audible")
    else:
        print("[FAIL] absorbed-target:crossed-cluster-warning-is-audible")
        FAILURES.append("absorbed-target:crossed-cluster-warning-is-audible")

    # Strong oracle: constrained explicit worker dummies plus independently
    # assembled iid/robust/cluster covariance systems. Exercise unweighted,
    # aweight and the frequency-weight score conventions.
    rng_abs = np.random.default_rng(20260719)
    aw_abs = rng_abs.uniform(0.3, 2.5, ya.size)
    fw_abs = rng_abs.integers(1, 5, ya.size)
    for vce_abs in ("unadjusted", "robust", "cluster"):
        for tag, kwargs in [
                ("unweighted", {}),
                ("aweight", {"weights": aw_abs}),
                ("fweight", {"weights": fw_abs, "fweights": True})]:
            vce_kwargs = ({"cluster": workera}
                          if vce_abs == "cluster" else {})
            got = gb.decompose(
                ya, x1a, {"OBS": za}, {"WORKER": workera},
                vce=vce_abs, absorbed_targets=np.array([True, False]),
                **vce_kwargs, **kwargs)
            oracle = absorbed_target_oracle(
                ya, x1a, za, workera, got["df_full"], vce=vce_abs,
                cluster=workera if vce_abs == "cluster" else None, **kwargs)
            stem = f"absorbed-target:{vce_abs}:{tag}"
            check(f"{stem}:b-active", got["b_full"][1:],
                  oracle["b_active"], 1e-11)
            check(f"{stem}:delta-observed",
                  got["delta"]["OBS"]["coef"], oracle["delta"][0], 1e-11)
            check(f"{stem}:delta-fe-slope-rows",
                  got["delta"]["WORKER"]["coef"][:-1],
                  oracle["delta"][1][:-1], 1e-11)
            check(f"{stem}:cov", got["cov"], oracle["cov"], 5e-11)
            _, base_cov = base_ols_cov(
                ya, x1a, vce=vce_abs,
                cluster=workera if vce_abs == "cluster" else None,
                **kwargs)
            check(f"{stem}:target-total-variance-is-base",
                  got["total_cov"][0, 0], base_cov[0, 0], 2e-12)
            expected_n = (int(np.sum(kwargs["weights"]))
                          if kwargs.get("fweights") else ya.size)
            check(f"{stem}:reported-row-count", got["n_obs"], ya.size, 0.0)
            check(f"{stem}:reported-effective-n", got["n_obs_effective"],
                  expected_n, 0.0)

    # Edge cell: every X1 target absorbed and no observed X2 leaves a valid
    # zero-column full-score block; robust/cluster inference must stay finite.
    all_abs = gb.decompose(
        ya, x1a[:, 0], fes={"WORKER": workera}, vce="cluster",
        cluster=workera, absorbed_targets=[0])
    if (all_abs["converged"] and np.all(np.isfinite(all_abs["cov"])) and
            all_abs["b_full_status"] == ["imposed_zero"]):
        print("[PASS] absorbed-target:all-x1-absorbed-zero-score-block")
    else:
        print("[FAIL] absorbed-target:all-x1-absorbed-zero-score-block")
        FAILURES.append("absorbed-target:all-x1-absorbed-zero-score-block")

    check_raises(
        "absorbed-target:requires-fe",
        lambda: gb.decompose(ya, x1a, {"OBS": za}, absorbed_targets=[0]),
        "requires at least one absorbed FE",
    )
    check_raises(
        "absorbed-target:declared-column-must-be-absorbed",
        lambda: gb.decompose(ya, x1a, {"OBS": za}, {"WORKER": workera},
                             absorbed_targets=[0, 1]),
        "must be omitted specifically",
    )

    run_external_absorbed_oracle(args, gb, ya, x1a, za, workera)

    if not args.b1x2_dir:
        print("[SKIP] b1x2 oracle checks (pass --b1x2-dir)")
    else:
        import pandas as pd
        with tempfile.TemporaryDirectory() as td:
            pd.DataFrame({"y": y, "x11": x1[:, 0], "x12": x1[:, 1],
                          "a1": x2a[:, 0], "a2": x2a[:, 1],
                          "b1": x2b[:, 0]}).to_csv(
                os.path.join(td, "gelb_a.csv"), index=False)
            yb, x1b, z, firm = sim_b()
            pd.DataFrame({"y": yb, "x11": x1b[:, 0], "x12": x1b[:, 1],
                          "z": z, "firm": firm}).to_csv(
                os.path.join(td, "gelb_b.csv"), index=False)
            rngw = np.random.default_rng(77)
            aw = rngw.uniform(0.2, 3.0, n)
            fw = rngw.integers(1, 5, n)
            pd.DataFrame({"y": y, "x11": x1[:, 0], "x12": x1[:, 1],
                          "a1": x2a[:, 0], "a2": x2a[:, 1], "b1": x2b[:, 0],
                          "aw": aw, "fw": fw}).to_csv(
                os.path.join(td, "gelb_w.csv"), index=False)
            do = os.path.join(td, "oracle.do")
            open(do, "w").write(ORACLE_DO.format(
                b1x2_dir=os.path.abspath(args.b1x2_dir), tdir=td))
            subprocess.run([args.stata, "-q", "-b", "do", do], cwd=td,
                           check=True, timeout=600)
            oa = pd.read_csv(os.path.join(td, "gelb_a_oracle.csv")).iloc[0]
            ob = pd.read_csv(os.path.join(td, "gelb_b_oracle.csv")).iloc[0]
            ow = pd.read_csv(os.path.join(td, "gelb_w_oracle.csv")).iloc[0]

        # (a) no absorbed effects: everything must match exactly
        rg0 = gb.decompose(y, x1, x2_groups={"A": x2a, "B": x2b}, gamma0=True)
        A_o, B_o, TC_o = unpack(oa, "d_def")
        vA, vB, vTC = unpack(oa, "v_def")
        gA, gB, _ = unpack(oa, "v_g0")
        check("b1x2:a:delta_A", r["delta"]["A"]["coef"], A_o, 1e-11)
        check("b1x2:a:delta_B", r["delta"]["B"]["coef"], B_o, 1e-11)
        check("b1x2:a:total", r["total"]["coef"], TC_o, 1e-11)
        check("b1x2:a:se_A_default", r["delta"]["A"]["se"], np.sqrt(vA), 1e-11)
        check("b1x2:a:se_B_default", r["delta"]["B"]["se"], np.sqrt(vB), 1e-11)
        check("b1x2:a:se_total_default", r["total"]["se"], np.sqrt(vTC), 1e-11)
        check("b1x2:a:se_A_gamma0", rg0["delta"]["A"]["se"], np.sqrt(gA), 1e-11)
        check("b1x2:a:se_B_gamma0", rg0["delta"]["B"]["se"], np.sqrt(gB), 1e-11)

        # robust and cluster (b1x2 stacked _robust, exact multipliers)
        rr = gb.decompose(y, x1, x2_groups={"A": x2a, "B": x2b}, vce="robust")
        rA, rB, rT = unpack(oa, "v_rob")
        check("b1x2:a:se_A_robust", rr["delta"]["A"]["se"], np.sqrt(rA), 1e-11)
        check("b1x2:a:se_B_robust", rr["delta"]["B"]["se"], np.sqrt(rB), 1e-11)
        check("b1x2:a:se_total_robust", rr["total"]["se"], np.sqrt(rT), 1e-11)
        cl = np.arange(n) % 40
        rc = gb.decompose(y, x1, x2_groups={"A": x2a, "B": x2b},
                          vce="cluster", cluster=cl)
        cA, cB, cT = unpack(oa, "v_cl")
        check("b1x2:a:se_A_cluster", rc["delta"]["A"]["se"], np.sqrt(cA), 1e-11)
        check("b1x2:a:se_B_cluster", rc["delta"]["B"]["se"], np.sqrt(cB), 1e-11)
        check("b1x2:a:se_total_cluster", rc["total"]["se"], np.sqrt(cT), 1e-11)

        # (w) weighted estimators: aweights and fweights, all vce modes
        cl = np.arange(n) % 40
        for tag, kw in [("awdef", dict(weights=aw)),
                        ("awg0", dict(weights=aw, gamma0=True)),
                        ("awrob", dict(weights=aw, vce="robust")),
                        ("awcl", dict(weights=aw, vce="cluster", cluster=cl)),
                        ("fwdef", dict(weights=fw, fweights=True)),
                        ("fwrob", dict(weights=fw, fweights=True, vce="robust")),
                        # fw x cluster: audit finding F2 (09jul2026) — a row
                        # standing for w copies must contribute w*z to its
                        # cluster's score sum (b1x2 on the original data ==
                        # b1x2 on the row-expanded data; sqrt(w) scoring
                        # understated these SEs ~2x before 2.14.2).
                        ("fwcl", dict(weights=fw, fweights=True, vce="cluster",
                                      cluster=cl))]:
            rw = gb.decompose(y, x1, x2_groups={"A": x2a, "B": x2b}, **kw)
            wA, wB, _ = unpack(ow, f"d_{tag}")
            vA_w, vB_w, _ = unpack(ow, f"v_{tag}")
            check(f"b1x2:w:{tag}:delta",
                  max(np.max(np.abs(rw["delta"]["A"]["coef"] - wA)),
                      np.max(np.abs(rw["delta"]["B"]["coef"] - wB))), 0, 1e-11)
            check(f"b1x2:w:{tag}:se",
                  max(np.max(np.abs(rw["delta"]["A"]["se"] - np.sqrt(vA_w))),
                      np.max(np.abs(rw["delta"]["B"]["se"] - np.sqrt(vB_w)))),
                  0, 1e-11)

        # (b) strong HDFE test: absorbed firm effects vs LSDV dummy block
        rb = gb.decompose(yb, x1b, x2_groups={"OBS": z},
                          fes={"FIRM": firm})
        rbg0 = gb.decompose(yb, x1b, x2_groups={"OBS": z},
                            fes={"FIRM": firm}, gamma0=True)
        O_o, F_o, TC_o = unpack(ob, "d_def")
        vO_g, vF_g, _ = unpack(ob, "v_g0")
        check("b1x2:b:delta_OBS_full", rb["delta"]["OBS"]["coef"], O_o, 1e-11)
        check("b1x2:b:delta_FIRM_x1rows", rb["delta"]["FIRM"]["coef"][:2],
              F_o[:2], 1e-11)
        check("b1x2:b:total_x1rows", rb["total"]["coef"][:2], TC_o[:2], 1e-11)
        cons_conv = abs((rb["delta"]["FIRM"]["coef"][2] - F_o[2]) -
                        (rb["total"]["coef"][2] - TC_o[2]))
        check("b1x2:b:cons-shift-isolated-to-FE-block", cons_conv, 0.0, 1e-10)
        check("b1x2:b:se_OBS_gamma0", rbg0["delta"]["OBS"]["se"],
              np.sqrt(vO_g), 1e-11)
        check("b1x2:b:se_FIRM_gamma0", rbg0["delta"]["FIRM"]["se"],
              np.sqrt(vF_g), 1e-11)
        check("b1x2:b:identity_gap", rb["identity_gap"], 0.0, 1e-10)
        check("hdfe:fe_total_coef", rb["fe_total"]["coef"],
              rb["delta"]["FIRM"]["coef"], 1e-12)
        check("hdfe:fe_total_cov", rb["fe_total"]["cov"],
              rb["cov"][3:6, 3:6], 1e-12)
        check("hdfe:total_cov-is-block-sum", rb["total"]["se"] ** 2,
              np.diag(sum(rb["cov"][g * 3:(g + 1) * 3,
                                     h * 3:(h + 1) * 3]
                          for g in range(2) for h in range(2))), 1e-12)

        # LSDV robust gamma0: within-representation of the stacked sandwich
        rbg0r = gb.decompose(yb, x1b, x2_groups={"OBS": z}, fes={"FIRM": firm},
                             gamma0=True, vce="robust")
        vO_gr, vF_gr, _ = unpack(ob, "v_g0rob")
        check("b1x2:b:se_OBS_gamma0_robust", rbg0r["delta"]["OBS"]["se"],
              np.sqrt(vO_gr), 1e-11)
        check("b1x2:b:se_FIRM_gamma0_robust", rbg0r["delta"]["FIRM"]["se"],
              np.sqrt(vF_gr), 1e-11)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
