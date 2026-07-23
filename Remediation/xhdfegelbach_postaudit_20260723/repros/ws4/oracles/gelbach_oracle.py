#!/usr/bin/env python3
"""WS4 independent Gelbach oracle (pure numpy/scipy, dense LSDV).

DERIVATION (independent of the xhdfe implementation; from OLS/FWL algebra):

Setup. n rows, strictly positive moment weights omega_i:
  - no weights: omega = 1, N_eff = n
  - aweights w: omega_i = n * w_i / sum(w)   (Stata convention), N_eff = n
  - fweights w (integer): omega_i = w_i, N_eff = sum(w). Ground truth for
    fweights is PHYSICAL ROW EXPANSION: every quantity must equal the
    unweighted run on the expanded data set.

Base design X1t = [X1, 1] (k1 = p+1 columns), P = (X1t' Om X1t)^{-1}.
Full design Zf = [X1(active), X2, D] where D is a full-rank pinned basis of
the FE dummy columns (dim 1 keeps all levels so the constant is inside D;
later dims drop one level per connected component of the mobility graph).
WLS: beta = (Zf' Om Zf)^{-1} Zf' Om y; u = y - Zf beta.

Point estimands (Gelbach 2016 / omitted-variable algebra):
  b_base = P X1t' Om y
  H_g    = X2_g ghat_g            (observed block)
  H_d    = D_d alphahat_d         (FE dim d; pinned; x1-rows of delta_d are
                                   pinning-invariant iff the mobility graph is
                                   connected, because pinnings differ by
                                   per-component constants)
  delta_g = P X1t' Om H_g
  Identity: X1t' Om u = 0 (X1t is in span(Zf)) implies
  b_base - [b_full; cons_alloc] = sum_g delta_g   exactly, in sample.

Inference (stacked estimating equations; two-step GMM linearization):
  Moments:  sum_i om_i zf_i u_i = 0;   sum_i om_i x1t_i v_gi = 0
  with v_g = H_g - X1t delta_g. The joint Jacobian is block triangular, so
  the influence function of delta_g is EXACTLY
     IF_i(delta_g) = P x1t_i om_i v_gi  +  Gamma_g [A_F^{-1} zf_i om_i u_i]_g
  where Gamma_g = P X1t' Om X2_g (aux projection coefficients) and A_F =
  Zf' Om Zf. Sandwich pieces per block pair (g,h), each k1 x k1:
     AUX(g,h)   = P M_aux(g,h) P            (meat of the aux scores)
     GG(g,h)    = Gamma_g V_full[g,h] Gamma_h'
     CROSS(g,h) = Gamma_g A_F^{-1}[g,:] M_cross(h) P  (+ transpose image)
  Multiplier conventions (validated against Gelbach's own b1x2 in Task D;
  they are conventions, not asymptotics):
     robust:   q_big = N/(N-1) on AUX+CROSS, q_vu = N/df_full on GG
     cluster:  q_big = Gc/(Gc-1),           q_vu = (N-1)/df_full * Gc/(Gc-1)
     N = N_eff. Score factor sf_i: omega_i for none/aweights; for fweights
     sqrt(w_i) under robust and w_i under cluster (both DERIVED from the
     physical-expansion definition, not copied: expanding rows w times makes
     the robust meat sum w*(z e)(z e)' per unique row and the cluster score
     sum w*z e per unique row).
  Unadjusted (iid-form, Gelbach App. B):
     V(g,h) = Omhat_vv[g,h] * P + s2 * Gamma_g (A_F^{-1})[g,h] Gamma_h'
     Omhat_vv[g,h] = sum_i om_i vt_gi vt_hi / df_full  (vt = weighted-centred v)
     s2 = sum om_i u_i^2 / df_full.
  FE blocks: the xhdfe contract gives FE blocks the AUX-only (gamma0 /
  "conditional") variance. The oracle can additionally compute the FULL LSDV
  variance including the dummy-coefficient block of V_full (fe_full mode) to
  QUANTIFY what the conditional convention omits.

Absorbed-target mode: the focal column is excluded from Zf (imposed zero) and
total_focal = b_base_focal exactly, so Var(total_focal) is the BASE-model
variance under the requested VCE:
     unadjusted: s2_base * P, s2_base = sum om r^2 / (N_eff - k1)
     robust:     N/(N-k1) * P M_base P
     cluster:    (N-1)/(N-k1) * Gc/(Gc-1) * P M_base P   (Stata regress).

Joint covariance with b_base (for share-SE truth): b_base is an M-estimator
on the same data with IF_i = P x1t_i om_i r_i (r = base residual); stack it to
get Cov(delta_g, b_base) and hence the CORRECT delta-method SE of the base
share delta_gj / b_base_j:  grad = (e_g/b, -delta_g/b^2).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- utilities

def _omega(n, weights, fweights):
    if weights is None:
        return np.ones(n), float(n)
    w = np.asarray(weights, float)
    if fweights:
        return w.copy(), float(w.sum())
    return w * (n / w.sum()), float(n)


def dummies(ids):
    ids = np.asarray(ids)
    _, codes = np.unique(ids, return_inverse=True)
    L = codes.max() + 1
    D = np.zeros((ids.shape[0], L))
    D[np.arange(ids.shape[0]), codes] = 1.0
    return D, codes


def mobility_components(fe_codes_list):
    """Union-find over levels of all FE dims; returns per-row component id."""
    offs, total = [], 0
    for c in fe_codes_list:
        offs.append(total)
        total += c.max() + 1
    parent = np.arange(total)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    n = fe_codes_list[0].shape[0]
    for i in range(n):
        r0 = find(offs[0] + fe_codes_list[0][i])
        for d in range(1, len(fe_codes_list)):
            rd = find(offs[d] + fe_codes_list[d][i])
            if rd != r0:
                parent[rd] = r0
    comp = np.array([find(offs[0] + fe_codes_list[0][i]) for i in range(n)])
    _, comp = np.unique(comp, return_inverse=True)
    return comp


def pinned_dummy_basis(fe_ids_list):
    """Full-rank pinned LSDV basis. Dim 0: all levels (constant in span).
    Dim d>=1: drop one level per connected component of the mobility graph.
    Returns (D, per_dim_col_slices, codes_list)."""
    codes_list = []
    for ids in fe_ids_list:
        _, c = np.unique(np.asarray(ids), return_inverse=True)
        codes_list.append(c)
    comp = mobility_components(codes_list) if len(codes_list) else None
    blocks, slices, start = [], [], 0
    for d, c in enumerate(codes_list):
        D, _ = dummies(c)
        if d >= 1:
            drop = []
            seen = set()
            for i in range(c.shape[0]):
                if comp[i] not in seen:
                    seen.add(comp[i])
                    drop.append(c[i])
            keep = np.setdiff1d(np.arange(D.shape[1]), np.array(sorted(set(drop))))
            D = D[:, keep]
        blocks.append(D)
        slices.append(slice(start, start + D.shape[1]))
        start += D.shape[1]
    if blocks:
        return np.hstack(blocks), slices, codes_list
    return np.zeros((len(fe_ids_list[0]) if fe_ids_list else 0, 0)), [], codes_list


def wls(y, X, om):
    A = X.T @ (X * om[:, None])
    b = X.T @ (om * y)
    beta = np.linalg.solve(A, b)
    return beta, A


def cluster_sum(Z, codes):
    if codes is None:
        return Z
    G = codes.max() + 1
    S = np.zeros((G, Z.shape[1]))
    np.add.at(S, codes, Z)
    return S


# ---------------------------------------------------------------- oracle

def oracle(y, x1, x2_blocks=None, fe_ids=None, vce="unadjusted", cluster=None,
           weights=None, fweights=False, gamma0=False, cov0=False,
           absorbed=None, fe_full_variance=False):
    """Independent dense Gelbach oracle.

    x2_blocks: list of (name, (n,q) array); fe_ids: list of (name, ids).
    absorbed: list of x1 column indices imposed to zero in the full model.
    Returns dict with b_base, b_full, delta (k1 x G), cov (G*k1 x G*k1),
    total, total_cov, plus joint pieces cov_delta_bbase (G*k1 x k1),
    V_base (k1 x k1, requested VCE), and (fe_full_variance) an alternative
    covariance treating FE dummy coefficients as estimated.
    """
    y = np.asarray(y, float)
    x1 = np.asarray(x1, float)
    n, p = x1.shape
    x2_blocks = x2_blocks or []
    fe_ids = fe_ids or []
    absorbed = sorted(absorbed or [])
    om, n_eff = _omega(n, weights, fweights)
    k1 = p + 1
    X1t = np.hstack([x1, np.ones((n, 1))])
    A1 = X1t.T @ (X1t * om[:, None])
    P = np.linalg.solve(A1, np.eye(k1))

    # base model
    b_base_vec, _ = wls(y, X1t, om)
    r_base = y - X1t @ b_base_vec

    # full model design (pinned)
    active = [c for c in range(p) if c not in absorbed]
    X2 = np.hstack([np.asarray(m, float).reshape(n, -1) for _, m in x2_blocks]) \
        if x2_blocks else np.zeros((n, 0))
    q = X2.shape[1]
    D, fe_slices, fe_codes = pinned_dummy_basis([ids for _, ids in fe_ids])
    if fe_ids:
        Zf = np.hstack([x1[:, active], X2, D])
    else:
        Zf = np.hstack([x1[:, active], X2, np.ones((n, 1))])
    beta, A_F = wls(y, Zf, om)
    u = y - Zf @ beta
    A_F_inv = np.linalg.solve(A_F, np.eye(A_F.shape[0]))
    pa = len(active)
    b_full = np.zeros(p)
    for j, c in enumerate(active):
        b_full[c] = beta[j]
    ghat = beta[pa:pa + q]

    # blocks: observed groups then FE dims (implementation order)
    names, Hs, gam, spans = [], [], [], []
    cursor = 0
    for name, m in x2_blocks:
        qg = np.asarray(m).reshape(n, -1).shape[1]
        Hg = X2[:, cursor:cursor + qg] @ ghat[cursor:cursor + qg]
        names.append(name)
        Hs.append(Hg)
        gam.append(P @ (X1t.T @ (om[:, None] * X2[:, cursor:cursor + qg])))
        spans.append((pa + cursor, pa + cursor + qg))
        cursor += qg
    for d, (name, _ids) in enumerate(fe_ids):
        sl = fe_slices[d]
        Hd = D[:, sl] @ beta[pa + q + sl.start: pa + q + sl.stop]
        names.append(name)
        Hs.append(Hd)
        gam.append(None if not fe_full_variance
                   else P @ (X1t.T @ (om[:, None] * D[:, sl])))
        spans.append((pa + q + sl.start, pa + q + sl.stop))
    G = len(names)

    delta = np.zeros((k1, G))
    vres = []
    for g in range(G):
        dg, _ = wls(Hs[g], X1t, om)
        delta[:, g] = dg
        vres.append(Hs[g] - X1t @ dg)

    # ----- degrees of freedom of the full model
    df_full = n_eff - Zf.shape[1]
    s2 = float((om * u * u).sum() / df_full)

    # ----- cluster codes
    ccodes = None
    n_cl = 0
    if cluster is not None:
        _, ccodes = np.unique(np.asarray(cluster), return_inverse=True)
        n_cl = ccodes.max() + 1
    if vce == "cluster" and ccodes is None:
        raise ValueError("cluster vce requires cluster ids")

    # ----- score factor
    if weights is not None and fweights:
        w = np.asarray(weights, float)
        sf = w if vce == "cluster" else np.sqrt(w)
    else:
        sf = om.copy()

    # ----- covariance
    cov = np.zeros((G * k1, G * k1))
    treat_full = [(gam[g] is not None) for g in range(G)]
    if vce == "unadjusted":
        # centred weighted residual cross-products / df_full
        Vt = np.column_stack(vres)
        Vt = Vt - (om @ Vt) / om.sum()
        Omv = (Vt.T @ (Vt * om[:, None])) / df_full
        for g in range(G):
            for h in range(G):
                block = Omv[g, h] * P
                if (not gamma0) and treat_full[g] and treat_full[h]:
                    s0, s1 = spans[g]
                    t0, t1 = spans[h]
                    block = block + s2 * gam[g] @ A_F_inv[s0:s1, t0:t1] @ gam[h].T
                cov[g * k1:(g + 1) * k1, h * k1:(h + 1) * k1] = block
        V_full_for_base = None
    else:
        if ccodes is None and vce != "robust":
            raise ValueError("vce must be unadjusted/robust/cluster")
        use_codes = ccodes if vce == "cluster" else None
        Nc = n_cl if vce == "cluster" else n
        if vce == "cluster":
            q_big = Nc / (Nc - 1.0)
            q_vu = (n_eff - 1.0) / df_full * Nc / (Nc - 1.0)
        else:
            q_big = n_eff / (n_eff - 1.0)
            q_vu = n_eff / df_full
        Sfull = Zf * (sf * u)[:, None]                     # full-model scores
        Saux = [X1t * (sf * vres[g])[:, None] for g in range(G)]
        Cf = cluster_sum(Sfull, use_codes)
        Ca = [cluster_sum(Saux[g], use_codes) for g in range(G)]
        M_ff = Cf.T @ Cf
        V_full_sand = q_vu * (A_F_inv @ M_ff @ A_F_inv)
        for g in range(G):
            for h in range(G):
                block = q_big * (P @ (Ca[g].T @ Ca[h]) @ P)
                if (not gamma0) and treat_full[g] and treat_full[h]:
                    s0, s1 = spans[g]
                    t0, t1 = spans[h]
                    block = block + gam[g] @ V_full_sand[s0:s1, t0:t1] @ gam[h].T
                    if not cov0:
                        # cross: Gamma_g (A_F^{-1} full-score, aux-score_h) P
                        Bg = A_F_inv[s0:s1, :] @ Cf.T @ Ca[h] @ P
                        Bh = A_F_inv[t0:t1, :] @ Cf.T @ Ca[g] @ P
                        block = block + q_big * (gam[g] @ Bg) \
                                      + q_big * (gam[h] @ Bh).T
                cov[g * k1:(g + 1) * k1, h * k1:(h + 1) * k1] = block
        V_full_for_base = V_full_sand

    total = delta.sum(axis=1)
    total_cov = np.zeros((k1, k1))
    for g in range(G):
        for h in range(G):
            total_cov += cov[g * k1:(g + 1) * k1, h * k1:(h + 1) * k1]

    # ----- base-model VCE under the requested vce (Stata regress conventions)
    df_base = n_eff - k1
    if vce == "unadjusted":
        s2_base = float((om * r_base * r_base).sum() / df_base)
        V_base = s2_base * P
    else:
        Sb = X1t * (sf * r_base)[:, None]
        Cb = cluster_sum(Sb, ccodes if vce == "cluster" else None)
        Mb = Cb.T @ Cb
        if vce == "cluster":
            corr = (n_eff - 1.0) / df_base * n_cl / (n_cl - 1.0)
        else:
            corr = n_eff / df_base
        V_base = corr * (P @ Mb @ P)

    # ----- joint covariance of (delta stack, b_base) for share-SE truth
    # (sandwich, robust/cluster scores; for unadjusted use iid-form analog)
    cov_delta_bbase = np.zeros((G * k1, k1))
    if vce == "unadjusted":
        # E[v_g r_base'] iid-form: r_base = u + sum_h v_h  (in sample, exactly)
        # so Cov(term1_g, b_base) = (Omega_vu[g] + sum_h Omega_vv[g,h]) P
        # and Cov(GG-part, b_base) via Gamma and A_F^{-1} cross with X1t:
        Vt = np.column_stack(vres)
        Vtc = Vt - (om @ Vt) / om.sum()
        uc = u - (om @ u) / om.sum()
        for g in range(G):
            omv_u = float((om * Vtc[:, g] * uc).sum() / df_full)
            row = (omv_u + sum(float((om * Vtc[:, g] * Vtc[:, h]).sum() / df_full)
                               for h in range(G))) * P
            if (not gamma0) and treat_full[g]:
                s0, s1 = spans[g]
                # Cov(ghat, b_base): s2 * A_F^{-1} Zf' Om X1t P
                Cgb = s2 * (A_F_inv @ (Zf.T @ (om[:, None] * X1t)) @ P)
                row = row + gam[g] @ Cgb[s0:s1, :]
            cov_delta_bbase[g * k1:(g + 1) * k1, :] = row
    else:
        use_codes = ccodes if vce == "cluster" else None
        Sb = X1t * (sf * r_base)[:, None]
        Cb = cluster_sum(Sb, use_codes)
        Sfull = Zf * (sf * u)[:, None]
        Cf = cluster_sum(Sfull, use_codes)
        for g in range(G):
            Sag = X1t * (sf * vres[g])[:, None]
            Cag = cluster_sum(Sag, use_codes)
            # This is a cross-block of the same stacked full+aux system used
            # by b1x2.  The audit copy accidentally omitted q_big here even
            # though it applies q_big to the corresponding blocks of
            # Var(delta) above.  q_vu remains exclusive to the separately
            # added full/full term.
            row = q_big * (P @ (Cag.T @ Cb) @ P)
            if (not gamma0) and treat_full[g]:
                s0, s1 = spans[g]
                row = row + q_big * (
                    gam[g] @
                    (A_F_inv[s0:s1, :] @ Cf.T @ Cb @ P)
                )
            cov_delta_bbase[g * k1:(g + 1) * k1, :] = row

    cov_total_bbase = np.zeros((k1, k1))
    for g in range(G):
        cov_total_bbase += cov_delta_bbase[
            g * k1:(g + 1) * k1, :
        ]
    if absorbed:
        for target in absorbed:
            cov_total_bbase[target, :] = V_base[target, :]

    return {
        "names": names, "b_base": b_base_vec[:p], "b_base_cons": b_base_vec[p],
        "b_full": b_full, "ghat": ghat, "delta": delta, "cov": cov,
        "total": total, "total_cov": total_cov, "V_base": V_base,
        "cov_delta_bbase": cov_delta_bbase,
        "cov_total_bbase": cov_total_bbase,
        "df_full": df_full, "df_base": df_base,
        "n_eff": n_eff, "P": P, "gam": gam, "s2": s2,
        "u": u, "vres": vres, "X1t": X1t, "om": om,
    }


def share_movement_se(delta_row, cov_row):
    """Delta-method SE for delta_g / sum(delta) from the row-group covariance."""
    G = delta_row.shape[0]
    denom = delta_row.sum()
    out = np.zeros(G)
    for g in range(G):
        grad = -delta_row[g] * np.ones(G) / denom**2
        grad[g] += 1.0 / denom
        out[g] = np.sqrt(max(0.0, grad @ cov_row @ grad))
    return out


def share_base_se_correct(delta_row, b_base_j, cov_row, cov_row_bbase, var_bbase):
    """CORRECT delta-method SE for delta_g / b_base_j including Cov(delta,b)."""
    G = delta_row.shape[0]
    out = np.zeros(G)
    for g in range(G):
        v = (cov_row[g, g] / b_base_j**2
             + delta_row[g]**2 * var_bbase / b_base_j**4
             - 2.0 * delta_row[g] * cov_row_bbase[g] / b_base_j**3)
        out[g] = np.sqrt(max(0.0, v))
    return out


def expand_fweights(arrays, w):
    """Physical row expansion for integer frequency weights."""
    w = np.asarray(w)
    idx = np.repeat(np.arange(w.shape[0]), w.astype(int))
    return [np.asarray(a)[idx] for a in arrays]
