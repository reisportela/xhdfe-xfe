#!/usr/bin/env python3
"""WS4 Task A: implementation vs independent oracle, per-fixture max errors.

Usage: python3 run_oracle_comparison.py  (writes ws4_oracle_comparison_raw.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/mangelo/Documents/GitHub/xhdfe"
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import xhdfe.gelbach as gb              # noqa: E402
from gelbach_oracle import (            # noqa: E402
    oracle, expand_fweights, share_movement_se, share_base_se_correct)

RESULTS = {}


def errpair(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        return {"abs": float("inf"), "rel": float("inf"),
                "note": f"shape {a.shape} vs {b.shape}"}
    if a.size == 0:
        return {"abs": 0.0, "rel": 0.0}
    d = float(np.max(np.abs(a - b)))
    scale = float(max(np.max(np.abs(b)), 1e-30))
    return {"abs": d, "rel": d / scale}


def impl_matrices(r):
    names = r["names"]
    delta = np.column_stack([r["delta"][nm]["coef"] for nm in names])
    se = np.column_stack([r["delta"][nm]["se"] for nm in names])
    return names, delta, se


def compare(tag, r, o, skip_fe_x1rows=False, absorbed=None):
    """Record implementation-vs-oracle errors for one fixture."""
    names, delta_i, se_i = impl_matrices(r)
    assert names == o["names"], (names, o["names"])
    k1 = len(r["labels"])
    G = len(names)
    out = {}
    out["b_base"] = errpair(r["b_base"], o["b_base"])
    out["b_full"] = errpair(r["b_full"], o["b_full"])
    out["df_full"] = errpair(r["df_full"], o["df_full"])
    out["df_base"] = errpair(r["df_base"], o["df_base"])
    out["base_cov"] = errpair(r["base_cov"], o["V_base"])
    # delta: x1 rows always; _cons rows only summed across blocks (per-block
    # _cons split depends on the FE normalization convention).
    p = len(r["b_base"])
    out["delta_x1rows"] = errpair(delta_i[:p, :], o["delta"][:p, :])
    # The per-block and summed _cons rows depend on the engine's FE/constant
    # allocation convention; they are only cross-checkable when no FE block is
    # present (explicit constant in both parameterizations).
    if all(r["group_kinds"][nm] == "x2" for nm in names):
        out["delta_cons_total"] = errpair(delta_i[p, :].sum(),
                                          o["delta"][p, :].sum())
    out["total"] = errpair(r["total"]["coef"][:p], o["total"][:p])
    se_o = np.sqrt(np.column_stack(
        [np.diag(o["cov"][g * k1:(g + 1) * k1, g * k1:(g + 1) * k1])
         for g in range(G)]))
    if skip_fe_x1rows:
        # compare observed blocks fully; FE blocks: totals only
        obs = [g for g, nm in enumerate(names) if r["group_kinds"][nm] == "x2"]
        out["se_observed"] = errpair(se_i[:, obs], se_o[:, obs])
        out["cov_observed"] = errpair(
            np.array([[r["cov"][g * k1 + i, h * k1 + j]
                       for h in obs for j in range(p)] for g in obs
                      for i in range(p)]),
            np.array([[o["cov"][g * k1 + i, h * k1 + j]
                       for h in obs for j in range(p)] for g in obs
                      for i in range(p)]))
    else:
        out["se_all"] = errpair(se_i, se_o)
        out["cov_full"] = errpair(r["cov"], o["cov"])
        out["cov_delta_bbase"] = errpair(
            r["cov_delta_bbase"], o["cov_delta_bbase"])
    if skip_fe_x1rows:
        rows = [g * k1 + j for g in range(G) for j in range(p)]
        out["cov_delta_bbase_x1"] = errpair(
            np.asarray(r["cov_delta_bbase"])[np.ix_(rows, range(p))],
            o["cov_delta_bbase"][np.ix_(rows, range(p))])
    out["cov_total_bbase_x1"] = errpair(
        np.asarray(r["cov_total_bbase"])[:p, :p],
        o["cov_total_bbase"][:p, :p])
    tc_o = o["total_cov"].copy()
    if absorbed:
        # the implementation's documented contract replaces the target-target
        # entries of total_cov with the base-model VCE; mirror that in the
        # oracle prediction and ALSO record the raw component-sum discrepancy.
        for j in absorbed:
            for j2 in absorbed:
                tc_o[j, j2] = o["V_base"][j, j2]
        for j in absorbed:
            out[f"abs_total_se_vs_baseVCE_col{j}"] = errpair(
                np.asarray(r["total_cov"])[j, j], o["V_base"][j, j])
            out[f"abs_componentsum_vs_baseVCE_col{j}"] = errpair(
                o["total_cov"][j, j], o["V_base"][j, j])
    out["total_cov_x1"] = errpair(np.asarray(r["total_cov"])[:p, :p],
                                  tc_o[:p, :p])
    out["identity_gap_impl"] = float(r["identity_gap"])
    out["converged"] = bool(r["converged"])
    RESULTS[tag] = out
    worst = max((v["rel"] for v in out.values() if isinstance(v, dict)),
                default=0.0)
    print(f"[{tag}] worst rel err = {worst:.3e}  identity_gap={r['identity_gap']:.2e}")
    return out


# ----------------------------------------------------------------- fixtures

def fixture_noFE(seed=101, n=400):
    rng = np.random.default_rng(seed)
    x1 = np.column_stack([rng.normal(size=n), rng.uniform(-1, 2, n)])
    a = np.column_stack([0.6 * x1[:, 0] + rng.normal(size=n),
                         rng.normal(size=n)])
    b = np.column_stack([-0.4 * x1[:, 1] + rng.normal(size=n)])
    y = (x1 @ [1.2, -0.7] + a @ [0.8, 0.3] + b @ [0.5]
         + rng.normal(size=n))
    cl = rng.integers(0, 25, n)
    aw = rng.uniform(0.5, 4.0, n)
    fw = rng.integers(1, 5, n).astype(float)
    return y, x1, a, b, cl, aw, fw


def fixture_1FE(seed=202, n=600, nf=30):
    rng = np.random.default_rng(seed)
    x1 = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    firm = rng.integers(0, nf, n)
    a = np.column_stack([0.5 * x1[:, 0] + 0.2 * firm / nf + rng.normal(size=n),
                         rng.normal(size=n)])
    b = np.column_stack([rng.normal(size=n) - 0.3 * x1[:, 1]])
    psi = rng.normal(0, 0.8, nf)
    y = x1 @ [1.0, -0.5] + a @ [0.7, 0.25] + b @ [0.4] + psi[firm] + rng.normal(size=n)
    cl = rng.integers(0, 20, n)
    aw = rng.uniform(0.5, 3.0, n)
    fw = rng.integers(1, 4, n).astype(float)
    return y, x1, a, b, firm, cl, aw, fw


def fixture_2FE(seed=303, n_workers=120, periods=6, nf=25):
    rng = np.random.default_rng(seed)
    worker = np.repeat(np.arange(n_workers), periods)
    n = worker.size
    # connected mobility: random firm draws with churn
    firm = rng.integers(0, nf, n)
    x1 = np.column_stack([rng.normal(size=n)])
    obs = np.column_stack([0.4 * x1[:, 0] + rng.normal(size=n)])
    alpha = rng.normal(0, 0.7, n_workers)
    psi = rng.normal(0, 0.5, nf)
    y = (x1 @ [0.9] + obs @ [0.6] + alpha[worker] + psi[firm]
         + rng.normal(0, 0.8, n))
    return y, x1, obs, worker, firm


def fixture_absorbed(seed=404, n_workers=100, periods=5):
    rng = np.random.default_rng(seed)
    worker = np.repeat(np.arange(n_workers), periods)
    n = worker.size
    female = rng.integers(0, 2, n_workers)[worker].astype(float)
    exper = np.tile(np.arange(periods), n_workers) + rng.normal(0, 0.2, n)
    job = 0.4 * female + 0.15 * exper + rng.normal(size=n)
    alpha = -0.3 * rng.integers(0, 2, n_workers).astype(float) + rng.normal(0, 0.6, n_workers)
    y = (0.2 * female + 0.05 * exper + 0.5 * job + alpha[worker]
         + rng.normal(0, 0.5, n))
    x1 = np.column_stack([female, exper])
    return y, x1, job, worker


def run_all():
    # ---------------- F1: no-FE (b1x2 domain), all vce, all weights
    y, x1, a, b, cl, aw, fw = fixture_noFE()
    grids = [
        ("unadjusted", None, None, False),
        ("robust", None, None, False),
        ("cluster", cl, None, False),
        ("unadjusted", None, ("aw", aw), False),
        ("robust", None, ("aw", aw), False),
        ("cluster", cl, ("aw", aw), False),
        ("unadjusted", None, ("fw", fw), False),
        ("robust", None, ("fw", fw), False),
        ("cluster", cl, ("fw", fw), False),
        ("robust", None, None, True),          # gamma0
        ("cluster", cl, None, "cov0"),         # cov0
    ]
    for vce, clu, wspec, flag in grids:
        kw = {}
        wo = {}
        wtag = "none"
        if wspec is not None:
            wtag, wv = wspec
            kw["weights"] = wv
            wo["weights"] = wv
            if wtag == "fw":
                kw["fweights"] = True
                wo["fweights"] = True
        g0 = flag is True
        c0 = flag == "cov0"
        r = gb.decompose(y, x1, {"A": a, "B": b}, None, vce=vce,
                         cluster=clu if vce == "cluster" else None,
                         gamma0=g0, cov0=c0, **kw)
        o = oracle(y, x1, [("A", a), ("B", b)], [], vce=vce,
                   cluster=clu if vce == "cluster" else None,
                   gamma0=g0, cov0=c0, **wo)
        compare(f"F1_noFE_{vce}_{wtag}" + ("_gamma0" if g0 else "")
                + ("_cov0" if c0 else ""), r, o)

    # ---------------- F2: one FE dim + two observed blocks
    y, x1, a, b, firm, cl, aw, fw = fixture_1FE()
    for vce, clu in [("unadjusted", None), ("robust", None), ("cluster", cl)]:
        for wtag, wv, fwq in [("none", None, False), ("aw", aw, False),
                              ("fw", fw, True)]:
            kw = {"weights": wv, "fweights": fwq} if wv is not None else {}
            r = gb.decompose(y, x1, {"A": a, "B": b}, {"FIRM": firm}, vce=vce,
                             cluster=clu if vce == "cluster" else None,
                             tol=1e-10, **kw)
            o = oracle(y, x1, [("A", a), ("B", b)], [("FIRM", firm)], vce=vce,
                       cluster=clu if vce == "cluster" else None,
                       weights=wv, fweights=fwq)
            compare(f"F2_1FE_{vce}_{wtag}", r, o)

    # ---------------- F3: two FE dims (connected), per-dim split check
    y, x1, obs, worker, firm = fixture_2FE()
    for vce, clu in [("unadjusted", None), ("robust", None),
                     ("cluster", worker)]:
        r = gb.decompose(y, x1, {"OBS": obs}, {"WORKER": worker, "FIRM": firm},
                         vce=vce, cluster=clu if vce == "cluster" else None,
                         tol=1e-10)
        o = oracle(y, x1, [("OBS", obs)], [("WORKER", worker), ("FIRM", firm)],
                   vce=vce, cluster=clu if vce == "cluster" else None)
        compare(f"F3_2FE_{vce}", r, o)

    # ---------------- F4: absorbed target under all three VCEs
    y, x1, job, worker = fixture_absorbed()
    for vce, clu in [("unadjusted", None), ("robust", None),
                     ("cluster", worker)]:
        r = gb.decompose(y, x1, {"JOB": job}, {"WORKER": worker}, vce=vce,
                         cluster=clu if vce == "cluster" else None,
                         tol=1e-10, absorbed_targets=[0])
        o = oracle(y, x1, [("JOB", job)], [("WORKER", worker)], vce=vce,
                   cluster=clu if vce == "cluster" else None, absorbed=[0])
        out = compare(f"F4_absorbed_{vce}", r, o, absorbed=[0])
        out["inference_status"] = r["inference_status"]
        out["absorbed_target_inference_valid"] = r["absorbed_target_inference_valid"]
        # total point identity: total[0] == b_base[0] exactly
        out["abs_total_equals_bbase"] = errpair(r["total"]["coef"][0],
                                                r["b_base"][0])

    # ---------------- F5: fweights == physical expansion (implementation only,
    # must be EXACT: estimates, SEs, df, N)
    y, x1, a, b, firm, cl, aw, fw = fixture_1FE()
    ye, x1e, ae, be, firme, cle = expand_fweights([y, x1, a, b, firm, cl], fw)
    for vce, clu_w, clu_e in [("unadjusted", None, None),
                              ("robust", None, None),
                              ("cluster", cl, cle)]:
        rw = gb.decompose(y, x1, {"A": a, "B": b}, {"FIRM": firm}, vce=vce,
                          cluster=clu_w if vce == "cluster" else None,
                          weights=fw, fweights=True, tol=1e-12)
        re_ = gb.decompose(ye, x1e, {"A": ae, "B": be}, {"FIRM": firme},
                           vce=vce, cluster=clu_e if vce == "cluster" else None,
                           tol=1e-12)
        nm, dw, sw = impl_matrices(rw)
        nm2, de, se_e = impl_matrices(re_)
        RESULTS[f"F5_fw_expansion_{vce}"] = {
            "b_base": errpair(rw["b_base"], re_["b_base"]),
            "b_full": errpair(rw["b_full"], re_["b_full"]),
            "delta": errpair(dw, de),
            "se": errpair(sw, se_e),
            "total_cov": errpair(rw["total_cov"], re_["total_cov"]),
            "df_full": errpair(rw["df_full"], re_["df_full"]),
            "n_eff_match": rw["n_obs_effective"] == re_["n_obs"],
        }
        worst = max(v["rel"] for v in RESULTS[f"F5_fw_expansion_{vce}"].values()
                    if isinstance(v, dict))
        print(f"[F5_fw_expansion_{vce}] worst rel err = {worst:.3e} "
              f"n_eff_match={RESULTS[f'F5_fw_expansion_{vce}']['n_eff_match']}")

    # ---------------- F6: metamorphic invariances (implementation only)
    y, x1, a, b, firm, cl, aw, fw = fixture_1FE()
    base = gb.decompose(y, x1, {"A": a, "B": b}, {"FIRM": firm},
                        vce="cluster", cluster=cl, tol=1e-10)
    nb, db, sb = impl_matrices(base)
    # order swap of blocks
    r2 = gb.decompose(y, x1, {"B": b, "A": a}, {"FIRM": firm},
                      vce="cluster", cluster=cl, tol=1e-10)
    n2, d2, s2 = impl_matrices(r2)
    perm = [n2.index(nm) for nm in nb]
    RESULTS["F6_order_invariance"] = {
        "delta": errpair(db, d2[:, perm]), "se": errpair(sb, s2[:, perm]),
        "total": errpair(base["total"]["coef"], r2["total"]["coef"]),
        "total_cov": errpair(base["total_cov"], r2["total_cov"]),
    }
    # within-block reparameterization: A -> A @ T (invertible)
    T = np.array([[2.0, 0.7], [-0.5, 1.3]])
    r3 = gb.decompose(y, x1, {"A": a @ T, "B": b}, {"FIRM": firm},
                      vce="cluster", cluster=cl, tol=1e-10)
    n3, d3, s3 = impl_matrices(r3)
    RESULTS["F6_reparam_invariance"] = {
        "delta": errpair(db, d3), "se": errpair(sb, s3),
        "cov": errpair(base["cov"], r3["cov"]),
    }
    # split-and-recombine: A split into two singleton blocks
    r4 = gb.decompose(y, x1, {"A1": a[:, :1], "A2": a[:, 1:], "B": b},
                      {"FIRM": firm}, vce="cluster", cluster=cl, tol=1e-10)
    n4, d4, s4 = impl_matrices(r4)
    da_sum = d4[:, n4.index("A1")] + d4[:, n4.index("A2")]
    k1 = len(base["labels"])
    ia1, ia2 = n4.index("A1"), n4.index("A2")
    cov4 = np.asarray(r4["cov"])
    var_sum = (cov4[ia1 * k1:(ia1 + 1) * k1, ia1 * k1:(ia1 + 1) * k1]
               + cov4[ia2 * k1:(ia2 + 1) * k1, ia2 * k1:(ia2 + 1) * k1]
               + cov4[ia1 * k1:(ia1 + 1) * k1, ia2 * k1:(ia2 + 1) * k1]
               + cov4[ia2 * k1:(ia2 + 1) * k1, ia1 * k1:(ia1 + 1) * k1])
    ia = nb.index("A")
    covb = np.asarray(base["cov"])
    RESULTS["F6_split_recombine"] = {
        "delta_sum": errpair(db[:, ia], da_sum),
        "cov_sum": errpair(covb[ia * k1:(ia + 1) * k1, ia * k1:(ia + 1) * k1],
                           var_sum),
        "total": errpair(base["total"]["coef"], r4["total"]["coef"]),
        "total_cov": errpair(base["total_cov"], r4["total_cov"]),
    }
    for tag in ("F6_order_invariance", "F6_reparam_invariance",
                "F6_split_recombine"):
        worst = max(v["rel"] for v in RESULTS[tag].values()
                    if isinstance(v, dict))
        print(f"[{tag}] worst rel err = {worst:.3e}")

    # ---------------- F7: share SEs — movement vs oracle; base share error
    y, x1, a, b, firm, cl, aw, fw = fixture_1FE()
    r = gb.decompose(y, x1, {"A": a, "B": b}, {"FIRM": firm},
                     vce="cluster", cluster=cl, tol=1e-10,
                     x1_names=["x11", "x12"], focal="x11")
    o = oracle(y, x1, [("A", a), ("B", b)], [("FIRM", firm)],
               vce="cluster", cluster=cl)
    tab = gb.tidy(r, share="movement", include_total=False, include_full=False)
    k1 = len(r["labels"])
    G = len(r["names"])
    row = 0
    cov_row = np.asarray(r["cov"])[np.ix_([g * k1 + row for g in range(G)],
                                          [g * k1 + row for g in range(G)])]
    d_row = np.array([r["delta"][nm]["coef"][row] for nm in r["names"]])
    se_mv_o = share_movement_se(d_row, cov_row)
    se_mv_i = np.array([t["share_std_error"] for t in tab])
    RESULTS["F7_share_movement_se"] = {"se": errpair(se_mv_i, se_mv_o)}
    print(f"[F7_share_movement_se] rel err = "
          f"{RESULTS['F7_share_movement_se']['se']['rel']:.3e}")
    # Base-share now uses the full joint delta method. Keep base_fixed as a
    # separately labelled historical/descriptive convention.
    tabb = gb.tidy(r, share="base", include_total=False,
                   include_full=False)
    se_b = np.array([t["share_std_error"] for t in tabb])
    tabf = gb.tidy(r, share="base_fixed", include_total=False,
                   include_full=False)
    se_bf = np.array([t["share_std_error"] for t in tabf])
    cov_row_bb = o["cov_delta_bbase"][
        np.ix_([g * k1 + row for g in range(G)], [row])].ravel()
    cov_row_o = o["cov"][np.ix_([g * k1 + row for g in range(G)],
                                [g * k1 + row for g in range(G)])]
    d_row_o = o["delta"][row, :]
    se_base_correct = share_base_se_correct(
        d_row_o, o["b_base"][row], cov_row_o, cov_row_bb,
        o["V_base"][row, row])
    RESULTS["F7_share_base_joint_vs_correct"] = {
        "se": errpair(se_b, se_base_correct),
        "share_values": [t["share"] for t in tabb],
    }
    RESULTS["F7_share_base_fixed_vs_correct"] = {
        "se_base_fixed_impl": se_bf.tolist(),
        "se_base_correct_oracle": se_base_correct.tolist(),
        "ratio_fixed_over_correct": (se_bf / se_base_correct).tolist(),
        "share_values": [t["share"] for t in tabf],
    }
    print("[F7 base-share] joint oracle rel err:",
          RESULTS["F7_share_base_joint_vs_correct"]["se"]["rel"])
    print("[F7 base-share] fixed/correct SE ratios:",
          np.round(se_bf / se_base_correct, 3))

    with open(os.path.join(HERE, "ws4_oracle_comparison_raw.json"), "w") as f:
        json.dump(RESULTS, f, indent=1, default=float)
    print("\nwrote ws4_oracle_comparison_raw.json with", len(RESULTS), "fixtures")


if __name__ == "__main__":
    run_all()
