#!/usr/bin/env python3
"""WS4 Task A6: Monte Carlo coverage of xhdfegelbach confidence intervals.

Population estimands are derived in closed form from the DGP moment structure
(joint projections), NOT from the implementation.

MC1 (standard two-block design, random X):
  x1 = z_g + xi (Var z = vz, Var xi = 1-vz);  a = 0.6 x1 + ea;  b = -0.4 x1 + eb
  y = 1.2 x1 + 0.8 a + 0.5 b + c_g + eps
  Population: Gamma_A = 0.6, Gamma_B = -0.4 (projection slopes on [1, x1])
    delta_A* = 0.48, delta_B* = -0.20, total* = 0.28, b_base* = 1.48
    movement shares: s_A* = 12/7, s_B* = -5/7.
    base shares: delta_A*/1.48 and delta_B*/1.48.
  iid variant: vz = 0, Var c = 0  -> vce unadjusted + robust.
  clustered variant: vz = 0.3, Var c = 0.5 -> vce cluster (40 clusters x 20).

MC2 (absorbed target, clustered at the absorbing worker FE):
  female_j ~ B(0.5); alpha_j = beta_f*... realized worker effect eta_j~N(0,tau2)
  job = kf*female + ke*exper + w;  y = bf*female + be*exper + g*job
        + lam*female + eta_j + eps   (worker FE = bf*female + lam*female + eta)
  Population (female row): b_base* = bf + g*kf + lam
    delta_job* = g*kf;  delta_FE* = bf + lam;  total* = b_base*.
  Base share of job block: g*kf / b_base*.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/mangelo/Documents/GitHub/xhdfe"
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import warnings                       # noqa: E402
import xhdfe.gelbach as gb            # noqa: E402
from gelbach_oracle import oracle, share_base_se_correct  # noqa: E402

Z = 1.959963984540054  # Phi^{-1}(0.975)
REPS = int(os.environ.get("WS4_MC_REPS", "1000"))


def cover(est, se, truth):
    return abs(est - truth) <= Z * se


def mc1(reps=REPS, clustered=False, seed0=50000):
    nG, m = 40, 20
    n = nG * m
    vz = 0.3 if clustered else 0.0
    vc = 0.5 if clustered else 0.0
    dA, dB, tot = 0.48, -0.20, 0.28
    sA, sB = dA / tot, dB / tot
    bbase = 1.2 + dA + dB
    sbA, sbB = dA / bbase, dB / bbase
    labels = ["delta_A", "delta_B", "total", "share_A", "share_B",
              "share_A_base", "share_B_base",
              "delta_A_gamma0", "delta_B_gamma0"]
    hits = {k: 0 for k in labels}
    used = 0
    for rep in range(reps):
        rng = np.random.default_rng(seed0 + rep)
        g = np.repeat(np.arange(nG), m)
        z = rng.normal(0, np.sqrt(vz), nG)[g] if vz > 0 else 0.0
        x1 = (z + rng.normal(0, np.sqrt(1 - vz), n))[:, None]
        a = 0.6 * x1[:, 0] + rng.normal(size=n)
        b = -0.4 * x1[:, 0] + rng.normal(size=n)
        c = rng.normal(0, np.sqrt(vc), nG)[g] if vc > 0 else 0.0
        y = 1.2 * x1[:, 0] + 0.8 * a + 0.5 * b + c + rng.normal(size=n)
        vce = "cluster" if clustered else "unadjusted"
        clu = g if clustered else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = gb.decompose(y, x1, {"A": a, "B": b}, None, vce=vce,
                             cluster=clu)
            rg = gb.decompose(y, x1, {"A": a, "B": b}, None, vce=vce,
                              cluster=clu, gamma0=True)
        row = 0
        eA = r["delta"]["A"]["coef"][row]
        eB = r["delta"]["B"]["coef"][row]
        seA = r["delta"]["A"]["se"][row]
        seB = r["delta"]["B"]["se"][row]
        et = r["total"]["coef"][row]
        set_ = r["total"]["se"][row]
        hits["delta_A"] += cover(eA, seA, dA)
        hits["delta_B"] += cover(eB, seB, dB)
        hits["total"] += cover(et, set_, tot)
        hits["delta_A_gamma0"] += cover(
            rg["delta"]["A"]["coef"][row], rg["delta"]["A"]["se"][row], dA)
        hits["delta_B_gamma0"] += cover(
            rg["delta"]["B"]["coef"][row], rg["delta"]["B"]["se"][row], dB)
        tab = gb.tidy(r, share="movement", include_total=False,
                      include_full=False)
        by = {t["component"]: t for t in tab if t["coefficient"] == "x1_1"}
        hits["share_A"] += cover(by["A"]["share"], by["A"]["share_std_error"], sA)
        hits["share_B"] += cover(by["B"]["share"], by["B"]["share_std_error"], sB)
        tab_base = gb.tidy(r, share="base", include_total=False,
                           include_full=False)
        by_base = {
            t["component"]: t for t in tab_base
            if t["coefficient"] == "x1_1"
        }
        hits["share_A_base"] += cover(
            by_base["A"]["share"], by_base["A"]["share_std_error"], sbA)
        hits["share_B_base"] += cover(
            by_base["B"]["share"], by_base["B"]["share_std_error"], sbB)
        used += 1
    return {k: hits[k] / used for k in labels}, used


def mc2(reps=REPS, seed0=90000):
    nW, T = 150, 5
    n = nW * T
    bf, be, gcoef, kf, ke, lam, tau = 0.15, 0.05, 0.5, 0.4, 0.15, 0.25, 0.6
    d_job = gcoef * kf                 # 0.20
    d_fe = bf + lam                    # 0.40
    b_base = d_job + d_fe              # 0.60
    share_job = d_job / b_base
    labels = ["total", "delta_job", "delta_FE",
              "share_job_base_fixed", "share_job_joint",
              "share_job_oracle",
              "total_robustVCE_misuse"]
    hits = {k: 0 for k in labels}
    used = 0
    for rep in range(reps):
        rng = np.random.default_rng(seed0 + rep)
        worker = np.repeat(np.arange(nW), T)
        female = rng.integers(0, 2, nW).astype(float)[worker]
        exper = np.tile(np.arange(T), nW) + rng.normal(0, 0.3, n)
        job = kf * female + ke * exper + rng.normal(size=n)
        eta = rng.normal(0, tau, nW)[worker]
        y = (bf * female + be * exper + gcoef * job + lam * female + eta
             + rng.normal(0, 0.7, n))
        x1 = np.column_stack([female, exper])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = gb.decompose(y, x1, {"job": job}, {"worker": worker},
                             vce="cluster", cluster=worker,
                             absorbed_targets=[0],
                             x1_names=["female", "exper"], focal="female")
            rr = gb.decompose(y, x1, {"job": job}, {"worker": worker},
                              vce="robust", absorbed_targets=[0],
                              x1_names=["female", "exper"], focal="female")
        row = 0
        hits["total"] += cover(r["total"]["coef"][row], r["total"]["se"][row],
                               b_base)
        hits["delta_job"] += cover(r["delta"]["job"]["coef"][row],
                                   r["delta"]["job"]["se"][row], d_job)
        hits["delta_FE"] += cover(r["delta"]["worker"]["coef"][row],
                                  r["delta"]["worker"]["se"][row], d_fe)
        hits["total_robustVCE_misuse"] += cover(
            rr["total"]["coef"][row], rr["total"]["se"][row], b_base)
        tab = gb.tidy(r, share="base_fixed", include_total=False,
                      include_full=False)
        by = {t["component"]: t for t in tab if t["coefficient"] == "female"}
        hits["share_job_base_fixed"] += cover(
            by["job"]["share"], by["job"]["share_std_error"], share_job)
        tab_joint = gb.tidy(r, share="base", include_total=False,
                            include_full=False)
        by_joint = {
            t["component"]: t for t in tab_joint
            if t["coefficient"] == "female"
        }
        hits["share_job_joint"] += cover(
            by_joint["job"]["share"],
            by_joint["job"]["share_std_error"],
            share_job)
        # Independent corrected b1x2-convention oracle.
        o = oracle(y, x1, [("job", job)], [("worker", worker)], vce="cluster",
                   cluster=worker, absorbed=[0])
        k1 = 3
        G = 2
        cov_row = o["cov"][np.ix_([g * k1 + row for g in range(G)],
                                  [g * k1 + row for g in range(G)])]
        cov_row_bb = o["cov_delta_bbase"][
            np.ix_([g * k1 + row for g in range(G)], [row])].ravel()
        se_corr = share_base_se_correct(
            o["delta"][row, :], o["b_base"][row], cov_row, cov_row_bb,
            o["V_base"][row, row])
        hits["share_job_oracle"] += cover(
            o["delta"][row, 0] / o["b_base"][row], se_corr[0], share_job)
        used += 1
    return {k: hits[k] / used for k in labels}, used


def main():
    out = {}
    for name, fn, kw in [("MC1_iid", mc1, {"clustered": False}),
                         ("MC1_clustered", mc1, {"clustered": True}),
                         ("MC2_absorbed_clustered", mc2, {})]:
        cov_rates, used = fn(**kw)
        mcse = {k: float(np.sqrt(v * (1 - v) / used))
                for k, v in cov_rates.items()}
        out[name] = {"reps": used, "coverage": cov_rates, "mc_se": mcse}
        print(f"== {name} (reps={used})")
        for k, v in cov_rates.items():
            print(f"   {k:26s} {v:6.3f}  (mc se {mcse[k]:.3f})")
    with open(os.path.join(HERE, "ws4_mc_coverage.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote ws4_mc_coverage.json")


if __name__ == "__main__":
    main()
