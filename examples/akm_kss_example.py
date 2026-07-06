#!/usr/bin/env python3
"""AKM + leave-out (KSS) worked example for xhdfe (Python front-end).

Two-way worker-firm estimation, leave-out variance decomposition, component
standard errors, Andrews-Mikusheva confidence intervals, the Andrews et al.
(2012) subsampling diagnostic, and interoperability with pytwoway /
LeaveOutTwoWay.

Numerical semantics follow Kline, Saggio & Solvsten (2020) as implemented in
Saggio's LeaveOutTwoWay (the canonical KSS reference); the leave-out set and
components are validated against it and against pytwoway.

Run with:  python examples/akm_kss_example.py
Requires the compiled xhdfe extension (pip install . from the repo root).
"""

import numpy as np

import xhdfe.akm as akm


def simulate_akm_panel(n_workers=400, n_firms=40, reps=5, seed=20260706):
    """A reproducible mover panel: every worker changes firm each period."""
    rng = np.random.default_rng(seed)
    worker = np.repeat(np.arange(n_workers), reps)
    firm = np.empty(n_workers * reps, dtype=np.int64)
    pos = 0
    for _ in range(n_workers):
        f = int(rng.integers(n_firms))
        for _ in range(reps):
            firm[pos] = f
            pos += 1
            nxt = int(rng.integers(n_firms - 1))
            f = nxt if nxt < f else nxt + 1
    alpha = rng.normal(0, 0.6, n_workers)
    psi = rng.normal(0, 0.4, n_firms)
    x1 = rng.normal(size=n_workers * reps)
    y = alpha[worker] + psi[firm] + 0.3 * x1 + rng.normal(size=worker.size) * 0.5
    return y, worker, firm, x1


y, worker, firm, x1 = simulate_akm_panel()

# 1. Sample preparation: the leave-one-out connected set (KSS-identified).
s = akm.leave_out_set(worker, firm)
print(f"leave-out sample: {s['n_obs']} obs, {s['n_workers']} workers "
      f"({s['n_movers']} movers), {s['n_firms']} firms")

# 2. AKM two-way estimation + variance decomposition. Default: leave-a-match-out,
#    exact leverages on a small sample (auto-switches to JLA on large data).
r = akm.akm_kss(y, worker, firm, leverages="exact")
kss = r["kss"]
print("\nKSS-corrected decomposition:")
print(f"  var(psi)        = {kss['var_psi']:.5f}")
print(f"  var(alpha)      = {kss['var_alpha']:.5f}")
print(f"  cov(alpha,psi)  = {kss['cov_alpha_psi']:.5f}")
print(f"  corr(alpha,psi) = {kss['corr_alpha_psi']:.5f}")
print(f"  shares of var(y): alpha {kss['share_var_alpha']:.3f}, "
      f"psi {kss['share_var_psi']:.3f}, 2cov {kss['share_2cov']:.3f}")

# 3. Component standard errors + Andrews-Mikusheva weak-id confidence intervals.
r = akm.akm_kss(y, worker, firm, leverages="exact",
                compute_se=True, eigen_diagnostics=True)
se = r["component_se"]
wk = r["weak_id"]["var_psi"]
print("\nInference on var(psi):")
print(f"  KSS point est.  = {se['theta_var_psi']:.5f}")
print(f"  standard error  = {se['se_var_psi']:.5f}")
print(f"  AM 95% CI       = [{wk['ci_lb']:.5f}, {wk['ci_ub']:.5f}]")
print(f"  F statistic     = {wk['f_stat']:.3f} (curvature {wk['curvature']:.3f})")

# 4. Controls partialled out (FWL); pass gpu=True for the CUDA backend.
r = akm.akm_kss(y, worker, firm, X=x1.reshape(-1, 1),
                leverages="jla", jla_draws=200)
print(f"\ncontrol coefficient on x1: {float(np.asarray(r['beta'])[0]):.5f}")

# 5. Andrews-Gill-Schank-Upward (2012) subsampling diagnostic: components should
#    stay flat as movers are dropped once bias is corrected.
diag = akm.subsampling_diagnostic(y, worker, firm,
                                  fractions=(0.0, 0.25, 0.5), n_reps=2)
print("\nsubsampling diagnostic (KSS var_psi vs mover fraction dropped):")
for rec in diag:
    print(f"  drop {rec['fraction']:.2f}: kss var_psi = {rec['kss']['var_psi']:.5f}")

# 6. Interoperability with pytwoway / LeaveOutTwoWay (export the leave-out set).
frame = akm.to_pytwoway_frame(y, worker, firm)   # columns i, j, y, t
print(f"\npytwoway-ready long frame: {len(frame)} rows, columns {list(frame.columns)}")
# akm.export_leaveout_csv("leaveout.csv", y, worker, firm)  # LeaveOutTwoWay input

print("\ndone.")
