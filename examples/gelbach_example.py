#!/usr/bin/env python3
"""Gelbach (2016) conditional decomposition worked example for xhdfe (Python).

Decomposes the movement of a base coefficient (the education coefficient)
between a short and a long regression into additive, order-invariant
contributions from declared covariate and fixed-effect blocks. This is
specification accounting, not causal mediation. Semantics match Gelbach's
b1x2.

Run with:  python examples/gelbach_example.py
Requires the compiled xhdfe extension.
"""

import numpy as np

import xhdfe.gelbach as gelbach

rng = np.random.default_rng(20260706)
n, n_firms = 5000, 60

# Education, job covariates and firm assignment share latent determinants, so
# they are correlated. The example describes coefficient movement only; it does
# not give educ, ability or firm assignment a validated causal ordering.
ability = rng.normal(size=n)
educ = 0.5 * ability + rng.normal(size=n)
age = 35 + 3 * ability + 5 * rng.normal(size=n)
latent = 0.4 * ability + rng.normal(size=n)
firm_id = np.floor((n_firms - 1) / (1 + np.exp(-latent))).astype(np.int64)
tenure = rng.normal(size=n) + 0.2 * ability
exper = rng.normal(size=n)
firmpay = rng.normal(0, 0.7, n_firms)[firm_id]
y = 0.5 * educ + 0.8 * ability + 0.1 * tenure + firmpay + rng.normal(size=n)

# Gelbach decomposition into an ability block, a job-covariate block and a
# firm fixed-effect block. Age remains a common control in both specifications;
# focal selects only the publication-facing coefficient.
res = gelbach.decompose(
    y, np.column_stack([educ, age]),
    x2_groups={"ability": ability,
               "job_covariates": np.column_stack([tenure, exper])},
    fes={"firm_fe": firm_id},
    x1_names=["educ", "age"], focal="educ",
)

print("base (short) educ coefficient:", round(float(res["b_base"][0]), 5))
print("full (long)  educ coefficient:", round(float(res["b_full"][0]), 5))
print("total movement:               ",
      round(float(res["total"]["coef"][0]), 5),
      "(se", round(float(res["total"]["se"][0]), 5), ")")
print("\ncontribution of each declared block to the movement:")
for name, d in res["delta"].items():
    print(f"  {name:6s}: {float(d['coef'][0]):+.5f}  (se {float(d['se'][0]):.5f})")
print(f"\nsummation identity residual = {res['identity_gap']:.2e}")
print("absorbed-FE aggregate (conditional/gamma0 SE):",
      round(float(res["fe_total"]["coef"][0]), 5),
      "(se", round(float(res["fe_total"]["se"][0]), 5), ")")
print("interpretation: coefficient-movement accounting; not causal mediation")

# Publication-ready signed shares. Movement shares use the stored joint
# covariance and may legitimately be negative or exceed 100 percent.
rows = gelbach.tidy(res, share="movement")
print("\nsigned shares of education-coefficient movement:")
for row in rows:
    if row["component_kind"] in ("x2", "fe"):
        print(f"  {row['component']:16s}: {100 * row['share']:+.2f}% "
              f"(se {100 * row['share_std_error']:.2f} pp)")

# Any linear combination of blocks uses the same joint covariance.
observed = gelbach.contrast(res, "educ", ["ability", "job_covariates"])
print("observed-block subtotal:", round(observed["estimate"], 5),
      "(se", round(observed["std_error"], 5), ")")

# Cluster-robust inference by firm.
res_cl = gelbach.decompose(
    y, np.column_stack([educ, age]),
    x2_groups={"ability": ability,
               "job_covariates": np.column_stack([tenure, exper])},
    fes={"firm_fe": firm_id},
    vce="cluster", cluster=firm_id,
    x1_names=["educ", "age"], focal="educ",
)
print("\ncluster-robust standard errors:")
for name, d in res_cl["delta"].items():
    print(f"  {name:6s}: se {float(d['se'][0]):.5f}")

print("\ndone.")
