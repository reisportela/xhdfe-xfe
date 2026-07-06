#!/usr/bin/env python3
"""Gelbach (2016) conditional decomposition worked example for xhdfe (Python).

Decomposes the movement of a base coefficient (the return to education) between
a short regression (y on educ) and a long regression (y on educ plus mediating
channels and a firm fixed effect) into an additive, order-invariant
contribution per channel. Semantics match Gelbach's b1x2.

Run with:  python examples/gelbach_example.py
Requires the compiled xhdfe extension.
"""

import numpy as np

import xhdfe.gelbach as gelbach

rng = np.random.default_rng(20260706)
n, n_firms = 5000, 60

# educ is correlated with ability and with firm sorting, so the short-regression
# return to education is inflated by both channels.
ability = rng.normal(size=n)
educ = 0.5 * ability + rng.normal(size=n)
# sort workers into firms on a logit of education (educ-firm sorting)
latent = 0.4 * educ + rng.normal(size=n)
firm_id = np.floor((n_firms - 1) / (1 + np.exp(-latent))).astype(np.int64)
tenure = rng.normal(size=n) + 0.2 * educ
exper = rng.normal(size=n)
firmpay = rng.normal(0, 0.7, n_firms)[firm_id]
y = 0.5 * educ + 0.8 * ability + 0.1 * tenure + firmpay + rng.normal(size=n)

# Gelbach decomposition of the education coefficient into a skill channel
# (ability), a job channel (tenure, exper) and a firm fixed-effect channel.
res = gelbach.decompose(
    y, educ,
    x2_groups={"skill": ability, "job": np.column_stack([tenure, exper])},
    fes={"firm": firm_id},
)

print("base (short) educ coefficient:", round(float(res["b_base"][0]), 5))
print("full (long)  educ coefficient:", round(float(res["b_full"][0]), 5))
print("total movement:               ",
      round(float(res["total"]["coef"][0]), 5),
      "(se", round(float(res["total"]["se"][0]), 5), ")")
print("\ncontribution of each channel to the movement:")
for name, d in res["delta"].items():
    print(f"  {name:6s}: {float(d['coef'][0]):+.5f}  (se {float(d['se'][0]):.5f})")
print(f"\nsummation identity residual = {res['identity_gap']:.2e}")

# Cluster-robust inference by firm.
res_cl = gelbach.decompose(
    y, educ,
    x2_groups={"skill": ability, "job": np.column_stack([tenure, exper])},
    fes={"firm": firm_id},
    vce="cluster", cluster=firm_id,
)
print("\ncluster-robust standard errors:")
for name, d in res_cl["delta"].items():
    print(f"  {name:6s}: se {float(d['se'][0]):.5f}")

print("\ndone.")
