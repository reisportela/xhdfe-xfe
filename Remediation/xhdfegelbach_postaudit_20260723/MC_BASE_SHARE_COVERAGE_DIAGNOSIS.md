# Joint base-share Monte Carlo diagnosis

Date: 23 July 2026

Scope: evidence only. No production, audit-trail, or literature file was
changed in this workstream.

## Algebra checked

For a reported base share \(s_g=\delta_g/b_{\mathrm{base}}\), the joint
delta-method variance is

\[
\operatorname{Var}(s_g)=
  \frac{\operatorname{Var}(\delta_g)}{b_{\mathrm{base}}^2}
  +\frac{\delta_g^2\operatorname{Var}(b_{\mathrm{base}})}
         {b_{\mathrm{base}}^4}
  -\frac{2\delta_g\operatorname{Cov}(\delta_g,b_{\mathrm{base}})}
         {b_{\mathrm{base}}^3}.
\]

Under robust or clustered inference, the cross block in the corrected
b1x2-convention oracle is

\[
q_{\mathrm{big}}\left[
P C_{a,g}'C_b P+
\Gamma_g A_F^{-1}C_f'C_bP
\right],
\]

where the second term is absent for `gamma0`/conditional FE blocks and
\(q_{\mathrm{big}}=N/(N-1)\) for robust or \(G/(G-1)\) for cluster. The audit
copy omitted `q_big` from this cross block even though it applied the same
factor to the corresponding auxiliary and full/aux blocks of
`Var(delta)`. The copied oracle under `Remediation/` was corrected; the audit
trail was not modified.

The deterministic grid in
`repros/ws4/oracles/ws4_oracle_comparison_raw.json` then matches the
implementation cross covariance at roughly \(10^{-15}\) in the ordinary
fixtures (and at the documented absorption tolerance in the two-FE fixture).
For absorbed targets, the authoritative identity
`Cov(total, b_base) == Var(b_base)` is exact on the target row.

## Why 100 replications looked bad

The original 100-replication clustered MC used 40 clusters and normal
critical values. Its second base share covered only 0.89, with a binomial
Monte Carlo standard error of 0.031. Re-running the exact DGP and exact
seed sequence for 5,000 replications gives:

| Design | Share A coverage | Share B coverage | MC SE |
|---|---:|---:|---:|
| iid, audit seed, 800 observations | 0.9498 | 0.9480 | about 0.0031 |
| cluster, audit seed, 40 x 20 | 0.9372 | 0.9330 | about 0.0034 |
| cluster, independent seed, 40 x 20 | 0.9340 | 0.9390 | about 0.0035 |
| cluster, independent seed, 120 x 20 | 0.9458 | 0.9468 | about 0.0032 |

Thus the 0.89 result combines two effects:

1. substantial simulation noise from only 100 replications; and
2. real, modest finite-cluster undercoverage of a normal-reference CR1
   interval with 40 clusters.

The second effect is not a covariance-algebra failure. At 40 clusters the
mean reported SE is about 0.97 times the empirical SD. At 120 clusters the
ratio is about 0.995 and coverage is 94.6--94.7%. Small-cluster
randomization/bootstrap inference is explicitly outside this remediation
tranche, so the estimator and critical-value contract were not changed.

The absorbed-target 100-replication result of 0.90 is simulation noise, not
the 40-cluster issue: that DGP has 150 independent worker clusters. The same
implementation and the same seed sequence over 5,000 replications produce
0.954 coverage (MC SE 0.0030), mean-SE/empirical-SD 1.0029, and zero maximum
error in the required `SE(total) == SE(b_base)` identity.

## Acceptance evidence

The 5,000-replication focused implementation MC gives:

| DGP/VCE | Share A coverage | Share B coverage |
|---|---:|---:|
| iid, unadjusted | 0.9482 | 0.9428 |
| iid, robust | 0.9556 | 0.9486 |
| clustered, 120 clusters | 0.9458 | 0.9468 |
| absorbed target, 150 worker clusters, job share | 0.9540 | n/a |

All required iid and adequately-clustered base-share experiments therefore
meet the 0.94 gate. The deliberately retained 40-cluster sensitivity does
not, and is reported rather than hidden.

## Reproduction

The scripts accept a just-built extension without replacing the repository
artifact:

```bash
XHDFE_GELBACH_EXTENSION=/path/to/py_hdfe_v11.cpython-312-x86_64-linux-gnu.so \
python3 repros/ws4/oracles/base_share_coverage.py \
  --reps 5000 --clusters 40 120 --rows-per-cluster 20

XHDFE_GELBACH_EXTENSION=/path/to/py_hdfe_v11.cpython-312-x86_64-linux-gnu.so \
python3 repros/ws4/oracles/absorbed_share_coverage.py --reps 5000
```

Machine-readable evidence:

- `repros/ws4/oracles/base_share_coverage.json`
- `repros/ws4/oracles/base_share_coverage_100.json`
- `repros/ws4/oracles/absorbed_share_coverage.json`
- `repros/ws4/oracles/absorbed_share_coverage_100.json`
- `repros/ws4/oracles/ws4_oracle_comparison_raw.json`
