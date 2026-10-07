# IV first-stage diagnostics: CPU/CUDA candidate validation

**7 October 2026 — `codex/iv-first-stage-f`, unissued 2.29.0.20261007 candidate.**

The new Stata first-stage diagnostics passed 27 CPU and 24 CUDA equation-level
comparisons on 6,000 synthetic observations. The largest scaled F-statistic
difference from the direct reference was **3.80e-12**. CPU and CUDA agreed within
**2.86e-15** over all 24 shared equations. Actual CUDA execution was required.
The existing CUDA IV solver has a separate redundancy limitation described below.
This report covers this feature on Linux, not a general release, cross-platform,
or performance certification. No pull request or release has been published.

## Implemented interface

```stata
xhdfe y x1 x2 d1, absorb(firm year) endogenous(d1) instruments(z1 z2) ///
    vce(cluster state) gpubackend(cpu) ffirst
matrix list e(first)
estat firststage
```

Use `gpubackend(cuda)` for CUDA absorption. `ffirst` displays the diagnostic
summary; `first` also displays the auxiliary regressions. Both store `e(first)`.
`estat firststage` computes the diagnostics on demand when necessary and returns
`r(first)`. Matrix columns identify endogenous regressors; rows are `pr2`, `F`,
`df`, `df_r`, and `pvalue`. Replay accepts `xhdfe, first` or `xhdfe, ffirst`.

For each endogenous regressor, xhdfe fits an unrestricted first stage with
included controls, excluded instruments, and fixed effects. It tests the
excluded instruments jointly using that fit's covariance matrix. A restricted
first stage with controls and fixed effects yields partial R-squared as
`1 - RSS_unrestricted / RSS_restricted`. Both fits use the final IV sample,
weights, inference settings, and selected backend. These additional fits are
performed only when requested.

These are per-equation instrument-relevance tests. Sanderson-Windmeijer
conditional tests, Kleibergen-Paap statistics, and overidentification tests are
not implemented by this change.

## Numerical results

| Check | CPU | CUDA |
| --- | ---: | ---: |
| Specifications completing the full contract | 21 | 18 |
| First-stage equations compared to a direct reference | 27 | 24 |
| Equations also compared to available `ivreghdfe` statistics | 23 | 21 |
| Maximum scaled F difference, direct reference | 3.791e-12 | 3.791e-12 |
| Maximum scaled F difference, `ivreghdfe` with documented df adjustment | 3.789e-12 | 3.789e-12 |
| Maximum coefficient `mreldif` across independent IV fits | 0 | 6.458e-15 |
| Maximum covariance `mreldif` across independent IV fits | 0 | 3.883e-14 |
| Maximum diagnostic `mreldif` across independent fits | 0 | 2.728e-15 |

The scaled F difference is `abs(F_xhdfe - F_reference) / max(1, abs(F_reference))`.
The direct reference is `reghdfe` followed by `testparm` with absorbed effects,
and Stata `regress` followed by `testparm` without absorbed effects. All
numerator and denominator degrees of freedom agreed exactly. The tests require
F differences below 1e-7 on this scale and p-value/partial-R-squared differences
below 1e-9. Independent-fit matrix comparisons require `mreldif < 1e-12`.
No native precision threshold was changed.

Representative F-statistics, rounded only for this table:

| Specification / endogenous variable | xhdfe CPU | xhdfe CUDA | Direct reghdfe | ivreghdfe, raw |
| --- | ---: | ---: | ---: | ---: |
| IID / d1 | 900.454315205 | 900.454315205 | 900.454315205 | 900.454315205 |
| Robust / d1 | 895.117090841 | 895.117090841 | 895.117090841 | 895.117090841 |
| Cluster state / d1 | 1235.938813034 | 1235.938813034 | 1235.938813034 | 1235.938813034 |
| Cluster firm, nested FE / d1 | 916.843703006 | 916.843703006 | 916.843703006 | 916.996893265 |
| Two endogenous variables, cluster state / d2 | 634.116841743 | 634.116841743 | 634.116841743 | 634.116841743 |
| Weak instruments / weak | 2.149358118 | 2.149358118 | 2.149358118 | 2.149358118 |

When an absorbed effect is nested in clusters, reghdfe uses an extra intercept
degree of freedom in its finite-sample correction. Consequently,
`F_xhdfe = F_ivreghdfe * (N-K-1)/(N-K)`, with `K = e(df_m) + e(df_a)` from
the direct reghdfe first stage. The nested-firm example uses the factor
0.9998329435349148. The test checks this relationship explicitly and also checks
unadjusted agreement with the direct reghdfe result. The underlying convention
is documented in the [ivreghdfe README](https://github.com/sergiocorreia/ivreghdfe#recent-updates).

The two-way state/year example has only ten year clusters. In this fixture,
ivreghdfe reports a singular system moment covariance and missing first-stage
F-statistics. The contract explicitly verifies that missing reference result;
xhdfe's per-equation F-statistics are checked against direct reghdfe tests.
Redundant-instrument cases also use direct reghdfe as their oracle.

## Coverage and preservation of estimation results

The suite includes one and two endogenous variables; IID, robust, one-way and
two-way clustered errors; analytic, frequency and probability weights;
interaction fixed effects; weak instruments; no included controls; zero
instrument columns; missing outcomes/instruments; and singleton removal.
CPU additionally covers no-absorb models with and without an intercept, and a
combined proportional-duplicate/repeated-control instrument case.

After `estat firststage`, coefficients, covariance matrices, sample flags, and
the IV command line must remain exactly unchanged. Full display/replay and
restoration after a failed diagnostic also pass exact matrix checks. Saved IV
residuals remain unchanged. Cached summaries are identical, `nosample` summaries
work when requested during estimation, and unsupported postestimation requests
fail with the expected errors. Non-default `statstyle()` and `ssc()` settings
are checked against direct xhdfe first-stage fits.

For each CUDA specification, the contract asserts `e(gpu_used)==1`,
`e(gpu_backend)=="cuda"`, and `e(gpu_status)=="used"` for the IV fit.
Both unrestricted and restricted first-stage fits must also report actual GPU
use through `r(first_gpu_used)`. CUDA requires at least one fixed-effect
dimension, so the two ordinary IV specifications are CPU-only.

## Follow-up: tighter tolerances and an explicit-dummy reference

The probability-weighted specification produced the largest F difference in
the original comparison. The targeted
[precision probe](../../tests/stata/iv_firststage_precision_probe.do) reruns
that same synthetic fixture at tolerances 1e-8, 1e-12, and 1e-14, using CPU and
actual CUDA. It also fits Stata `regress` with explicit firm/year dummies and
the same probability weights and state-clustered VCE. The explicit-dummy
reference avoids iterative fixed-effect absorption.

| Tolerance | Absolute F difference: CPU vs reghdfe | Absolute F difference: CUDA vs reghdfe | Maximum CPU/CUDA F difference from explicit dummies |
| --- | ---: | ---: | ---: |
| 1e-8 | 5.99289e-9 | 5.99334e-9 | 4.77485e-12 |
| 1e-12 | 5.99266e-9 | 5.99175e-9 | 5.91172e-12 |
| 1e-14 | 5.99266e-9 | 5.99221e-9 | 5.45697e-12 |

The reference F is 1580.845842736720 with explicit dummies, versus
1580.845842742718 from reghdfe. Tightening tolerance leaves the small discrepancy
essentially unchanged. Reghdfe takes four iterations at all three settings;
xhdfe takes four, five, and six. All fits use the same 6,000 observations,
29 clusters, and F degrees of freedom `(2, 28)`; the absorbed fits agree on
129 absorbed degrees of freedom.

For the four common slopes `x1 x2 z1 z2`, the maximum absolute coefficient
difference between xhdfe and reghdfe is 9.99201e-16; the maximum absolute
covariance-entry difference is 1.63052e-15. Against the explicit-dummy fit,
xhdfe agrees within 4.44089e-16 in coefficients and 2.98156e-19 in covariance
entries. The coefficient/covariance comparison excludes the intercept because
the explicit-dummy parametrization uses a different intercept normalization.
The reported first-stage diagnostic F and a separate xhdfe first-stage `test`
agree within 1.36425e-12 across all six CPU/CUDA fits.

This evidence points to a tiny difference in clustered covariance evaluation.
The independent dummy-variable calculation agrees substantially more closely
with xhdfe in this fixture. The check supports retaining the implementation and
default tolerance; it does not justify changing the estimator to reproduce the
last digits of the other command. The exact arithmetic operation responsible
inside the reference implementation was not investigated further.

All assertions passed with the marker `IV_FIRSTSTAGE_PRECISION_PASS`.
The probe uses the same ado/plugin hashes recorded below. Executed scripts,
CSV summaries, full-precision result matrices in `.dta`, and a log with its
license banner removed are retained in the coordination workspace under
`pipeline/results/iv_firststage_precision_20261007/`; the remote run is
`/srv/projetos/P008/04_to_test_xhdfe/runs/iv_firststage_precision_20261007`.
No estimator code or default setting changed during this follow-up.

## Existing CUDA IV redundancy limitation

Two extra probes reproduced problems **without any first-stage diagnostics**
using unchanged xhdfe 2.28.1 ado code and the same native plugin:

- Adding a proportional duplicate `zdup = 2*z1` caused the CUDA IV fit to fail
  with return code 198, reporting rank-deficient FWL-residualized instruments,
  on all five repeated fits.
- Repeating included control `x1` inside `instruments(z1 z2 zzero x1)` produced
  coefficient `mreldif` as large as 2.696e-4 across five repeated CUDA fits.
  Ordinary `instruments(z1 z2)` and `instruments(z1 z2 zzero)` repeated within
  1.982e-16 and 1.279e-16 respectively.

The candidate encountered the same problems before its diagnostic code ran.
These CUDA configurations are **not counted as successful comparisons**.
The permanent test logs `NOT_TESTED redundant_duplicate cuda`; zero-column
redundancy is tested successfully on both backends. Use independent excluded
instruments on CUDA and avoid repeating included controls. Repairing the native
redundancy handling is separate work; this branch leaves the native solver
unchanged and documents the limitation in Stata help.

## Reproduction and provenance

Run [the standalone contract](../../tests/stata/iv_firststage_contract.do) in a
scratch directory with the candidate on the adopath and its matching plugin:

```bash
XHDFE_STATA_ADOPATH=/path/to/candidate/stata stata-mp -q -b do /path/to/tests/stata/iv_firststage_contract.do cpu
XHDFE_STATA_ADOPATH=/path/to/candidate/stata stata-mp -q -b do /path/to/tests/stata/iv_firststage_contract.do cuda
```

Require the exact terminal markers `IV_FIRSTSTAGE_CONTRACT_PASS cpu` and
`IV_FIRSTSTAGE_CONTRACT_PASS cuda`; Stata's shell exit code alone is insufficient.
Each run exports the full comparison CSV. The CPU contract is included in
`tests/stata/run_stata_tests.sh`.

- Host: Athena, Linux x86_64, StataNow/MP 19.5.
- GPU: NVIDIA H100 NVL; NVIDIA driver 570.195.03.
- Installed references: ivreghdfe 1.1.3 (04Jan2023), reghdfe 6.12.5 (27Dec2023),
  ivreg2 4.1.12 (14Aug2024).
- Private branch base: `516ceca139985522ee3c302ef90bdd038a593679`.
- Public branch base: `f71420bc70893f37d42a2d1ef8a118bf0cabebc3`.
- Native plugin reused unchanged from the existing 2.28.1 H100 build; all 53
  C++/CUDA/header source files under `stata/src` and `stata/include` match its
  saved source tree byte-for-byte. This ado-only feature required no rebuild.
- Plugin SHA256:
  `3f4be3983be8e32756e181a5ecab6014b5d195a3043d87fd4f9e433c17267dc6`.
- Candidate `stata/xhdfe.ado` SHA256:
  `684601075f558214e36fc410c45d6d543baef3848b9def64aa13b4e11e59fc2a`.
- Candidate `stata/xhdfe_estat.ado` SHA256:
  `bc82c3019cbb302b725ce3e63f4c10f1ac85d53461b5950dc95dd565d04c8db1`.
- Test script SHA256:
  `2c3f167bef9f4756de5d7bb9a0e13a29cee2f05a5a8fc61383047d9d66c41e59`.

Remote artifacts are retained under
`/srv/projetos/P008/04_to_test_xhdfe/runs/iv_firststage_20261007`.
The coordination workspace holds derived CSVs, logs with the license banner
removed, `summary.json`, source hashes, and the baseline redundancy probe under
`pipeline/results/iv_firststage_20261007/`. All data used here were generated by
the test; no restricted research data were accessed.
