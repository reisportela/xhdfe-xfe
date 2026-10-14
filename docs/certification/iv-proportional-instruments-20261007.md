# Proportional IV instruments on CUDA — 7 October 2026

Candidate **2.29.2.20261007**, on `codex/iv-first-stage-f`, repairs the
reproduced CUDA refusal for exactly proportional instrument columns. The
focused repair checks pass. Broader testing also reproduces two pre-existing
multiway-covariance edge-case failures, with identical outputs before and after
this change. This is scoped development validation, not release certification.

## Cause and repair

An instrumented copy of the unchanged native IV helper reproduces the failure
with 6,000 synthetic observations, two fixed effects and `zdup=2*z1`.
The original columns are exactly proportional. CPU absorption preserves the
relation; CUDA absorption leaves a maximum discrepancy of
`8.8817841970012523e-16`. The ordinary rank check finds four directions, but
the wider-precision check counts that absorbed rounding error as a fifth
(last pivot about `8.45e-17`). The final FWL check then rejects three excluded
columns with rank two. This is an **undue verifier refusal**, not evidence that
the regression is underidentified or needs tighter absorption targets.

`src/iv_input.hpp` removes nonzero user-instrument columns exactly proportional
to an earlier retained column, using the original estimation sample after
singleton and positive-weight filtering. `HdfeRegressorV11::fit` applies it
before absorption. Consequently the redundant column never reaches the rank
check that mistakes transformation error for identifying information.

Exact binary64 cross-products use bounded, portable integer products, with
no epsilon, division, overflow/underflow-prone floating multiplication, or
arbitrary-precision dependency. Independent pairs normally exit after three
sampled rows; a possible proportional pair is checked on every row. A one-bit
difference anywhere preserves the column. Full scans respect the resolved
OpenMP team and work observer. Independent inputs allocate no replacement
N-by-k matrix. Rank tolerances, convergence targets, iteration budgets and
estimator definitions are unchanged.

The repair is deliberately limited to pairwise proportional raw instruments.
It does not establish arbitrary linear dependencies involving several columns,
or dependencies created only by fixed-effect absorption. Zero columns retain
their existing handling. Underidentified models must still fail.

## Targeted results

| Permanent contract | CPU | CUDA |
| --- | ---: | ---: |
| Stata proportional instruments, three placements and five repetitions | 210 fits PASS | 180 fits PASS |
| Stata repeated-control regression guard | 210 fits PASS | 180 fits PASS |
| First-stage diagnostics versus reghdfe / ivreghdfe | 29 equations PASS | 27 equations PASS |
| Independent proportional / near-distinct Walsh fixtures | 28 cases PASS | 28 cases PASS |
| Existing projection-conditioning contract | 128 cases PASS | 38 cases PASS |
| Saved-FE IV / existing cache contract, now including proportional columns | 38 cases PASS | 32 cases PASS |

Every accepted CUDA row checks actual GPU use. The Stata proportional tests
also compare structural slopes with independent explicit-dummy `ivregress
2sls` and ivreghdfe; self-instrumentation uses its OLS equivalent. They cover
robust and clustered covariance, analytic/frequency/probability weights,
multiple endogenous variables, missing values, singleton removal, zero
columns and weak instruments. CPU additionally covers no absorption and no
constant. Sample, degrees of freedom, RSS, covariance, metadata and failed-fit
state are checked. An insufficient distinct instrument span is rejected.

For the original unadjusted Stata reproducer, the largest coefficient change
from adding/reordering proportional columns is **2.22e-16 on CPU** and
**4.44e-16 on CUDA**. Maximum covariance-entry changes are **5.42e-20** and
**4.34e-19**, respectively. Across the entire proportional battery, including
weak instruments, maxima are **5.365e-13 / 1.918e-13** for coefficients and
**6.353e-12 / 1.112e-11** for covariance entries, CPU / CUDA.

The repaired proportional case reports F = **1350.851651882072** on both
backends; the direct reghdfe reference reports **1350.851651882074**, with two
restrictions and denominator df 5,867. Across the whole first-stage battery,
the maximum F difference remains **5.993115337332711e-9**, the previously
investigated probability-weighted rounding discrepancy. No test threshold was
relaxed.

Additional guards pass:

- 4,396 independent rational-arithmetic proportionality cases per OpenMP team
  (1, 2, 8, 16, 24 and 48), including extreme binary64 scales, subnormals,
  signed zeros and one-bit distinctions; all requested workers are observed.
  The same production-header test passes with Apple clang on macOS, serial.
- The existing CPU instrument-span contract: 24 valid cases plus one expected
  underidentification refusal, including general redundant combinations.
- The native underidentification guard: 30 correctly rejected cases and 30
  accepted valid companion cases.

## Broader covariance failures retained

The results below describe the frozen 2.29.2 candidate. They were subsequently
repaired and rechecked in [2.29.3](iv-covariance-precision-20261007.md); the
original failing observations remain part of the evidence.

The unchanged independent high-precision intercept/covariance contracts each
return **44 PASS / 2 FAIL**, in both tolerance modes on both CPU and CUDA:
176 passing and eight failing candidate rows in total. The two distinct
fixtures have `delta=2^-27`, zero regressor means and equivalent two-way cluster
partitions, with either no weights or frequency weights. Coefficients and
residuals pass; the intercept covariance fails, with diagonal-scaled full-V
errors approximately **8.641e-4** and **2.495e-3**.

All 184 cases were rerun on the unchanged frozen 2.28.1 native module.
Coefficients, covariance matrices, failed checks and pass/fail outcomes are
**identical** to the candidate in every case. These are pre-existing numerical
covariance failures, classified as demonstrated estimator errors relative to
the independent oracle; they are not regressions caused by this filter. Their
cause and repair remain a separate issue. Failed reports remain in the
evidence inventory and are not counted as successes or removed from the suite.

## Harness corrections and provenance

The new multiple-endogenous weighted CUDA fixture initially reused weights
constant over only 16 rows. Its extra Walsh directions require a complete
64-row cycle to preserve the reference score's orthogonality to fixed effects.
That invalid combination produced 12 failed covariance checks. This was an
**evaluation-harness error**: the corrected fixture uses full-cycle weights and
asserts score-space, FE and residual orthogonality before fitting. Original
results and the initial script are retained. Estimator code and numerical
acceptance limits did not change after this finding.

The first covariance-contract launches lacked `mpmath`; these are recorded as
**missing coverage**, not estimator failures. Mpmath 1.3.0 was installed in the
isolated run's dependency folder, leaving the timing campaign's environment
unchanged, and all four contracts were executed as described above.

Evidence is under workspace `pipeline/results/iv_proportional_fix_20261007/`,
including `summary.json`, per-case CSV/JSON reports, sanitized Stata logs,
failed runs, trace logs, source hashes and reproducible build/run scripts.
The Athena run is
`/srv/projetos/P008/04_to_test_xhdfe/runs/iv_proportional_fix_20261007`.

The isolated incremental build recompiled the changed native translation unit
with the respective frozen CMake and Stata flags, including their different
Eigen alignment settings. It reused only unchanged translation-unit objects
from the frozen 2.28.1 build. A source inventory compared 105 existing files in
`src`, `include`, `stata/src` and `stata/include`: only the two copies of
`hdfe_regressor_v11.cpp` changed, and each received the new `iv_input.hpp`.
The existing CUDA verifier objects retain their original strict-FP build.
No frozen campaign source, artifact or environment was modified.

- Python native module SHA256:
  `1186a265f7ebc964737a934a554a9e23de43157f9aa6ab5319bc631ca975bd63`.
- Stata CUDA plugin SHA256:
  `24193fb4a091789fb5a8cc97e12a883993d4b2a442477f8323c78d78dea361c1`.
- StataNow/MP 19.5; NVIDIA H100 NVL; Linux CUDA target sm_90.
- Parent checkpoints: private `bcc315b0`, public `c1b7ca38`.

The user explicitly authorized these small diagnostics during a separate
Athena timing campaign. Start/finish timestamps and process snapshots record
the overlap; affected timing results may need repeating. These checks provide
no performance evidence. Full core24-by-eight performance validation,
macOS/Windows estimator builds, R runtime checks and release artifact
certification remain unexecuted. The configured active-checkout gate still
refers to absent `/home/mangelo` paths and a stale approved-state manifest;
actual Mac branches and tested Athena source/artifact hashes were checked
without altering that manifest. The candidate remains unissued.
