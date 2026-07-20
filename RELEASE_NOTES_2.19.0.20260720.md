# xhdfe 2.19.0.20260720 — state-of-the-art Gelbach decomposition

This minor release turns `xhdfegelbach` into a cross-language empirical
workflow for linear coefficient-movement accounting while preserving the
established xhdfe estimator path. It adds a distinct, opt-in constrained
estimand for targets absorbed by fixed effects, richer reporting, and
certification-oriented build guards. Gelbach results remain specification
accounting; the command does not identify causal mediation or mechanisms.

## Absorbed-target allocation

- Python `absorbed_targets=`, R `absorbed_targets=`, and Stata
  `absorbedtargets()` implement the same constrained estimand when a declared
  X1 target belongs to the span of an added fixed effect.
- The full-model target coefficient is imposed at zero and labelled
  `imposed_zero`; it is never presented as an estimated within-FE coefficient.
- The backend fails closed unless every declared target is omitted specifically
  because of the absorbed FEs and every undeclared X1/X2 column is identified.
- The Gelbach summation identity remains exact. The standard estimand and its
  arithmetic branch are unchanged; an A/B oracle found 0 differences in 240
  standard-path output blocks.
- For a declared target, `total = b_base - 0` and its covariance is equal by
  construction to the requested base-model covariance. Cluster-FE Monte Carlo
  coverage was 0.952 in 500 repetitions.
- Inference is certified only when clustering matches an FE dimension that
  absorbs every declared target. Unadjusted, robust, or crossed clustering is
  retained for descriptive accounting but emits a prominent warning and an
  explicit invalid-inference status.

## Empirical reporting across Python, R, and Stata

- `focal` / `focal()` selects displayed coefficients without changing either
  model or any full-precision result.
- Signed component shares support movement, base-coefficient, and explicitly
  labelled fixed-base-denominator conventions. Negative shares and totals above
  100 percent are preserved rather than renormalized.
- Python `gelbach.tidy()` / `gelbach.contrast()` and R
  `xhdfe_gelbach_tidy()` / `xhdfe_gelbach_contrast()` expose publication-ready
  rows and joint-covariance linear combinations.
- Stata now prints one integrated panel per focal coefficient, marks
  `0 (imposed)` in the coefficient row, and stores the full matrices and
  metadata in `r()`.
- Cross-frontend metadata now uses one contract: zero-based backend indices,
  presentation names, an absorbed mask, identification status, inference
  status, effective N, and singleton counts.
- Six executable examples cover standard and absorbed-target applications in
  Python, R, and Stata. Help files document every option, result object,
  estimand, covariance layout, example, warning, and deliberate limitation.

## Correctness and provenance hardening

- The absorbed-target estimand is checked against an external constrained-LSDV
  Stata oracle, in addition to the established official `b1x2` oracle for the
  standard estimand.
- Empty X1 now raises a catchable error in Release builds instead of reaching an
  Eigen assertion or undefined behavior.
- CMake defaults to Release and refuses non-Release production builds unless an
  explicit diagnostic opt-out is supplied. Plugin build scripts independently
  reject live `__assert_fail` references and missing Linux OpenMP linkage.
- Eigen is pinned to the vendored in-repository 3.4.0 headers; artifact
  provenance no longer depends on a sibling checkout or host installation.
- The local production plugins were rebuilt with `-DNDEBUG`, `-O3`,
  `-march=native`, OpenMP, and H100 `sm_90`. Public release workflows continue
  to build CPU-only Linux, Windows, and macOS plugins from the identical
  sources on native/declared runners.
- `xhdfeakm` now reports a bounded count when many non-stayer rows hit the
  unit-leverage guard, preventing diagnostic text from exceeding Stata limits;
  no KSS estimate, tolerance, or convergence decision changed.
- Copyright-restricted local papers under `literature/` are now protected by
  `.gitignore` and are not distributed.

## Validation and performance

- Gelbach core oracle, cross-frontend parity, help contract, AKM/KSS validator,
  C++ mirror alignment, complete R suite, and the 28-file Stata suite passed on
  Release artifacts.
- All shipped local artifacts have zero dynamic `__assert_fail` references;
  both Stata plugins link `libgomp`, and both local CUDA artifacts contain
  `sm_90` only.
- A 500,000-observation CPU/CUDA smoke used the real H100
  (`gpu_used=1`, backend `cuda`, status `used`) and preserved CPU/GPU parity.
- The shared host did not provide a quiet timing window. Five interleaved A/B
  pairs nevertheless excluded a standard-path slowdown of 30 percent or more
  in all unadjusted/cluster, weighted/unweighted CPU and CUDA cells
  (one-sided sign test `p=0.03125` per cell). This is a catastrophic-regression
  guard, not a claim that smaller timing differences are absent.

Versions: shared C++/Python/R package `2.19.0.20260720`; Stata `xhdfe`
`2.19.0`; `xhdfegelbach` `1.3.0`; `xhdfeakm` `1.7.2`;
`xhdfeconnected` remains `1.2.1`; `xfe` remains `1.10.1`. All production Stata
files carry the common release date `20jul2026`.
