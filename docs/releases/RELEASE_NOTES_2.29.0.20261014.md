# xhdfe 2.29.0.20261014

First public 2.29 release, superseding 2.28.1.20260929; cut on 14oct2026, with
every production file stamped with that date. Development labels used before
this release (2.29.0.20261006 delivered privately for testing, the PR #13
series 2.29.0–2.29.5, the integration labels 2.29.6 and 2.29.0.20261013) were
never published; their notes remain as development history.

## Performance: maximum-performance absorption

The absorption engine was rebuilt for throughput without changing the
estimator, the stopping rules or the two tolerance contracts:

- approximate-Cholesky preconditioner built in an arena with parallel
  aggregation (bit-identical factor), lockstep multi-RHS MLSMR solve with
  explicit fused multiply-add kernels (team-invariant results), fused packed
  Gauss–Seidel, parallel singleton scan, indexers, DOF and cluster passes;
- data-size dispatch gates so small designs keep the previous paths, a joint
  memory budget for the lockstep solve, and arena ownership/release;
- Stata plugins and native modules carry a fused multiply-add liveness gate.

Core24 × 8 cells of the **pre-PR candidate 2.29.0.20261006** against frozen
local builds of the 2.28.1 sources (three interleaved pairs, 16 threads, real
CUDA use; `Verifications/core24_2290_vs_2281_20261012/`): 188 paired cells,
152 faster beyond the descriptive 7 % host-noise band, 36 inside it
(inconclusive), 0 slower beyond it; geometric mean ratio 0.77; sum of cell
medians 6627 → 6133 s. The promotion campaign of the integrated release
(`Verifications/pr13_integracao_2290_20261012/`): the minimal cells
(`main_95_21_ready`, `akm_v02_firstreg`, `simulated_panel` × 8) with four
arms and four interleaved pairs on 13oct2026 are faster than 2.28.1 in 24/24
cells and in 4/4 pairs of each (ratios 0.47–0.92); the remaining core24 × 8
cells with three arms (release, pre-PR candidate, frozen 2.28.1) on
13–14oct2026 give, on cells at or above 1 s, 54 faster beyond the 7 % band,
14 inside it and 0 slower, and the four CUDA comparable cells that 2.28.1
refuses (`directors`, `pf_difficult_10m_3fe`) are estimated. Against the
pre-PR candidate the release is inside the band in 55 of 70 adjudicated
cells, faster in 12 and slower in 3 (by at most 11 % on 2–7 s legs).
`main_95_21_ready` (46.2 M rows) C++ CPU comparable 63.9 → 29.5 s. With the
PR #13 fast-slope policy, the firm-specific AKM regression (55.9 M rows)
falls from 415 to 112 s on CPU (164 → 30 iterations) and from 241 to 53 s
on CUDA (1934 → 38) in `xhdfe-fast`; comparable results are bit-identical.

## Corrections

- Multiway clustered covariance with a recovered constant: the slope block
  keeps its PSD correction and the constant's row is repaired (range and
  Schur condition), both modes and grouped models (policy D-C). IV: exactly
  equivalent cluster partitions and a positive augmented score matrix are
  recognised before the ill-conditioned bread, and the wide IV covariance is
  built from cluster influences and corrected before any binary64 rounding.
- CUDA comparable fits that previously refused (directors, difficult 10 M
  three-FE) now converge with bounded corrections; grouped FE column moments
  are checked in the existing reductions; strict CUDA continuation keeps the
  requested tolerance; OpenMP allocation failures propagate as exceptions.
- Stata frontend: factor-variable and time-series responses, `noabsorb`,
  `constant`, automatic FE variable names for `g##c.(x z)`, protected user
  variables on repeated `savefe`, models with only absorbed slopes.
- IV in Stata: included controls repeated in `instruments()` are removed;
  exactly proportional instrument columns are removed before absorption on
  CPU and CUDA; first stages inherit `vce(dkraay #)` and the user's
  `absorb()` text.
- Heterogeneous slopes in `xhdfe-fast` use the historical aggregate
  certificate only; comparable keeps the block certificate and continuation.
  This changes Fast results on such designs at the level of the documented
  Fast approximation: on the firm-specific AKM regression the Fast
  coefficients sit 1.1e-7 (6e-5 of one standard error) from the Comparable
  ones after 30 iterations, where 2.28.1 iterated 164 times to 4e-11.
- R: the multiway-clustered standard error of the recovered constant follows
  policy D-C (the parity fixture against the Python module was refreshed;
  slopes are unchanged). Source installs from a clone (`R CMD INSTALL
  r/xhdfe`) record the build configuration (CUDA, architecture, `-march`) in
  a stamp so a later install with another configuration recompiles instead
  of linking objects from the previous one.

## New features

- `vce(dkraay [#])`: Driscoll–Kraay standard errors (Bartlett kernel,
  observed periods, reghdfe 6.14.1 conventions), `e(dkraay_bw)`,
  `e(dkraay_lags)`, `e(dkraay_timevar)`.
- IV first-stage diagnostics: `first`, `ffirst`, `estat firststage`,
  `e(first)` (partial R², excluded-instrument F, df, p), `e(first_gpu_used)`.
- `e(slope_block_checked)` / `slope_block_checked_` / `slope_block_checked`
  report whether the slope block certificate ran.

## Known limitations and pending adjudications

- Two synthetic ill-conditioned heterogeneous-slope designs refuse estimation
  in `reghdfe-comparable` on CUDA (block FE-moment certificate 2.8e-8 against
  the 8e-9 limit after five continuation stages), identically on frozen 2.28.1
  builds. Adjudicated on 13oct2026 with a direct absorption probe: the refused
  state's coefficients are within 1e-14 of the explicit-dummy oracle, but its
  projection error (1.1e-7) does not meet the comparable FE-moment contract and
  does not improve with more iterations; the refusal is legitimate and
  conservative, CPU certifies the same designs, and `xhdfe-fast` on CUDA
  accepts them under its own contract.
- `gpubackend(cuda)` requires at least one absorbed dimension; models without
  fixed effects are refused rather than silently run on the CPU.
- Adjusted R² for saturated models with non-positive residual degrees of
  freedom keeps the 2.28.1 and reghdfe 6.14.1 convention (1); a candidate
  revision had reported it missing under `statstyle(reghdfe)` and was reverted.
- Memory of the AKM first-stage regression at one thread exceeds 2.28.1 by
  15.4 % (authorised exception); large designs use less memory than 2.28.1.
- Windows and macOS binaries are not certified by this candidate; the public
  CI gates validate them before publication. R with CUDA is built from source.

Companion versions: xfepout 1.13.3, xhdfeakm 1.8.1, xhdfegelbach 1.6.1,
xhdfegelbachbootstrap 1.0.1. The validation record of the cut release is
`VALIDATION_2.29.0.<date>.md`; integration evidence is in
`Verifications/pr13_integracao_2290_20261012/` (private repository).
