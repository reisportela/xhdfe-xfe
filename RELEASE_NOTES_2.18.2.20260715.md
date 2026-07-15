# xhdfe 2.18.2.20260715 — fail loudly on unidentified IV designs

This patch release hardens the IV/2SLS surface while preserving the established
`reghdfe`-compatible estimator path for valid models.  It also corrects an R
parity test that compared a non-identified fixed-effect representation instead
of its identified group projection.

## IV identification hardening

- Zero, duplicated, exactly collinear, post-FWL rank-deficient and
  underidentified instrument designs now fail before the legacy projection.
- The preflight checks instrument rank, residualized excluded-instrument rank,
  useful first-stage rank, finite cross-products and the post-solve
  normal-equation residual.
- Rank decisions are scale-aware and were tested under column and row
  permutations, rescaling from `1e-12` to `1e12`, multiple endogenous
  regressors/instruments, weights, fweights, absorbed fixed effects, controls,
  wide designs and large samples.
- Weak but full-rank and near-rank-but-complete IV designs remain accepted; no
  economic weak-IV cutoff was introduced.
- Release-mode NaN/Inf checks use IEEE-754 bit classification and remain active
  under the existing build flags.

## Zero-diff valid path

Models that pass preflight retain the existing arithmetic and solver:
`Z'Z`, `Z'Q`, `LDLT(Z'Z)`, `solve(Z'Q)`, then `Z * gamma`.  There is no solver
handoff, polishing step, additional estimation iteration, stopping-rule
change, tolerance relaxation, backend change or output-format change.

The directed 53-case oracle matrix rejected all 22 invalid cases and accepted
all 28 valid cases plus three pre-declared near-threshold cases at 1, 2, 4 and
8 CPU threads.  Deterministic valid outputs were bit-identical to 2.18.1.
Real-H100 CUDA cases used the GPU and remained inside the measured A/A
floating-point envelope.

## R test rigor

The group-level-outcome fixture has four exact null directions, so its raw
individual fixed effects are not unique.  The R parity test now checks the
identified group projection against an independent dense QR oracle while
retaining the existing `1e-9` decomposition accuracy gate.  No R package
implementation or estimator tolerance changed.

## Validation and performance

- Full `core23 x 8`: 184/184 runs converged; all 92 CUDA rows used a real GPU.
- Stata certification suite: 26/26 do-files passed.
- R full tests and `R CMD check`: tests passed, exit code 0.
- CPU and real-H100 CUDA AKM/KSS validators passed; maximum CPU/GPU difference
  remained `1.11e-16`.
- ASan/UBSan passed the complete 53-case IV matrix.
- Directed IV timing showed improvements across nearly all cells (up to about
  10%); the sole `+0.019%` measurement was below A/A noise and not
  reproducible.  No reproducible slowdown was found.
- Python/C++, Stata, R and distribution C++ mirrors are byte-aligned.

The rejected ABI-hardening prototype, global fast-math experiments and all
`xtwoway` work are excluded from this release.

Package metadata now uses a setuptools-compatible PEP 621 license declaration,
and Python/R/CITATION links point to the public `xhdfe-xfe` distribution
repository.  These metadata fixes do not change runtime behavior.

Versions: shared C++/Python/R package `2.18.2.20260715`; Stata `xhdfe`
`2.18.2`; `xhdfeakm` remains `1.7.1`; `xhdfeconnected` and `xhdfegelbach`
remain `1.2.1`; `xfe` remains `1.10.1`.  Companion dates are refreshed to the
common release date without changing their behavior.
