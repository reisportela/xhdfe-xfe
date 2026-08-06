# xhdfe 2.23.1 / xfe 1.11.0 — 06aug2026

Targeted release hotfix. The estimator, numerical tolerances, convergence
criteria, C++/CUDA kernels and `xfe` implementation are unchanged from 2.23.0.

## Stata display

- Fixed `r(509)` after an otherwise successful visible `xhdfe` fit or replay.
  The display routine passed `e(b)` directly to `colsof()` in a context where
  Stata rejects matrix-valued `e()` operators. It now counts the rows of the
  already-copied `omit_reason` matrix, which has one row per displayed
  coefficient.
- Added a non-silent replay regression test so the certification suite cannot
  hide this failure behind quiet estimation calls.

This changes only post-estimation display. Coefficients, variance estimates,
degrees of freedom, fixed effects, convergence state and backend selection are
not recomputed or modified.

## Autonomous source archive

- `xhdfe-src.zip` now includes the repository-level scientific build gates,
  CMake test probes, Python manifest and compatibility module required by the
  build paths it documents.
- The archive validator now fails closed on those entries, checks shell syntax,
  configures CMake with testing enabled and builds the fast-math non-finite
  liveness gate before accepting the ZIP.
- This closes the missing `tools/check_verifier_device_fma.sh` failure that made
  `xhdfegpu` compile all CUDA objects and then exit with `r(198)` at the final
  scientific gate.
- Build failures now print the captured compiler log; the former unsupported
  `type, lines(20)` diagnostic could hide the original error.

## Validation

- Stata certification suite: 36/36 tests passed, including the new visible
  replay regression.
- A clean source ZIP passed integrity, Rcpp SHA-256, CMake configuration and
  non-finite liveness-gate validation.
- `xhdfegpu` built and installed both `sm_90` OpenMP plugins from the corrected
  offline ZIP in an isolated Stata adopath. Installed bytes matched the build
  outputs exactly.
- On the H100, the installed `xhdfe` and `xfe` plugins both reported
  `gpu_used=1`, `gpu_backend=cuda` and `gpu_status=used`; `xhdfe` additionally
  reported `e(version)="2.23.1 06aug2026"`.

## Performance scope

No solver, recovery, cache, threading or GPU code changed. The repaired display
operation is constant-time metadata handling after estimation, so the core23
runtime matrix is unaffected by construction; no performance trade-off or
tolerance relaxation is introduced.
