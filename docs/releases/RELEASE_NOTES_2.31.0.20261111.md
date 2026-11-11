# xhdfe 2.31.0.20261111

This release corrects Student-t inference on Windows, makes Stata
`residuals()` return the fit's own residuals, and corrects weighted
reghdfe-comparable fits on CPU. It also carries the routing and performance
changes of the 2.31 line.

- **Student-t inference on Windows.** An x87 precision-control
  incompatibility in Student-t inference could reject otherwise valid fits on
  Windows with `r(198)`, including OLS fits without absorbed fixed effects.
  The failure was reproduced with the distributed Windows 64-bit Stata plugin
  from 2.30.1.20261020 under StataNow/MP 19.5. The affected code is present in
  the reviewed release tags from 2.26.2.20260903 onwards; earlier Windows
  binaries were not individually tested. `xhdfeakm` with `controls()`
  (version **1.8.2**) no longer evaluates Student-t quantiles in its internal
  control-partialling fit, whose standard errors, p-values and intervals it
  never reports; its estimates are unchanged. On Linux and x86 macOS (64-bit
  x87 precision at start-up) this correction changes no result.
- **Stata `residuals()`.** In `xhdfe-fast` and `reghdfe-comparable` the
  stored vector is now the fit's own within residual. It is the vector behind
  `e(b)`, `e(V)` and `e(rss)`, and the same vector Python returns as
  `residuals_`. The extra absorption of y - Xb that 2.30.1 ran for this option
  no longer runs. On `main_95_21_ready` (46 million rows, CUDA Comparable, one
  fit each) the fit took 14.9 s without `residuals()`, 16.5 s with it, and
  147.3 s with the route of 2.30.1. `strict-residual` keeps the 2.30.1
  behaviour. `XHDFE_STATA_LEGACY_RESIDUAL_REFINE=1` restores the old route for
  comparisons.
- **Weighted reghdfe-comparable fits on CPU.** 2.30.1 skipped the bounded
  MLSMR retry for weighted fits. Below the thresholds of the automatic
  selector (for example one regressor), a weighted fit could then stop with
  the sweep's honest exit: converged and certified, but outside the
  comparable coefficient contract. Such fits now take that retry. It applies
  with up to three fixed effects, the Auto method and criterion, and
  tolerance 1e-8. The published iterations count the primary and the retry,
  within `max_iter`. On the 240,000-row three-way public fixture with one
  regressor, coefficients and standard errors now agree with `reghdfe` 6.14.1
  at tolerance 1e-12, within |Δb| <= 1e-8·max(1,|b|) and |ΔSE|/SE <= 1e-7. The
  2.30.1 route missed the standard-error limit by a factor of 26 to 67.
  This correction leaves unweighted fits unchanged: bit for bit on CPU, and
  with the same route and iteration counts on CUDA.
  `XHDFE_P8_CPU_WEIGHTED_RETRY=0` restores the 2.30.1 route.
- **Variance of `_cons` with a nested intercept effect.** When a fixed effect
  without slopes is nested within a one-way cluster, outside `xhdfe-fast`, the
  `_cons` entries of `e(V)` now use the exact identity for zero cluster
  residual sums.
- **Performance, routing and defaults.** Visible changes from 2.30.1:
  - automatic OpenMP teams use one thread per 25,000 rows when no thread
    count is requested. This applies to absorption without saved fixed
    effects, within the automatic limits;
  - automatic teams are shared between simultaneous fits of the same user,
    reported as `e(thread_limit_code)` 6;
  - Stata reads its input in one pass, in parallel on qualified hosts;
  - the packed sweep covers designs with up to 16 regressors;
  - a sweep that the automatic selector would discard is left early, reported
    as `e(auto_retry_status)` `left_early`;
  - the CUDA Comparable conjugate-gradient route continues through a slow
    mode instead of treating it as a breakdown;
  - the CUDA Comparable tail with three fixed effects can be handed over to
    the host MLSMR;
  - `xhdfe-fast` on CUDA no longer refuses with "did not converge" after a
    device CG breakdown;
  - new diagnostics: `e(cuda_tail)` and `e(cuda_tail_code)` in `xhdfe` and
    `xfepout`; `r(cuda_tail)`, `r(cuda_tail_code)` and `r(point_cuda_tail)`
    in the Gelbach commands.

  In the core24 × 8 matrix against 2.30.1, the geometric means of the time
  ratios are as follows. The matrix uses 16 explicit threads and three
  interleaved pairs, so it does not measure the automatic teams.

  | Variant | CPU comparable | CPU fast | CUDA comparable | CUDA fast |
  |---|---|---|---|---|
  | Stata | 0.80 | 0.78 | 0.64 | 0.79 |
  | Python | 0.89 | 0.90 | 0.74 | 1.00 |

  Slight losses remain after repetitions: `main_95_21_ready` Python CPU and
  CUDA Comparable, about +1%. Three cases stay under review:
  - `simulated_panel` Python CUDA Fast, +3.1% median;
  - a sub-second Stata CUDA Comparable case, +8%;
  - `akm_v02_secondreg` Stata CUDA Comparable, +3.5%, which varies up to 1.6×
    between pairs of the same binary.

  Against that matrix's candidate, the P3 correction gave aggregates
  compatible with neutrality (geometric means 0.995–1.006). Two measured
  losses were accepted by the maintainer on 11 November 2026 under the
  performance trade-off allowance. In both, the final method and the
  iterations are identical in the two arms, and the mechanism is not
  identified:
  - +3.09% (+0.104 s) in Stata CPU Fast on `pf_simple_10m_3fe`;
  - +11.09% (+0.067 s) in Stata CUDA Comparable on
    `synthetic-uniform-harder`.

  Twenty-one cases remain under review. On CPU without weights, this release
  is bit-for-bit identical to the P3-corrected candidate in the 13 compared
  cases. The final build was not timed directly against 2.30.1.
- **Cache migration.** A new solve-contract generation invalidates
  previously computed absorption transforms. Old caches do not need to be
  deleted.
- **Versions.** `xhdfe` 2.31.0; `xfepout` **1.14.0** (automatic thread
  default and the new `e(cuda_tail_code)` diagnostic); `xhdfegelbach`
  **1.6.2** and `xhdfegelbachbootstrap` **1.0.2** (CUDA completion
  diagnostics); `xhdfeakm` **1.8.2**. Other commands change only their date.

## Known limitations

- **Weighted CUDA Comparable fits with three fixed effects** keep the
  inherited sweep exit: converged and certified, but outside the comparable
  contract against `reghdfe` at tolerance 1e-12. On the public 240,000-row
  three-way fixture, coefficients differ by up to 4.4e-7 and standard errors
  by up to 2.3e-5 (relative). Use the CPU backend for such fits.
- **`savefe` in reghdfe-comparable** uses the effect-capturing sweep. Its exit
  can leave coefficients and standard errors outside the comparable contract:
  in the measured cases up to 4.5e-7 relative in coefficients and 1e-4 in the
  covariance.
- **Strict-residual on CUDA.** With the automatic method and criterion and no
  weights, CUDA strict-residual may pass the relative certificate without the
  final strict check. On a public fixture, coefficients then differ by 1.5e-7
  and standard errors by 1.7e-4 (relative) from an exact reference. This is
  inherited and present in 2.28.1. Use the CPU backend for strict-residual
  results.
- **Vector stored by `residuals()` under CUDA strict-residual.** Its accuracy
  can differ between runs: 2.1e-6 or 1.4e-13 on the public three-way fixture.
- **`residuals()` on CUDA** has the accuracy of the CUDA fit itself. On a
  path-like two-FE public fixture the error is 8.4e-4 (Fast) and 1.3e-6
  (Comparable) relative to an exact reference. The extra re-absorption of
  earlier versions gave 4.5e-3 (measured with 2.28.1).
- **`xhdfeakm` and the level of the outcome (issue #18).** Under
  `leaveoutlevel(match)`, KSS-corrected variance components depend on
  outcome shifts for matches of unequal length. `se` and `ci` also depend on
  the outcome level, under both `obs` and `match`. A correction is planned
  for the next version.
- **Windows: near-perfect fits.** Near-perfect additive-cell fixtures
  returned exact coefficients and certified fits on Windows, but with
  relative covariance errors of 2.65e-9, 1.74e-4 and 1.82e+2. The
  corresponding relative residual noise was 2^-12, 2^-20 and 2^-30. The same
  fixtures on Linux returned 2.2e-16. The exact additive-cell projection
  needs extended-precision arithmetic, which Windows hosts run at 53 bits.
- **Student-t tails on Apple Silicon.** On macOS arm64, where `long double`
  equals `double`, the measured case (df = 1,000,000, tail 1e-20) has a
  relative error of 1.2e-11 against a 5e-12 contract. The test stops at its
  first failure. This is inherited.
- **Iteration accounting** of unweighted CPU routing and repair stages, and of
  the IT sweeps before a CUDA conjugate-gradient stage, is unchanged from
  2.30.1. A strict-residual CPU repair can report up to 100,010 iterations
  against `max_iter` 100,000.

Validation, test lists and measurement conditions are in
[VALIDATION_2.31.0.20261111.md](VALIDATION_2.31.0.20261111.md).
