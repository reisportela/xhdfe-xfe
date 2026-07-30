# xhdfe 2.17.0.20260710 — safer and more auditable Gelbach decomposition

Date: 2026-07-10

This development release candidate strengthens `xhdfegelbach` without changing
the Gelbach estimand or any existing default point estimate, variance formula,
convergence target, or non-Gelbach estimator path.

## Scientific and interpretive safeguards

- Describes the result as coefficient-movement accounting, not causal
  mediation, in Stata output, Python/R documentation and all three examples.
- Removes “mediating channel” language and the examples' mechanical
  `educ -> tenure/firm` construction.
- Labels absorbed-FE standard errors as conditional/`gamma0` and adds a
  preferred aggregate absorbed-FE result (`fe_total`) without changing the
  existing per-dimension output.

## API and correctness

- Wires `tol` through Python, R, Stata and the shared C++ core. The default is
  `1e-8`, preserving the historical effective behavior; explicitly supplied
  values now control FE absorption instead of being ignored by Python.
- Fails closed when the base or full design is rank deficient after absorption,
  avoiding arbitrary attribution across duplicate, overlapping or collinear
  columns.
- Requires unique, non-empty block names and validates finite inputs, positive
  weights and the minimum cluster count in the friendly wrappers.
- Stata now returns `r(b_base)`, `r(b_full)`, `r(cov)`, `r(total_cov)` and
  `r(fe_total)`, plus machine-readable interpretation metadata.

## Validation

- 51 Gelbach checks against hand algebra, the original `b1x2` 4.1.0 and LSDV,
  including unadjusted/robust/cluster, aweights, fweights and
  `fweight x cluster`.
- Stata/R/Python parity for deltas, SEs, totals, complete covariance matrices,
  base/full coefficients, FE aggregate and diagnostics to at most `3.55e-15`.
- Adversarial slow-mode FE graph: default path converges and matches an explicit
  sparse-dummy oracle; the legacy A/B path fails closed.
- Stata certification: 26 test files passed.
- AKM/KSS CPU and CUDA suites passed; CUDA parity max difference `1.11e-16`.
- 46.2M-row Stata smoke: CPU `68.719 s`; H100 CUDA `14.348 s`,
  `gpu_used=1`, backend `cuda`, absorption converged in 77 iterations.

## Release status

The implementation is validated in the private development repository. It is
not a completed public release until the production-facing files are
byte-synchronized to `reisportela/xhdfe-xfe`, public CPU-only artifacts are
rebuilt, and the online release is explicitly authorized and cut.
