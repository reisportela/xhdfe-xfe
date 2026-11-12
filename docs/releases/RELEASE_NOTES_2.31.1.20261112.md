# xhdfe 2.31.1.20261112 / xhdfeakm 1.9.0

This release addresses issue #18 with an explicit location-invariant
AKM/KSS alternative while preserving the existing default. Publication
requires the exact version-tag artifacts to pass the release gates.

`xhdfeakm, sigmacentering(personyear)` selects a person-year-centred point
estimator. Python and R use `sigma_centering="person_year"`. The default is
`leaveouttwoway`, retaining existing numerical semantics and inference.

The alternative uses the weighted mean on the final retained person-year
sample after control partialling. It makes the KSS point components invariant
to a common outcome shift with unequal match lengths. Observation-level point
formulae and the existing stayer centre are retained. The selected convention
is returned in result metadata. The sample, controls, leave-out definition,
solver tolerances, leverage streams, plugin components and effect normalisation
are unchanged.

This is a statistical choice, not a higher-precision setting. The estimated
centre can introduce finite-sample bias under arbitrary heteroskedasticity;
there is no uniform unbiasedness or RMSE improvement. Exact rational examples
also show that preserving the target, exact unbiasedness and location
invariance is impossible for covariance and combined variance within the
quadratic-estimator class on some currently supported leave-match-out graphs.

With SE/CI, the alternative provides regular-case delta standard errors and
simultaneous 95% worker-cluster projection intervals for firm variance and
worker–firm covariance. They use the full design, including control-estimation
uncertainty, and preserve the point formula and leave-out unit. Projection
intervals cover weak signals under explicit asymptotic conditions and can be
conservative; they need not contain the corrected point. The delta SEs require
first-order regularity and are conditional on the JLA seed. The implementation
supports 2–128 firms and unit weights. Worker-variance inference, sigma-lowess
and lincom remain unavailable in this alternative. See
[`akm-projection-inference.md`](../akm-projection-inference.md).

Legacy inference remains available, including its sensitivity to outcome
location and its treatment of the FWL-adjusted outcome as fixed. Its SE/CI
concern `theta_*`, distinct from the main `kss_*` components.

The Student-t repair from the 2.31.0 source base is retained. The auxiliary
control fit does not evaluate unreported quantiles. Final local CPU/CUDA
contracts, Stata and R tests, full-model oracle comparisons and the
1,000-control reproducer passed. All 20 legacy cases in each backend retain
bit-identical results against the frozen 2.31.0 build. Scoped paired timings
support non-regression within host resolution; they are not a new whole-package
performance certificate.

Native macOS SPI checks are distinct from licensed Stata/MP host checks;
the latter remain a documented coverage limit. Validation, platform results
and publication evidence are recorded in
[`VALIDATION_2.31.1.20261112.md`](VALIDATION_2.31.1.20261112.md).
