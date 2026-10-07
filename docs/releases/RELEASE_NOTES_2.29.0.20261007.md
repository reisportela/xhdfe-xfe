# xhdfe 2.29.0.20261007 — development candidate

This feature branch is unissued and has not been submitted as a pull request.
The version and date identify the candidate; they are not a publication claim.

## First-stage excluded-instrument tests in Stata

IV users can request `first` for full first-stage regressions and their diagnostic
summary, or `ffirst` for the summary alone. `e(first)` stores partial R-squared,
the excluded-instrument F-statistic, its two degrees of freedom, and its p-value,
with one column per endogenous regressor. `estat firststage` returns the same
summary in `r(first)` and can calculate it on demand after an ordinary IV fit.
Replay supports `xhdfe, first` and `xhdfe, ffirst`.

The auxiliary fits use the IV estimation sample and the same weights, fixed
effects, VCE, DoF/SSC settings, and backend. The structural estimates and sample
are restored even if a postestimation diagnostic fails. `nosample` supports
summaries requested during estimation; later full regressions require the sample.
The C++ estimator and plugin interface are unchanged. Diagnostics are opt-in and
perform additional xhdfe OLS regressions.

The statistics follow xhdfe/reghdfe first-stage finite-sample conventions.
When fixed effects are nested within clusters, their additional intercept
degree of freedom produces the documented `(N-K-1)/(N-K)` factor relative to
ivreghdfe's F-statistic. The tests verify both the direct reghdfe result and this
ivreghdfe relationship. These are per-equation relevance tests; conditional
Sanderson-Windmeijer, Kleibergen-Paap, and overidentification tests are outside
this candidate's scope.

## Validation

`tests/stata/iv_firststage_contract.do` provides small synthetic CPU/CUDA tests
against `ivreghdfe`, direct `reghdfe` first stages, and Stata `regress` without
absorbed effects. Its CUDA mode requires actual GPU use in the IV and auxiliary
fits. The test covers multiple endogenous variables, weights, missing data,
singletons, redundant instruments, weak instruments, and postestimation state.
Execution evidence is recorded separately in the candidate validation report.

The [7 October validation](../certification/iv-firststage-20261007.md) passed
27 CPU and 24 CUDA first-stage comparisons. An existing CUDA IV limitation with
proportional duplicate instruments or repeated included controls was reproduced
on unchanged 2.28.1 code and is documented separately; this candidate does not
change the native IV solver.
