# Regressions exposed by the reghdfe 6.14.1 suite

These are independent, offline regression fixtures. The upstream suite and its
SHA-256 manifest are retained under the private repository's
`upstream_tests/reghdfe/vendor/`. The complete execution controller, original
outcomes, documented ports and candidate are in
`Verifications/perf_max_20260929/implementacao_fecho_revisao_20261010/reghdfe_upstream_20261011/`.
These historical paths are relative to the original server worktree
`/home/mangelo/Documents/GitHub/xhdfe/build-worktrees/perf-max-20260929`.
The standalone fixtures listed below are included in this testing checkout.

* `new_frontend_regressions.do`: native areg/regress oracles for FV/TS
  responses and regressors, estimation sample, b, V and predictions.
* `savefe_lifecycle.do`: explicit dummy regression oracle, a model with no
  free coefficients, repeated automatic FE outputs, and negative controls
  proving that source variables and manually created variables are protected.
* `pure_fe_certificate.cpp`: exactly represented dyadic oracle and 20
  admission controls for the real native certificate. Includes material
  errors, a regressor, weights, NaN, non-convergence and distinct Fast and
  Comparable contracts. Since 2.29.0 (PR #13) Fast accepts on the aggregate
  certificate alone, so the exact regressor/weighted projections are
  certified in Fast and two material-error controls (kinds 8 and 9) prove
  that the aggregate check still refuses a wrong projection in both modes.

The do-files take one argument: the directory containing the candidate's ado
and plugin files. Run each in a fresh Stata batch process with its working
directory and STATATMP inside a new test directory. Require the terminal
markers `NEW_FRONTEND_REGRESSIONS_PASS` and `SAVEFE_LIFECYCLE_PASS`; process
exit 0 alone is insufficient. The campaign's isolated runner demonstrates the
environment setup without installing packages or changing the user's ado tree.

Compile the C++ fixture against the **same headers, Eigen ABI settings and
plugin** being tested; the campaign's `compile_certificate_probe.py` records
the full command and plugin hash. All 20 lines must say `pass=1` and the
native process must exit 0. This is not a performance test or a general proof
of covariance accuracy.

The public binary32 counterexample is retained here in
`public_cuda_float_fixture.txt`, with the exact rational projection oracle
`rational_float_oracle.py`. It takes `--residual` and a **new** `--output` path;
the fixture defaults to the file next to the script. Its source data are public
Stata auto data plus seeded synthetic variables. No confidential microdata
are fixtures or test outputs here.

The suite's documented port uses an explicit `fetolerance(1e-8)` for FE
comparison. Product defaults are unchanged. Keep literal results and ported
results separate; refusals, upstream defects and unsupported features are not
numerical PASS results.
