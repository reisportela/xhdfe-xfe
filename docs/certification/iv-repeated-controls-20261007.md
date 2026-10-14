# Stata IV repeated-control repair — 7 October 2026

Candidate **2.29.1.20261007**, on `codex/iv-first-stage-f`, fixes exact
variable-name overlap between included exogenous regressors and `instruments()`.
It removes those redundant entries before absorption and recalculates the
excluded-instrument count. Included controls already enter the native instrument
matrix. An empty excluded list returns rc 198 and clears estimates.

The native estimator, rank thresholds and plugin are unchanged. This is a
Stata frontend repair; it does not fix proportional columns under different
variable names or make a claim about other language frontends.

## Reproduction and validation

The permanent guard is `tests/analytic_precision/iv_repeated_control_contract.do`.
It uses 6,000 synthetic observations and checks three placements of repeated
included controls, including wildcard expansion, five times per specification.
Its frozen pre-fix CUDA run fails on the first repeated-control fit, with a
coefficient-vector `mreldif` of 1.65351732988338e-4. This reproduces the known
estimator error before applying the repair.

| Candidate guard | CPU | CUDA |
| --- | ---: | ---: |
| Specifications | 14 | 12 |
| Repeated-control fits | 210 | 180 |
| Largest absolute coefficient difference from the clean specification | 0 | 3.87246e-13 |
| Largest absolute covariance-entry difference from the clean specification | 0 | 1.48361e-11 |
| Largest coefficient-vector `mreldif` | 0 | 2.20806e-14 |
| Largest covariance-matrix `mreldif` | 0 | 1.66137e-13 |

The largest absolute differences occur in the deliberately weak-instrument
case. For the original unadjusted reproducer, CUDA's maximum absolute
coefficient difference is 4.44089e-16 and its covariance-entry difference is
1.08420e-19. Across 180 paired CPU/CUDA fits, the maximum absolute difference
in the first endogenous coefficient is 6.43041e-13. All candidate checks pass
their predeclared tolerances; none were relaxed after observing results.

Coverage includes conventional, robust and clustered covariance (including
nested firm effects), analytic/frequency/probability weights, multiple
endogenous regressors, a zero instrument, weak instruments, missing values and
singleton removal. CPU additionally covers no fixed effects and no constant.
The guard preserves coefficients, covariance, sample, N, RSS, degrees of
freedom, instrument metadata and the original command line. All accepted CUDA
fits require `gpu_used == 1`, backend `cuda`, and status `used`.

Structural slopes are checked against both ivreghdfe and Stata's independent
`ivregress 2sls` with explicit FE dummies. The special case where an endogenous
variable instruments itself is checked against its OLS equivalent using
reghdfe and explicit-dummy regress; this verifies that the filter removes only
included exogenous names. Three underidentified specifications per backend
must fail with rc 198 and empty estimates.

The full first-stage contract was also rerun, including two newly added
repeated-control specifications: **29 CPU and 26 CUDA first-stage equations
passed** (23 and 20 specifications). The largest absolute F difference from the
direct reference remains 5.993115337332711e-9, the previously investigated and
accepted probability-weighted covariance rounding difference. Postestimation
state-preservation checks passed. CUDA proportional duplicates remain an
explicitly untested limitation in that contract, never counted as a pass.

Two test-infrastructure corrections preceded the final runs: quiet Stata logs
begin with the source commands rather than a `do` wrapper, and ivreghdfe rejects
self-instrumentation syntax, requiring the OLS-equivalent comparison above.
The initial logs and scripts are retained. Neither required an estimator change
or a tolerance adjustment.

## Provenance and boundaries

- Remote run: `/srv/projetos/P008/04_to_test_xhdfe/runs/iv_repeated_controls_fix_20261007`.
- Workspace evidence: `pipeline/results/iv_repeated_controls_fix_20261007/`;
  final results are under `validated/cpu/` and `validated/cuda/`.
- StataNow/MP 19.5; ivreghdfe 1.1.3; reghdfe 6.12.5; ivreg2 4.1.12.
- Actual GPU: NVIDIA H100 NVL, driver 570.195.03.
- Candidate `xhdfe.ado` SHA256:
  `2bb163d7bddd420ffc4bd933cd3c6c8cf5d7351eb3f97fae15c8050b93fcce09`.
- Reused plugin SHA256:
  `3f4be3983be8e32756e181a5ecab6014b5d195a3043d87fd4f9e433c17267dc6`.
  All 53 C++/CUDA/header files in `stata/src` and `stata/include` match the
  saved source tree associated with that binary; no native binary was rebuilt.
- `manifest.json` records tested script/runtime hashes and the parent commit;
  `summary.json` records aggregate and per-specification differences.
- Private/public production text and changed shared files are byte-identical.
  Both release-metadata checks, shell syntax and `git diff --check` passed.

The configured active-checkout gate still refers to absent `/home/mangelo`
directories and an older approved-state manifest. Actual Mac branches,
worktrees, loaded ado versions and isolated Athena runtime hashes were checked
directly; the manifest was not refreshed to suppress that failure. This report
is scoped validation of the Stata repair, not full native performance,
cross-platform or release certification. The branch remains unissued.
