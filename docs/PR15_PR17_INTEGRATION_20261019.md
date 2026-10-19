# Integration validation: PR #15 and PR #17

19 October 2026. Combined source validation on the local Apple Silicon Mac and Athena.

- Base main: `a377a886687c7c05478b17620d5fe9769ecfac5f`.
- PR #15 input: `2381fba7162a8a447b98f6ebd4c7ed3a4d97bd6c`.
- PR #17 input: `7c974305224dd75a55ce247e7a3cce9dbd6d2b35`.
- Tested combined Git tree: `2ecff42e7ca0a03f40b8141b8d06f226b1f94c19`. The report itself is a subsequent documentation-only addition.

The branches combine without conflicts. Their only overlapping file is the Stata test runner. Fresh CMake/plugin builds were made for Mac ARM64, Intel/Rosetta and Athena CPU/CUDA (H100, sm_90). The Mac packages use the isolated OpenMP runtime from PR #15; both slices pass the private-export guard. No production installation was changed.

## Correctness and integration checks

- **1,026 precision fits passed:** three synthetic cases, three seeds, three row orders, five CPU modes and four CUDA modes. Mac ARM64/Intel use 1/4 threads; Athena CPU/CUDA use 1/16 threads. Coefficients, clustered SEs, sample size, convergence, precision certification and requested backend are asserted.
- **150 real Stata host cases passed:** 60 ARM64, 60 Intel/Rosetta and 30 Athena, including OLS, IV/first-stage diagnostics, partial-out, repeated calls and actual worker counts. Mac coverage includes automatic thread selection.
- Stored-version, categorical-interaction, first-stage and first-stage-alias contracts pass on Mac ARM64, Intel/Rosetta, Athena CPU and Athena CUDA. The interaction contract checks 26 OLS and 2 IV cases per configuration.
- Existing Stata contracts: **41 pass on Mac** (one Linux-specific cache test is skipped) and **42 pass on Athena**, each in a fresh process.
- The registered CG CTest passes on Mac (four underlying core tests); **9/9 CTests pass on Athena**.
- **16 macOS packaging control tests pass**, alongside source-mirror alignment and release-metadata checks. The packaging controls use mocks and are separate from the real Stata execution evidence above.

| Mode | Fits | Maximum absolute coefficient error | Maximum absolute SE error |
|---|---:|---:|---:|
| default | 216 | 2.303e-16 | 3.974e-14 |
| fast | 216 | 3.516e-14 | 1.635e-10 |
| strict | 216 | 2.292e-16 | 6.409e-13 |
| lsmr | 162 | 2.335e-16 | 3.853e-15 |
| cg | 216 | 2.303e-16 | 3.974e-14 |

Reference comparisons use areg with explicit effects and reghdfe/ivreghdfe. Acceptance thresholds remain coefficient error < 1e-12 + 1e-7*abs(reference coefficient), and SE error < 1e-12 + 1e-6*reference SE.

The legacy derived-statistics test initially stopped with r(601) on both hosts: it prepends source/stata to adopath, but the clean build placed plugins in a separate package directory. Copying the exact freshly built candidate plugins/runtime into the isolated source/stata directory resolves this layout assumption; the unchanged contract then passes. Initial failure logs and successful retry logs are both retained. No estimator source was changed for the retry.

## Initial performance comparison

192 timed fits on 2,000,000 seeded synthetic observations: two-way OLS, two-way IV, dominant-group three-way OLS, and firm-specific slopes; comparable and fast modes; three alternating baseline/candidate pairs. No warm-ups or discarded fits. The first OLS fit includes plugin/context loading on both builds. Baseline is the original main core/ado, with the same isolated runtime as the candidate on Mac.

Each entry is the median of three trials. Athena applies the 60% CPU-idle admission threshold; GPU sharing is permitted. Short/shared-host timings have limited precision.

| Platform/backend | Threads | Specification | Mode | Baseline seconds | Combined seconds | Change |
|---|---:|---|---|---:|---:|---:|
| Mac cpu | 1 | dominant3 | Comparable | 1.116 | 1.175 | +5.3% |
| Mac cpu | 1 | dominant3 | Fast | 0.719 | 0.720 | +0.1% |
| Mac cpu | 1 | iv2 | Comparable | 1.656 | 1.663 | +0.4% |
| Mac cpu | 1 | iv2 | Fast | 1.613 | 1.611 | -0.1% |
| Mac cpu | 1 | ols2 | Comparable | 0.556 | 0.547 | -1.6% |
| Mac cpu | 1 | ols2 | Fast | 0.478 | 0.467 | -2.3% |
| Mac cpu | 1 | slopes | Comparable | 0.707 | 0.709 | +0.3% |
| Mac cpu | 1 | slopes | Fast | 0.670 | 0.657 | -1.9% |
| Mac cpu | 4 | dominant3 | Comparable | 0.549 | 0.570 | +3.8% |
| Mac cpu | 4 | dominant3 | Fast | 0.426 | 0.419 | -1.6% |
| Mac cpu | 4 | iv2 | Comparable | 1.259 | 1.258 | -0.1% |
| Mac cpu | 4 | iv2 | Fast | 1.212 | 1.224 | +1.0% |
| Mac cpu | 4 | ols2 | Comparable | 0.407 | 0.401 | -1.5% |
| Mac cpu | 4 | ols2 | Fast | 0.334 | 0.336 | +0.6% |
| Mac cpu | 4 | slopes | Comparable | 0.439 | 0.430 | -2.1% |
| Mac cpu | 4 | slopes | Fast | 0.413 | 0.405 | -1.9% |
| Athena cpu | 16 | dominant3 | Comparable | 0.687 | 0.722 | +5.1% |
| Athena cpu | 16 | dominant3 | Fast | 0.644 | 0.684 | +6.2% |
| Athena cpu | 16 | iv2 | Comparable | 1.916 | 1.799 | -6.1% |
| Athena cpu | 16 | iv2 | Fast | 2.009 | 1.886 | -6.1% |
| Athena cpu | 16 | ols2 | Comparable | 0.640 | 0.601 | -6.1% |
| Athena cpu | 16 | ols2 | Fast | 0.521 | 0.510 | -2.1% |
| Athena cpu | 16 | slopes | Comparable | 0.768 | 0.785 | +2.2% |
| Athena cpu | 16 | slopes | Fast | 0.669 | 0.723 | +8.1% |
| Athena cuda | 16 | dominant3 | Comparable | 0.474 | 0.522 | +10.1% |
| Athena cuda | 16 | dominant3 | Fast | 0.472 | 0.511 | +8.3% |
| Athena cuda | 16 | iv2 | Comparable | 1.570 | 1.626 | +3.6% |
| Athena cuda | 16 | iv2 | Fast | 1.565 | 1.556 | -0.6% |
| Athena cuda | 16 | ols2 | Comparable | 0.835 | 0.861 | +3.1% |
| Athena cuda | 16 | ols2 | Fast | 0.389 | 0.395 | +1.5% |
| Athena cuda | 16 | slopes | Comparable | 0.509 | 0.534 | +4.9% |
| Athena cuda | 16 | slopes | Fast | 0.497 | 0.514 | +3.4% |

All 192 fits pass sample, degrees-of-freedom, iteration-count, backend and worker-count comparisons. Maximum baseline/combined coefficient difference is 1.332e-15; maximum SE difference is 2.602e-18.

The initial comparable three-way medians were +5.3% on Mac/1 thread, +3.8% on Mac/4, +5.1% on Athena CPU and +10.1% on Athena CUDA. All original trials are retained; the GPU estimate varied substantially across pairs.

## Focused follow-up

96 additional timed fits repeat the dominant-group three-way specification in both modes, with six alternating baseline/candidate pairs per configuration. This focused batch starts with the comparable fit, so its timing includes plugin/context loading; the original batch measured that specification after its OLS/IV fits. Both variants have the same ordering within each comparison. These absolute times are therefore reported separately, not pooled with the initial table. No warm-ups or discarded trials.

| Platform/backend | Threads | Mode | Baseline seconds | Combined seconds | Change |
|---|---:|---|---:|---:|---:|
| Mac cpu | 1 | Comparable | 1.247 | 1.333 | +6.9% |
| Mac cpu | 1 | Fast | 0.730 | 0.746 | +2.2% |
| Mac cpu | 4 | Comparable | 0.623 | 0.641 | +3.0% |
| Mac cpu | 4 | Fast | 0.434 | 0.434 | -0.0% |
| Athena cpu | 16 | Comparable | 0.790 | 0.897 | +13.4% |
| Athena cpu | 16 | Fast | 0.619 | 0.599 | -3.2% |
| Athena cuda | 16 | Comparable | 0.978 | 0.994 | +1.7% |
| Athena cuda | 16 | Fast | 0.509 | 0.510 | +0.3% |

All 96 focused fits pass the paired numerical/sample/iteration/backend/worker checks. The follow-up controller initially expected the wrong log filename after the first baseline batch on each host. Those completed fits were recovered from their successful logs and timing CSVs, and retained; the remaining fits resumed without rerunning or discarding them.

## Scope and release limits

The integration checks support merging both source fixes. Performance is not free: the focused comparable three-way fit is about 3–7% slower on the Mac and 13.4% slower on Athena CPU; its CUDA median is +1.7%. Focused fast-mode medians range from -3.2% to +2.2%. The original comparison and focused comparison have different fit positions and are both reported above. This localized overhead is accepted for the correction of falsely certified results; it does not establish a general speedup or zero performance cost.

The full historical dataset campaign was not rerun. The complete sequential Stata suite was not used as a clean acceptance gate: the previously documented test-order limitation remains distinct from the fresh-process contracts above. The previously observed Mac full-CTest baseline failures were not reclassified as successes; only the targeted CG CTest was rerun there.

Native Intel hardware, Stata SE, Windows and the R frontend remain untested in this integration. Rosetta is explicitly translated execution. This is source-level integration evidence, not certification of a newly published binary release. Release artifacts must be rebuilt and validated on their exact bytes before publication. GitHub-hosted workflows remain skipped under the Mac/Athena-only testing restriction.

All measured trials, source/build hashes, commands and logs are retained in the maintainer workspace under `pipeline/results/integration_pr15_pr17_20261019_193117/`, including `summary.json`, `performance_summary.csv`, the focused result tables and both hosts' completion receipts. Raw Stata logs remain outside Git because they contain host/license metadata.
