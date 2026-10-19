# Issue 16: CG precision fix and validation

19 October 2026. Validation of branch `codex/issue-16-cg-precision`, based on `a377a886687c7c05478b17620d5fe9769ecfac5f` (2.29.1).

The candidate fixes the wrong, precision-certified coefficients and clustered standard errors reported in [issue #16](https://github.com/reisportela/xhdfe-xfe/issues/16). The reproduced cases now match `areg` plus explicit dummies, `reghdfe`, and `ivreghdfe`, on CPU and CUDA, including reversed and shuffled observations. The targeted timing comparison shows no material systematic slowdown. This is source validation on the available machines, not release certification.

## Cause and implementation

The ordinary CG accelerator updated every RHS until the slowest RHS stopped. An already solved outcome could therefore continue with a nonzero rounding residual. Near-zero curvature then produced very large steps that introduced components outside the FE span. The FE-orthogonality certificate can accept such a corrupted transform, and absorbing it again preserves the error.

- Freeze each RHS separately when it reaches the existing CG stopping criterion, in packed CPU, SoA CPU and CUDA implementations.
- Reject a step above `1/sqrt(double epsilon)` (67,108,864), as well as existing nonfinite/nonpositive breakdowns, before applying it. This scale-invariant bound limits amplification of sweep roundoff; it does not relax the requested tolerance.
- Restart unsuccessful CPU CG from the original response/design using MLSMR. On CUDA, restart from the original inputs using native GPU norm-change MAP with CG disabled. The GPU retry releases the original device workspace before allocating the replacement.
- Keep the normal packed direction loop vectorizable; reset frozen lanes only when a new lane stops. The normal path adds scalar checks to existing reductions, not another FE-accuracy pass.
- Preserve the observed CG iteration count on early failure instead of overwriting it with `max_iter`. In the reported aggregate CPU case, the final count is 20, comprising 7 attempted CG iterations and 13 MLSMR iterations.
- Advance the absorption-cache contract generation so previously certified, faulty transforms cannot be reused. A test writes an incorrect result with the old build and confirms the new build recomputes the correct result in read mode without rewriting that old cache.

Public tolerance settings and the heterogeneous-slope fast-mode policy are unchanged. Canonical sources and the Stata, R and packaged `share/` mirrors are byte-identical; the repository-wide source-alignment check passes. No compiled plugin is added to Git and no production installation is replaced.

## Original reported cases

Seed 1 and original row order. Previous values are from the published Intel plugin reproduced through Rosetta; fixed values are from the rebuilt Intel plugin on the same host. Both use one thread and default mode. The reference coefficient is also checked against `areg` with explicit effects (reduced form / first stage for IV); reference SEs are clustered by district.

| Case | Previous coefficient | Fixed coefficient | Reference coefficient | Previous SE | Fixed SE | Reference SE | Iterations, previous → fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rare binary outcome | 0.000807533865692 | 0.000197168296634 | 0.000197168296634 | 0.000943694371516 | 0.000380524485565 | 0.000380524485565 | 6 → 6 |
| District aggregate OLS | 0.00127021943846 | 0.000806667729235 | 0.000806667729235 | 0.00228671076084 | 0.000400079170045 | 0.000400079170049 | 100,003 → 20 |
| District aggregate IV | -2.74662965715 | -3.49588108178e-05 | -3.49588108177e-05 | 7.31797247543 | 0.000116985753006 | 0.000116985753004 | 14 → 20 |

The previous 100,003 count was inflated by failure bookkeeping; it did not represent 100,000 executed CG iterations. CUDA default-mode counts are now 6, 14 and 15 for the three cases. The unchanged CUDA build refused the original default cases, and its strict mode could return a wrong certified coefficient after shuffling rows; the candidate passes those permutations.

## Precision across the validation matrix

1,026 fits: three regressions × three seeds × three row orders, with five modes on CPU and four on CUDA. Explicit LSMR is CPU-only. Mac ARM64 and Intel/Rosetta were tested at 1 and 4 threads; Athena CPU and CUDA at 1 and 16 threads. Every accepted fit must report the correct sample, convergence, certification and requested backend, as well as reference coefficients and SEs.

| Mode | Fits | Maximum absolute coefficient difference | Maximum absolute SE difference |
|---|---:|---:|---:|
| default | 216 | 2.303e-16 | 3.974e-14 |
| fast | 216 | 3.516e-14 | 1.635e-10 |
| strict | 216 | 2.292e-16 | 6.409e-13 |
| lsmr | 162 | 2.335e-16 | 3.853e-15 |
| cg | 216 | 2.303e-16 | 3.974e-14 |

All 1,026 checks pass. The numerical thresholds were fixed before running: coefficient error `< 1e-12 + 1e-7*abs(reference b)` and SE error `< 1e-12 + 1e-6*reference SE`. Default-mode errors are much smaller than those thresholds. Fast-mode SE differences remain larger, as expected under its existing stopping policy.

## Performance

One seeded synthetic dataset with 2,000,000 rows; four specifications: two-way OLS, two-way IV, three-way OLS with a dominant FE group, and firm-specific slopes. Each specification is evaluated in comparable and fast modes. These are targeted performance checks, not a repeat of the full historical dataset campaign.

Three alternating baseline/candidate pairs per platform/thread/backend configuration; fresh Stata process for each eight-fit batch. No warm-up fits or discarded observations. The first OLS fit includes plugin/context loading on both builds. Timers cover estimation, excluding data generation. Both variants use the same build flags and runtime on each host; Mac builds share the isolated OpenMP runtime from the issue-14 investigation. Athena uses the established 60% CPU-idle admission threshold and permits the authorized shared GPU use. All raw trials and admission records are retained.

### Mac, CPU, 1 thread

| Specification | Mode | Baseline seconds | Fixed seconds | Change | Iterations, baseline → fixed |
|---|---|---:|---:|---:|---:|
| ols2 | Comparable | 0.478 | 0.478 | +0.0% | 2 → 2 |
| ols2 | Fast | 0.385 | 0.386 | +0.3% | 2 → 2 |
| iv2 | Comparable | 1.366 | 1.369 | +0.2% | 2 → 2 |
| iv2 | Fast | 1.321 | 1.327 | +0.5% | 2 → 2 |
| dominant3 | Comparable | 0.922 | 0.939 | +1.8% | 5 → 5 |
| dominant3 | Fast | 0.578 | 0.598 | +3.5% | 2 → 2 |
| slopes | Comparable | 0.582 | 0.602 | +3.4% | 2 → 2 |
| slopes | Fast | 0.533 | 0.538 | +0.9% | 2 → 2 |

### Mac, CPU, 4 threads

| Specification | Mode | Baseline seconds | Fixed seconds | Change | Iterations, baseline → fixed |
|---|---|---:|---:|---:|---:|
| ols2 | Comparable | 0.345 | 0.347 | +0.6% | 2 → 2 |
| ols2 | Fast | 0.284 | 0.291 | +2.5% | 2 → 2 |
| iv2 | Comparable | 1.104 | 1.074 | -2.7% | 2 → 2 |
| iv2 | Fast | 1.020 | 1.052 | +3.1% | 2 → 2 |
| dominant3 | Comparable | 0.485 | 0.474 | -2.3% | 5 → 5 |
| dominant3 | Fast | 0.355 | 0.365 | +2.8% | 2 → 2 |
| slopes | Comparable | 0.388 | 0.366 | -5.7% | 2 → 2 |
| slopes | Fast | 0.353 | 0.358 | +1.4% | 2 → 2 |

### Athena, CPU, 16 threads

| Specification | Mode | Baseline seconds | Fixed seconds | Change | Iterations, baseline → fixed |
|---|---|---:|---:|---:|---:|
| ols2 | Comparable | 0.631 | 0.644 | +2.1% | 2 → 2 |
| ols2 | Fast | 0.519 | 0.545 | +5.0% | 2 → 2 |
| iv2 | Comparable | 2.045 | 1.850 | -9.5% | 2 → 2 |
| iv2 | Fast | 1.895 | 1.898 | +0.2% | 2 → 2 |
| dominant3 | Comparable | 0.699 | 0.692 | -1.0% | 5 → 5 |
| dominant3 | Fast | 0.626 | 0.608 | -2.9% | 2 → 2 |
| slopes | Comparable | 0.696 | 0.719 | +3.3% | 2 → 2 |
| slopes | Fast | 0.657 | 0.652 | -0.8% | 2 → 2 |

### Athena, CUDA, 16 threads

| Specification | Mode | Baseline seconds | Fixed seconds | Change | Iterations, baseline → fixed |
|---|---|---:|---:|---:|---:|
| ols2 | Comparable | 0.924 | 0.926 | +0.2% | 2 → 2 |
| ols2 | Fast | 0.424 | 0.395 | -6.8% | 2 → 2 |
| iv2 | Comparable | 1.609 | 1.585 | -1.5% | 2 → 2 |
| iv2 | Fast | 1.567 | 1.580 | +0.8% | 2 → 2 |
| dominant3 | Comparable | 0.540 | 0.531 | -1.7% | 5 → 5 |
| dominant3 | Fast | 0.537 | 0.541 | +0.7% | 2 → 2 |
| slopes | Comparable | 0.577 | 0.558 | -3.3% | 2 → 2 |
| slopes | Fast | 0.540 | 0.542 | +0.4% | 2 → 2 |

All 192 timed fits completed and passed their convergence/backend checks. Samples and degrees of freedom matched baseline. Maximum old/new coefficient difference was 1.332e-15; maximum SE difference was 2.602e-18.

For the comparable three-way case that exercises the changed CG path, median timing differences were +1.8% (Mac/1), −2.3% (Mac/4), −1.0% (Athena CPU) and −1.7% (Athena CUDA). Across all specifications, differences ranged from −9.5% to +5.0%; these short, shared-host trials do not establish a general speedup. The first CPU prototype had a repeatable ~5.5% overhead in the single-thread CG case; restoring the vectorizable direction loop reduced that to ~1.8%. Earlier prototype measurements remain in the evidence directory.

## Additional regression checks and limits

- Four core CG tests pass, including independent unaccelerated/explicit-dummy references, exact-zero RHS lanes, row permutation, and scales `1e-10`, `1`, `1e10`. The old core fails the new near-redundant-FE tests. The core test is registered in CTest.
- The new Stata regression contract is integrated into `tests/stata/run_stata_tests.sh` and fails on the baseline before the patch.
- 24 independent FE-space/weighted/retained-effect contracts pass on the Mac.
- All 42 existing Stata contracts pass individually on Athena. On the Mac, 41 pass; the remaining cache receipt contract uses Linux-only `/usr/bin/sha256sum` and GNU `stat`, and passes on Athena.
- Four existing CUDA IV contracts pass: first-stage diagnostics, repeated controls, proportional instruments and first-stage aliases.
- All 9 Athena CTests pass. On the Mac, 9 of 12 pass. The other 3 fail identically on the unchanged baseline: extreme-tail Student-t arithmetic/bit-expectation checks and a host compiler-hash provenance check. Logs retain both results.
- The sequential Stata suite stops at `reghdfe-derived-statistics.do` on Athena on both baseline and candidate, while that contract passes in a fresh process on both builds. Fresh-process execution makes the existing test-order limitation explicit rather than claiming a clean sequential-suite run.
- Intel Mac execution uses Rosetta, not native Intel hardware. Stata SE, Windows and the R frontend were not exercised. Stata MP, the C++ core through Python, and Athena H100 CUDA were exercised. Release packaging/native-platform certification remains separate.

## Reproduction and evidence

Permanent tests: `tests/test_cg_breakdown.py`, `tests/stata/cg_near_redundant_contract.do`. For the Stata contract, set `XHDFE_STATA_ADOPATH` to the isolated candidate and run the do-file with `cpu 1`, `cpu 16`, `cuda 1` or `cuda 16`. The Python core test uses `XHDFE_TEST_BUILD_DIR` to select the rebuilt module.

Maintainer evidence is retained under `pipeline/results/issue16_fix_20261019/` in the parent workspace: `summary.json`, `performance_summary.csv`, `performance.do`, `run_performance.py`, `cache_control.json`, final correctness CSVs, build/source manifests, all original/prototype/final logs, and the retrieved Athena receipts. The corresponding remote work is isolated under `runs/issue16_fix_20261019`. Raw Stata logs stay outside Git because they contain host/license metadata.
