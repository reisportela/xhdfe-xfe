# Fast-mode heterogeneous slopes: implementation and performance test

Run `fast_slope_policy_20261009_101640`. Prepared 9 October 2026. **Status: complete; 28/28 fits recorded.**

Implementation on `codex/iv-first-stage-f`, based on public commit `c5f5d2be` and private commit `ca81eccc`. The old anchor is v2.23.1 (`d9958df3`). These tests completed before the local commits authorized on 11 October 2026; no push or installation accompanied the commits.

Local development metadata is now 2.29.5.20261009 (unissued). The isolated benchmark snapshot retains the parent 2.29.4 labels; subsequent local edits only update version/date metadata and documentation. Native implementation hashes identify the tested code independently of those labels.

The full-sample fast-mode runs restore the historical 30 CPU / 38 CUDA iterations. Across Python and Stata, they remove 93–96% of the slowdown relative to the historical old medians and match or improve on the fresh old-version anchors. CPU coefficients match the old version to all recorded digits; the largest old-versus-new GPU coefficient difference is 8.02e-14 and the largest SE difference across all paths is below 3.26e-13.

## Policy being tested

Fast mode skips the stronger heterogeneous-slope block certificate and block-driven polishing. It retains the historical aggregate certificate, normal solver/finite/backend safeguards and same-method repair when that aggregate check fails. Comparable mode retains its current block acceptance and continuation. `slope_block_checked` explicitly distinguishes skipped diagnostics from measured zero residuals.

## Firm-specific AKM: full-sample fast-mode timing

Same 55,947,171 retained observations, four experience regressors, firm-specific seniority terms, worker/firm two-way clustering, 16 threads, public tolerance 1e-8 and max_iter 10,000. All fits use fresh processes and no warm-up fits. Python has three interleaved baseline/candidate repetitions per backend; Stata and current-session old anchors have one fit per condition. Initial CPU admission requires at least 60% idle in three 20-second samples; a 2-second pre-fit check returns to that admission gate if idle falls below 60%. Shared GPU use is recorded.

| Interface / backend | Current branch (s) | Candidate (s) | Old rerun (s) | Historical old median (s) | Faster than branch | Slowdown recovered vs old rerun |
|:--|--:|--:|--:|--:|--:|--:|
| Python / CPU | 404.27 | 114.79 | 134.94 | 96.71 | 71.6% | 107.5% |
| Python / CUDA | 287.61 | 59.97 | 61.95 | 49.94 | 79.1% | 100.9% |
| Stata / CPU | 422.24 | 133.46 | 135.03 | 111.64 | 68.4% | 100.5% |
| Stata / CUDA | 316.43 | 76.74 | 78.12 | 59.45 | 75.7% | 100.6% |

| Interface / backend | Time change vs branch (s) | Iteration change vs branch | Historical slowdown removed |
|:--|--:|--:|--:|
| Python / CPU | -289.48 | -118 | 94.1% |
| Python / CUDA | -227.64 | -1,896 | 95.8% |
| Stata / CPU | -288.79 | -118 | 93.0% |
| Stata / CUDA | -239.70 | -1,896 | 93.3% |

Historical medians come from the August five-trial benchmark. Fresh old-version anchors help distinguish host conditions from code effects. One-fit Stata/old-anchor times are less precise than the repeated Python comparisons; all observations and host snapshots are retained. A recovery above 100% means the candidate was faster than that single old anchor, not proof of a general advantage over the old release.

| Interface / backend / variant | Fits | Median time (s) | Observed range (s) |
|:--|--:|--:|--:|
| python / cpu / baseline | 3 | 404.27 | 394.39–406.84 |
| python / cpu / candidate | 3 | 114.79 | 110.57–120.31 |
| python / cpu / old | 1 | 134.94 | 134.94–134.94 |
| python / cuda / baseline | 3 | 287.61 | 283.50–295.30 |
| python / cuda / candidate | 3 | 59.97 | 58.60–66.56 |
| python / cuda / old | 1 | 61.95 | 61.95–61.95 |
| stata / cpu / baseline | 1 | 422.24 | 422.24–422.24 |
| stata / cpu / candidate | 1 | 133.46 | 133.46–133.46 |
| stata / cpu / old | 1 | 135.03 | 135.03–135.03 |
| stata / cuda / baseline | 1 | 316.43 | 316.43–316.43 |
| stata / cuda / candidate | 1 | 76.74 | 76.74–76.74 |
| stata / cuda / old | 1 | 78.12 | 78.12–78.12 |

The existing in-fit profiler separates absorption (including its certificate/refinement work) from the rest of the fit:

| Python backend / variant | Median absorption (s) | Median remaining fit time (s) |
|:--|--:|--:|
| cpu / baseline | 362.37 | 42.45 |
| cpu / candidate | 72.57 | 44.55 |
| cpu / old | 87.88 | 47.06 |
| cuda / baseline | 245.57 | 42.04 |
| cuda / candidate | 22.05 | 37.92 |
| cuda / old | 19.91 | 42.05 |

## Iterations and precision against the archived reghdfe reference

Maximum absolute gaps across the four nonconstant coefficients; maxima across repetitions. Agreement with this saved reference is not a universal error bound. Fast mode deliberately gives up the newer block guarantee.

| Interface / backend / variant | Fits | Median iterations | Maximum coefficient gap | Maximum SE gap | Maximum SE gap (%) | Maximum coefficient gap in reference SEs |
|:--|--:|--:|--:|--:|--:|--:|
| python / cpu / baseline | 3 | 148 | 1.4677e-11 | 3.7642e-12 | 3.8036e-07% | 5.5347e-09 |
| python / cpu / candidate | 3 | 30 | 1.1410e-07 | 5.8326e-09 | 2.8062e-04% | 6.0826e-05 |
| python / cpu / old | 1 | 30 | 1.1410e-07 | 5.8326e-09 | 2.8062e-04% | 6.0826e-05 |
| python / cuda / baseline | 3 | 1,934 | 3.8261e-11 | 3.0162e-12 | 3.6122e-07% | 1.4148e-08 |
| python / cuda / candidate | 3 | 38 | 2.1777e-08 | 1.6410e-09 | 1.2445e-04% | 3.1372e-05 |
| python / cuda / old | 1 | 38 | 2.1777e-08 | 1.6410e-09 | 1.2446e-04% | 3.1372e-05 |
| stata / cpu / baseline | 1 | 148 | 1.4473e-11 | 3.7369e-12 | 3.8046e-07% | 5.4173e-09 |
| stata / cpu / candidate | 1 | 30 | 1.1410e-07 | 5.8326e-09 | 2.8062e-04% | 6.0826e-05 |
| stata / cpu / old | 1 | 30 | 1.1410e-07 | 5.8326e-09 | 2.8062e-04% | 6.0826e-05 |
| stata / cuda / baseline | 1 | 1,934 | 3.7979e-11 | 3.0166e-12 | 3.6123e-07% | 1.4025e-08 |
| stata / cuda / candidate | 1 | 38 | 2.1777e-08 | 1.6410e-09 | 1.2445e-04% | 3.1372e-05 |
| stata / cuda / old | 1 | 38 | 2.1777e-08 | 1.6410e-09 | 1.2445e-04% | 3.1372e-05 |

Direct candidate-versus-old comparison across all posted coefficients (including the constant), taking the maximum over every completed candidate repetition:

| Interface / backend | Candidate fits | Max coefficient change | Max SE change |
|:--|--:|--:|--:|
| python / cpu | 3 | 0.0000e+00 | 1.0443e-14 |
| python / cuda | 3 | 8.0158e-14 | 3.2108e-13 |
| stata / cpu | 1 | 0.0000e+00 | 1.5699e-14 |
| stata / cuda | 1 | 1.0603e-14 | 3.2540e-13 |

## Comparable-mode controls

Paired controls use the first 500,000 prepared Python rows or raw Stata rows before each frontend’s sample filters. These are within-frontend comparisons, not a claim that the two subset definitions coincide.

| Interface / backend | Retained N, baseline / candidate | Iterations, baseline / candidate | Max coefficient change | Max SE change | Check |
|:--|--:|--:|--:|--:|:--|
| python / cpu | 439,508 / 439,508 | 1,783 / 1,783 | 0.000e+00 | 0.000e+00 | PASS |
| python / cuda | 439,508 / 439,508 | 1,095 / 1,109 | 2.524e-11 | 5.502e-11 | PASS |
| stata / cpu | — | — | — | — | MATCHED_REFUSAL; rc 498 / 498 |
| stata / cuda | 190,852 / 190,852 | 845 / 893 | 8.325e-13 | 2.959e-11 | PASS |

Stata / CPU returned the identical iteration-limit refusal in both builds. This checks refusal behavior, not numerical agreement: `HDFE absorption: did not converge; iterations=10000/10000; requested tolerance=1.000000e-08; relative residual=6.121568e-09. No estimates returned. The iteration limit was reached; increase max_iter/maxiter() and retry.`

Of the 28 fit cases, 26 return validated estimates and 2 are matched existing refusals. These counts exclude four initial Python subset input-validation errors, preserved under `initial_subset_harness_errors`. The corrected harness slices every cluster and slope-value vector; no full-sample timing used the erroneous subset code.

Comparable CPU numerical outputs are identical on the successful Python subset. CUDA outputs agree within the stated comparison limits but are not bit-identical, and their iteration counts differ as shown. These controls do not establish comprehensive comparable-mode runtime non-regression.

## Focused validation and limits

The final wrapper tests cover 32 acceptance/refusal cases. Six arithmetic fixtures exercise the actual certificate, including units, weights, non-finite values and failed aggregate checks; these were relinked and passed against the final native archive after all timing runs. Each build also passed 66 existing slope-unit/rank cases. The real weighted/retained-effects fits produce 14 successful baseline/candidate pairs and two pre-existing comparable CUDA refusals in each build; refusals are not counted as numerical passes.

The initial prototype incorrectly removed aggregate-driven repair too. The correction preserves it; the initial failed tests remain under `initial_revision_checks` and are excluded from performance conclusions. See `FAILURE_CLASSIFICATION.md` in the run directory.

This is a focused performance and compatibility test, not full release certification. Comparable controls cover a subset and synthetic fixtures. Broader dataset, group/individual and release-platform coverage is pending before promotion. Private/public source mirrors are maintained. The implementation and this evidence report are committed locally after validation; publication remains outside this test.

## Evidence

- `summary.json`: machine-readable aggregates and checks.
- `performance_summary.csv`: all completed configuration summaries.
- Individual fit JSON/logs and `job_status.json`: every observation and host context.
- `focused_validation.json`: paired fixture results and unchanged refusals.
- `final_validation.json`, `postflight.log`: final audit, including 108 matching native source/header files and hashes of all 12 candidate artifact observations.
- `artifact_sha256.txt`, `parent_custody.json`: binary and source provenance.
- Experiment identifier: `fast_slope_policy_20261009_101640`. Full scripts and derived outputs are retained under that identifier in the research workspace; restricted source data remain on Athena.
