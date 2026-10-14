# IV multiway covariance precision — 7 October 2026

Candidate **2.29.3.20261007**, on `codex/iv-first-stage-f`, fixes the reproduced
covariance error on CPU and CUDA. Focused correctness tests pass. CPU cost
ranges from about 1–2% in the equivalent-cluster reproducer to 8–15% with
many distinct cluster intersections. Concurrent Athena CUDA timings also show
a cost in the distinct-cluster cases; clean GPU timing and complete release
gates remain pending. This report is scoped development validation.

## What changed

The affected path already uses OlsWide, approximately 106 significand bits.
It now forms clustered covariance from solved cluster influences, avoiding
cancellation in bread × meat × bread, and retains the augmented covariance
through final SSC scaling, standardization and eigenvalue correction.
Binary64 conversion happens after that postprocessing. The existing full-
matrix correction followed by a separate slope-block correction is preserved.

The change is confined to wide IV multiway inference with a recovered absorbed
intercept. It adds no GPU kernel, absorption pass, higher precision type, rank
threshold, tightened convergence target or in-fit mathematical oracle. The
ordinary OLS and ordinary-precision IV paths retain their previous calculation.
`src/wide_covariance.hpp`, `src/ols.cpp`, `include/ols.hpp` and the main regressor
are synchronized across native source mirrors and the public checkout.

## Numerical results

The two original failing cases now have these intercept variances, identically
on CPU and CUDA:

| Case | Before | After | Absolute error versus MP120 after repair |
| --- | ---: | ---: | ---: |
| Unweighted | 0.0019711179779727454 | 0.0019694161555621850 | 3.81e-19 |
| Frequency weighted | 0.0023316100709152589 | 0.0023258062233082817 | 2.23e-19 |

The references are independent Walsh-score MP90 calculations cross-checked at
MP120. No reference value is derived from the candidate's estimates. The
coefficient and full-covariance limits remain 1e-9 and 1e-8. The maximum full-
covariance error across the original contracts is **4.902e-15**, scaled by the
reference diagonal. The new PSD controls have maximum scaled error
**3.828e-15**; their scale uses the larger absolute raw/projected diagonal so
that zero projected variances remain gradeable.

| Permanent contract | CPU | CUDA |
| --- | ---: | ---: |
| Original intercept/full covariance, both tolerance modes | 92 PASS | 92 PASS |
| New positive/indefinite covariance controls, both modes | 88 PASS | 88 PASS |
| Stata covariance entries and `e(vcv_psd_fixed)`, both modes | 88 fits PASS | 88 fits PASS |
| Projection-conditioning guard | 128 PASS | 38 PASS |
| Proportional / near-distinct instruments | 28 PASS | 28 PASS |
| Saved-FE IV and existing cache guard | 38 PASS | 32 PASS |
| Instrument span and underidentification guard | 25 PASS | CPU-only contract |
| Stata first-stage diagnostics versus reghdfe / ivreghdfe | 29 equations PASS | 27 equations PASS |
| Stata proportional-instrument repetitions | 210 fits PASS | 180 fits PASS |
| Stata repeated-control repetitions | 210 fits PASS | 180 fits PASS |

The largest scaled Stata covariance error is **2.735e-12**, below the same
1e-8 limit. Every accepted CUDA row requires actual GPU use. The PSD cases include both
negative eigenvalues that must be corrected and positive covariances that must
be preserved, shifted regressor means, analytic/frequency weights and SSC
choices. The earlier tiny first-stage F rounding difference remains within the
unchanged reference limits.

The new 44-case native contract produces 34 PASS / 10 FAIL on the frozen
pre-fix build, 41 PASS / 3 FAIL on the initial precision-retention patch, and
44 PASS on the final patch. The three remaining initial failures motivated the
stable cluster-influence evaluation; a standalone helper probe isolated the
loss upstream of eigenvalue correction. Original failed results and harness
mistakes are retained and classified in the
[lessons](../LICOES_IV_COVARIANCE_PRECISION_20261007.md).

## Focused CPU performance

Apple M1 Pro, eight cores, OpenMP enabled; frozen pre-fix 2.29.2 versus final
candidate, identical compiler flags and the same C++ driver. Twenty cells cross
two tolerance modes, 1/8 explicitly requested workers, and five regression
shapes. A follow-up adds eight cells with distinct cluster partitions, including
25,980 intersections at 32,768 observations. Each cell has 11 interleaved baseline/candidate pairs, alternating
order, two warmups per process, and multiple fresh fits per measurement. The
recorded worker count equals the request. Compilation finished before timing;
process/load snapshots, raw timings and hashes are retained.

| Shape | Rows | Baseline median range | Candidate median range | Paired median time change across modes/teams |
| --- | ---: | ---: | ---: | ---: |
| Affected weak IV, two cluster partitions | 512 | 4.29–6.00 ms | 4.38–6.06 ms | +0.9% to +2.4% |
| Affected weak IV, two cluster partitions | 32,768 | 245.8–270.8 ms | 245.5–271.4 ms | -0.8% to +1.0% |
| Weak IV, one cluster partition | 512 | 4.02–5.57 ms | 4.02–5.54 ms | -0.5% to +0.4% |
| Ordinary IV, two fixed effects | 65,536 | 15.58–20.49 ms | 15.34–20.29 ms | -1.0% to -0.3% |
| Ordinary OLS, two fixed effects | 65,536 | 13.15–15.74 ms | 12.92–15.70 ms | -2.0% to -0.4% |
| Affected weak IV, distinct cluster partitions | 512 | 4.45–6.11 ms | 4.98–6.63 ms | +8.2% to +12.0% |
| Affected weak IV, distinct cluster partitions | 32,768 | 281.6–302.4 ms | 321.7–342.3 ms | +13.2% to +14.6% |

The equivalent-cluster reproducer costs approximately **0.06–0.11 ms more**
at 512 observations; its larger equivalent-cluster version varies by about one
percent. With distinct partitions, the stable influence calculation does more
work per cluster/intersection: the cost becomes **0.46–0.56 ms** at 512 rows and
**39.9–40.6 ms** at 32,768 rows. Paired bootstrap intervals support these positive
costs. This is a meaningful localized performance trade-off, so the fix is
retained for review and should be optimized before a PR. Correctness alone is
not a performance promotion. Ordinary-control changes should be read as timing
variation, not advertised speedups. These synthetic measurements quantify the
repair's local cost; they are not a release benchmark matrix.

## Concurrent Athena CUDA timing follow-up

At the user's explicit request, the same frozen pre-fix and final native modules were timed concurrently with the separate Athena campaign. The run completed 28 cells, 11 alternating before/after pairs per cell, 616 timed batches and **9,658 timed fits** in approximately 198 seconds. Every fit confirmed `gpu_used`, `gpu_attempted` and CUDA status code 1; requested 1/8 host threads were observed, coefficient error stayed below 3.54e-14, and coefficients, covariance and RSS were finite. Module hashes matched the previously validated binaries before and after the run.

These measurements use the native Python binding on the H100 NVL, with separate persistent baseline/candidate processes and a fresh model for every fit. Each cell has two warmups and three pilot fits per process; measured batches target 0.2 seconds. Input generation, warmups and diagnostics are outside the timer. The estimators were neither rebuilt nor changed for this follow-up.

| Case | Rows | Paired median time-change range across modes/teams |
| --- | ---: | ---: |
| Affected IV, equivalent clusters | 512 | +0.1% to +26.4% |
| Affected IV, equivalent clusters | 32,768 | -5.8% to +2.5% |
| Affected IV, distinct clusters | 512 | +7.7% to +11.3% |
| Affected IV, distinct clusters | 32,768 | +10.6% to +12.7% |
| Weak IV, one-way clustering | 512 | -2.8% to +7.2% |
| Ordinary IV | 65,536 | +0.4% to +4.4% |
| Ordinary OLS | 65,536 | -3.7% to +2.1% |

The distinct-cluster penalty is consistent with the clean local CPU result. Covariance evaluation runs on the host CPU even when CUDA performs absorption. For 32,768 rows, paired median extra time was approximately 32.6–37.4 ms. Ordinary-control differences and the 26.4% outlier among tiny equivalent-cluster cells demonstrate the uncertainty from concurrent load; they are not reliable estimates of new ordinary-path overhead.

The separate campaign process was present before and after the run. Host one-minute load ranged approximately 13–21; between-batch GPU snapshots ranged 0–11% utilization. All observations are retained. For the large distinct-cluster case, paired bootstrap intervals were approximately +8.0% to +17.9% and +8.8% to +16.4% with one host thread, but widened to -18.2% to +58.5% in one eight-thread cell. These are measurements under contention, not clean-host performance certification.

Evidence: `gpu_performance_20261007_130735/README.md` contains all 28 rows and intervals; `results/` preserves per-fit timings, process/GPU snapshots, raw and paired summaries, and module-custody receipts. The recommendation remains to optimize the affected covariance path before a PR. Clean-host CUDA timing and the full release matrix remain pending.

## Evidence and remaining gates

Evidence is under the coordination workspace's
`pipeline/results/iv_covariance_fix_20261007/`:

- `results_final/`: final native JSON, Stata CSV and sanitized logs;
- `results/` and `initial_candidate/`: previous/initial results and failure custody;
- `local_performance_final/`: 440 timed process batches, metadata, per-fit
  observations, paired summaries and bootstrap intervals;
- `local_performance_distinct/`: a further 176 timed process batches with many
  distinct cluster intersections, with the same interleaved protocol;
- `gpu_performance_20261007_130735/`: the authorized concurrent CUDA timing
  campaign, including every paired result and host/GPU load snapshot;
- `psd_isolation_*`: independent covariance versus PSD-helper isolation;
- `candidate_manifest.json`: final source hashes, parent commits and checks;
- build scripts/logs and binary hashes for the isolated Athena candidates.

Athena used separate source/build directories, the frozen CUDA/OpenMP flags,
H100 code, and the installed Stata/reference packages. Unchanged objects were
reused; both changed native translation units were rebuilt for Python and
Stata. The final Stata plugin links libgomp. No existing installation or
separate timing campaign was replaced.

Athena's independent 55.9-million-observation timing campaign remained active.
Small correctness diagnostics overlapped it under the user's authorization;
those runs provide no timing evidence. A later explicit request authorized the
concurrent CUDA timing follow-up above. **Clean CUDA timing is deferred.** The
full core24 × eight-cell release matrix, required three-dataset interleaved
baseline comparisons, grouped performance, large plugin CPU smoke, R runtime,
Windows and complete release packaging checks have not been repeated for this
candidate. No global speed or release-certification claim is made.

The configured checkout gate still references absent `/home/mangelo/...`
paths; its failure is retained. The actual macOS private/public checkouts,
branches, source mirrors, isolated loaded binaries and package metadata are
verified directly. The changes are saved as a local development checkpoint;
performance optimization and the remaining release gates stay open. No PR or
publication has been made.
