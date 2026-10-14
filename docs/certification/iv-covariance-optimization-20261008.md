# Covariance optimization experiment, 2026-10-08

Status: focused accuracy and performance evaluation complete. This is a
2.29.4.20261008 development candidate, not release certification.

## Finding

The optimization preserves the corrected covariance and produces a modest
end-to-end benefit when the model has many cluster intersections. At 1,048,576
rows the original four CPU/CUDA and fast/comparable cells improve by 2.14% to
4.74%; all five pairs in each cell favor the candidate. The separate 11-pair
CPU-fast repeat improves by 3.82%, with all eleven pairs favoring the candidate.
The 5,242,880-row matched-warmup results are reported separately below because
the original large timing procedure exposed a process-history effect.

Few-cluster and ordinary/group controls do not establish a consistent
performance penalty. No numerical regression was found within the tested
scope. Keep the implementation as a reviewable candidate; broader release
certification remains separate.

## Baseline and implementation

The baseline is the already-correct covariance fix in private commit
`48ee7b65978d13de138816edbd3d7eb88dc26064` and public commit
`49dfd4abbb9f25bd5861d9e1c1bc965905fe8925`. The pre-fix implementation is not
the comparison baseline for this optimization.

Only `run_ols_wide` in `src/ols.cpp` changes. For the retained wide multiway
covariance, cluster scores occupy a contiguous buffer, the existing
factorized solve uses reusable scratch space, and the original-unit influence
vector is reset and reused. The code skips the discarded off-diagonal meat,
intercept cross/total, sandwich and basis-transform covariance calculations.
Component score-diagonal range checks and checks on the returned stable
covariance remain. Legacy calculations stay active where they supply results;
the inverse Gram matrix remains available.

The stable contribution arithmetic, observation/group order, precision type,
SSC and PSD policy remain unchanged. No absorption, GPU kernel, cache policy,
estimator, tolerance or API change. The covariance work still executes on the
CPU for either absorption backend. All five C++ source mirrors match in both
local checkouts. Production metadata is aligned at 2.29.4.20261008; no compiled
artifact is installed into a canonical production location.

## Accuracy evidence

- Existing 677 native contract rows pass: IV intercept covariance, genuine PSD
  corrections, projection conditioning, exact proportional instruments, IV
  cache paths and instrument span. CPU and actual CUDA, both tolerance modes
  where applicable.
- Stata: both backends pass covariance, first-stage, proportional-instrument
  and repeated-control contracts, including the 176 covariance fits. After
  metadata alignment, covariance and first-stage contracts were repeated on
  both backends against the final 2.29.4 ado/help sources and the tested plugin;
  all four final contracts pass.
- Added 84 direct native cases cover 2/3/5 regressors, explicit constants in
  first/last position and no constant, weights, response scales 2^-80/1/2^80
  and conventional/minimum cluster SSC. Returned values match the frozen
  baseline exactly. Independent MP90/MP120 reference maximum scaled covariance
  error: 2.092800705919071e-16, below the unchanged 1e-8 limit.
- OLS range contract: 19 passes and 2 expected safe range errors; normal-case
  comparisons to the frozen baseline pass.
- Group-only sample/extraction tests: 24 cases per backend pass. Group/
  individual direct test: 96 CPU cases pass. On CUDA, both builds have
  identical 64 passes and 32 absorption refusals, including identical refusal
  messages. Those pre-existing refusals remain an explicit coverage limit.
- Every weak-IV performance covariance, including pilot, warmup, conditioning
  and timed observations, passes the unchanged independent MP90/MP120 oracle:
  268/268. Each build has
  134 checked observations.
  Maximum scaled error is
  4.517390060831218e-12 for the candidate and
  4.660110778125622e-12 for the frozen baseline
  (unchanged limit 1e-8). Degrees of freedom and all cluster/intersection
  counts match the oracle.

The supplementary native fixture initially used distinct offsets without a
constant, so the baseline correctly did not take the intended ill-conditioned
path. Correcting that fixture was an evaluation-harness repair, not an
estimator change. An isolated SciPy test dependency resolved the missing
group-test dependency. Initial failures and classifications are retained.

## Performance: many-intersection case at 1.05M rows

Negative values mean the candidate is faster. Percentages are medians of
within-pair runtime ratios, not ratios of separate runtime medians. The
empirical intervals describe the retained paired observations; small samples
and a shared host limit precision.

Original five-pair study, 826,796 intersections:

| Rows | Fixture | Backend | Mode | Pairs | Median time change | Empirical 95% interval | Median seconds change |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1,048,576 | weak_many | CPU | fast | 5 | -2.14% | [-4.85%, -1.56%] | -0.174 |
| 1,048,576 | weak_many | CUDA | fast | 5 | -3.53% | [-4.52%, -2.36%] | -0.284 |
| 1,048,576 | weak_many | CPU | comparable | 5 | -4.74% | [-23.48%, -1.53%] | -0.271 |
| 1,048,576 | weak_many | CUDA | comparable | 5 | -3.62% | [-4.45%, -2.69%] | -0.293 |

Explicit 11-pair repeat of the affected CPU-fast case and ordinary OLS control:

| Rows | Fixture | Backend | Mode | Pairs | Median time change | Empirical 95% interval | Median seconds change |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1,048,576 | ols | CPU | fast | 11 | -4.83% | [-7.71%, -3.26%] | -0.056 |
| 1,048,576 | weak_many | CPU | fast | 11 | -3.82% | [-3.88%, -2.74%] | -0.308 |

The original ordinary OLS CPU-fast control at 1.05M rows was +2.30%, while the
same 5.24M control was -1.71%. Its explicit 11-pair repeat is -4.83%. The initial
slowdown did not reproduce. The conflicting timing environments do not justify
claiming a stable ordinary-OLS speedup from this narrow covariance change.

The complete-table review identified the remaining initial ordinary IV
CPU-fast control at +2.48% (the 5.24M counterpart was -0.10%). A final
11-pair repeat, declared before running it and using the identical worker,
fixtures and modules, checks that ambiguity:

| Rows | Fixture | Backend | Mode | Pairs | Median time change | Empirical 95% interval | Median seconds change |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1,048,576 | iv | CPU | fast | 11 | -0.88% | [-5.52%, +5.92%] | -0.011 |

Its median is -0.88%, with an empirical interval of [-5.52%, +5.92%].
The initial slowdown does not reproduce; a small effect in either direction
cannot be excluded by this shared-host timing control.

## Performance: matched warmups at 5.24M rows

The original weak-fixed CPU cell exposed an immediate-repeat/process-order
effect in both builds: about 22 seconds when immediately repeated versus about
27 seconds after switching processes. Alternating AB/BA with five pairs leaves
one order represented three times, producing misleading paired medians.
The physical cause has not been established. Original observations are retained
below and must not be interpreted as isolated code effects.

A separately declared follow-up places one same-build conditioning fit
immediately before every measured fit, retaining and validating the
conditioning results. Code, fixtures, defaults and tolerances remain unchanged.
It covers the fixed-cluster detector (67,591 intersections) and the affected
many-intersection case (4,127,645 intersections), in fast mode:

| Rows | Fixture | Backend | Mode | Pairs | Median time change | Empirical 95% interval | Median seconds change |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 5,242,880 | weak_fixed | CPU | fast | 3 | +0.15% | [-3.98%, +5.93%] | +0.033 |
| 5,242,880 | weak_many | CPU | fast | 3 | -3.20% | [-22.05%, -2.58%] | -0.934 |
| 5,242,880 | weak_many | CUDA | fast | 3 | -3.79% | [-26.87%, -3.05%] | -1.090 |

Only three pairs were collected per cell. The fixed-cluster estimate is near
zero; the apparent original large slowdown disappears. The affected CPU and
CUDA cases improve in all three pairs, although large observations widen the
intervals.
Treat these estimates as focused diagnostics, not precise universal gains.

## Group controls and scope

Eight supplementary group/individual control cells completed 400 measured fits
and 80 warmups. They cover sum/mean aggregation, CPU/CUDA and fast/comparable,
with 384 group observations and two host threads. All outputs match baseline
exactly. Median changes range from -1.22% to +1.25%, each empirical interval
includes zero, and absolute median changes are at most 0.205 milliseconds
per fit. This adds group-path control coverage, not a large group benchmark.

The five studies comprise 804 measured fits and 168 warmup or
conditioning fits, plus a separate 32-fit pilot (1004 fits in total).
The original large study has 32 cells, 320 measured fits and 64 warmups;
the control repeat has two cells and 44 measured fits; the matched-warmup
follow-up has three cells and 18 measured fits; the final ordinary IV repeat
adds 22 measured fits. All requested actual backends
and thread teams are asserted; iteration counts match within each pair.
The original large study's maximum known-slope error is
1.323230414129739e-11, with exact known RSS. Follow-up accuracy is also checked.

## Timing protocol and limitations

Frozen modules in separate persistent processes; exactly one fit at a time.
Fresh model each fit; data preparation and validation outside the timer.
Large cases use CPU/CUDA, 16 host threads, N=1,048,576 and 5,242,880,
ordinary OLS/IV controls, ill-conditioned IV with fixed and growing cluster
intersections. The original study uses fast/comparable modes, five alternating
AB/BA pairs after one warmup per build/cell. Follow-up differences are explicitly
described above. All slow observations are retained. The independent large
fixture and MP90/MP120 oracle are unchanged from the previous correctness study.

Host: Athena, Intel Xeon Gold 6542Y (24 physical cores, 48 logical CPUs,
one socket and NUMA node), H100 NVL sm_90. Other users may share the host.
No task-owned builds or test campaigns overlap the timed fits. Host and GPU
load are recorded. These results cover synthetic affected-path regressions and
controls; the full core24 x 8 release matrix has not been run.

## Reproducibility

Evidence directory:
`pipeline/results/iv_covariance_optimization_20261008_102603/`.

Final summaries are `performance_summary.json`,
`all_performance_covariance_validation.json`, `validation_summary.json`
and `working_tree_manifest.json`. Raw per-fit records, warmups, order,
load snapshots, fixture hashes and worker diagnostics are retained under
`large_performance/`, `group_performance/`, `control_repeat/` and
`matched_warmup/` and `ordinary_iv_repeat/`. The analysis scripts reproduce
the reported statistics.

The artifact hashes are:

- Frozen accurate baseline module:
  `e22c14a08c74fa688127e6b9b05ac25f7d4e21324f4668b899449c021418b80b`
- Optimization module:
  `384e7de40adb375df097b861a5a24164f4cc6646e293d7f96fba93e073ac2c64`
- Optimized Stata plugin:
  `9e68e23c4a4433a924836d5d2fe295ae035f39788438a14ae94445ec5cfb6a1d`
- `src/ols.cpp` and its four mirrors:
  `ce86e348172bf7c073d5dcd5dd99f77d569e89eb81a342dfc9c5b92983ee65c4`

The native build verifies all 109 source/header files against the local
snapshot, with only the two compiled `ols.cpp` copies changed relative to the
frozen build. Unchanged object files are reused with unchanged flags and
dependencies. The native artifacts predate the metadata-only version edit;
final Stata checks use the updated package sources and the identical plugin.

The active-checkouts checker still targets absent Linux paths and a stale
approved release hash on this Mac. Its failure is recorded; the manifest was
not rewritten to suppress it. Explicit actual checkout/module paths govern
this experiment. Both local source checkouts are aligned and remain on
`codex/iv-first-stage-f`. No production installation, push or PR has
occurred for this candidate.

## Original 32-cell results, retained without filtering

The 5.24M rows remain here for auditability despite the process-history
limitation discussed above. They are not the final causal performance estimate.

| Rows | Fixture | Backend | Mode | Pairs | Median time change | Empirical 95% interval | Median seconds change |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1,048,576 | iv | CPU | fast | 5 | +2.47% | [+1.46%, +6.84%] | +0.031 |
| 1,048,576 | iv | CUDA | fast | 5 | -1.38% | [-3.92%, +4.69%] | -0.017 |
| 1,048,576 | iv | CPU | comparable | 5 | -2.08% | [-4.37%, +1.78%] | -0.026 |
| 1,048,576 | iv | CUDA | comparable | 5 | -5.16% | [-9.15%, +2.36%] | -0.062 |
| 1,048,576 | ols | CPU | fast | 5 | +2.30% | [+0.94%, +14.19%] | +0.027 |
| 1,048,576 | ols | CUDA | fast | 5 | -1.75% | [-3.52%, +6.05%] | -0.021 |
| 1,048,576 | ols | CPU | comparable | 5 | +0.11% | [-4.26%, +5.92%] | +0.001 |
| 1,048,576 | ols | CUDA | comparable | 5 | +2.20% | [-0.87%, +3.89%] | +0.024 |
| 1,048,576 | weak_fixed | CPU | fast | 5 | -1.09% | [-2.19%, -0.23%] | -0.072 |
| 1,048,576 | weak_fixed | CUDA | fast | 5 | -0.78% | [-1.52%, -0.25%] | -0.052 |
| 1,048,576 | weak_fixed | CPU | comparable | 5 | +0.27% | [-4.06%, +1.23%] | +0.018 |
| 1,048,576 | weak_fixed | CUDA | comparable | 5 | -0.93% | [-2.58%, +0.95%] | -0.061 |
| 1,048,576 | weak_many | CPU | fast | 5 | -2.14% | [-4.85%, -1.56%] | -0.174 |
| 1,048,576 | weak_many | CUDA | fast | 5 | -3.53% | [-4.52%, -2.36%] | -0.284 |
| 1,048,576 | weak_many | CPU | comparable | 5 | -4.74% | [-23.48%, -1.53%] | -0.271 |
| 1,048,576 | weak_many | CUDA | comparable | 5 | -3.62% | [-4.45%, -2.69%] | -0.293 |
| 5,242,880 | iv | CPU | fast | 5 | -0.10% | [-2.18%, +1.79%] | -0.008 |
| 5,242,880 | iv | CUDA | fast | 5 | -1.65% | [-28.57%, -0.25%] | -0.128 |
| 5,242,880 | iv | CPU | comparable | 5 | -0.40% | [-2.34%, +2.44%] | -0.031 |
| 5,242,880 | iv | CUDA | comparable | 5 | +0.56% | [-2.70%, +1.47%] | +0.043 |
| 5,242,880 | ols | CPU | fast | 5 | -1.71% | [-3.02%, -0.93%] | -0.127 |
| 5,242,880 | ols | CUDA | fast | 5 | +1.61% | [-3.07%, +5.08%] | +0.114 |
| 5,242,880 | ols | CPU | comparable | 5 | -0.15% | [-3.41%, +1.00%] | -0.011 |
| 5,242,880 | ols | CUDA | comparable | 5 | -3.18% | [-9.66%, +3.74%] | -0.249 |
| 5,242,880 | weak_fixed | CPU | fast | 5 | +19.16% | [-26.39%, +21.86%] | +4.280 |
| 5,242,880 | weak_fixed | CUDA | fast | 5 | -15.51% | [-22.57%, +18.54%] | -4.016 |
| 5,242,880 | weak_fixed | CPU | comparable | 5 | -1.30% | [-16.37%, +23.47%] | -0.360 |
| 5,242,880 | weak_fixed | CUDA | comparable | 5 | -13.95% | [-27.61%, +25.11%] | -3.556 |
| 5,242,880 | weak_many | CPU | fast | 5 | +2.74% | [-15.67%, +12.89%] | +0.869 |
| 5,242,880 | weak_many | CUDA | fast | 5 | -3.79% | [-17.01%, +14.75%] | -1.385 |
| 5,242,880 | weak_many | CPU | comparable | 5 | +10.48% | [-15.00%, +13.22%] | +3.027 |
| 5,242,880 | weak_many | CUDA | comparable | 5 | -13.62% | [-19.85%, +11.63%] | -4.405 |
