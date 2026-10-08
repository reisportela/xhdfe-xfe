# Covariance optimization lessons, 2026-10-08

## Retain the accurate covariance and remove discarded work

The previous covariance fix computed stable cluster-influence outer products
but also completed an older meat/sandwich/intercept calculation whose result
was replaced. Cluster-score matrices and two solve/transform temporaries also
allocated storage for every group. With millions of intersections, those
costs repeat millions of times.

The candidate stores retained-path scores contiguously, reuses scratch vectors,
and skips discarded covariance arithmetic. Its stable contribution calculation,
summation order and precision are unchanged. Score-diagonal range checks and
checks on returned quantities remain. Other paths still compute the legacy
quantities they use. These are implementation changes, not tolerance tuning.

Regression coverage:
`tests/analytic_precision/wide_cluster_covariance_reuse.cpp` and
`wide_cluster_covariance_reuse_contract.py`, together with the existing IV
intercept/PSD contracts. The new direct check covers constant positions,
regressor counts, weights, large/small response units and both cluster SSC
rules, using an independent MP90/MP120 reference.

## Evaluation-harness error: offsets can remove the intended conditioning

The initial supplemental fixture used two near-parallel varying columns but
gave them different offsets without a constant column. The offsets themselves
made the represented Gram matrix well-conditioned, so the frozen baseline did
not enter the wide path. The check correctly failed its path assertion.

Repair: use centered varying columns for no-constant cases, and keep offset
cases with an explicit constant. Apply the same construction to the independent
reference. No estimator or threshold changes. The initial fixture and outcome
remain in the run evidence.

## Missing coverage: supplemental dependencies and pre-existing CUDA refusals

The group/individual reference imports SciPy; the frozen minimal Python runtime
did not have it. Install it only in the experiment's test-dependency directory.

After that repair, the direct group test has 96 CPU passes. On CUDA the candidate
and frozen baseline have the same 64 passes and 32 absorption refusals, with
exact equality of every result/refusal record. Those refusals occur before the
changed covariance code. They remain an explicit coverage limit and do not
authorize changing the absorption solver within this optimization.

Do not label the complete supplemental CUDA battery a pass, discard the
refusals, relax its tolerances, or interpret this narrow comparison as release
certification. Preserve the baseline comparison and classify separately from
new covariance regressions.

## Reproducibility

See [the validation report](certification/iv-covariance-optimization-20261008.md).
Evidence is under
`pipeline/results/iv_covariance_optimization_20261008_102603/`, outside the Git roots.



## Evaluation-harness limitation: immediate process history at 5.24M rows

The first large weak-fixed CPU cell produced a bimodal timing pattern in both
builds: roughly 22 seconds for an immediately repeated fit, versus 27 seconds
after switching processes. Five alternating AB/BA pairs left the paired median
dominated by whichever order occurred three times. The physical cause has not
been established; a particular cache or scheduler explanation is not evidence.

Preserve and report the original timings. A separately declared follow-up uses
one same-build conditioning fit immediately before every measured fit, with
unchanged fixtures, estimator defaults, code and tolerances. It covers the
large fixed-cluster detector and the many-intersection case on CPU and CUDA.
Conditioning results are recorded and checked, not silently discarded. This
focused diagnostic and its limited repetition do not replace release-wide
performance certification.

## Final focused evaluation

The matched-conditioning fixed-cluster CPU estimate is +0.15%, while the large
many-intersection estimates are -3.20% on CPU and -3.79% with CUDA. Each has only
three pairs and broad empirical intervals; retain that uncertainty. The original
1.05M-row many-intersection cases all favor the candidate, with a separate
11-pair CPU-fast repeat at -3.82%. Initial ordinary OLS and IV control slowdowns
do not reproduce in their explicit 11-pair repeats. Small effects remain hard
to resolve on the shared host; these checks do not establish universal speedups.

All 268 weak-IV performance covariance outputs pass the unchanged independent
reference, including conditioning fits. The optimization remains a local
development candidate with full source/metadata alignment in both local
checkouts and no production binary installation or release certification.
