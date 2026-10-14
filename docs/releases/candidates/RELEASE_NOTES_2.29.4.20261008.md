> Development candidate on the PR #13 branch; never published. The first public 2.29 release is
> 2.29.0.<release date>; see `../RELEASE_NOTES_2.29.0.20261013.md` (candidate) and the release index.

# xhdfe 2.29.4.20261008 — development candidate

This unissued candidate retains the covariance precision repair from
[2.29.3](RELEASE_NOTES_2.29.3.20261007.md) and reduces repeated work in its
wide multiway clustered covariance path. Contiguous cluster scores and reusable
solve/transform buffers avoid per-group allocations. Calculations whose
covariance results were discarded are skipped; the stable influence formula,
precision, summation order, degrees-of-freedom and PSD policies are unchanged.

Focused native and Stata CPU/CUDA accuracy checks pass, including 84 added
native cases with independent MP90/MP120 references. All 268 weak-IV performance
covariance checks pass the unchanged reference limit. The supplemental group
CUDA tests retain the same 32 absorption refusals as the frozen baseline.

Paired Athena timings against the correct 2.29.3 baseline show modest reductions
in total fit time for many-intersection regressions: the 1.05M-row CPU-fast
11-pair repeat improves by 3.82%, and CUDA-fast by 3.53% in the original five-pair
study. At 5.24M rows, a three-pair follow-up with matched conditioning fits gives
3.20% on CPU and 3.79% with CUDA. Timing noise limits precision; few-cluster
and ordinary controls show no consistent penalty. Original timings, including
a detected process-order effect and initial slower control observations, are
retained in the report.

This candidate has not been pushed, issued or promoted through the
full release matrix. See the
[validation report](../../certification/iv-covariance-optimization-20261008.md).
