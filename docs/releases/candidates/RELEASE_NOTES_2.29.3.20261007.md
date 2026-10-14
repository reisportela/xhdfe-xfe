> Development candidate on the PR #13 branch; never published. The first public 2.29 release is
> 2.29.0.<release date>; see `../RELEASE_NOTES_2.29.0.20261013.md` (candidate) and the release index.

# xhdfe 2.29.3.20261007 — development candidate

This unissued candidate includes the first-stage diagnostics and redundant-
instrument repairs from [2.29.2](RELEASE_NOTES_2.29.2.20261007.md).

Ill-conditioned IV regressions with absorbed intercepts and multiway clustered
standard errors now preserve covariance precision through the final eigenvalue
correction. Cluster influence outer products avoid cancellation in the wide
sandwich, and the resulting covariance stays wide through degrees-of-freedom
scaling and the existing standardized full-matrix/slope-block PSD rule.

The repair fixes spurious changes to otherwise accurate intercept variances
and preserves genuine negative-eigenvalue corrections. It uses the existing
precision type and changes no absorption, rank, convergence or reporting
threshold. Ordinary OLS and ordinary-precision IV keep their existing paths.

Focused CPU/CUDA Python and Stata checks pass. Local CPU measurements find an
approximately 0.06–0.11 ms cost for the tiny equivalent-cluster reproducer.
Many distinct cluster intersections increase the measured cost to 8–15% on
weak-IV stress cases. Ordinary controls show no slowdown. That narrow path
needs performance work before a PR. Concurrent Athena CUDA measurements show
approximately 8–13% paired median costs in the distinct-cluster cases, with
wide uncertainty from overlapping load. Clean CUDA timing and the complete
release performance matrix remain pending. See the
[validation report](../../certification/iv-covariance-precision-20261007.md).
