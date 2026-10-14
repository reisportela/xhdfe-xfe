> Development candidate on the PR #13 branch; never published. The first public 2.29 release is
> 2.29.0.<release date>; see `../RELEASE_NOTES_2.29.0.20261013.md` (candidate) and the release index.

# xhdfe 2.29.2.20261007 — development candidate

This unissued candidate includes the first-stage diagnostics and repeated-control
repair from [2.29.1](RELEASE_NOTES_2.29.1.20261007.md).

CPU and CUDA IV estimation now remove exactly proportional nonzero instrument
columns before absorbing fixed effects. Previously, CUDA absorption could turn
an exact duplicate such as `zdup=2*z1` into a tiny numerical direction, and the
subsequent rank checks rejected an otherwise identified model.

The filter preserves distinct, nearly proportional instruments and changes no
rank tolerances, convergence targets or estimator definitions. Identification
still fails when the remaining instruments do not identify every endogenous
regressor. General linear dependencies involving several columns and dependencies
created only by absorption remain outside this repair.

Native source mirrors and package metadata identify candidate 2.29.2.20261007.
The focused CPU/CUDA evidence and remaining release gates are recorded in the
[validation report](../../certification/iv-proportional-instruments-20261007.md).
