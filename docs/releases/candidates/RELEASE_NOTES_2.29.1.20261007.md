> Development candidate on the PR #13 branch; never published. The first public 2.29 release is
> 2.29.0.<release date>; see `../RELEASE_NOTES_2.29.0.20261013.md` (candidate) and the release index.

# xhdfe 2.29.1.20261007 — development candidate

This unissued candidate includes the first-stage diagnostics introduced in
[2.29.0](RELEASE_NOTES_2.29.0.20261007.md).

Stata IV regressions now remove included exogenous regressors from the explicit
`instruments()` list before absorption. Those controls already enter the
instrument matrix automatically. Previously, repeating them could make CUDA
estimates vary across identical fits. CPU and CUDA now receive the same clean
instrument specification. Lists containing only included controls return an
identification error with no estimates retained.

The filter uses exact variable names after Stata expands the varlist; it does
not change numerical rank thresholds or the native plugin. Proportional
instruments under different variable names remain a separate CUDA limitation.

The permanent CPU/CUDA guard is
`tests/analytic_precision/iv_repeated_control_contract.do`, with additional
first-stage checks in `tests/stata/iv_firststage_contract.do`.
See the [validation report](../../certification/iv-repeated-controls-20261007.md).
