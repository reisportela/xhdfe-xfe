# 7 October 2026 — included controls repeated as IV instruments

Classification before the repair: **demonstrated estimator error**. Adding an
included exogenous regressor to `instruments()` leaves the mathematical
instrument span unchanged. In the saved 6,000-observation reproduction, CUDA
accepted this specification but changed coefficients across identical runs;
the largest absolute difference from the clean model was 2.27451e-4. The clean
model agreed with explicit reference estimates, and CPU preserved the span.

The repair removes exact variable-name overlap with included exogenous
regressors in the Stata frontend before absorption. Those controls already
enter the native instrument matrix automatically. A list containing only
included controls must still fail identification, rather than run as OLS.
Sample marking retains these variables through the structural regressor list.

This change does not classify numerical near-collinearity, change a rank
threshold, or fix proportional columns under different variable names. The
native CUDA rank issue remains separate.

The permanent reproducer is
`tests/analytic_precision/iv_repeated_control_contract.do`. Validation and the
fixing version are recorded in `docs/certification/iv-repeated-controls-20261007.md`.
