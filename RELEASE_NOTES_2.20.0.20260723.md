# xhdfe 2.20.0.20260723 — Gelbach post-audit inference and diagnostics

This release closes the actionable conditions of the empirical-utility audit
for `xhdfegelbach` and adds one inferential extension:
joint-covariance inference for shares normalized by the base coefficient.
The Gelbach estimand, decomposition identity, stacked-GMM conventions,
classification threshold, solver tolerances, convergence criteria, and
standard non-Gelbach `xhdfe` path are unchanged.

## Joint inference for base-coefficient shares

- The public result contract now includes the requested-VCE base covariance,
  `Cov(delta, b_base)`, and `Cov(total, b_base)`.
- `shares(base)` in Stata and `share="base"` in Python/R use the complete
  delta-method variance

  ```text
  Var(delta/b) =
      Var(delta)/b^2
    + delta^2 Var(b)/b^4
    - 2 delta Cov(delta,b)/b^3.
  ```

- The resulting convention is labelled
  `joint_base_covariance_delta_method`.
- `base_fixed` remains available, numerically unchanged, and explicitly
  labelled as descriptive fixed-denominator scaling.
- A requested share with an undefined denominator is missing and emits one
  visible warning; Stata also retains that warning in `r(notes)`.

## Identification and finite-sample diagnostics

- Every frontend returns the per-X1 squared residual-norm ratio after
  absorbing the declared fixed effects.
- An X1 column with a ratio in `(1e-9, 1e-4]` remains in the standard estimand
  but receives a visible, recorded near-FE-collinearity warning. The
  `XHDFE_GELBACH_NEAR_COLLINEAR_WARN=0` switch suppresses only that warning;
  it does not change classification or any estimate.
- One-way clustered results return the retained cluster count. Fewer than 30
  clusters triggers a caution note without changing the requested VCE or
  substituting another procedure.
- Results now expose `df_base`, `df_full`, and the observed-block full-model
  coefficients (`gamma`) alongside the existing component and sample
  metadata.
- A saturated full model with no positive residual degrees of freedom raises
  a catchable error rather than returning non-finite inference.

## CPU/CUDA contract

- Python and R add an opt-in `gpu` request matching Stata's Gelbach option.
- All three frontends expose truthful requested/attempted/used/backend/status
  diagnostics and the full-model absorption diagnostics.
- CUDA applies only to the full-model FE-absorption phase. Base regression,
  fixed-effect recovery, component construction, covariance algebra, and
  reporting remain CPU work.
- `threads()` / `num_threads` is a per-phase OpenMP cap; phases execute
  sequentially and may use fewer threads.

## Documentation and empirical boundaries

- The three help surfaces document the new covariance matrices, warning
  thresholds, cluster and degrees-of-freedom metadata, `gamma`, GPU status
  fields, and full `share=base` formula.
- Binary outcomes are explicitly described as linear probability models;
  logit-scale decomposition is a separate estimator.
- Formula/factor notation is not interpreted by the decomposition wrappers.
  Researchers should generate a full-rank numeric indicator/interaction
  matrix, omit the intercept and one reference category, and pass explicit
  named blocks.
- Autonomous release media include the pinned official CRAN source archive
  `third_party/Rcpp_1.1.2.tar.gz` plus its SHA-256/license provenance, and the
  R documentation gives a network-disabled local-library installation route.
- The examples are described accurately as the standard and absorbed-target
  examples executed in Stata, Python, and R.

This tranche does not add multiway clustering, wild-cluster bootstrap,
common HDFE in both specifications, nonconditional recovered-FE covariance,
IV/LATE allocation, dynamic-panel corrections, nonlinear or distributional
decompositions, Oaxaca/KHB/mediation estimators, or a Stata tidy/export
subcommand.

## Version surface

- Shared C++/Python/R package and release tag: `2.20.0.20260723`.
- Stata `xhdfe`: `2.20.0`.
- Stata `xhdfegelbach`: `1.4.0`.
- Stata `xhdfeakm`: `1.7.2` (number unchanged).
- Stata `xhdfeconnected`: `1.2.1` (number unchanged).
- Stata `xfe`: `1.10.1` (number unchanged).
- All production Stata files carry the common release date `23jul2026`.

Release acceptance remains conditional on the rebuilt-Release validation
gates and artifact hashes recorded in
`XHDFEGELBACH_CODEX_POSTAUDIT_REMEDIATION_REPORT_20260723.md`; these release
notes do not anticipate those test results.
