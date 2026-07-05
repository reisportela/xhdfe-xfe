# Overview

## What xhdfe does

`xhdfe` estimates linear models with one or more high-dimensional fixed effects
(HDFE): the absorbed-dummy designs — worker–firm panels, patent–inventor data,
multi-way categorical effects — that would be infeasible to fit by explicit
dummy expansion. It follows the conventions of
[`reghdfe`](https://github.com/sergiocorreia/reghdfe) (Correia 2016): the same
absorbed-degrees-of-freedom logic, singleton handling, reporting, and stored
results. Under the default `reghdfe-comparable` tolerance mode, coefficients
match `reghdfe` at the same nominal tolerance.

## One estimator, three front-ends

Stata, Python, and R are thin wrappers over a single compiled C++ core, so a
model specified in any of the three produces the same estimates. The core
absorbs fixed effects with the method of alternating projections
(Guimaraes–Portugal 2010; Gaure 2013), accelerated and, on ill-conditioned
graphs, automatically handed off to a stable conjugate-gradient / Schwarz /
LSMR solver. The regression on the residualized data, the variance-covariance
estimation, and the degrees-of-freedom accounting all live in the same core.

## Feature summary

| Capability | Notes |
| --- | --- |
| Multiway HDFE | Any number of absorbed dimensions; two-way categorical interactions. |
| Heterogeneous slopes | Group-specific slopes with or without the level FE. |
| IV / 2SLS | Endogenous regressors instrumented after absorption. |
| Weights | Analytic, frequency, probability, importance. |
| Inference | Unadjusted, robust (HC1-style), and one-/multi-way clustering. |
| Small-sample corrections | reghdfe-style DoF adjustments and fixest-style `ssc` controls. |
| Fixed-effect recovery | Per-observation FE contributions (`savefe` / `fixef()` / `retain_fes`). |
| Group / individual FEs | Group-level outcomes with absorbed individual effects. |
| Mobility groups | Connected-component identification across FE dimensions. |
| GPU (optional) | CUDA absorber, opt-in per call; fail-closed in Stata/R, check `gpu_used_` in Python. See the [GPU guide](gpu.md). |

## Backends

CPU is the reference backend and defines the numerical results. The optional
CUDA backend accelerates the fixed-effect absorber on large problems; it is
requested explicitly (`gpubackend(cuda)` in Stata, `backend = "cuda"` in R,
`XHDFE_GPU_BACKEND=cuda` in Python). Stata and R are fail-closed — they error
rather than silently returning CPU results if the GPU path is unavailable or
fails; in Python, `.fit()` falls back to CPU instead, so check
`reg.gpu_used_ == 1`. See the [GPU (CUDA) guide](gpu.md) for how to install with
the GPU feature in every version of the package.

## Tolerance modes

- `reghdfe-comparable` (default) — the accelerated absorber stops when one full
  sweep moves the data by less than `tolerance()` in relative norm, the same
  meaning `reghdfe` attaches to its tolerance, so coefficients are directly
  comparable at the same nominal tolerance.
- `xhdfe-fast` — a faster stopping rule for exploration and speed benchmarking;
  effective precision is data-dependent and can be looser on ill-conditioned
  designs.
- `strict-residual` — a heavier audit mode that treats the final group-mean
  residual check as authoritative.

## Validation

`xhdfe` is cross-validated against `reghdfe` (Stata),
[`fixest`](https://github.com/lrberge/fixest) (R),
[`pyfixest`](https://github.com/py-econometrics/pyfixest) (Python), and
[`FixedEffectModels.jl`](https://github.com/FixedEffects/FixedEffectModels.jl)
(Julia), and the three packages are checked against each other for numerical
agreement. The software is a proof of concept; validate estimates for your own
research design. See [`../DISCLAIMER.md`](../DISCLAIMER.md).

## References

- Guimaraes, P., and P. Portugal. 2010. "A simple feasible procedure to fit models with high-dimensional fixed effects." *Stata Journal* 10(4): 628–649.
- Gaure, S. 2013. "OLS with multiple high dimensional category variables." *Computational Statistics & Data Analysis* 66: 8–18.
- Correia, S. 2016. "reghdfe: Estimating linear models with multi-way fixed effects." 2016 Stata Conference.
