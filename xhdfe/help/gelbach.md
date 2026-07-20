# xhdfe Gelbach decomposition help

Release status: the absorbed-target and empirical-reporting extensions are
available in xhdfe 2.19.0.20260720 (`xhdfegelbach` 1.3.0). Inspect the installed
package version with `python -m xhdfe --version`.

`xhdfe.gelbach` implements Gelbach's order-invariant accounting of the movement
of coefficients between one base and one full linear specification. It is
specification accounting, not causal mediation. A causal interpretation needs
a separately justified research design.

## Open this help

```bash
python -m xhdfe gelbach
xhdfe-help gelbach
xhdfe-help gelbach --path
```

Inside Python:

```python
import xhdfe

print(xhdfe.help_text("gelbach"))
```

## Estimands

The standard estimand compares, on the same sample and with the same weights,

```text
base: y = X1 b_base + error
full: y = X1 b_full + sum_g X2_g gamma_g + absorbed fixed effects + error
```

with an implicit intercept in both models. The returned block contributions
sum to `b_base - b_full` up to floating-point error. Standard mode fails closed
when an X1 or X2 column is not identified in the full model.

`absorbed_targets=` activates the distinct `absorbed_target_allocation`
estimand. It is for an X1 target that belongs to the span of an added FE, such
as a worker-invariant group indicator with worker FE. The target's full-model
coefficient is imposed at zero and labelled `imposed_zero`; it is not an
estimated within-FE coefficient. Every undeclared omission and generic rank
failure remains an error.

## Standard example: focal plus common controls

```python
import numpy as np
from xhdfe import gelbach

result = gelbach.decompose(
    y,
    np.column_stack([target, age, baseline_score]),
    x2_groups={
        "human_capital": np.column_stack([education, education_sq]),
        "job": np.column_stack([tenure, experience]),
    },
    fes={"firm": firm_id, "occupation": occupation_id},
    vce="cluster",
    cluster=worker_id,
    x1_names=["target", "age", "baseline_score"],
    focal="target",
)

table = gelbach.tidy(result, share="movement")
observed_total = gelbach.contrast(
    result, "target", ["human_capital", "job"]
)
```

`focal` is reporting metadata only. Here `age` and `baseline_score` remain in
X1 and therefore remain in both specifications. Moving a common control to
`x2_groups` would change the base model and the decomposition.

## `decompose`

```python
gelbach.decompose(
    y,
    x1,
    x2_groups=None,
    fes=None,
    vce="unadjusted",
    cluster=None,
    gamma0=False,
    cov0=False,
    tol=1e-8,
    num_threads=0,
    weights=None,
    fweights=False,
    absorbed_targets=None,
    x1_names=None,
    focal=None,
)
```

Arguments:

- `y`: finite numeric vector of length n.
- `x1`: finite `(n, p)` base-design matrix, without a constant. A 1-D vector
  is treated as one column. At least one X1 column is required.
- `x2_groups`: mapping from unique nonempty group names to finite vectors or
  matrices. Each entry is one simultaneous added block.
- `fes`: mapping from unique names to length-n integer FE identifiers. Exact
  integer-valued floating arrays are accepted and converted.
- `vce`: `"unadjusted"`, `"robust"`, or `"cluster"`.
- `cluster`: one length-n cluster identifier vector. It is required only for
  `vce="cluster"`; one-way clustering requires at least two clusters. Passing
  it with another VCE is an error. Multiway clustering is not implemented.
- `gamma0`: retain only the auxiliary-regression variance, matching `b1x2`.
- `cov0`: omit robust stacked cross terms, matching `b1x2`; it is a no-op for
  unadjusted inference.
- `tol`: positive FE-absorption tolerance. It does not control the separate
  FE-collinearity classifier.
- `num_threads`: OpenMP thread request; zero uses the library default.
- `weights`: finite, strictly positive analytic weights by default.
- `fweights`: interpret `weights` as positive integer frequency weights.
- `absorbed_targets`: zero-based X1 indices or a length-p Boolean mask. Every
  declared target must be classified as FE-absorbed by the backend.
- `x1_names`: p unique, nonempty names; `_cons` is reserved. Names affect
  labels and selectors only, never the design matrix.
- `focal`: X1 names, zero-based indices, or a length-p Boolean mask selecting
  one or more reporting targets. It never removes an X1 column from a model.

At least one observed group or FE dimension is required. All block names must
be unique across `x2_groups` and `fes`. The full design must pass every rank
guard. Polynomials, factor indicators, splines, bins and interactions are
supported as explicit numeric columns grouped by the researcher.

### Sample and weights

Unlike the Stata command, the Python wrapper does not silently mark out missing
rows: every supplied numeric input must be finite. Construct one common sample
before calling `decompose`. Recursive singleton removal can still occur inside
an FE fit. Inspect:

- `n_obs_input`: rows supplied to the backend;
- `n_obs`: retained row count after FE processing;
- `n_obs_effective`: retained rows, or the sum of frequency weights;
- `n_singletons_dropped`: recursively removed FE singleton observations.

The same sample and weight inner product are used for base, full and auxiliary
projections.

### Returned dictionary

Coefficient and component objects:

- `b_base`, `b_full`: length-p X1 coefficient arrays;
- `b_full_status`: `estimated` or `imposed_zero` per X1 column;
- `delta[name]`: `coef`, `se`, and `se_type` arrays over
  `[x1 columns..., _cons]`;
- `total`: `coef`, `se`, `cov`, and `se_type` for total movement;
- `total_cov`: top-level alias of `total["cov"]`;
- `cov`: joint covariance of every component. Its order is group-major:
  `[group_1:x1..., group_1:_cons, group_2:x1..., group_2:_cons, ...]`;
- `fe_total`: aggregate FE `members`, `coef`, `se`, `cov`, and `se_type`, or
  `None` when there are no FE groups.

Names and classification:

- `names`, `group_kinds`, `labels`, `x1_names`;
- `focal_indices`, `focal_names`;
- `focal_status`, `absorbed_mask`, `absorbed_targets`,
  `absorbed_target_names`;
- `estimand`: `coefficient_movement` or `absorbed_target_allocation`;
- `identity_status`: `exact_ols` or `exact_ols_constrained`;
- `causal_interpretation`: always `False`.

Inference and diagnostics:

- `vce`, `gamma0`, `cov0`, `tol`, `df_full`;
- `total_se_type`, `inference_status`;
- `absorbed_target_inference_valid`, `absorbing_fe_index`;
- `fe_collinear_ss_ratio_tol` and `fe_collinear_relative_norm_tol`;
- `identity_gap`, `converged`, and `notes`;
- the four sample-count fields documented above.

Always require `converged is True` and inspect `notes`. `identity_gap` is only a
summation consistency check; it does not certify each FE split, inference, or
causal interpretation.

## Inference contract

For observed blocks, the covariance follows the random-design stacked-moment
variance of Gelbach's official `b1x2`. It includes uncertainty in the auxiliary
projections and is not the smaller variance conditional on the realised design.

Absorbed-FE component inference is `conditional_gamma0`: uncertainty from
estimating the absorbed effects is not fully included. A total combining
observed and FE blocks is therefore labelled `mixed_*_conditional_fe`. With
multiple mobility components, the aggregate `fe_total` and overall movement
are the preferred normalization-invariant objects; per-FE-dimension splits can
depend on normalization.

For an absorbed target, `total_j = b_base_j - 0` is the base-coefficient
estimator itself, so its target-target total variance equals the requested base
VCE. Inference for a target invariant within an absorbing FE must be clustered
at that FE dimension. Unadjusted, robust, or crossed clustering is retained for
descriptive accounting but emits a warning and sets
`absorbed_target_inference_valid=False`.

The severe-near-collinearity diagnostic can also emit `RuntimeWarning` while
leaving points unchanged. Such a warning means the within-block SE split is
sensitive; tightening `tol` does not select a unique correct split.

## `tidy`

```python
gelbach.tidy(
    result,
    *,
    focal=None,
    include_intercept=False,
    include_total=True,
    include_full=True,
    conf_level=0.95,
    share=None,
    share_tol=1e-12,
)
```

This is post-processing only; no model is re-estimated. It returns a list of
dictionaries suitable for pandas, Polars or a CSV writer.

- `focal`: optional override of the result's reporting selector.
- `include_intercept`: add the implicit `_cons` row.
- `include_total`: add `total_movement`.
- `include_full`: add `full_model_residual` for X1 coefficients. Its SE is
  missing because that covariance is not in the public result contract.
- `conf_level`: normal-approximation confidence level strictly between 0 and 1.
- `share`: `None`, `"movement"`, `"base"`, or `"base_fixed"`.
- `share_tol`: nonnegative absolute denominator threshold. Shares at or below
  it are undefined and returned as NaN.

Every component row contains `coefficient`, `component`, `component_kind`,
`estimate`, `std_error`, `conf_low`, `conf_high`, `conf_level`, and `se_type`.
When shares are requested it also contains `share`, `share_std_error`,
`share_conf_low`, `share_conf_high`, `share_defined`, `share_denominator`,
`share_se_type`, `share_units`, and `share_tol`.

Share meanings:

- `movement`: `delta_g / sum_h(delta_h)`. Its SE uses the full joint component
  covariance and the delta method. The total share is one with SE zero.
- `base`: `delta_g / b_base`. Points are returned, but ratio SEs and intervals
  are missing because `Cov(delta, b_base)` is not public.
- `base_fixed`: the explicit descriptive convention that scales component SEs
  while holding the reported base coefficient fixed. It is labelled
  `fixed_base_denominator_scaling`, not full ratio inference.

Shares remain signed. Negative values and totals above 100 percent are never
truncated or renormalized. Contributions in original units remain primary.

## `contrast`

```python
gelbach.contrast(result, focal, groups, *, conf_level=0.95)
```

`focal` must select exactly one X1 coefficient. `groups` may be:

- a sequence of group names, all with weight one;
- a mapping `{group_name: weight}`;
- a numeric sequence with one weight in `result["names"]` order.

The function returns `coefficient`, a complete name-to-weight mapping,
`estimate`, `std_error`, `conf_low`, `conf_high`, `conf_level`, and `se_type`.
It uses the joint covariance, including cross-component terms, and never
re-estimates the model. A contrast containing an FE block is labelled
`joint_covariance_including_conditional_fe`.

## Absorbed-target example

```python
result = gelbach.decompose(
    y,
    np.column_stack([group, experience]),
    x2_groups={"job": job_controls},
    fes={"worker": worker_id, "firm": firm_id},
    vce="cluster",
    cluster=worker_id,
    absorbed_targets=[0],
    x1_names=["group", "experience"],
    focal="group",
)

assert result["b_full_status"][0] == "imposed_zero"
assert result["absorbed_target_inference_valid"]
```

## CPU, CUDA and frontends

The Python `gelbach.decompose` signature currently has no `gpu=` switch and
should be treated as the CPU companion interface. The Stata `xhdfegelbach`
command exposes explicit `gpu` and backend diagnostics. All frontends share the
same estimator implementation and are validated for numerical parity on their
common feature surface.

Index conventions:

- Python input and output indices are zero-based;
- R accepts one-based input indices but returns zero-based metadata;
- Stata accepts variable names and returns zero-based metadata plus names.

## Deliberate boundaries

The current command does not implement:

- IV/2SLS Gelbach decompositions or LATE allocation;
- multiway clustering or wild-cluster bootstrap;
- nonconditional sampling uncertainty for recovered FE components;
- a separate HDFE set absorbed in both base and full models;
- split-panel jackknife/dynamic-panel corrections;
- kernel, MM-quantile, KHB/GLM, distributional, Oaxaca, or causal-mediation
  estimators.

Low-dimensional FE common to both specifications can be represented by
explicit indicator columns in X1 and hidden from reporting with `focal`.
Because the intercept is implicit, use a full-rank reference coding rather
than all category indicators.
High-dimensional entries in `fes` are added full-model components.

## Further examples and validation

- `examples/gelbach_example.py`: standard focal/common-control/HDFE workflow;
- `examples/gelbach_absorbed_target.py`: imposed-zero target workflow;
- `VALIDATE_GELBACH.py`: b1x2, LSDV and absorbed-target oracles;
- `VALIDATE_GELBACH_FRONTENDS.py`: Python/Stata/R parity and example gates;
- `XHDFEGELBACH_EMPIRICAL_APPLICATION_COVERAGE_20260720.md`: paper-by-paper
  coverage and scientific boundaries.

Reference: Gelbach, J. B. (2016), “When Do Covariates Matter? And Which Ones,
and How Much?”, *Journal of Labor Economics* 34(2), 509–543.
