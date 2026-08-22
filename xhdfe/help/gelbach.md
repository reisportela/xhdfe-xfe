# xhdfe Gelbach decomposition help

Release version: xhdfe 2.24.2.20260822 (`xhdfegelbach` 1.6.0). This version
includes the bootstrap, table and waterfall interfaces documented below.
Inspect the installed package version with `python -m xhdfe --version`.

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
base: y = X1 b_base + common fixed effects C + error
full: y = X1 b_full + sum_g X2_g gamma_g
             + common fixed effects C + added fixed effects A + error
```

with an implicit intercept in both models. `common_fes` conditions both
specifications and is not decomposed; `fes` contains the added, decomposable FE
dimensions. Equivalently, the coefficient-movement identity is formed after
partialling out the common-FE space. The returned observed and added-FE block
contributions sum to `b_base - b_full` for every X1 slope up to floating-point
error. Without common FEs, the historical intercept identity is also retained.
With common FEs, the intercept allocation is normalization-dependent and its
public point estimates, contributions and covariance entries are deliberately
missing. Standard mode fails closed when an X1 or X2 column is not identified
in the relevant common-FE-conditioned design.

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
    common_fes={"year": year_id},
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
X1 and therefore remain in both specifications; the high-dimensional year
effect is absorbed in both through `common_fes`. Moving a common control to
`x2_groups` or moving a common FE to `fes` would change the base model and the
decomposition.

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
    gpu=False,
    connected="diagnose",
    connectivity_fes=None,
    common_fes=None,
    sample_info=False,
    fe_variance_ratio_min=0.35,
)
```

Arguments:

- `y`: finite numeric vector of length n.
- `x1`: finite `(n, p)` base-design matrix, without a constant. A 1-D vector
  is treated as one column. At least one X1 column is required.
- `x2_groups`: mapping from unique nonempty group names to finite vectors or
  matrices. Each entry is one simultaneous added block.
- `fes`: mapping from unique names to length-n integer FE identifiers added
  only in the full specification. Each dimension is a decomposable FE block.
  Exact integer-valued floating arrays are accepted and converted.
- `common_fes`: optional mapping from unique names to length-n integer FE
  identifiers absorbed in both base and full specifications. These FEs
  condition the decomposition and do not become contribution blocks. The
  argument follows the historical arguments so earlier positional calls
  retain their meaning.
- `sample_info`: opt in to retained-sample provenance. When true, the result
  materializes the zero-based retained positions into the arrays supplied by
  the caller, a length-n Boolean mask, and a stable non-cryptographic hash.
  The default is false and avoids the extra O(n) hash/output work.
- `fe_variance_ratio_min`: nonnegative residual-to-total X1 squared-norm
  threshold for the conditional FE-variance validity gate; default `0.35`.
  It changes metadata and inferential labels only.
- `vce`: `"unadjusted"`, `"robust"`, or `"cluster"`.
- `cluster`: one length-n cluster identifier vector. It is required only for
  `vce="cluster"`; one-way clustering requires at least two clusters. Passing
  it with another VCE is an error. Multiway clustering is not implemented.
- `gamma0`: retain only the auxiliary-regression variance, matching `b1x2`.
- `cov0`: omit robust stacked cross terms, matching `b1x2`; it is a no-op for
  unadjusted inference.
- `tol`: positive FE-absorption tolerance. It does not control the separate
  FE-collinearity classifier.
- `num_threads`: OpenMP team-size request for each computational phase; zero
  uses the automatic policy. A positive request bypasses automatic
  workload-size caps and is limited only by the processor/OpenMP capacity
  visible to the process. Phases execute sequentially: this is the team
  budget for an active phase, not a sum of simultaneously reserved teams.
  The returned diagnostics distinguish the request, effective budget, team
  size, and workers observed doing useful work.
- `weights`: finite, strictly positive analytic weights by default.
- `fweights`: interpret `weights` as positive integer frequency weights.
- `absorbed_targets`: X1 names, zero-based indices, or a length-p Boolean
  mask. Names are resolved against `x1_names` (or the generated `x1_1`, ...
  labels). Every declared target must be classified as FE-absorbed by the
  backend.
- `x1_names`: p unique, nonempty names; `_cons` is reserved. Names affect
  labels and selectors only, never the design matrix.
- `focal`: X1 names, zero-based indices, or a length-p Boolean mask selecting
  one or more reporting targets. It never removes an X1 column from a model.
- `gpu`: request CUDA for the full-model FE-absorption phase only. The base
  regression, component construction and covariance calculations remain CPU
  work. The request is opt-in and a non-use/fallback is reported truthfully in
  the returned diagnostics.
- `connected`: `"diagnose"` (default) reports the connectivity contract without
  changing the fit. `"require"` fails closed unless there are no common FEs and
  exactly two retained added-FE dimensions form one mobility component. It
  deliberately rejects common-FE and 3+ added-FE designs because their
  per-dimension split is not yet connectivity-certified.
- `connectivity_fes`: optional pair of FE names or zero-based indices selecting
  the retained-sample pair diagnostic among the added dimensions in `fes`.
  Common FEs cannot be selected. The default is the first two added-FE
  dimensions. With common FEs or 3+ added FEs this is diagnostic only and
  never upgrades the global split to identified.

At least one observed group or added FE dimension is required: common FEs alone
do not define anything to decompose. All block names must be unique across
`x2_groups`, `common_fes`, and `fes`. The common-conditioned base and full
designs must pass every rank guard. Polynomials, factor indicators, splines,
bins and interactions are supported as explicit numeric columns grouped by the
researcher. Formula or factor-variable notation is not parsed: generate a
full-rank set of numeric indicator/interaction columns first, omit a reference
category, and pass those columns as one or more named blocks.

With a binary outcome this command estimates a linear probability model;
decomposition on a logit or other nonlinear scale is a separate estimator and
is not supplied here.

### Sample and weights

Unlike the Stata command, the Python wrapper does not silently mark out missing
rows: every supplied numeric input must be finite. Construct one common sample
before calling `decompose`. Recursive singleton removal can still occur inside
an FE fit. Inspect:

- `n_obs_input`: rows supplied to the backend;
- `n_obs`: retained row count after FE processing;
- `n_obs_effective`: retained rows, or the sum of frequency weights;
- `n_singletons_dropped`: recursively removed FE singleton observations.

Set `sample_info=True` when an empirical pipeline must retain auditable row
membership rather than counts alone. The additional fields are:

- `sample_index`: strictly increasing, zero-based positions into the supplied
  arrays;
- `sample_mask`: Boolean vector of length `n_obs_input`, exactly equivalent to
  `sample_index`;
- `sample_hash`: 16-character retained-sample identifier;
- `sample_hash_algorithm`: `fnv1a64-le-v1`;
- `sample_index_scope`: `input_rows_zero_based`;
- `sample_info_requested`: whether the opt-in was active.

The canonical hash byte stream is the ASCII domain tag
`xhdfe-gelbach-sample-v1`, followed by unsigned 64-bit little-endian
`n_obs_input`, retained count, and each retained zero-based position. It binds
both membership and input row order. FNV-1a is used as a compact reproducible
provenance identifier, not as a cryptographic integrity or security
guarantee. When `sample_info=False`, the five materialized provenance fields
are `None`; the default numerical path and result values are unchanged.

The union of common and added FE dimensions defines recursive singleton
removal. The base is then estimated on that retained full-model sample without
performing a second, specification-specific singleton drop. The same sample,
weight inner product and common-FE projection are used for base, full and
auxiliary projections.

With at least two added FE dimensions, the retained selected pair also reports
`n_mobility_components`, `largest_mobility_component_n_obs`,
`largest_mobility_component_share`, and
`largest_mobility_component_weight_share`. The first two shares need not
refer to the same component: one maximizes physical rows and the other
retained weight. `connectivity_fe_indices` and `connectivity_fe_names` record
the authoritative added-FE pair. Without common FEs,
`mobility_component_scope` preserves the historical
`first_two_fe_dimensions`/`selected_fe_pair` labels. With common FEs it is
`first_two_added_fe_dimensions` under the default or
`selected_added_fe_pair` after an explicit selector.

### Returned dictionary

Coefficient and component objects:

- `b_base`, `b_full`: length-p X1 coefficient arrays;
- `b_full_status`: `estimated` or `imposed_zero` per X1 column;
- `gamma`: mapping from each observed block name to its full-model coefficient
  vector, in the same column order supplied for that block. Absorbed FE blocks
  have no finite-dimensional `gamma` entry. This backward-compatible field is
  not the auxiliary projection loading;
- `beta2`, `beta2_cov`: full-model X2 coefficients and their requested-VCE
  covariance in original X2 column order;
- `auxiliary_loadings`: the true auxiliary projection matrix `Gamma`, with
  rows `[x1 columns..., _cons]` and columns in original X2 order;
- `regularity[name]`: observed-block diagnostics including the block slices of
  `beta2`, `beta2_cov`, and `auxiliary_loadings`; loading signal/rank/condition
  summaries; the joint `beta2_g=0` Wald test; rowwise loading tests; product
  gradient norms; and per-row regularity flags/status;
- `delta[name]`: `coef`, `se`, and `se_type` arrays over
  `[x1 columns..., _cons]`, plus the observed-block regularity metadata (FE
  entries are explicitly not applicable);
- `total`: `coef`, `se`, `cov`, and `se_type` for total movement;
- `total_cov`: top-level alias of `total["cov"]`;
- `cov`: joint covariance of every component. Its order is group-major:
  `[group_1:x1..., group_1:_cons, group_2:x1..., group_2:_cons, ...]`;
- `base_cov`: requested-VCE covariance of `[b_base..., base intercept]`;
- `cov_delta_bbase`: cross-covariance between the group-major component vector
  and `[b_base..., base intercept]`;
- `cov_total_bbase`: cross-covariance between total movement and
  `[b_base..., base intercept]`;
- `fe_total`: aggregate FE `members`, `coef`, `se`, `cov`, and `se_type`, or
  `None` when there are no added FE groups. Common FEs are excluded.

Names and classification:

- `names`, `group_kinds`, `labels`, `x1_names`;
- `common_fe_names`, `n_common_fes`, and `common_fes_applied`;
- `focal_indices`, `focal_names`;
- `focal_status`, `absorbed_mask`, `absorbed_targets`,
  `absorbed_target_names`;
- `estimand`: `coefficient_movement` or `absorbed_target_allocation`;
- `identity_status`: `exact_ols`, `exact_ols_constrained`,
  or `exact_ols_conditional_common_fes`;
- `causal_interpretation`: always `False`.

Inference and diagnostics:

- `vce`, `gamma0`, `cov0`, `tol`, `df_base`, `df_full`, and `n_clusters`
  (`0` outside clustered inference);
- `total_se_type`, `inference_status`;
- `intercept_inference_available` and `intercept_status`;
- `regular_inference_valid`, `regular_inference_status`,
  `regular_inference_all_valid`, and `regularity_test_alpha`;
- `absorbed_target_inference_valid`, `absorbing_fe_index`;
- `x1_fe_collinear_ratio`, `fe_variance_status`,
  `fe_variance_ratio_min`, `x1_near_collinear_mask`,
  `fe_collinear_ss_ratio_tol`, `fe_collinear_relative_norm_tol`, and
  `near_fe_collinear_ss_ratio_warn_upper`;
- `few_cluster_warning_threshold`;
- `n_mobility_components`, `largest_mobility_component_n_obs`,
  `largest_mobility_component_share`, and
  `largest_mobility_component_weight_share`;
- `fe_split_identified`, `fe_split_status`, and
  `mobility_component_scope`;
- `connectivity_fe_index1`, `connectivity_fe_index2`,
  `connectivity_fe_indices`, `connectivity_fe_names`,
  `connectivity_pair_explicit`, `connectivity_pair_status`, and
  `connected_mode`;
- `threads_requested`, `threads_effective`,
  `recovery_threads_effective`, `threads_used`,
  `parallel_workers_active`, `thread_capacity`, `openmp_enabled`,
  `thread_limit_code`, and `thread_limit_reason`;
- phase diagnostics `fullfit_threads_used`,
  `fullfit_parallel_workers_active`, `recovery_threads_used`,
  `recovery_parallel_workers_active`, `covariance_threads_used`, and
  `covariance_parallel_workers_active`;
- `gpu_requested`, `gpu_attempted`, `gpu_used`, `gpu_backend`, `gpu_status`,
  `gpu_status_code`, `gpu_absorption_converged`, and
  `gpu_absorption_iterations`;
- `identity_gap`, `converged`, and `notes`;
- the four sample-count fields documented above, plus the opt-in sample
  provenance fields.

Always require `converged is True` and inspect `notes`. `identity_gap` is only a
summation consistency check; it does not certify each FE split, inference, or
causal interpretation.

The FE-collinearity classifier imposes the absorbed-target boundary at
`ratio <= 1e-9`. A focal with `1e-9 < ratio <= 1e-4` remains in the standard
estimand but is marked in `x1_near_collinear_mask` and emits a warning because
its component split and SEs may be numerically fragile. The
`XHDFE_GELBACH_NEAR_COLLINEAR_WARN=0` environment switch suppresses that
warning only; it does not change classification, coefficients, SEs, or any
rank guard.

For one-way clustered inference, `n_clusters` below
`few_cluster_warning_threshold` (30) produces a recorded and visible warning.
This is a caution flag, not an automatic finite-cluster correction or
wild-cluster bootstrap.

## Inference contract

For observed blocks, the covariance follows the random-design stacked-moment
variance of Gelbach's official `b1x2`. It includes uncertainty in the auxiliary
projections and is not the smaller variance conditional on the realised design.
With `common_fes`, this contract applies to slopes after partialling out those
FEs. The retained `(p+1)` public layout preserves cross-frontend compatibility,
but the `_cons` row and column are NaN, `intercept_inference_available=False`,
and `intercept_status="not_certified_common_fes"`.

Absorbed-FE component inference is `conditional_gamma0`: uncertainty from
estimating the absorbed effects is not fully included. A total combining
observed and FE blocks is therefore labelled `mixed_*_conditional_fe`. With no
common FEs and exactly two added FE dimensions,
`fe_split_status="identified_two_way"` certifies only the X1 rows when their
retained selected-pair graph has one component.
Multiple components return `normalization_dependent` and emit a
`RuntimeWarning`. `connected="require"` turns this into a fail-closed
precondition. Three or more added FE dimensions return
`not_certified_multiway` whether the selected pair is connected or
disconnected, because a pair diagnostic is not an exact multiway rank
certificate. Any common FE makes the added-FE split
`not_certified_with_common_fes`; the diagnostic pair remains informative but
does not certify a normalization of the larger common-plus-added FE system.
The FE intercept-row allocation is never certified by this flag. In all
uncertified cases, the aggregate `fe_total` and overall movement are the
preferred normalization-invariant objects.

This conditional FE variance has a separate validity gate. For each X1 row,
`fe_variance_status` is `conditional_only_between_fe_dominant` when added FEs
are present and `x1_fe_collinear_ratio <= fe_variance_ratio_min` (default
`0.35`); otherwise it is `valid_first_order`. A triggered row keeps every
point estimate, covariance and numerical SE unchanged, but FE-component and
mixed-total SE labels gain `_conditional_only_diagnostic`, tidy confidence
intervals become `diagnostic_only_between_fe_dominant`, and one warning routes
the user to the existing pairs bootstrap. This is a detection and reporting
gate, not nonconditional recovered-FE inference.

For an absorbed target, `total_j = b_base_j - 0` is the base-coefficient
estimator itself, so its target-target total variance equals the requested base
VCE. Inference for a target invariant within an absorbing FE must be clustered
at that FE dimension. Unadjusted, robust, or crossed clustering is retained for
descriptive accounting but emits a warning and sets
`absorbed_target_inference_valid=False`.

The severe-near-collinearity diagnostic can also emit `RuntimeWarning` while
leaving points unchanged. Such a warning means the within-block SE split is
sensitive; tightening `tol` does not select a unique correct split.

### Product-regularity gate

For observed block `g` and reported coefficient row `r`,
`delta_rg = Gamma_rg beta2_g`. Its first derivative is
`[beta2_g, Gamma_rg]`. If both parts of this gradient are zero, the usual
first-order delta approximation is nonregular (Gelbach 2016, footnote 14).

The shared core therefore:

1. tests `beta2_g=0` jointly with the requested VCE;
2. if that null is not rejected, tests the corresponding row
   `Gamma_rg=0` using marginal requested-VCE z tests with a within-row
   Bonferroni adjustment;
3. marks the contribution `regular_beta_nonzero` or
   `regular_loading_nonzero` only when one of those nulls is rejected at half
   the family-wise level, i.e. half of `regularity_test_alpha`; the public
   value `0.05` is the family-wise level for the union of the two component
   tests;
4. otherwise reports `nonregular_not_ruled_out` (or `not_certified` when the
   test cannot be evaluated).

Failure to reject is not proof that the true gradient is zero. It is a
conservative warning that regular first-order inference has not been
established. The numerical SE and interval are retained for diagnosis and
reproducibility, but `notes`, `RuntimeWarning`, tidy output and contrasts label
them `diagnostic_only_nonregular_not_ruled_out`. This gate covers observed X2
contributions. Recovered-FE components retain their separate
conditional-inference qualification.

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
    share_t_min=3.0,
)
```

This is post-processing only; no model is re-estimated. It returns a list of
dictionaries suitable for pandas, Polars or a CSV writer.

- `focal`: optional override of the result's reporting selector.
- `include_intercept`: add the implicit `_cons` row. With common FEs its
  estimates and inferential fields are NaN by the deliberate intercept
  boundary.
- `include_total`: add `total_movement`.
- `include_full`: add `full_model_residual` for X1 coefficients. Its SE is
  missing because that covariance is not in the public result contract.
- `conf_level`: normal-approximation confidence level strictly between 0 and 1.
- `share`: `None`, `"movement"`, `"base"`, or `"base_fixed"`.
- `share_tol`: nonnegative absolute denominator threshold. Shares at or below
  it are undefined and returned as NaN.
- `share_t_min`: minimum absolute denominator t-statistic for treating
  first-order delta-method share intervals as reliable; default `3.0`.

Every component row contains `coefficient`, `component`, `component_kind`,
`estimate`, `std_error`, `conf_low`, `conf_high`, `conf_level`, `se_type`,
`regular_inference_valid`, `regular_inference_status`, and
`confidence_interval_status`.
When shares are requested it also contains `share`, `share_std_error`,
`share_conf_low`, `share_conf_high`, `share_defined`, `share_denominator`,
`share_se_type`, `share_units`, `share_tol`, `share_t_min`,
`share_denominator_t`, and `share_interval_status`.

Share meanings:

- `movement`: `delta_g / sum_h(delta_h)`. Its SE uses the full joint component
  covariance and the delta method. The total share is one with SE zero.
- `base`: `delta_g / b_base`. Its full delta-method variance is
  `Var(delta_g)/b_base^2 + delta_g^2 Var(b_base)/b_base^4
  - 2 delta_g Cov(delta_g,b_base)/b_base^3`, using `base_cov` and
  `cov_delta_bbase`. The total row uses `cov_total_bbase`; rows are labelled
  `joint_base_covariance_delta_method`.
- `base_fixed`: the explicit descriptive convention that scales component SEs
  while holding the reported base coefficient fixed. It is labelled
  `fixed_base_denominator_scaling`, not full ratio inference.

Shares remain signed. Negative values and totals above 100 percent are never
truncated or renormalized. Contributions in original units remain primary.
If a selected denominator is nonfinite or no larger than `share_tol`,
`tidy()` emits one `RuntimeWarning` for that call and returns NaN share fields
for the affected coefficient. Base-denominator shares are defined for X1 rows,
not the optional `_cons` reporting row.

Defined shares are additionally diagnosed using
`share_denominator_t = abs(denominator) / se(denominator)`, with
`base_cov` for base shares and `total_cov` for movement shares. Values below
`share_t_min` retain the point share and numerical SE, but set
`share_interval_status="weak_denominator_delta_method_unreliable"`, suffix
`share_se_type` with `_weak_denominator_diagnostic_only`, and emit one warning
directing the user to `gelbach.bootstrap(method="pairs")`. Values at or above
the threshold are `valid_first_order`. `share_tol` retains its independent
absolute-denominator semantics.

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
`joint_covariance_including_conditional_fe`. If an included observed
contribution has not passed the product-regularity gate, the contrast warns
and exposes `regular_inference_valid=False`,
`regular_inference_status="contains_nonregular_not_ruled_out"`, and a
diagnostic-only confidence-interval status.

## Full-refit pairs bootstrap

```python
result = gelbach.bootstrap(
    y,
    x1,
    x2_groups=None,
    fes=None,
    *,
    method="pairs",
    bootstrap_cluster=None,
    reps=999,
    seed=0,
    conf_level=0.95,
    ci_method="percentile",
    min_valid_reps=None,
    store_draws=True,
    require_gpu_used=False,
    share_tol=1e-12,
    **decompose_kwargs,
)
```

This is the full-refit counterpart of the bootstrap surface reviewed in
PyFixest 0.50.1. `method="pairs"` samples observations with replacement.
`method="cluster_pairs"` samples the explicitly declared
`bootstrap_cluster` blocks. The bootstrap unit is never guessed from the
point-estimate `vce` or `cluster` arguments: inferential clustering and
resampling design are separate choices.

The point estimate retains every `decompose_kwargs` setting. Each bootstrap
replication resamples `y`, `x1`, every observed block, every common and added
FE identifier, and analytic weights together; it then runs the same public
`decompose()` path from scratch. The replicate VCE is unadjusted because only
the re-estimated point functional enters the empirical bootstrap
distribution. Base/full fitting, singleton removal, auxiliary projections,
FE recovery, connectivity diagnostics and the coefficient-movement identity
are therefore recomputed on every draw.

Python uses `SeedSequence` plus one independent PCG64 child stream per
replication. The result is invariant to NumPy's global RNG state and repeated
calls with the same seed are bitwise reproducible on the same estimator
build. RNG algorithms deliberately differ across frontends: Stata uses its
native sequential RNG and R uses L'Ecuyer-CMRG streams, so an equal integer
seed is reproducible within a frontend but does not imply identical resamples
across Python, R and Stata. Every attempted draw appears in
`result["bootstrap"]["ledger"]` with
status, reason, sampled/retained row counts, singleton count, identity gap,
GPU use and RNG spawn key; aggregated reasons are in `failure_counts`. The
command fails closed unless at least
`min_valid_reps` succeed; the default is 90 percent of `reps`.

The complete `result["bootstrap"]` metadata schema is `method`,
`resampling_unit`, `bootstrap_cluster_name`, `seed`, `rng`, `reps_requested`,
`reps_valid`, `reps_failed`, `min_valid_reps`, `conf_level`, `ci_method`,
`interval_status`, `share_tol`, `component_names`, `coefficient_names`,
`ledger`, `failure_counts`, `intervals`, `draws`, `draws_stored`, `point_vce`,
`replicate_vce`, `gpu_required`, and `gpu_used_all_valid`.

`ci_method` supports percentile and basic intervals. The `intervals` mapping
contains bootstrap SEs, lower/upper limits and valid counts under the exact
keys `delta`, `total`, `b_base`, `b_full`, `share_base`, `share_movement`,
`full_share_base`, and `total_share_base`, covering:

- every group contribution (`delta`) and total movement;
- base and full coefficients;
- component shares relative to the base coefficient;
- component shares relative to total movement;
- full/base and total-movement/base ratios.

Frequency-weight bootstrap is deliberately rejected. Resampling one
compressed row is not the same empirical distribution as resampling its
expanded copies; expand the data explicitly before bootstrapping. Analytic
weights remain attached to sampled observations. `require_gpu_used=True`
turns any requested-CUDA fallback in the point or a valid replication into a
failure instead of accepting silent CPU execution.

These intervals are resampling-based diagnostics. They do not automatically
solve a product-nonregularity, weak-identification, few-cluster or
normalization-identification problem; the existing status fields and warnings
remain authoritative.

## Tables and waterfall plots

```python
rows = gelbach.etable(
    result,
    panels="all",
    format="records",       # dataframe/df, markdown/md, latex/tex, html, gt
    type=None,              # PyFixest-style alias for format
    focal="target",
    keep="human|job",
    drop=None,
    exact_match=False,
    labels={"human": "Human capital"},
    include_other=True,
    digits=3,
    caption=None,
    notes=None,
    conf_level=0.95,
    interval="auto",
    share_tol=1e-12,
    share_t_min=3.0,
)

specification = gelbach.waterfall_data(
    result,
    focal="target",
    keep="human",
    drop=None,
    exact_match=False,
    labels={"human": "Human capital"},
    include_other=True,
    share_tol=1e-12,
)

figure, axis = gelbach.coefplot(
    result,
    focal="target",
    annotate_shares=True,
    title=None,
    figsize=None,
    keep="human",
    drop=None,
    exact_match=False,
    labels={"human": "Human capital"},
    notes=None,
    include_other=True,
    share_tol=1e-12,
    ax=None,
    save="gelbach-waterfall.png",
    show=False,
)
```

`etable()` reports levels, shares of the base coefficient (`share_base`, with
the PyFixest alias `share_full`) and shares of the movement
(`share_movement`, alias `share_explained`). It includes the base endpoint,
selected components, total movement and full endpoint. Bootstrap intervals
are selected automatically for a bootstrap result; `interval="normal"`
explicitly requests the analytic reporting contract. Records are
dependency-free. DataFrames need pandas, `gt` needs pandas plus
`great_tables`, and the markup formats return strings. Default table notes
retain the point estimator's diagnostic `result["notes"]` so reporting does
not silently drop warnings.

`keep` and `drop` accept regular expressions unless `exact_match=True`;
labels are applied after selection. In both `etable()` and
`waterfall_data()`, filtering hides no accounting mass by default: omitted
components are summed into `Other (filtered)`. In `etable()` the aggregate is
present in every panel, and its SE uses the summed sub-block of the joint
component covariance, including cross-component terms.
`include_other=False` restores the former filtered shape and warns that the
displayed rows no longer preserve the accounting identity.
`waterfall_data()` is dependency-free, and the residual step makes the plotted
path end exactly at the full coefficient.
`coefplot()` uses optional Matplotlib and stores the underlying waterfall
specification on `axis._xhdfe_gelbach_waterfall`.

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

Python accepts `gpu=True` as an opt-in request for CUDA in the full-model
FE-absorption phase. It does not promise that CUDA will be profitable or
available: always inspect `gpu_requested`, `gpu_attempted`, `gpu_used`,
`gpu_backend`, and `gpu_status`. In particular, `gpu_backend == "cuda"` is
reported only when CUDA was actually used; a hidden or unavailable device is
reported as CPU/non-used rather than silently labelled CUDA. All frontends
share the same estimator implementation and are validated for numerical
parity on their common feature surface.

Index conventions:

- Python input and output indices are zero-based;
- R accepts one-based input indices but returns zero-based metadata;
- Stata accepts variable names and returns zero-based metadata plus names.

## Deliberate boundaries

The current command does not implement:

- automatic restriction/refitting on the largest connected component;
- an exact multiway-FE rank certificate for per-dimension contributions;
- IV/2SLS Gelbach decompositions or LATE allocation;
- multiway clustering, wild-cluster bootstrap, BCa/studentized intervals or
  weak-inference confidence sets;
- a claim that the pairs bootstrap by itself cures cells that fail the
  product-regularity gate;
- nonconditional sampling uncertainty for recovered FE components; the
  `fe_variance_status` runtime gate detects a region where the retained
  conditional intervals are diagnostic, but does not estimate the omitted
  uncertainty;
- inference or a normalization-independent decomposition for the intercept
  when `common_fes` is nonempty;
- a connectivity certificate for separate added-FE contributions conditional
  on common FEs;
- split-panel jackknife/dynamic-panel corrections;
- kernel, MM-quantile, KHB/GLM, distributional, Oaxaca, or causal-mediation
  estimators.

High-dimensional FE common to both specifications belong in `common_fes`;
high-dimensional entries in `fes` are added full-model components. Explicit
low-dimensional indicators in X1 remain supported when their individual
coefficients are substantively desired, but they are not needed merely to
condition both specifications on a categorical effect.

## Further examples and validation

- `examples/gelbach_example.py`: standard focal/common-control/HDFE workflow;
- `examples/gelbach_absorbed_target.py`: imposed-zero target workflow;
- `tests/validation/VALIDATE_GELBACH.py`: b1x2, LSDV and absorbed-target oracles;
- `tests/validation/VALIDATE_GELBACH_FRONTENDS.py`: Python/Stata/R parity and gates executing
  the standard and absorbed-target examples in each of the three frontends;
- `tests/validation/VALIDATE_GELBACH_PYFIXEST_FEATURES.py`: iid/cluster resampling oracles,
  seed/ledger/fail-closed gates, interval arithmetic, tables and waterfall
  identity;
- `docs/certification/gelbach-empirical-coverage-20260720.md`: paper-by-paper
  coverage and scientific boundaries.

Reference: Gelbach, J. B. (2016), “When Do Covariates Matter? And Which Ones,
and How Much?”, *Journal of Labor Economics* 34(2), 509–543.
