# xhdfe Python help

Package documentation version: 2.24.1.20260816. Use `python -m xhdfe --version`
to inspect the installed package rather than relying on this static document.

`xhdfe` is the Python package wrapper around the v11 xhdfe C++ backend. It
exposes the same compiled estimator that older scripts imported as
`py_hdfe_v11`, while adding a package namespace, install metadata, and packaged
help.

The package is a proof of concept. Estimates, standard errors, and recovered
fixed effects should be validated for the user's research design, just as with
the Stata package.

## Installation

From the repository root:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e .
```

The build uses CMake and compiles the existing `py_hdfe_v11` extension into the
`xhdfe` package. CPU is the default backend.

Portable source installs do not require `-march=native`. Native tuning defaults
to off on Apple and Windows: Apple Silicon hosts can also build x86_64 Python
environments under Rosetta, and Windows binaries do not inherit the build
machine's instruction set. The explicit safe-path override is:

```bash
XHDFE_ENABLE_MARCH_NATIVE=OFF python -m pip install .
```

CUDA builds require the source (build from a clone or `git+` URL, not a
prebuilt wheel) and the NVIDIA toolkit. Set the architecture to the GPU's
compute capability (`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`;
for example, `9.0` maps to `90`) and request CUDA with environment variables:

```bash
XHDFE_ENABLE_CUDA=ON CMAKE_CUDA_ARCHITECTURES=90 python -m pip install .
```

Equivalent CMake definitions can be passed through `CMAKE_ARGS`.

### Windows native runtime loading

GNU/MinGW builds recursively bundle every detected non-system DLL dependency
beside the compiled extension. CPython changed its DLL search behaviour in 3.8;
on xhdfe's supported Python versions (3.9 and later), `xhdfe` registers that
exact installed package directory with `os.add_dll_directory()` before loading
the extension and retains the returned handle for the lifetime of the process.
A certified wheel therefore needs neither the build toolchain on `PATH` nor a
user-written `os.add_dll_directory(...)` workaround. The build fails closed if
the active toolchain cannot supply a matching binary. MSVC builds do not take
this MinGW-specific path. The recursive closure includes transitive dependencies
such as `libdl.dll` when required by the selected `libgomp`. The prebuilt
Windows asset in this release is for CPython 3.12 x86-64; other Python ABIs
require a source build and were not separately Windows-certified here.

## Imports

Preferred package import:

```python
import xhdfe

reg = xhdfe.HdfeRegressor()
```

The legacy import remains available after installation:

```python
import py_hdfe_v11

reg = py_hdfe_v11.HdfeRegressor()
```

Companion modules use the same compiled core:

```python
from xhdfe import akm, gelbach
```

For the complete Gelbach decomposition API, including absorbed targets, focal
reporting, signed shares and contrasts, run:

```bash
python -m xhdfe gelbach
xhdfe-help gelbach
```

## Minimal example

```python
import os
import numpy as np
import xhdfe

n = 2000
rng = np.random.default_rng(0)
y = rng.normal(size=n)
X = rng.normal(size=(n, 3))
firm_id = rng.integers(0, 200, size=n)
year_id = rng.integers(0, 20, size=n)

reg = xhdfe.HdfeRegressor(se_type="robust", tol=1e-8)
reg.fit(y, X, fes=[firm_id, year_id])

print(reg.coef_)
print(reg.summary())

# Optional: request CUDA after installing a CUDA-enabled build
os.environ["XHDFE_GPU_BACKEND"] = "cuda"
reg_gpu = xhdfe.HdfeRegressor(se_type="robust", tol=1e-8)
reg_gpu.fit(y, X, fes=[firm_id, year_id])
assert reg_gpu.gpu_used_ == 1
assert reg_gpu.gpu_status_code_ == 1
os.environ.pop("XHDFE_GPU_BACKEND", None)
```

## Optional formula interface

The dataframe/formula layer is optional. It delegates estimation to the same
compiled `HdfeRegressor`; the array API remains available and unchanged.
Install the extra from a source checkout with:

```bash
python -m pip install '.[formula]'
# If xhdfe is already installed from a release asset:
python -m pip install 'formulaic>=1.2.1,<2' 'pandas>=1.3'
```

The extra is imported lazily: importing `xhdfe`, `feols`, or
`prepare_formula` does not itself import Formulaic, pandas, or SciPy.

```python
import xhdfe

reg = xhdfe.feols(
    "wage ~ tenure + experience + C(education) | firm + year",
    data=d,
    se_type="cluster",
    clusters="firm",
)

print(reg.coef_names_)
print(reg.tidy())
```

`data` may be a dataframe or a mapping of column names to equal-length vectors.
Custom transforms must be supplied explicitly through a `context` mapping;
Formulaic never captures arbitrary caller-local names implicitly.

The supported operators use standard R/Formulaic semantics. The closest common
Stata factor-variable spellings are:

| Purpose | Formula interface | Closest Stata spelling |
|---|---|---|
| Categorical main effect | `C(g)` | `i.g` |
| Choose category 3 as reference | `C(g, Treatment(reference=3))` | `ib3.g` |
| Continuous product only | `x:z` | `c.x#c.z` |
| Main effects plus product | `x*z` | `c.x##c.z` |
| Category main effects plus explicit slope interactions | `C(g)*x` | `i.g##c.x` |
| Arithmetic square | `I(x**2)` | `c.x#c.x` |
| Remove the intercept | `0 + x` or `x - 1` | `noconstant` |
| Endogenous regressors and their instruments | `... \| d ~ z1 + z2` | `endogenous(d) instruments(z1 z2)` |

Use `C(...)` explicitly for categorical variables, especially numeric category
codes and pandas nullable strings. Object and pandas categorical columns are
normally encoded automatically, but numeric columns without `C(...)` are
continuous. Formulaic treatment coding uses the first level in its encoded
level order as the default reference when an intercept is present. Declare a
pandas `Categorical` order or specify `Treatment(reference=...)` when the
reference must be reproducible and explicit. With `0 + C(g)`, all category
columns are retained. For a non-syntactic column name, use Formulaic's quoted
lookup, for example `Q("industry code")` or `C(Q("industry code"))`. In this
first release, the data argument of `C(...)` must be a direct column lookup,
either bare or through `Q(...)`; create a derived category as a dataframe
column before using it in the formula. Names containing literal quote
characters are not supported through `Q(...)`.

The interface deliberately follows R/Formulaic rather than implementing a
second Stata parser. In particular, `C(g):x` with an intercept produces one
slope for every category, whereas standalone Stata `i.g#c.x` omits its base
category interaction. These are not merely different coefficient bases: their
dimensions and fitted values can differ. Use `x + C(g):x` for a common slope
plus category slope deviations (the analogue of `c.x i.g#c.x`), or `C(g)*x`
when category main effects are also required (the analogue of `i.g##c.x`),
holding the reference category fixed. Also, `x:x` and `x**2` are formula
operators and do not square `x`; use `I(x**2)` for arithmetic.

Only bare column names are accepted after `|` in this first interface:

```python
reg = xhdfe.feols("y ~ x1 + x2 | firm + year", data=d)
```

Those columns are encoded as identifiers and passed to the unchanged HDFE
absorber. They are never materialized as a dense Formulaic dummy matrix. Put a
high-cardinality effect such as `firm` after `|`; `C(firm)` on the regression
RHS intentionally creates explicit dummy columns and can require substantial
memory. FE interactions or transforms after `|` and the `.` shorthand are
rejected explicitly in this first version; list RHS columns or use the array
API for `group`, `individual`, or native heterogeneous-slope workflows.

### Instrumental variables / 2SLS

A third formula part requests 2SLS, using the same grammar as the R frontend:

```python
reg = xhdfe.feols("y ~ x1 | firm + year | d ~ z1 + z2", data=d)
```

Its left side lists the endogenous regressors and its right side the
**excluded** instruments; both sides accept several terms joined with `+`, and
both go through Formulaic, so transforms and interactions such as
`I(z1**2)` or `z1:z2` are available there too. The exogenous regressors are
added to the instrument matrix by the core, so they must not be repeated after
the second `~`. The fixed-effect part is optional, which makes `y ~ x1 | d ~ z1`
a fixed-effect-free 2SLS fit.

The endogenous columns are appended to the design after the exogenous ones, so
`coef_names_` reads exogenous terms first, then endogenous terms, then the
recovered intercept; `endogenous_index_` records their positions. Estimation is
the same compiled 2SLS path the array API reaches through `instruments` and
`endogenous_idx`, including its residual convention: coefficients are solved on
the instrumented design while inference uses residuals from the actual
regressors. Two spellings are rejected before estimation because they would
otherwise surface as a rank failure deep in the core: repeating an endogenous
regressor before the last `|`, and listing an exogenous regressor as an
instrument.

Numeric FE and cluster arrays follow the native nonnegative-ID contract: a
negative value such as pandas' `-1` missing-category sentinel is rejected, as
are non-finite numeric identifiers. String, categorical, datetime, and
other label columns are factorized without conversion through floating point.
Nonnegative `int64` identifiers, including values above `2**53`, also retain
their exact integer identity.

`weights` and `clusters` accept either arrays or dataframe column names. A
two-dimensional cluster array has shape `(n, q)`; a list/tuple of arrays is
interpreted as `q` separate length-`n` cluster vectors. Frequency weights use
`weights=...` together with `fweights=True`. The formula layer validates
positive integer values, checked cumulative `int64` totals, and exact
representability through the core's current float64 Python binding; it rejects
an integer that would be rounded in transit. Clustered inference still requires
the native constructor option `se_type="cluster"`:

```python
reg = xhdfe.feols(
    "y ~ x | firm + year",
    data=d,
    weights="sampling_weight",
    clusters=["firm", "region"],
    se_type="cluster",
)
```

Missing values fail closed by default across the response, regressors, fixed
effects, weights, and clusters. The first formula release supports only
`na_action="raise"`; clean, drop, or impute rows explicitly before estimation
so every input stays positionally aligned.

The returned object is a Python subclass of the native `HdfeRegressor`, so its
coefficients, standard errors, diagnostics, retained fixed effects, and methods
remain available. Formula metadata adds `formula_`, `coef_names_`, `fe_names_`,
`fe_levels_`, `cluster_levels_`, `cluster_names_`, `se_type_`, `var_labels_`,
`data_index_`, `estimation_index_`, `intercept_index_`, `used_fast_path_`, and,
for an IV formula, `endog_names_`, `instrument_names_`, and
`endogenous_index_` (all three empty for a formula without an IV part).
`cluster_names_` records the cluster variables (`cluster_levels_` only records
categorical levels), `se_type_` records the canonical standard-error family,
and `var_labels_` snapshots descriptive labels carried by the estimation
frame. The native core stores its intercept
last; `coef_names_` and `tidy()` use that same order. If a real regressor is
itself named `Intercept`, the native intercept is labelled `Intercept [xhdfe]`
to keep result keys unique.

Formula parsing and dataframe materialization necessarily add overhead. Simple
numeric lookups and numeric `:` interactions use a direct NumPy construction
path, promoting inputs to FP64 before interaction arithmetic; categories and
transforms use Formulaic. Numeric columns referenced directly or inside Python
transforms such as `I(x**2)`—on either side of `~`—are promoted before Formulaic
evaluation, including in a mixed formula such as `C(g) + x:z`. If one non-FP64
column is used both as `C(g)` and as a numeric regressor, create a separate FP64
column for the numeric role; the wrapper rejects that ambiguous mixed use rather
than risk collapsing large integer category IDs. For repeated fits of the exact
same response/design/FE snapshot, prepare it once:

```python
prepared = xhdfe.prepare_formula(
    "y ~ x1 + x2 | firm + year",
    data=d,
    num_threads=1,
)

first = prepared.fit()
second = prepared.fit()
```

`PreparedFormula` owns read-only array copies, so later dataframe mutations do
not alter the prepared design. When variables or samples change inside a tight
loop, construct arrays once and call `HdfeRegressor.fit(y, X, ...)` directly for
the lowest overhead.

## Constructor

```python
xhdfe.HdfeRegressor(
    se_type="unadjusted",
    tol=1e-8,
    max_iter=100000,
    check_interval=1,
    convergence="auto",
    fit_intercept=True,
    num_threads=0,
    default_threads=0,
    max_threads=0,
    min_parallel_rows=20000,
    target_rows_per_thread=500000,
    drop_singletons=True,
    retain_fes=False,
    symmetric_sweep=False,
    absorption_method="auto",
    jacobi_relaxation=0.0,
    level=95.0,
    keepsingletons=None,
    dofadjustments=None,
    groupvar=None,
    ssc_k_adj=None,
    ssc_k_fixef=None,
    ssc_k_exact=None,
    ssc_g_adj=None,
    ssc_g_df=None,
    ssc_t_df=None,
    tolerance_mode="reghdfe-comparable",
)
```

Main options:

- `se_type`: `unadjusted`, `homoskedastic`, `robust`, or `cluster`.
- `tol`: fixed-effect absorber convergence tolerance.
- `max_iter`: maximum absorber iterations.
- `convergence`: `auto`, `normchange`, `reghdfe`, or `both`.
- `fit_intercept`: append an intercept to the slope regression.
- `num_threads`: authoritative OpenMP request when positive; zero uses
  auto-threading. Positive requests bypass automatic row-count, FE-shape, and
  `max_threads` heuristics and are clamped only to the logical processors and
  OpenMP runtime limit visible to the process. A serial build fails loudly for
  requests above one.
- `drop_singletons`: drop singleton observations before estimation.
- `keepsingletons`: reghdfe-compatible override for `drop_singletons`.
- `retain_fes`: recover per-observation fixed-effect contributions.
- `absorption_method`: `auto`, `gauss-seidel`, `symmetric-gauss-seidel`,
  `jacobi`, `mlsmr`, `lsmr`, or `auto-mlsmr` with documented aliases.
- `tolerance_mode`: `reghdfe-comparable` (default), `xhdfe-fast`, or
  `strict-residual`.
- `dofadjustments`: `all`, `none`, `firstpair`, `pairwise`, `clusters`, and
  `continuous`, as a string or sequence.
- `ssc_*`: fixest-style small-sample correction controls.

## Fit

```python
reg.fit(
    y,
    X,
    fes=None,
    weights=None,
    clusters=None,
    instruments=None,
    endogenous_idx=[],
    group=None,
    individual=None,
    aggregation="mean",
    slopes=None,
)
```

Arguments:

- `y`: 1-D numeric array with `n` observations.
- `X`: 2-D numeric array with `n` rows. For large jobs, Fortran-contiguous
  arrays avoid a copy.
- `fes`: sequence of 1-D integer arrays, one per fixed-effect dimension.
- `weights`: optional 1-D weights.
- `clusters`: one cluster array, an `(n, q)` matrix, or a sequence of cluster
  arrays.
- `instruments` and `endogenous_idx`: 2SLS inputs. `endogenous_idx` uses
  zero-based column positions in `X`.

  With absorbed fixed effects, the reported intercept is the finite
  normalization `mean(y) - mean(X) @ beta`, under zero-mean absorbed
  contributions. It is normalization-dependent, not a structural IV
  parameter.
- `group`: group-level outcome identifier.
- `individual`: individual identifier for group/individual fixed effects.
- `aggregation`: `mean`, `avg`, `average`, or `sum`.
- `slopes`: heterogeneous absorbed slopes. Each entry can be
  `{"fe_index": j, "values": z, "include_intercept": True}`.
- `fweights`: set `True` to read `weights` as positive-integer frequency
  weights (literal row replication) instead of analytic weights. The flag
  applies to that call only.

### Input validation

Estimation inputs are validated before any work starts, and a violation
raises rather than propagating into the results:

- `y`, `X`, `instruments` and slope values must be finite. `NaN` or `Inf`
  raises an error naming the offending row (and column). Drop or impute
  missing data yourself — this library will not do it silently.
- Fixed-effect, cluster, group and individual identifiers must be
  non-negative. **`pandas.factorize()` returns `-1` for missing
  categories**, so factorized columns containing missing values are
  rejected. Previously a `-1` fixed effect was silently dropped and a `-1`
  cluster silently became an extra cluster, which changed the standard
  errors without any warning.
- `weights` must be finite and non-negative; with `fweights=True` they must
  additionally be positive integers representable as `int64`. Note that a
  weight of exactly `0` is accepted and drops the row from the fit without
  any diagnostic.

Under `fweights`, `nobs_` reports the weight total (as Stata's `e(N)` does),
`nobs_rows_` the number of input rows, and `sample_index_` indexes rows. The
weight total is carried through a double, so it is exact only up to `2**53`;
beyond that — a frequency total above about `9.0e15` — `nobs_` can be off by
one unit.

A fit reported as `converged_ == True` never returns a non-finite
coefficient, nor a non-finite standard error for a coefficient whose
inference is identified. Inference is legitimately unavailable — and is
therefore exempt from that contract — for omitted (collinear) terms, an
unidentified intercept, a saturated design, fewer than two clusters, and
non-positive residual degrees of freedom.

**Do not infer that a coefficient was dropped from the value of its
standard error.** A dropped regressor is reported as a zero coefficient
whose standard error is `NaN` on most paths but `0.0` on some degenerate
ones, and a zero coefficient with a small standard error is also a
perfectly ordinary estimate. Use the explicit flags instead:

- `omitted_` — boolean per coefficient;
- `omitted_reason_` — `0` kept, `1` collinear with the absorbed fixed
  effects, `2` other collinearity;
- `any_omitted_` — whether anything was dropped at all.

Finite inputs can still overflow their FP64 cross-products when the design is
scaled extremely (roughly `|X| >= 1e153`). Such a fit fails explicitly rather
than being misclassified as ordinary collinearity. Rescale `y` and/or `X` and
fit again.

Heterogeneous slope example:

```python
reg.fit(
    y,
    X,
    fes=[worker_id, year_id, firm_id, firm_id],
    clusters=[worker_id, firm_id],
    slopes=[
        {"fe_index": 2, "values": firm_seniority, "include_intercept": True},
        {"fe_index": 3, "values": firm_seniority_sq, "include_intercept": False},
    ],
)
```

## Group and individual fixed effects

`group` without `individual` collapses data by group and estimates on the
collapsed sample.

`group` with `individual` uses the group/individual absorber. The individual
identifier must also be present among `fes`. This mode does not support IV or
heterogeneous slopes. Fixed-effect recovery through `retain_fes` is not
available for the combined group/individual path.

## Fixed-effect recovery

Use `retain_fes=True` in the constructor:

```python
reg = xhdfe.HdfeRegressor(retain_fes=True)
reg.fit(y, X, fes=[firm_id, year_id])
effects = reg.fe_effects_
```

Recovery diagnostics:

- `fe_recovery_iterations_`
- `fe_recovery_max_delta_`
- `fe_recovery_converged_`

## Output attributes

After `fit`, the regressor exposes:

- Estimates: `coef_`, `stderr_`, `tvalues_`, `pvalues_`, `conf_int_`,
  `covariance_`, `residuals_`.
- Sample stats: `nobs_`, `nobs_full_`, `num_singletons_`, `sample_index_`.
- Degrees of freedom: `df_resid_`, `df_resid_unadj_`, `df_m_`, `df_a_`,
  `df_a_levels_`, `df_a_exact_`, `df_a_nested_`.
- Fit stats: `r2_`, `r2_within_`, `rss_`, `tss_`, `tss_within_`,
  `saturated_`, `num_iterations_`, `converged_`. `r2_` and `r2_within_` are
  `NaN` when the corresponding total sum of squares is zero (a constant
  outcome), matching Stata's `regress` and `reghdfe`.
- Absorption certificate: `abs_residual_` is the maximum absolute
  `||D' W v_tilde||_2`, `abs_residual_rel_` is the corresponding maximum
  scale-normalized normal-equation residual
  `||D' W v_tilde||_2 / (||D' W||_F ||v||_2)`, where `v` is the original
  pre-absorption right-hand side, and
  `precision_certified_` reports whether the explicit post-solve check meets
  its numerical limit. For combined `group`/`individual` absorption this
  certificate is authoritative: any continuation sweeps are counted in
  `num_iterations_`, and `converged_` is true only when
  `precision_certified_` is also true.

  When a fit stops with `precision_certified_` false, `fit` emits a
  `py_hdfe_v11.PrecisionWarning` (a `RuntimeWarning` subclass, so it can be
  filtered with `warnings.simplefilter`). On badly conditioned absorption
  graphs the stopping rule can be satisfied while the coefficients are still
  materially further from the exact within solution than the nominal
  tolerance suggests; inspect `abs_residual_rel_` and consider
  `tolerance_mode="strict-residual"` or a tighter `tol`. The warning changes
  no estimate.
- Fixed-effect stats: `fe_num_levels_`, `groupvar_`, `fe_effects_`.
- Cluster stats: `num_clusters_`, `cluster_counts_`,
  `cluster_combo_counts_`, `cluster_scale_`. With fewer than two clusters
  the point estimates are still returned but all inference
  (`stderr_`, `tvalues_`, `pvalues_`, `conf_int_`) is `NaN`, matching
  `regress`/`reghdfe`; earlier releases reported standard errors of order
  `1e-16`, which produced spurious significance.
- Runtime diagnostics: `threads_requested_`, `threads_effective_`,
  `threads_used_` (largest team observed doing real work),
  `parallel_workers_active_` (largest useful-worker count in one region),
  `thread_capacity_`, `openmp_enabled_`, `thread_limit_code_`,
  `thread_limit_reason_`, `absorption_method_used`,
  `gpu_attempted_`, `gpu_used_`, `gpu_status_code_`,
  `gpu_absorption_converged_`, `gpu_absorption_iterations_`.

`summary()` returns a formatted text table.

## Regression tables with maketables

Fitted results implement the plug-in format of
[maketables](https://github.com/py-econometrics/maketables), so a result can be
passed directly to `ETable` without registration or an adapter:

```python
import maketables as mt
import xhdfe

m1 = xhdfe.feols("y ~ x1 | firm", data=d, se_type="cluster", clusters="firm")
m2 = xhdfe.feols("y ~ x1 + x2 | firm + year", data=d, se_type="cluster",
                 clusters="firm")

print(mt.ETable([m1, m2], drop="Intercept").make(type="tex"))
```

maketables is **not** an xhdfe dependency. The integration consists only of
duck-typed `__maketables_*` attributes; xhdfe never imports maketables.

Absorbed fixed effects appear as indicator rows. These model statistics are
available through `model_stats=[...]`:

| Key | Meaning |
| --- | --- |
| `N` | observations used, after singleton dropping |
| `r2`, `r2_within` | overall and within R-squared |
| `rmse` | `sqrt(RSS/N)` |
| `n_clusters` | minimum cluster count used for multiway inference |
| `se_type` | `iid`, `robust`, or `by: firm+year` |
| `N_full` | observations before singletons were dropped |
| `n_singletons` | singleton observations dropped |
| `df_absorbed` | degrees of freedom absorbed by the fixed effects |

`N`, `r2`, and `r2_within` are shown by default. The within R-squared is
reported only when the fit absorbed a fixed effect. The coefficient table
exposes estimates, standard errors, t statistics, and p-values. Confidence
interval tokens are intentionally omitted because xhdfe supports arbitrary
confidence levels and must not label a non-95% interval as `ci95l`/`ci95u`.

The native intercept is retained in the coefficient vector under absorption,
so `drop="Intercept"` is the usual spelling for a table following the
`reghdfe` convention. Array-API fits also tabulate: coefficients use `b0`,
`b1`, ...; the dependent variable is `y`; and unnamed absorbed dimensions use
`fe1`, `fe2`, ... so their presence is not hidden.

If the estimation frame carries labels under
`df.attrs["variable_labels"]`—as frames read by `maketables.import_dta` do—xhdfe
captures a read-only snapshot at fit time for table row and column labels.

## GPU selection

GPU acceleration targets the fixed-effect absorber. CPU remains the reference
path.

Runtime backend selection uses:

```bash
XHDFE_GPU_BACKEND=cpu
XHDFE_GPU_BACKEND=cuda
XHDFE_GPU_BACKEND=metal
```

CPU is the package default. CUDA requires a CUDA-enabled build and a CUDA device.
Metal is currently reserved.

## Help commands

Print the packaged help:

```bash
python -m xhdfe
xhdfe-help
```

List or open companion topics:

```bash
xhdfe-help --topics
xhdfe-help gelbach
python -m xhdfe gelbach
```

Print the help resource path:

```bash
python -m xhdfe --path
```

Inside Python:

```python
import xhdfe

print(xhdfe.help_text())
print(xhdfe.help_text("gelbach"))
```
