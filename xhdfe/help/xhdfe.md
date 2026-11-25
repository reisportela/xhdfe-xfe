# xhdfe Python help

Package documentation version: 2.35.0.20261125. Use `python -m xhdfe --version`
to inspect the installed package rather than relying on this static document.

Version 2.32.0 adds the reduced-system tier, on by default for eligible CPU
fits (see "Reduced-system tier" below); `reduced_pcg=False` in the constructor
turns it off for that regressor, and `XHDFE_REDUCED_PCG=0` for the process.

Version 2.31.0 corrects Student-t inference under the 53-bit x87 precision of
Windows hosts, corrects weighted `reghdfe-comparable` fits on CPU and carries
the routing and performance changes listed in the release notes. Old
absorption caches are invalidated automatically.

Version 2.30.1 corrected an inherited CUDA Comparable covariance error for
ordinary models with three fixed effects, no weights and no saved effects.
The remaining projection is completed at the existing target and within the
total iteration budget, and GPU repair diagnostics include all attempted
work. The 2.30.1 release record documents the accepted runtime cost relative
to 2.29.1; Fast retains its separate approximation contract.

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

Open `python -m xhdfe akm` for the connected-set and AKM/KSS arguments,
`python -m xhdfe akm-inference` for the inference guide, or
`python -m xhdfe examples` for public-data examples.

## Minimal example

Use pandas to read the public Grunfeld investment panel from statsmodels.

```python
import pandas as pd
import xhdfe

url = "https://raw.githubusercontent.com/statsmodels/statsmodels/main/statsmodels/datasets/grunfeld/grunfeld.csv"
data = pd.read_csv(url)
firm = pd.factorize(data["firm"])[0]
reg = xhdfe.HdfeRegressor(se_type="cluster")
reg.fit(
    data["invest"].to_numpy(),
    data[["value", "capital"]].to_numpy(),
    fes=[firm, data["year"].to_numpy()],
    clusters=firm,
)
print(reg.summary())
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

`data` may be a pandas or Polars DataFrame, or a mapping of column names to
equal-length vectors. Pass prepared data; materialize a Polars LazyFrame before
calling the estimator. The numeric path reads columns as NumPy arrays.
Formulaic evaluates other formulas through pandas or Narwhals; categorical
Polars columns can be converted internally to pandas and require PyArrow.
Polars is optional: install `polars[pyarrow]>=1.0` when using that input format.
Polars nulls count as missing values.

Numeric formula columns are promoted to float64 before evaluating products,
powers, and other transforms, so integer arithmetic cannot overflow before the
model matrix is built. Category and identifier columns retain their original
types, and the input DataFrame is not modified. A non-float64 column used both
numerically and through `C(...)` requires a separate float64 column for its
numeric role, preserving the original category IDs.

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
their exact integer identity. Integer identifiers are passed to the absorber
as they are, while label columns must be factorized first; on large data,
storing a FE or cluster identifier as an integer column is the faster input.

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
    target_rows_per_thread=0,
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
    fe_tolerance=1e-6,
    fe_recovery_method="hybrid",
    stats_style="reghdfe",
    reduced_pcg=None,
    reduced_pcg_cuda=None,
    weighted_cpu_route=False,
    reduced_pcg_apply_fine=None,
    slope_solver=None,
    slope_scores=None,
    slope_csr_owner=None,
)
```

Main options:

- `se_type`: `unadjusted`, `homoskedastic`, `robust`, or `cluster`.
  Driscoll–Kraay covariance is available in Stata through `vce(dkraay ...)`.
- `tol`: fixed-effect absorber convergence tolerance.
- `max_iter`: maximum absorber iterations.
- `convergence`: `auto`, `normchange`, `reghdfe`, or `both`.
- `fit_intercept`: append an intercept to the slope regression.
- `num_threads`: authoritative OpenMP request when positive; zero uses
  auto-threading. Positive requests bypass automatic row-count, FE-shape, and
  `max_threads` heuristics and are clamped only to the logical processors and
  OpenMP runtime limit visible to the process. A serial build fails loudly for
  requests above one.
- `target_rows_per_thread`: rows per thread when auto-threading sizes the team.
  The default `0` is automatic: one thread per 25,000 rows when fixed effects
  are absorbed with `retain_fes=False`, within the problem-size cap (one thread
  below 100,000 rows, at most 8 below 250,000 and at most 16 otherwise), and
  one thread per 500,000 rows with `retain_fes=True` or without fixed effects.
  A positive value applies that fixed number of rows per thread on every path;
  `500000` restores the automatic rule of version 2.30.1. Setting
  `XHDFE_AUTO_ROWS_PER_THREAD` to a positive integer before the first fit
  replaces the 25,000 for the process. On Linux and macOS the automatic teams
  of one user's simultaneous fits share the machine: such a fit takes the
  smallest of its automatic team, the logical processors divided by the number
  of fits running and the processors not taken by the other fits' absorptions,
  never fewer threads than `target_rows_per_thread=500000` would give. A
  positive `num_threads` is never reduced. A lowered team reports
  `thread_limit_code_ == 6` and `thread_limit_reason_ == "auto_team_coordination"`.
  `XHDFE_AUTO_TEAM_COORDINATION=0` before the first fit turns this off.
- `drop_singletons`: drop singleton observations before estimation.
- `keepsingletons`: reghdfe-compatible override for `drop_singletons`.
- `retain_fes`: recover per-observation fixed-effect contributions.
- `absorption_method`: `auto`, `gauss-seidel`, `symmetric-gauss-seidel`,
  `jacobi`, `mlsmr`, `lsmr`, or `auto-mlsmr` with documented aliases.
  `reduced-pcg` is only reported (see "Reduced-system tier" below);
  requesting it raises an error. `krylov-pcg`
  (`xhdfe.AbsorptionMethod.krylov_pcg`, code 9 in Stata and R) is only
  reported, when the Krylov-PCG route is forced, for example by a mobility
  profile that suggests it; requesting it raises an error. In
  `absorption_method_used`, `gauss_seidel`, `symmetric_gauss_seidel` and
  `jacobi` (codes 1 to 3) name the sweep selected for the fit; they are also
  reported when a route without a code of its own served it: the exact
  projections of weighted designs whose effects have few levels, the exact sparse
  solve and the conjugate-gradient accelerator of small path-like two-way
  graphs, and the conjugate-gradient stage that can follow the GPU sweeps.
- `reduced_pcg`: the reduced-system tier for the fits of this regressor:
  `True` on, `False` off, `None` (default) the process default (see
  "Reduced-system tier" below). Any other value raises `TypeError`.
- `reduced_pcg_cuda`: the experimental CUDA variant of that tier for the fits
  of this regressor that run on CUDA: `True` requests it, `False` does not,
  `None` (default) the process default (see "Reduced-system tier" below). It
  has no effect on CPU fits. Any other value raises `TypeError`.
- `reduced_pcg_apply_fine`: the finer blocks of the segment pass of the tier's
  operator for the fits of this regressor: an integer from 1 to 8 (1 keeps the
  blocks of 2.34.2), `"auto"` (the automatic rule of the data), or `None`
  (default) the process value (see "Finer blocks of the segment pass" below).
  An integer is read through `__index__` (numpy integers are accepted; `bool`
  and `numpy.bool_` are not). Any other integer or string raises `ValueError`,
  any other type `TypeError`. It is the last argument, after
  `weighted_cpu_route`; the property of the same name reads it and sets it
  between fits, so that one regressor can fit under several requests.
- `tolerance_mode`: `reghdfe-comparable` (default), `xhdfe-fast`, or
  `strict-residual`. `strict-residual` treats a final absolute check as
  authoritative, may use additional iterations up to `max_iter`, and raises a
  non-convergence error if that check is not met. On the CPU the check is the
  largest absolute group mean of the absorbed outcome and of every absorbed
  regressor for sweeps, for fits with both `group` and `individual`, and for
  ordinary LSMR or MLSMR projections without saved effects, including cached projections. A failed
  LSMR or MLSMR absolute check can trigger one continuation in the same solver
  within the remaining `max_iter` budget. For the sparse direct and Krylov-PCG
  solvers, the solver's stopping rule and the relative certificate apply.
  An ordinary categorical CUDA fit checks the largest absolute group mean of
  its reported residuals, using the fit's weights. It checks the structural
  residual for IV and the reconstruction when the fit retains effects. The
  unweighted automatic Comparable-based Strict route can continue from its
  projected state. Other ordinary CUDA fits can run one device LSMR projection
  within the remaining `max_iter` budget if the absolute check fails.
  `weighted_cpu_route=True` applies the CPU checks to an eligible fit.
  Fits with both `group` and `individual` use the grouped-absorption checks.
  The check of the delivered residuals is absolute, in the units of the
  dependent variable. When the residuals are so large in those units that
  the tolerance is within a few units of roundoff of them (observed on one
  synthetic panel, with `tol=1e-8`, from residuals of about 2e7 to 4e7), the
  check may be unattainable in double precision: the fit can then use
  iterations up to `max_iter` and raise a non-convergence error, in some runs
  well before that limit, and close to that point the same fit can be
  accepted in one run and refused in the next. Rescale the dependent
  variable, raise `tol`, or use `reghdfe-comparable`, whose criteria are
  relative.
  Ordinary no-slope LSMR/MLSMR uses
  `min(effective tolerance, 1e-11)` in comparable/strict modes; fast mode and
  requested tolerances at or below `1e-11` are unchanged.
  For heterogeneous slopes, fast mode uses the historical aggregate residual
  check and omits the stronger block checks and block-driven refinement.
  Aggregate failures can still trigger refinement or refusal. The default
  comparable mode retains its block checks. `slope_block_checked_` reports
  whether they ran; block metrics are zero placeholders when it is false.
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
    fweights=False,
    iweights=False,
    pweights=False,
    weighted_cpu_route=None,
)
```

Arguments:

- `y`: 1-D numeric array with `n` observations.
- `X`: 2-D numeric array with `n` rows. For large jobs, Fortran-contiguous
  arrays avoid a copy.
- `fes`: sequence of 1-D identifier arrays, one per fixed-effect dimension.
- `weights`: optional 1-D weights.
- `clusters`: one cluster array, an `(n, q)` matrix, or a sequence of cluster
  arrays.
- `instruments` and `endogenous_idx`: 2SLS inputs. `endogenous_idx` uses
  zero-based column positions in `X`.

  Redundant instrument columns are accepted when their retained span identifies
  the endogenous regressors. An effective zero span or an underidentified model
  returns an error.

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
- `iweights`: importance weights. With unadjusted VCE, the reported
  observation count uses `floor(sum(weights))` and weights are unnormalised.
  Robust and clustered inference use the analytic-weight convention.
- `pweights`: probability weights. Use a robust or clustered VCE with
  `reg.fit()`. `feols(..., pweights=True)` defaults to robust inference
  without clusters and clustered inference with clusters.
- `weighted_cpu_route`: per-call override for the weighted CUDA route.
  `None` uses the constructor value.

Supply `weights` with any of the three weight flags and choose at most one
flag. The default weight convention is analytic.

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

`slope_solver="schur"` requests the reduced solver with a diagonal preconditioner
for eligible unweighted OLS with two ordinary effects and two slope dimensions
on the same identifiers, at least one with an intercept. `"network"` requests
the forest preconditioner and `"direct"` sparse direct factorisation, both CPU
only. The route requires the reduced-system tier to be on, automatic absorption
and convergence in `reghdfe-comparable` mode, and no retained effects or saved
group variable. Turn off the absorption cache, FE-structure cache and mobility
profile for these fits. CUDA also needs `reduced_pcg_cuda=True` or its environment
setting. Accepted fits report method code 9 and pass the slope certificate.

Set the option in the `HdfeRegressor` constructor or set its `slope_solver`
property between fits. Supply slope terms to the array `fit` method as above.
The `feols` formula interface accepts ordinary fixed-effect terms.
`"off"` disables the route and its helpers. `None` (the default)
checks the per-fit environment switches `XHDFE_EXPERIMENT_SLOPE_DIRECT=1`,
`XHDFE_EXPERIMENT_SLOPE_NETWORK=1`, then `XHDFE_EXPERIMENT_SLOPE_SCHUR=1`, and
otherwise leaves the solver off.

`slope_scores` and `slope_csr_owner` are optional booleans for two-way cluster
score reuse (when the cluster partitions match the ordinary effects) and
parallel sparse assembly. Both default to enabled when an active mode is chosen
explicitly. When the mode comes from the environment they follow
`XHDFE_EXPERIMENT_NETWORK_SCORES=1` and `XHDFE_EXPERIMENT_NETWORK_CSR_OWNER=1`
respectively.

After a fit, read `slope_solver_request_`, `slope_solver_source_`,
`slope_solver_used_`, `slope_solver_status_`, and `slope_solver_reason_`
for the request and outcome. `slope_solver_discarded_iters_` counts iterations
lost in a declined attempt. `slope_solver_scores_used_` identifies covariance
that consumed the scores and `slope_solver_csr_owner_used_` identifies successful
parallel assembly even when a later step declined. Set
`XHDFE_EXPERIMENT_SLOPE_TRACE=1` in the environment for the optional trace.

## Group and individual fixed effects

`group` without `individual` collapses data by group and estimates on the
collapsed sample.

`group` with `individual` uses the group/individual absorber. The individual
identifier must also be present among `fes`. This mode does not support IV or
heterogeneous slopes. Fixed-effect recovery through `retain_fes` is not
available for the combined group/individual path.

On CPU, `absorption_method="auto"` uses a joint matrix-free LSMR operator for
the standard and individual fixed effects; `absorption_method="lsmr"` selects
it explicitly. An automatic LSMR result that cannot pass the independent
certificate falls back to the previous certified sweep route, and
`absorption_method_used` reports which result was retained. On CUDA, `auto`
keeps the existing group/individual GPU solver and accepts it only when the
authoritative certificate passes. Explicit LSMR requests remain CPU-only.
Explicit `gauss-seidel` and
`symmetric-gauss-seidel` remain available; `jacobi`, `schwarz`, `mlsmr`, and
`auto-mlsmr` are unsupported for combined group/individual fits.

## Reduced-system tier

The reduced-system tier is on by default. Before the route chosen by
`absorption_method="auto"` runs, an ordinary fit with
`tolerance_mode="reghdfe-comparable"`, `convergence="auto"`, two to five fixed
effects and no CUDA or Metal request (`XHDFE_GPU_BACKEND`), and without
weights, instruments, slopes, `retain_fes=True`, `groupvar`, `group` or the
writing of a mobility profile, eliminates exactly the fixed effect with the
most levels and solves for the others by preconditioned conjugate gradients on
the CPU, with the fit's threads, in at most `min(max_iter, 1500)` iterations.
For a requested tolerance of at least `1.25e-12`, a column stops when its
reduced-system acceptance quantity is at most `1e-12`, confirmed after
recomputing the reduced residual. Existing exits for trivial or compatible
systems remain available. Smaller requested tolerances retain the former
stopping criterion. Passing this fixed stopping target does not replace the
independent final precision certificate. The result is published only if it
passes that certificate, including the requested tolerance, and the unchanged
acceptance trigger, with a verified `abs_residual_rel_` of at most `1e-11`. A
published fit reports `absorption_method_used` equal to
`xhdfe.AbsorptionMethod.reduced_pcg` (`summary()` prints `reduced-pcg`; code 8
in Stata and R), `reduced_pcg_status_ == 1` and the tier's iterations in
`num_iterations_` (plus those of a normal-equation correction, should one
follow); the CPU Auto retry does not run. The method is reported, never
requested. The design follows fereg 0.6.0 (Schmieder 2026); see
"Acknowledgements and references" below.

Otherwise the attempt declines: `reduced_pcg_status_ == 2`,
`reduced_pcg_reason_` names the reason (for example `stall`, `ceiling`,
`projected`, `memory`, `certificate` or `trigger`) and
`reduced_pcg_discarded_iterations_` counts its iterations. Its work is
discarded and the route runs from the original data with its own `max_iter`
budget, so the estimates, `num_iterations_`, `absorption_method_used` and the
thread diagnostics are those of the route, as with the tier off. The attempt
declines with reason `memory` when its working-memory estimate exceeds the
larger of 1 GiB and 16 bytes per observation for each of the outcome, the
regressors and the fixed effects, or when an allocation fails.

A fit outside the conditions above, or whose solver a matching mobility
profile forces, reports `reduced_pcg_status_ == 0` and the condition in
`reduced_pcg_reason_`: `off` when the tier is switched off, `not_reached` for
a path that never reaches it, and `cache` when another route's transform was
read from the absorption cache (a transform the tier published is read back
with status 1 and reason `none`). While the tier is on for a fit,
absorption-cache entries written with it off or by an earlier version are not
read, and while it is off, entries written with it on are not read. The
internal fits of `xhdfe.akm` and `xhdfe.gelbach` never enter the tier.

The constructor argument `reduced_pcg` switches the tier for the fits of one
regressor, so that fits with and without it can be compared in the same
process: `reduced_pcg=False` turns it off and `reduced_pcg=True` turns it on,
whatever the process default; `None` (the default) follows the process
default. The process default is the environment variable `XHDFE_REDUCED_PCG`,
read once per process (`0` turns the tier off for every regressor that leaves
`reduced_pcg=None`), and otherwise on. A fit with the tier switched off reports
`reduced_pcg_status_ == 0` and `reduced_pcg_reason_ == "off"`, and gives the
results of the tier switched off for the whole process. The formula interface
passes the argument through (`xhdfe.feols(..., reduced_pcg=False)`).
`XHDFE_REDUCED_PCG_TRACE=1` writes the tier's plan and decision to standard
error, and `2` adds the verified residual of each column of a published fit.

Finer blocks of the segment pass. Where the tier's kernel for the segment pass
of its operator runs (builds compiled with GCC; an absorbed effect whose rows of
the reduced system take at least 2 MiB), that pass can run over blocks finer
than those of the tier's other passes. The constructor argument
`reduced_pcg_apply_fine` sets the factor for the fits of one regressor: an
integer from 1 to 8 asks for that factor, which is then halved while the
private sums of the small effects would take more than 4 MiB and kept only if
its extra bytes fit the attempt's memory budget (otherwise the coarse blocks
run); `1` keeps the blocks of 2.34.2; `"auto"` starts from 8 and halves the
factor while the cells of the operator's segments per finer block stay below
16 384, then applies the same rules (a rule of the data only, never of the
threads). A factor above 1 changes the last bits of the fits the kernel takes,
within the precision contract. When an attempt that took finer blocks
declines because an allocation failed, the tier repeats it once on the coarse
blocks and publishes that second result, and the fit writes no absorption-cache
entry. `None` follows the process value: the environment variable
`XHDFE_REDUCED_PCG_APPLY_FINE`, read once per process (`auto` or an integer
from 0 to 8, `0` read as 1), otherwise the built-in value, `1`. The fit
reports the request in force (`reduced_pcg_fine_request_`, `"auto"` or `"1"`
to `"8"`), its source (`reduced_pcg_fine_source_`: `"request"`,
`"environment"` or `"built-in"`), the factor the attempt chose and its blocks
(`reduced_pcg_fine_used_`, `reduced_pcg_fine_parts_`; 0 when no attempt on the
CPU reached that choice), the applications of the operator through the kernel
(`reduced_pcg_hinted_applies_`), a failed reservation of the finer blocks
(`reduced_pcg_fine_alloc_failed_`), the repetition and the factor, blocks and
iterations of the discarded first attempt (`reduced_pcg_fine_retry_`,
`reduced_pcg_fine_retry_factor_`, `reduced_pcg_fine_retry_parts_`,
`reduced_pcg_fine_retry_iters_`), and whether the absorption came from
the absorption cache (`reduced_pcg_cache_hit_`). Read the factor and the
applications together: `reduced_pcg_fine_used_` and `reduced_pcg_fine_parts_`
say which blocks the attempt chose, not that the kernel ran on them;
`reduced_pcg_hinted_applies_` counts the applications of the operator that
went through the kernel, and it is 0 when the chosen blocks were applied in
none.

The tier has an experimental CUDA variant, off by default and the one
exception to the condition of no CUDA request above. A fit that runs on CUDA
(`XHDFE_GPU_BACKEND=cuda`), meets the other conditions, requests the variant
and uses no mobility profile, absorption cache or FE-structure cache attempts
the tier with the products of the reduced system computed on the GPU; the
recurrence, the stopping rule, the acceptance trigger and the certificate are
those described above, computed on the CPU. The constructor argument
`reduced_pcg_cuda` requests it for the fits of one regressor: `True` requests
it and `False` does not, whatever the process default; `None` (the default)
follows the process default, the environment variable `XHDFE_REDUCED_PCG_CUDA`
(`1` requests the variant; read once per process), and otherwise off. It has
no effect on CPU fits, and `reduced_pcg=False` (or `XHDFE_REDUCED_PCG=0` with
`reduced_pcg=None`) keeps both the tier and its CUDA variant off. The
regressor has no backend argument: each fit follows the value of
`XHDFE_GPU_BACKEND` at the time it runs, so one process can alternate CPU and
CUDA fits by setting `os.environ["XHDFE_GPU_BACKEND"]` between them. A
published fit reports `absorption_method_used` equal to
`xhdfe.AbsorptionMethod.reduced_pcg`, `reduced_pcg_status_ == 1`,
`gpu_used_ == 1` and `gpu_status_code_ == 1`; requesting CUDA alone does not
prove entry, so check all of them. A declined attempt runs the CUDA route that
the fit takes without the request. A CUDA fit without the request reports
`reduced_pcg_reason_ == "gpu"`; with the request, a fit kept out by a mobility
profile or a cache reports `"mobility_profile"`, `"absorption_cache"` or
`"fe_structure_cache"` (the first that applies, in that order; a file
`xhdfe_mobility_profile.txt` in the working directory is a mobility profile)
and runs that same route. An attempt that cannot obtain device memory, or
cannot page-lock the rows it exchanges with the GPU, declines with
`reduced_pcg_reason_` equal to `"memory"` or `"pinned"` and runs that same
route. Where that confirmed stopping rule is in force (a requested tolerance
of at least `1.25e-12` and `XHDFE_REDUCED_PCG_CHAT_STOP` not `0`), the product
used to recompute the reduced residual is computed on the GPU as well; the
stopping decisions stay on the CPU. `XHDFE_REDUCED_PCG_CUDA_REPLACEMENT=0`,
set in the environment before the first fit (read once per process), keeps
that product on the CPU. The acceptance trigger and the certificate, computed
on the CPU, are unchanged. The formula
interface passes the argument through
(`xhdfe.feols(..., reduced_pcg_cuda=True)`).

For comparison with the former stopping criterion, set
`XHDFE_REDUCED_PCG_CHAT_STOP=0` before starting Python. This process switch
restores that criterion; it does not switch the tier off or change its
acceptance trigger or final certificate. An unsuccessful attempt declines to
the unchanged route. The switch is read once per process; requested
tolerances below `1.25e-12` retain the former criterion with either setting.

## Known limitations

- CUDA checks the fixed-effect moments of each absorbed column in
  `reghdfe-comparable` and `strict-residual` for ordinary weighted fits
  with two to four effects and unweighted fits with four effects. A failed
  check can trigger one device LSMR projection within the iteration budget.
  For analytic, probability and importance weights, the Comparable
  correction covers up to 2,000,000 estimation rows.
  For eligible fits, `weighted_cpu_route=True`
  requests the CPU fit and reports `gpu_status_code_ == 7` (see "Weighted fits
  on CUDA" below). Weighted `group` fits with `individual` keep the GPU
  route and lack CUDA accuracy measurements for this contract.
  In the ordinary weighted Auto CPU path, the published Comparable
  projection may use the ordinary selector with four effects and
  `retain_fes=True`, or for just-identified IV with three effects.
- With `retain_fes=True` in Comparable mode, a fit can publish a
  separate certified within projection and recover its effects afterwards.
  The outcome reconstruction is then limited by the accuracy of FE recovery.
  Explicit-method, IV, heterogeneous-slope and grouped routes can publish
  the effect-capturing sweep, whose accuracy is design-dependent.
- On CUDA, `residuals_` has the accuracy of the CUDA fit (for example 8.4e-4
  relative for `xhdfe-fast` and 1.3e-6 for `reghdfe-comparable` on a
  path-like two-FE public fixture).
- Windows, near-perfect fits: near-perfect additive-cell fixtures returned
  exact coefficients and certified fits with relative covariance errors of
  2.65e-9, 1.74e-4 and 1.82e+2 at relative residual noise of 2^-12, 2^-20 and
  2^-30; the same fixtures on Linux returned 2.2e-16. The exact additive-cell
  projection needs extended-precision arithmetic, which Windows hosts run at
  53 bits.

- A fit that converges by the solver's stopping rule but does not pass the
  precision certificate raises an error and returns no estimates, in Python,
  R and Stata alike. Ill-conditioned specifications can still require a refusal.
  In `reghdfe-comparable`, heterogeneous slopes under an explicit
  `convergence="reghdfe"` can require continuation at tighter internal
  tolerances with the first certified stage returned
  (`slope_accuracy_retry_stages_` counts them); the default
  `convergence="auto"` checks the slope block. `xhdfe-fast` keeps its
  aggregate certificate without the Comparable block continuation.
- `aggregation="sum"` without an ordinary absorbed fixed effect: the
  constant is in the span of the membership columns only for a uniform team
  size or through a data-dependent combination of individual columns, which
  xhdfe decides by projecting the ones vector on the absorbed effects as one
  separate right-hand side. When
  the constant is not spanned the fit has no intercept and is reported as one:
  `model_has_constant_` is `False`, the trailing intercept entry of `coef_` is
  NaN, `tss_` is uncentered and the R-squared derives from it; coefficients,
  RSS and residuals are unchanged from earlier versions and equal those of the
  no-intercept model.
- CUDA with small, ill-conditioned `group`/`individual` designs (condition
  number above about 1e7): the device solver can leave residuals at the 1e-6
  relative level where the CPU direct projection is exact to rounding. On
  designs small enough for that projection (at most about 1,500 absorbed
  columns) the device result is compared with it and refused with an
  informative `RuntimeError` when a retained column deviates by more than ten
  times the forward tolerance of the mode; use the CPU backend for such
  designs.
- CUDA with `tolerance_mode="reghdfe-comparable"` on ill-conditioned ordinary
  designs: unweighted OLS without saved effects also checks the normal
  equations against the original regressors. A failed check receives one
  continuation at the same method and tolerance, within the remaining
  iteration budget. A persistent failure returns no estimates and does not
  substitute CPU estimates. The 2.29.0 validation includes successful CUDA
  fits on directors and the difficult 10M-row 3-FE benchmark, with bounded
  corrections. Other ill-conditioned specifications can still require a
  refusal. The opt-in accuracy-retry environment settings remain available,
  but cannot bypass this necessary check.
- Degrees of freedom of the individual block: one exact redundancy against an
  ordinary fixed effect is subtracted whenever the constant is provably in the
  span of the membership columns (`aggregation="mean"`, or `"sum"` with a
  uniform team size); other redundancies are not detected by default
  (`dofadjustments="exact"` computes the exact rank).

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
- Comparable CUDA `absorption_method="auto"` uses device LSMR for ordinary
  unweighted two-FE graphs identified by the existing low group-occupancy detector.
  It preserves ordinary degrees of freedom and the ordinary Krylov floor
  `min(effective tolerance, 1e-11)`, and reports `lsmr` as the method used.
  Fast, explicit methods/convergence, slopes and retained effects keep their
  established primary routing. A broken ordinary Comparable CG recurrence
  restarts with device LSMR within the remaining iteration budget.
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
  Ordinary no-slope LSMR/MLSMR additionally reports
  `krylov_internal_tolerance_`, `krylov_max_final_backward_error_`,
  `krylov_max_condition_`, and
  `krylov_max_condition_times_backward_error_`; these are deterministic
  maxima across right-hand sides, with `krylov_max_condition_` recording the
  maximum running condition estimate. They remain zero on non-Krylov paths.
  The distinct Krylov-PCG path is not covered and continues to report zeros;
  a PCG precision floor remains future work rather than part of this change.
  The reduced-system tier is a different conjugate-gradient solver with the
  stopping rule described under "Reduced-system tier"; it leaves these four
  fields at zero itself.

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

  With absorbed fixed effects the intercept is the constant recovered by
  the FE normalisation, `mean(y) - mean(X) @ beta`. When a one-way cluster
  nests an absorbed intercept dimension (every cluster is a union of its
  levels; a slope term counts only with `include_intercept=True`), the
  cluster sums of the residuals are exactly zero and the intercept's row and
  column of `covariance_` are exactly `m @ V @ m` and `-V @ m`, with
  `m = mean(X)` and `V` the slope block; this applies on every absorption
  route and backend, but not with `tolerance_mode="xhdfe-fast"`, instruments
  or Driscoll-Kraay errors. That variance is zero when `m = 0` or there are
  no regressors: the intercept's `stderr_` is then `0.0` and its `tvalues_`,
  `pvalues_` and `conf_int_` are `NaN`. This does not mean a structural
  parameter known without uncertainty; the constant is a normalisation of
  the absorbed effects. Saturated designs and fewer than two clusters keep
  the conventions above.
- Runtime diagnostics: `threads_requested_`, `threads_effective_`,
  `threads_used_` (largest team observed doing real work),
  `parallel_workers_active_` (largest useful-worker count in one region),
  `thread_capacity_`, `openmp_enabled_`, `thread_limit_code_`,
  `thread_limit_reason_`, `absorption_method_used`,
  `reduced_pcg_status_`, `reduced_pcg_reason_`,
  `reduced_pcg_discarded_iterations_` (see "Reduced-system tier"),
  `gpu_attempted_`, `gpu_used_`, `gpu_status_code_` (0 not requested, 1 used,
  2 unavailable, 3 not converged, 4 failed, 5 CPU cache/profile result, 7
  weighted fit run on the CPU by design: see "Weighted fits on CUDA"),
  `gpu_absorption_converged_`, `gpu_absorption_iterations_`,
  `cuda_tail_code_`, `cuda_tail_`.

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

`gpu_used_ == 1` proves GPU execution, not universal numerical certification.
The 2.29.0 core24 validation includes the four CUDA Comparable cells that
2.28.1 refused. Separate extreme-offset, heterogeneous-slope and group/individual
CUDA cases retain documented refusals; use CPU for those specifications.
Consult the versioned validation record for the exact tested artifacts and
scope. Python callers must check
`gpu_used_` and `gpu_status_code_`: an unavailable CUDA backend can return a
CPU result. Fast and Comparable remain distinct approximation contracts.

### Weighted fits on CUDA

Set `weighted_cpu_route=True` in `xhdfe.feols()`, `HdfeRegressor()` or
`reg.fit()` to request CPU for an eligible weighted CUDA fit. The
regressor's default is `False`. In `reg.fit()`, `None` uses the
regressor's setting and a Boolean overrides it for that call only.

With the option enabled, a fit with weights (frequency, analytic, probability or importance; all ones
and zero weights included) and two or more absorbed effect dimensions, in
`reghdfe-comparable` or `strict-residual`, whose effective backend is CUDA
(`XHDFE_GPU_BACKEND=cuda`, or a build whose default backend is CUDA), is
absorbed on the CPU and says so: `gpu_used_` is 0, `gpu_attempted_` is 0, and
`gpu_status_code_` is 7 (`weighted_cpu_route`). On CUDA such fits could pass
the certificate and still lie outside the comparable contract against
`reghdfe` at tolerance 1e-12 (public 240,000-row fixture: coefficients up to
4.5e-7 and standard errors up to 2.8e-5 relative). The dimensions are counted
as `fit` receives them, before redundant ones are removed: `fes=[a, a]` is two.
A weighted fit with `group` and without `individual` is the collapsed weighted
fit and is routed too; its CPU fit runs, as before, without the routing retry
and without the saved-effects projection (the `group` entry turns both off).
With `individual` a fit keeps the GPU route and was not measured against the
contract on CUDA; `xhdfe-fast`, unweighted fits and fits with one absorbed
effect keep the GPU route. A routed fit takes the time of
the CPU backend. Its results other than the GPU status code are bit for bit
those of the same fit with `XHDFE_GPU_BACKEND=cpu`, with the same module,
threads and initial cache and profile state (`reduced_pcg_reason_` included),
wherever that CPU fit repeats itself (an automatic fit that benchmarks methods
by time while writing a mobility profile from empty does not); the identity is
stated for builds whose default backend is the CPU (all released ones). It is
an identity with the CPU backend, not a guarantee against `reghdfe`, and the
route is not a general correction (see "Known limitations"). A routed
`strict-residual` fit applies the CPU checks described under `tolerance_mode`.
For ordinary weighted CPU fits with three fixed effects and without saved effects,
explicit `absorption_method="gauss-seidel"` in Comparable mode checks each column's
fixed-effect moments and can perform one Gauss–Seidel block correction within `max_iter`.
An explicit `absorption_method="schwarz"`
with weights gets the CPU's answer, which for an ordinary fit is a refusal. A
routed fit is the CPU fit also for the optional absorption cache and mobility
profile, so in a process that writes profiles a later fit may read what a
routed fit wrote, as it reads what a CPU fit wrote. With `weighted_cpu_route=True`, a weighted Gelbach
decomposition reports the status of the full-model fit it reports
(`gpu_status == "weighted_cpu_route"`), except when the fixed-effect recovery
falls back to the full refit (`XHDFE_GELBACH_WARM_RECOVERY=0`, or a warm start
that is not certified): the status is then that refit's, made after the CUDA
request has ended (code 0 when nothing else requests CUDA), as for an
unweighted model. A bootstrap using `require_gpu_used=True` refuses a fit
whose actual backend is CPU.

This routing applies to model fits. Stata's `xfepout` and R's
`xhdfe_demean()` keep their CUDA partial-out route with weights.

For unweighted Comparable fits with three fixed effects, the CUDA path
completes the projection after its sweeps. This completion normally runs on
the GPU. When a short GPU probe predicts a long GPU solve (graphs that mix
slowly) and the CPU would itself absorb the fit with MLSMR, the CPU finishes it
with MLSMR from the state the GPU sweeps reached, at the same tolerance, within
the same iteration budget and under the same precision certificate, using the
fit's thread team. `cuda_tail_` names it: `device-lsmr`, `host-mlsmr` (then
`gpu_used_` stays 1 and `absorption_method_used` is MLSMR), or `none` when no
completion ran; `cuda_tail_code_` is 0, 1 or 2. At tight tolerances the GPU
sweeps can instead hand over to a conjugate-gradient stage; when that stage
stops without converging, the fit restarts from the original data with GPU
LSMR. The same probe and the same CPU decision apply to that restart: when both
point to the CPU, the CPU solves the fit with MLSMR from the original data
(never from the conjugate-gradient state) and `cuda_tail_` is `host-mlsmr`; a
restart kept on the GPU reports `none` (2.30.1 behaviour), so `none` also
covers that GPU restart (the trace below names it `restart->device-lsmr`).
`num_iterations_` and `gpu_absorption_iterations_` count all of it: the GPU
sweeps (including those before a conjugate-gradient stage that is then
restarted), the conjugate-gradient iterations, the GPU solver and the CPU
iterations, all charged to one `max_iter` budget that is never reset: the
conjugate-gradient stage only receives what the sweeps left, and a fit whose
sweeps and conjugate-gradient stage use the whole budget raises with no
estimates (`iterations=max_iter/max_iter`). One exception, on the CPU and on
the GPU alike: when the conjugate-gradient stage converges (no restart),
`num_iterations_` counts its iterations but not the sweeps before it. The
environment variable `XHDFE_CUDA_TAIL_HANDOVER=0` keeps the completion and the
restart on the GPU (2.30.1 behaviour, including its budget and iteration count:
the sweeps before a conjugate-gradient stage are then neither charged to its
budget nor counted). For diagnosis,
`XHDFE_GI_LSMR_TRACE=1` writes the decision to standard error: one
`cuda_tail_probe` line per checkpoint (predicted remaining GPU work, CPU route
selector), one `cuda_tail_handover` line when the CPU completes (iterations,
seconds), one `cuda_it_to_cg` and one `cuda_cg_end` line around a
conjugate-gradient stage, and one `cuda_tail_restart` line per restart
(iterations and seconds of every stage, GPU and CPU apart). The same variable
traces the LSMR solves of group/individual absorptions. The decision uses
solver quantities only, never elapsed time.

## Mobility profiles

`XHDFE_MOBILITY_PROFILE` selects a profile path and
`XHDFE_MOBILITY_MODE=off|read|write|auto` controls its use. Version-2 profiles
distinguish `standard` from `group_individual` fits. Group/individual profiles
require two domain-separated signatures covering the post-singleton operator,
incidence and aggregation scale, weights and frequency-weight semantics, RHS
count, threads, tolerance, backend, and solver-selection context. Legacy
version-1 profiles, scope mismatches, and signature mismatches are safe misses.
Writing a profile (`write`, or `auto` without a matching profile) keeps the fit
out of the reduced-system tier, as does a matching profile that suggests a
Krylov or sparse solver.

A profile is a method-selection hint, not a transformed-data cache. In
group/individual Auto it cannot promote Gauss-Seidel or symmetric Gauss-Seidel:
CPU keeps joint LSMR and CUDA keeps the grouped GPU solver. Explicit GS/SGS
requests remain available. Profile write/auto does not benchmark sweep methods
and writes only after the requested route converges and certifies. It suggests
LSMR only after safe CPU LSMR and otherwise records Auto, so an explicit GS/SGS
diagnostic run cannot promote that method later. CPU-only LSMR hints are not
forced on CUDA runs.

The CPU absorption cache uses `XHDFE_ABSORPTION_CACHE` for the file path and
`XHDFE_ABSORPTION_CACHE_MODE` for the mode. It stores transformed columns
and checks the sample, input values, weights and solver settings before reuse.
The FE-structure cache uses `XHDFE_FE_STRUCTURE_CACHE` and
`XHDFE_FE_STRUCTURE_MODE` for its own file and mode. It stores the
identifiers and singleton selection for a matching FE structure. Both modes
accept `off`, `read`, `write` and `auto`. `read` leaves the file
untouched. With `auto`, the fit can refresh an invalid or outdated entry.
Use distinct files when combining the two caches. The combined absorption
key uses the original FE identifiers and singleton policy. Use `auto` or
`write` to refresh an older combined absorption entry.

For combined group/individual CUDA fits, `xhdfe-fast` uses an internal
`min(tol, 1e-12)` floor before certification. Fast mode therefore cannot
re-enable the materially inaccurate early GS stop fixed in version 2.26.0.

For group/individual fits, `precision_certified` requires the canonical
preconditioned-operator check for `A = W^(1/2) D C`: weighted residuals,
moments `C D' W r`, and a Frobenius norm over active normalized columns. The
consistent-system residual ratio or the least-squares optimality ratio must
pass for every right-hand side. The public residual diagnostics retain their
established scale for compatibility.

## Acknowledgements and references

`xhdfe` follows the estimator, defaults and reporting of `reghdfe` (Sergio
Correia) and builds on the work of other researchers:

- `fixest` (Laurent Bergé and contributors): accelerated alternating
  projections and small-sample corrections.
- `pyfixest` (Alexander Fischer and contributors; https://pyfixest.org/,
  https://github.com/py-econometrics/pyfixest): code and documentation
  studied during development, and interface, numerical and benchmark
  references.
- `FixedEffectModels.jl` (Matthieu Gomez and contributors): diagonally
  preconditioned LSMR.
- `within` (Alexander Fischer and Kristof Schröder;
  https://github.com/py-econometrics/within) and Fischer and Schröder (2026),
  "Graph Preconditioning for High-Dimensional Fixed Effects Regression",
  mimeo, draft of June 2026 (https://github.com/py-econometrics/within-paper):
  the direct starting point of the graph-preconditioned MLSMR/additive-Schwarz
  route.
- `fereg` (Johannes F. Schmieder; https://github.com/johannes-schmieder/fereg)
  and Schmieder (2026), "fereg: Fast high-dimensional fixed-effects
  regressions in Stata", CEPR Discussion Paper No. 22170, CEPR Press, Paris &
  London (https://cepr.org/publications/dp22170). The reduced-system tier of
  version 2.32 follows the design implemented in fereg 0.6.0 and described
  there, and cites it as such: exact elimination of the effect with the most
  levels and Jacobi-preconditioned conjugate gradients on the remaining
  system, for two to five fixed effects, with up to eight columns per pass, a
  limit of 1,500 iterations for that stage and a stall test of the same form
  (the residual must halve within each window of iterations). As that paper
  notes, the elimination is the block reduction of Somaini and Wolak (2016),
  and iterative solutions have precedents in Abowd, Creecy and Kramarz (2002)
  and in Mittag (2019). `xhdfe` does not implement what Schmieder (2026)
  presents as fereg's distinguishing contribution in this step, the escalation
  to stronger preconditioners, including combinatorial multigrid, on poorly
  connected data: where the Jacobi-preconditioned iteration stalls, `xhdfe`
  hands the fit to its previous routes. `xhdfe`'s part is an independent C++
  implementation inside its own routing, certificate, fallbacks and
  diagnostics. `xhdfe` contains no fereg source code; its C++ implementation
  was written after studying the paper and fereg's public source. Two
  changes of version 2.31.0 followed benchmarks against fereg: reading the
  Stata data in parallel on qualified hosts, as fereg does, and larger
  automatic thread teams (one thread per 25,000 rows instead of 500,000),
  where fereg by default uses all logical processors and sizes the parallel
  work of each stage by its amount of work. fereg
  credits the precedent of parallel copying from Stata to David Roodman's
  `julia` package (Roodman 2025).

Comparison packages, current releases checked on 14 November 2026: `reghdfe`
6.14.1 (8 July 2026), `fixest` 0.14.2 (CRAN, 26 June 2026), `pyfixest`
0.70.0, `FixedEffectModels.jl` 2.2.0 (2 November 2026) and fereg 0.6.0.

References cited above:

- Abowd, John M., Robert H. Creecy, and Francis Kramarz. 2002. "Computing
  person and firm effects using linked longitudinal employer-employee data."
  Longitudinal Employer-Household Dynamics Technical Paper TP-2002-06, U.S.
  Census Bureau.
- Fischer, Alexander, and Kristof Schröder. 2026. "Graph Preconditioning for
  High-Dimensional Fixed Effects Regression." Mimeo, draft of June 2026.
  https://github.com/py-econometrics/within-paper
- Mittag, Nikolas. 2019. "A simple method to estimate large fixed effects
  models applied to wage determinants." Labour Economics 61: 101766.
  doi:10.1016/j.labeco.2019.101766
- Roodman, David. 2025. "Julia as a universal platform for statistical
  software development." Stata Journal 25(2): 255-284.
  doi:10.1177/1536867X251341105
- Schmieder, Johannes. 2026. "fereg: Fast high-dimensional fixed-effects
  regressions in Stata." CEPR Discussion Paper No. 22170. CEPR Press, Paris &
  London. https://cepr.org/publications/dp22170
- Somaini, Paulo, and Frank A. Wolak. 2016. "An algorithm to estimate the
  two-way fixed effects model." Journal of Econometric Methods 5(1): 143-152.
  doi:10.1515/jem-2014-0008

## Help commands

Print the packaged help:

```bash
python -m xhdfe
xhdfe-help
```

List or open companion topics:

```bash
xhdfe-help --topics
xhdfe-help examples
xhdfe-help akm
xhdfe-help akm-inference
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
