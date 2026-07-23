# xhdfe Python help

Published package version: 2.20.0.20260723. Use `python -m xhdfe --version`
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

Portable source installs do not require `-march=native`. On Apple platforms,
native tuning defaults to off so Apple Silicon hosts can also build x86_64
Python environments under Rosetta. The explicit safe-path workaround is:

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
- `num_threads`: force a thread count when positive; zero uses auto-threading.
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
- `group`: group-level outcome identifier.
- `individual`: individual identifier for group/individual fixed effects.
- `aggregation`: `mean`, `avg`, `average`, or `sum`.
- `slopes`: heterogeneous absorbed slopes. Each entry can be
  `{"fe_index": j, "values": z, "include_intercept": True}`.

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
  `saturated_`, `num_iterations_`, `converged_`.
- Fixed-effect stats: `fe_num_levels_`, `groupvar_`, `fe_effects_`.
- Cluster stats: `num_clusters_`, `cluster_counts_`,
  `cluster_combo_counts_`, `cluster_scale_`.
- Runtime diagnostics: `threads_used_`, `absorption_method_used`,
  `gpu_attempted_`, `gpu_used_`, `gpu_status_code_`,
  `gpu_absorption_converged_`, `gpu_absorption_iterations_`.

`summary()` returns a formatted text table.

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
