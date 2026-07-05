# xhdfe quick start

`xhdfe` estimates linear models with multiple high-dimensional fixed effects.
The Stata command, the Python package, and the R package are three front ends to
**one compiled C++ core**, so the estimator, defaults, and numerical results are
the same in all three languages. CPU is the reference backend; an optional CUDA
GPU absorber can accelerate the fixed-effect step.

By default `xhdfe` runs in `reghdfe-comparable` tolerance mode, so at the same
nominal tolerance its coefficients match
[reghdfe](https://github.com/sergiocorreia/reghdfe) (Correia 2016). It also
exposes `fixest`/`pyfixest`-style small-sample corrections. See
[comparison.md](comparison.md) for a command/option map across the ecosystem.

> **Platform.** Source installs target Linux, Windows with Rtools/MinGW, and
> macOS Apple Silicon/Intel where a C++17 toolchain is available. CUDA builds
> are currently Linux-only and require the NVIDIA toolkit.

---

## Install (recap)

The repository ships source. Prebuilt Stata plugins and distribution ZIPs are
attached to GitHub **Releases**, not stored in the tree. See the top-level
`README.md` for full instructions; in brief:

**Stata.** Install online from the release net-install site:

```stata
net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
net install xfe,   from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
```

The online package uses Stata platform-specific `g` lines to select the plugin
for the user's OS. You can also download a Release and `net install` from the
folder that contains `xhdfe.pkg` and `stata.toc`, or point Stata at the
repository's `stata/` folder:

```stata
net install xhdfe, from("/path/to/xhdfe/stata") replace
net install xfe,   from("/path/to/xhdfe/stata") replace
```

The online plugins and release ZIPs are **CPU-only**. Building from source is
**required for NVIDIA GPU (CUDA)** — the online material ships no source. Clone
the repo, then:

```bash
# CPU (OpenMP recommended)
bash stata/tools/build-plugin.sh --linux --openmp     # produces stata/xhdfe.plugin

# GPU (Linux + NVIDIA only): set XHDFE_CUDA_ARCH to your GPU's compute capability
#   nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. 9.0 -> 90, 8.6 -> 86
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash stata/tools/build-plugin.sh --linux --openmp
```

**Python** (requires Python >= 3.9), from the repository root:

```bash
python -m pip install .
# CUDA build (optional):
XHDFE_ENABLE_CUDA=ON CMAKE_CUDA_ARCHITECTURES=90 python -m pip install .
```

**R** (Linux x86-64 + GCC):

```r
remotes::install_github("reisportela/xhdfe-xfe", subdir = "r/xhdfe")
```

or, from a clone, `R CMD INSTALL r/xhdfe` (CUDA:
`XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 R CMD INSTALL r/xhdfe`).

---

## Cookbook

The recipes below show the **same eight estimation tasks** in Stata, Python, and
R. Each column uses that language's natural public/synthetic data — Stata's
built-in `nlswork` worker panel, and a small synthetic worker–firm panel in
Python and R — but performs the same conceptual estimation, so you can read
across languages.

**Setup — Stata** (public dataset shipped with Stata):

```stata
webuse nlswork, clear
```

`nlswork` is a worker panel with `idcode` (worker), `year`, `occ_code`,
`ind_code`, outcome `ln_wage`, and regressors `grade`, `ttl_exp`, `tenure`,
`union`, `hours`, `south`, ...

**Setup — Python** (synthetic):

```python
import numpy as np
import xhdfe

rng = np.random.default_rng(2026)
n = 5000
firm_id = rng.integers(0, 300, size=n)          # first FE dimension
year_id = rng.integers(0, 8,   size=n)          # second FE dimension
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
z  = rng.normal(size=n)                          # instrument
xe = 0.6 * z + 0.3 * x1 + rng.normal(size=n)     # endogenous regressor
w  = rng.uniform(0.5, 1.5, size=n)               # analytic weights
y  = 1 + 0.5 * x1 - 0.2 * x2 + 0.7 * xe + rng.normal(size=n)

X = np.column_stack([x1, x2])                    # design matrix (n x k)
```

**Setup — R** (synthetic):

```r
library(xhdfe)

set.seed(2026)
n <- 5000
d <- data.frame(
  firm = sample(300, n, replace = TRUE),
  year = sample(2015:2022, n, replace = TRUE),
  x1   = rnorm(n),
  x2   = rnorm(n),
  z    = rnorm(n),                       # instrument
  w    = runif(n, 0.5, 1.5)             # analytic weights
)
d$xe <- 0.6 * d$z + 0.3 * d$x1 + rnorm(n)  # endogenous regressor
d$y  <- 1 + 0.5 * d$x1 - 0.2 * d$x2 + 0.7 * d$xe + rnorm(n)
```

### (a) Basic two-way fixed-effect regression

**Stata**

```stata
xhdfe ln_wage grade ttl_exp tenure, absorb(idcode year)

* CUDA build only: request GPU absorption and verify it was used
xhdfe ln_wage grade ttl_exp tenure, absorb(idcode year) gpubackend(cuda)
display e(gpu_used)
display "`e(gpu_backend)'"
```

**Python**

```python
reg = xhdfe.HdfeRegressor()
reg.fit(y, X, fes=[firm_id, year_id])
print(reg.summary())

# CUDA build only: request GPU absorption and verify it was used
import os
os.environ["XHDFE_GPU_BACKEND"] = "cuda"
reg_gpu = xhdfe.HdfeRegressor()
reg_gpu.fit(y, X, fes=[firm_id, year_id])
assert reg_gpu.gpu_used_ == 1
assert reg_gpu.gpu_status_code_ == 1
os.environ.pop("XHDFE_GPU_BACKEND", None)
```

**R**

```r
m <- xhdfe(y ~ x1 + x2 | firm + year, data = d)
summary(m)

# CUDA build only: request GPU absorption and verify it was used
m_gpu <- xhdfe(y ~ x1 + x2 | firm + year, data = d, backend = "cuda")
stopifnot(m_gpu$gpu_used == 1, m_gpu$gpu_status == "used")
```

### (b) Clustered standard errors (one-way and multiway)

**Stata**

```stata
* one-way cluster on worker
xhdfe ln_wage grade ttl_exp tenure, absorb(idcode year) vce(cluster idcode)

* multiway cluster on worker and occupation (Cameron-Gelbach-Miller)
xhdfe ln_wage grade ttl_exp tenure, absorb(idcode year) vce(cluster idcode occ_code)
```

**Python**

```python
# one-way
reg = xhdfe.HdfeRegressor(se_type="cluster")
reg.fit(y, X, fes=[firm_id, year_id], clusters=[firm_id])

# multiway
reg = xhdfe.HdfeRegressor(se_type="cluster")
reg.fit(y, X, fes=[firm_id, year_id], clusters=[firm_id, year_id])
```

**R**

```r
# one-way (passing `cluster` implies vcov = "cluster")
m <- xhdfe(y ~ x1 + x2 | firm + year, data = d, cluster = ~ firm)

# multiway
m <- xhdfe(y ~ x1 + x2 | firm + year, data = d, cluster = ~ firm + year)
```

### (c) Robust (heteroskedasticity-consistent) standard errors

**Stata**

```stata
xhdfe ln_wage grade ttl_exp tenure, absorb(idcode year) vce(robust)
```

**Python**

```python
reg = xhdfe.HdfeRegressor(se_type="robust")
reg.fit(y, X, fes=[firm_id, year_id])
```

**R**

```r
m <- xhdfe(y ~ x1 + x2 | firm + year, data = d, vcov = "robust")
```

### (d) Heterogeneous (group-specific) slopes

`fe##c.x` (Stata) = `fe[x]` (R) absorbs a level FE **plus** a group-specific
slope on `x`; `fe#c.x` = `fe[[x]]` absorbs the slope only. In Python, list the
slope in `slopes=` and point `fe_index` at its carrier dimension in `fes`
(`include_intercept=True` mirrors `##`).

**Stata**

```stata
xhdfe ln_wage grade union, absorb(idcode##c.ttl_exp year) vce(cluster idcode)
```

**Python**

```python
reg = xhdfe.HdfeRegressor(se_type="cluster")
reg.fit(
    y, X,
    fes=[firm_id, year_id],
    clusters=[firm_id],
    slopes=[{"fe_index": 0, "values": x2, "include_intercept": True}],
)
```

**R**

```r
m <- xhdfe(y ~ x1 | firm[x2] + year, data = d, cluster = ~ firm)
```

### (e) Instrumental variables / 2SLS

**Stata**

```stata
xhdfe ln_wage grade ttl_exp union, absorb(idcode year) ///
    endogenous(union) instruments(south)
```

**Python**

```python
# put the endogenous column in X and reference it by 0-based position
X_iv = np.column_stack([x1, xe])     # column 1 (xe) is endogenous
reg = xhdfe.HdfeRegressor()
reg.fit(
    y, X_iv,
    fes=[firm_id, year_id],
    instruments=z.reshape(-1, 1),
    endogenous_idx=[1],
)
```

**R**

```r
# third formula part: endogenous ~ instruments
m <- xhdfe(y ~ x1 | firm + year | xe ~ z, data = d)
```

### (f) Weights

Analytic weights (Stata `aweight`) below; frequency weights are also supported
(`fweight` / `weights_type = "frequency"`).

**Stata**

```stata
xhdfe ln_wage grade ttl_exp tenure [aweight=hours], absorb(idcode year)
```

**Python**

```python
reg = xhdfe.HdfeRegressor()
reg.fit(y, X, fes=[firm_id, year_id], weights=w)
```

**R**

```r
m <- xhdfe(y ~ x1 + x2 | firm + year, data = d, weights = ~ w)
# frequency weights: weights = ~ w, weights_type = "frequency"
```

### (g) Saving and recovering fixed effects

Recovered values are per-observation FE **contributions**, not category ids, and
may not be separately identified in every design.

**Stata**

```stata
* reghdfe-style __hdfe*__ variables
xhdfe ln_wage grade ttl_exp union, absorb(idcode year, savefe)

* custom prefix, and also save residuals
xhdfe ln_wage grade ttl_exp union, absorb(idcode year) ///
    residuals(u) savefes(fe_)
```

**Python**

```python
reg = xhdfe.HdfeRegressor(retain_fes=True)
reg.fit(y, X, fes=[firm_id, year_id])
effects = reg.fe_effects_             # one contribution vector per FE dimension
```

**R**

```r
m <- xhdfe(y ~ x1 + x2 | firm + year, data = d, save_fe = TRUE)
alpha <- fixef(m)                     # list, one full-length vector per FE
head(alpha$firm)
```

### (h) Reading results — `e()` / result attributes

**Stata** (`eclass`; `ereturn list` shows everything):

```stata
xhdfe ln_wage grade ttl_exp tenure, absorb(idcode year)
ereturn list
display e(N)               // observations
display e(r2_within)       // within R-squared
display e(df_a)            // absorbed degrees of freedom
display e(iterations)      // absorber iterations
display e(converged)       // 1 if converged
matrix list e(b)           // coefficient vector
matrix list e(V)           // variance-covariance matrix
```

**Python** (attributes on the fitted regressor):

```python
reg.coef_          # point estimates
reg.stderr_        # standard errors
reg.pvalues_       # p-values
reg.nobs_          # observations
reg.r2_within_     # within R-squared
reg.df_a_          # absorbed degrees of freedom
reg.num_iterations_, reg.converged_
reg.summary()      # formatted text table
```

**R** (S3 methods and result fields):

```r
summary(m)
coef(m); vcov(m); confint(m)
nobs(m)
m$r2_within        # within R-squared
m$df_a             # absorbed degrees of freedom
m$iterations; m$converged
```

---

## Choosing a backend (CPU vs CUDA)

CPU is the **reference** backend and the default everywhere. The optional CUDA
backend accelerates only the fixed-effect absorber and is most useful on large
problems. It is opt-in per call and **fail-closed** — if a GPU backend is
requested but is unavailable, fails, or does not converge on the device, the
call stops with an error instead of silently returning CPU results. Always
confirm real GPU use with the diagnostics below.

**Stata** — request CUDA and verify:

```stata
xhdfe ln_wage ttl_exp tenure, absorb(idcode year) gpubackend(cuda)
display e(gpu_used)                 // must be 1
display "`e(gpu_backend)'"          // must be "cuda"
display "`e(gpu_status)'"           // must be "used"
```

If you rebuild or switch the plugin binary inside one Stata session, run
`discard` (with no arguments) before rerunning.

**Python** — select the backend by environment variable (requires a CUDA build):

```python
import os
os.environ["XHDFE_GPU_BACKEND"] = "cuda"
reg.fit(y, X, fes=[firm_id, year_id])
assert reg.gpu_used_ == 1
```

**R** — pass `backend = "cuda"` (requires a CUDA build):

```r
m <- xhdfe(y ~ x1 + x2 | firm + year, data = d, backend = "cuda")
stopifnot(m$gpu_used == 1, m$gpu_status == "used")
```

Building the CUDA backend requires the NVIDIA CUDA toolkit (`nvcc`) and a device
of compute capability `sm_75` or newer; set the target with `XHDFE_CUDA_ARCH`
(e.g. `90` for an H100).

---

## Full documentation

- **Stata:** `help xhdfe` (and `help xfe`); source in `stata/xhdfe.sthlp`.
- **Python:** `python -m xhdfe`, the `xhdfe-help` console script, or
  `xhdfe.help_text()` in a session; source in `xhdfe/help/xhdfe.md`.
- **R:** `?xhdfe` (mirrors the Stata help section by section) and `r/README.md`;
  a full feature tour is in `r/examples/xhdfe_r_showcase.qmd`.
- **Ecosystem map:** [comparison.md](comparison.md).
