# xhdfe for R

R package exposing the xhdfe C++ backend — the same compiled estimator behind
the Stata `xhdfe` command and the Python `xhdfe` package. The full estimator
surface is exposed to R: multiway high-dimensional fixed effects,
heterogeneous (group-specific) slopes, IV/2SLS, analytic and frequency
weights, robust and multiway-clustered inference with reghdfe/fixest-style
small-sample controls, singleton dropping and DoF adjustments, fixed-effect
recovery, group-level outcomes with individual fixed effects, mobility
groups, and the optional CUDA GPU absorber with fail-closed semantics.

**Supported CPU source-install platforms:** Linux, macOS (Apple Silicon and
Intel where an R C++17 toolchain is available), and Windows with Rtools. CUDA
builds are currently Linux-only.

**Stata options without a dedicated R argument.** The Stata command's
diagnostic/cache options — `mobilityprofile`, `mobfile()`,
`absorptioncache()`, `absorptioncachemode()`, `festructurecache`,
`fescache()`, `fescachemode()` — have no R arguments, but the underlying
machinery lives in the shared C++ core and is driven by environment
variables that work identically from R (set before the fit):
`XHDFE_MOBILITY_PROFILE` / `XHDFE_MOBILITY_MODE`, `XHDFE_ABSORPTION_CACHE` /
`XHDFE_ABSORPTION_CACHE_MODE`, `XHDFE_FE_STRUCTURE_CACHE` /
`XHDFE_FE_STRUCTURE_MODE`. Stata's display-only options (`noheader`,
`cformat()`, ...) have no R counterpart by design (use R's own printing).

## Layout

- `xhdfe/` — the R package source.
  - `xhdfe/src/` — Rcpp binding (`rcpp_xhdfe.cpp`) plus a **byte-for-byte
    mirror** of the canonical C++ core (`src/`, `include/` at the repo root)
    and the vendored Eigen 3.4.0 headers. Do not edit the mirrored files
    here; edit the canonical core and run `tools/refresh_r_core.sh`.
- `tools/refresh_r_core.sh` — refreshes and verifies the core mirror.
- `tools/gen_parity_data.R` + `tools/gen_parity_fixture.py` — regenerate the
  R↔Python parity fixture used by the test suite (requires the reference
  `build/` Python module).
- `examples/xhdfe_r_showcase.qmd` — executable feature tour + the complete
  core23 benchmark-suite examples.
- `Rlib/`, `Rlib_cuda/` — local install libraries used during development and
  validation (CPU and CUDA builds respectively); not part of the package.

## Install (CPU)

From this directory:

```bash
R CMD INSTALL xhdfe
```

The build replicates the reference CMake Release flags of the C++ core
(`-O3 -ffast-math` plus the platform's available OpenMP flags; see
`xhdfe/src/Makevars` and `xhdfe/src/Makevars.win`). The default build is
portable for the architecture selected by R. For a local machine-specific
benchmark build, set `XHDFE_MARCH=native`; for a shareable Linux x86-64 build,
set a portable psABI level such as `XHDFE_MARCH=x86-64-v3`. Verify any install
with `xhdfe_info()` (reports compiler, `-march`, fast-math and CUDA arch).

Known `R CMD check` results on the supported platform: 1 WARNING (the
`.cu`/`.hpp` sources in `src/` — kept intentionally so tarball installs can
rebuild the CUDA absorber) and 1 NOTE (`std::cout`/`std::cerr` in the core —
opt-in, env-gated diagnostics that are silent by default).

## Install (CUDA, optional)

Requires Linux, the NVIDIA CUDA toolkit (nvcc on PATH or `CUDA_HOME` set), and
the package source (a clone or source tarball — a CPU install has no GPU). Use
`XHDFE_ENABLE_CUDA=auto` to detect the local GPU architecture:

```bash
XHDFE_ENABLE_CUDA=auto R CMD INSTALL xhdfe
```

To install straight from GitHub with GPU support, set the same variable in R
first:

```r
Sys.setenv(XHDFE_ENABLE_CUDA = "auto")
remotes::install_github("reisportela/xhdfe-xfe", subdir = "r/xhdfe")
```

For an explicit target, set `XHDFE_CUDA_ARCH=90`.
Verify the build with `xhdfe_info()` (its `cuda_arch` field shows e.g. `sm_90`).
GPU use is opt-in per call (`backend = "cuda"`) and fail-closed: if the GPU
is unavailable or absorption does not complete on it, the call errors instead
of silently returning CPU results — mirroring the Stata command's error 498
contract. CPU remains the reference backend.

## Quick start

```r
library(xhdfe)
m <- xhdfe(y ~ x1 + x2 | firm + year, data, cluster = ~firm)
summary(m)
fixef(xhdfe(y ~ x1 | firm + year, data, save_fe = TRUE))
# heterogeneous slopes (Stata: absorb(firm##c.tenure)):
xhdfe(y ~ x1 | worker + firm[tenure], data)
# IV (fixest-style):
xhdfe(y ~ exo | firm + year | endo ~ instr, data)
# CUDA build only:
m_gpu <- xhdfe(y ~ x1 + x2 | firm + year, data, backend = "cuda")
stopifnot(m_gpu$gpu_used == 1, m_gpu$gpu_status == "used")
```

Worker-firm (AKM) leave-out (KSS) decomposition and the Gelbach companion
(same compiled backend; validated against Saggio's LeaveOutTwoWay and
`pytwoway`):

```r
# leave-out (KSS) variance decomposition with component SEs + AM q=1 CIs
fit <- xhdfe_akm_kss(y, worker, firm, compute_se = TRUE, eigen_diagnostics = TRUE)
fit$kss          # var(alpha), var(psi), cov, corr, shares of var(y)
fit$component_se # KSS standard errors
fit$weak_id      # Andrews-Mikusheva confidence intervals
# Gelbach conditional decomposition
xhdfe_gelbach(y, x1 = educ, x2_groups = list(skill = ability),
              fes = list(firm = firm))
```

See `?xhdfe` for the full documentation (it mirrors the Stata help file
section by section), `?xhdfe_akm_kss` / `?xhdfe_gelbach` for the worker-firm
layer, and `examples/xhdfe_r_showcase.qmd` plus the top-level `examples/`
scripts for complete tours.

## Validation

`xhdfe/tests/testthat/` contains:

- `test-parity-python.R` — the R binding must reproduce the reference Python
  module (same core, same flags) across 24 spec entries: OLS, 2/3-way FEs,
  robust/one-way/multiway cluster, analytic weights, IV, heterogeneous
  slopes, singleton handling, DoF/SSC controls, all three tolerance modes,
  no-constant, confidence levels, FE recovery, mobility groups, MLSMR and
  symmetric-GS overrides, and group()/individual() fits — coefficients at
  1e-9, inference at 1e-7, iteration counts and dof exactly.
- `test-interface.R` — exact agreement with `lm()` on dummy-expanded designs,
  `fixest::feols` agreement (coefficients and cluster SEs under
  `ssc(fixef.K = "nested", cluster.df = "min", t.df = "min")`), frequency
  weights vs exact replicated-rows equivalence, formula grammar, S3 methods,
  prediction identities (y = xb + d + e), and fail-closed GPU behavior.

Run with:

```bash
Rscript -e '.libPaths(c("Rlib", .libPaths())); testthat::test_local("xhdfe")'
```
