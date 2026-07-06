# xhdfe

**Linear regression with multiple high-dimensional fixed effects — in Stata, Python and R, on one fast C++ core.**

`Version 2.12.0` · `License: MIT` · `Stata + Python + R` · `Optional CUDA GPU`

---

## What is xhdfe?

`xhdfe` estimates linear models with any number of high-dimensional fixed
effects (HDFE) — the worker–firm, patent–inventor, and multi-way panel designs
common in applied economics. It mirrors the defaults and reporting of
[`reghdfe`](https://github.com/sergiocorreia/reghdfe) (Correia 2016): under the
default `reghdfe-comparable` tolerance mode, coefficients match `reghdfe` at the
same nominal tolerance. The same estimator is exposed through three front-ends —
a Stata command, a Python package, and an R package — all sitting on a **single
compiled C++ core**. CPU is the reference backend; an optional CUDA GPU absorber
is available for large problems.

## Features

- **Multiway HDFE** — any number of absorbed fixed-effect dimensions, plus two-way categorical interactions.
- **Heterogeneous (group-specific) slopes** — `fe#c.x` and `fe##c.x` designs.
- **IV / 2SLS** with absorbed fixed effects.
- **Weights** — analytic, frequency, probability, and importance weights.
- **Robust and multiway-cluster** standard errors.
- **DoF adjustments** — reghdfe-style singleton dropping and degrees-of-freedom logic, plus fixest-style small-sample corrections (`ssc`).
- **Fixed-effect recovery** — `savefe` / `savefes` (Stata), `fixef()` (R), `retain_fes` (Python).
- **Group-level outcomes with individual fixed effects** — the `group()` / `individual()` machinery.
- **Mobility groups** and connected-component diagnostics.
- **Optional GPU** — CUDA absorber with explicit request and status reporting; fail-closed (never a silent CPU fallback).
- **AKM / worker-firm post-estimation** — leave-out (KSS) variance decomposition with plug-in, AGSU and KSS corrections, exact and Johnson-Lindenstrauss leverages, component standard errors, Andrews-Mikusheva weak-identification confidence intervals, fweights, and a leave-one-out connected-set utility (`xhdfeakm` / `xhdfeconnected` in Stata, `xhdfe.akm` in Python, `xhdfe_akm_kss()` in R); validated against Saggio's LeaveOutTwoWay (the canonical KSS implementation) and pytwoway. See [`docs/akm-kss.md`](docs/akm-kss.md).
- **Gelbach decomposition** — `xhdfegelbach` / `xhdfe.gelbach` / `xhdfe_gelbach()`, validated against Gelbach's `b1x2`.

---

## Choose your language

The three packages call the same C++ estimator, so results agree across
languages. Pick your front-end below. **For GPU (CUDA) acceleration in any of
them, see the [GPU (CUDA) guide](docs/gpu.md)** — it walks through installing
with the GPU feature, requesting it, and verifying it in Stata, Python, and R.

### Stata

**Option A — install online from the release net-install site.** The release
workflow publishes a Stata site whose `.pkg` files select the plugin for the
current OS:

```stata
net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
net install xfe,   from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
```

The online package uses Stata platform-specific `g` lines for Linux, macOS
Apple Silicon/Intel, and Windows when a Windows plugin artifact exists.

> **These prebuilt plugins are CPU-only** (OpenMP, no CUDA). The online
> net-install and the release ZIPs never ship a GPU plugin. For NVIDIA GPU
> acceleration you must build from source with the CUDA flag — see **Option C**
> below. There is no way to obtain a GPU plugin from the online material alone,
> because it contains no source, only the compiled CPU binaries.

**Option B — install a prebuilt release ZIP.** Download the distribution ZIP from the
[Releases](https://github.com/reisportela/xhdfe-xfe/releases) page and unzip it. It
bundles the platform plugin. In Stata, point `net install` at the unzipped
folder that contains `xhdfe.pkg` and `stata.toc`:

```stata
net install xhdfe, from("/path/to/unzipped/xhdfe/stata") replace
net install xfe,   from("/path/to/unzipped/xhdfe/stata") replace
```

**Option C — build the plugin from source (required for GPU).** Get the source
(clone the repo, or download a *source* release ZIP that bundles the C++ backend
and build scripts) and build the plugin, then add the folder to your `adopath`:

```bash
git clone https://github.com/reisportela/xhdfe-xfe.git
cd xhdfe-xfe

# CPU build (Linux + GCC; OpenMP recommended)
bash stata/tools/build-plugin.sh --linux --openmp        # produces stata/xhdfe.plugin
bash stata/tools/build-xfe-plugin.sh --linux --openmp    # produces stata/xfe.plugin
```

For **NVIDIA GPU (CUDA)**, use `--cuda auto`. The build script checks
`nvidia-smi`, selects the local GPU architecture, and enables CUDA:

```bash
bash stata/tools/build-plugin.sh --linux --openmp --cuda auto
bash stata/tools/build-xfe-plugin.sh --linux --openmp --cuda auto
```

CUDA builds are **Linux + NVIDIA only** and require the CUDA toolkit (`nvcc`).
For an explicit target, use `--cuda 90`; for a shareable multi-GPU binary, use
`--cuda-archs "75,80,86,89,90"`. See `stata/BUILD_CUDA.md` for verification.
Then point Stata at the folder:

```stata
adopath + "/path/to/xhdfe/stata"
```

Minimal example (public data shipped with Stata):

```stata
sysuse auto, clear
xhdfe price weight length, absorb(rep78)
xhdfe price weight length, absorb(rep78) vce(cluster rep78)

* Optional: request CUDA after installing a CUDA-enabled plugin
xhdfe price weight length, absorb(rep78) vce(cluster rep78) gpubackend(cuda)
display e(gpu_used)                 // must be 1
display "`e(gpu_backend)'"          // must be "cuda"

webuse nlswork, clear
xhdfe ln_wage grade age ttl_exp tenure not_smsa south, absorb(idcode year)
xhdfe ln_wage grade age ttl_exp tenure not_smsa south, absorb(idcode year occ_code)
```

`xfe` is a companion command that partials out variables (residualizes against
absorbed fixed effects) using the same core. See `help xhdfe` and `help xfe`.

### Python

Install from the repository. Python source builds require CMake, a C++ compiler,
and the Python development headers for the Python you are using (`Python.h`).
On Linux, install the matching system package first, for example
`python3-dev` on Debian/Ubuntu or `python3-devel` on Fedora/RHEL/Rocky. On
clusters without sudo, use a conda/mamba environment or a Python module that
includes development headers.

```bash
python -m pip install "git+https://github.com/reisportela/xhdfe-xfe.git"
# or, from a clone:
git clone https://github.com/reisportela/xhdfe-xfe.git && cd xhdfe-xfe && python -m pip install .
```

**With the GPU (CUDA) feature** (Linux + NVIDIA only; needs the CUDA toolkit
`nvcc` and always builds from source — never a prebuilt wheel). Use
`XHDFE_ENABLE_CUDA=auto` to detect the local GPU architecture:

```bash
# from a clone:
XHDFE_ENABLE_CUDA=auto python -m pip install .
# or straight from GitHub:
XHDFE_ENABLE_CUDA=auto python -m pip install "git+https://github.com/reisportela/xhdfe-xfe.git"
```

For an explicit target, set `XHDFE_CUDA_ARCH=90` or
`CMAKE_CUDA_ARCHITECTURES=90`.
At runtime request the GPU with `os.environ["XHDFE_GPU_BACKEND"] = "cuda"` (see
the example below) and confirm with `reg.gpu_used_ == 1`.

Minimal example:

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

### R

Install from GitHub (the package lives in the `r/xhdfe` subdirectory). This
gives you the **CPU** build:

```r
# install.packages("remotes")
remotes::install_github("reisportela/xhdfe-xfe", subdir = "r/xhdfe")
```

**With the GPU (CUDA) feature** (Linux + NVIDIA only; needs the CUDA toolkit
`nvcc` and always builds from source). Use `XHDFE_ENABLE_CUDA=auto` to detect
the local GPU architecture:

```r
Sys.setenv(XHDFE_ENABLE_CUDA = "auto")
remotes::install_github("reisportela/xhdfe-xfe", subdir = "r/xhdfe")
```

or, from a clone: `XHDFE_ENABLE_CUDA=auto R CMD INSTALL r/xhdfe`. For an
explicit target, set `XHDFE_CUDA_ARCH=90`.
GPU use is then per call via `backend = "cuda"` (fail-closed if unavailable);
`xhdfe_info()` reports the CUDA arch the package was built for.

Minimal example (a small simulated worker–firm panel):

```r
library(xhdfe)

set.seed(2026)
n <- 600
d <- data.frame(
  worker = sample(80, n, replace = TRUE),
  firm   = sample(30, n, replace = TRUE),
  x1     = rnorm(n),
  x2     = rnorm(n)
)
d$y <- 0.5 * d$x1 - 0.2 * d$x2 + 0.05 * d$worker + 0.03 * d$firm + rnorm(n)

# Two-way fixed effects (worker + firm), clustered by firm
m <- xhdfe(y ~ x1 + x2 | worker + firm, data = d, cluster = ~ firm)
summary(m)

# Optional: request CUDA after installing a CUDA-enabled build
m_gpu <- xhdfe(y ~ x1 + x2 | worker + firm, data = d,
               cluster = ~ firm, backend = "cuda")
stopifnot(m_gpu$gpu_used == 1, m_gpu$gpu_status == "used")
```

The R formula grammar is fixest-style: `y ~ x | fe1 + fe2` for absorbed FEs,
`fe[slope]` / `fe[[slope]]` for heterogeneous slopes, `f1^f2` for a combined
interaction FE, and `| endo ~ inst` for IV. See
[`r/README.md`](r/README.md) for the CUDA build and the platform note, and
`?xhdfe` for the full documentation.

---

## Worker-firm (AKM) and Gelbach post-estimation

Beyond general-purpose HDFE regression, `xhdfe` ships a worker-firm layer that
follows the Kline-Saggio-Solvsten (2020) leave-out methodology (validated
against Saggio's LeaveOutTwoWay and `pytwoway`) and a Gelbach (2016)
decomposition — all on the same compiled backend, in Stata, Python and R.
Installation is the same as the core (they are part of the one package).

Stata:

```stata
* leave-one-out connected set, then the AKM/KSS variance decomposition
xhdfeconnected worker firm, generate(insample)
xhdfeakm y, worker(worker) firm(firm) ci          // KSS SEs + Andrews-Mikusheva CIs
xhdfegelbach y, x1(educ) x2groups("skill = ability") fes(firm)
```

Python:

```python
import xhdfe.akm as akm, xhdfe.gelbach as gelbach
r = akm.akm_kss(y, worker, firm, compute_se=True, eigen_diagnostics=True)
print(r["kss"], r["component_se"], r["weak_id"])
g = gelbach.decompose(y, educ, x2_groups={"skill": ability}, fes={"firm": firm})
```

R:

```r
fit <- xhdfe_akm_kss(y, worker, firm, compute_se = TRUE, eigen_diagnostics = TRUE)
g   <- xhdfe_gelbach(y, x1 = educ, x2_groups = list(skill = ability),
                     fes = list(firm = firm))
```

The plug-in, AGSU (homoskedastic) and KSS (heteroskedasticity-robust leave-out)
decompositions report the variance of worker effects, of firm effects, their
covariance and correlation, and the shares of wage variance; with `se`/`ci`
they add component standard errors and Andrews-Mikusheva weak-identification
confidence intervals. Exact and Johnson-Lindenstrauss leverages, frequency
weights and an optional CUDA solver are supported. Runnable examples in all
three languages live in [`examples/`](examples/); a felsdvsimul walkthrough is
in [`docs/akm-kss.md`](docs/akm-kss.md).

---

## Repository layout

| Path | Contents |
| --- | --- |
| `src/`, `include/`, `third_party/eigen-3.4.0/` | The shared C++ core (MAP absorber plus Krylov / Schwarz / LSMR variants) and header-only Eigen. |
| `python/`, `xhdfe/` | Python package (`import xhdfe`; the `HdfeRegressor` class). |
| `r/` | R package (`r/xhdfe/`), examples, and helper tools. |
| `stata/` | Stata package: `xhdfe.ado`, `xfe.ado`, help files, plugin sources (`src/`), and build scripts (`tools/`). |
| `docs/` | Quickstart and overview. |
| `CMakeLists.txt`, `pyproject.toml`, `setup.py` | Build configuration for the C++ core and Python bindings. |

---

## Documentation

- **Quickstart & overview:** [`docs/quickstart.md`](docs/quickstart.md), [`docs/overview.md`](docs/overview.md).
- **GPU (CUDA):** [`docs/gpu.md`](docs/gpu.md) — install-with-GPU, request, and verify in Stata/Python/R.
- **AKM + leave-out (KSS) & Gelbach:** [`docs/akm-kss.md`](docs/akm-kss.md); `help xhdfeakm`, `help xhdfeconnected`, `help xhdfegelbach`.
- **Release workflow:** [`docs/release-workflow.md`](docs/release-workflow.md).
- **Stata:** `help xhdfe`, `help xfe`.
- **R:** `?xhdfe`, `?fixef.xhdfe`, `?predict.xhdfe`; feature tour in `r/examples/`.
- **Python:** `python -m xhdfe` or `xhdfe-help` at the shell, or `xhdfe.help_text()` inside Python.

## Validation & accuracy

Under the default `reghdfe-comparable` tolerance mode, `xhdfe` coefficients,
standard errors, and recovered fixed effects match `reghdfe` at the same nominal
tolerance (down to the conditioning of the problem). The three packages are
cross-checked against each other and against the wider ecosystem —
[`reghdfe`](https://github.com/sergiocorreia/reghdfe) (Stata),
[`fixest`](https://github.com/lrberge/fixest) (R),
[`pyfixest`](https://github.com/py-econometrics/pyfixest) (Python), and
[`FixedEffectModels.jl`](https://github.com/FixedEffects/FixedEffectModels.jl)
(Julia). This software is released as a **proof of concept**: please validate
estimates for your own research design. See [`DISCLAIMER.md`](DISCLAIMER.md).

## Citation

If you use `xhdfe` in academic work, please cite it (see
[`CITATION.cff`](CITATION.cff)):

> Portela, Miguel, and Tiago Tavares. 2026. *xhdfe: High-dimensional fixed
> effects regression via a C++ backend.* Version 2.12.0.
> https://github.com/reisportela/xhdfe-xfe

## License

MIT — see [`LICENSE`](LICENSE). `xhdfe` bundles the Eigen 3.4.0 headers
(primarily MPL-2.0, with parts under BSD-3-Clause and Apache-2.0); see
[`NOTICE`](NOTICE).

## Authors

- **Miguel Portela** — NIPE / Universidade do Minho and BPLIM / Banco de Portugal.
- **Tiago Tavares** — NIPE / Universidade do Minho.

*Development note:* `xhdfe` was built with AI-assisted tooling, but only the two
listed humans are authors; no software tool or AI system is credited as an
author or co-author.

## Acknowledgements

`xhdfe` validates against and interoperates with prior HDFE software. Full
credit goes to [`reghdfe`](https://github.com/sergiocorreia/reghdfe) by Sergio
Correia (Stata), [`fixest`](https://github.com/lrberge/fixest) by Laurent
Berge (R), [`pyfixest`](https://github.com/py-econometrics/pyfixest) by
Alexander Fischer and collaborators (Python), and
[`FixedEffectModels.jl`](https://github.com/FixedEffects/FixedEffectModels.jl)
by Matthieu Gomez and collaborators (Julia).

The worker-firm (AKM) leave-out layer follows, and is validated at machine
precision against, [`LeaveOutTwoWay`](https://github.com/rsaggio87/LeaveOutTwoWay)
by Raffaele Saggio (MATLAB) — the canonical Kline-Saggio-Solvsten (2020)
implementation — and interoperates with, and is cross-checked against,
[`pytwoway`](https://github.com/tlamadon/pytwoway) by Thibaut Lamadon and
collaborators (Python). The Gelbach decomposition is validated against
`b1x2` by Jonah Gelbach (Stata). Full credit to their authors.

We thank Paulo Guimaraes, Marta Silva, and Nelson Areal for discussions and
workshop collaboration around earlier versions of the project. We especially
thank Sergio Correia for feedback on benchmarking, tolerances, and
`reghdfe`-comparable validation. All remaining errors are ours.

## References

Methods implemented by `xhdfe`'s worker-firm post-estimation layer:

- Abowd, J. M., F. Kramarz, and D. N. Margolis. 1999. High wage workers and
  high wage firms. *Econometrica* 67(2): 251-333. (AKM two-way model.)
- Andrews, M. J., L. Gill, T. Schank, and R. Upward. 2008. High wage workers
  and low wage firms: negative assortative matching or limited mobility bias?
  *Journal of the Royal Statistical Society A* 171(3): 673-697. (AGSU
  homoskedastic correction.)
- Kline, P., R. Saggio, and M. Solvsten. 2020. Leave-out estimation of
  variance components. *Econometrica* 88(5): 1859-1898. (KSS leave-out
  heteroskedasticity-robust correction and inference.)
- Andrews, D. W. K., and A. Mikusheva. 2016. Conditional inference with a
  functional nuisance parameter. *Econometrica* 84(4): 1571-1612.
  (Weak-identification q=1 confidence intervals used by KSS.)
- Gelbach, J. B. 2016. When do covariates matter? And which ones, and how
  much? *Journal of Labor Economics* 34(2): 509-543. (Conditional
  decomposition of coefficient movements.)

## Contributing

Contributions, bug reports, and validation cases are welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Platform support

The C++ core, Python bindings, and R package target Linux x86-64, Windows
x86-64, and macOS Apple Silicon/Intel source builds with the local platform
toolchain. CUDA GPU acceleration is optional on Linux with the NVIDIA toolkit.
Prebuilt Stata release assets are CPU-only: Linux CPU/OpenMP and macOS
universal plugins are published, and the Windows CPU plugin is included when
the release build produces the artifact. CUDA GPU builds are source builds on
Linux with the NVIDIA toolkit.
