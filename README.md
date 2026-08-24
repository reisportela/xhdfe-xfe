# xhdfe

**Linear regression with multiple high-dimensional fixed effects — in Stata, Python and R, on one fast C++ core.**

`Version 2.25.0` · `License: MIT` · `Stata + Python + R` · `Optional CUDA GPU`

---

## What is xhdfe?

`xhdfe` estimates linear models with any number of high-dimensional fixed
effects (HDFE) — the worker–firm, patent–inventor, and multi-way panel designs
common in applied economics. It follows the defaults and reporting conventions
of [`reghdfe`](https://github.com/sergiocorreia/reghdfe) (Correia 2016). Under
the default `reghdfe-comparable` tolerance mode, the certification suite checks
coefficients and inference against `reghdfe` at the same nominal tolerance. A
Stata command, a Python package, and an R package all call the **same compiled
C++ core**. CPU is the reference backend; an optional CUDA GPU absorber is
available for large problems.

As an illustration, the table below reports median estimator-call runtimes for
an AKM-style wage regression using Portuguese matched employer-employee data.
The specification uses 55,947,171 observations and absorbs 5,948,793 worker,
799,265 firm, and 36 year fixed effects. It includes common seniority controls
and clusters standard errors at the worker level:

| Implementation | Backend | Seconds | Speedup vs. `reghdfe` |
| --- | ---: | ---: | ---: |
| `reghdfe`, Stata | CPU | 3,667.5 | 1.0x |
| `FixedEffectModels.jl` | CPU | 916.3 | 4.0x |
| `fixest` | CPU | 486.5 | 7.5x |
| `pyfixest` | CPU | 110.4 | 33.2x |
| `FixedEffectModels.jl` | CUDA | 105.7 | 34.7x |
| `xhdfe`, Stata | CPU | 61.3 | 59.8x |
| `xhdfe`, Python | CPU | 53.6 | 68.4x |
| `xhdfe`, Stata | CUDA | 22.7 | 161.7x |
| `xhdfe`, Python | CUDA | 14.5 | 252.2x |

The `xhdfe` rows use the speed-oriented `xhdfe-fast` mode; the default
`reghdfe-comparable` mode is somewhat slower but matches `reghdfe` more tightly.

## Features

- **Multiway HDFE** — any number of absorbed fixed-effect dimensions, plus two-way categorical interactions.
- **Dual Python API** — low-overhead arrays or an optional R-style formula frontend with categories and interactions.
- **Publication tables** — fitted Python results plug directly into `maketables.ETable` without an adapter or runtime dependency.
- **Heterogeneous (group-specific) slopes** — `fe#c.x` and `fe##c.x` designs.
- **IV / 2SLS** with absorbed fixed effects.
- **Weights** — analytic, frequency, probability, and importance weights.
- **Robust and multiway-cluster** standard errors.
- **DoF adjustments** — reghdfe-style singleton dropping and degrees-of-freedom logic, plus fixest-style small-sample corrections (`ssc`).
- **Fixed-effect recovery** — `savefe` / `savefes` (Stata), `fixef()` (R), `retain_fes` (Python).
- **Group-level outcomes with individual fixed effects** — the `group()` / `individual()` machinery.
- **Mobility groups** and connected-component diagnostics.
- **Optional GPU** — CUDA absorber with explicit request and status reporting; fail-closed (never a silent CPU fallback).
- **AKM / worker-firm post-estimation** — plug-in, AGSU, and KSS leave-out
  variance decompositions; exact or Johnson-Lindenstrauss leverages; component
  inference; and leave-one-out connected-set preparation. See
  [`docs/akm-kss.md`](docs/akm-kss.md).
- **Gelbach decomposition** — `xhdfegelbach` / `xhdfe.gelbach` /
  `xhdfe_gelbach()`, with multiple focal coefficients and covariate blocks,
  common and added HDFEs, explicitly declared absorbed targets,
  joint-covariance inference, bootstrap, tables, and plots. See the
  [companion documentation](#worker-firm-akm-and-gelbach-post-estimation).

---

## Choose your language

The three frontends call the same C++ estimator and are tested for agreement on
shared specifications. Pick your language below. For CUDA installation,
selection, and verification, see the [GPU guide](docs/gpu.md).

### Stata

One command installs the estimator, the CPU plugin for the current OS, and
every companion command:

```stata
net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
```

The installed commands are:

| Command | What it does |
| --- | --- |
| `xhdfe` | HDFE linear regression — the estimator (`help xhdfe`). |
| `xfepout` | Partials out / residualizes variables against the fixed effects, no regression (`help xfepout`). |
| `xhdfeakm` | Worker-firm (AKM) leave-out (KSS) variance decomposition (`help xhdfeakm`). |
| `xhdfeconnected` | Largest leave-one-out connected set — KSS sample prep (`help xhdfeconnected`). |
| `xhdfegelbach` | Gelbach (2016) decomposition of coefficient movements (`help xhdfegelbach`). |
| `xhdfegelbachbootstrap` | Full-refit iid- or cluster-pairs bootstrap for Gelbach results (`help xhdfegelbachbootstrap`). |
| `xhdfegelbachetable` | Publication-oriented Gelbach tables (`help xhdfegelbachetable`). |
| `xhdfegelbachcoefplot` | Identity-preserving Gelbach waterfall plot (`help xhdfegelbachcoefplot`). |
| `xhdfegpu` | Builds and installs a CUDA GPU plugin for this machine (`help xhdfegpu`). |

To install only the standalone partial-out tool after publication:

```stata
net install xfepout, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
```

> **Stata command rename.** `xfepout` replaces the former `xfe` command. The
> numerical implementation and syntax are otherwise unchanged, and no Python
> or R API was renamed. If a previous standalone installation still provides
> `xfe`, run `ado uninstall xfe` before installing `xfepout`; the new package
> deliberately does not ship a permanent `xfe` alias.

#### Minimal example

This example uses public data shipped with Stata:

```stata
sysuse auto, clear
xhdfe price weight length, absorb(rep78) vce(cluster rep78)

webuse nlswork, clear
xhdfe ln_wage grade age ttl_exp tenure not_smsa south, absorb(idcode year occ_code)
```

#### Optional CUDA build

On Linux with an NVIDIA GPU and the CUDA toolkit, run this once after
`net install`:

```stata
xhdfegpu
```

`xhdfegpu` detects the GPU, compiles for its architecture, and replaces the CPU
plugins in place. Reload Stata, request CUDA, and verify that it was used:

```stata
discard
xhdfe price weight length, absorb(rep78) gpubackend(cuda)
display e(gpu_used)          // 1
```

For an offline machine, copy the release's self-contained `xhdfe-src.zip` and
pass it directly to the builder:

```stata
xhdfegpu, zip("/path/to/xhdfe-src.zip")
```

The command requires `nvcc` and a C++ compiler. See `help xhdfegpu` and the
[GPU guide](docs/gpu.md).

#### Install from a release ZIP (offline, no GitHub)

Download and unzip a distribution from
[Releases](https://github.com/reisportela/xhdfe-xfe/releases), then point Stata
at the folder containing `xhdfe.pkg` and `stata.toc`:

```stata
net install xhdfe, from("/path/to/unzipped/xhdfe/stata") replace
```

#### Build the plugin by hand (advanced)

For a manual build, clone the repository or unpack `xhdfe-src.zip`:

```bash
git clone https://github.com/reisportela/xhdfe-xfe.git
cd xhdfe-xfe
# CPU/OpenMP
bash stata/tools/build-plugin.sh --linux --openmp
bash stata/tools/build-xfepout-plugin.sh --linux --openmp
# CUDA, with architecture auto-detection
bash stata/tools/build-plugin.sh --linux --openmp --cuda auto
bash stata/tools/build-xfepout-plugin.sh --linux --openmp --cuda auto
```

Use `--cuda 90` for an explicit target or
`--cuda-archs "75,80,86,89,90"` for a multi-GPU build. Then add the `stata/`
folder to `adopath`. See [`stata/BUILD_CUDA.md`](stata/BUILD_CUDA.md).

### Python

Python builds from source and requires CMake, a C++ compiler, and matching
Python development headers (`python3-dev` on Debian/Ubuntu or `python3-devel`
on Fedora/RHEL/Rocky):

```bash
python -m pip install "git+https://github.com/reisportela/xhdfe-xfe.git"
# or, from a clone:
git clone https://github.com/reisportela/xhdfe-xfe.git && cd xhdfe-xfe && python -m pip install .
```

Contributor setup and the complete test suite are documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

#### Minimal example

```python
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
```

On macOS, the standard AppleClang source build is supported but has no OpenMP
threading. For a multi-threaded build, install Homebrew GCC and use its
versioned C++ compiler (replace `15` below if Homebrew reports a different
major version):

```bash
brew install gcc
CXX=g++-15 python -m pip install .
```

For CUDA on Linux, install from source with the NVIDIA toolkit available. An
explicit request fails rather than silently producing a CPU-only build:

```bash
XHDFE_ENABLE_CUDA=auto python -m pip install .
```

For an explicit target, set `XHDFE_CUDA_ARCH=90` or
`CMAKE_CUDA_ARCHITECTURES=90`. At runtime set
`XHDFE_GPU_BACKEND=cuda` and confirm `reg.gpu_used_ == 1`; see the
[GPU guide](docs/gpu.md).

The optional R-style formula frontend uses the same native estimator while
adding named dataframe designs. Install the extra from a source checkout:

```bash
python -m pip install '.[formula]'
```

```python
model = xhdfe.feols(
    "y ~ x1 + x2 + C(industry) | firm + year",
    data=d,
    se_type="cluster",
    clusters="firm",
)
print(model.tidy())
```

`C(g)` corresponds to Stata's `i.g`; fixed effects after `|` remain identifier
columns rather than dummy variables. A third formula part requests 2SLS:

```python
model = xhdfe.feols("y ~ x1 | firm + year | d ~ z1 + z2", data=d)
```

The array API remains the lowest-overhead route for repeated small regressions.
See the [Python formula interface](xhdfe/help/xhdfe.md#optional-formula-interface)
for formula syntax, categorical terms, interactions, and IV/2SLS.

Fitted results implement the duck-typed plug-in format of
[maketables](https://github.com/py-econometrics/maketables), so publication
tables need no adapter or registration step:

```python
import maketables as mt

print(mt.ETable([model_a, model_b], drop="Intercept").make(type="tex"))
```

`maketables` is not an `xhdfe` runtime dependency; install it separately if you
want this integration. The packaged Python help lists the available
coefficient and model statistics.

### R

Install from GitHub (the package lives in the `r/xhdfe` subdirectory). This
gives you the **CPU** build:

```r
# install.packages("remotes")
remotes::install_github("reisportela/xhdfe-xfe", subdir = "r/xhdfe")
```

#### Minimal example

This example uses a small simulated worker–firm panel:

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
```

For CUDA on Linux, set `XHDFE_ENABLE_CUDA=auto` before installation. The build
requires the NVIDIA toolkit and fails if the requested GPU path is unavailable:

```r
Sys.setenv(XHDFE_ENABLE_CUDA = "auto")
remotes::install_github("reisportela/xhdfe-xfe", subdir = "r/xhdfe")
```

From a clone, use `XHDFE_ENABLE_CUDA=auto R CMD INSTALL r/xhdfe`; set
`XHDFE_CUDA_ARCH=90` for an explicit target. Select CUDA per call with
`backend = "cuda"` and verify the build with `xhdfe_info()`.

For a network-disabled installation from `xhdfe-src.zip` or the autonomous
offline bundle, install the pinned Rcpp source into a local library first:

```bash
mkdir -p r/Rlib
R_PROFILE_USER=/dev/null R_ENVIRON_USER=/dev/null \
  R_LIBS_USER="$PWD/r/Rlib" \
  R CMD INSTALL --library="$PWD/r/Rlib" third_party/Rcpp_1.1.2.tar.gz
R_PROFILE_USER=/dev/null R_ENVIRON_USER=/dev/null \
  R_LIBS_USER="$PWD/r/Rlib" XHDFE_ENABLE_CUDA=OFF \
  R CMD INSTALL --library="$PWD/r/Rlib" r/xhdfe
```

The R formula grammar is fixest-style: `y ~ x | fe1 + fe2` for absorbed FEs,
`fe[slope]` / `fe[[slope]]` for heterogeneous slopes, `f1^f2` for a combined
interaction FE, and `| endo ~ inst` for IV. See
[`r/README.md`](r/README.md) for the CUDA build and the platform note, and
`?xhdfe` for the full documentation.

---

## Worker-firm (AKM) and Gelbach post-estimation

Beyond general-purpose HDFE regression, `xhdfe` ships a worker-firm layer that
follows the Kline-Saggio-Sølvsten (2020) leave-out methodology (validated
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

The AKM layer reports plug-in, AGSU, and KSS leave-out components for worker and
firm effects, their covariance, and sorting. Exact and
Johnson-Lindenstrauss leverages, frequency weights, component inference,
weak-identification confidence intervals, and an optional CUDA solver are
available. It is validated against Saggio's `LeaveOutTwoWay` and `pytwoway`.

Gelbach's standard mode decomposes the movement from a base model to a full
model. Its separate absorbed-target mode covers a declared focal variable that
belongs to an added FE span, with inference clustered at that FE dimension.
The current interfaces support multiple focal coefficients and covariate
blocks, common and added HDFEs, joint-covariance inference, retained-sample and
regularity diagnostics, full-refit pairs bootstrap, tables, and plots. The
decomposition is specification accounting, not evidence of causal mediation.

Run `help xhdfeakm`, `help xhdfegelbach`, `python -m xhdfe gelbach`, or
`?xhdfe_gelbach` for the complete contracts. Runnable examples live in
[`examples/`](examples/), with an AKM walkthrough in
[`docs/akm-kss.md`](docs/akm-kss.md).

---

## Documentation

- **Quickstart & overview:** [`docs/quickstart.md`](docs/quickstart.md), [`docs/overview.md`](docs/overview.md).
- **GPU (CUDA):** [`docs/gpu.md`](docs/gpu.md) — install-with-GPU, request, and verify in Stata/Python/R.
- **AKM + leave-out (KSS) & Gelbach:** [`docs/akm-kss.md`](docs/akm-kss.md);
  `help xhdfeakm`, `help xhdfeconnected`, `help xhdfegelbach`,
  `help xhdfegelbachbootstrap`, `help xhdfegelbachetable`,
  `help xhdfegelbachcoefplot`,
  `python -m xhdfe gelbach`, and `?xhdfe_gelbach`.
- **Release workflow:** [`docs/release-workflow.md`](docs/release-workflow.md).
- **Release history and certification:** [`docs/releases/`](docs/releases/),
  [`docs/certification/`](docs/certification/).
- **Stata:** `help xhdfe`, `help xfepout`.
- **R:** `?xhdfe`, `?fixef.xhdfe`, `?predict.xhdfe`; feature tour in `r/examples/`.
- **Python:** `python -m xhdfe` or `xhdfe-help` at the shell, or `xhdfe.help_text()` inside Python.

## Validation, development provenance, and contributions

### Validation and accuracy

Under the default `reghdfe-comparable` tolerance mode, `xhdfe` coefficients,
standard errors, and recovered fixed effects are validated against `reghdfe` at
the same nominal tolerance, subject to the conditioning of each problem.
Certification covers convergence, numerical agreement, backend use, and
cross-frontend parity on tested CPU and CUDA builds. It certifies those builds
and cases, not every possible research design. Users should validate their own
specifications; see [`docs/certification/`](docs/certification/) and
[`DISCLAIMER.md`](DISCLAIMER.md).

### Algorithmic and software provenance

`xhdfe` was developed within an existing ecosystem of HDFE algorithms and
software. The roles of the principal packages were distinct:

| Project | Role in the development of `xhdfe` |
| --- | --- |
| [`reghdfe`](https://github.com/sergiocorreia/reghdfe), by Sergio Correia | Canonical econometric and Stata-behaviour reference for the estimator, defaults, reporting, and validation. |
| [`fixest`](https://github.com/lrberge/fixest), by Laurent Bergé and contributors | Algorithmic and performance reference for accelerated MAP, including the Irons-Tuck route used by the headline `xhdfe` GPU benchmarks. |
| [`pyfixest`](https://github.com/py-econometrics/pyfixest), by Alexander Fischer and contributors | Public code and documentation studied during development, plus interface, DGP, numerical, and cross-language benchmark references. |
| [`FixedEffectModels.jl`](https://github.com/FixedEffects/FixedEffectModels.jl), by Matthieu Gomez and contributors | Reference for diagonally preconditioned LSMR and for cross-language numerical and performance comparisons. |
| [`within`](https://github.com/py-econometrics/within), by Alexander Fischer and Kristof Schröder | Direct algorithmic starting point for `xhdfe`'s graph-preconditioned MLSMR/additive-Schwarz route. |

Fischer and Schröder's public `within` materials and their 2026 mimeo,
*Graph Preconditioning for High-Dimensional Fixed Effects Regression*, were the
direct starting point for `xhdfe`'s graph-preconditioned
MLSMR/additive-Schwarz route. The factor-pair graph decomposition,
bipartite-Laplacian representation, approximate-Cholesky local solves, and
additive-Schwarz preconditioner inside modified LSMR are their underlying
algorithmic contribution and should be cited as such.

The headline GPU benchmarks instead use Irons-Tuck-accelerated MAP, as in
`fixest`; the native CUDA kernels do not implement the Fischer–Schröder graph
preconditioner. `xhdfe` contributes the integrated C++/OpenMP and CUDA
implementation, three language frontends, broader feature surface, execution
and batching choices, routing and fallbacks, diagnostics, precision
certification, and cross-platform validation. These contributions do not alter
the credit for the algorithms on which they build.

### AI-assisted development record

AI-assisted work used public software and documentation in the development
environment, with source inspection, documentation study, and black-box
benchmarking recorded as distinct forms of evidence. The authors retain
authorship and responsibility for `xhdfe` and its documentation.

## Citation

If you use `xhdfe` in academic work, please cite it (see
[`CITATION.cff`](CITATION.cff)):

> Portela, Miguel, and Tiago Tavares. 2026. *xhdfe: High-dimensional fixed
> effects regression via a C++ backend.* Version 2.25.0.
> https://github.com/reisportela/xhdfe-xfe

## License

MIT — see [`LICENSE`](LICENSE). `xhdfe` bundles the Eigen 3.4.0 headers
(primarily MPL-2.0, with parts under BSD-3-Clause and Apache-2.0); see
[`NOTICE`](NOTICE). Autonomous release media also carry the unmodified official
Rcpp 1.1.2 source archive under its upstream GPL (>= 2) license solely as the
R package's offline build dependency; see
[`third_party/RCPP_SOURCE_PROVENANCE.md`](third_party/RCPP_SOURCE_PROVENANCE.md).

## Authors

- **Miguel Portela** — NIPE / Universidade do Minho and BPLIM / Banco de Portugal.
- **Tiago Tavares** — NIPE / Universidade do Minho.

## Acknowledgements

`xhdfe` is a high-performance, `reghdfe`-compatible implementation: the
`reghdfe` estimator, defaults, and reporting conventions remain its reference.
The roles of the prior HDFE packages, the direct Fischer–Schröder algorithmic
credit, and the distinct `xhdfe` contribution are summarized in
[Validation, development provenance, and contributions](#validation-development-provenance-and-contributions).

The AKM layer builds on the worker-firm literature and on
[`pytwoway`](https://github.com/tlamadon/pytwoway) by Thibaut Lamadon and Adam
A. Oppenheimer. Its leave-out decomposition is validated against `pytwoway` and
against Raffaele Saggio's canonical
[`LeaveOutTwoWay`](https://github.com/rsaggio87/LeaveOutTwoWay)
implementation. `xhdfe` also exports its leave-out sample in the
`pytwoway`/`bipartitepandas` format, while leaving structural CRE and BLM models
to `pytwoway`. The Gelbach decomposition is validated against Jonah Gelbach's
`b1x2`.

We thank Paulo Guimaraes, Marta Silva, and Nelson Areal for discussions and
workshop collaboration around earlier versions of the project. We especially
thank Sergio Correia for feedback on benchmarking, tolerances, and
`reghdfe`-comparable validation. We thank Alexander Fischer and Kristof Schröder
for making the `within` materials available and for direct discussion of their
graph-preconditioned fixed-effects solver. All remaining errors are ours.

## References

High-dimensional fixed effects — the `reghdfe` universe `xhdfe` replicates:

- Cornelissen, T. 2008. The Stata command `felsdvreg` to fit a linear model
  with two high-dimensional fixed effects. *Stata Journal* 8(2): 170-189.
- Guimaraes, P., and P. Portugal. 2010. A simple feasible procedure to fit
  models with high-dimensional fixed effects. *Stata Journal* 10(4): 628-649.
- Gaure, S. 2013. OLS with multiple high dimensional category variables.
  *Computational Statistics & Data Analysis* 66: 8-18.
- Correia, S. 2016. `reghdfe`: Estimating linear models with multi-way fixed
  effects. Stata Conference, Stata Users Group.
- Correia, S., P. Guimaraes, and T. Zylkin. 2020. Fast Poisson estimation with
  high-dimensional fixed effects. *Stata Journal* 20(1): 95-115.
- Fischer, A., and K. Schröder. 2026. *Graph Preconditioning for
  High-Dimensional Fixed Effects Regression*. Mimeo.

Worker-firm (AKM) leave-out layer — what `xhdfe` borrows from the `pytwoway`
literature (see [Acknowledgements](#acknowledgements)):

- Abowd, J. M., F. Kramarz, and D. N. Margolis. 1999. High wage workers and
  high wage firms. *Econometrica* 67(2): 251-333. (AKM two-way model.)
- Andrews, M. J., L. Gill, T. Schank, and R. Upward. 2008. High wage workers
  and low wage firms: negative assortative matching or limited mobility bias?
  *Journal of the Royal Statistical Society A* 171(3): 673-697. (AGSU
  homoskedastic correction.)
- Kline, P., R. Saggio, and M. Sølvsten. 2020. Leave-out estimation of
  variance components. *Econometrica* 88(5): 1859-1898. (KSS leave-out
  heteroskedasticity-robust correction and inference.)
- Andrews, I., and A. Mikusheva. 2016. A geometric approach to nonlinear
  econometric models. *Econometrica* 84(3): 1249-1264.
  (Weak-identification q=1 confidence intervals used by KSS.)
- Gelbach, J. B. 2016. When do covariates matter? And which ones, and how
  much? *Journal of Labor Economics* 34(2): 509-543. (Conditional
  decomposition of coefficient movements.)

## Contributing

Contributions, bug reports, and validation cases are welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Platform support

`xhdfe` supports CPU builds on Linux x86-64, Windows x86-64, and macOS Apple
Silicon/Intel. CUDA acceleration is available on Linux with the NVIDIA toolkit;
see the [GPU guide](docs/gpu.md) for installation and verification across Stata,
Python, and R.
