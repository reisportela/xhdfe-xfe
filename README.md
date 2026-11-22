# xhdfe

**Linear regression with multiple high-dimensional fixed effects — in Stata, Python and R, on one fast C++ core.**

`Version 2.34.2` · `License: MIT` · `Stata + Python + R` · `Optional CUDA GPU`

Version 2.34.2 optimises eligible CPU reduced-system fits and parallelises
parts of large regressions, including the preparation of the Stata estimation
sample. See [the release notes](docs/releases/RELEASE_NOTES_2.34.2.20261122.md).

Version 2.34.1 brings to the public release the work of versions 2.32 to 2.34:
the reduced-system route for `reghdfe-comparable` fits on CPU, corrections to
`savefe`, to heterogeneous slopes and to weighted projections, lower memory use,
and CUDA libraries that load on demand. See
[the release notes](docs/releases/RELEASE_NOTES_2.34.1.20261121.md) and
[the validation record](docs/releases/VALIDATION_2.34.1.20261121.md).

Version 2.31.1 added to `xhdfeakm` 1.9.0 an
explicit person-year-centred point estimator and full-model worker projection
inference while retaining the legacy default. See [the option and its statistical limits](docs/akm-projection-inference.md)
and [its validation record](docs/releases/VALIDATION_2.31.1.20261112.md).

This release also includes the [2.31.0 source changes](docs/releases/RELEASE_NOTES_2.31.0.20261111.md),
which correct Student-t inference on Windows, where an x87 precision-control
incompatibility could reject valid fits with `r(198)`. Stata `residuals()` now
stores the fit's own residuals in Fast and Comparable mode, without an extra
absorption. Weighted reghdfe-comparable fits on CPU below the automatic
selector's thresholds now take the bounded MLSMR retry.
Unweighted CPU results are bit-for-bit unchanged, and Fast and Comparable keep
their distinct contracts. See the
[release notes and validation](docs/releases/RELEASE_NOTES_2.31.0.20261111.md)
for the exact tests, known limitations and platform coverage.

---

## What is xhdfe?

`xhdfe` estimates linear models with any number of high-dimensional fixed
effects (HDFE) — the worker–firm, patent–inventor, and multi-way panel designs
common in applied economics. It follows the defaults and reporting conventions
of [`reghdfe`](https://github.com/sergiocorreia/reghdfe) (Correia 2016). Under
the default `reghdfe-comparable` tolerance mode, numerical validation uses
independent finite-sample oracles, with `reghdfe` assessed separately as a comparator. A
Stata command, a Python package, and an R package all call the **same compiled
C++ core**. CPU is the reference backend; an optional CUDA GPU absorber is
available for large problems.

As a historical illustration, the table below reports median estimator-call runtimes for
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

These timings were measured before 24 August 2026 (the table has been in this
README since release 2.25.0, commit `0d54ecae`) with the releases of each
package available then; they predate the current releases listed under
[Algorithmic and software provenance](#algorithmic-and-software-provenance) and
the release of `fereg`, and they are not a comparison with those releases.
The `xhdfe` rows use the speed-oriented `xhdfe-fast` mode; the default
`reghdfe-comparable` mode is somewhat slower but matches `reghdfe` more tightly.
These historical timings do not benchmark every current workflow.
The [2.29.0 validation record](docs/releases/VALIDATION_2.29.0.20261014.md)
covers core24 across eight language, backend and tolerance-mode combinations,
including the four CUDA Comparable cells that 2.28.1 refused. It records
numerical checks, recovery coverage, performance comparisons and remaining
CUDA limitations. Its timings and platform receipts identify the artifacts
tested. The [2.29.1 validation record](docs/releases/VALIDATION_2.29.1.20261014.md)
separately records the corrective release's tests and coverage, and the
[2.30.1 validation record](docs/releases/VALIDATION_2.30.1.20261020.md) records
the integrated fixes and the ready-only performance scope of that release. The
[2.31.0 validation record](docs/releases/VALIDATION_2.31.0.20261111.md) records
that source base's acceptance against `reghdfe` at a tight tolerance and against the
previously validated build.
GPU execution alone does not certify every output.

## Features

- **Multiway HDFE** — any number of absorbed fixed-effect dimensions, plus categorical interactions with two or more components (`a#b#c`).
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
- **Optional GPU** — CUDA absorber with status reporting. Stata and R reject an unfulfilled explicit request; Python callers must inspect GPU-use diagnostics.
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

Restart Stata before testing an updated native plugin. The Windows plugins
include their GNU/OpenMP runtimes, so no MinGW installation or DLL search-path
workaround is needed. macOS plugins include the matching OpenMP runtime.

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

OpenMP is required by production builds on every platform. On macOS,
AppleClang needs a separately configured OpenMP runtime. One source-build
option is Homebrew GCC; use its
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
| [`pyfixest`](https://github.com/py-econometrics/pyfixest) ([pyfixest.org](https://pyfixest.org/)), by Alexander Fischer and contributors | Public code and documentation studied during development, plus interface, DGP, numerical, and cross-language benchmark references. |
| [`FixedEffectModels.jl`](https://github.com/FixedEffects/FixedEffectModels.jl), by Matthieu Gomez and contributors | Reference for diagonally preconditioned LSMR and for cross-language numerical and performance comparisons. |
| [`within`](https://github.com/py-econometrics/within), by Alexander Fischer and Kristof Schröder | Direct algorithmic starting point for `xhdfe`'s graph-preconditioned MLSMR/additive-Schwarz route. |
| [`fereg`](https://github.com/johannes-schmieder/fereg), by Johannes F. Schmieder | Design followed by the reduced-system route (`reduced-pcg`) of version 2.32, as implemented in `fereg` 0.6.0 and described in Schmieder (2026) ([author's copy](https://johannes-schmieder.com/papers/fereg.pdf), [CEPR DP22170](https://cepr.org/publications/dp22170)); benchmark that led to two changes of version 2.31.0 and reference of the performance work of November 2026. |
| [`fereg`](https://github.com/leoliu0/fereg), by Leo Liu | A different package with the same name, for Stata, Python and R on one Rust core. The November 2026 comparison informed the CPU operator and parallel fit changes in xhdfe 2.34.2. |
| [`reghdfejl`](https://github.com/droodman/reghdfejl), by David Roodman | Stata front end to `FixedEffectModels.jl` through his [`julia`](https://github.com/droodman/julia.ado) package; comparison engine in the Stata benchmarks. `julia` is the precedent that `fereg` credits for copying Stata data in parallel, which `xhdfe` adopted in version 2.31.0. |

Comparison packages, current releases checked at their sources on 21 November
2026: `reghdfe` 6.14.1 (8 July 2026), `fixest` 0.14.2 (CRAN, 26 June 2026),
`pyfixest` 0.70.0 (8 November 2026), `FixedEffectModels.jl` 2.2.1 (16 November
2026), `fereg` by Schmieder 0.6.7 (20 November 2026) and `fereg` by Liu 0.5.0
(21 November 2026). `reghdfejl` was benchmarked at 1.1.9, its GitHub release
of 12 October 2026.

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

The reduced-system route of version 2.32 (`reduced-pcg`) follows the design
implemented in `fereg` 0.6.0 and described in Schmieder (2026), *fereg: Fast
high-dimensional fixed-effects regressions in Stata* (CEPR Discussion Paper No.
22170), and cites it as such: with two to five fixed effects, it eliminates
exactly the effect with the most levels and solves the remaining system by
Jacobi-preconditioned conjugate gradients. Which effect is eliminated, the
range of two to five effects, the joint treatment of up to eight columns per
pass, the limit of 1,500 iterations for that stage and the form of its stall
test (the residual must halve within each window of iterations) follow `fereg`.
As Schmieder (2026) notes, the elimination is the block reduction of Somaini and
Wolak (2016), and iterative solutions have precedents in Abowd, Creecy and
Kramarz (2002) and in Mittag (2019). `xhdfe` does not implement what Schmieder
(2026) presents as `fereg`'s distinguishing contribution in this step, the
escalation to stronger preconditioners, including combinatorial multigrid, on
poorly connected data: where the Jacobi-preconditioned iteration stalls, `xhdfe`
hands the fit to its previous routes. `xhdfe`'s part is an independent C++
implementation inside its own routing, acceptance certificate, fallbacks,
diagnostics and three language front ends. `xhdfe` contains no `fereg` source
code; its C++ implementation was written after studying the paper and
`fereg`'s public source.

Two changes of version 2.31.0 followed benchmarks against `fereg`: reading the
Stata data in parallel on qualified hosts, as `fereg` does, and larger automatic
thread teams (one thread per 25,000 rows instead of 500,000), where `fereg` by
default uses all logical processors and sizes the parallel work of each stage
by its amount of work. `fereg` credits the precedent of parallel copying from
Stata to David Roodman's `julia` package (Roodman 2025).

The performance work of November 2026 (versions 2.32 to 2.34) was measured
against successive releases of Schmieder's `fereg`, from 0.6.0 to 0.6.7, on
public benchmark datasets and on large linked panels. In the same month we
studied [`fereg`](https://github.com/leoliu0/fereg) by Leo Liu, whose solver, in
its author's words, "eliminates the largest fixed effect exactly and solves the
remaining system with a preconditioned conjugate-gradient method on
cell-compressed sufficient statistics". We compared it with `xhdfe`'s
reduced-system route on a large panel. The comparison informed the CPU
operator prefetch and the work on Stata input, fit statistics and sample
marking in version 2.34.2. These changes were implemented in xhdfe.

The numerical methods that `xhdfe` exposes are established ones and are cited
as such in the [References](#references): partialling out by the
Frisch–Waugh–Lovell theorem, alternating projections as applied to fixed
effects by Guimarães and Portugal (2010) and Gaure (2013) and as implemented in
`reghdfe` (Correia 2017), conjugate gradients (Hestenes and Stiefel 1952), LSMR
(Fong and Saunders 2011) and the Irons–Tuck acceleration (Irons and Tuck 1969).
The approximate-Cholesky local solves of the graph-preconditioned route are
those that Fischer and Schröder take from the Laplacian-solver literature
(Spielman and Teng 2014; Gao, Kyng and Spielman 2026). Multiway clustering
follows Cameron, Gelbach and Miller (2011), and the Driscoll–Kraay covariance
follows Driscoll and Kraay (1998).

### AI-assisted development record

AI-assisted work used public software and documentation in the development
environment, with source inspection, documentation study, and black-box
benchmarking recorded as distinct forms of evidence. The authors retain
authorship and responsibility for `xhdfe` and its documentation.

## Citation

If you use `xhdfe` in academic work, please cite it (see
[`CITATION.cff`](CITATION.cff)):

> Portela, Miguel, and Tiago Tavares. 2026. *xhdfe: High-dimensional fixed
> effects regression via a C++ backend.* Version 2.34.2.
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
credit, the `fereg` credit for the reduced-system route, and the distinct `xhdfe`
contribution are summarized in
[Validation, development provenance, and contributions](#validation-development-provenance-and-contributions).

The AKM layer builds on the worker-firm literature and on
[`pytwoway`](https://github.com/tlamadon/pytwoway) by Thibaut Lamadon and Adam
A. Oppenheimer. Its leave-out decomposition is validated against `pytwoway` and
against Raffaele Saggio's canonical
[`LeaveOutTwoWay`](https://github.com/rsaggio87/LeaveOutTwoWay)
implementation. `xhdfe` also exports its leave-out sample in the
`pytwoway`/`bipartitepandas` format, while leaving structural CRE and BLM models
to `pytwoway`. The Gelbach decomposition is validated against Jonah Gelbach's
`b1x2`. Its design also drew on the decomposition interface of `pyfixest`
(groups of covariates, tables, plots and bootstrap intervals), on Guimarães,
Portugal and Reis (2026) as the starting point for the decomposition with
high-dimensional fixed effects, on Gaure's `lfe` article in *The R Journal*
(2013) for what is estimable about absorbed effects, and on the literature
that delimits how such a decomposition
may be read: Kohler, Karlson and Holm (2011) for nonlinear probability models,
Fortin, Lemieux and Firpo (2011) for its relation to Oaxaca–Blinder
decompositions, Cameron, Gelbach and Miller (2008) for bootstrap inference with
clustered errors, and Acharya, Blackwell and Sen (2016) and Tingley et al.
(2014) for causal mediation, which the decomposition does not identify.

We thank Paulo Guimaraes and Nelson Areal for discussions and workshop
collaboration around earlier versions of the project.

We are especially grateful to Marta Silva, whose sustained and rigorous testing
has been fundamental to the development and validation of `xhdfe`. Her
contribution goes well beyond reporting individual bugs: she has systematically
tested whether features are present and work as intended, scrutinized numerical
precision and the accuracy of the estimates, and compared `xhdfe` results with
`reghdfe` across demanding empirical specifications. Her careful reports —
including recent evidence on convergence and residual-sum-of-squares
discrepancies in `group()` / `individual()` models — have helped us reproduce
problems, identify their numerical origins, and validate the resulting
solutions. The reliability, precision, and `reghdfe` compatibility of `xhdfe`
have materially benefited from her persistence, judgement, and detailed
feedback.

We especially thank Sergio Correia for feedback on benchmarking, tolerances, and
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
- Correia, S. 2017. Linear models with high-dimensional fixed effects: An
  efficient and feasible estimator. Working paper.
  http://scorreia.com/research/hdfe.pdf (the estimator that `reghdfe`
  implements.)
- Constantine, N., and S. Correia. 2021. `reghdfe`: Stata module for linear and
  instrumental-variable/GMM regression absorbing multiple levels of fixed
  effects. https://ideas.repec.org/c/boc/bocode/s457874.html
- Correia, S., P. Guimaraes, and T. Zylkin. 2020. Fast Poisson estimation with
  high-dimensional fixed effects. *Stata Journal* 20(1): 95-115.
- Bergé, L. R., K. Butts, and G. McDermott. 2026. Fast and user-friendly
  econometrics estimations: The R package `fixest`. arXiv preprint
  arXiv:2601.21749. doi:10.48550/arXiv.2601.21749
- The PyFixest Authors. `pyfixest`: Fast high-dimensional fixed effect
  estimation in Python. https://github.com/py-econometrics/pyfixest
  (https://pyfixest.org/)
- Gomez, M., and contributors. `FixedEffectModels.jl`: Fast estimation of
  linear models with IV and high dimensional categorical variables. Julia
  package. https://github.com/FixedEffects/FixedEffectModels.jl
- Roodman, D. `reghdfejl`: High-dimensional fixed-effect estimation in Stata
  using Julia. https://github.com/droodman/reghdfejl
- Liu, L. `fereg`: Fast high-dimensional fixed-effects regressions for Stata,
  Python and R, on one Rust core. https://github.com/leoliu0/fereg
- Fischer, A., and K. Schröder. 2026. *Graph Preconditioning for
  High-Dimensional Fixed Effects Regression*. Mimeo, draft of June 2026.
  https://github.com/py-econometrics/within-paper
- Fischer, A., and K. Schröder. `within`: Fast solvers for high-dimensional
  fixed effects. Rust crate and Python package.
  https://github.com/py-econometrics/within
- Abowd, J. M., R. H. Creecy, and F. Kramarz. 2002. Computing person and firm
  effects using linked longitudinal employer-employee data. Longitudinal
  Employer-Household Dynamics Technical Paper TP-2002-06, U.S. Census Bureau.
- Mittag, N. 2019. A simple method to estimate large fixed effects models
  applied to wage determinants. *Labour Economics* 61: 101766.
  doi:10.1016/j.labeco.2019.101766
- Somaini, P., and F. A. Wolak. 2016. An algorithm to estimate the two-way
  fixed effects model. *Journal of Econometric Methods* 5(1): 143-152.
  doi:10.1515/jem-2014-0008
- Roodman, D. 2025. Julia as a universal platform for statistical software
  development. *Stata Journal* 25(2): 255-284. doi:10.1177/1536867X251341105
- Schmieder, J. F. 2026. `fereg`: Fast high-dimensional fixed-effects
  regressions in Stata. CEPR Discussion Paper No. 22170. CEPR Press, Paris &
  London. https://cepr.org/publications/dp22170 (author's copy, November 2026:
  https://johannes-schmieder.com/papers/fereg.pdf; software:
  https://github.com/johannes-schmieder/fereg)

Numerical methods and inference:

- Frisch, R., and F. V. Waugh. 1933. Partial time regressions as compared with
  individual trends. *Econometrica* 1(4): 387-401. doi:10.2307/1907330
- Lovell, M. C. 1963. Seasonal adjustment of economic time series and multiple
  regression analysis. *Journal of the American Statistical Association*
  58(304): 993-1010. doi:10.1080/01621459.1963.10480682
- Hestenes, M. R., and E. Stiefel. 1952. Methods of conjugate gradients for
  solving linear systems. *Journal of Research of the National Bureau of
  Standards* 49(6): 409-436. doi:10.6028/jres.049.044
- Irons, B. M., and R. C. Tuck. 1969. A version of the Aitken accelerator for
  computer iteration. *International Journal for Numerical Methods in
  Engineering* 1(3): 275-277. doi:10.1002/nme.1620010306
- Fong, D. C.-L., and M. Saunders. 2011. LSMR: An iterative algorithm for
  sparse least-squares problems. *SIAM Journal on Scientific Computing* 33(5):
  2950-2971. doi:10.1137/10079687X
- Spielman, D. A., and S.-H. Teng. 2014. Nearly linear time algorithms for
  preconditioning and solving symmetric, diagonally dominant linear systems.
  *SIAM Journal on Matrix Analysis and Applications* 35(3): 835-885.
  doi:10.1137/090771430
- Gao, Y., R. Kyng, and D. A. Spielman. 2026. AC(k): Robust solution of
  Laplacian equations by randomized approximate Cholesky factorization. *SIAM
  Journal on Scientific Computing* 48(3): A1643-A1684. doi:10.1137/24M1673577
- Cameron, A. C., J. B. Gelbach, and D. L. Miller. 2011. Robust inference with
  multiway clustering. *Journal of Business & Economic Statistics* 29(2):
  238-249. doi:10.1198/jbes.2010.07136
- Driscoll, J. C., and A. C. Kraay. 1998. Consistent covariance matrix
  estimation with spatially dependent panel data. *Review of Economics and
  Statistics* 80(4): 549-560. doi:10.1162/003465398557825

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

Gelbach decomposition — further sources studied in its design (see
[Acknowledgements](#acknowledgements)):

- Guimarães, P., P. Portugal, and H. Reis. 2026. Labor market sorting in
  Portugal, with a practitioner corner on the decomposition of the regression
  coefficient estimates. *Banco de Portugal Economic Studies* XII(3).
- Gaure, S. 2013. lfe: Linear group fixed effects. *The R Journal* 5(2):
  104-116. doi:10.32614/RJ-2013-031
- Kohler, U., K. B. Karlson, and A. Holm. 2011. Comparing coefficients of
  nested nonlinear probability models. *Stata Journal* 11(3): 420-438.
  doi:10.1177/1536867X1101100306
- Fortin, N., T. Lemieux, and S. Firpo. 2011. Decomposition methods in
  economics. In *Handbook of Labor Economics*, vol. 4A, 1-102.
  doi:10.1016/S0169-7218(11)00407-2
- Cameron, A. C., J. B. Gelbach, and D. L. Miller. 2008. Bootstrap-based
  improvements for inference with clustered errors. *Review of Economics and
  Statistics* 90(3): 414-427. doi:10.1162/rest.90.3.414
- Acharya, A., M. Blackwell, and M. Sen. 2016. Explaining causal findings
  without bias: Detecting and assessing direct effects. *American Political
  Science Review* 110(3): 512-529. doi:10.1017/S0003055416000216
- Tingley, D., T. Yamamoto, K. Hirose, L. Keele, and K. Imai. 2014. mediation:
  R package for causal mediation analysis. *Journal of Statistical Software*
  59(5): 1-38. doi:10.18637/jss.v059.i05

## Contributing

Contributions, bug reports, and validation cases are welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Platform support

`xhdfe` supports CPU builds on Linux x86-64, Windows x86-64, and macOS Apple
Silicon/Intel. CUDA acceleration is available on Linux with the NVIDIA toolkit;
see the [GPU guide](docs/gpu.md) for installation and verification across Stata,
Python, and R.
