# xhdfe

**Linear regression with high-dimensional fixed effects in Stata, Python and R on one C++ core.**

`Version 2.35.1` · `License: MIT` · `Stata + Python + R` · `Optional CUDA GPU`

[Installation and examples](#choose-your-language) ·
[Benchmarks](#benchmarks) ·
[Working paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7271338) ·
[Release notes](docs/releases/RELEASE_NOTES_2.35.1.20261125.md) ·
[2.35.1 validation](docs/releases/VALIDATION_2.35.1.20261125.md) ·
[2.35.0 validation](docs/releases/VALIDATION_2.35.0.20261125.md)

## When to use xhdfe

- **Fit the same linear specification in Stata, Python or R.** The three
  interfaces share a C++ estimator for OLS and 2SLS, with absorbed interactions
  and heterogeneous slopes, weights and multiway clustering. Stata follows
  `reghdfe` conventions. `numthreads(#)` requests OpenMP in the same Stata
  process and `parallel()` is accepted for compatibility. CPU is the reference
  backend and supported models can use NVIDIA CUDA.
  [Syntax comparison](docs/comparison.md) · [GPU guide](docs/gpu.md)
- **Work with group-level outcomes and individual fixed effects.** For example,
  use `group(patent_id) individual(inventor_id)` to fit patent-level outcomes
  with inventor effects. `reghdfe` also supports this group-outcome/individual-FE
  design. The tagged Stata help for Liu's `fereg` and `reghdfejl` lists it as
  unsupported. In Stata `xhdfe`, `savefe` and `savefes()` are unavailable when
  `group()` and `individual()` are specified together.
  [xhdfe help](stata/xhdfe.sthlp) · [reghdfe help](https://scorreia.com/help/reghdfe.html#opt_group_fes) · [Liu 0.5.1 help](https://github.com/leoliu0/fereg/blob/v0.5.1/stata/fereg.sthlp) · [reghdfejl 1.1.9 help](https://github.com/droodman/reghdfejl/blob/v1.1.9/reghdfejl.sthlp)
- **Prepare a worker–firm sample and compute leave-out variance components.**
  In Stata, use `xhdfeconnected` to construct the leave-one-out sample, then
  `xhdfeakm` for AKM plug-in, AGSU and KSS decompositions and inference for
  supported components. KSS variance decomposition extends beyond the
  estimation and FE recovery documented for the two Stata `fereg` distributions.
  Saggio's `LeaveOutTwoWay` also implements the KSS estimator.
  [AKM and KSS walkthrough](docs/akm-kss.md) · [Inference scope](docs/akm-projection-inference.md) · [LeaveOutTwoWay](https://github.com/rsaggio87/LeaveOutTwoWay)
- **Explain several coefficient movements in one Gelbach decomposition.** For
  example, decompose changes in the education and experience coefficients when
  adding covariate blocks and firm effects to a model with common year effects.
  `xhdfe` reports the joint covariance for contrasts across contributions and
  supports a separately declared FE-absorbed target. PyFixest's `decompose()`
  also implements Gelbach, with one focal coefficient per call.
  [xhdfe Gelbach help](stata/xhdfegelbach.sthlp) · [PyFixest decomposition](https://pyfixest.org/reference/estimation.models.feols_.Feols.decompose.html)

## What other packages offer

`xhdfe` focuses on linear OLS and 2SLS. These packages provide options for
other estimators, inference or execution workflows.

| Package | Capabilities outside xhdfe 2.35.1 | Documentation |
| --- | --- | --- |
| `reghdfe` 6.14.1 and the Stata ecosystem | `reghdfe, parallel(#)` runs separate Stata processes using `parallel`. `ivreghdfe` 1.1.4 provides LIML and two-step GMM with absorbed effects. `ppmlhdfe` provides Poisson PML with separation checks. | [reghdfe help](https://scorreia.com/help/reghdfe.html) · [ivreghdfe](https://github.com/sergiocorreia/ivreghdfe#comparison-with-other-commands) · [ppmlhdfe help](https://scorreia.com/help/ppmlhdfe.html) |
| Johannes F. Schmieder's `fereg` 0.6.8 (Stata) | `feppml` for Poisson PML, `memory(low)` and `poolsize(#)` for low-memory work blocks, and explicit combinatorial multigrid (CMG) preconditioning. `solver(cg, reduce(eliminate) precondition(cmg))` is an experimental route for two categorical intercept effects. | [Tagged README](https://github.com/johannes-schmieder/fereg/blob/v0.6.8/README.md) |
| Leo Liu's `fereg` 0.5.1 (Stata, Python and R) | One Rust core with Poisson PML and fixed-effects logit. The Stata `felogit` provides incidental-parameter bias corrections and average partial effects. The linear estimator offers Conley spatial standard errors. | [Tagged README](https://github.com/leoliu0/fereg/blob/v0.5.1/README.md) |
| `fixest` 0.14.2 (R) | Fixed-effects GLMs via `feglm()`, cohort-specific staggered DiD with `sunab()` and spatial inference with Conley covariance. | [GLMs](https://lrberge.github.io/fixest/reference/feglm.html) · [Staggered DiD](https://lrberge.github.io/fixest/reference/sunab.html) · [Conley covariance](https://lrberge.github.io/fixest/reference/vcov_conley.html) |
| PyFixest 0.70.0 (Python) | Dedicated difference-in-differences estimators such as `did2s()`, fixed-effects GLMs and wild cluster bootstrap tests for eligible OLS fits. | [0.70.0 documentation](https://pyfixest.org/changelog.html) · [Estimation methods](https://pyfixest.org/reference/) |
| `FixedEffectModels.jl` 2.2.1 (Julia) and `reghdfejl` 1.1.9 (Stata) | Julia-native model fitting and experimental Apple Metal demeaning in single precision. The Stata wrapper calls Julia and offers parallel pairs-bootstrap standard errors through `vce(bootstrap, ...)`. | [GPU methods in 2.2.1](https://github.com/FixedEffects/FixedEffectModels.jl/blob/v2.2.1/README.md#gpus) · [reghdfejl 1.1.9 help](https://github.com/droodman/reghdfejl/blob/v1.1.9/reghdfejl.sthlp) |

PyFixest and `FixedEffectModels.jl` also offer CUDA demeaning for supported
models. [PyFixest backends](https://pyfixest.org/changelog.html) · [Julia GPU methods](https://github.com/FixedEffects/FixedEffectModels.jl/blob/v2.2.1/README.md#gpus)

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
- **Optional GPU** — An unfulfilled explicit CUDA request raises an error by
  default in Stata, Python and R. Check `e(gpu_used)`, `gpu_used_` or `gpu_used`
  to confirm GPU execution. [Supported paths and diagnostics](docs/gpu.md)
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

One command installs the estimator, the portable CPU plugin for the current OS, and
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
| `xhdfegpu` | Builds hybrid CPU/CUDA plugins on Linux and reports their build targets (`help xhdfegpu`). |

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

To optimise the CPU path for this host and target an H100 explicitly, use
this command in place of the default.

```stata
xhdfegpu, cputarget(native) arch(90)
```

The default hybrid build uses a portable CPU target with a clean environment.
`cputarget(native)` optimises for the build host. Keep portable builds in a
`PLUS` shared across nodes with
different CPUs. The optional `arch()` selects the CUDA device target
independently. The installer verifies the staged plugins before replacement.

Open a new Stata session and check the installed build and GPU execution.

```stata
xhdfegpu, status
sysuse auto, clear
xhdfe price weight length, absorb(rep78) gpubackend(cuda)
display e(gpu_used)          // 1
display "`e(gpu_status)'"     // used
```

Status shows the resolved path on disk and reports separately when metadata
comes from a module bound in the current session. Legacy plugins may report
build fields as `unrecorded`.

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
- **R:** `?xhdfe`, `?fixef.xhdfe`, `?predict.xhdfe`; public examples in `?xhdfe_examples` and a feature tour in `r/examples/`.
- **Python:** `python -m xhdfe` or `xhdfe-help` at the shell, or `xhdfe.help_text()` inside Python.

## Benchmarks

These results use the 25 November 2026 measurement snapshot for xhdfe 2.35.0,
including earlier sealed runs with the same inputs, settings and timing
boundary. The xhdfe timings use a Linux build of the 2.35.0 source with OpenMP,
host-native CPU optimisation and CUDA for the H100. Each series uses three
fresh processes. The host had an NVIDIA H100 NVL and a 16-processor Stata/MP
licence. xhdfe, both `fereg`
packages, PyFixest, `fixest` and the Julia engines were configured for 24
native threads where supported. Stata's native `reghdfe` used its 16 licensed
processors. Direct Julia CPU used 24 Julia threads and 24 BLAS threads. The
observer recorded CPU work on up to 48 tasks within a sampling interval
across the two pools. The xhdfe runs used `reghdfe-comparable` mode.

Times are in seconds. The clock starts with the input in memory and after any
instrumented start-up or synthetic warm-up. It covers the estimation command
or estimator-specific preparation, fit and table formatting where applicable.
The Stata `reghdfejl` clock includes the Stata–Julia interface. PyFixest H100 uses
Torch CUDA in FP64 with compilation disabled during the measurement. Direct
`FixedEffectModels.jl` H100 uses CUDA FP64 and warms the kernels on synthetic
data at the size of the estimation sample. Round numbers identify repetitions
within each series. Comparisons across separate measurement blocks are unpaired.

**Bold medians** have three completed fits and passed the benchmark's full
numerical checks. The xhdfe CPU fit supplies the reference for sample, FE
cardinalities, identified slopes, standard errors, covariance and relevant
degrees of freedom. The comparison limits are `1e-8` for scaled coefficients
and `1e-7` for relative standard errors and scaled covariance. The covariance
error for entry `j,k` is `abs(V[j,k] - Vref[j,k]) / sqrt(Vref[j,j] * Vref[k,k])`
when the reference diagonal scale is positive.
**†** marks completed raw timings with a numerical difference. The saturated
design marked **§** uses a separate check of the identified slopes.

### Ready worker–firm panel

The full source has **47,569,720** rows. The estimation sample has **46,156,187**
observations and **5,074,679** worker, **610,089** firm and **26** year FE levels.

| Estimator | Run 1 | Run 2 | Run 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| `xhdfe` 2.35.0 (Stata, CPU) | 8.363 | 8.170 | 8.409 | **8.363** |
| `xhdfe` 2.35.0 (Stata, H100) | 6.015 | 5.910 | 5.894 | **5.910** |
| Liu `fereg` 0.5.1 (Stata, CPU) | 8.680 | 8.715 | 8.719 | **8.715** |
| Schmieder `fereg` 0.6.8 (Stata, CPU) | 22.727 | 22.782 | 22.588 | **22.727** |
| `reghdfe` 6.14.1 (Stata, CPU) | 1,082.798 | 1,068.244 | 1,082.791 | **1,082.791** |
| PyFixest 0.70.0 (Python, CPU) | 61.735 | 60.333 | 62.405 | **61.735** |
| PyFixest 0.70.0 (Python, H100) | 39.716 | 40.134 | 39.439 | **39.716** |
| `fixest` 0.14.2 (R, CPU) | 345.985 | 348.110 | 340.338 | **345.985** |
| `fixest` 0.14.2 (R, H100) | — | — | — | No GPU backend |
| `FixedEffectModels.jl` 2.2.1 (Julia, CPU) | 241.222 | 248.194 | 243.924 | **243.924** |
| `FixedEffectModels.jl` 2.2.1 (Julia, H100 FP64) | 25.272 | 24.705 | 24.414 | **24.705** |
| `reghdfejl` 1.1.9 (Stata–Julia, CPU) | 445.668 | 446.542 | 447.348 | **446.542** |
| `reghdfejl` 1.1.9 (Stata–Julia, H100 FP32) | 108.728 † | 112.926 † | 114.527 † | 112.926 † |

The `reghdfejl` H100 row reports three raw clocks for its public FP32 absorption
path. Its largest relative standard-error difference is **5.07%**, and its
largest scaled covariance difference is **9.88%**, against xhdfe CPU. It has
zero qualifying FP64 runs for a comparable median. The documented `fixest`
0.14.2 `feols()` interface provides a CPU estimator. [fixest API](https://lrberge.github.io/fixest/reference/feols.html)

The Ready xhdfe CPU series used `reducedpcg(on)` and `reducedpcgfine(auto)`,
with method 8 and an observed fine-block factor of 8 at both 16 and 24 threads.
At the 16-processor CPU budget, xhdfe took 9.568, 11.145, 10.218 s, a median of
**10.218 s**. The `reghdfe` median at 16 processors was
**1,082.791 s**. The two series were measured in separate blocks.

### The 24 core24s1 designs

Each row shows source rows → reference-sample observations, followed by the
number of unique levels in each absorbed dimension in that estimation sample.
For the Marta design, the source and retained data are long-format rows and
the estimation units are groups. The second AKM regression uses the designated
1% extract. Its employer intercept and heterogeneous slopes share the
`NPC_FIC` dimension, which is counted once. The columns use xhdfe 2.35.0,
Liu `fereg` 0.5.1, Schmieder `fereg` 0.6.8, `reghdfe` 6.14.1 and
PyFixest 0.70.0, with the devices shown.

| Design and estimation sample | xhdfe CPU | xhdfe H100 | Liu | Schmieder | reghdfe | PyFixest CPU | PyFixest H100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **credit**<br>516,810 → 516,154<br>`id1` 13,290 · `id2` 140 | **0.253** | **0.424** | **0.263** | **0.401** | **0.950** | **1.310** | **0.321** |
| **credit2**<br>19,094 → 19,075<br>`id1` 582 · `id2` 140 | **0.196** | **0.386** | **0.226** | **0.208** | **0.384** | **0.802** | **0.086** |
| **directors**<br>525,012 → 31,230<br>`id1` 12,926 · `id2` 12,085 | **0.376** | **0.566** | **0.312** | **0.823** | **1.158** | **1.226** | 0.966 † |
| **enron**<br>367,662 → 346,494<br>`id1` 25,286 · `id2` 25,286 | **0.332** | **0.575** | **5.824** | **0.568** | **2.185** | **1.760** | **0.470** |
| **github**<br>548,843 → 540,674<br>`id1` 21,531 · `id2` 21,348 · `id3` 1,847 | **1.171** | **0.781** | **0.569** | **1.710** | **14.797** | 2.741 † | 1.422 † |
| **patents**<br>500,008 → 176,115<br>`id1` 49,348 · `id2` 67,524 | **0.448** | **0.644** | **1.188** | **1.241** | **2.883** | **2.136** | 0.978 † |
| **schools**<br>413,444 → 413,248<br>`id1` 206,624 · `id2` 11,861 · `id3` 4 | **0.550** | **0.550** | **0.493** | **0.898** | **5.924** | **2.172** | **0.832** |
| **soccer**<br>73,487 → 73,487<br>`id1` 528 · `id2` 521 | **0.200** | **0.382** | **0.223** | **0.213** | **0.367** | **1.120** | **0.097** |
| **synthetic-assortative**<br>499,155 → 465,221<br>`id1` 99,503 · `id2` 78,875 · `year` 10 | **0.572** | **0.675** | **0.784** | **1.036** | **11.714** | **2.860** | 0.980 † |
| **synthetic-complete**<br>500,000 → 500,000<br>`id1` 1,000 · `id2` 500 | **0.258** | **0.426** | **0.220** | **0.258** | **0.689** | **1.216** | **0.274** |
| **synthetic-uniform-easy**<br>500,000 → 432,350<br>`id1` 148,357 · `id2` 20 | **0.243** | **0.487** | **0.242** | **0.453** | **0.933** | **1.423** | **0.419** |
| **synthetic-uniform-hard**<br>500,000 → 480,287<br>`id1` 112,743 · `id2` 112,708 | **0.315** | **0.496** | **0.349** | **0.691** | **1.718** | **1.623** | **0.454** |
| **synthetic-uniform-harder**<br>500,000 → 317,745<br>`id1` 118,120 · `id2` 118,466 | **0.430** | **0.570** | **0.606** | **0.864** | **2.330** | **2.568** | **0.550** |
| **synthetic-zigzag** §<br>10,002 → 10,002<br>`id1` 5,000 · `id2` 5,001 | 0.227 § | 0.390 § | 0.208 § | Refused | 3.875 † | Failed | Failed |
| **workers**<br>504,315 → 451,626<br>`id1` 20,400 · `id2` 169,111 · `id3` 3,814 · `id4` 3 | **1.316** | **0.934** | **1.383** | **1.794** | **32.246** | 5.007 † | 1.557 † |
| **main_95_21_ready**<br>47,569,720 → 46,156,187<br>`w_id` 5,074,679 · `f_id` 610,089 · `year` 26 | **8.363** | **5.910** | **8.715** | **22.727** | **1,082.791** | **61.735** | **39.716** |
| **simulated_panel**<br>173,163,263 → 173,163,263<br>`worker_id` 20,000,000 · `firm_id` 1,000,000 · `occupation_id` 500 · `year` 15 | **12.603** | **12.640** | **8.623** | **29.916** | Censored &gt;420 s | **108.622** | GPU OOM |
| **akm_v02_firstreg**<br>57,821,050 → 55,947,171<br>`idtrab` 5,948,793 · `ano` 36 · `NPC_FIC` 799,265 | **10.733** | **9.170** | **12.988** | **47.108** | Censored &gt;420 s | **94.394** | **54.690** |
| **pf_difficult_10m_3fe**<br>10,000,000 → 10,000,000<br>`indiv_id` 1,000,000 · `firm_id` 43,478 · `year` 10 | **6.553** | **12.570** | **1.824** | **36.396** | Censored &gt;420 s | 13.175 † | 30.295 † |
| **pf_simple_10m_2fe**<br>10,000,000 → 10,000,000<br>`indiv_id` 1,000,000 · `year` 10 | **1.251** | **1.659** | **0.623** | **2.436** | **26.448** | **5.851** | **5.101** |
| **pf_simple_10m_3fe**<br>10,000,000 → 10,000,000<br>`indiv_id` 1,000,000 · `firm_id` 43,478 · `year` 10 | **1.735** | **1.925** | **1.112** | **3.242** | **68.592** | **8.142** | **5.674** |
| **pf_difficult_10m_2fe**<br>10,000,000 → 10,000,000<br>`indiv_id` 1,000,000 · `year` 10 | **1.249** | **1.622** | **0.605** | **2.488** | **24.731** | **6.341** | **5.883** |
| **akm_v02_secondreg_small** (1% sample)<br>580,643 → 539,327<br>`idtrab` 57,466 · `ano` 36 · `NPC_FIC` 52,732 | **1.254** | **1.157** | **27.480** | Censored &gt;420 s | 177.782 † | 13.628 † | No support |
| **marta_group_individual_3m**<br>3,000,000 rows → 1,104,000 groups (2,760,000 retained rows)<br>`p_ntrab` 345,000 · `ntrab` 24,576 | **6.961** | **1.607** | Refused | Refused | **87.637** | No support | No support |

**§** `synthetic-zigzag` is saturated, with zero mathematical residual degrees
of freedom. Its rational oracle checks the two identified slopes only. The
default `reghdfe` fits return different slopes from that oracle and their
completed clocks are marked †. PyFixest fails while computing adjusted R².

**Censored** means the estimation clock reached the recorded 420 s limit.
**GPU OOM** records the first out-of-memory attempt on the full simulated panel
and two subsequent repetitions omitted after the failure. **No support** marks
an unsupported specification or backend. **Refused** marks an explicit
estimator rejection, and **Failed** marks an unsuccessful fit. Each displayed
median uses three completed clocks.

[xhdfe 2.35.0 release validation](docs/releases/VALIDATION_2.35.0.20261125.md)

## Validation, development provenance, and contributions

### Validation and accuracy

The default `reghdfe-comparable` mode is evaluated against independent
finite-sample oracles and `reghdfe` on the designs covered by the validation
suite. The 25 November 2026 validation covers the new slope solver, savefe cache and CPU block controls and records an inherited CUDA covariance limit.
[2.35.0 validation](docs/releases/VALIDATION_2.35.0.20261125.md)

The 22 November 2026 validation documents the earlier 2.34.2 and 2.34.1
comparison. The current release's measurement snapshot is in the
[benchmarks](#benchmarks) above.
[2.34.2 validation](docs/releases/VALIDATION_2.34.2.20261122.md) ·
[Certification scope](docs/certification/) · [Disclaimer](DISCLAIMER.md)

Earlier release records document the CPU reduced-system route and recovery
changes in 2.34.1, the scope of worker projection inference in 2.31.1, and
Windows inference and residual storage fixes in 2.31.0.
[2.34.1 release notes](docs/releases/RELEASE_NOTES_2.34.1.20261121.md) ·
[2.34.1 validation](docs/releases/VALIDATION_2.34.1.20261121.md) ·
[2.31.1 validation](docs/releases/VALIDATION_2.31.1.20261112.md) ·
[2.31.0 release notes](docs/releases/RELEASE_NOTES_2.31.0.20261111.md) ·
[2.31.0 validation](docs/releases/VALIDATION_2.31.0.20261111.md)

The earlier worker–firm timings remain in the README archive.
[Archived 2026 benchmark](https://github.com/reisportela/xhdfe-xfe/blob/1842265accd347ee5c462c07f73f80d3a0631376/README.md#L48-L72)

### Algorithmic and software provenance

`xhdfe` was developed within an existing ecosystem of HDFE algorithms and
software. The roles of the principal packages were distinct:

| Project | Role in the development of `xhdfe` |
| --- | --- |
| [`reghdfe`](https://github.com/sergiocorreia/reghdfe), by Sergio Correia | Canonical econometric and Stata-behaviour reference for the estimator, defaults, reporting, and validation. |
| [`fixest`](https://github.com/lrberge/fixest), by Laurent Bergé and contributors | Algorithmic and performance reference for accelerated MAP, including the Irons-Tuck route used by the historical `xhdfe` GPU benchmarks. |
| [`pyfixest`](https://github.com/py-econometrics/pyfixest) ([pyfixest.org](https://pyfixest.org/)), by Alexander Fischer and contributors | Public code and documentation studied during development, plus interface, DGP, numerical, and cross-language benchmark references. |
| [`FixedEffectModels.jl`](https://github.com/FixedEffects/FixedEffectModels.jl), by Matthieu Gomez and contributors | Reference for diagonally preconditioned LSMR and for cross-language numerical and performance comparisons. |
| [`within`](https://github.com/py-econometrics/within), by Alexander Fischer and Kristof Schröder | Direct algorithmic starting point for `xhdfe`'s graph-preconditioned MLSMR/additive-Schwarz route. |
| [`fereg`](https://github.com/johannes-schmieder/fereg), by Johannes F. Schmieder | Design followed by the reduced-system route (`reduced-pcg`) of version 2.32, as implemented in `fereg` 0.6.0 and described in Schmieder (2026) ([author's copy](https://johannes-schmieder.com/papers/fereg.pdf), [CEPR DP22170](https://cepr.org/publications/dp22170)); benchmark that led to two changes of version 2.31.0 and reference of the performance work of November 2026. |
| [`fereg`](https://github.com/leoliu0/fereg), by Leo Liu | A different package with the same name, for Stata, Python and R on one Rust core. The November 2026 comparison informed the CPU operator and parallel fit changes in xhdfe 2.34.2. |
| [`reghdfejl`](https://github.com/droodman/reghdfejl), by David Roodman | Stata front end to `FixedEffectModels.jl` through his [`julia`](https://github.com/droodman/julia.ado) package; comparison engine in the Stata benchmarks. `julia` is the precedent that `fereg` credits for copying Stata data in parallel, which `xhdfe` adopted in version 2.31.0. |

The November 2026 Stata comparisons used `reghdfejl` 1.1.9. The local
22 November comparison used the Stata distributions of Schmieder's `fereg`
0.6.8 and Liu's `fereg` 0.5.0.

Fischer and Schröder's public `within` materials and their 2026 mimeo,
*Graph Preconditioning for High-Dimensional Fixed Effects Regression*, were the
direct starting point for `xhdfe`'s graph-preconditioned
MLSMR/additive-Schwarz route. The factor-pair graph decomposition,
bipartite-Laplacian representation, approximate-Cholesky local solves, and
additive-Schwarz preconditioner inside modified LSMR are their underlying
algorithmic contribution and should be cited as such.

The historical GPU benchmarks used Irons-Tuck-accelerated MAP, as in `fixest`.
The Fischer–Schröder graph preconditioner runs on CPU. `xhdfe` integrates these
methods into its C++/OpenMP and CUDA implementation, with Stata, Python and R
frontends, worker–firm and Gelbach workflows, and its own batching, routing,
fallbacks, diagnostics and cross-platform validation.

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
> effects regression via a C++ backend.* Version 2.35.1.
> https://github.com/reisportela/xhdfe-xfe

> Portela, Miguel, and Tiago Tavares. *xhdfe: Fast High-Dimensional
> Fixed-Effects Estimation in Stata* (August 11, 2026).
> [SSRN 7271338](https://ssrn.com/abstract=7271338) ·
> [DOI 10.2139/ssrn.7271338](https://doi.org/10.2139/ssrn.7271338).

## License

MIT — see [`LICENSE`](LICENSE). `xhdfe` bundles the Eigen 3.4.0 headers
(primarily MPL-2.0, with parts under BSD-3-Clause and Apache-2.0); see
[`NOTICE`](NOTICE). Autonomous release media also carry the unmodified official
Rcpp 1.1.2 source archive under its upstream GPL (>= 2) license solely as the
R package's offline build dependency; see
[`third_party/RCPP_SOURCE_PROVENANCE.md`](third_party/RCPP_SOURCE_PROVENANCE.md).

## Authors

- **Miguel Portela** — NIPE / Universidade do Minho and BPLIM / Banco de Portugal.
  [Website](https://reisportela.github.io/) ·
  [miguel.portela@eeg.uminho.pt](mailto:miguel.portela@eeg.uminho.pt)
- **Tiago Tavares** — Department of Economics, ISEG – Lisbon School of Economics
  & Management, University of Lisbon.
  [Website](https://www.tgstavares.com/) ·
  [tgstavares@iseg.ulisboa.pt](mailto:tgstavares@iseg.ulisboa.pt)

## Acknowledgements

We are grateful to Paulo Guimarães for encouraging us to develop `xhdfe` and
for his support throughout its development. We also thank him and Nelson
Areal for discussions and workshop collaboration on earlier versions.

Miguel Portela thanks the [Banco de Portugal Microdata Research Laboratory
(BPLIM)](https://bplim.bportugal.pt/) for its support and for the favourable
research environment in which he worked on `xhdfe`.

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

We especially thank Sergio Correia for developing [`reghdfe`](https://scorreia.com/software/reghdfe/)
and for making his benchmark datasets available in
[Datasets for benchmarking high-dimensional fixed-effect algorithms and their implementations](https://scorreia.com/data/hdfe/index.html).
We are grateful for his feedback on benchmarking, tolerances, and
`reghdfe`-comparable validation.

We thank Alex Fischer for taking the time to discuss our work and for sharing
[PyFixest](https://pyfixest.org/) with us.

We thank Alexander Fischer and Kristof Schröder for sharing their paper,
[*Graph Preconditioning for High-Dimensional Fixed Effects Regression*](https://github.com/py-econometrics/within-paper).
We also thank them for making the `within` materials available and for discussing
their graph-preconditioned fixed-effects solver with us.

All remaining errors are ours.

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
