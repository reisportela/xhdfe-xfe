# xhdfe

**Linear regression with multiple high-dimensional fixed effects — in Stata, Python and R, on one fast C++ core.**

`Version 2.11.0` · `License: MIT` · `Platform: Linux x86-64 + GCC` · `Optional CUDA GPU`

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
- **Optional GPU** — CUDA absorber with fail-closed semantics (never a silent CPU fallback).

---

## Choose your language

The three packages call the same C++ estimator, so results agree across
languages. Pick your front-end below.

### Stata

**Option A — install a prebuilt release.** Download the distribution ZIP from the
[Releases](https://github.com/reisportela/xhdfe-xfe/releases) page and unzip it. It
bundles the platform plugin. In Stata, point `net install` at the unzipped
folder that contains `xhdfe.pkg` and `stata.toc`:

```stata
net install xhdfe, from("/path/to/unzipped/xhdfe/stata") replace
net install xfe,   from("/path/to/unzipped/xhdfe/stata") replace
```

**Option B — build the plugin from source.** Clone the repo and build the
plugin (Linux + GCC; OpenMP recommended), then add the folder to your `adopath`:

```bash
git clone https://github.com/reisportela/xhdfe-xfe.git
cd xhdfe-xfe
bash stata/tools/build-plugin.sh --linux --openmp     # produces stata/xhdfe.plugin
```

```stata
adopath + "/path/to/xhdfe/stata"
```

Minimal example (public data shipped with Stata):

```stata
sysuse auto, clear
xhdfe price weight length, absorb(rep78)
xhdfe price weight length, absorb(rep78) vce(cluster rep78)

webuse nlswork, clear
xhdfe ln_wage grade age ttl_exp tenure not_smsa south, absorb(idcode year)
xhdfe ln_wage grade age ttl_exp tenure not_smsa south, absorb(idcode year occ_code)
```

`xfe` is a companion command that partials out variables (residualizes against
absorbed fixed effects) using the same core. See `help xhdfe` and `help xfe`.

### Python

Install from the repository (needs CMake and a C++ compiler):

```bash
python -m pip install "git+https://github.com/reisportela/xhdfe-xfe.git"
# or, from a clone:
git clone https://github.com/reisportela/xhdfe-xfe.git && cd xhdfe-xfe && python -m pip install .
```

Optional CUDA build (NVIDIA toolkit required; set the arch for your GPU):

```bash
XHDFE_ENABLE_CUDA=ON CMAKE_CUDA_ARCHITECTURES=90 python -m pip install .
```

Minimal example:

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

### R

Install from GitHub (the package lives in the `r/xhdfe` subdirectory):

```r
# install.packages("remotes")
remotes::install_github("reisportela/xhdfe-xfe", subdir = "r/xhdfe")
```

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
```

The R formula grammar is fixest-style: `y ~ x | fe1 + fe2` for absorbed FEs,
`fe[slope]` / `fe[[slope]]` for heterogeneous slopes, `f1^f2` for a combined
interaction FE, and `| endo ~ inst` for IV. See
[`r/README.md`](r/README.md) for the CUDA build and the platform note, and
`?xhdfe` for the full documentation.

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
> effects regression via a C++ backend.* Version 2.11.0.
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

## Contributing

Contributions, bug reports, and validation cases are welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Platform support

The C++ core, Python bindings, and R package target **Linux x86-64 with GCC**.
CUDA GPU acceleration is optional (requires the NVIDIA toolkit, e.g. `sm_90`).
macOS and Windows are not supported yet.
