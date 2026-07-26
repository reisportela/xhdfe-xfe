# xhdfe

High-dimensional fixed effects (HDFE) estimator in C++ backend and a Stata wrapper (`xhdfe`).

TASKS:
- test the accuracy of the estimated FEs when we use the group option

## Disclaimer

This package is provided as a proof of concept.
While the estimates, standard errors, and fixed effects coincide with those
produced by Stata's `reghdfe` in the tested cases, the software has not undergone
systematic validation or stress testing.

The code is provided "as is", without any warranty. Users are responsible for
verifying the correctness and suitability of the results for their own research.

## Motivation

The project focuses on:

1. Replicating the core linear HDFE workflow and postestimation behavior used in Stata's reghdfe.
2. Delivering large speedups on large datasets through a C++ backend.

The design and defaults are inspired by and draw heavily on:

- `reghdfe` (Stata): https://github.com/sergiocorreia/reghdfe
- `FixedEffectModels.jl` (Julia): https://github.com/FixedEffects/FixedEffectModels.jl
- `fixest` (R): https://github.com/lrberge/fixest
- `pyfixest` (Python): https://github.com/py-econometrics/pyfixest

## What's in this repo

- C++ core (v11 build target): `include/`, `src/`
- Stata package and plugin (v11 backend): `stata/`
- Python package and bindings (v11): `xhdfe/`, `python/`
- Tests and validation harness: `test/tests_xhdfe/`
- Notebooks and examples: `notebook/`, `data/`
- Benchmarks and timing notebooks/scripts: `benchmarks/`
- Bundle artifacts: `xhdfe_v11_full.zip`

## Active code map (where the live xhdfe code is)

The single place to EDIT the estimator is the repository root:

- **Canonical C++ core: `src/` + `include/`** (plus `python/py_hdfe_v11.cpp`
  for the binding). All algorithm work happens here.
- **Mirrors — never edit by hand**: `stata/src` + `stata/include` (Stata
  plugin build inputs) and `share/xhdfe_estimation_cpp/{src,include}` +
  `share/xhdfe_estimation_cpp/stata/{src,include}` (distribution source
  mirrors). They are byte-for-byte copies of the canonical files, refreshed
  by copying and verified with:

  ```bash
  bash tools/check_cpp_core_alignment.sh
  ```

- **Active Stata artifacts**: `stata/xhdfe.ado`, `stata/xhdfe.sthlp`,
  `stata/xhdfe.plugin`, `stata/xfe.ado`, `stata/xfe.sthlp`,
  `stata/xfe.plugin`. The plugins are produced from the `stata/src` mirror by
  `stata/tools/build-plugin.sh` / `build-xfe-plugin.sh` (on this workstation:
  OpenMP + CUDA `sm_90`; see CLAUDE.md). Files named
  `stata/*.plugin.bak_*` are local safety copies, not active artifacts.
- **Active Python module build dirs**: `build/` (CPU, Release,
  `-march=native`) and `build_cuda/` (CUDA `sm_90`, Release,
  `-march=native`). These are the DEFAULTS that benchmark scripts and
  notebooks load (`HDFE_BUILD_DIR_CPU` / `HDFE_BUILD_DIR_GPU` override).
  After any accepted C++ change, rebuild both so the defaults track the
  current code. Every other `build_*` directory (dated names such as
  `build_audit_*`, `build_opt2_*`, `build_benchmark_*`,
  `build_release_*_dist_*`) is a frozen experiment/distribution snapshot and
  is NOT kept current.

Quick check that a compiled artifact contains the current optimization layer
(10jun2026): `strings <artifact> | grep -c XHDFE_UNCAP_LARGE_N` must print
≥ 1 for `build/py_hdfe_v11*.so`, `build_cuda/py_hdfe_v11*.so`,
`stata/xhdfe.plugin`, and `stata/xfe.plugin`.

## AKM + leave-out (KSS) variance decomposition and Gelbach companion (July 25, 2026)

`xhdfe` now estimates the two-way AKM model on the leave-out connected set
and reports the variance decomposition in three flavours — **plug-in, AGSU
(homoskedastic), KSS (heteroskedastic leave-out)** — with exact and JLA
leverage paths (deterministic random streams; FP64-equivalent reductions
across thread counts/backends), across the three
front-ends on the same compiled core:

- Python: `xhdfe.akm.akm_kss(...)` (+ `subsampling_diagnostic`, interop
  exports, `compute_se=True` component standard errors, `Z=` KSS lincom
  projections, `gpu=True` CUDA solves — 11.4x at 2M rows on an H100),
  `xhdfe.gelbach.decompose(...)` (Gelbach 2016 conditional decomposition
  with absorbed FE blocks; homoskedastic/robust/cluster SEs, aweights and
  fweights, all matching `b1x2` exactly), an opt-in `gpu=True` request with
  truthful backend diagnostics, plus reporting-only focal selection,
  `gelbach.tidy(...)` for signed shares and joint-covariance intervals and
  `gelbach.contrast(...)` for joint-covariance linear combinations. The
  release 2.21.0 adds `gelbach.bootstrap(...)`,
  `gelbach.etable(...)`, `gelbach.waterfall_data(...)` and
  `gelbach.coefplot(...)`; run
  `xhdfe-help gelbach` for the complete estimand, inference, result-schema,
  example, and limitation reference;
- Stata: `xhdfeakm` (options `se`, `gpu`; see `stata/xhdfeakm.sthlp`) and
  `xhdfegelbach` (weights, `focal()`, signed `shares()`, `gpu`, and the
  explicit absorbed-target estimand), plus the companion commands
  `xhdfegelbachbootstrap`, `xhdfegelbachetable` and
  `xhdfegelbachcoefplot`; see their `.sthlp` files;
- R: `xhdfe_akm_kss()` / `xhdfe_akm_leave_out_set()` / `xhdfe_gelbach()`,
  including opt-in `gpu = TRUE`, with `xhdfe_gelbach_tidy()` and
  `xhdfe_gelbach_contrast()`. The package also exports
  `xhdfe_gelbach_bootstrap()`, `xhdfe_gelbach_etable()`,
  `xhdfe_gelbach_waterfall_data()` and `xhdfe_gelbach_coefplot()`; see
  `?xhdfe_gelbach`, `?xhdfe_gelbach_tidy`, and
  `?xhdfe_gelbach_contrast`.

In `xhdfegelbach 1.5.0` (shared package `2.21.0.20260725`), all three
frontends additionally support common HDFE, selectable two-way connectivity
diagnostics, retained-sample provenance, conservative weak-denominator and
between-FE variance gates, full-refit pairs bootstrap, and identity-preserving
tables and plots. The covariance contract includes `Cov(delta, b_base)`, so
`shares(base)` / `share = "base"` report the full ratio delta-method SE and
interval; the descriptive `base_fixed` convention remains available and
unchanged. See
[`RELEASE_NOTES_2.21.0.20260725.md`](RELEASE_NOTES_2.21.0.20260725.md)
for the complete versioned change surface and explicit deferred extensions.

Numerical semantics follow Saggio's LeaveOutTwoWay (the canonical KSS
implementation), matched at machine precision; JLA results are bit-identical
across Stata, Python and R under the same seed. Validation:
`VALIDATE_AKM_KSS.py`, `VALIDATE_GELBACH.py`; benchmarks:
`benchmarks/akm_kss/`; worked example: `New_Features/akm_kss_vignette.md`; design
log: `New_Features/PLAN_AKM_KSS.md`, `New_Features/PROGRESS_AKM_KSS.md`.

## Update (June 16, 2026)

- New in `xhdfe 2.11.0` / `xfe 1.10.0`: in the default
  `tolerancemode(reghdfe-comparable)`, ill-conditioned / poorly connected
  multi-way fixed-effect graphs now hand off automatically from the
  alternating-projections accelerator (which would otherwise stall and fall back
  to thousands of plain sweeps) to a stable per-column conjugate-gradient solve on
  the symmetric demeaning operator. On such designs this cuts runtime by roughly an
  order of magnitude at equal or better precision — e.g. the 540k-observation
  3-way `github` benchmark: C++ CPU ~27s→~5s, C++ CUDA ~4.6s→~0.9s, Stata CPU
  ~54s→~7s, Stata CUDA ~5.5s→~0.85s; coefficients match `reghdfe` to ~1e-11.
- Well-conditioned datasets are unaffected and numerically identical to 2.10.0
  (verified bit-identical at single thread); no feature, precision, default-output
  or convergence change on any CPU/GPU path, and `tolerancemode(xhdfe-fast)` is
  unchanged.
- Stata plugins rebuilt (OpenMP + CUDA `sm_90`) and GPU-validated
  (`e(gpu_used)==1`, `e(gpu_backend)=="cuda"`, `e(gpu_status)=="used"`).

## Latest update (June 10, 2026)

- Large-n optimization layer accepted (see
  `AUDIT_OPTIMIZATION_CLAUDE_20260610.md`): −22…−30% on the two large
  benchmarks across C++ CPU, C++ CUDA, and Stata plugin CPU, with no
  feature/precision/convergence change; the 15 small benchmark datasets keep
  the previous code path byte-for-byte (n ≥ 4 194 304 gate).
- Stata plugins rebuilt (OpenMP + CUDA `sm_90`) and validated end-to-end
  (CPU + real-GPU runs, savefe/fweight/group/tolerancemode smokes).
- Default tolerance mode is `reghdfe-comparable` since 2.7.0 (parity work,
  09-10jun2026); iteration counts are higher than pre-2.7.0 records by
  design.

## Update (June 12, 2026)

- At the 16jun2026 release, the Stata package versions were
  `xhdfe 2.11.0 16jun2026` and
  `xfe 1.10.0 16jun2026`.
- CUDA plugin builds now require real CUDA execution when `gpubackend(cuda)` is
  requested; unavailable GPU execution reports an error instead of silently
  returning CPU output.
- The 12jun2026 xhdfe build includes the current C++ absorption, OpenMP plugin,
  and GPU convergence
  fixes used in the Sergio benchmark reruns.

## Build (C++ library, v11)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Build / install (Python package, v11)

```bash
python -m pip install .
```

This builds the existing C++/pybind11 module and installs it as
`xhdfe.py_hdfe_v11`. The preferred import is:

```python
import xhdfe

reg = xhdfe.HdfeRegressor()
```

For compatibility with older scripts, `import py_hdfe_v11` remains available
after installation.

The packaged Python help is available with:

```bash
python -m xhdfe
xhdfe-help
```

Development installs use:

```bash
python -m pip install -e .
```

CUDA package builds can be requested by passing the existing CMake options:

```bash
XHDFE_ENABLE_CUDA=ON CMAKE_CUDA_ARCHITECTURES=90 python -m pip install .
```

Portable source installs do not require `-march=native`. On Apple platforms,
`XHDFE_ENABLE_MARCH_NATIVE` defaults to `OFF` so native Apple Silicon hosts can
also build x86_64 Python environments under Rosetta. The explicit workaround
for older checkouts or unusual cross-target builds is:

```bash
XHDFE_ENABLE_MARCH_NATIVE=OFF python -m pip install .
```

## Build (Python module only, v11)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target py_hdfe_v11 -j
```

The compiled module will be in `build/` (e.g., `build/py_hdfe_v11*.so`).
This direct build path is kept for benchmarks and notebooks that import from a
build directory. See `QUICK_START.md` for smoke tests. For full Python API
options, see `xhdfe/help/xhdfe.md` or run `python -m xhdfe` after installation.

Notes:
- Eigen3 and pybind11 are fetched automatically if not found.
- OpenMP is optional (single-threaded build works without it).
- `-march=native` is optional local tuning. It is disabled by default on Apple
  platforms and only applied when CMake verifies that the active compiler/target
  accepts it.
- For platform-specific instructions and GPU/Metal builds, see `COMPILATION_NOTES.md`.

### Platform notes

- macOS (Apple Silicon): you can force arm64 builds with
  `-DCMAKE_OSX_ARCHITECTURES=arm64` if needed.
- Windows (Visual Studio): configure with `cmake -S . -B build -A x64` and build
  with `cmake --build build --config Release`.
- Windows (MSYS2/MinGW): use a MinGW generator (e.g., `-G "MinGW Makefiles"`).

## Stata command: installation

The Stata command lives in `stata/` and is installed via `net install`.

### Option A (local checkout)

```stata
net install xhdfe, from("/path/to/this/repo/stata") replace
help xhdfe
```

### Option B (online release site)

```stata
net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
net install xfe,   from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
help xhdfe
```

The online Stata site is published from the release workflow after `xhdfe` is
synced to `xhdfe-xfe`. Its `.pkg` files use Stata platform-specific `g` lines
so Stata downloads the plugin that matches the user's OS (`LINUX64`/`LINUX64P`,
`MACARM64`/`OSX.ARM64`, `MACINTEL64`/`OSX.X8664`, and `WIN64` when the Windows
artifact exists). The development URL under `xhdfe/main/stata` should be
treated as a local/developer install surface, not the cross-OS public one.

### Building the plugin

The package includes a platform-specific `xhdfe.plugin`. If it is missing or incompatible
with your machine, build it from source (Linux/Unix):

```bash
bash stata/tools/build-plugin.sh
```

## Stata quick usage

```stata
webuse nlswork, clear
xhdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year) vce(cluster idcode)
```

## Tests and validation

The cross-validation harness lives in `test/tests_xhdfe`. A typical run is:

```bash
python test/tests_xhdfe/generate_data.py
stata-mp -q -b do test/tests_xhdfe/run_reghdfe_tests.do
python test/tests_xhdfe/run_xhdfe_tests.py
python test/tests_xhdfe/compare_results.py
```

Note: the harness expects the v11 Python module `py_hdfe_v11` to be available at
`build` in the project root. If you are running from this repo, you may need
to adjust `run_xhdfe_tests.py` or build the v11 module separately.

## Reports

The latest validation report lives at:

- `test/tests_xhdfe/report.md`
- `test/tests_xhdfe/report.html`

## Benchmarks (simulated data)

This folder collects the code and outputs used to benchmark xhdfe (C++ v11 / `py_hdfe_v11`) against:

- Stata: `reghdfe`
- R: `fixest::feols()`
- Julia: `FixedEffectModels.jl`

### Folder layout

- `benchmarks/Cpp_xhdfe_simulated_panel.ipynb`: xhdfe runs on the 173M-row simulated panel (robust, cluster, retain_fes + save).
- `benchmarks/Cpp_xhdfe_simulated_panel_Full.ipynb`: adds an NLSWORK smoke test and toy patents `group()/individual()` examples.
- `benchmarks/R_fixest_simulated_panel.Rmd` + `benchmarks/R_fixest_simulated_panel.nb.html`: fixest panel runs and timed export steps.
- `benchmarks/R_fixest_simulated_panel_NLSWORK.Rmd` + `benchmarks/R_fixest_simulated_panel_NLSWORK.nb.html`: fixest on NLSWORK.
- `benchmarks/Julia_FixedEffectModels.jl_simulated_panel.ipynb`: Julia runs + FE export.
- `benchmarks/Stata_reghdfe_simulated_panel.do` + `benchmarks/Stata_reghdfe_REGS_simulated.txt`: Stata script and log.

### Datasets benchmarked

- Simulated LEED-like worker-firm-occupation panel: `data/simulated_panel.parquet` (173,163,263 rows).
- Simulated toy patents long: `data/simulated_toy-patents-long.parquet` (100,000,000 edge rows used by `group()/individual()` examples).
- NLSWORK sample: `data/nlswork.dta` / `data/nlswork.parquet` (small smoke test in the full notebook).

### Computation times (table first)

Times below are the wall-clock runtimes printed in the benchmark outputs in `benchmarks/`.
All times are seconds, with minutes in parentheses when >= 60s.

| Dataset / step | N (reported by estimator) | C++ (xhdfe) | R (`fixest`) | Julia (`FixedEffectModels.jl`) | Stata (`reghdfe`) |
|---|---:|---:|---:|---:|---:|
| Simulated panel: read Parquet -> memory | 173,163,263 | 8.53 | 4.95 | 40.45 | 40.29 |
| Simulated panel: robust SEs | 173,163,263 | 52.15 | 119.27 (1.99m) | 238.16 (3.97m) | 1667.21 (27.79m) |
| Simulated panel: robust SEs on GPU | 173,163,263 | na | na | 1207.50 (20.12m) | na |
| Simulated panel: cluster SEs by `firm_id` | 173,163,263 | 71.49 (1.19m) | 136.06 (2.27m) | 243.55 (4.06m) | 1236.07 (20.60m) |
| Simulated panel: fit + retain FEs | 173,163,263 | 161.17 (2.69m) | 140.18 (2.34m) | 414.72 (6.91m) | 1551.75 (25.86m) |
| Simulated panel: append fixed effects | 173,163,263 | na | 148.69 (2.10m) | na | na |
| Simulated panel: write FE-augmented Parquet | 173,163,263 | 37.63 | 36.28 | 70.89 (1.18m) | 107.45 (1.79m) |
| Simulated panel: total (fit + append + save) | 173,163,263 | 198.80 (3.31m) | 325.15 (5.42m) | 493.50 (8.23m) | 1659.20 (27.65m) |

| Toy patents step | N (reported by estimator) | xhdfe (C++ v11) | reghdfe (Stata) |
|---|---:|---:|---:|
| Toy patents: read Parquet -> memory | 100,000,000 edge rows | 2.83 | 28.50 |
| Toy patents: `absorb(inventor_id) group(patent_id) individual(inventor_id) aggregation(mean)` | 23,824,067 groups | 61.29 (1.02m) | 481.78 (8.03m) |
| Toy patents: `absorb(year inventor_id) group(patent_id) individual(inventor_id) aggregation(mean)` | 23,824,067 groups | 38.79 | 546.36 (9.11m)|
| Toy patents: `absorb(inventor_id) group(patent_id) individual(inventor_id) aggregation(sum)` | 23,824,067 groups | 84.22 (1.40m) | 466.47 (7.77m) |
| Toy patents: `absorb(year) group(patent_id) aggregation(mean)` | 23,824,067 groups | 3.71 | 61.29 (1.02m) |

### Discussion / notes

- Timings are as run: xhdfe uses `num_threads=32` for the panel notebook and `num_threads=18` for toy patents; Julia reports `Threads.nthreads() == 16`; fixest reports 32 threads.
- The fixest simulated-panel HTML captures data load + FE extraction/append/save, but not the estimation runtime; the missing cells are left as `n/a`.
- The Stata `reghdfe` script is included under `benchmarks/`, but the committed log does not include timing output.
- Toy patents uses `group()/individual()` aggregation; the input has 100M rows but the estimator reports `N = 23,824,067` after the internal aggregation.
- Current benchmark dependency versions, including the live pyfixest version, are
  recorded in `benchmarks/BENCHMARK_VERSIONS.md` (`pyfixest 0.60.0` as checked
  on 2026-07-06).


## Authors / contact

- Miguel Portela, NIPE / Universidade do Minho and BPLIM / Banco de Portugal.
  Email: miguel.portela@eeg.uminho.pt.
  Website: https://reisportela.github.io
- Tiago Tavares, NIPE / Universidade do Minho.
  Email: tgstavares@eeg.uminho.pt.
  Website: https://www.tgstavares.com

Only the listed human authors are authors or co-authors of xhdfe. No software
tool or AI system is credited as an author or co-author.

Repository: https://github.com/reisportela/xhdfe

## Acknowledgements

We thank **Paulo Guimarães** and **Marta Silva** for discussions of HDFE models.
We also thank **Nelson Areal** for workshop collaboration related to earlier
versions of this proof of concept.
We especially thank **Sergio Correia** for feedback on benchmarking,
tolerances, and `reghdfe`-comparable validation.
We warmly thank **Alexander Fischer** for sharing the latest updates on his and
Kristof Schröder's novel fixed-effects demeaning strategy — a modified LSMR
solver with an additive-Schwarz (domain-decomposition) preconditioner built
from the worker–firm bipartite graph (the
[`within`](https://github.com/py-econometrics/within) project). This graph-based
approach has been very helpful for our ongoing work on high-dimensional
demeaning.

`xhdfe` validates against and interoperates with prior HDFE software. Full
credit goes to **reghdfe** by Sergio Correia (Stata), **fixest** by Laurent
Berge (R), **pyfixest** by Alexander Fischer and collaborators (Python), and
**FixedEffectModels.jl** by Matthieu Gomez and collaborators (Julia).

By design, `xhdfe` is first and foremost a high-performance replica of
`reghdfe`: it mirrors reghdfe's estimator, defaults, and reporting, and
reghdfe-comparable results are its reference. From the worker-firm (AKM)
literature and from **pytwoway** — Thibaut Lamadon and Adam A. Oppenheimer's reference
Python toolkit for two-way worker-firm models (AKM and the leave-out, CRE and
BLM estimators) — `xhdfe` adopts *only* what adds value inside that reghdfe
universe: the leave-out (KSS) bias-corrected variance decomposition, the
leave-out connected set, and the Gelbach decomposition, implemented natively on
the same C++ core. It does not attempt to reproduce pytwoway.

`xhdfe` links to pytwoway in two concrete ways. First, **validation**: its
leave-out decomposition is checked at machine precision against pytwoway and
against **LeaveOutTwoWay** by Raffaele Saggio, the canonical Kline-Saggio-Sølvsten
(2020) implementation. Second, **interoperability**: `xhdfe` exports the
leave-out sample to the pytwoway / bipartitepandas format, so a cleaned two-way
sample moves between the two tools. The combination is most useful in labour
economics with large linked employer-employee data: run the fast HDFE regression
and the leave-out variance decomposition (variance of worker and firm effects,
their covariance, and worker-firm sorting) in a familiar reghdfe workflow with
`xhdfe`, and reach for pytwoway for the broader structural models (CRE, BLM)
that are deliberately outside `xhdfe`'s scope. The Gelbach companion is
validated against **b1x2** by Jonah Gelbach. Full credit to their authors.
Relevant methods: Abowd, Kramarz & Margolis (1999, *Econometrica*); Andrews,
Gill, Schank & Upward (2008, *JRSS-A*); Kline, Saggio & Sølvsten (2020,
*Econometrica*); Andrews & Mikusheva (2016, *Econometrica*); Gelbach (2016,
*Journal of Labor Economics*).

This work connects to the workshop presentation:
“Parallel and Cross-Language Computing: A Hands-On Workshop for Empirical Researchers”,
presented at BPLIM’s workshop “SPEEDING UP EMPIRICAL RESEARCH: TOOLS AND TECHNIQUES FOR FAST COMPUTING”,
15–16 Dec 2025:
https://github.com/BPLIM/Workshops/tree/master/BPLIM2025

## License

MIT License.
