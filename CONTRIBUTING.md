# Contributing to xhdfe

Thanks for your interest in improving xhdfe. This document explains how the
project is laid out, how to build and test each component from source, and what
we expect from a change before it can be merged.

xhdfe is a single high-dimensional-fixed-effects estimator, exposed identically
to Stata, Python, and R on one compiled C++ core. CPU is the reference backend;
the CUDA GPU absorber is optional. Results mirror `reghdfe` (Correia, 2016) at
the same nominal tolerance under the default `reghdfe-comparable` tolerance mode.

**Supported CPU source-install platforms:** Linux, Windows with Rtools/MinGW,
and macOS Apple Silicon/Intel where a C++17 toolchain is available. CUDA builds
are currently Linux-only and require the NVIDIA toolkit.

## Repository layout and the one-core rule

There is exactly **one canonical C++ core**, and everything else mirrors it:

- `src/` + `include/` (+ `include/hdfe/`) — the canonical core:
  `hdfe_regressor_v11.cpp`, `fe_absorption.cpp`, `schwarz_demean.cpp`,
  `iv.cpp`, `ols.cpp`, and the optional `fe_absorption_cuda.cu` /
  `fe_absorption_metal.mm`.
- `third_party/eigen-3.4.0/` — vendored, header-only Eigen (unmodified).
- `python/py_hdfe_v11.cpp` + `xhdfe/` — the pybind11 binding and Python package.
- `stata/src/` + `stata/include/` — a **byte-for-byte mirror** of the core,
  plus the Stata-only plugin glue `xhdfe_plugin.cpp` and `xfe_plugin.cpp`.
- `r/xhdfe/src/` — a **byte-for-byte mirror** of the core (`include/`, `eigen/`)
  plus the Rcpp binding `rcpp_xhdfe.cpp` (+ `RcppExports.cpp`).

**Edit the canonical core only.** Never edit a mirrored core file under
`stata/` or `r/xhdfe/src/` directly — such edits are overwritten on the next
refresh and silently diverge the packages.

After changing the core, refresh the mirrors:

- R: `bash r/tools/refresh_r_core.sh` — copies the core into `r/xhdfe/src` and
  verifies the result is byte-identical (fails loudly on any mismatch).
- Stata: copy the changed `src/*.{cpp,cu,hpp}` and `include/*` into `stata/src`
  and `stata/include` respectively. The plugin glue `xhdfe_plugin.cpp` /
  `xfe_plugin.cpp` is unique to `stata/src` and is not part of the core mirror.

CPU behavior is authoritative. If you touch the CUDA path it must stay
explicitly validated against the CPU result — a CUDA build that silently falls
back to CPU, fails to converge, or disagrees with CPU is not acceptable.

## Building from source

### C++ core and Python module

Reference Release build via pip (uses CMake under the hood):

```bash
python -m pip install .          # or: python -m pip install -e .  (editable)
```

Or configure CMake directly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Optional CUDA absorber (requires the NVIDIA CUDA toolkit with `nvcc` on PATH).
Set the CUDA architecture to your GPU's compute capability (e.g. `75` for T4,
`80` for A100, `90` for H100):

```bash
XHDFE_ENABLE_CUDA=ON CMAKE_CUDA_ARCHITECTURES=90 python -m pip install .
```

The core is compiled with fast-math Release flags
(`-O3 -march=native -ffast-math -funroll-loops -fopenmp`), and OpenMP is
required for the reference threading behavior. These are deliberate: a plain
`-O2` / generic-`-march` build is **not** the reference numerical behavior.

### R package

```bash
R CMD INSTALL r/xhdfe
```

The install replicates the reference Release flags (see `r/xhdfe/src/Makevars`).
Install-time environment knobs: `XHDFE_MARCH` (target `-march`, default `native`;
set e.g. `x86-64-v3` for a portable binary), `XHDFE_ENABLE_CUDA` (`ON` to compile
the CUDA absorber, default `OFF`), `XHDFE_CUDA_ARCH` (compute capability, single
target, default `90`), and `CUDA_HOME` (toolkit root, default: `nvcc` on PATH):

```bash
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 R CMD INSTALL r/xhdfe
```

Verify any install with `xhdfe_info()`, which reports the compiler, `-march`,
fast-math, and CUDA arch. On the supported platform `R CMD check` reports one
expected WARNING (the `.cu`/`.hpp` sources kept for tarball CUDA rebuilds) and
one NOTE (env-gated, silent-by-default `std::cout`/`std::cerr` diagnostics).

### Stata plugin

From the `stata/` folder (the build scripts fetch `stplugin.h/.c` and Eigen on
first run, so the first build needs network access):

```bash
bash tools/build-plugin.sh --linux --openmp        # -> stata/xhdfe.plugin
bash tools/build-xfe-plugin.sh --linux --openmp    # -> stata/xfe.plugin
```

Production plugin builds must use OpenMP. For CUDA, add
`XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=<cc>` (single target) or
`XHDFE_CUDA_ARCHS="75,80,86,89,90"` (multi-arch fatbin for redistribution).
See `stata/BUILD_CUDA.md` for the GPU build and Stata-side verification steps.

## Running the tests

The primary automated suite is the R package's `testthat` (edition 3) tests in
`r/xhdfe/tests/testthat/`:

```bash
R CMD INSTALL r/xhdfe
Rscript -e 'testthat::test_local("r/xhdfe")'
```

- `test-interface.R` — exact agreement with `lm()` on dummy-expanded designs,
  `fixest::feols` agreement (coefficients and cluster SEs), weighted fits, the
  formula grammar, S3 methods, and fail-closed GPU behavior (needs `fixest`).
- `test-regressions.R` — pinned fixes for previously found defects.
- `test-parity-python.R` — checks the R binding against the committed reference
  fixture `tests/testthat/fixtures/parity_reference.json` (needs `jsonlite`;
  skips if absent). Regenerating that fixture uses `r/tools/gen_parity_data.R` +
  `r/tools/gen_parity_fixture.py` and **requires a built reference Python
  module** — you need not regenerate it for a normal change; the committed
  fixture is used as-is.

The Stata package has a certification-test harness under `tests/stata/`,
adapted from Sergio Correia's `reghdfe` test-suite style. It runs built-in
Stata baselines (`regress`, `areg`), then checks `xhdfe` coefficients, variance
matrices, stored scalars, postestimation predictions, singleton handling, and
compatibility options:

```bash
bash stata/tools/build-plugin.sh --linux --openmp
tests/stata/run_stata_tests.sh
```

On macOS, build the local plugin with the platform-appropriate flags first,
for example `bash stata/tools/build-plugin.sh --no-openmp --no-march-native`.
The older worked comparison do-files under `stata/example/`
(`2.Detailed_Comparision_xhdfe_reghdfe.do`, `3.Check_Options_reghdfe_xhdfe.do`)
remain useful for exploratory option-by-option comparisons against `reghdfe`.

## Non-regression expectation

Every change must **preserve** the estimator's numerical results, convergence
behavior, default output, and public interfaces. This holds across CPU and —
when the CUDA path is touched — GPU, and for both the plain (no-`savefe`) and
the fixed-effect-recovery (`savefe` / `fixef()` / `retain_fes`) paths. If a
touched path can affect both, validate both. No speedup is worth reduced
feature coverage, lower precision, relaxed tolerances, changed default output,
or a silent CPU fallback after a requested CUDA run. Reference comparisons are
`reghdfe` (Stata) and the sibling packages `fixest` (R), `pyfixest` (Python),
and `FixedEffectModels.jl` (Julia).

## Reporting bugs

Open a GitHub issue at https://github.com/reisportela/xhdfe-xfe/issues and include:

- OS and compiler version (`gcc --version`) and, for GPU issues, the CUDA
  toolkit/driver and GPU model.
- The xhdfe version: `xhdfe_info()` in R, `xhdfe.__version__` in Python, or the
  help-file version (`help xhdfe`) in Stata.
- A **minimal reproducible example on public or synthetic data** (e.g.
  `sysuse auto` / `webuse nlswork` in Stata, or a seeded generator in R/Python).
  Do not attach private data.

## Proposing changes

1. Fork the repo and create a topic branch.
2. Edit the **canonical core** (or the relevant binding), never a mirror.
3. Refresh the mirrors (`r/tools/refresh_r_core.sh`; copy the Stata mirror) so
   all packages stay in sync.
4. Add or update tests, and confirm the non-regression expectation above.
5. Open a pull request describing the change and the validation you ran.

## Releases and binaries

Prebuilt binaries — the Stata `.plugin` files and distribution ZIPs — are **not**
committed to the repository. They are attached to
[GitHub Releases](https://github.com/reisportela/xhdfe-xfe/releases). The repository
carries source only; build from source or download a Release.

## Respectful collaboration

Please be kind, constructive, and professional in issues, reviews, and pull
requests. We welcome contributions from everyone and expect all participants to
treat each other with respect.
