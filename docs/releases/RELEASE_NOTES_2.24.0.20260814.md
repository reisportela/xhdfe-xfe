# xhdfe 2.24.0 / xfe 1.11.0 — 14aug2026

Python interface and Windows source-build packaging release. The estimator,
fixed-effect absorber, numerical tolerances, convergence criteria, C++/CUDA
kernels, Stata estimation behavior, and R estimation behavior are unchanged
from 2.23.1.

## Optional Python formula interface

Python now offers an additive R/Formulaic-style frontend while retaining the
existing low-overhead array API:

```python
model = xhdfe.feols(
    "y ~ x1 + x2 + C(industry) | firm + year",
    data=d,
    se_type="cluster",
    clusters="firm",
)
```

- `C(g)` supplies treatment-coded categorical regressors.
- `x:z` is a product-only interaction; `x*z` expands to the two main effects
  and their interaction.
- `I(x**2)` requests an arithmetic square.
- Fixed effects after `|` are encoded as group identifiers and passed to the
  native absorber; they are never expanded into a dense dummy matrix.
- `weights`, frequency weights, one- or multiway clusters, explicit transform
  contexts, result names, source/estimation indices, and `tidy()` are supported.
- `prepare_formula()` creates a read-only design snapshot for repeated fits.

The frontend follows Formulaic semantics rather than adding a second Stata
parser. In particular, `C(g):x` with an intercept contains one explicit slope
per category and is not the same model as standalone Stata `i.g#c.x`.

Formulaic (`formulaic>=1.2.1,<2`) and pandas (`pandas>=1.3`) are declared in
the optional `formula` extra and are imported lazily.
Base imports and the array API do not import Formulaic, pandas, or SciPy. From a
source checkout, install the frontend with:

```bash
python -m pip install '.[formula]'
```

The release workflow publishes Python packages as GitHub Release assets; it
does not claim a PyPI publication. The autonomous offline archive closes the
base native package. The optional Formulaic/pandas/SciPy stack is not vendored
and must already be available in an offline Python environment.

## Formula performance

Simple numeric formulas use a direct NumPy materialization path. An
environment-specific development microbenchmark found that the fixed parsing
cost matters most at very small sample sizes and becomes small relative to
estimation at larger samples. Reusing a `PreparedFormula` made repeated fits
close to the pre-built-array route in that benchmark. These timings are
illustrative rather than a cross-platform release gate; coefficients, standard
errors, residuals, convergence, and iteration counts remain covered by the
formula-versus-array parity tests.

## Windows Python source builds

When CMake selects GNU/MinGW on Windows, the build now inspects the resulting
`.pyd` and recursively bundles its non-system GNU runtime DLL closure beside
the extension. This covers runtimes such as `libgcc_s_seh-1.dll`,
`libstdc++-6.dll`, `libgomp-1.dll`, and `libwinpthread-1.dll`, including
Strawberry Perl toolchains. Missing or conflicting runtime files fail the wheel
build instead of producing an artefact that fails later at import time. MSVC
builds are unchanged.

Before the native import, the package now registers the installed directory
containing those DLLs with `os.add_dll_directory()` and retains the returned
handle for the lifetime of the process. This closes Python 3.8+'s restricted
DLL-search behaviour without relying on Strawberry Perl's `PATH` entry or a
user-side workaround.

The logic has fail-closed unit coverage. Release certification additionally
requires a real Windows/Strawberry wheel build, inspection of the wheel DLL
closure, and a successful import and regression after Strawberry's compiler
directory is removed from `PATH`.

## R and Stata scope

No R or Stata feature was added for this Python-only frontend. R already has a
formula interface with factors and interactions; Stata already has native
factor-variable notation. Their package metadata and release dates are
restamped only to preserve the repository's unified release identity:

- shared C++/Python/R package and release tag: `2.24.0.20260814`;
- Stata `xhdfe`, `xhdfe_p`, `xhdfe_estat`, and `xhdfegpu`: `2.24.0`;
- Stata `xfe`: unchanged at `1.11.0`;
- `xhdfeakm`, `xhdfeconnected`, `xhdfegelbach`, and its presentation helpers:
  unchanged feature versions;
- all production Stata text files listed by `xhdfe.pkg`/`xfe.pkg`: common
  release date `14aug2026` (the separately versioned experimental
  `xhdfe_hetero` surface is not part of those packages).

## Source-candidate validation

- Python `unittest` discovery suite: 51/51 tests passed against the current compiled
  binding, including formula/array parity and frequency weights.
- Formula-focused suite: 35 tests passed; lazy optional imports, categorical
  coding, interactions, missing-data failures, identifier exactness, clusters,
  weights, result metadata, and FP64 transform arithmetic are covered.
- Windows runtime packaging suite: eleven fail-closed dependency-closure and loader tests
  passed on the source candidate.
- The existing direct-array API passed a separate smoke without importing the
  optional formula stack.

These source tests are not by themselves approval of final release assets.
The version tag creates a draft. Publication remains blocked until the exact
wheel and source distribution pass installed-artifact tests, the Windows wheel
passes its clean-runtime gate, and the exact CI CUDA plugins pass the existing
maintainer-H100 real-use gate. Only the matching `publish-v*` marker may then
publish the draft and Stata net-install snapshot.
