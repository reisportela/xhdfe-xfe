# xhdfe 2.24.0 / xfe 1.11.0 — 15aug2026

Python interface and Windows packaging release. The estimator, fixed-effect
absorber, numerical tolerances, convergence criteria, C++/CUDA kernels, Stata
estimation behaviour, and R estimation behaviour are unchanged from 2.23.1.

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
- `x:z` is a product-only interaction; `x*z` expands to both main effects and
  their interaction.
- `I(x**2)` requests an arithmetic square.
- Fixed effects after `|` are encoded as group identifiers and passed to the
  native absorber, not expanded into a dense dummy matrix.
- Weights, frequency weights, one- or multiway clusters, explicit transform
  contexts, result names, source/estimation indices, and `tidy()` are supported.
- `prepare_formula()` creates a read-only design snapshot for repeated fits.

The frontend follows Formulaic semantics rather than adding a second Stata
parser. In particular, `C(g):x` with an intercept contains one explicit slope
per category and is not the same model as standalone Stata `i.g#c.x`.

Formulaic (`formulaic>=1.2.1,<2`) and pandas (`pandas>=1.3`) are an optional,
lazy-loaded `formula` extra. Base imports and the array API do not import that
stack. From a source checkout:

```bash
python -m pip install '.[formula]'
```

The release workflow publishes Python packages as GitHub Release assets; it
does not claim a PyPI publication. Formulaic/pandas/SciPy are not vendored in
the autonomous offline archive and must already be available in an offline
Python environment.

## Formula performance

Simple numeric formulas use a direct NumPy materialisation path. The fixed
parsing cost matters most for very small regressions; a prepared formula avoids
that repeated cost and keeps loop use close to the pre-built-array route. The
array API remains the preferred zero-overhead interface. Numerical and sample
parity are release-tested against manually constructed arrays.

## Generic Windows runtime closure

When CMake selects GNU/MinGW on Windows, the build inspects the resulting
`.pyd`, resolves every non-system PE dependency recursively, and copies the
validated closure beside the extension. The resolver is dependency-driven,
not a list of five Strawberry filenames: missing, ambiguous, unlicensed,
wrong-architecture, conflicting, or unreferenced DLLs fail the build.

The wheel embeds a manifest containing the PE graph, architecture, size,
SHA-256, provider path, provider SHA-256, resolution method, and licence family
for every runtime. The release validator independently reconstructs the graph
from direct wheel members. Before loading the native extension, xhdfe registers
the installed package directory with `os.add_dll_directory()` and retains the
handle for the lifetime of the process.

The prebuilt Windows asset targets CPython 3.12 x86-64. It is built on a
GitHub-hosted Windows x86-64 runner with Strawberry Perl 5.42.0.1 / GCC 13.2,
with machine-specific instruction tuning disabled. Windows source builds also
default to portable tuning; an explicitly local-only build can opt in. The
clean-environment gate removes Strawberry from `PATH` and exercises the base,
legacy, and formula imports. Other Python ABIs use the source distribution and
are not separately Windows-certified in this release.

The Stata Windows bundle uses the same closure principle: its plugin roots,
runtime directory, and ledgers must describe exactly one reachable PE graph;
the online package generates one `g WIN64` entry per validated DLL.

## Runtime provenance and licence materials

Release artefacts that carry GNU/MinGW runtime libraries also carry the exact
applicable GCC Runtime Library Exception, winpthreads, and dlfcn-win32 licence
texts. The adjacent
`xhdfe-2.24.0.20260815-corresponding-source.zip` release asset contains the
exact upstream sources, distribution patches/build recipes, provider source
packages, and a byte-level mapping from every released runtime to its provider.

The CUDA bundle records the exact `nvcc` link invocations and rejects NVIDIA
libraries beyond the static CUDA runtime/device runtime used by the plugins.
It carries the CUDA 12.6 EULA and CCCL 2.5.0 licence materials; NVIDIA inputs
are not represented as GNU Corresponding Source. The Linux wheel separately
maps its auditwheel-private `libgomp` member to the exact provider and source
RPM.

These custody controls are additive packaging changes; they do not change the
estimator or its backend selection.

## R and Stata scope

No R or Stata estimation feature was added for this Python-only frontend. R
already has a formula interface with factors and interactions; Stata already
has factor-variable notation. Their metadata are restamped only to preserve
the unified release identity:

- shared C++/Python/R package and release tag: `2.24.0.20260815`;
- Stata `xhdfe`, `xhdfe_p`, `xhdfe_estat`, and `xhdfegpu`: `2.24.0`;
- Stata `xfe`: unchanged at `1.11.0`;
- other production ado feature versions are unchanged;
- production Stata text files listed by `xhdfe.pkg`/`xfe.pkg`: common release
  date `15aug2026`.

## Certification boundary

Source-level tests and local package checks are prerequisites, not release
approval. A preflight tag must build and assemble every exact artefact. The
version tag creates a draft only. Publication remains blocked until the exact
CI CUDA bundle passes Stata CPU/CUDA functional validation and the large
OpenMP regression gate on the maintainer H100. Only the matching `publish-v*`
marker may publish that draft and its net-install snapshot.
