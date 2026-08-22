# xhdfe 2.24.2 / xfe 1.11.0 - 22aug2026

Stata correctness and Windows installation hotfix. The estimator definition,
objective, numerical tolerances, convergence criteria, C++/CUDA absorption
kernels, Python API, R API, and default output formatting are unchanged.

## Large individual identifiers in Stata

`group()`/`individual()` estimation previously read the numeric
`individual()` variable twice. The fixed-effect ingestion path already
dense-coded sparse numeric identifiers safely, but the second read narrowed
the raw Stata `double` to a C++ `int`. Valid categorical identifiers above
`INT32_MAX` therefore failed before estimation with:

```text
xhdfe plugin: integer out of range in individual
r(198)
```

The plugin now validates the raw individual identifier while reading its
canonical `absorb()` dimension and reuses that dimension's compact
`Eigen::VectorXi` codes in grouped estimation. Accepted identifiers must be
finite, nonnegative, integer-valued, and no greater than `2^53`, the largest
integer boundary at which Stata `double` values have guaranteed unit
resolution. Values above that boundary, fractional values, and negative
values fail with specific diagnostics.

The hot per-observation representation remains 32-bit. No FE, permutation,
offset, GPU, cache, or solver array was widened. The redundant second Stata
read and redundant `N x int32` individual vector were removed.

Regression coverage includes `INT32_MAX-1`, `INT32_MAX`, `INT32_MAX+1`,
`INT32_MAX+2`, `10^10`, `10^11`, `10^12`, `2^53-1`, and `2^53`; repeated
identifiers; `group()` plus multiple `absorb()` dimensions;
`aggregation(mean)` and `aggregation(sum)`; and `residuals()`. Results match an
equivalent `egen group()` recoding for coefficients, covariance, returned
scalars, sample membership, and residuals.

## Windows `net install` runtime delivery

The generated Stata package manifests listed the MinGW/OpenMP runtime DLLs
with lowercase platform directive `g WIN64`. Stata treats `.dll` as an
ancillary suffix under lowercase `g`, so `net install` could complete without
downloading the runtime closure. Windows machines that already exposed
compatible MinGW DLLs could load the plugin, while a clean or enterprise
machine could fail with `r(9999)`.

The manifests now use uppercase `G WIN64` for every DLL named by the validated
Windows runtime ledger. Uppercase `G` forces installation into Stata's system
directories. The plugin mappings remain lowercase `g`, as `.plugin` is already
an installation suffix. Site validation rejects any runtime DLL that lacks
exactly one uppercase `G WIN64` mapping.

The release closure contains the exact DLL set derived recursively from the
Windows PE imports, currently:

```text
libgcc_s_seh-1.dll
libgomp-1.dll
libwinpthread-1.dll
```

## Validation evidence

- The pre-fix Stata reproducer returned `r(198)` for an individual identifier
  above `INT32_MAX`; the dense-coded control completed.
- The patched CPU/OpenMP plugin compiled and linked `libgomp.so.1`.
- The targeted Stata group/individual certification completed with exit 0.
- Wide-ID and dense-ID estimates were identical in the tested mean and sum
  specifications; the sum/residual case returned 100 estimation observations,
  100 nonmissing residuals, and identical residuals within `1e-10`.
- Negative, fractional, and above-`2^53` identifiers returned `r(198)` with
  specific diagnostics.
- The Stata site/corresponding-source suite passed 21 of 21 tests.
- The installed-wheel formula/maketables gate passed 83 of 83 tests without
  skips, including the post-2.24.1 IV formula coverage.
- A real Stata package-semantics probe showed lowercase `g` reporting
  `nothing to install` for a `.dll`, while uppercase `G` installed the same
  file into `PLUS`.

Windows Stata execution is not available on the GitHub-hosted release runner;
the Windows plugin gate therefore certifies PE32+/OpenMP compilation, link and
recursive runtime closure. Linux CPU/CUDA runtime gates and H100 real-use
diagnostics remain mandatory before final publication.

## Version scope

- Shared C++/Python/R package and release tag: `2.24.2.20260822`.
- Stata `xhdfe`, `xhdfe_p`, `xhdfe_estat`, and `xhdfegpu`: `2.24.2`.
- Stata `xfe`: unchanged at `1.11.0`.
- Other command feature versions are unchanged.
- Production Stata text files share release date `22aug2026`.
