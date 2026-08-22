# xhdfe 2.24.2 / xfe 1.11.0 - 22aug2026

Correctness, interface, and Windows installation release. The estimator
definition, objective, numerical tolerances, convergence criteria, absorption
kernels, CUDA orchestration, and default output formatting are unchanged. The
release also incorporates the Python formula IV frontend and the corrected
frequency-weight scale used by multiway PSD covariance repair that entered the
public main after 2.24.1.

## Python formula IV/2SLS

The Python formula frontend now accepts the same three-part IV grammar as the R
frontend:

```text
y ~ exogenous | fixed_effects | endogenous ~ excluded_instruments
```

The implementation parses quoted and nested separators safely, supports both
the numeric fast path and Formulaic path, passes the named endogenous positions
to the existing native 2SLS estimator, and exposes frozen instrument/design
metadata. Sixteen regression tests cover single and multiple endogenous
regressors, clustering, weights, transformed terms, categorical blocks,
intercept handling, prepared formulas, and malformed specifications.

This feature was contributed by fqueiro in public PR #8. The release preserves
that authorship and credit.

## Frequency-weight scaling in multiway PSD repair

The scale used to repair a non-PSD multiway-cluster covariance matrix now
computes sample standard deviations under `fweights` as literal row
replication. The correction uses compensated wide accumulation and applies in
the shared core and all maintained C++/Stata/R mirrors. It does not alter
unweighted or analytic-weight paths.

This correction was contributed by Tiago Tavares in public PR #9. The release
preserves that authorship and credit.

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
