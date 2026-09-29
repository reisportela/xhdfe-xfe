# Validation scope for 2.28.1.20260929

## Python frontend

The source implementation passed 97 tests without errors, failures or skips,
and the same 97 tests passed in an isolated package layout. The recorded local
stack was Python 3.12, NumPy 2.2.6, pandas 2.3.3, Formulaic 1.2.1,
Narwhals 2.21.2, Polars 1.36.1, PyArrow 24.0.0 and maketables 0.1.8.

The permanent independent oracle in
`tests/analytic_precision/polars_formula_contract.py` checks coefficients,
complete covariance, residuals, sample and degrees of freedom against explicit
OLS and HC1 for int16, int32 and int64 inputs. The six Polars cases failed on
the old formula path and pass after promotion before arithmetic. Tests also
cover transformed responses, interactions, quoted names, IV matrix blocks,
category roles, Enum metadata, identifiers above 2^53, missing values and input
preservation. Explicit category levels are used when contrast ordering matters.

The shared gate rejects a missing Polars installation, an omitted analytic
suite and import paths that point back to the source checkout instead of the
intended installed package. These three negative controls were exercised.
CI repeats the contract against installed package artifacts, including the
Formulaic 1.2.1 compatibility floor in the release job.

## Scope of performance measurements

Seven interleaved pairs on 250,000-row prepared inputs compared the current
frontend with frozen 2.28.0 sources. All prepared y/X/FE arrays were identical
for the previously correct float64 specifications. String-ID formula
materialization medians fell by approximately 69–77% across pandas/Polars
and the numeric/Formulaic paths. These are frontend timings, not estimator
speedups or a new core24 performance campaign.

An allocation-lifetime-sensitive pandas numeric measurement was investigated
with a separate 24-pair control, releasing each result before the next call:
17.31 ms versus 17.54 ms (+1.36%). Small timing differences under variable host
load are not adjudicated as general gains or regressions. Already-float64
Polars columns avoid unnecessary replacement while retaining validation.

## Inherited native coverage and limitations

The C++/CUDA sources and algorithms are unchanged from 2.28.0. The
[2.28.0 validation record](VALIDATION_2.28.0.20260924.md) remains the source
for its core24 × 8 campaign, recovery coverage, accepted performance costs
and known CUDA refusals. This patch does not claim to have rerun that entire
campaign or to resolve those inherited numerical limitations. Fast and
Comparable retain their distinct contracts and tolerances.

## Artifact publication gates

The release sequence requires fresh CI builds, native OpenMP worker checks on
Linux/Windows/macOS, clean installed-wheel tests and source/runtime provenance.
The version tag creates a draft. Its exact Linux CUDA plugins must then pass
fresh-process H100 checks, and the exact Stata CPU/net-install artifacts must
load and pass numerical checks before the publish marker is pushed.

The release's offline bundle carries native receipts and provider ledgers;
the published checksums identify the distribution files. Local source tests
and earlier release artifacts do not substitute for these version-tag gates.
See the [release procedure](../release-workflow.md).

## Executed release gates

The [preflight](https://github.com/reisportela/xhdfe-xfe/actions/runs/36553876660)
and the independent
[version-tag build](https://github.com/reisportela/xhdfe-xfe/actions/runs/36558408298)
passed on public commit `9d221ae21206686b2b30857c1cb883087ad4c143`.
All 13 final draft assets matched both GitHub's SHA-256 digests and the
downloaded `SHA256SUMS.txt` before runtime testing.

| Gate | Observed result |
| --- | --- |
| Linux CPU/CUDA-host, Windows, macOS arm64/x86_64 plugins | Ten native plugin cases matched final artifact hashes and demonstrated one/two useful OpenMP workers with analytic numerical agreement. |
| macOS serial controls | Both plugins on both architectures rejected the separate serial builds. |
| Installed Linux/Windows Python wheels and R package | Actual one/two-worker gates and analytic b/full covariance/RSS checks passed. |
| Downloaded Linux wheel, isolated local installation | The complete 97-test frontend gate passed again without skips or source shadowing. |
| Exact Linux CUDA ZIP on H100 with licensed Stata | Fast/Comparable analytic b/full V, xfepout, savefe/reconstruction and CPU/CUDA cache cases passed with actual GPU use. |
| Exact Linux CPU ZIP with licensed Stata | The analytic fixture, documented auto-data example, cache cases and large CPU smoke passed. |
| Platform ZIPs, net-install snapshot, corresponding sources | Package layout, runtime/source closure and provenance validators passed. |

The freshly built Linux CPU and CUDA plugin pairs are byte-identical to their
2.28.0 counterparts. In fresh Stata processes on the large CPU sample,
the previous public package and the final 2.28.1 package returned identical
coefficients, full covariance, RSS, sample signature and degrees of freedom:
46,156,187 estimation observations, 23 iterations and 16 actual workers.
Fit times were 80.854 s and 80.182 s. This single pair is an artifact smoke,
not a new performance campaign or a general speedup claim.

Both canonical local checkouts were separately rebuilt and exercised on CPU
and H100. Their native modules, active Python imports and Stata entry points
were identified by path and hash. Local sm_90 builds remain distinct from
the portable CI distribution artifacts.

## R check qualifications

The version-tag `R CMD check` completed its examples and tests, with two
warnings and four notes. The same pattern was present in the exact 2.28.0
tag build: nonstandard source filenames, a GCC `-Wstringop-overread` warning,
unavailable suggested packages (`fixest`, `nanoparquet`, `gt`), installed size,
GNU make and compiled stdout/stderr symbols. The separate R OpenMP/numerical
gate passed with zero recorded analytic and one/two-worker differences.

These qualifications are retained; the result is not a warning-free CRAN
check and does not establish that the compiler diagnostic is false. The
native sources were not changed to suppress these diagnostics.
