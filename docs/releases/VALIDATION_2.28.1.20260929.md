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
