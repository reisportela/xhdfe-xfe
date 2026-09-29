# xhdfe 2.28.1.20260929

Release date: 29 September 2026. The xhdfe command version is 2.28.1;
the shared Python/R/package identity is 2.28.1.20260929.

This patch fixes integer overflow while constructing Python formula matrices
from Polars input. Numeric formula columns are promoted to float64 before
products, powers and other expressions are evaluated. The same checks on
numeric/categorical roles apply to pandas and Polars. Category metadata and
large integer IDs are preserved, and the caller's DataFrame is not modified.
Already-float64 Polars columns are reused after validation.

It also incorporates Francisco Queiró's label-ID encoding optimization from
[PR #12](https://github.com/reisportela/xhdfe-xfe/pull/12): a pandas dtype
inference pass avoids the Python element-by-element numeric scan only for
label-only object columns. Mixed columns retain the existing validation.

Polars remains an optional input format. Formulaic/Narwhals can convert
categorical columns internally to pandas, which requires PyArrow; the help
now documents this dependency and the need to supply prepared data. No data
preparation engine is embedded in xhdfe.

The ordinary CI and installed-wheel release gate now share a complete
97-test frontend contract, including independent OLS/HC1 references for
int16/int32/int64 transforms. Missing dependencies, skipped tests, omitted
test suites and source-checkout shadowing fail the gate.

The C++/CUDA estimator and its stopping rules are unchanged. Stata and R
receive the shared release identity and documentation updates. Companion
versions remain xfepout 1.13.2, xhdfeakm 1.8.1, xhdfegelbach 1.6.1 and
xhdfegelbachbootstrap 1.0.1, with the common 29sep2026 date. The quarantined
experimental xhdfe_hetero command remains outside the installable Stata package.

The [validation record](VALIDATION_2.28.1.20260929.md) distinguishes the new
frontend checks from the inherited native coverage and limitations.
Publication uses fresh CI artifacts, exact-asset CUDA validation on the H100,
OpenMP gates on every native platform, and an independently checked net-install
snapshot.
