# xhdfe 2.24.1 / xfe 1.11.0 — 16aug2026

Python regression-table integration release. The estimator, fixed-effect
absorber, numerical tolerances, convergence criteria, C++/CUDA kernels, Stata
estimation behaviour, and R estimation behaviour are unchanged from 2.24.0.

## Direct maketables integration

Fitted Python results now implement maketables' documented duck-typed plug-in
format. Formula and native array-API fits can be passed directly to `ETable`
without an adapter, registration step, or new xhdfe runtime dependency:

```python
import maketables as mt
import xhdfe

model = xhdfe.feols(
    "y ~ x1 + x2 | firm + year",
    data=d,
    se_type="cluster",
    clusters="firm",
)
print(mt.ETable([model], drop="Intercept").make(type="tex"))
```

The integration exposes coefficient estimates, standard errors, t statistics,
p-values, dependent-variable and fixed-effect metadata, variance-estimator
information, variable labels, and the following model statistics: `N`, `r2`,
`r2_within`, `rmse`, `n_clusters`, `se_type`, `N_full`, `n_singletons`, and
`df_absorbed`.

Formula fits retain canonical cluster names, standard-error type, and a
read-only snapshot of supported dataframe variable labels. Array-API fits use
positional coefficient names and explicit generic fixed-effect labels when no
formula metadata exists, so an absorbed dimension is never silently hidden.

Confidence-interval plug-in tokens are deliberately omitted. xhdfe supports an
arbitrary confidence level and must not label a non-95% interval as `ci95l` or
`ci95u`. Existing confidence intervals remain available through the normal
xhdfe result interface.

The hook attachment is idempotent and transactional. Failure to attach optional
table descriptors cannot leave a partially modified result class or prevent the
native estimator from loading.

## Packaging and validation

The release workflow pins `maketables==0.1.8` for the integration gate and
requires the new module and tests in wheels, Python source distributions, and
the autonomous source archive. The base package still does not depend on or
import maketables.

Local acceptance covered 116 repository tests and 67 formula/maketables tests,
including end-to-end `ETable` rendering for formula and array results. A
deterministic before/after estimator payload remained byte-identical, with
convergence and iteration count unchanged.

## Contribution credit

This feature was proposed and initially implemented by
[@fqueiro](https://github.com/fqueiro) in
[public PR #7](https://github.com/reisportela/xhdfe-xfe/pull/7). The released
implementation adapts that contribution to xhdfe's confidence-level semantics,
native array metadata, transactional attachment, and release-artifact gates.

## Version scope

No R or Stata estimation feature is added by this Python integration. Their
metadata are restamped only to preserve the unified release identity:

- shared C++/Python/R package and release tag: `2.24.1.20260816`;
- Stata `xhdfe`, `xhdfe_p`, `xhdfe_estat`, and `xhdfegpu`: `2.24.1`;
- Stata `xfe`: unchanged at `1.11.0`;
- other production ado feature versions: unchanged;
- production Stata text files listed by `xhdfe.pkg`/`xfe.pkg`: common release
  date `16aug2026`.

## Certification boundary

Local source tests are prerequisites, not publication approval. A preflight tag
must build and assemble every exact artefact. The version tag creates a draft
release only. Publication remains blocked until the exact CI CUDA bundle passes
CPU/CUDA functional validation, real-use diagnostics, numerical parity, and the
large OpenMP regression gate on the maintainer H100. Only the matching
`publish-v*` marker may publish that draft and its net-install snapshot.
