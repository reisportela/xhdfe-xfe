# xhdfe 2.21.0.20260725 — flexible HDFE Gelbach decomposition

This release completes the linear HDFE Gelbach feature tranche and promotes
the previously local bootstrap/reporting companions to the installable
package. The estimator remains an order-invariant accounting of coefficient
movement between one base and one full linear specification; it is not an
automatic causal-mediation procedure.

## Broader estimand surface

- `xhdfe.gelbach`, `xhdfe_gelbach`, and `xhdfegelbach` accept multiple focal
  coefficients, multiple multicolumn observed blocks, any number of added FE
  dimensions, and high-dimensional fixed effects common to the base and full
  models.
- Common FEs condition both specifications without being reported as
  artificial contributions. With common FEs, slope identities and inference
  remain available while the normalization-dependent intercept allocation is
  explicitly not certified.
- Observed targets and explicitly declared FE-absorbed targets are supported.
  Generic rank failures and undeclared omissions continue to fail closed.
- A selectable pair of added FE dimensions can be diagnosed for retained-sample
  mobility connectivity. `connected(require)` certifies only the supported
  two-way, no-common-FE domain and rejects unsupported multiway claims.
- Multiple linear contrasts can be computed from the joint contribution
  covariance without re-estimating the model.

## Inference diagnostics

- The full result exposes observed-block coefficients, auxiliary loadings,
  requested-VCE Wald diagnostics, product-gradient diagnostics, and a
  conservative per-cell regularity status.
- The two component tests in the regularity union gate now use `alpha/2`, so
  the public `regularity_test_alpha` is the family-wise level.
- Share intervals retain the existing delta-method point and standard error,
  but are marked diagnostic when the denominator has
  `|estimate / SE| < 3`. The package routes those cases to the existing pairs
  bootstrap.
- FE-block and mixed-total intervals are marked conditional-only when the
  focal residual-to-total variance ratio is at most `0.35`. No variance value
  is silently replaced.
- Near-FE collinearity, few-cluster designs, absorbed-target inference, and
  backend use remain visible in the returned metadata and warnings.

## Bootstrap and research reporting

- New public companions:
  - Python: `gelbach.bootstrap`, `gelbach.etable`,
    `gelbach.waterfall_data`, and `gelbach.coefplot`;
  - R: `xhdfe_gelbach_bootstrap`, `xhdfe_gelbach_etable`,
    `xhdfe_gelbach_waterfall_data`, and `xhdfe_gelbach_coefplot`;
  - Stata: `xhdfegelbachbootstrap`, `xhdfegelbachetable`, and
    `xhdfegelbachcoefplot`.
- The bootstrap performs a full refit for every iid-pairs or explicitly
  declared cluster-pairs draw. It uses independent streams, records every
  failed and valid replication, enforces a minimum valid count, and can require
  real CUDA use.
- Percentile and basic intervals are available for components, totals, base
  and full coefficients, and supported share conventions.
- Tables are available as data frames/display output, Markdown, LaTeX, HTML,
  CSV, and optional Great Tables output where supported.
- Stata tables now use the same joint-covariance delta method as Python and R
  for component/base, component/movement, and total/base shares. Weak
  denominators retain the numerical interval but are labelled diagnostic
  rather than being presented as conventional inference.
- Filtered tables and waterfall plots include an `Other (filtered)` row by
  default. Its standard error uses the complete joint covariance sub-block, so
  visible components continue to add to the printed total.
- Stata multi-label parsing, bootstrap-interval selection, and the
  retained-sample provenance example have been corrected.
- LaTeX row terminators and note control sequences were corrected and are
  regression-tested with a real `pdflatex` compilation. Reporting help now
  documents defaults, full bootstrap provenance metadata, RNG differences
  across frontends, and Stata's volatile post-graph `r()` behavior.

## Auditability and sample provenance

- Opt-in sample provenance returns input, retained, effective-N, and singleton
  counts; zero-based retained positions; a retained-sample mask; and a stable,
  order-sensitive `fnv1a64-le-v1` identifier.
- Stata can generate a sample ledger variable distinguishing retained rows,
  singleton removals, and observations outside the initial `if`/`in` sample.
- Python, R, and Stata use the same field names, status vocabulary, default
  thresholds, and core implementation.

## Compatibility and boundaries

- `b1x2` remains an oracle for the classic OLS subset, not the boundary of the
  public API.
- Existing point estimates, covariance entries, solver tolerances,
  convergence criteria, normalization rules, and the default non-Gelbach
  `xhdfe`/`xfe` paths are unchanged by the final inference gates.
- The Python source distribution now resolves Eigen from the packaged
  canonical vendored tree when the R source mirror is absent, while a full
  repository build continues to verify and prefer the byte-aligned R mirror.
  Isolated sdist installation therefore remains fully offline and does not
  fall back to a system or network dependency.
- Frequency weights remain supported by the decomposition itself; compressed
  frequency weights are deliberately rejected by pairs bootstrap because row
  resampling is not equivalent to resampling the expanded sample.
- Multiway or wild-cluster inference, automatic largest-component sample
  selection, nonconditional recovered-FE uncertainty, IV/LATE, dynamic-panel,
  nonlinear, distributional, formula, and path-dependent narrative engines
  remain outside this linear Gelbach command.

## Distribution and release gate

- Release assets include Linux CPU/OpenMP and CUDA fatbin Stata plugins,
  Windows 11 x86_64 CPU/OpenMP plugins with colocated runtime DLLs, macOS
  universal plugins, a Linux Python wheel and source distribution, an R source
  package, and an autonomous offline source/runtime archive with a SHA-256
  manifest.
- `SHA256SUMS.txt` authenticates every separately downloadable release asset.
- A version tag creates a draft only. The exact CUDA fatbins produced by CI
  must be downloaded and exercised on the maintainer H100 before the matching
  `publish-v*` certification marker can promote the draft and net-install site.

## Version surface

- Shared C++/Python/R package and release tag: `2.21.0.20260725`.
- Stata `xhdfe`: `2.21.0`.
- Stata `xhdfegelbach`: `1.5.0`.
- New Stata companions: `1.0.0`.
- Stata `xhdfeakm`: `1.7.2` (number unchanged).
- Stata `xhdfeconnected`: `1.2.1` (number unchanged).
- Stata `xfe`: `1.10.1` (number unchanged).
- All production Stata files carry the common release date `25jul2026`.

Release certification evidence, packaged-install checks, backend diagnostics,
and artifact hashes are recorded in
`docs/certification/gelbach-release-certification-20260725.md`.
