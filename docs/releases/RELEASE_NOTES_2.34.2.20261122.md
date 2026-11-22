# xhdfe 2.34.2.20261122

- In eligible CPU `reghdfe-comparable` fits with distinct cells and a large
  fixed effect, GCC builds prefetch rows of the reduced system and accumulate
  each cell with vector instructions. The kernel tests compare the prefetch
  path bit for bit with the original loop. The process-level diagnostic
  `XHDFE_REDUCED_PCG_PREFETCH=0` selects the original loop.
- Large fits can compute independent fit statistics and column norms on
  separate workers and split copies of residuals and fixed-effect identifiers
  into blocks. Each statistic uses its original reduction.
- After reading wide identifiers in parallel, Stata checks their order on the
  same team. It can also prepare the changes to `e(sample)` in parallel and
  writes the affected rows in observation order. Thread diagnostics account for
  the teams and workers observed in these stages.
- This release ships `xfepout` 1.14.5, rebuilt with the shared sources.
- Use `net get xhdfe` from the package site to copy two standalone do-files.
  After installing xhdfe, run `do xhdfe_example_cluster.do` or
  `do xhdfe_example_partial_out.do` with the datasets supplied with Stata.

The comparison with Leo Liu's [`fereg`](https://github.com/leoliu0/fereg)
informed the operator and fit changes. The reduced-system design also draws on
Johannes F. Schmieder's [`fereg`](https://github.com/johannes-schmieder/fereg).
