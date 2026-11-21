# xhdfe 2.34.1.20261121

This release brings to the public repository the work of versions 2.32 to
2.34, which until now lived only in the development repository. The main
changes are a new default route for `reghdfe-comparable` fits on CPU,
corrections to `savefe`, to heterogeneous slopes and to weighted projections,
lower memory use, and CUDA libraries that load on demand. `xfepout` is
**1.14.5**. The previous public release was 2.31.1.20261112.

## What changes for every user

- **Reduced-system route for `reghdfe-comparable` on CPU (2.32).** Eligible
  fits now eliminate one fixed effect exactly and solve the system of the
  remaining effects by preconditioned conjugate gradients. The design follows
  `fereg`, by Johannes F. Schmieder; see the credits in the repository README.
  The route is on by default and reports itself as absorption method 8.
  It stops when the normalised residual of the reduced system reaches 1e-12,
  confirmed on a residual recomputed from the iterate, and the existing
  absorption certificate stays mandatory. A fit that does not reach that
  target continues on the route of 2.31.1 and returns the result of that
  route. `reducedpcg(off)` (Stata) or `reduced_pcg=False` (Python, R) restores
  the previous behaviour for a fit.
- **`savefe` in `reghdfe-comparable` (2.32).** With saved effects, 2.31.1
  could publish a covariance, and on chain-like designs coefficients, that
  differed from the same fit without `savefe` by more than the precision
  contract. The fit now takes its coefficients, covariance and residuals from
  the exact projection. Two consequences: such fits are slower (16 % on a
  46-million-row panel; more with weights, see the limitations below), and
  the identity y = xb + constant + effects + residual now closes at the
  precision of the saved effects rather than at machine precision.
  `XHDFE_SAVEFE_ACCURATE_PROJECTION=0` restores the old behaviour for a
  process.
- **Heterogeneous slopes with a regressor absorbed by the effects (2.32).**
  In `reghdfe-comparable`, a regressor that the effects absorb no longer makes
  the fit refuse; the "Heterogeneous slopes" example of `help xhdfe` was
  refused in 2.31.1. The regressor is omitted and the fit returns the numbers
  of the same command without it. Fits that were already accepted with such a
  regressor now stop where the fit without it stops; their results stay
  within the precision contract of the previous ones.
- **Weighted projections on CPU (2.33).** Three cases that converged and were
  certified with a covariance outside the contract are corrected: explicit
  Gauss–Seidel with analytic weights and three effects; `savefe` with weights
  and four effects; and `savefe` with instruments. `strict-residual` on CPU
  now checks the absolute group means of every absorbed column, and its
  repairs are charged to the iteration budget of the request.
- **Lower memory use and clustering at several levels (2.32).**
- **`xfepout` 1.14.5.** Weighted projections with two or three effects that
  the delegated method refused are retried; the options of the parser are
  documented; and the plugin no longer needs the CUDA libraries at load time.
- **Help files.** The Stata help was revised and carries 20 public examples.

## GPU (built from source)

- **CUDA libraries load on demand (2.34.1).** The plugins and the hybrid
  Python module no longer list cuBLAS and cuSPARSE as shared dependencies;
  the functions are resolved at the first use of the network route. On a
  machine where those libraries are not on the loader path, CPU fits now run
  and the first use of that route fails with an error and no estimates.
- **New build requirement.** Building the GPU plugin or module, with
  `xhdfegpu` or by hand, now needs the cuBLAS and cuSPARSE development files
  of the CUDA toolkit (the headers `cublas_v2.h` and `cusparse.h`). A full
  toolkit installation has them; a compiler-only installation does not, and
  its build stops at the missing header.
- **`strict-residual` on CUDA (2.32.2)** now checks the residual it delivers.
- **Projections on CUDA (2.33).** Weighted fits, and unweighted fits with
  four effects, that were certified with a covariance outside the contract
  are now checked per effect and column on the device and, on failure,
  completed by LSMR. `weightedcpuroute(on)` sends eligible weighted fits to
  the CPU and says so in the results; it is off by default.
- **Reduced-system route on CUDA (2.32), experimental and off by default:**
  `reducedpcgcuda(on)`, in `reghdfe-comparable` only.

## Experimental, off by default

- **Solver for firm-specific slopes (2.34.0).** For eligible specifications
  with heterogeneous slopes in `reghdfe-comparable`, an opt-in route
  eliminates the worker and period effects and solves the system of the
  firms. It is selected by environment switches documented in the repository
  and is not part of the default behaviour.

## Known limitations

- **Fits on which the reduced-system route gives up.** On some designs the
  new route does not reach its target and the fit continues on the previous
  route, having paid for the attempt. Against 2.31.1 this was measured on one
  hard three-effect design of ten million rows as 1.24 times the time with 16
  threads and 1.47 times with 4. `reducedpcg(off)` avoids the attempt.
- **Few threads.** The new route gains most with many threads. With one
  thread it can be slower than 2.31.1 on panels where it needs tens of
  iterations.
- **More threads than cores.** An explicit request above the number of
  physical cores can be slower than in 2.31.1 on the same panels; the request
  is honoured and reported, not capped.
- **`savefe` with weights** can take about three times the time and twice
  the memory of 2.31.1 on sparse panels, as the price of the correction.
- **GPU, weighted `reghdfe-comparable` fits.** The repair of 2.33 covers
  analytic, probability and importance weights within a size band; outside
  it a weighted fit on CUDA can still be certified outside the contract.
  `weightedcpuroute(on)` avoids this at the cost of running on the CPU.
- **Windows, nearly perfect fits** (residual below a millionth of the
  signal) can lose the covariance, as in 2.31.1.
- **A CUDA library that fails to load** is remembered for the life of the
  process; recovering needs a new Stata session or interpreter.
- **`BUILD_SHARED_LIBS=ON`** is not a supported build configuration in this
  release.

## Credits

The work of November 2026 behind this release drew on other people's software
and papers, which the repository README ("Algorithmic and software
provenance", "Acknowledgements" and "References") and `help xhdfe`
("Acknowledgements") credit in full, with links:

- [`fereg`](https://github.com/johannes-schmieder/fereg), by Johannes F.
  Schmieder, and his paper *fereg: Fast high-dimensional fixed-effects
  regressions in Stata* ([author's copy](https://johannes-schmieder.com/papers/fereg.pdf),
  CEPR Discussion Paper No. 22170): the design of the reduced-system route
  follows `fereg` 0.6.0, and the performance work was measured against its
  releases 0.6.0 to 0.6.7. The elimination step is the block reduction of
  Somaini and Wolak (2016).
- [`fereg`](https://github.com/leoliu0/fereg), by Leo Liu, a different package
  with the same name: studied in November 2026 as an independent
  implementation and performance reference. This release contains no change
  taken from it.
- [`within`](https://github.com/py-econometrics/within) and the paper of
  Alexander Fischer and Kristof Schröder, the starting point of the
  graph-preconditioned route.
- [`reghdfe`](https://github.com/sergiocorreia/reghdfe) (Sergio Correia), the
  reference for the estimator, defaults and reporting;
  [`fixest`](https://github.com/lrberge/fixest) (Laurent Bergé and
  contributors); [`pyfixest`](https://github.com/py-econometrics/pyfixest)
  (the PyFixest authors);
  [`FixedEffectModels.jl`](https://github.com/FixedEffects/FixedEffectModels.jl)
  (Matthieu Gomez and contributors); and
  [`reghdfejl`](https://github.com/droodman/reghdfejl) (David Roodman).

`xhdfe` contains no source code of either `fereg`.

See [the validation record](https://github.com/reisportela/xhdfe-xfe/blob/main/docs/releases/VALIDATION_2.34.1.20261121.md)
for what was run, on which data and with which results.
