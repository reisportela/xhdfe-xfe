# xhdfe 2.30.1.20261020

This correction resolves the inherited CUDA Comparable covariance discrepancy
identified after 2.30.0. Ordinary models with three fixed effects, no weights
and no saved effects now complete the remaining projection at the existing
tolerance and within the original iteration budget. Cooperative FP64 GPU
reductions reduce the cost of that completion.

- **Covariance and partial-out accuracy.** Independent dummy-variable
  references validate the complete coefficient and covariance output,
  including rescaling and row-order controls. The same correction improves
  `xfepout` projections; its version is **1.13.5**.
- **GPU repair accounting.** A failed GPU attempt consumes iterations even
  when it returns no usable result. Subsequent repair stages now receive only
  the remaining budget and report the accumulated work. Exhaustion leaves
  no estimates.
- **Cache migration.** Previously computed absorption transforms are
  invalidated by a new solve-contract generation. Exact new-generation hits
  remain supported; old caches do not need to be deleted manually.
  Re-estimate affected models after updating: loading the package does not
  recalculate previously saved estimation results.
- **Preserved contracts.** CPU estimation, Fast/Comparable semantics,
  public tolerances, fixed-effect recovery and existing interfaces retain
  their contracts. The release includes Tiago Tavares's
  [macOS/runtime, stored-version and interaction fixes](https://github.com/reisportela/xhdfe-xfe/pull/15)
  and [CG precision correction](https://github.com/reisportela/xhdfe-xfe/pull/17).

The maintainer accepts the measured runtime cost versus **2.29.1.20261014**.
On ready, three interleaved pairs per execution variant give a median paired
increase of **54.6%** for native CUDA Comparable and **22.3%** for Stata CUDA
Comparable. Their candidate medians are **11.11 s** and **18.67 s**.
Native CUDA Fast records **+7.3%**; the complete eight-cell table and
measurement conditions are in the [validation record](VALIDATION_2.30.1.20261020.md).
The earlier correct, unpublished implementation took about 24.2 s in native
CUDA Comparable. These results do not establish a universal speedup.

Expected numerical refusals remain explicit. Three difficult synthetic
`xfepout` CUDA cases with 16 workers refuse in both the published baseline and
this candidate after failing the necessary precision check. They leave no
estimates and are counted separately from successful numerical checks.
The full core24 performance campaign remains deferred by the maintainer.

Linux, Windows and native macOS ARM/Intel artifacts require the release
OpenMP, runtime-isolation and exact-artifact gates. Execution of the final
plugins inside licensed Stata/MP on native Mac ARM and Intel is a separate
host check; native SPI probes and Intel/Rosetta runs do not replace it.
The validation record states its actual outcome and any publication exception.
