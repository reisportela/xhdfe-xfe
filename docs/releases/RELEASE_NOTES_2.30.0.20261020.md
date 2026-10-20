# xhdfe 2.30.0.20261020

This version integrates Tiago Tavares's [PR #15](https://github.com/reisportela/xhdfe-xfe/pull/15)
and [PR #17](https://github.com/reisportela/xhdfe-xfe/pull/17), with corrections
and permanent regression tests from the private development repository.

- **macOS Stata/MP runtime isolation.** The bundled LLVM OpenMP runtime exposes
  only the public OpenMP/compiler ABI, preventing its private weak C++ symbols
  from coalescing with Stata's runtime. Both architectures retain OpenMP;
  the fix does not force single-thread execution. The source investigation
  reproduced and resolved the crash on Apple Silicon and Intel under Rosetta.
  Native Intel Stata/MP validation of the final release artifacts is recorded
  separately; Rosetta is not a substitute for that test.
- **CG precision and restart budget.** Solved right-hand sides remain frozen,
  unstable steps are rejected before application, and cold restarts use the
  original data within the remaining iteration budget. Comparable/Strict also
  check the proposed final step against their existing stopping tolerance.
  Fast retains its published energy-based stopping contract; it is not forced
  through the additional Comparable check. Old absorption caches are invalidated.
- **CUDA absorption.** Comparable cold recovery preserves the existing tolerance
  floor and uses the device solver where applicable. Ordinary, unweighted,
  sparse two-effect Auto fits use the existing device LSMR operator after exact
  shortcuts, with the ordinary Krylov floor and original sample/DoF semantics.
  Independent dummy-variable references cover the previously incorrect result.
  Other primary Auto paths and the Fast policy are unchanged.
- **Stata results and syntax.** `e(version)` agrees with the version query;
  categorical interactions accept three or more components in `absorb()`,
  `cluster()` and `xfepout`; documented `ib#.`/`b#.` prefixes and `ab()` work.
  Ordinary saturated fits report undefined adjusted R-squared as missing under
  the reghdfe-style convention. Existing group/individual and legacy conventions
  are preserved. `xfepout` is version **1.13.4**, dated 20 October 2026.
- **Permanent test core.** `tests/core` combines the core24 benchmark controller,
  an immutable MIT-licensed reghdfe test snapshot, documented compatibility
  adaptations and independent numerical/negative controls. Reference-only and
  unsupported cases remain distinct from numerical PASS results.

For this release, the maintainer explicitly deferred the complete core24 timing
campaign. The performance gate is `ready` across native/Stata, CPU/CUDA and
Fast/Comparable, with three interleaved pairs against **2.29.1.20261014**.
No full core24 performance certification or universal speedup is claimed.

See the [validation record](VALIDATION_2.30.0.20261020.md) for results, exact
artifact checks and remaining limitations. Historical source investigations:
[macOS runtime](candidates/ISSUE_14_STATA_FIXES_20261016.md),
[CUDA weak graph](../LICOES_CUDA04_20261020.md), and
[Fast stopping policy](../LICOES_CG05_FAST_20261020.md).
