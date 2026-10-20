# xhdfe 2.30.0.20261016

This release addresses [issue #14](https://github.com/reisportela/xhdfe-xfe/issues/14),
reported by a Stata/MP user through Tiago Tavares, and integrates
[PR #15](https://github.com/reisportela/xhdfe-xfe/pull/15) and
[PR #17](https://github.com/reisportela/xhdfe-xfe/pull/17) with follow-up work.
This is an unpublished candidate; its date will be restamped when released.

- **Stata/MP on macOS no longer crashes with more than one OpenMP thread.** The
  bundled LLVM OpenMP runtime `xhdfe_libomp.dylib` is linked with an export list
  restricted to the public OpenMP/compiler ABI (`tools/macos_openmp.exports`,
  following LLVM's `exports_so.txt`, which LLVM applies on Linux but not on
  Darwin). The 2.28.1 and 2.29.1 runtimes exported their private C++ template
  instantiations as weak definitions; dyld coalesced them with Stata/MP's own
  `libomp.dylib`, so a worker thread of this runtime executed the host runtime's
  `__kmp_suspend_64<false,true>` on this runtime's state and the process aborted
  (SIGABRT 134 or SIGSEGV 139, uncatchable by `capture`). Stata/SE, Linux (GCC
  libgomp) and Windows were never affected. On StataNow MP 19.5 the published
  runtime crashed 5/5 eight-fit processes at four threads and the filtered
  runtime completed 5/5, with 60/60 real-host cases on Apple Silicon (native)
  and Intel (Rosetta). New gates: `tools/check_macos_openmp_exports.py` on each
  thin slice and on the installed universal, with every OpenMP symbol the plugins
  import required among the runtime's exports; `MH_WEAK_DEFINES` refused by the
  portable `tools/validate_macos_openmp.py`; `tools/run_stata_openmp_host.py`
  for licensed Stata hosts. File names, install names and package layout are
  unchanged. Native Intel hardware was not available during development; see
  the validation record for the remaining coverage.
- **`e(version)` after a fit reports the release version.** 2.29.1 stored
  `2.29.0 14oct2026` after estimation while the ado header and `xhdfe, version`
  were correct. The query and the estimation result now share one helper
  (`_xhdfe_package_version`; likewise `_xfepout_package_version`),
  `tools/validate_release_metadata.py` checks those literals against the release
  identity, and `tests/stata/version_contract.do` checks the header, the query,
  OLS, IV, `estat firststage` and replay.
- **Categorical interactions with three or more components.** `absorb(a#b#c)`,
  `vce(cluster a#b#c)` and the `xfepout` `absorb()`/`clustervars()` grammar
  accept any number of categorical components, materialised with `egen group()`
  over all components after the estimation sample is defined; the result is
  identical to grouping the components manually. Categorical-only `##` in
  `absorb()` remains one joint fixed effect; in `cluster()` and `clustervars()`
  two-component `##` keeps its expansion to main effects plus interaction and
  `##` with more than two components is refused with a message asking for the
  dimensions to be listed. Heterogeneous slopes keep exactly one categorical and
  one continuous component. New tests: `tests/stata/absorb_interaction_contract.do`
  and `tests/stata/cluster_interaction_contract.do`.
- The documented factor prefixes `ib#.` and `b#.` now parse in interaction terms of
  `absorb()`, `cluster()` and the `xfepout` options; a double-escaped regular
  expression had always let them reach `unab` and fail with r(101).
- `vce(cluster a##b)` keeps its documented expansion to the two main dimensions plus
  their interaction; the help now states that `reghdfe` treats the same term as one
  joint factor, so `a#b` is the portable spelling. `##` with more than two components
  is refused in cluster options.
- `xfepout` 1.13.4: interaction grammar, factor prefixes and version helper as above.
- `tests/test_macos_openmp.py` packaging controls repaired after the FMA liveness
  gate changed the builders' tool calls; the suite had been failing unnoticed
  because `pytest` skips it without the documented `--out` invocation.

- Nearly redundant fixed effects: solved right-hand sides are frozen during
  CG acceleration; breakdown restarts from original input within the remaining
  iteration budget. A proposed final step inconsistent with the same public
  tolerance is rejected before application. This corrects an independently
  demonstrated full-covariance error without changing public tolerances.
- Persistent caches from previous CG implementations are invalidated.
  Independent OLS/IV covariance, ordering, budget and cache tests cover CPU
  and CUDA; group/individual interaction coverage is extended.

The absorption changes require a new core24 performance campaign. See
[current validation and remaining gates](VALIDATION_PR15_PR17_20261019.md),
the [historical PR #15 record](VALIDATION_2.30.0.20261016.md), and
the development report of PR #15,
[candidates/ISSUE_14_STATA_FIXES_20261016.md](candidates/ISSUE_14_STATA_FIXES_20261016.md).
