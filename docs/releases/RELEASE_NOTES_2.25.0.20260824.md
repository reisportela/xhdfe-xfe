# xhdfe 2.25.0 / xfepout 1.12.0 - 24aug2026

Stata frontend rename, numerical fail-closed hardening, and autonomous
source/release validation release. The estimator definition, objective,
supported feature surface, public Python and R APIs, default output formatting,
and convergence targets are unchanged. No tolerance is relaxed.

## Stata partial-out command renamed to `xfepout`

The standalone Stata partialling-out command is now `xfepout`. It replaces the
former `xfe` command and ships as `xfepout.ado`, `xfepout.sthlp`,
`xfepout.pkg`, and `xfepout.plugin`. The syntax and numerical implementation
remain the same, while command identity, stored `e(cmd)`, diagnostics, cache
defaults, build scripts, release assets, package manifests, tests, and online
installation metadata now use the unambiguous `xfepout` name.

This is an intentional Stata-only migration. The release does not ship a
permanent `xfe` alias, and release validation rejects retired `xfe` frontend
files. Users with a previous standalone installation should run:

```stata
ado uninstall xfe
net install xfepout, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
```

No Python or R API is renamed. The shared C++ source file name
`xfe_plugin.cpp` remains an internal build detail; the installed plugin and
all public Stata surfaces are `xfepout`.

## Weighted CUDA precision repair

Weighted CUDA absorption now has a final strict-residual repair route when a
genuine GPU solve satisfies the cheaper native update criterion but misses the
independent public precision certificate. The retry:

- remains on CUDA and is attempted only after the cheaper repair is
  insufficient;
- switches to strict residual certification and tightens a positive tolerance
  to at most `1e-10`;
- accumulates the full GPU iteration count in the public diagnostics; and
- still fails closed if convergence, CUDA use, or precision certification is
  not obtained.

Successful unweighted paths and weighted paths that already certify do not
enter this retry.

## Non-finite guards under optimized builds

GNU/Clang builds now combine `-ffast-math` with
`-fno-finite-math-only`, preserving the package's explicit NaN/Inf rejection
contract under optimization. CMake, direct Stata builds, Python extension
builds, and R package builds use the same rule. Native Stata build scripts also
compile and execute the IEEE bit-level liveness probe before producing an
installable non-Windows plugin. Cross-built Windows binaries remain covered by
the source and package validation gates because they cannot be executed on the
Linux build host.

## Source, packaging, and cross-platform validation

The release machinery now:

- builds and stages `xfepout` consistently for Linux CPU/CUDA, Windows, and
  universal macOS packages;
- packages the fixtures and optional test dependencies required for autonomous
  source checks;
- records the R reference-math contract and reports runtime thread limits
  without assuming that every platform can form the requested team;
- removes Bash `mapfile` dependence from Stata site staging/validation paths;
- verifies that retired `xfe` files cannot enter source archives or the online
  Stata site; and
- removes the obsolete generated `xhdfe_py_hdfe_v11_help.html` file in favour
  of the maintained packaged Markdown help.

The root README is streamlined around installation, minimal examples,
documentation, validation boundaries, provenance, and acknowledgements.

## Validation evidence

- Public PR #10 job `32667748660` passed the Linux C++/Python build, maintained
  source-mirror check, source installation, fast Python contract validation,
  Linux OpenMP Stata-plugin build, and R installation/test suite.
- Unified release metadata passed for `2.25.0.20260824`, including 20 production
  Stata text files dated `24aug2026`, exact maintained source mirrors, exact
  project/Stata LICENSE and NOTICE copies, and mandatory README credit to
  Alexander Fischer and Kristof Schröder.
- Local Release validation passed 4/4 CTest gates, 129/129 Python unit tests,
  the fast Gelbach help/provenance contracts, 44/44 Windows and corresponding-
  source packaging tests, 36 Stata certification groups plus the Gelbach
  bootstrap/reporting smoke, 1163 R CPU assertions, and 1182 R CUDA assertions.
- The autonomous `xhdfe-src.zip` closure passed archive integrity, offline
  dependency, 21 corresponding-source, Python compile, CMake non-finite guard,
  license-hash, and Rcpp provenance gates; the validated local archive SHA-256
  was `793400e2aed5caff8cea890e7e26823ab468e54a41a4c3c49430e2ba8945c560`.
- Both Stata plugins built with OpenMP and CUDA `sm_90`; `libgomp` linkage,
  embedded `sm_90` cubins, IEEE guard liveness, and PTX/SASS verifier rules
  passed. Host probes identified one idle H100 NVL and `cudaGetDeviceCount=1`.
- Weighted `xfepout` CPU/CUDA checks passed for analytic and frequency weights
  with real CUDA use; maximum transformed-variable differences were
  `1.25e-14`. Weighted Gelbach CPU/CUDA matrix checks and AKM/KSS CPU/CUDA
  checks also passed; AKM/KSS's maximum component difference was `1.11e-16`.
- The large V08 Stata gate used 46,156,187 estimation observations and 32
  effective threads: CPU `65.078 s`, CUDA `16.476 s`, both converged; CUDA use
  was real and coefficient/SE differences were at most `2.35e-13`/`1.02e-12`.
- The full `core23` crossed with C++/Stata, CPU/CUDA, and fast/comparable modes
  passed its functional contract. CUDA single-pair residual/iteration signals
  were adjudicated with repeated self-control envelopes and clean-GPU
  telemetry; no estimator tolerance was widened. A private harness correction
  treats two coefficient/SE self-envelopes below `8 * machine epsilon` as
  representation-equivalent while remaining fail-closed above that floor.

## Performance evidence and trade-offs

The full-matrix timings were collected as fresh, interleaved baseline/candidate
pairs against release 2.24.2 built with the same GCC/OpenMP/CUDA `sm_90`
toolchain. Timing is not the functional acceptance gate, and pairs with more
than three percentage points of background-load imbalance were excluded from
balanced summaries.

- Balanced one-pair aggregate ratios (candidate/baseline) were `1.015` and
  `0.990` for C++ CPU fast/comparable, `0.994` for C++ CUDA fast, `1.003` for
  Stata CPU fast, `0.963` for Stata CUDA fast, and `1.028` for Stata CUDA
  comparable. C++ CUDA comparable and Stata CPU comparable needed targeted
  repetitions because their initial aggregates were dominated by one noisy
  large case or too few balanced pairs.
- Three-pair repeats found a localized Stata CPU-fast slowdown of about 7.5%
  on the 173-million-row simulated panel, while the same panel's Stata CPU-
  comparable median was about 1.3% faster and its Stata CUDA-comparable median
  about 0.9% faster. The difficult 10-million-row 3FE DGP was about 2.2% slower
  in Stata CPU fast and 7.6% slower in Stata CPU comparable.
- The first AKM Stata CUDA-fast repeat was about 6.6% slower. Five-pair CUDA-
  comparable envelopes measured about +3.0% on the first AKM and +3.5% median
  on the difficult 3FE DGP. Other repeated cells showed sign reversals or
  parity; in particular, the apparent 20% C++ CUDA-comparable slowdown on the
  second AKM reversed direction across interleaved pairs and is not a stable
  estimate.

These are explicit workload-specific trade-offs of preserving fail-closed
NaN/Inf validation under optimized builds; they are not a universal speedup
claim. No feature, precision, estimator definition, convergence target, or
backend-use contract regressed. Final publication remains gated on the tagged
cross-platform workflow, exact H100 plugin read-back, release-asset checksums,
and the online `gh-pages` package surface.

## Version scope

- Shared C++/Python/R package and release tag: `2.25.0.20260824`.
- Stata `xhdfe`, `xhdfe_p`, `xhdfe_estat`, and `xhdfegpu`: `2.25.0`.
- Stata `xfepout`: `1.12.0`.
- Other companion-command feature versions are unchanged.
- All production Stata text files carry the release date `24aug2026`.

The `xfepout` rename and associated hardening were contributed by Tiago
Tavares in public PR #10; the release preserves that authorship and credit.
