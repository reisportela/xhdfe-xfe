# Issue 14: Stata fixes and validation, 16 October 2026

This candidate addresses the three complaints in [issue #14](https://github.com/reisportela/xhdfe-xfe/issues/14):
a multithreaded Intel Stata MP crash, stale stored version information, and
rejected categorical interactions with three components. It starts from
`a377a886687c7c05478b17620d5fe9769ecfac5f` (2.29.1.20261014).

All builds and execution in this investigation used the maintainer's local
Apple Silicon Mac or Athena. No external CI was dispatched. The existing
release requirements remain in force; these results do not certify a release.

## Changes

1. **OpenMP runtime isolation.** The macOS runtime builder now exports the
   public OpenMP/compiler ABI using `tools/macos_openmp.exports`, following
   LLVM's `openmp/runtime/src/exports_so.txt`. Private C++ weak template
   definitions and internal `__kmp_*` symbols no longer appear as global
   exports. A new checker validates each thin build and each final universal
   slice. The corresponding-source validator requires both new build inputs.
2. **Version identity.** The version query and OLS/IV result posting share
   `_xhdfe_package_version`. Release metadata validation checks that value
   against the release date/version, and a Stata contract checks the ado
   header, displayed query, OLS, IV, replay and first-stage postestimation.
3. **Categorical absorb interactions.** `absorb(a#b#c)` and longer categorical
   interactions now materialize all components with `egen group()` after
   sample construction. Missing-value screening includes every component.
   Factor prefixes, aliases, categorical `##`, strings, weights and IV
   auxiliaries are covered. The help distinguishes a joint interaction from
   the already supported additive model `absorb(a b c)`. Heterogeneous
   slopes retain their existing two-component grammar.

The C++ estimator, numerical tolerances, fast-mode refinement policy and thread
selection are unchanged. The runtime keeps its own identity and useful worker
teams; the fix does not depend on a single-thread fallback or
`KMP_DUPLICATE_LIB_OK`.

## Crash investigation

The published 2.29.1 macOS plugins were held fixed while the bundled runtime
was replaced. LLVM sources were pinned to
`87f0227cb60147a26a1eeb4fb06e3b505e9c7261` (llvmorg-20.1.8).
The ARM runtime was built natively; the Intel runtime was cross-built on the
same Mac and executed through Rosetta in StataNow MP 19.5.

The original stack and symbol inspection pointed to private weak OpenMP
templates being shared between Stata MP's runtime and the plugin's runtime.
Both libraries exported the same `__kmp_suspend_64<false,true>` template.
Changing the runtime's filename/install name alone had not hidden those symbols.

A controlled relink used the **same compiled Intel runtime objects**, with only
the export filter removed. Five fresh Stata processes per variant then ran the
same eight-fit, four-thread reproducer:

| Runtime | Completed processes | Crashed processes |
|---|---:|---:|
| Published 2.29.1 | 0/5 | 5/5, SIGABRT |
| Locally rebuilt, filter removed | 1/5 | 4/5, SIGABRT |
| Locally rebuilt, filter enabled | 5/5 | 0/5 |

An earlier unfiltered trial also passed, so the failure is intermittent for
that build. Both successful and failing trials were retained. This comparison
supports the private-symbol collision diagnosis and the export-isolation fix;
it does not establish universal compatibility across Stata/macOS releases.

The new export checker rejects both published runtime slices and accepts the
candidate's 771 ARM / 881 Intel public ABI symbols.

## Correctness and host checks

| Check | Local ARM | Local Intel through Rosetta | Athena CPU | Athena CUDA |
|---|---|---|---|---|
| Original reproducer, candidate | 24/24 fits | 24/24 fits | n/a | n/a |
| Real-host OLS, IV, first-stage, xfepout | 60/60 cases | 60/60 cases | n/a | n/a |
| Categorical interaction OLS cases | 26/26 | 26/26 | 26/26 | 26/26 |
| Interaction IV/alias/first-stage cases | 2/2 | 2/2 | 2/2 | 2/2 |
| Version query/result contract | Pass | — | Pass | Pass |
| Existing first-stage and alias suites | — | — | Pass | Pass |
| Existing main Stata suite, testall.do | — | — | Pass | — |
| Existing native plugin SPI probe, both plugins | Pass | Native gate not run | — | — |

The 60-case host runs requested 1, 2, 4 and automatic threads, with repeated
OLS/IV calls in both tolerance modes and repeated residual extraction.
They assert actual worker counts and numerical agreement with one thread.
Automatic-thread cases used two million observations. One-thread reference
fits are correctness controls, not discarded timing warm-ups.

The interaction fixtures cover three/four components, an additional FE, analytic
weights, missing values, singletons, conventional/robust/one-way/two-way
clustered covariance, strings, reordered/repeated components and saved-FE aliases.
Samples and degrees of freedom match the grouped-ID reference.
CUDA cases assert actual GPU use.

| Interaction validation | Largest absolute coefficient difference vs reghdfe | Largest absolute SE difference vs reghdfe | Largest grouped-ID b / full-V relative difference |
|---|---:|---:|---:|
| Local ARM | 2.665e-15 | 1.112e-13 | 0 / 0 |
| Local Intel through Rosetta | 1.998e-15 | 1.112e-13 | 0 / 0 |
| Athena CPU | 1.887e-15 | 1.112e-13 | 0 / 0 |
| Athena CUDA | 1.998e-15 | 1.112e-13 | 1.862e-16 / 6.502e-19 |

These are maxima reported by the synthetic fixture, not an accuracy guarantee
for arbitrary data. Grouped-ID comparisons use the complete default coefficient
and covariance matrices. For the multiway-cluster reghdfe comparison, both fits
use `noconstant`: reghdfe PSD-corrects its full covariance including the
recovered constant, whereas xhdfe's documented D-C policy corrects the slope
block. The existing difference was exposed during test development and is not
changed here. String categories use equivalent numeric IDs for reghdfe.

Frozen 2.29.1 fails the version contract with stale `e(version)` and rejects
the three-component interaction with r(198). These negative controls establish
that the new tests detect the reported defects.

## Focused timing comparison

Two million synthetic observations; OLS, IV with first-stage diagnostics, and
firm-specific slopes; fast mode; tolerance 1e-8; clustering by firm. Three
alternating old/new pairs per configuration, **zero warm-ups, no discarded
fits**, 72 timed fits in total. Every three-fit batch starts a fresh Stata
process. Data generation and process startup are outside the estimator timer.
All scalar output fields are stored as doubles and exported with 17-digit
precision.

Athena retained the 60% CPU-idle admission threshold (observed admission idle:
71.0–85.8%). Another user's GPU allocation remained present; observed GPU
utilization before each batch was 0%. The server was shared.

| Host / backend / threads | Model | Old median (s) | New median (s) | Difference (s) | Change | Old/new iterations |
|---|---|---:|---:|---:|---:|---:|
| Mac ARM / cpu / 1 | ols | 0.468 | 0.466 | -0.002 | -0.4% | 2/2 |
| Mac ARM / cpu / 1 | iv | 1.399 | 1.396 | -0.003 | -0.2% | 2/2 |
| Mac ARM / cpu / 1 | slopes | 0.553 | 0.566 | +0.013 | +2.4% | 2/2 |
| Mac ARM / cpu / 4 | ols | 0.352 | 0.339 | -0.013 | -3.7% | 2/2 |
| Mac ARM / cpu / 4 | iv | 1.084 | 1.070 | -0.014 | -1.3% | 2/2 |
| Mac ARM / cpu / 4 | slopes | 0.343 | 0.344 | +0.001 | +0.3% | 2/2 |
| Athena / cpu / 16 | ols | 0.554 | 0.549 | -0.005 | -0.9% | 2/2 |
| Athena / cpu / 16 | iv | 1.730 | 1.790 | +0.060 | +3.5% | 2/2 |
| Athena / cpu / 16 | slopes | 0.598 | 0.594 | -0.004 | -0.7% | 2/2 |
| Athena / cuda / 16 | ols | 0.647 | 0.601 | -0.046 | -7.1% | 2/2 |
| Athena / cuda / 16 | iv | 1.466 | 1.514 | +0.048 | +3.3% | 2/2 |
| Athena / cuda / 16 | slopes | 0.456 | 0.450 | -0.006 | -1.3% | 2/2 |

All old/new sample sizes, degrees of freedom and iteration counts match.
CPU coefficients, SEs and first-stage F values match exactly in the recorded
outputs. On CUDA, maximum absolute old/new differences are 6.66e-16 for
coefficients, 1.09e-18 for SEs and 6.99e-10 for the first-stage F (about
1.25 million). These are numerical roundoff-sized differences.

Median changes range from -7.1% to +3.5%, with short runs and shared-host noise.
This supports absence of a large regression in the tested cases. It is not
a new full-dataset benchmark or proof of a speedup. Old/new Intel multithreaded
timing is unavailable because the published runtime crashes.

## Packaging and retained evidence

- Mach-O architecture, deployment targets and runtime closure: passed for both
  plugins and both candidate runtime slices; code signature verification passed.
- Existing standalone native ARM plugin probes: passed, including all four
  independent IV Driscoll–Kraay cases.
- Release metadata and maintained source mirrors: passed.
- Local packaging control suite: 16 tests passed. Its old mock predated the
  builders' FMA `lipo -archs` / `arch` checks; the mock now models those calls
  without executing native code. The original failing log was retained.
- Source archive closure, 21 corresponding-source tests, CMake configuration
  and non-finite guard build: passed. The initial default AppleClang attempt
  could not discover OpenMP; explicit paths to the newly built local SDK
  resolved that toolchain configuration, without disabling OpenMP.
- No production installation, release asset or Athena source checkout was
  replaced.

Evidence directory in the maintainer workspace:
`pipeline/results/issue14_fixes_20261016_131014/`. It contains reproducer,
unfiltered relink, host and packaging receipts, all correctness logs, timing
controllers/data, and `quantitative_summary.json`. Raw Stata logs remain local
because they contain installation/license metadata.

Candidate universal runtime SHA-256:
`77209062a9998d69da411d2859b1904ae565dfe98ca1a2451cb6c78cf512fa53`.

The unchanged macOS plugin SHA-256 values are:
- xhdfe: `f4967cb8000bf62a73a35db7f3d9d109194388c865b94b8546966f4cf92ab151`
- xfepout: `87e41792106cb355530488394aef8a1874a753edd547d2a9e7072dbc9f1728a2`

## Remaining release coverage

Native Intel hardware and Stata SE were unavailable. Rosetta is valuable
reproduction evidence but does not replace the native Intel gate. Only
StataNow MP 19.5 was exercised on the Mac; other host versions remain untested.
Windows and the Python/R frontends were not rerun for this Stata-only change.

Before publishing, rebuild the actual release artifacts and repeat the
architecture/runtime gates and real licensed Stata host checks on those exact
bytes, following `.github/RELEASE_OPENMP_CHECKLIST.md`. A new release/version
bump and installation are separate from this source PR.
