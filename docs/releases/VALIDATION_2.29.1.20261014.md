# Validation — xhdfe 2.29.1.20261014

**Published 14 October 2026 at 18:17:43 UTC:**
[v2.29.1.20261014](https://github.com/reisportela/xhdfe-xfe/releases/tag/v2.29.1.20261014),
at public source commit `8ce9b96cd2b1deca6ff052cead8bfbffbc7a67ef`.
All 13 exact assets and a fresh public Stata installation were verified.

The initial local correction checkpoint was validated before
publication, from private/public base commits
`a332096e67db877b612bfa16a2b6146e2a6fda4f` and
`78b1d2966cfec7b956cd3460e86aaea69d107018`. The tables below identify that local
checkpoint; the additional platform work and completed release gates are
recorded below.

Evidence is retained in the private repository under
`Verifications/remediacao_main_2291_20261014/`. The earlier adversarial audit and
its failed counterexamples were preserved unchanged.

## Corrections

| Finding | Change | Final evidence |
|---|---|---|
| F1: wrong IV DK covariance with common regressor means | Retain time-cluster influences from the existing wide IV factorisation, form the recovered constant before rounding, and use those influences in the Bartlett covariance | At the original bandwidth-three counterexample, the scaled error falls from `0.0878038099` to about `3.4e-15`; CPU/CUDA and both tolerance modes |
| F2: named saved effects make first stages fail with r(110) | Auxiliary regressions inherit absorbed terms without output aliases, preserving time-series expressions | Eight alias scenarios per backend, including interactions, slopes, lags, weights, DK, replay and `nosample`, pass in both checkouts |
| F3: the evaluator accepts a wrong/non-finite OLS constant | Export and grade complete parameters/V, identified contrasts and the recovered constant; add constant and cross-covariance mutations | Fourteen actual native/Stata positive/negative controls pass; wrong constants, NaN, wrong constant variance/cross-covariance and missing complete evidence are rejected |
| F4: no CUDA/Fast jobs in the main battery | Add the applicable Fast/CUDA rows | 45 additional planned jobs; all eight interface/backend/mode combinations represented |

No solver tolerance, iteration target, estimator definition, default output
format or Python API was changed. The additional wide influence calculation
is requested only by IV DK covariance. Other covariance paths retain their
existing calculation.

## Final functional checks

| Check | Result |
|---|---|
| Full analytic campaign, 25 fixtures / 1,029 jobs | xhdfe: **664 numerical PASS, 223 expected refusals, zero failures**; custody unchanged |
| T01 mode-specific layer | 578 PASS, 219 expected refusals; joint fitted-value metric remains a diagnostic, not an adopted gate |
| Python suite | 176 passed, 17 skipped, 102 subtests |
| Main Stata suite | 40 tests; required completion marker present |
| Rebuilt CTest targets | 10/10 passed |
| DK extended independent MP90/MP120 guard, each canonical plugin | 72 CPU PASS; 66 CUDA PASS and 6 documented no-FE CUDA refusals |
| Existing DK oracle | 39/39 CPU PASS |
| IV multiway covariance, Comparable | 44/44 CPU and 44/44 CUDA PASS |
| Absorption-cache identity/corruption/concurrent publication | 38/38 PASS |
| IV cache, including one-ULP mutations | 16/16 PASS |
| FE recovery units and methods | Six CPU and six CUDA cases PASS |
| Explicit native CPU thread requests | 1, 2, 8, 16, 24 and 48 effective workers, numerical parity |
| Active package modules in both checkouts | Analytic CPU/CUDA checks with 1, 2, 8, 16, 24 and 48 useful workers; real CUDA use |
| Metadata, source mirrors and Stata help | Unified version/date checks and SMCL lint PASS; nine help files translated by Stata |
| R manuals | Both main Rd files parsed and rendered to HTML |

The main campaign's comparator results are retained separately: 25 explicit
OLS PASS; reghdfe has 77 PASS, 7 expected refusals and 33 outcomes that do not
satisfy this oracle/contract. They do not constitute xhdfe failures. The
extended covariance check includes xhdfe's documented D-C convention.

The complete parameter check retains the historical battery's declared
numerical thresholds; a campaign PASS does not replace every Precision
Contract requirement. The separate DK/IV guards use the predeclared `1e-9`
coefficient and `1e-8` diagonal-scaled full-covariance limits.

## Large CPU artifact smoke

No confidential observations were used. A deterministic synthetic design has
**48,000,000 observations**, three absorbed dimensions, four orthogonal
within-regressors and known coefficients/RSS. The old private 2.29.0 plugin was
frozen before editing. Each arm ran in a fresh Stata process; no build or other
validation campaign ran concurrently with these measurements.

| Arm | Fit seconds | Iterations | Observed useful workers | b/V difference from frozen arm |
|---|---:|---:|---:|---:|
| Frozen private 2.29.0 | 17.396 | 18 | 16 | reference for parity |
| Corrected private 2.29.1 | 16.167 | 18 | 16 | 0 in exported values |
| Corrected public-checkout 2.29.1 | 15.938 | 18 | 16 | 0 in exported values |

The coefficient/RSS checks also pass their independent analytic truth. Samples
and degrees of freedom agree. This is an operational large CPU check and one
sequential observation per arm, **not a new speedup or core24 performance claim**.
See `LARGE_SYNTHETIC_CPU_RESULTS.json` for host context and artifact hashes.

## Build and help provenance

Both canonical `build/` and `build_cuda/` directories were rebuilt in Release
mode with required OpenMP and native CPU tuning; CUDA targets sm_90. The Stata
plugins use the required CUDA/OpenMP/native recipe. FMA and non-finite guards
passed, as did the verifier device-code gate. `xfepout.plugin` is unchanged:
the absorption implementation it uses was not changed by this correction.

| Artifact | SHA-256 |
|---|---|
| Private Stata plugin | `f1880c48399c2b45e8bb230d2d12ee92dd66fcd48efd88a9961e926cd0d66c40` |
| Public-checkout Stata plugin | `cac1eae267685cd4779eaca4605ee5b129064f41ab7c27c1ff0bd0da723b129a` |
| CPU module, both checkouts | `54e04347987e1cc73004c134ad8bbeddbc2584bfd59daead09ebabb33b21fe9d` |
| Private CUDA/package module | `8ccac8531f100daeee1fa3de550176398affad68898d0159c73db737e588b948` |
| Public CUDA/package module | `14c0c50be9f26a62f9e11076b3a33f6abf54eafdb517157eb5d915e22b04d0ce` |

Stata, Python and R help versions were aligned. Outdated Python/R descriptions
of Fast slope polishing and the four previously refused CUDA benchmark cells
were corrected. Stata help now covers the explicit `noabsorb` and `constant`
options and the corrected first-stage/DK behaviour. The offline source mirror
also contained an old `xhdfe_estat` and outdated companion dates; these now
match the canonical files. The alignment guard checks all installed ado/help
files instead of only `xhdfe.ado` and `xhdfe.sthlp`.

Stata batch mode ignores requests to open the GUI Viewer. The checks here are
source lint, path/version verification and actual SMCL-to-text translation,
not a claim to have visually inspected the GUI.

## Intermediate failures retained and adjudicated

- The first complete-covariance oracle assumed that the multiway matrix was
  positive definite. It was corrected to implement the existing D-C policy;
  the estimator and thresholds were not changed for this harness failure.
- The first full campaign exposed loss of bits when a Stata local macro
  transported `_cons`. It now uses a scalar. The failed campaign remains in
  `analytic_final/`; the successful fresh campaign is `analytic_r2/`.
- The reserved permutation test initially allowed `xtset` to undo the row
  permutation. Its corrected execution is retained separately.
- Public source copying initially preserved mtimes, allowing CMake to reuse
  old objects. Cache version/compile-log inspection detected this before
  acceptance. The reviewed files received current mtimes and both modules
  were rebuilt with explicit evidence that `ols.cpp` and
  `hdfe_regressor_v11.cpp` compiled. `PUBLIC_NATIVE_FRESH_RESULT.json`
  supersedes those initial native build steps; no cache cleaning was needed.

## Additional platform corrections and release gates

The new native DK IV gate exposed two further errors. MinGW's software FMA
gave a coefficient of `0.50000003803288495` instead of exactly `0.5`; the
same value was independently reproduced with the upstream runtime's arithmetic
sequence. Clang fast-math discarded a wide product's compensation term, giving
`0.30298477495325127` on macOS Intel and in an independent Linux/Clang build.
The ordinary one/two-worker fixtures had passed in both cases.

The MinGW correction replaces only the wide product's required roundoff
operation. The Clang correction scopes precise arithmetic to the wide-number
definitions. Neither adds a fit-time verifier, changes tolerances or increases
the solver's iteration target. The GCC Linux/CUDA route retains its existing
arithmetic. The Windows native gate passed the correction; the corrected local
Clang plugin passed all four DK IV mode/thread cases. The final native macOS
and version-tag gates subsequently passed, as recorded below.

Permanent exact-rational tests cover 37,632 product/range cases and the complete
wide-pair value. GCC, Clang and Clang UBSan pass; an uncorrected Clang negative
control fails. At subnormal halfway cases, the grader compares the exact sum
of the high/low words because the pair's representation is not unique.

An isolated R source build/check passed its audit and 1,181 testthat assertions
with zero failures, six test warnings and seven skips (five unavailable AKM
data cases and two CUDA cases in a CPU build). `R CMD check` reports one WARNING
for source header filenames and one NOTE for C++ console symbols. The installed
package passed the independent one/two-worker analytic gate. This is not a
claim of clean CRAN validation or native Windows/macOS R execution.

The maintainer explicitly waived **only a new core24 × 8 timing campaign** for
this corrective release on 14 October. Numerical, multiplatform, CUDA and
fresh-install gates remain mandatory. Fresh synthetic artifact smokes do not
establish a new core24 performance claim. The Windows R build-configuration
stamp remains a separately scoped follow-up.

The prior active-checkout manifest was already locally modified and sealed
2.29.0. Its exact bytes were preserved for recovery. The final 2.29.1 seal
follows publication, public readback and final documentation commits, and
includes the ignored active native modules as well as the source/help files.

## Exact release-artifact closure

| Gate | Result |
|---|---|
| [Preflight 38598859387](https://github.com/reisportela/xhdfe-xfe/actions/runs/38598859387) | All 11 required build/platform/assembly jobs passed |
| [Version-tag build 38604829130](https://github.com/reisportela/xhdfe-xfe/actions/runs/38604829130) | All 12 required jobs passed; fresh draft artifacts |
| Asset authentication | All 13 GitHub digests agree; the 12 payloads also match `SHA256SUMS.txt` |
| Native final-byte checks | Ten plugin/platform cases, two Python wheels and four macOS serial negative controls; observed useful workers and analytic agreement |
| [Native arithmetic 38597574837](https://github.com/reisportela/xhdfe-xfe/actions/runs/38597574837) | 37,632 exact-rational cases on GCC 13, Clang 18, MinGW 13, and Apple Clang 17 Intel/ARM; [receipt](NATIVE_ARITHMETIC_2.29.1.20261014.json) |
| Exact Linux Stata packages | Analytic b/full-V, partial-out, recovery and cache gates passed; real H100 use |
| Exact DK IV extended guard | 72 CPU PASS; 66 CUDA PASS and six documented no-FE CUDA refusals; maximum scaled full-V error about `3.24e-9`, below the fixed `1e-8` limit |
| Exact first-stage alias guard | Eight scenarios per CPU/CUDA backend passed |
| Installed Linux Python wheel | 97 frontend tests passed, no skips; actual one/two-worker gate passed |
| Tagged R source package | CI build/check/tests and one/two-worker gate passed; two warnings and four notes remain explicit qualifications |
| [Publication 38613871888](https://github.com/reisportela/xhdfe-xfe/actions/runs/38613871888) | Existing draft and existing net-install snapshot published without rebuilding |
| Fresh public readback | 49 site files, 22 installed runtime files and 15 ancillary files match the authenticated snapshot |

The final canonical CPU/CUDA builds passed 11/11 CTests in each of the four
build directories. Fresh imports and Stata processes passed CPU/CUDA analytic
checks in both checkouts, including native requests of 1, 2, 8, 16, 24 and 48
workers. All ten canonical GNU artifacts remained byte-identical to the local
remediation artifacts; the earlier full analytic campaign therefore still
identifies their exact bytes. The current shared source/help surface is aligned.
Three unchanged historical release-note differences are retained for separate
review; they are outside the current command/help comparison.

The exact portable CPU release artifacts were additionally compared in five
interleaved pairs, each using 48 million synthetic observations and a fresh
Stata process. All ten fits returned identical b/V, sample and 18 iterations
with 16 actual workers. The paired time ratio's median was `0.986`; individual
ratios ranged from `0.880` to `1.155`, with variable timings in both arms. No
build or other local campaign overlapped those timings. These are operational
artifact checks, not a speedup or new core24 performance certification.

The native Windows/macOS gates exercise the SPI and estimator, including the
new IV/DK counterexample; they do not replace a licensed Stata GUI/ado suite on
each OS. Linux Stata/H100, the installed Python wheel and the R package have
their separately recorded checks. No universal numerical certification is claimed.
