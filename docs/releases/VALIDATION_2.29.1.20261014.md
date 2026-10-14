# Local validation — xhdfe 2.29.1.20261014

14 October 2026. **Local corrections validated; not published.** The private
and public canonical working trees contain the same production source/help
changes. Their Git HEADs remain `a332096e67db877b612bfa16a2b6146e2a6fda4f`
and `78b1d2966cfec7b956cd3460e86aaea69d107018`; no commit, push or release was made.
The published version remains 2.29.0.20261014.

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

## Remaining release gates

**NOT_RUN in this task:** fresh native Windows/macOS validation, R numerical
package installation/checks, sanitizers and a new complete core24 × 8 timing
campaign. No universal numerical or performance certification is claimed.
The Windows R build-configuration stamp remains a separately scoped follow-up.

The existing active-checkout manifest was already locally modified before this
task and still seals the published 2.29.0 state. It was preserved; expected
candidate/source/version mismatches must not be hidden by refreshing its hashes.
The new candidate's provenance lives in the separate remediation receipts.
Publication requires its own reviewed gates and authorisation.
