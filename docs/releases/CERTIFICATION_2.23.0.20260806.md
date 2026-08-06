# Certification — xhdfe 2.23.0.20260806

Date: 06aug2026
Reference release: 2.22.1.20260730
Local target: Linux x86_64, OpenMP, NVIDIA H100 NVL (`sm_90`)

## Verdict

**Local scientific GO.** The candidate preserves the estimator, public
tolerances, convergence limits, interfaces and supported backends. Every
candidate row executed for this release converged and passed the independent
precision certificate. Every CUDA row also reported `gpu_used=1`, backend
`cuda`, status `used`, status code `1`, and converged CUDA absorption.

The public release remains subject to the two-stage release workflow: tag CI
must build the draft assets, and the exact attached CUDA plugins must pass the
maintainer-H100 gate before the matching `publish-v*` tag may publish the
release and `gh-pages` snapshot.

## Scientific stop lines

The seven-case CPU inventory passed its frozen pins. The same seven cases then
ran on CUDA against an independent one-thread CPU `strict-residual(1e-10)`
reference. All cases preserved the sample and stayed below the pre-declared
limits:

- maximum relative coefficient error: `1e-7`;
- maximum relative standard-error error: `1e-7`;
- candidate `abs_residual_rel`: `1e-8`.

The worst CUDA case was `workers/fast`: relative coefficient error
`9.253e-10`, relative standard-error error `8.347e-8`, and
`abs_residual_rel=3.956e-10`. Forced methods remain observable: an explicitly
forced inaccurate sweep is warned and is not silently replaced.

## Core23 matrix

The current release artifacts were exercised on the `core23` surface in all
eight interface/backend/mode cells. Six cells completed all 23 current A/B
rows. C++/CPU comparable completed 22 current rows; the public baseline exceeded
both 420 s and 900 s on `akm_v02_secondreg`, for which the already-custodied
2.22.1 row (same baseline hash and unchanged certified path) remains the timing
reference. Stata/CPU comparable reran the two scientifically affected rows and
uses the complete 05aug2026 matrix for the other unchanged paths.

| Cell | Current candidate evidence | Result |
|---|---:|---|
| C++ CPU fast | 23/23 | all converged and certified |
| C++ CPU comparable | 22/23 current + 1 custodied | all completed rows certified |
| C++ CUDA fast | 23/23 | all certified; real CUDA use |
| C++ CUDA comparable | 23/23 | all certified; real CUDA use |
| Stata CPU fast | 23/23 | all converged and certified |
| Stata CPU comparable | 2 affected current + 21 custodied | affected rows certified |
| Stata CUDA fast | 23/23 | all certified; real CUDA use |
| Stata CUDA comparable | 23/23 | all certified; real CUDA use |

The historical comparator reports functional FAIL where the candidate corrects
an uncertified/loosely certified 2.22.1 result, or where it treats any tiny
increase in an already negligible residual as failure. Those findings are
retained as evidence and are not relabelled PASS. Candidate acceptance instead
uses convergence, the independent certificate, strict-reference stop lines,
sample identity and honest backend diagnostics.

## Performance adjudication

The precision repair necessarily adds work only where 2.22.1 returned a
scientifically inaccurate result. For example, `workers/fast` in 2.22.1 differs
from the strict reference by up to `1.151e-3` in coefficients; MLSMR is the
fastest tested method that removes that error.

Unaffected CPU paths remained essentially stable. A five-pair repeat of
`pf_simple_10m_2fe/comparable`, whose first observation suggested a 22% loss,
reduced the balanced median ratio to `1.0211`. The large protected Stata CPU
fast row `main_95_21_ready` was `185.664 s` versus `184.751 s` (+0.49%), and
`akm_v02_firstreg` was `257.445 s` versus `255.051 s` (+0.94%).

On CUDA, localized losses coexist with materially larger gains. Three-pair C++
comparable medians were `1.0333` for `main_95_21_ready`, `1.0435` for
`akm_v02_firstreg`, `0.9916` for `pf_difficult_10m_3fe`, and `0.9355` for
`akm_v02_secondreg`. `simulated_panel` had a noisy `1.2006` median; profiling
localized the variation to the host-memory-bound CUDA certificate. Its source
and compiled object are byte-identical between baseline and candidate, and the
two versions use the same seven solver iterations and produce the same
scientific result. The timing is therefore recorded as host-memory variability,
not hidden or attributed to a solver change.

## Build, interface and packaging gates

- CTest CPU: 4/4; CTest CUDA: 4/4.
- Fast-math non-finite source/liveness gate: PASS.
- CUDA verifier FMA gate: PASS.
- Python CG, AKM frontend and CUDA-envelope tests: PASS.
- Stata CPU/CUDA audits and the 36-part Stata suite: PASS on the exact final
  plugin hashes carried into this release.
- R CPU and H100 `sm_90` audits/testthat: PASS on the byte-identical final core.
- C++ core mirrors for Python, Stata, R and distribution: byte-identical.
- Final local plugins link OpenMP; local CUDA plugins target `sm_90` and report
  real H100 use.
- A wheel built from the sdist imports both `xhdfe` and the declared top-level
  compatibility module `py_hdfe_v11`; the sdist contains the liveness,
  fail-closed and FMA gate sources/scripts required to configure from source.

Final local artifact identities:

- core source SHA-256: `006595808219de5edfb92c7a3d4af15d2babc755a6694c9b22de2794bf2a2cf1`;
- CPU Python extension: `23cf52a0a8c827e562225f0e2da129a05c4de087e3aeb2e1ae40cdbdfead05a9`;
- CUDA Python extension: `8afa9d85d9ba1238cb52f24d3a4734c9fd43ca6ae5a181645cd66754b920ccc2`;
- local H100 xhdfe plugin: `63be6fb837330544e9374d071ae52c6232c7b9f7e38128de04aaf724822fa4d1`;
- local H100 xfe plugin: `014d5ec3ebddff6037345f383688974cd0e165617554e41e2f4fd562bdbb0519`.
