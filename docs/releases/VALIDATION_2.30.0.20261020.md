# Validation record — xhdfe 2.30.0.20261020

Status: **published with documented coverage limits**, authorised by the
maintainer on 20 October 2026. The [release](https://github.com/reisportela/xhdfe-xfe/releases/tag/v2.30.0.20261020)
was published at 10:53:51 UTC from public source commit
`1dd3addded1350b656bfda0544e961606b7d0a85`. The final licensed Stata/MP checks on
native Mac ARM and Intel remain **NOT_RUN**, under the explicit publication
exception recorded below. This is not a claim of universal certification.

The operational comparison baseline is the frozen published **2.29.1.20261014**.
The historical precision-contract reference remains 2.28.1.20260929; neither
public tolerances nor acceptance thresholds were changed to obtain a pass.

## Final numerical source

The final source uses absorption cache generation `0x226248414439` and identical
maintained C++ mirrors. Local builds use native CPU tuning, OpenMP and CUDA
`sm_90`. The generation-4439 functional campaign has unchanged pinned inputs.
The release date restamp changes metadata only; its before/after inventory
retains the exact functional-source and binary identities.

| Gate | Result and scope |
| --- | --- |
| C++ CTest | 16/16 passed in main CI and in each of the four canonical CPU/CUDA build directories |
| Python frontend | 97 passed |
| Stata sequential suite | 40 main tests, standalone contracts and 72 IV Driscoll–Kraay cases passed; CUDA regressions require actual GPU use |
| Adapted reghdfe core, CPU | 53 ordinary passes, one expected-difference control and one published-contract pass; three reference-only cases |
| Adapted reghdfe core, CUDA | 51 ordinary passes, one expected-difference control and three published-contract passes; three reference-only cases not applicable; 326 actual CUDA calls |
| Literal diagnostic runs | reghdfe: 52 pass, 6 fail; xhdfe: 42 pass, 13 fail, 3 reference-only passes. These original outcomes are retained and explained by the adaptation ledger |
| Independent analytic campaign | 664 xhdfe passes, 223 expected unsupported cases; T01: 578 passes, 219 expected unsupported cases. Unsupported cases are not numerical successes |
| CG final-step oracle | 18 OLS/IV/order/mode cases passed on each of CPU and CUDA |
| CUDA weak-graph regression | Independent sparse dummy-variable oracle passed; real CUDA, rescaling/order/worker controls and bounded-refusal tests passed |
| Fast regression | Analytic CPU/CUDA controls passed; the real GitHub fixture again returns in 336 iterations. Archived broken binary is rejected by the negative control |
| Full CUDA corpus execution | All 24 specifications × Fast/Comparable completed with actual CUDA. Candidate-only functional preflight; timings are not acceptance evidence |
| R package | Local CUDA audit: 1,229 passes, six warnings, no skips. Final CI package check: two warnings and four notes, no errors; tests and actual-worker probe passed. Optional unavailable Suggestions remain a check limitation |

The additional CTest checks Student-t fixture provenance. A producer-host
compiler hash had incorrectly rejected a different consumer host; the corrected
harness binds the actual CLI reference bytes, retains opt-in strict producer
verification, and rejects forged references and incorrect numerical outputs.
No estimator, numerical reference or tolerance was changed for this correction.

Fast covariance differences on the GitHub fixture remain diagnostics. A generic
per-entry covariance test fails on both 2.29.1 and the fixed candidate and is
not adopted as a universal Fast forward-error contract. Two independently
projected references agree (LSQR/LSMR, high-precision reduced solve); coefficient,
return-status, sample and backend checks pass. Two native `areg` reference
attempts timed out and are recorded as not completed, never as passes.

## Performance scope approved for this release

`main_95_21_ready` × eight execution cells × three interleaved baseline/candidate
pairs = **48 completed fits**, against frozen 2.29.1. Data loading is excluded,
caches are off, processes are fresh, and no warm-ups or completed fits are
discarded. Host and GPU activity are recorded. All fits pass the applicable
sample, convergence, iteration, source and backend checks, with 16 actual workers.

| Execution cell | 2.29.1 median (s) | 2.30.0 median (s) | Median paired change |
| --- | ---: | ---: | ---: |
| Native CPU Fast | 23.076 | 22.641 | -3.80% |
| Native CPU Comparable | 27.717 | 27.539 | -1.97% |
| Native CUDA Fast | 5.310 | 5.083 | -3.44% |
| Native CUDA Comparable | 6.770 | 6.863 | +1.38% |
| Stata CPU Fast | 28.676 | 28.930 | +3.21% |
| Stata CPU Comparable | 34.174 | 34.572 | -0.95% |
| Stata CUDA Fast | 12.349 | 12.058 | -2.36% |
| Stata CUDA Comparable | 13.721 | 13.865 | +1.12% |

The change column is the median of paired ratios, not the ratio of the two
medians. The observed changes are within documented shared-host variability;
no material runtime regression is established and no universal speedup is
claimed. Iteration counts agree in all pairs. CPU coefficients and covariance
are identical to the baseline. Across all cells, maximum current-versus-baseline
scaled differences are `1.0521e-15` for coefficients and `9.0644e-13` for full
covariance. Baseline parity is not an independent mathematical oracle.

**REVIEW_REFERENCE_COVARIANCE:** the approximately `3.2e-8` diagonal-scaled
CPU/CUDA Comparable covariance difference on ready is present in both 2.29.1
and 2.30.0. It remains a missing-coverage finding pending an independent
full-ready covariance adjudication. It is neither a newly demonstrated
estimator error nor a passing independent covariance check.

The full core24 timing campaign is **deferred by the maintainer**. Earlier
partial campaigns were held when an overly broad CUDA route caused a slowdown
and when an added Comparable check caused a Fast refusal. Those changes were
narrowed/corrected; the earlier measurements remain historical and are not
combined with the final ready campaign.

## Platforms, assets and installation

- [Main CI](https://github.com/reisportela/xhdfe-xfe/actions/runs/39397053574)
  passed the final source, including 16 CTests and Python/R checks.
- [Final version-tag workflow](https://github.com/reisportela/xhdfe-xfe/actions/runs/39399375729)
  passed all 13 required jobs. The ARM job was retried after an HTTP 408 while
  fetching pinned LLVM source; the tag and source did not change.
- All 13 downloaded assets match GitHub digests and the checksum manifest.
  Archive/source closure checks passed. Exact native receipts cover ten
  plugin/platform cases, two installed Python wheels and four macOS serial
  negative controls. Both final Mac slices passed native runtime export,
  isolation, useful-worker and numerical probes.
- Exact Linux CPU/CUDA packages passed licensed Stata/H100 analytic, full-V,
  IV, recovery, cache and new CG/CUDA regressions. Both packages passed the
  full Stata suite and 72 IV Driscoll–Kraay cases each. The installed Linux
  wheel passed all 97 frontend tests without skips and the new CG contracts.
- A separate full-ready smoke of the exact portable public CPU plugins passed
  b/V, sample and actual-worker checks: 42.918 s for 2.29.1 and 44.130 s for
  2.30.0, 23 iterations and 16 workers each. This single pair verifies the
  shipped artifact; it is not a separate performance certification.
- [Publication](https://github.com/reisportela/xhdfe-xfe/actions/runs/39411095882)
  published the tested draft and exact net-install snapshot. A fresh isolated
  Stata installation passed its version and analytic smoke. All 49 public
  site files and 37 installed/ancillary files matched the authenticated
  snapshot; the 13 public assets retained their verified IDs and digests.
- Both canonical `main` checkouts contain identical common production source
  files. Their CPU and CUDA `sm_90` builds, active Python imports and Stata
  plugins passed fresh-process checks, including requests for 1, 2, 8, 16, 24
  and 48 useful workers. Public portable release binaries and locally tuned
  H100 binaries have different build targets and are validated separately.

**Explicit maintainer exception, 20 October 2026:** the maintainer authorised
publication with the documented limits ("Publicar com os limites documentados").
Final licensed Stata/MP execution on native Mac ARM and Intel remains
**NOT_RUN**. Tiago's prior source-candidate evidence covers native ARM and
Intel/Rosetta; native SPI probes are not Stata/MP host-coexistence tests.
The issue-14 runtime fix is included, but the crash is not independently
certified closed on native Intel Stata/MP. The exception permits this release;
it does not turn either unexecuted host check into PASS. The inherited ready
covariance review and deferred full core24 timings remain explicit limits.

Private reproducibility evidence is under
`Verifications/integracao_pr15_pr17_20261019/`, including
`release_2_30_0_20261020/`. Its final manifests record source and binary hashes,
local/online main alignment, test receipts and preserved local work. No
confidential source records are published.
