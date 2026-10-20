# Validation record — xhdfe 2.30.0.20261020

Status: final source validation completed; ready timings and final release
artifact/platform validation are in progress. Publication is not yet certified.

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
| C++ CTest | 15/15 passed |
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
| R package | Required CUDA audit passed; package check completed with two existing package warnings (source filenames and compiled output streams), no errors. Optional unavailable Suggestions remain a check limitation |

Fast covariance differences on the GitHub fixture remain diagnostics. A generic
per-entry covariance test fails on both 2.29.1 and the fixed candidate and is
not adopted as a universal Fast forward-error contract. Two independently
projected references agree (LSQR/LSMR, high-precision reduced solve); coefficient,
return-status, sample and backend checks pass. Two native `areg` reference
attempts timed out and are recorded as not completed, never as passes.

## Performance scope approved for this release

`main_95_21_ready` × eight execution cells × three interleaved baseline/candidate
pairs = 48 fits, against frozen 2.29.1. Data loading is excluded, caches are off,
processes are fresh, and no warm-ups or completed fits are discarded. Host and
GPU activity are recorded. Final results will be added before publication.

The full core24 timing campaign is **deferred by the maintainer**. Earlier
partial campaigns were held when an overly broad CUDA route caused a slowdown
and when an added Comparable check caused a Fast refusal. Those changes were
narrowed/corrected; the earlier measurements remain historical and are not
combined with the final ready campaign.

## Platforms, assets and installation

Final version-tag artifacts, native OpenMP worker checks, H100 validation of the
exact CUDA assets, public installation readback and canonical local/online
alignment are pending. The macOS source fix has real Stata/MP evidence on ARM
and Intel/Rosetta. Native Intel Stata/MP on final assets is still unexecuted;
the crash is not claimed independently closed on that hardware.

Private reproducibility evidence is under
`Verifications/integracao_pr15_pr17_20261019/`, including
`release_2_30_0_20261020/`. No confidential source records are published.
