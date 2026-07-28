# xhdfe 2.18.0.20260711 — companion GPU features and fail-loud inference

This release ships the optimized `xhdfeakm`, `xhdfeconnected`, and
`xhdfegelbach` companion stack and applies the independently reproduced
findings from the 11 July 2026 adversarial pytwoway/precision audit.

## Correctness and diagnostics

- Match-level `var(alpha)` component SE/CI are now missing with an explicit
  note on every match-level sample. This follows canonical
  `leave_out_COMPLETE`, which reports only `var(psi)` and
  `cov(alpha,psi)` inference there. The removed movers-only extension had
  SE/empirical-SD ratios of 0.56–0.65 and AM coverage of 0.68–0.77.
  Observation-level inference is unchanged and remains the supported
  `var(alpha)` surface.
- Negative simulated component-variance estimates retain the established
  `SE=0` truncation but now emit a note/warning. Undefined Andrews–Mikusheva
  intervals return two missing bounds; the former `-1e300` sentinel can no
  longer reach users.
- AKM controls omitted for collinearity identify their one-based columns in
  `notes`/`r(notes)` and trigger Python/R warnings.
- Gelbach detects severely near-collinear observed x2 blocks with a
  bounded-cost diagnostic. Values are unchanged, while notes/warnings make
  tolerance- and rounding-sensitive block SEs explicit.
- Python's lazy `from xhdfe import akm, gelbach` loader no longer recurses,
  and both secondary Gelbach validators accept `--module-dir` so they cannot
  silently exercise a stale extension.

## GPU, verbose, and performance

- `xhdfeconnected` adds `threads()`, `gpu`, `verbose`, real-backend
  diagnostics, and deterministic CUDA radix sorting above its measured 10M
  row profitability gate.
- `xhdfegelbach` adds `gpu` and `verbose`, uses CUDA for profitable full-model
  absorption, keeps certified FE recovery/covariance on CPU, and reports the
  effective backend.
- Gelbach cluster VCE streams cluster scores instead of materializing the
  full stacked-score matrix. FMA contraction is intentionally accepted: on
  converged audited cells it changes only the final SE ulp (maximum
  `1.388e-17` absolute), while coefficients and deltas remain bit-identical.
- The local Linux plugin is CUDA `sm_90`, OpenMP, and `-march=native`. A
  paired 46.2M-row smoke took 76.418s versus 77.535s for the 2.17.1 plugin on
  the same host; coefficients/SEs match.

## External-oracle clarification

pytwoway 0.3.21 has a broadcasting bug in `fe.py` when constructing the
alpha-alpha inverse block (`Dwinv` is added as a vector rather than a
diagonal matrix). Exact-trace `var(alpha)` and exact HE covariance quadrants
that touch this block are not valid xhdfe oracles. The xhdfe help now states
the scope of that issue.

## Validation

- Fixed-design F1 gate, R=300: observation-level `var(alpha)` SE/SD 0.877 and
  AM coverage 0.967; match inference missing and noted in 300/300 draws.
- KSSMC-2, R=150: all four SE clamps and all 15 undefined CIs were noted;
  finite intervals remained equal to the independent port within
  `9.75e-15` relative.
- CPU and CUDA AKM/KSS validators: all checks passed; real H100 CPU/GPU
  parity maximum `1.11e-16`.
- Gelbach hand oracle, adversarial slow-graph gates, and Stata/R/Python
  frontend parity: all passed (frontend maximum `3.55e-15`).
- Stata repository suite: 26/26 files passed. Real-H100 companion gate:
  `xhdfeconnected` mask/counts exact, all weighted Gelbach rows used CUDA,
  and maximum CPU/GPU differences were `2.78e-17` (delta), `2.17e-19` (SE),
  and `2.12e-22` (covariance).
- Existing affected-candidate `core23 x 8` gate rechecked successfully:
  184/184 runs, 704/704 coefficients, 23 datasets, and real CUDA on every GPU
  row; aggregate runtime was 21.50% below its baseline.

Versions: shared C++/Python/R package `2.18.0.20260711`; Stata `xhdfe`
`2.18.0`; `xhdfeakm` `1.7.0`; `xhdfeconnected` and `xhdfegelbach` `1.2.0`;
`xfe` remains `1.10.1` (date stamp refreshed, no xfe behavior change).
