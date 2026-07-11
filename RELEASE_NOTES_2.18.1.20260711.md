# xhdfe 2.18.1.20260711 — clear AKM warnings and live progress

This patch release makes long `xhdfeakm` runs observable while they are still
running and replaces the generic “see r(notes)” warning with the complete
diagnostic at the point where the command returns.

## User-facing changes

- `xhdfeakm` now prints the full inferential warning automatically. For
  example, match-level `var(alpha)` inference explains immediately that its
  SE/CI is unavailable under the canonical `leave_out_COMPLETE` oracle and
  that observation-level inference requires `leaveoutlevel(obs)`.
- Non-convergence prints both the reliability warning and the available
  diagnostic details. `r(notes)` remains available for programs, but users no
  longer need to inspect it before another command overwrites `r()`.
- With `verbose`, every progress line is flushed to the Stata Results window
  and the GUI is polled before the next numerical phase. Leave-out-set
  construction, FWL, solver setup, point effects, JLA draws, corrected
  components, SE simulations, eigen diagnostics and completion therefore
  appear live instead of arriving as an end-of-command dump.
- The same output-only flush applies to the shared verbose sink used by
  `xhdfeconnected` and `xhdfegelbach`.
- Without `verbose`, the compact default output is unchanged. No estimator,
  tolerance, stopping rule, result, API or backend-selection rule changed.

## Validation

- Interactive Stata live test, 1,000,000 observations, 400 JLA draws and 200
  SE simulations per component: progress arrived while `plugin call` was
  still running, with approximately one-second updates and ETA. The run
  converged in 32.1 seconds and printed the full match-level `var(alpha)`
  warning automatically.
- Stata certification suite: 26/26 do-files passed.
- CPU and real-H100 CUDA AKM/KSS validators: all checks passed; maximum
  CPU/CUDA difference remained `1.11e-16`.
- Real-H100 companion gate: CUDA was used; maximum CPU/GPU differences were
  `1.60e-17` (Gelbach delta), `1.08e-19` (SE), and `4.24e-22`
  (covariance).
- Mandatory 46.16-million-observation QP smoke: CPU `67.130s`; real CUDA
  `16.350s`; both converged and returned identical coefficients and standard
  errors.
- Local Stata plugin: CUDA `sm_90`, OpenMP (`libgomp`) and `-march=native`.
- Default CPU and CUDA Python module directories rebuilt in Release mode;
  R package installation and Python/R version checks passed.

Versions: shared C++/Python/R package `2.18.1.20260711`; Stata `xhdfe`
`2.18.1`; `xhdfeakm` `1.7.1`; `xhdfeconnected` and `xhdfegelbach` `1.2.1`;
`xfe` remains `1.10.1`.
