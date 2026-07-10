# xhdfe / xfe 2.16.1 — KSS progress and thread-state fix

Released 10 July 2026.

## Highlights

- Adds optional, throttled progress reporting to `xhdfeakm` / KSS without
  changing default output. Use `verbose` in Stata, `verbose=True` in Python,
  or `verbose=TRUE` in R.
- Reports the current KSS phase, completed and total JLA/SE probes, elapsed
  time, and ETA. Progress callbacks run only on the caller thread and never
  inside OpenMP regions.
- Fixes an inherited OpenMP/Eigen thread-state issue after internal FWL
  residualization with `controls()`. The caller's thread state is now restored
  exception-safely before the KSS solver is constructed.
- Exposes `fwl_threads_used` and `threads_used` diagnostics in the Python, R,
  and Stata interfaces. `XHDFE_AKM_TEAM` is now applied after FWL and therefore
  works as documented.
- Preserves the conservative tuned FWL team as the default KSS ceiling when
  controls are present; users may override it explicitly.

## Correctness and compatibility

No estimator definition, seed, tolerance, stopping rule, reduction formula,
normalization, or default output was changed. Quiet and verbose executions are
numerically identical. Existing Python, R, and Stata calls remain compatible.

## Validation

- CMake CPU and CUDA `sm_90` builds passed locally.
- `VALIDATE_AKM_KSS.py` passed for CPU and CUDA; CUDA was genuinely used and
  the maximum CPU/CUDA difference reported by the gate was `1.11e-16`.
- The Linux Stata plugin links OpenMP (`libgomp`) and contains `sm_90` CUDA
  code; the Stata certification suite passed 26/26 tests, including the thread
  override gate.
- R AKM tests passed, including progress capture and exact quiet/verbose
  invariance.
- A paired ordinary-`xhdfe` run on the 47.6-million-row QP panel converged in
  the same 20 iterations with matching displayed coefficients and standard
  errors. The new build was not slower in that paired run.

The complete `core23 × 8` timing matrix was not rerun because the workstation
was heavily oversubscribed and the timings would not have been adjudicable.
The change is confined to AKM/KSS and its front ends; the ordinary absorption
core is unchanged.

## Distribution

The release workflow builds CPU Stata plugin bundles for Linux x86-64,
Windows x86-64, and macOS universal, publishes the Stata net-install site, and
attaches the autonomous offline source/install bundle. CUDA remains an opt-in
local build; the validated workstation build targets the local H100 (`sm_90`).
