# xhdfe 2.25.1 / xfepout 1.12.0 - 27aug2026

Correctness and convergence hotfix for group-level outcomes with individual
fixed effects. The estimator definition, objective, supported public Python
and R interfaces, default output formatting, and numerical tolerances are not
relaxed.

## Joint LSMR for `group()` / `individual()` on CPU

CPU `absorptionmethod(auto)` now uses a joint matrix-free LSMR operator for
the standard and individual fixed-effect dimensions after the long data are
collapsed to group-level rows. The operator supports `aggregation(mean)` and
`aggregation(sum)`, analytic and frequency weights, and deterministic
multi-thread execution. `absorptionmethod(lsmr)` selects this CPU route
explicitly.

This fixes a poorly conditioned incidence-graph case where the previous
Gauss-Seidel default could reach 100,000 iterations, satisfy a backward-error
certificate, and still return coefficients and RSS materially different from
`reghdfe`. Automatic LSMR remains fail-closed; if it cannot pass the
independent certificate, `auto` retries the previous sweep path from the
original data and reports the method actually retained. Explicit LSMR never
changes algorithm silently.

Explicit `gauss-seidel` and `symmetric-gauss-seidel` remain available in this
mode. `jacobi`, `schwarz`, `mlsmr`, and `auto-mlsmr` requests are rejected
rather than executing a different algorithm under a misleading method label.

## CUDA certificate hardening

CUDA `auto` retains the existing group/individual GPU solver. A result is now
reported as converged and `gpu_status=used` only after the authoritative
group/individual certificate passes. Explicit LSMR with CUDA is rejected with
a clear CPU-only diagnostic instead of executing GPU Gauss-Seidel while
reporting LSMR.

## Validation evidence before the tagged workflow

- An adversarial cycle that previously reached 100,000 iterations with a
  slope error of approximately `8.19e-4` converged under joint LSMR in 500
  iterations; the slope differed from `reghdfe` by `4.3e-15` and RSS was
  `1.54e-27`.
- Independent explicit dummy-design oracles covered unequal group sizes,
  `mean` versus `sum`, analytic weights, literal frequency weights, strict
  certification, and one-versus-multiple threads. Maximum slope error was
  below `8e-16`, maximum residual error below `1e-12`, and thread variants
  were identical in the tested cells.
- The maintained Stata certification suite passed 36 groups. CPU and CUDA
  CTest gates passed 4/4, and the H100 group/individual CPU-CUDA matrix passed
  16/16 cells with real GPU use and no silent fallback.
- A 100-million-row long-format patent workload produced 23,824,067 group
  observations, converged in 16 LSMR iterations in 58.642 seconds with eight
  effective threads, and reported slope `1.4349168836862`, RMSE
  `1.3643441261346`, and a passing precision certificate. This is a
  workload-specific functional/timing observation, not a universal speed-up
  claim.
- A no-`group()` sample gate retained byte-identical coefficients and the
  same 23 iterations; one timing pair was 2.064 seconds before versus 2.069
  seconds after, consistent with host noise.
- The adversarial cycle with `numthreads(12)` formed and used all 12 requested
  workers; its slope differed from `reghdfe` by `6.22e-15` and RSS by
  `2.29e-28`.

The tagged cross-platform workflow remains the release gate. Publication is
blocked until the exact Linux CUDA plugins from that workflow pass CPU and
CUDA Stata validation on the maintainer H100, including real-use diagnostics,
and the published assets and net-install site pass independent read-back.

## Version scope

- Shared C++/Python/R package and release tag: `2.25.1.20260827`.
- Stata `xhdfe`, `xhdfe_p`, `xhdfe_estat`, and `xhdfegpu`: `2.25.1`.
- Stata `xfepout`: `1.12.0` (unchanged functionality, release date refreshed).
- Other companion-command feature versions are unchanged.
- All production Stata text files carry the release date `27aug2026`.

## Online Stata installation

After the H100-certified publication marker completes:

```stata
net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
net install xfepout, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
```

The previous `v2.25.0.20260824` release and its assets remain available as
immutable rollback material.
