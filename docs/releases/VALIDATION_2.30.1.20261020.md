# Validation record — xhdfe 2.30.1.20261020

Status: local candidate validation completed in the scopes recorded below.
Exact release assets, installation and publication remain separate gates.

The timing reference is frozen **2.29.1.20261014**. The historical precision
contract remains unchanged; public tolerances and oracle thresholds were
not adjusted to obtain a pass.

## Numerical evidence

Independent full-ready dummy-variable references, refined in extended
precision with MP80 inference, identify an inherited CUDA Comparable
diagonal-scaled covariance error of approximately `3.2e-8`, above the `1e-8`
contract. Independent CPU and CUDA implementations of the reference agree.
The corrected eight-cell campaign has maximum Comparable coefficient error
`9.05e-15` and covariance error `1.27e-11`. Fast is assessed against its
separate published approximation contract.

The final local modules and Stata plugins passed 664 xhdfe analytic cases, with 223 expected
unsupported cases and no lost previous pass. The adapted reghdfe core passed
its CPU and CUDA contracts; reference-only and expected-difference outcomes
remain separate. The new three-effect covariance, scaling, worker-count and
GPU iteration-budget controls passed. A 150-iteration negative control
refuses without stale estimates; the positive control converges with real
CUDA use and an independent covariance check.

Additional `xfepout` coverage has nine numerical passes and three expected
CUDA refusals at 16 workers. The same three rows refuse in the published
version with identical iterations and necessary residual checks. They are
not numerical successes. Five injected corrupt-log controls are rejected
by the evaluator.

The integration changes absorption-cache generation to `0x22624841443a`.
Twenty checks across the CPU and CUDA-capable modules passed old-generation
rejection, exact new-generation hits and read-only cache custody. The
unbumped negative control was rejected in both tolerance modes.

Fresh local native CPU and `sm_90` CUDA builds passed all 16 CTests each.
Both Stata plugins use native CPU tuning and OpenMP. The full sequential
Stata suite, standalone contracts and extended IV Driscoll–Kraay checks
passed. All 24 corpus specifications completed in CUDA Fast and Comparable,
with actual CUDA use: 48/48 functional outcomes. These corpus runs are not
timing comparisons or independent mathematical oracles.

The Python frontend
passed its 97 CPU tests and the separate 16-case CUDA analytic contract;
the latter requires real CUDA use. The general CPU suite is not repurposed
as a CUDA test because it includes unsupported no-FE cases and exact CPU
comparisons.

The CUDA-enabled R package passed 1,229 tests with zero failures or skips;
six test warnings remain. `R CMD check` has the same two warning blocks as
the preceding version: non-standard CUDA/header support files and shared
C++ console/termination symbols. The optional Suggestions `nanoparquet`
and `gt` were unavailable. The R audit and actual one/two-worker OpenMP
checks passed; the executed tests include real CUDA assertions.

## Ready performance scope

Eight execution variants × three interleaved baseline/candidate pairs =
**48 completed fits**. Processes are fresh, caches are off and no warm-up is
used. Loading and receipt construction are excluded from fit times.
All fits report convergence, the required sample and 16 actual workers;
CUDA rows prove actual CUDA use. There were no interruptions or concurrent
GPU processes in the retained contexts.
These measurements use locally tuned Linux artifacts and the maintainer's
H100; portable release artifacts are validated separately.

| Execution cell | 2.29.1 median (s) | Corrected candidate median (s) | Median paired change |
| --- | ---: | ---: | ---: |
| Native CPU Fast | 23.844 | 22.602 | -3.46% |
| Native CPU Comparable | 31.659 | 30.829 | -2.62% |
| Native CUDA Fast | 5.604 | 5.911 | +7.34% |
| Native CUDA Comparable | 7.248 | 11.106 | +54.63% |
| Stata CPU Fast | 30.285 | 31.318 | +0.20% |
| Stata CPU Comparable | 38.610 | 38.165 | -0.99% |
| Stata CUDA Fast | 13.314 | 13.156 | -1.21% |
| Stata CUDA Comparable | 15.045 | 18.671 | +22.31% |

Changes are medians of paired ratios, not ratios of medians. The maintainer
explicitly accepts the residual cost of the demonstrated correction.
Small differences remain subject to shared-host variability; the native
CUDA Fast increase is retained as an observed local cost.

Three earlier interleaved native CUDA Comparable pairs against the first
correct, unpublished implementation measured a reduction from 24.182 s to
11.185 s, or 53.82% by paired ratios. That comparison is a separate experiment
and is not a speedup claim against 2.29.1.

The full core24 timing matrix is **deferred by the maintainer**.
Candidate-only corpus execution establishes functional coverage, not
performance acceptance.

After the cache-generation bump and clean rebuilds, large final-artifact
checks again passed the same sample, degrees of freedom, convergence,
worker and independent full-covariance controls. Native CPU took 30.079 s,
Stata CPU 37.255 s and Stata CUDA 18.626 s. These three artifact checks
validate the rebuilt binaries; the interleaved table remains the performance
evidence.

## Final release gates

Pending: Linux, Windows and macOS ARM/Intel release artifacts; exact
downloaded-artifact H100/runtime validation; isolated installation and
private/public local/online alignment.

The source fixes for issue #14 are included. Final licensed Stata/MP host
execution on native Mac ARM and Intel remains **NOT_RUN** at this stage.
Tiago's native ARM and Intel/Rosetta source-candidate tests, and native
SPI/runtime probes, are distinct evidence. The prior 2.30.0 publication
exception is not recorded here as a test pass for new artifacts.

Private evidence is under
`Verifications/ready_covariance_review_20261020/factor_performance_20261020/`
and `Verifications/ready_covariance_review_20261020/release_2_30_1_20261020/`.
