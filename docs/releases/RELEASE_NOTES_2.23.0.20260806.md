# xhdfe 2.23.0 / xfe 1.11.0 — 06aug2026

Correctness and certification release. It preserves the estimator definition,
public tolerances, convergence targets, numerical precision, interfaces and
default output formatting while closing the functional and scientific blockers
identified after 2.22.1.

## Scientific precision and convergence

- Automatic absorption now repairs a converged result that fails the
  independent precision certificate. Explicitly forced methods remain
  observable and retain a typed warning instead of being silently substituted.
- `partial_out()` resets convergence and certificate state before work begins,
  preventing stale success state after a failed call.
- CPU repair can continue from the current iterate or use MLSMR without changing
  the requested public tolerance or iteration limit.
- The bounded CUDA four-FE repair applies only to unweighted, default-Auto,
  no-fixed-effect-recovery fits with at most two million observations and a
  requested tolerance no looser than `1e-8`. Fast mode uses the scale-invariant
  update criterion directly; comparable mode solves internally at `1e-10` while
  the public certificate remains tied to the caller's original tolerance.
- A requested CUDA backend that is unavailable may retain the documented CPU
  fallback. Once CUDA work has actually started, a backend failure is reported
  fail-closed rather than being hidden by a CPU result.

The final frozen inventory passed all seven CPU cases and all seven CUDA probes.
Every CUDA probe converged, certified and reported real GPU use. Against a CPU
`strict-residual(1e-10)` reference, the worst case (`workers`, fast mode) had
maximum relative errors of `9.256e-10` for coefficients and `8.348e-8` for
standard errors, below the pre-declared `1e-7` stop line.

## Functional contracts

- Non-finite input and cross-product checks remain live under the production
  fast-math flags through IEEE bit-level predicates.
- Empty samples, invalid weight combinations, frequency-weight arithmetic,
  grouped data, DoF tokens and mobility combinations fail with typed,
  front-end-consistent errors.
- Token-free Python and R `dofadjustments` inputs mean the default `all`, as an
  absent Stata option does; `clusters` and `continuous` no longer enable
  mobility implicitly.
- Stata posts canonical `e(dofmethod)`, weighted `e(N)`, `e(wtype)` and
  `e(wexp)`, and uses reghdfe-compatible FE labels and no-FE DoF conventions.
- Grouped reghdfe shorthand such as `absorb(fe##c.(x z))` is accepted.

## Python source-package closure

- The declared top-level compatibility module `py_hdfe_v11` is now included in
  both source and wheel distributions.
- The Python sdist now contains the CMake liveness and fail-closed test sources
  and the build-time non-finite/FMA guard scripts required to configure and
  compile outside the repository checkout.
- Linux release wheels and plugins are built in the `manylinux_2_28` environment
  and fail closed if their GLIBC/GLIBCXX requirements exceed the declared floor.

## Performance

The paired large CPU smoke showed no regression: the candidate/baseline median
ratio was `0.98558` (candidate 1.44% faster; balanced-pair ratio `0.99289`). The
scientifically repaired `workers` CUDA fast path costs about `0.268` seconds
relative to the former result that did not pass its precision certificate, and
is 17.6% faster than the first scientifically valid repair. No invalid baseline
is presented as a performance acceptance target.

## Release provenance

The public tag workflow builds Linux CPU/OpenMP and CUDA fatbin plugins, Windows
x86_64 OpenMP plugins, macOS universal plugins, the Python wheel/sdist, the R
source package, per-platform Stata bundles, a net-install snapshot and an
offline autonomous source/binary archive. The release remains a draft until the
exact attached CUDA plugins pass the Stata functional gate on the maintainer
H100; only the matching `publish-v*` marker may publish the release and
`gh-pages` snapshot.

The local scientific and performance evidence used for the release decision is
summarized in [CERTIFICATION_2.23.0.20260806.md](CERTIFICATION_2.23.0.20260806.md).
