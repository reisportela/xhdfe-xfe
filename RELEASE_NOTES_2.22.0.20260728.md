# xhdfe 2.22.0 / xfe 1.10.2 — 28jul2026

Stability and compatibility release. No interface changes, no changes to
default output, no changes to estimator definitions, tolerances or stopping
criteria on healthy problems.

## Critical fix — silent wrong answers in `tolerancemode(xhdfe-fast)` at tight tolerances

In 2.21.0 and earlier, `xhdfe-fast` combined with a user tolerance at or below
`1e-11` could return **wrong, non-reproducible coefficients while reporting
`converged`**, on CPU and on GPU, on ill-conditioned fixed-effect structures
(the public `github` benchmark dataset reproduces it deterministically). Users
who tightened the tolerance — the natural act of someone seeking accuracy — got
silently wrong answers *because* they tightened it.

Root cause: the Irons–Tuck divergence safeguard (adaptive restart plus hand-off
to the stable block-CG) was active only under the `reghdfe-comparable` stopping
rule, while the historical change-of-norm criterion — the one that can fire on
a flattened-but-diverged trajectory — is exactly the one `xhdfe-fast` uses.

The fix decouples the two: the stopping criterion keeps its documented
mode-dependent semantics, and the divergence safeguard now runs in every mode
and on every accelerated absorber (packed, SoA, general/`savefe`/weights, and
heterogeneous-slopes). On healthy trajectories the safeguard never fires and
results are **bit-for-bit identical** to the previous behaviour — verified over
the full core-23 benchmark corpus (48/48 CPU cells bit-identical at the default
tolerance; all GPU cells statistically indistinguishable from the run-to-run
reproducibility floor across 1000+ validation runs).

On the GPU, the repaired fast path now also hands off to the block-CG when the
safeguard detects divergence (previously it ground out plain sweeps):
`github` at `tol=1e-12` drops from ~12 000 to ~450 iterations with correct
coefficients.

Kill switches (diagnostics only): `XHDFE_ACCEL_GUARD_ALWAYS=0` restores the
previous gating; `XHDFE_CUDA_GUARD_CG_BAIL=0` restores the suspend-only GPU
behaviour.

## Compatibility fix — Linux binaries now load on Enterprise Linux

All Linux binaries of 2.21.0 (wheel and Stata plugins) required
`GLIBC_2.38`/`GLIBCXX_3.4.32` — an Ubuntu 24.04 CI floor leak — and therefore
failed to load on RHEL/AlmaLinux/Rocky 9, Ubuntu 22.04 LTS, Debian 12 and
SLES 15. Linux release artifacts are now built in a `manylinux_2_28`
environment (glibc 2.28 floor) and every artifact is gated by a fail-closed
`tools/check_binary_floor.sh` check in CI. The rebuilt wheel's extension
requires only `GLIBC_2.23`/`GLIBCXX_3.4.21`.

## New execution-certificate diagnostics

All three frontends now surface the absorption precision certificate:
`abs_residual`, `abs_residual_rel` and `precision_certified`
(Stata: `e(abs_residual)`, `e(abs_residual_rel)`, `e(precision_certified)`;
Python: `abs_residual_`, `abs_residual_rel_`, `precision_certified_`;
R: list fields of the same names). `converged` keeps its historical
stopping-rule semantics; `precision_certified` is the authoritative statement
about the achieved absorption residual.

## Versions

| artifact | version |
|---|---|
| xhdfe (Stata, Python, R, shared package) | 2.22.0.20260728 |
| xfe | 1.10.2 |
| xhdfeakm | 1.7.2 (date restamp) |
| xhdfeconnected | 1.2.1 (date restamp) |
| xhdfegelbach | 1.5.0 (date restamp) |
