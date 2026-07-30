# xhdfe 2.22.1 / xfe 1.10.3 — 30jul2026

Hardening and certification release on top of 2.22.0. No interface changes,
no changes to default output, no changes to estimator definitions,
tolerances or stopping criteria. All 2.22.0 release notes apply unchanged
(2.22.0 was staged but never published online; this release supersedes it
and ships both changelogs).

## Fix — invalid analytic weights are now rejected loudly

`weights` (analytic) containing a negative or non-finite value were
previously accepted silently in the C++/Python core entry points: every
weighted moment downstream was corrupted while the only guard — a positive
weight total — still passed. Both `fit()` and `partial_out()` now validate
every analytic weight (IEEE-finite, non-negative; bit-level check immune to
fast-math) and raise a clear error. Zero weights remain allowed with their
existing semantics. Frequency weights keep their stricter positive-integer
validation. Valid inputs are unaffected — coefficient paths are proven
bit-identical (e(b)/e(V) byte-compared in %21x across the release binaries).

## Fix — internal `e(version)` strings

`xhdfe` reported `e(version)` 2.21.0 and `xfe` 1.10.1 internally while the
headers said 2.22.0/1.10.2 (a staging omission in the unpublished 2.22.0).
All version surfaces now agree: 2.22.1 30jul2026 / 1.10.3 30jul2026.

## GPU-certificate verification machinery (default-off, no runtime impact)

Groundwork for the device-native GPU acceptance certificate, all inert
unless explicitly enabled:

- A shadow device-side certificate (`XHDFE_GPU_CERT_SHADOW=1`, host remains
  authoritative) computing directed-rounding brackets for the absorption
  residual on-device, with order-independent error bounds, a weight-sums
  integrity cross-check that fails closed, and out-of-range group-id
  detection. Validated against a 256-bit exact oracle (containment of the
  true residual inside the device bracket on multiple datasets), a
  46-cell decision-identity battery vs the host certificate, an adversarial
  input corpus (subnormals, underflow-to-zero weighted inputs, saturated
  fits, invalid weights), and weighted sweeps.
- The host GPU-acceptance certificate is ~3× cheaper on multi-column fits
  (one observation pass per group instead of one per column; proven
  bit-identical, to the last bit of `e(abs_residual_rel)` on CPU).
- The verifier translation unit is compiled under a strict-FP contract in
  every build system (no fast-math, no FMA contraction, full-precision
  div/sqrt, no FTZ), enforced fail-closed at build time by a PTX+SASS gate
  (`tools/check_verifier_device_fma.sh`) wired into CMake, the Stata plugin
  build scripts, and the R package Makevars.

## Certification summary

The release binaries were certified by an interleaved balanced A/B campaign
over the 23-specification core benchmark corpus across all eight execution
variants (C++/Python and Stata plugin × fast and reghdfe-comparable × CPU
and CUDA sm_90): timing parity against the previously accepted binaries on
every Stata variant (medians 0.989–1.002), zero functional findings on the
strict CPU cells (1e-10 pair contract), and GPU differences bounded by the
documented run-to-run non-reproducibility envelope of CUDA accumulation.
Fast/comparable defaults, convergence criteria and outputs are unchanged.
