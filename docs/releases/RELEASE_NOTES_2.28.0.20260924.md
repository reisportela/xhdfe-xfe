# xhdfe 2.28.0.20260924 — release candidate

Candidate date: 24 September 2026. The xhdfe command version is 2.28.0;
the shared Python/R/package identity is 2.28.0.20260924. Publication follows the artifact gates in the validation record.

This version incorporates the audited integration documented in
[the validation record](VALIDATION_2.28.0.20260924.md).
It includes the OpenMP/thread-local crash fix, stable OLS/IV handling of
offsets and redundant instruments, corrected weighted and multiway inference,
FE recovery and extraction fixes, stronger cache identity, and frontend
sample, formula, prediction and state-preservation corrections.

Observable corrections include dropping zero analytic/probability weights
before singleton pruning, bootstrapping the retained estimation sample,
preserving distinct fractional identifiers, and preserving existing generated
variables when an AKM backend call fails. Fast and Comparable retain their
distinct numerical and stopping contracts.

Companion versions are xfepout 1.13.2, xhdfeakm 1.8.1, xhdfegelbach 1.6.1,
and xhdfegelbachbootstrap 1.0.1. The unchanged connected-set, table and plot
frontends retain their own version numbers, with the common 24sep2026 date.
The experimental xhdfe_hetero command is outside this bump.

The core24 x 8 campaign completed 192 cells: 188 candidate cells converged,
with four known CUDA Comparable refusals. All CPU and Fast cells converged.
Independent oracles, complete covariance and sample checks, recovery tests,
and the declared performance trade-offs are recorded in the validation record.
The version bump changes metadata and documentation; the validated native
binaries are preserved byte for byte.

The known CUDA refusals remain; use CPU for those specifications.

Windows Stata plugins now statically link the GNU/OpenMP runtimes. This removes
the loader dependency on DLLs that Stata installs into letter directories and
addresses [issue #11](https://github.com/reisportela/xhdfe-xfe/issues/11).
The native Windows gate loads the exact plugins from a separate `plus/x`
directory with a clean search path and checks one-versus-two-worker numerical
agreement. It also reproduces the previous dynamic-DLL layout failure.
Restart Stata after updating the plugin, then reinstall with `net install ...,
replace` and rerun the estimation.

OpenMP is mandatory across release platforms. macOS universal plugins carry
their matching runtime and require native checks on both architectures.
The preflight, draft and publication stages retain source/runtime provenance,
checksums and the separate exact-asset H100 gate.
