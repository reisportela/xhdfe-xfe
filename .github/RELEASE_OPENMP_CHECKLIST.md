# OpenMP release checklist — 2.29.1.20261014

The 2.29.1 release repeats every gate below with fresh version-tag artifacts.
Completion is pending the current preflight, version-tag build, exact H100
validation and public installation readback. The historical completion record
below belongs to 2.28.1 and does not substitute for these executions.

The native xhdfe probe now also checks the IV Driscoll–Kraay counterexample
against an independent MP90/MP120 reference, in both tolerance modes and with
one/two threads. Locally, the corrected plugin passes all four cases and the
frozen 2.29.0 plugin is rejected for its incorrect full covariance.

## Historical gate definitions and 2.28.1 completion

The preflight validates the build and distribution gates below. The version-tag
workflow repeats them; publication additionally requires validation of the exact
version-tag CUDA assets on the maintainer H100. Local builds cannot substitute
for those assets.

- [x] OpenMP-capable macOS arm64/x86_64 toolchains, with deployment targets
  macOS 11 and 10.12 respectively. Both final universal slices run natively.
- [x] Both Stata plugins compile/link with OpenMP on every platform. Release
  configuration refuses serial fallback. Native serial negative controls
  demonstrate rejection for both macOS plugins on both architectures.
- [x] CMake/Python/R production builds require OpenMP. Installed Linux/Windows
  wheels and the checked R package report two actual estimator workers and
  match the one-worker analytic reference.
- [x] Final Linux CPU/CUDA-host, Windows and macOS plugins have compile/link
  and native useful-worker evidence. CUDA device execution is validated on H100.
- [x] Runtime closure, installation layout and provenance are checked. Windows
  Stata embeds GNU/OpenMP/winpthreads runtimes; Python keeps its private DLL
  closure. macOS ships a compatible LLVM runtime and source/licence materials.
- [x] Required failures block assembly and publication. Numerical outputs,
  actual workers, architecture, loaded runtime and exact artifact hashes are
  checked together; compile flags alone are insufficient.
- [x] Preflight, ZIP and net-install validation use the same gates. The active
  private/public release workflows and production sources are synchronized.
- [x] README/help and release notes describe the actual targets, runtime
  requirements and remaining numerical coverage limits.

Completed on 29 September 2026:

- [Preflight 36553876660](https://github.com/reisportela/xhdfe-xfe/actions/runs/36553876660): all build, platform and assembly gates passed.
- [Version-tag run 36558408298](https://github.com/reisportela/xhdfe-xfe/actions/runs/36558408298): fresh builds, installed packages and draft assembly passed.
- All 13 draft assets matched their GitHub SHA-256 digests and `SHA256SUMS.txt`.
  Native receipts matched the final plugin/wheel bytes: ten plugin platform
  cases, two Python wheels and four macOS serial negative controls. The R
  package passed its one/two-worker analytic gate; its pre-existing check
  warnings and notes remain explicit in the validation record.
- The exact Linux CUDA assets passed licensed Stata/H100 analytic b/full-V,
  recovery and cache tests with actual GPU use. Linux CPU assets passed the
  same analytic fixture and a 46.16-million-observation smoke. The downloaded
  Linux wheel passed the 97-test frontend gate again in an isolated prefix.
- Every platform ZIP, the net-install snapshot and the corresponding-source
  bundle passed their closure validators. The offline bundle carries the
  native receipts and provider ledgers used above.

The publication marker is permitted only after these checks; public installation
readback and canonical-checkout sealing complete the publication procedure.
See `docs/releases/VALIDATION_2.28.1.20260929.md` for scope and inherited limits.
