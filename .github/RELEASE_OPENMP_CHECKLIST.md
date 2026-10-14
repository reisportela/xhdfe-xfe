# OpenMP release checklist — 2.29.1.20261014

The 2.29.1 release repeated every gate below with fresh version-tag artifacts
and was published on 14 October 2026 after exact H100 validation and public
installation readback. The historical 2.28.1 record remains below.

The native xhdfe probe now also checks the IV Driscoll–Kraay counterexample
against an independent MP90/MP120 reference, in both tolerance modes and with
one/two threads. Locally, the corrected plugin passes all four cases and the
frozen 2.29.0 plugin is rejected for its incorrect full covariance.

## 2.29.1 completed gates

- [x] [Final preflight 38598859387](https://github.com/reisportela/xhdfe-xfe/actions/runs/38598859387): all 11 required build, platform and assembly jobs passed.
- [x] [Version-tag run 38604829130](https://github.com/reisportela/xhdfe-xfe/actions/runs/38604829130): all 12 required jobs passed, including fresh draft assembly.
- [x] All 13 exact assets match their GitHub digests; the 12 payloads also
  match `SHA256SUMS.txt`.
- [x] Final native receipts match ten plugin/platform cases, two installed
  Python wheels and four macOS serial negative controls. Every production
  plugin has one/two-worker numerical evidence; both final macOS slices ran natively.
- [x] [Exact arithmetic 38597574837](https://github.com/reisportela/xhdfe-xfe/actions/runs/38597574837): 37,632 rational-reference cases passed in each of five native compiler/platform configurations.
- [x] Exact Linux CUDA assets passed licensed Stata/H100 analytic, covariance,
  recovery, cache and first-stage checks with real GPU use. The installed
  Linux wheel passed all 97 frontend tests without skips.
- [x] Five interleaved CPU pairs on 48 million synthetic rows have identical
  b/V, sample and convergence. This is an artifact smoke, not a new core24 claim.
- [x] [Publication 38613871888](https://github.com/reisportela/xhdfe-xfe/actions/runs/38613871888) published the existing draft and site; all 49 site files and 37 installed/ancillary files matched the authenticated snapshot.

The CI R package check has two warnings and four notes, with its tests and
actual-worker gate passed. The complete scope is in
`docs/releases/VALIDATION_2.29.1.20261014.md`.

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
