# OpenMP release checklist — 2.30.1.20261020

## 2.30.1 release gates

The candidate corrects the inherited three-effect CUDA covariance error and
GPU repair accounting, with cooperative GPU reductions and cache generation
`0x22624841443a`. The maintainer accepts the measured ready runtime cost
versus 2.29.1 and continues to defer the full core24 performance campaign.

- [x] Fresh local CPU/CUDA native builds: 16/16 CTests each; both Stata plugins built with OpenMP and native CPU tuning, CUDA `sm_90`.
- [x] Final local functional core: 664 xhdfe numerical passes, 223 expected unsupported cases, adapted reghdfe CPU/CUDA contracts and independent covariance/budget controls.
- [x] Twenty cache migration checks; Python 97 CPU tests and 16 independent CUDA formula cases; R 1,229 passes with the documented warnings and no skips.
- [x] Complete sequential Stata suite and candidate-only core24 CUDA execution, 48/48; final large CPU/CUDA artifact checks retain sample, independent b/V, convergence and actual workers.
- [ ] Fresh Linux, Windows, native macOS ARM/Intel version-tag artifacts, exact runtime/worker receipts and archive checks.
- [ ] Exact downloaded H100/CPU runtime gates and isolated installation.
- [ ] Final licensed native Mac ARM and Intel Stata/MP host checks: **NOT_RUN** at this stage; retain the distinction from native SPI and Rosetta evidence.
- [ ] Public release/net-install readback and both canonical `main` checkouts, builds and active artifacts reconciled.

Current evidence: [2.30.1 validation](../docs/releases/VALIDATION_2.30.1.20261020.md).
The historical 2.30.0 publication exception below is not a passing host test
for a newly rebuilt artifact.

## 2.30.0 release gates

The source integrates PR #15/#17 and the bounded CPU/CUDA recovery fixes.
The maintainer deferred the full core24 performance campaign for this release;
ready-only timings do not waive the artifact and actual-worker checks below.

- [x] [Fresh version-tag builds](https://github.com/reisportela/xhdfe-xfe/actions/runs/39399375729), 13 authenticated assets, ten native plugin/platform worker receipts, two installed-wheel checks and four macOS serial negative controls.
- [x] Exact downloaded Linux CPU/CUDA artifacts validated on Athena/H100, including full Stata suites, CG/CUDA regressions and installed-wheel tests.
- [x] Final macOS runtime export/import isolation on both native slices.
- [ ] Real licensed Stata/MP host checks on final native ARM and Intel artifacts: **NOT_RUN; explicit maintainer publication exception, 20 October 2026**.
- [x] [Public release](https://github.com/reisportela/xhdfe-xfe/releases/tag/v2.30.0.20261020), exact net-install readback (49 site files, 37 installed/ancillary files) and validated canonical runtime/source identity.

The maintainer explicitly authorised "Publicar com os limites documentados"
after reviewing the exact assets and release notes. This permits publication
with the native Stata/MP coverage gap; it is not a passing host test. The
inherited ready CPU/CUDA covariance review and deferred full core24 timings
also remain documented limits.

Current evidence and limitations: [2.30.0 validation](../docs/releases/VALIDATION_2.30.0.20261020.md).

## Issue 14 follow-up: Stata host coexistence

The historical native SPI gates below use a host without OpenMP. They do not
exercise coexistence with Stata MP's own runtime. Issue #14 reproduced an Intel
Stata MP crash despite those gates passing. See
[`ISSUE_14_STATA_FIXES_20261016.md`](../docs/releases/candidates/ISSUE_14_STATA_FIXES_20261016.md)
for the candidate fix, negative controls and current coverage.

For 2.30.0 and subsequent macOS follow-up:

- [x] Build with `tools/macos_openmp.exports`; reject private exports with
  `tools/check_macos_openmp_exports.py` on both final runtime slices.
- [x] Retain all existing native SPI, runtime-identity and serial-negative gates.
- [ ] Run `tools/run_stata_openmp_host.py` on each final packaged artifact in
  licensed native Apple Silicon and Intel Stata MP, with `--threads 1 2 4 0`.
  This checks OLS, IV/first-stage diagnostics and `xfepout`, useful workers,
  numerical agreement and repeated calls in the real host. **NOT_RUN on the
  final 2.30.0 assets; publication exception above.**
- [ ] Record Stata version/edition and check Stata SE when available. A missing
  edition is a coverage limitation, not a passing result.

Example (use a fresh output directory):

```sh
python3 tools/run_stata_openmp_host.py \
  --stata /path/to/StataMP --adopath /path/to/installed/x \
  --arch arm64 --threads 1 2 4 0 --out /path/to/new/host-receipt
```

The supplementary runner can select Intel Stata through Rosetta and records
that translation explicitly. Rosetta results do not satisfy the native Intel
release gate. The issue-14 work used only the maintainer's Mac and Athena;
it is not full release certification.

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
