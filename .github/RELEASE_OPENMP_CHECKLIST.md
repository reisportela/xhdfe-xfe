# OpenMP release checklist — 2.28.1.20260929

The preflight validates the build and distribution gates below. The version-tag
workflow repeats them; publication additionally requires validation of the exact
version-tag CUDA assets on the maintainer H100. Local builds cannot substitute
for those assets.

- [ ] OpenMP-capable macOS arm64/x86_64 toolchains, with deployment targets
  macOS 11 and 10.12 respectively. Both final universal slices run natively.
- [ ] Both Stata plugins compile/link with OpenMP on every platform. Release
  configuration refuses serial fallback. Native serial negative controls
  demonstrate rejection for both macOS plugins on both architectures.
- [ ] CMake/Python/R production builds require OpenMP. Installed Linux/Windows
  wheels and the checked R package report two actual estimator workers and
  match the one-worker analytic reference.
- [ ] Final Linux CPU/CUDA-host, Windows and macOS plugins have compile/link
  and native useful-worker evidence. CUDA device execution is validated on H100.
- [ ] Runtime closure, installation layout and provenance are checked. Windows
  Stata embeds GNU/OpenMP/winpthreads runtimes; Python keeps its private DLL
  closure. macOS ships a compatible LLVM runtime and source/licence materials.
- [ ] Required failures block assembly and publication. Numerical outputs,
  actual workers, architecture, loaded runtime and exact artifact hashes are
  checked together; compile flags alone are insufficient.
- [ ] Preflight, ZIP and net-install validation use the same gates. The active
  private/public release workflows and production sources are synchronized.
- [x] README/help and release notes describe the actual targets, runtime
  requirements and remaining numerical coverage limits.

Execution evidence for this release is recorded after its preflight, version-tag
and exact-asset checks complete. Unchecked rows are pending for 2.28.1, even
though the corresponding build gates passed for the preceding release.
The offline bundle contains native receipts and provider ledgers.
The frontend checks and inherited native limits are described in
`docs/releases/VALIDATION_2.28.1.20260929.md`.
