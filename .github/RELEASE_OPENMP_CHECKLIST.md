# OpenMP release checklist — 2.28.0.20260924

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

Evidence: [preflight native jobs](https://github.com/reisportela/xhdfe-xfe/actions/runs/36022535095)
passed on all platforms. The separate
[assembly retry](https://github.com/reisportela/xhdfe-xfe/actions/runs/36029203100)
validates the packaging corrections against those authenticated native inputs.
The release's offline bundle contains the native receipts and provider ledgers. The independent Windows
letter-directory negative control reproduces the previous loader error; both
static plugins then load without compiler paths and perform real parallel work.
The numerical campaign, accepted performance costs and known CUDA refusals are
recorded in `docs/releases/VALIDATION_2.28.0.20260924.md`.
