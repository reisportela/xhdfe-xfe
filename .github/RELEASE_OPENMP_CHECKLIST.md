# Required before the next release: OpenMP on every native target

Owner requirement, 6 September 2026: release builds must enable OpenMP on all
platforms. The existing macOS exception is withdrawn. The installed/public
2.26.2 binaries are historical artifacts; this checklist does not relabel them.

Private implementation update, 13 September 2026: `release.yml` now calls
`macos-openmp.yml`. It builds the pinned LLVM runtime and both plugins on
native arm64/x86_64 runners, checks serial negative controls, assembles and
signs universal bytes, and requires native tests of those final bytes before
release assembly. The runtime, licence, receipts and rebuild sources are
included in the package paths. Both plugin builders, CMake and normal R
source installation require OpenMP; serial diagnostics use explicit opt-outs.

Local validation passed: CMake/R configuration controls, simulated macOS
build/packaging controls, and a real Linux SPI probe of the existing xhdfe
and xfepout plugins with one/two observed workers and exact fixture agreement.
The macOS toolchain and native jobs have **not** run. Linux/Windows/R/Python
requirements outside this new macOS job retain their own pending evidence.
The boxes below remain release requirements, not claims inferred from code.
The canonical repositories/workflows have not been updated or published.

- [ ] Configure an OpenMP-capable toolchain/runtime for macOS ARM64 and
      x86_64, preserving the advertised minimum OS versions or explicitly
      reviewing any change to them.
- [ ] Compile xhdfe and xfepout with OpenMP on both slices; forbid retrying a
      failed release compilation without OpenMP.
- [ ] Make OpenMP required in all release CMake/Python/R builds; preserve
      diagnostic serial builds as separate, non-release configurations.
- [ ] Inspect final Linux, CUDA-host, Windows and macOS binaries for OpenMP
      compile/link evidence; inspect both slices of universal Mach-O files.
- [ ] Validate every non-system runtime dependency and its installation path;
      package the matching runtime, licences and rebuild inputs as needed.
      Validate loading without the build machine's compiler/Homebrew paths.
- [ ] Execute useful estimator work with at least two observed workers on
      each supported native architecture and compare precision with one
      worker; preserve convergence, inference, FE recovery and performance
      requirements in AGENTS.md. Compile/link evidence is recorded separately
      from runtime evidence when a CI runner cannot perform both.
- [ ] Enforce failure before artifact upload/assembly/publication if any
      required OpenMP gate is absent or fails. Exercise a serial negative
      control to demonstrate that the release gate rejects it.
- [ ] Apply the same requirements to preflight builds, release ZIPs and the
      net-install snapshot, and synchronize the public/private workflows.
- [ ] Update release notes to describe the validated targets and runtime
      requirements accurately; remove the previous macOS no-OpenMP exception
      only after the corresponding implementation and tests pass.

Keep the numerical estimator, tolerances, interfaces and default CPU backend
unchanged while correcting the build. Do not publish a replacement release
from this policy-only change.
