# Validation of xhdfe 2.35.1.20261125 and xfepout 1.14.7

This update changes CUDA installation on Linux, adds a build metadata query to
both Stata plugins and updates the release packages. The 2.35.0 timing table
in the README retains its original inputs, build identity and results.

## Local checks

| Check | Result | Evidence |
| --- | --- | --- |
| Version and date across package files and mirrors | PASS | `tools/validate_release_metadata.py --expected-version 2.35.1.20261125` |
| Installer protocol | PASS, 8 tests | `tests/test_xhdfegpu_install.py` checks invalid inputs, explicit and inherited CPU targets, source and ZIP precedence, paths with spaces, installation without xfepout, the staged Stata gate and rollback after the second replacement fails. Maintainer receipt `protocolo.log`. |
| SSC package assembly | PASS, 15 tests | `tests/test_stata_ssc_zip.py` verifies the flat ZIP against its declared members, including the two runtime helpers. Maintainer receipt `ssc.log`. |
| Linux installation with licensed Stata/MP | PASS | A fresh PLUS accepted the platform-mapped site and a compiled `stata/` snapshot. Ordinary `net install` put both helpers alongside the ado. `net get` copied the two example do-files and the existing licence material. The site validator passed. Maintainer receipt `site-linux.log`. |
| Stata status | PASS on local hybrid builds | Stata returned the build metadata for both installed roles in the portable and native Linux builds for `sm_90`. The tests also exercise missing and legacy modules. Maintainer receipts `status-B-portable.log` and `status-C-native.log`. |
| Source archive layout and dependency checks | PASS on local staging | The source ZIP builder and autonomous archive validator passed. The two helper entries match the installable Stata sources. Maintainer receipt `source-zip.log`. |

The portable Linux CPU-only candidate, installer smoke on H100 and final
downloaded artifacts have separate checks. For CUDA, the fit must report
actual GPU use and agree numerically with the CPU and the synthetic reference.

## CI and published artifacts

Record the preflight and version-tag workflow IDs, exact asset digests, the
licensed Linux Stata/MP runtime checks and installation from the published
snapshot here once those checks finish. CI verifies the declared CPU target,
CUDA architectures, version and OpenMP for the exact plugins. The
Windows/Linux worker probes and the native macOS probes exercise estimator
work through the Stata plugin interface. Licensed Stata/MP checks on Windows,
native macOS arm64 and native macOS x86_64 remain **NOT_RUN**.

[OpenMP release checklist](../../.github/RELEASE_OPENMP_CHECKLIST.md)
