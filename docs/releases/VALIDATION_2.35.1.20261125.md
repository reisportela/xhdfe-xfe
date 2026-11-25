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

The public release is pinned to `043d322cd43a064972e5c6f29bf74a8590cd3493`. The
14 published assets match the reviewed version-tag draft. Readback
matched all 54 Stata site files. A fresh public Linux installation
verified 42 installed and ancillary files, including both generic
plugin aliases, the two installer helpers and both example do-files.
[Release](https://github.com/reisportela/xhdfe-xfe/releases/tag/v2.35.1.20261125)

[Preflight 45658833721](https://github.com/reisportela/xhdfe-xfe/actions/runs/45658833721) · [Version tag 45669489138](https://github.com/reisportela/xhdfe-xfe/actions/runs/45669489138) · [Publication 45685145559](https://github.com/reisportela/xhdfe-xfe/actions/runs/45685145559)

The portable Linux runtime check used licensed Stata/MP on the distributed
CPU and CUDA binaries. It recorded 39 passing version pairs over 10
design and device cells and 15 passing CPU/H100 comparisons. Each pair
checked sample, convergence, coefficients and the full covariance matrix.
The H100 arms reported actual CUDA use. The CPU fast-mode gate checked four
Stata and four C++/Python fits for bitwise identity. The follow-up canonical
checks exercised the Stata aliases and the existing Python and C++ builds
in both Linux checkouts, including real H100 execution and liveness tests.

The local native Stata matrix checked overlapping `core24samples` and
`core24s1` suites. Across their union it recorded
186 passing version pairs and
186 passing CPU/H100 comparisons. Each suite accounted
for 144 version pairs with its own designs and options.

### Full Ready panel

The two Linux measurement blocks used three fresh Stata sessions per arm
with 24 observed native workers and a Stata/MP licence exposing 16 processors.
Entries are median estimation times in seconds, after data loading.

| Build | 2.35.0 CPU | 2.35.0 H100 | 2.35.1 CPU | 2.35.1 H100 |
| --- | ---: | ---: | ---: | ---: |
| Local native plugins | 8.728 | 6.017 | 8.763 | 5.851 |
| Distributed CI bundles with portable host CPU code | 11.029 | 6.129 | 10.270 | 6.036 |

The standard GitHub `net install` supplies the portable CPU plugin. The CI
H100 timings use the distributed CUDA bundle with portable host CPU code.
The native block uses host-specific CPU compilation, as requested by
`xhdfegpu, cputarget(native)`. The preflight and version-tag bundles match
byte for byte across the measured Linux execution files. Pairing applies
within each block on the same device.

### Timing follow-up for the six flagged cells

A is 2.35.0 and B is 2.35.1. The first timing column preserves the six-pair
medians that triggered follow-up. Each quartet shows the execution order,
the ratio of the geometric mean command times B/A and the difference of
the mean command times B−A in seconds. The clock starts after data loading.

| Build | Design and device | Initial B/A (B−A) | Three quartets in order | Median quartet B/A (B−A) | Quartets above screen |
| --- | --- | ---: | --- | ---: | ---: |
| Portable | `akm_v02_firstreg` · H100 | 1.0353 (+0.0685 s) | ABBA 0.9818 (-0.0360 s)<br>BAAB 0.9958 (-0.0085 s)<br>ABBA 1.0188 (+0.0360 s) | 0.9958 (-0.0085 s) | 0/3 |
| Portable | `marta_group_individual_3m` · CPU | 1.0649 (+0.4925 s) | BAAB 0.9778 (-0.1755 s)<br>ABBA 0.9566 (-0.3520 s)<br>BAAB 0.9778 (-0.1910 s) | 0.9778 (-0.1910 s) | 0/3 |
| Native | `credit` · H100 | 1.0974 (+0.0420 s) | ABBA 1.1087 (+0.0435 s)<br>BAAB 1.1014 (+0.0460 s)<br>ABBA 0.9750 (-0.0110 s) | 1.1014 (+0.0435 s) | 2/3 |
| Native | `enron` · H100 | 1.1077 (+0.0560 s) | BAAB 0.9799 (-0.0115 s)<br>ABBA 1.0692 (+0.0380 s)<br>BAAB 0.9786 (-0.0135 s) | 0.9799 (-0.0115 s) | 1/3 |
| Native | `schools` · H100 | 1.0548 (+0.0305 s) | ABBA 1.0465 (+0.0285 s)<br>BAAB 0.9239 (-0.0470 s)<br>ABBA 1.0115 (+0.0070 s) | 1.0115 (+0.0070 s) | 1/3 |
| Native | `synthetic-uniform-harder` · H100 | 1.0908 (+0.0505 s) | BAAB 1.0246 (+0.0145 s)<br>ABBA 1.0318 (+0.0180 s)<br>BAAB 0.9948 (-0.0030 s) | 1.0246 (+0.0145 s) | 1/3 |

For `credit` on H100, the three quartets measured +43.5 ms (ABBA), +46 ms (BAAB) and −11 ms (ABBA). The median difference was +43.5 ms (+10.14%), with the share attributable to the version still uncertain.

The portable screen used a ratio above 1.03. The native review also
required a difference above 0.001 s and used all three qualifying
quartets as its criterion for a stable timing loss. Counts use the unrounded
contrasts. The follow-up covers three quartets per cell on this shared host.
Smaller performance differences remain possible.

The native Linux installer was exercised end to end in an isolated Stata PLUS.
It built both plugins, passed the staged CPU and H100 checks with useful OpenMP
workers and installed the two aliases. A new Stata session read the recorded
native CPU target, `sm_90`, CUDA and OpenMP from each installed plugin.

The README's `sysuse auto` example reported actual H100 execution in both
fresh sessions, each with 69 estimation observations. Inserting `xhdfegpu, status` before
the fit left the printed full-precision coefficients and covariance matrix
identical.

The Linux, Windows and macOS native plugin-interface probes checked useful
OpenMP estimator work and build metadata. Both macOS slices were validated,
including the serial negative controls. Licensed Stata/MP execution on
Windows, native macOS arm64 and native macOS x86_64 remains **NOT_RUN**.

### Iteration counts on the difficult synthetic three-effect panels

Both panels on the H100 used the same absorption method and the same coefficient and
full covariance agreement criteria. The A/B fit pairs and the adjacent
same-build controls all passed the numerical checks. The signed iteration
change is the 2.35.1 count minus the 2.35.0 count. The two error columns show
the largest scaled errors among the version pairs and same-build controls.

| Design | A/B pairs with different counts | 2.35.0/2.35.0 control range | 2.35.1/2.35.1 control range | Median A/B change | Max scaled coefficient error | Max scaled covariance error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sampled panel (2,500,000 rows) | 3/3 | 627–986 (2/3 pairs differ) | 574–951 (3/3 pairs differ) | +1 | 8.35e-15 | 5.04e-13 |
| Full panel (10,000,000 rows) | 3/3 | 419–588 (3/3 pairs differ) | 419–551 (3/3 pairs differ) | -1 | 4.88e-15 | 3.06e-13 |


On both panels, iteration counts varied within each build and the control
ranges overlapped. Both median A/B count differences fell within the
observed control spread. The sealed route review found the observed A/B
variation indistinguishable from the variations in the same-build controls
on this H100. Smaller version effects remain unresolved.

[OpenMP release checklist](../../.github/RELEASE_OPENMP_CHECKLIST.md)
