# Validation — 2.34.2.20261122

The release integrates CPU reduced-system prefetch, parallel work after absorption, and Stata identifier and sample bookkeeping. The public version tag points to `e08615709e9482e5a734a3cd367bebfc96a57b67`.

## Published artifacts

GitHub published the authenticated draft on 22 November 2026 at 21:24:15 UTC. The [preflight](https://github.com/reisportela/xhdfe-xfe/actions/runs/45058111683) and [version-tag build](https://github.com/reisportela/xhdfe-xfe/actions/runs/45064820628) passed the Linux, Windows and macOS checks and assembled fourteen assets. The [publication job](https://github.com/reisportela/xhdfe-xfe/actions/runs/45071872342) promoted the same draft and site. All fourteen [release assets](https://github.com/reisportela/xhdfe-xfe/releases/tag/v2.34.2.20261122) retain their authenticated IDs and SHA-256 digests. All 52 files read from the public `gh-pages` commit `9dfcd07842cd9ef7d630644276e87a483d69331c` match the accepted net-install snapshot.

- The native CI receipts bind worker and numerical checks to the final Windows x86-64, macOS arm64 and macOS x86-64 plugin bytes. The licences and matched OpenMP runtimes are in the platform packages. The Windows wheel and manylinux wheel have installed worker receipts. Linux also exercised the R package worker probe. The macOS and Windows Stata-plugin checks use native SPI hosts. Licensed Stata runs in this validation took place on Linux.
- The exact draft Linux CPU and CUDA plugins ran in licensed Stata. The ordinary fixture exercised both tolerance modes and `xfepout`. Additional checks covered reduced-system routing, `savefe`, slopes, weights and an eight-million-row CPU fit with 16 workers. The exact manylinux wheel was installed in an isolated environment and its coefficients agreed with both Stata and the closed-form oracle.
- On the 46.2-million-observation `main_95_21_ready` estimation sample, the exact draft finished in 17.414 s on CPU and 6.966 s with real H100 use. Both runs converged in 116 iterations. The largest relative coefficient and standard-error differences between CPU and H100 were `8.67e-18` and `8.09e-15`.
- The optional firm-slope route (method 9) ran on the H100 with the NVIDIA libraries available to the loader. With those libraries absent, the documented error returned without estimates. The portable comparison below also exercises the ordinary firm-slope route with its default options.

A fresh `net install` of `xhdfe` and `xfepout` and `net get` from the public site matched all 40 installed and ancillary files byte for byte, including both demonstration do-files. The newly installed package passed the ordinary CPU fixture.

The two Stata demonstration do-files were downloaded with `net get` and run with the exact draft's portable plugins. The clustered example matched the native explicit-dummy oracle on the estimation sample and covariance, including its nesting correction. The partial-out example matched the native regression slopes and the projection check against the previous release. The flat [SSC asset](https://github.com/reisportela/xhdfe-xfe/releases/download/v2.34.2.20261122/xhdfe-ssc-2.34.2.20261122.zip) has 47 components, each byte-identical to its counterpart in the public site. Its SHA-256 is `9d70c708758fd730e324a5f640e93fa6b13bd2dabd8f469d77ddc7740790d951`.

## Agreement and run times

The local `core24s1` matrix compared native builds from 2.34.1.20261121 and 2.34.2.20261122 at 16 threads. It crossed the 24 specifications with CPU and H100, Stata and the C++ binding, and Fast and Comparable. The 666 pairs passed the functional checks, including 650 verified H100 executions. The maximum scaled coefficient, relative standard-error and diagonal-scaled covariance differences were `5.55e-11`, `4.64e-10` and `9.28e-10`. Each pair retained its estimation sample and backend. The timing decision retains cells whose differences were too variable for a call.

On the local native builds, the summed Stata CPU Comparable medians moved from 136.375 s to 109.462 s. The corresponding C++ sum moved from 120.445 s to 98.978 s. Stata measures the command and printed output after data loading. The C++ clock measures the binding's `fit()` call with the inputs prepared.

The portable Stata check used three interleaved pairs per sentinel and execution contract, plus three for `akm_v02_secondreg_small` on H100/Comparable with default options. The 39 pairs passed numerically. The Linux runtime surfaces of the measured preflight and exact draft bundles are byte-identical. The table gives the sums of the median fit times across the three sentinels.

| Stata contract | 2.34.1 (s) | 2.34.2 (s) |
| --- | ---: | ---: |
| CPU Fast | 102.354 | 95.841 |
| CPU Comparable | 86.000 | 64.033 |
| H100 Fast | 46.962 | 39.136 |
| H100 Comparable | 65.050 | 59.600 |

For the local Stata/H100 Comparable firm-slope cell, the maintainer accepted a median paired cost of 0.267 s. The within-version controls show long-tailed run time and iteration distributions in both versions. The follow-up median difference was −0.112 s per pair and +0.120 s per block. The three portable pairs were 7.191 to 2.833 s, 2.933 to 8.061 s, and 6.577 to 2.786 s. The three pairs retain convergence, sample and covariance agreement.

The permanent analytic battery returned 664 xhdfe passes and 223 predeclared refusals. The native CTest suites returned 25 CPU and 27 CUDA passes. Detailed logs and the staged runtime identities are retained in the development evidence directories.
