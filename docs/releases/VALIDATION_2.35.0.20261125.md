# Validation scope for xhdfe 2.35.0.20261125

## Local qualification of the estimator sources

The 24 November source qualification compared 2.35.0 with a frozen local
rebuild of the published 2.34.2 sources. All 144 paired Stata comparisons
agreed in estimation sample, coefficients and full covariance. The 144 C++
pairs and the 72 candidate Stata CPU/H100 comparisons also passed. The design
matrix used samples for the large panels, with a 1% whole-worker sample for
the second AKM regression. CPU Fast identity was checked separately in three
fits. These receipts qualify the local estimator builds.

A separate full-panel Stata comparison used 46,156,187 observations and
absorbed worker, employer and year effects. CPU command medians at 24 native
threads were 8.363 s with `reducedpcgfine(auto)` for 2.35.0 and 8.930 s for
2.34.2, from three interleaved rounds. The Stata/MP licence exposed 16
processors for the host. The H100 medians were 5.910 s for 2.35.0 and 6.281 s
for 2.34.2 at 24 requested native threads, drawn from separate measurement
blocks. Each clock started with the data in memory. The medians describe
those configurations and host conditions.

## Inherited numerical limitation

In the well-conditioned, 1,332-observation synthetic slope fixture with two
cluster dimensions, ordinary H100 absorption gives a relative difference
of about −1.563e-8 against the explicit-OLS intercept variance after an
independent SVD projection. The
diagonal-scaled full-covariance criterion in this check is 1e-8. The
published 2.34.2 and the 2.35.0 source qualification reproduce this
discrepancy. In the same fixture the explicit Schur route gives a maximum
diagonal-scaled covariance difference of about 6.3e-14. The ordinary method
also declines the established near-collinear synthetic case.

## Distributed artifacts

The release workflow records the preflight and version-tag builds, final
artifact hashes, actual OpenMP worker checks and portable runtime closure.
Its native SPI fixtures exercise both macOS architectures and the Windows
Stata installation layout. Licensed Stata execution and H100 fit checks
require the separate host validation of the exact draft assets.

## Publication and exact artifacts

The release tag identifies `d8e6577d1f7c693564d1380c0821de7f4f248f1c`. Publication retained all
14 authenticated draft assets. The public reader
matched 52 Stata site files to the snapshot at
`daad9d511a6202b27c7f1f7865c0fdb1fda50603`. A fresh installation and ancillary download matched
40 files, including the two plugin aliases and both
examples. [Published release](https://github.com/reisportela/xhdfe-xfe/releases/tag/v2.35.0.20261125)

The OpenMP jobs checked useful workers and numerical agreement for the
production plugins, the installed Linux and Windows wheels and the R
package. The macOS jobs checked both native slices and the serial rejection
controls. The Windows Stata plugin loader used the installed layout. The
exact Linux artifacts passed the separate licensed Stata CPU and H100
checks, and both the site and the flat SSC archive passed their Linux
installation checks. Licensed Stata/MP runs on macOS ARM64, macOS Intel
and Windows are recorded as **NOT_RUN**.

The full-input Stata matrix passed 144 paired main–release comparisons
and 72 main CPU/H100 comparisons. The full-input C++ extension recorded 144 passing pairs out of 144.

Nine alternating confirmation pairs covered the three cells initially flagged for timings. The two-order repeatability criterion returned no confirmed material timing loss. In the difficult 10-million-row, three-effect fixture, both versions used CUDA MLSMR with 24 observed threads. The candidate used 492–575 absorption iterations and the release 468–527, with more iterations for the candidate in all six pairs. Fit-time ratios ranged from 0.9856 to 1.0933. The medians were 1.0418 across six pairs and 1.0297 across the three confirmations. They were 1.0541 with the candidate first and 1.0297 with the release first. Three of the six ratios exceeded 1.03.

[Preflight 45595081309](https://github.com/reisportela/xhdfe-xfe/actions/runs/45595081309) · [Version tag 45605311261](https://github.com/reisportela/xhdfe-xfe/actions/runs/45605311261) · [Publication 45628048791](https://github.com/reisportela/xhdfe-xfe/actions/runs/45628048791)
