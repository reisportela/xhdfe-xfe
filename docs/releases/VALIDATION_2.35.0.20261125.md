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
