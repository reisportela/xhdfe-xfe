# xhdfe 2.29.5.20261009 — development candidate

Explicit `xhdfe-fast` fits with heterogeneous slopes use the historical
aggregate absorption certificate and skip the stronger per-block certificate
and block-driven continuation. Solver convergence, finite-value checks and
the aggregate acceptance limit remain active. If the aggregate check fails,
the existing same-method repair or refusal still applies.

The default `reghdfe-comparable` mode retains its block checks and refinement.
This is an intentional fast-mode precision trade-off: difficult slope designs
can return less accurate absorbed variables than comparable mode. It does not
change the public tolerance, estimator specification, sample or covariance
formula. Ordinary fixed effects and group/individual policy are unchanged.

The new `slope_block_checked_` diagnostic (Stata: `e(slope_block_checked)`, R:
`slope_block_checked`) records whether the block check ran. Existing block
metrics contain zero placeholders when it did not; those zeros are not measured
residuals. `precision_certified` refers to the selected mode's acceptance check.

The full firm-specific AKM test on Athena restores the old 30 CPU / 38 CUDA
iterations. Three-pair Python medians fall from 404.27 to 114.79 seconds on CPU
and from 287.61 to 59.97 seconds on CUDA. Single Stata comparisons fall from
422.24 to 133.46 seconds and from 316.43 to 76.74 seconds. These results recover
93–96% of the slowdown relative to historical old medians; fresh old-version
anchors give similar or longer times than the candidate. No warm-up fits are
used. Coefficients and standard errors match the old fast results to numerical
precision, with the maximum SE difference below 3.26e-13.

Permanent tests cover 32 acceptance/refusal cases, six real certificate
fixtures, weights, slope-unit changes and retained effects. Both builds pass
66 slope-unit/rank cases. The main battery has 26 successful fits and one
matched Stata CPU comparable-mode iteration-limit refusal pair. Comparable
CPU numerical outputs are identical on the successful Python subset; CUDA
coefficient/SE differences stay below 2.6e-11 / 5.6e-11. GPU control iteration
counts differ; this does not certify comparable-mode runtime non-regression.
Two existing comparable CUDA refusal pairs also remain in the synthetic suite.
The corrected subset harness and all rejected initial attempts are documented
in the [validation report](../FAST_SLOPE_POLICY_VALIDATION_20261009.md).

This development candidate is recorded in a local commit after the focused
validation above. It has not been pushed, installed, issued or promoted through
the full release matrix.
