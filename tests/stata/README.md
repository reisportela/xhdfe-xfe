# Stata certification tests

This directory contains Stata-side certification tests for `xhdfe`. The style is
adapted from Sergio Correia's `reghdfe/test` suite: each test runs a trusted
Stata baseline, runs `xhdfe`, and compares the public estimation contract
instead of only checking that commands execute.

The oracles are built-in Stata commands (`regress`, `areg`) plus the community
packages `reghdfe` and `ivreghdfe`. **Both `reghdfe` and `ivreghdfe` are hard
prerequisites**: `testall.do` runs every test unconditionally and the suite
aborts (rc=601) if either oracle is missing. Install them first with
`ssc install reghdfe` and `ssc install ivreghdfe` (plus `ftools`).

## Run

Build or install a matching Stata plugin first, then run:

```bash
tests/stata/run_stata_tests.sh
```

The runner defaults to `stata-mp` and adds the repository `stata/` directory to
the Stata `adopath`. Override these with:

```bash
STATA_BIN=/path/to/stata-mp XHDFE_STATA_ADOPATH=/path/to/xhdfe/stata tests/stata/run_stata_tests.sh
```

To build the local plugin before running the tests:

```bash
XHDFE_BUILD_STATA_PLUGIN=1 tests/stata/run_stata_tests.sh
```

Logs are written under `tests/stata/output/`.

## Coverage

- no-absorb OLS parity against `regress`
- one-way absorbed OLS parity against `areg`
- robust and clustered VCE parity against `areg`
- analytic and frequency-weight parity against `areg`
- factor-variable parsing under `absorb()`
- multiway fixed-effect parity against `reghdfe`
- multiway clustered-VCE parity against `reghdfe`
- heterogeneous-slope parity against `reghdfe`
- saved fixed-effect reconstruction
- named/aliased saved-FE mapping vs `reghdfe` (regression guard for mixed unnamed+named `absorb()` terms)
- IV/2SLS parity against `ivreghdfe`
- probability-weight and importance-weight behavior
- `group()` / `individual()` behavior and aliases
- missing-data, sample-restriction, and expected-error behavior
- DoF/SSC/stat-style option acceptance and aliases
- string fixed-effect/cluster IDs and interaction-ID materialization
- collinearity and omitted-coefficient marking
- `noconstant` and slope-only constant reporting
- `estat summarize`, `estat ic`, and `estat vce`
- `nosample` postestimation behavior
- forced absorption-method coefficient parity
- postestimation smoke and identity checks for `score`, `stdp`, `margins`,
  replay, `test`, `estat vce`, and heterogeneous-slope predictions
- `predict` parity for `xb`, `d`, `xbd`, residuals, and `dresiduals`
- singleton/drop and convergence-option behavior

## Upstream notice

Several test scenarios and the certification-test layout are adapted from the
MIT-licensed `reghdfe` test suite:

Copyright (c) 2014 Sergio Correia

The original license is MIT. Preserve this notice when modifying this Stata test
suite.
