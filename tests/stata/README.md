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

Logs are written under `tests/stata/output/`. The runner also executes the
separate `xhdfegelbach_bootstrap_smoke.do` gate and requires its explicit PASS
marker.

`xhdfegelbach_bootstrap_cuda_smoke.do` is the opt-in H100/CUDA gate. It
requires affirmative CUDA use for the point estimate and every valid
observation-pairs and declared-cluster-pairs replication; it is intentionally
not part of the CPU/default runner.

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
- IV first-stage diagnostics against `ivreghdfe` and direct `reghdfe` first
  stages, including the documented nested-cluster degrees-of-freedom factor
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
- data order and `sortedby` marker left untouched by successful and failed
  commands, on unsorted, panel-sorted and regressor-sorted data, with
  time-series operators and with automatic outputs used as sort keys
  (`part1/sort-order.do`)
- Gelbach iid/declared-cluster full-refit bootstrap reproducibility, manual
  first-draw and basic-CI oracles, analytic weights, added/common FEs,
  fail-closed validation, multi-format escaped tables and
  identity-preserving waterfall filtering
- explicit OpenMP teams of 1 to 48 threads and the automatic rows-per-thread
  rule at its row-count boundaries (`part1/threads-contract.do`), which
  `run_stata_tests.sh` runs in three fresh processes: with the built-in density
  and with `XHDFE_AUTO_ROWS_PER_THREAD=500000`, both with `XHDFE_PARALLEL_READ=0`
  so that `e(threads_used)` is the fit's own team, and with the built-in density
  and the parallel read left to the host; all with `OMP_NUM_THREADS` unset; the
  runner skips it with a message when `nproc` reports fewer than 48 logical
  processors

`openmp_host_contract.do` is not part of the runner: `tools/run_stata_openmp_host.py`
runs it for each thread request with `XHDFE_PARALLEL_READ=0` (the fit's own
workers) and again with the parallel read left to the host.

## Version and categorical-interaction contracts

Three standalone contracts run inside `run_stata_tests.sh` and can be run alone
in a scratch directory, each with an explicit `cpu` or `cuda` argument and a
`..._PASS <backend>` marker:

```bash
XHDFE_STATA_ADOPATH=/path/to/xhdfe/stata stata-mp -b do /path/to/xhdfe/tests/stata/version_contract.do cpu
XHDFE_STATA_ADOPATH=/path/to/xhdfe/stata stata-mp -b do /path/to/xhdfe/tests/stata/absorb_interaction_contract.do cuda
XHDFE_STATA_ADOPATH=/path/to/xhdfe/stata stata-mp -b do /path/to/xhdfe/tests/stata/cluster_interaction_contract.do cpu
```

`version_contract.do` requires the ado header, `xhdfe, version`, and `e(version)`
after OLS, IV, `estat firststage` and replay to agree. The two interaction
contracts compare `absorb()`, `cluster()`, and `xfepout` `absorb()`/`clustervars()`
terms with three or more categorical components against the manually grouped
identifier (bit identity of samples, DoF and `e(b)`/`e(V)`) and against `reghdfe`,
and check the negative controls (`##` with more than two components in cluster
options, continuous components in interaction terms).

## Input-read contract

The plugin reads its numeric inputs in one pass by observation and falls back
to the column readers on any anomaly; `XHDFE_FUSED_READ=0` keeps the column
readers. On a host qualified for it (Linux x86-64, Stata 19 MP) a pass of at
least 2,000,000 values (rows times the variables the fused pass reads) with
more than one thread goes through the parallel reader: the sample is selected
from the marker variable the ado appends, and several threads read blocks of
16,384 rows. The team of each pass is the fit's own team for its rows, so
every threading option governs it, with at most one thread per block (and 16
in automatic mode); `e(threads_used)` and `e(parallel_workers_active)` count
the read's team and workers. Any anomaly discards the parallel pass and the
serial pass reads everything again, so errors and messages are those of the
serial readers; calls that receive an alias variable (`fralias`) keep the
serial readers. `XHDFE_PARALLEL_READ=0`
keeps the serial readers on every host; `XHDFE_PARALLEL_READ=1` forces the
parallel reader on any host and exists only for qualification runs;
`XHDFE_PARALLEL_READ_MIN_VALUES=<n>` replaces the size threshold and is a test
hook; `XHDFE_PARALLEL_READ_PRETOUCH=0` lets the worker threads touch the
destination pages first instead of the calling thread, for timing comparisons.

`fused_read_contract.do` runs 47 fits, 49 on Stata 18 and later (identifiers
that fit `int32` and identifiers that do not, also only in a later block,
clusters shared with and distinct from the fixed effects, weights, slopes,
instruments, strings, interactions, sample restrictions, `in` ranges, samples
of one block of 16,384 rows, of one row more and of five blocks, three
automatic fits with `targetrowsperthread()` and `maxthreads()`, one automatic
fit on `in 1/5000` whose `e(threads_used)` must be the selection's team of 3 in
the parallel run and 1 elsewhere, with `e(threads_effective)` 1, saved outputs,
`group()`/`individual()`, alias variables), four invalid individual
identifiers and an empty sample, and writes every result in hexadecimal; its
second argument sets `numthreads()` (default 1; the automatic fits use 0).
`run_stata_tests.sh` runs it in four fresh processes: the column and the fused
reader with one thread, then the serial fused reader (`XHDFE_PARALLEL_READ=0`)
and the forced parallel reader (`XHDFE_PARALLEL_READ=1`,
`XHDFE_PARALLEL_READ_MIN_VALUES=1`) with four threads and `OMP_NUM_THREADS`
unset. Each pair must give identical result files and all four runs identical
input fingerprints (`plugin_profile label=input_digest` under
`XHDFE_PROFILE_CPU=1`); the `plugin_profile label=read_parallel_mode` lines
must show the parallel reader with teams of three and four, at least one
anomaly sent back to the serial reader, the automatic read following
`targetrowsperthread()` and `maxthreads()` (three threads, two threads, serial),
the serial reader kept for alias variables, and no parallel pass in the serial
run. The four-thread pair is skipped with a message on hosts with fewer than
four logical processors.

## First-stage diagnostics

The standalone `iv_firststage_contract.do` uses 6,000 synthetic observations.
Run it in a scratch/output directory with the candidate package on the adopath:

```bash
XHDFE_STATA_ADOPATH=/path/to/xhdfe/stata stata-mp -b do /path/to/xhdfe/tests/stata/iv_firststage_contract.do cpu
XHDFE_STATA_ADOPATH=/path/to/xhdfe/stata stata-mp -b do /path/to/xhdfe/tests/stata/iv_firststage_contract.do cuda
```

Require the explicit `IV_FIRSTSTAGE_CONTRACT_PASS` marker. Each run writes a CSV
comparison with `reghdfe` and `ivreghdfe`, and the CUDA run requires actual GPU
use in the IV regression and both auxiliary regressions. The tests cover
multiple endogenous variables, robust/clustered errors, weights, missing values,
singleton removal, redundant instruments, weak instruments, replay, `nosample`,
and restoration of the IV estimates after successful and failed postestimation.
The few-cluster two-way case pins ivreghdfe's unavailable system statistic and
checks the per-equation F tests against reghdfe. No timing or general
release-certification claim follows from these small tests.

The CPU run covers 23 specifications (29 first-stage equations); CUDA covers
21 specifications (27 equations). No-absorb regressions are CPU-only. Both
backends cover zero instrument columns, repeated included controls, and exactly
proportional instruments under different variable names.
The proportional-instrument repair is covered on both backends. Repeated-fit
comparisons allow FP64 roundoff;
postestimation restoration and cached-result checks require exact equality.
See the [initial candidate validation](../../docs/certification/iv-firststage-20261007.md)
and [repeated-control follow-up](../../docs/certification/iv-repeated-controls-20261007.md).

`iv_firststage_precision_probe.do` is a separate, opt-in CPU/CUDA investigation
of the probability-weighted case. It compares tolerances 1e-8, 1e-12, and 1e-14
with reghdfe/ivreghdfe and an explicit-dummy `regress` reference, recording
first-stage coefficients, covariance entries, F-statistics, and sample/DoF
checks. It requires CUDA and the `IV_FIRSTSTAGE_PRECISION_PASS` marker.
Run it from a scratch directory with `XHDFE_STATA_ADOPATH` set as above.

The standalone `../analytic_precision/iv_repeated_control_contract.do` accepts
the same `cpu`/`cuda` argument and adopath environment variable. It checks that
adding included controls to `instruments()` preserves coefficients, covariance,
sample, RSS, degrees of freedom and IV metadata across repeated runs, while
underidentified models still fail with empty estimates. It includes explicit
FE-dummy `ivregress 2sls` and `ivreghdfe` coefficient comparisons; the special
self-instrumentation case uses its explicit-dummy OLS equivalent. Require
the `IV_REPEATED_CONTROL_CONTRACT_PASS` marker. Its CPU mode also runs in
`run_stata_tests.sh`.

The standalone `../analytic_precision/iv_covariance_psd_contract.do` compares
88 fits per backend with MP120 full-covariance references, checks the PSD-fix
flag, and exercises both tolerance modes and genuine negative-eigenvalue cases.
Generate its input directory with `iv_covariance_psd_stata_fixtures.py`; pass
`cpu` or `cuda` and that directory to the do-file. See the analytic battery's
README for the commands and accuracy limits.

## Upstream notice

Several test scenarios and the certification-test layout are adapted from the
MIT-licensed `reghdfe` test suite:

Copyright (c) 2014 Sergio Correia

The original license is MIT. Preserve this notice when modifying this Stata test
suite.
