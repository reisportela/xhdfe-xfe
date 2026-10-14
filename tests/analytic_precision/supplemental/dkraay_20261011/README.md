# Driscoll–Kraay: independent offline verification

## Run from this checkout

From the repository root, use a new output directory for each invocation:

```sh
OPENBLAS_NUM_THREADS=1 python3 tests/analytic_precision/supplemental/dkraay_20261011/independent_oracle.py Verifications/dk_cpu_tiago
OPENBLAS_NUM_THREADS=1 python3 tests/analytic_precision/supplemental/dkraay_20261011/independent_oracle.py Verifications/dk_cuda_tiago --backend cuda
```

The runner loads this checkout's `stata/` by default; `--package /path/to/stata`
selects another candidate explicitly. It uses fresh Stata processes and private
test ado directories, and writes only synthetic fixtures and their results.
The oracle and fixtures are unchanged from the original campaign; the CLI and
path setup were adapted for this testing delivery. Require all 39 CPU cases or
all 32 CUDA cases to pass, `complete: true` in `RESULTS.json`, and exit code 0.
NumPy, pandas, Stata and the candidate's native runtime are prerequisites.

`FINAL_RESULTS.json` is the original 11 October evidence receipt, not a receipt
for a new run or for this checkout's later build.

## Historical campaign

The private 2.29.0.20261006 Stata candidate and original tests are preserved in
`Verifications/perf_max_20260929/implementacao_fecho_revisao_20261010/dkraay_20261011/`
under `/home/mangelo/Documents/GitHub/xhdfe/build-worktrees/perf-max-20260929`.
See its `RESULTADO.md` and `FINAL_RESULTS.json`. That historical environment is
separate from this delivery.

From that campaign directory, using a new output label on each run:

```sh
OPENBLAS_NUM_THREADS=1 python3 independent_oracle.py replay_cpu
OPENBLAS_NUM_THREADS=1 python3 independent_oracle.py replay_cuda --backend cuda
python3 run_diagnostic.py contracts.do replay_contracts
XHDFE_CERTIFY=1 python3 run_diagnostic.py grader_controls.do replay_controls
python3 campaign.py --engine xhdfe --label replay_upstream --only dkraay --equivalent-contracts --fe-tolerance 1e-8
```

CUDA commands require a GPU-visible execution context. All fixtures are
synthetic or public; no packages are installed and the user's ado directories
remain unchanged. Dependencies are isolated in the campaign.

`independent_oracle.py` solves explicit WLS/2SLS with dummy/incidence matrices
and assembles Bartlett covariance from a direct lag sum. It does not use
xhdfe/reghdfe coefficients, residuals or degrees of freedom as oracle inputs.
Coverage: bandwidth/default/cap, unbalanced samples, missing and negative time,
observed-period gaps, permutations, sample selection, singletons, four weights,
weight rescaling, constant/no constant, omitted terms, IV, grouped sum/mean,
and actual CUDA absorption. `contracts.do` adds savefe, prediction, cache,
frequency replication, syntax, one-period inference and explicit processor tests.

`rational_reference.py` proves the public two-FE auto example using exact
fractions. It explains why the raw upstream 1e-12 comparison fails. The
documented port compares this one case with the rational answer at the published
contract; all other upstream assertions retain their original tolerances.
Both raw and adapted results are retained. `grader_controls.do` injects errors
into b and V and requires the corrected grader to reject both.

These checks do not certify the full core24×8 performance matrix, other
interfaces, Windows or macOS. The existing opt-in internal covariance proof
reports DK as unsupported; the independent offline checks are separate evidence.
