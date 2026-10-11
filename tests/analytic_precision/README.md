**Permanent analytic precision certification**

This battery compares xhdfe against finite-sample mathematical truth, with
unmodified reghdfe as a separately graded comparator. It does not designate
either package as the other's oracle. The implemented runner targets the
active Linux C++/Python and Stata artifacts, on CPU and CUDA.

The 25 deterministic fixtures cover OLS with/without a constant, identifiable
contrasts under collinearity, one/two/three/disconnected FE structures,
heterogeneous slopes, aweights/fweights/pweights/iweights (the last in Stata),
conventional/robust/one-way/multiway covariance, singleton removal, group-only
data, and genuine multiple group–individual incidence with sum/mean and
uniform/variable group sizes. A 4,096-group Laplacian case retains the difficult
numerical reference counterexample. The quick grouped matrix uses 256 groups.

Every fixture constructs `y = X beta + D alpha + e` with `X' W e = D' W e = 0`
in the actual finite sample. The identified beta and residuals are therefore
known, rather than merely close to population parameters. Explicit FE matrices
and an independent SVD/FWL solution cross-check the construction. Stata's
`regress` supplies an additional explicit-dummy check. No patched reghdfe or
xhdfe solver is used in an oracle.

Covariances are computed independently from the known residuals and design.
The default multiway convention uses one common `min(G)/(min(G)-1)` factor.
Grouped individual-FE inference records both exact rank-based and documented
conservative-DoF covariance; the latter is the predeclared command contract.
The grader never derives an expected covariance from candidate `e(V)` or its
reported DoF. For multiway clustering, the explicit OLS baseline uses
conventional VCE, since repairing a full dummy covariance to positive
semidefiniteness is not invariant to FE normalization; the independent
multiway sandwich is the covariance oracle for HDFE fits.

Run from the repository root with NumPy, pandas, SciPy, configured `build/`
and `build_cuda/` modules, the current Stata plugin, and Stata/MP with original
reghdfe installed. Use a **new** output directory for every prepared campaign:

```bash
OPENBLAS_NUM_THREADS=1 python3 -B tests/analytic_precision/run.py prepare --output Verifications/analytic_precision/new-run
OPENBLAS_NUM_THREADS=1 python3 -B tests/analytic_precision/run.py run --output Verifications/analytic_precision/new-run --backend cpu
OPENBLAS_NUM_THREADS=1 python3 -B tests/analytic_precision/run.py run --output Verifications/analytic_precision/new-run --backend cuda
OPENBLAS_NUM_THREADS=1 python3 -B tests/analytic_precision/run.py report --output Verifications/analytic_precision/new-run
```

CUDA execution must run in a GPU-visible context. In a restricted Codex
sandbox, follow AGENTS.md's host-level GPU probe policy. The runner records
`nvidia-smi`, and a CUDA precision row requires actual `gpu_used == 1`.
Fresh processes, temporary files and CUDA caches stay within the campaign.

`prepare` runs independent-oracle assertions and negative controls against the
grader, then hashes scripts, fixtures, active binaries and canonical reference
files into `manifest.json`. `run` refuses changed custody, preserves raw FP64
artifacts and logs, and resumes only attempts with completed assessment files;
an incomplete attempt is not silently overwritten. The default timeout is
420 seconds per attempt. Partial runs may use `--filter`, `--limit`, or a
smaller `prepare --cases`/`--interfaces`/`--backends` selection. Their smaller
coverage must remain explicit. Names must come from the supported catalogue;
unknown/duplicate cases, interfaces or backends are rejected before outputs
are created. Preparation always includes CPU; `--backends cpu` omits CUDA.

`report` creates a new report JSON, includes every planned row (unexecuted
rows are `NOT_RUN`), and returns nonzero unless all scoped xhdfe positive and
negative contracts pass with unchanged custody. It also reports the default
xhdfe subset separately. Passing defaults does not erase failures in explicit
methods or interfaces. reghdfe failures remain visible without automatically
failing an independently correct xhdfe result.

The documented controls mutate beta despite a forged convergence certificate,
covariance, sample size and residual finiteness; request a GPU while returning
CPU diagnostics; and replace an expected unsupported error with an unrelated
missing-command failure. These must be rejected. Unsupported combinations
count separately from precision PASS, including Stata Schwarz, grouped
savefe, and heterogeneous `strict-residual`. Rejection requires an informative
error and no finite coefficient, covariance or residual arrays. Returning
nonconverged estimates cannot satisfy an expected rejection. The grader's
negative controls also test errors carrying stale estimates.
Requested and effective methods are both retained: heterogeneous
Jacobi resolves to SGS, and Auto-MLSMR may resolve to a sweep.

Default coefficient tolerance is 1e-8, strict coefficient tolerance 1e-10,
residual tolerance 1e-7, covariance `atol=1e-12, rtol=1e-7`, and RSS tolerance
`1e-8 + 1e-10*abs(oracle RSS)`. FE recovery contributes an additional 1e-6
identity gate. These thresholds precede the campaign and cannot be relaxed
after observing a failure. Convergence flags alone cannot obtain PASS.

Boundaries explicitly pending in the manifest include IV/2SLS, R/formula
interfaces, persistent-cache signature mutation, weighted singleton cascades,
macOS/Windows execution, and full-data performance. This battery supplements
the existing inference, feature, release and benchmark suites; it does not
replace them or certify every possible dataset/platform configuration.

A separate Stata IV guard, `iv_repeated_control_contract.do`, covers exact
variable-name overlap between included controls and `instruments()`. From a
scratch directory run `stata-mp -b do /path/to/iv_repeated_control_contract.do cpu`
(or `cuda`) with `XHDFE_STATA_ADOPATH` selecting the candidate. It compares
the clean and repeated-control specifications, coefficients against independent
explicit-dummy `ivregress 2sls` and `ivreghdfe`, repeated estimates and covariance,
sample/DoF/RSS, backend use, and failed-fit state. Require the
`IV_REPEATED_CONTROL_CONTRACT_PASS` marker. This focused IV test is separate from
the analytic campaign above; proportional duplicates and general numerical-rank
decisions remain outside its coverage. The special self-instrumentation case
uses explicit-dummy OLS and reghdfe because ivreghdfe rejects that syntax.

The proportional-instrument repair has separate permanent guards:

- `iv_exact_proportional_contract.cpp` reads represented binary64 inputs and
  checks the production filter. Compile it with the candidate's flags and
  `-Iinclude -Isrc -I/path/to/Eigen`; use `-fopenmp -DHDFE_USE_OPENMP` for
  explicit-team coverage. Run `iv_exact_proportional_contract.py --probe BIN
  --out NEW.json --threads 1,2,8,16,24,48`. Its independent `Fraction` oracle
  covers 4,396 cases per team, including subnormals, extreme exponents, signed
  zeros and one-bit differences. A 6,000-row check verifies that a distinction
  in a late row survives and every requested worker actually performs work.
- `iv_proportional_instrument_contract.py --module MODULE --backend cpu|cuda
  --out NEW.json` runs 28 cases per backend against the conditioning contract's
  independent Walsh scores. It combines exactly proportional columns with tiny
  genuine directions, two endogenous regressors, weights, robust/clustered
  covariance, both tolerance modes, positive-weight sample selection, and
  underidentification. The multiple-endogenous weighted fixture explicitly
  checks the score-space and FE orthogonality assumptions before evaluating a fit.
- `iv_proportional_instrument_contract.do cpu|cuda`, with
  `XHDFE_STATA_ADOPATH`, checks 210 CPU / 180 CUDA fits against the clean
  specification, independent explicit-dummy 2SLS/OLS and ivreghdfe/reghdfe.
  Require `IV_PROPORTIONAL_INSTRUMENT_CONTRACT_PASS`. The original first-stage
  contract now also runs the proportional case on CUDA.
- `n1_iv_cache_contract.py` crosses its independent IV score and saved-FE
  recovery fixtures with and without proportional columns. Its existing OLS
  persistent-cache checks are unchanged.

These tests do not certify arbitrary multicolumn dependencies, FE-induced
redundancy, cross-platform packages, or runtime performance.

The multiway IV covariance repair has two complementary independent guards:

- The unchanged `iv_intercept_covariance_contract.py --module MODULE --out
  NEW.json --mode xhdfe-fast|reghdfe-comparable` covers 46 full covariance cases
  per mode. `iv_intercept_covariance_cuda_contract.py` runs the same references
  and requires real CUDA use. It includes equivalent cluster partitions, raw
  regressor means, analytic/frequency weights, SSC choices and perfect fits.
- `iv_covariance_psd_contract.py --module MODULE --backend cpu|cuda --mode
  xhdfe-fast|reghdfe-comparable --out NEW.json` adds 44 cases per backend/mode.
  Fixed partitions (seeds 2/102 and 11/111) require actual eigenvalue correction;
  equivalent partitions must preserve the positive sandwich. The MP90 and
  MP120 references independently construct the cluster scores and sandwich,
  then apply the documented standardized full-matrix and separate slope-block
  rule. At least 12 controls must detect an implementation that skips the
  projection. Every coefficient/covariance limit remains 1e-9/1e-8.
- For Stata, first run `iv_covariance_psd_stata_fixtures.py NEW_DIRECTORY`, then
  `do iv_covariance_psd_contract.do cpu|cuda NEW_DIRECTORY` from a scratch
  directory with the candidate selected by `XHDFE_STATA_ADOPATH`. This tests
  88 fits per backend, including `e(vcv_psd_fixed)` and real CUDA diagnostics,
  against the same independent MP120 oracle. Require the
  `IV_COVARIANCE_PSD_CONTRACT_PASS` marker. A wrapper do-file avoids ambiguity
  in the log filename chosen by Stata batch mode for multiple arguments.

These covariance tests do not substitute for full release performance gates.

The scope, source hashes, fixtures and complete per-row results must accompany
any certification claim. Preserve failed evidence, investigate oracle or
interface-contract mistakes separately from product failures, and prepare a
fresh campaign after correcting the test implementation.

Additional permanent grouped guards are `near_null_contract.py` (an analytic
beta=0.7 design, including good-fit/bad-refit state revocation) and
`group_formula_contract.py` (prepared snapshots, group sample mapping, missing
identifiers and literal frequency-weight semantics). Both take explicit
`--module`, `--backend` and a new `--out`; the formula guard also takes
`--formula-source`. They are separate scoped evidence, and must be run on the
same candidate bytes as the main campaign. The Stata test suite includes
`part1/group-individual-analytic.do` (64/128-node chains, explicit OLS, both
aggregations and absorb orders) and `part1/group-individual-near-null.do`.

The near-null guard must run at both `--nodes 96` and `--nodes 600` on CPU
and CUDA. The latter has 19,200 groups and 1,200 FE patterns and exposed a
silent verifier bypass that the smaller tests did not detect. The R equivalent
is `near_null_r.R library dependencies backend new-output.json 600` (96 is the
default final argument). A correct fit preceding every bad refit is mandatory;
refusing all fits cannot satisfy this guard. Preserve both results and report
any resource refusal as a limitation, never a verified numerical success.

`ols_inference_range_contract.py` is a separate CPU guard for positive robust
or clustered variances lost through intermediate underflow, despite a
representable analytical answer. It uses 20 small Walsh fixtures and one
failed numerical refit after success. Ordinary, rescaled, near-collinear and
weighted designs have closed-form covariance references; exact zero variances
from disjoint score support or within-cluster cancellation remain valid, with
undefined t/p/CI following the existing NaN convention. Supply `--module`,
`--module-sha256` and a new JSON `--out` in an existing directory. An optional
`--compare-normal-from` receipt from the same worker compares the nine ordinary
controls. The full-covariance scaled limit is 1e-8. The two extreme aweight
cases permit a specific numerical-range error with empty result state;
these are reported as safe refusals, separately from successful estimates.
Run this guard on the same candidate bytes as the main campaign. Its CPU-only
scope does not replace Stata/R/CUDA validation or performance measurements.

`disconnected_chain_contract.py` checks two, three and nine disjoint patent
chains, with both aggregations, weights and strict/default criteria. Its small
synthetic CSV fixture has a provenance/hash sidecar. The reference uses the
closed-form projector separately on every component, then pools the normal
equations. This catches false rejections that a single-chain test can miss.

A classification-only review may use `regrade.py` with the original evaluator
preserved in the campaign's `source_snapshot/`. It verifies unchanged execution
sources, fixtures and binaries, hashes every raw result, reruns the grader's
negative controls and asserts that the PASS/failure set cannot change. Its
only permitted relabeling is a sample-provenance failure previously included
under the broader false-convergence label. It writes separate evidence and
does not overwrite the original campaign or change any numerical threshold.

## T01 Fast/Comparable adjudication

`evaluate.py` retains the historical verdict and thresholds. T01 adds a
separate `t01` object through `evaluate_t01.py`; it never converts a historical
failure into PASS. `precision_contract.py` freezes the mode rules:

- Fast keeps `max(abs(C delta_beta)) <= 1e-8` and covariance
  `atol=1e-12, rtol=1e-7`, with N1 `tau=tol`.
- Comparable inherits those gates and additionally applies the historical
  coefficient scale `1e-9*max(1,abs(C beta_ref))`, covariance
  `1e-8*sqrt(Vref_jj*Vref_kk)+A_jk`, and N1
  `tau=min(tol,1e-9)`.

`A_jk` is generated with each fixture before any fit. It uses `gamma_m`, the
Gram condition and absolute products in that fixture's bread/meat covariance
graph. It has no empirical floor and does not depend on a candidate/reference
difference. If the first-order condition `gamma_m*kappa(G)<1` fails, an exact
covariance match may still pass; any nonzero discrepancy is
`CONTRACT_INCOMPLETE`, not silently accepted. Undefined t/p/CI when both the
coefficient and standard error are zero are outside the gate.

The joint fitted-value metric and the provisional `1e-8` H2 value are recorded
only as diagnostics. `precision_certified` reported by a fit is never used as
an independent oracle. A valid requested fit that refuses or returns a wrong
result fails G1; `UNSUPPORTED_EXPECTED` is reserved for combinations documented
as unsupported before execution. Group/individual near-null protection remains
required in the default path.

Prepare separate default and audit campaigns with `--certificate-mode default`
and `--certificate-mode audit`; the latter sets only `XHDFE_CERTIFY=1` in
addition to the pre-existing runner environment. Pin native modules with
`--native-cpu-module`/`--native-cuda-module` and an isolated Stata package with
`--stata-package-root`. Their paths and SHA-256 values enter the campaign
manifest. `t01_regrade.py CAMPAIGN` adds the frozen adjudication to immutable
raws and writes a new report without overwriting the historical one.

The files under `supplemental/` preserve focused savefe, pure-slope, extreme
weight, leverage/noise and redundant-FE controls with source provenance in
`supplemental_cases.json`. For the three valid extreme-weight/leverage/redundant
contracts, an informative empty refusal is `FAIL_VALID_REFUSAL`; it is not an
expected success. These controls, the paired default/audit campaigns, R/formula
coverage and the mandatory core24 x 8 matrix remain separate gates.

`run_r.py` derives the R matrix/formula jobs from a prepared main campaign.
Preparation requires explicit installed CPU/CUDA xhdfe library roots and may
receive repeated `--dependency-library` roots. It hashes every file in each
installed xhdfe package and the exact DLL. The R worker loads xhdfe with
`lib.loc` and verifies both `find.package("xhdfe")` and the loaded DLL path;
dependency and base/system libraries remain available. `--vanilla`, isolated
temporary/cache paths and the campaign's default/audit `CERTIFY` mode prevent
ambient user configuration from selecting another xhdfe. Missing rows remain
`COVERAGE_MISSING`; a partial R run cannot report scoped PASS.

`run_supplemental.py` pins CPU/CUDA modules, both installed R packages, the
Stata ado/plugin and executable paths before preparing 13 focused jobs. It
aggregates all planned rows, including absent ones, and distinguishes a valid
certificate refusal from a wrong returned result where the result schema makes
that distinction possible. The independent rational/reference files are
custody inputs, not executed product oracles.

`aggregate_t01.py` requires main, R and supplemental reports for both default
and audit modes. Only the complete default set can produce
`default_scoped_functional_pass`. Audit verifier refusals are reported
separately and do not overturn a correct default fit; audit harness/coverage
gaps and wrong returned estimates remain failures. The aggregate always leaves
`full_G1_claim=false` while IV/X-hat normal-defect coverage, formal covariance
interval gaps, core24 x 8 and the group/individual 3M gate remain outstanding.


## Wide cluster covariance storage reuse (2026-10-08)

`wide_cluster_covariance_reuse.cpp` exercises the retained wide multiway
covariance path directly, including 2/3/5 regressors, explicit constants in
first/last position, no explicit constant, weights, response scales 2^-80,
1 and 2^80, and minimum/conventional cluster adjustments. All 84 cases require
this path to have been entered and expose both the retained and returned full
covariance. This covers buffer sizing/reset and the original-units transform.

Compile the driver against each frozen baseline/candidate `libxhdfe.a`, using
that build's Eigen headers and OpenMP/CUDA link dependencies. Run the two
executables into separate JSONL files, then grade outside the estimator:

```bash
python tests/analytic_precision/wide_cluster_covariance_reuse_contract.py \
  --baseline baseline.jsonl --candidate candidate.jsonl --out reuse_validation.json
```

The grader constructs an independent MP90/MP120 augmented sandwich reference,
checks the unchanged 1e-8 scaled covariance and 1e-9 coefficient/RSS limits,
and requires exact baseline/candidate output equality for these fixtures.
Existing IV intercept/PSD contracts cover the full Python/Stata estimation
and PSD postprocessing; this test supplies additional native-path coverage.

## Fast heterogeneous-slope acceptance policy

`fast_slope_policy_contract.cpp` calls the real certificate on an approximate
projection whose historical aggregate residual passes while the comparable
block residual fails. It also checks the exact projection, weighted cases,
slope-unit changes, recertification scope, non-convergence, non-finite values,
and an aggregate residual that must still be rejected in fast mode. Link it
against the matching native archive with that build's include/compile/link flags.

`fast_slope_fit_contract.py --module MODULE --backend cpu|cuda --out OUTPUT`
records real weighted/unweighted fits with and without retained effects, both
modes, explicit-dummy OLS coefficient references, full covariance and residuals.
Run baseline and candidate separately and compare comparable outputs. The
acceptance-wrapper contract additionally verifies fast mode never performs a
block-driven retry, and comparable mode retains its continuation/refusal rules.
The new `slope_block_checked_` diagnostic (Stata: `e(slope_block_checked)`, R:
`slope_block_checked`) distinguishes skipped checks from zero residuals. The
five existing block metrics are zero placeholders when it is false.
