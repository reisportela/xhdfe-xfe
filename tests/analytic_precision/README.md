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
Fast is planned explicitly on both CPU and CUDA, for each applicable interface.

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
The HDFE receipts also carry the complete coefficient/covariance block. The
identified reported constant follows the weighted zero-mean FE normalisation;
the independent multiway reference applies the documented D-C repair when
needed. Mutations of the constant, its variance and its cross-covariances must
fail even when the slope block is correct. Explicit-dummy OLS retains its
separate slope-contrast check because its extra coefficients use another basis.

Run from the repository root with NumPy, pandas, SciPy, configured `build/`
and `build_cuda/` modules, the current Stata plugin, and Stata/MP with original
reghdfe installed. Use a **new** output directory for every prepared campaign:

```bash
OPENBLAS_NUM_THREADS=1 python3 -B tests/analytic_precision/run.py prepare \
  --output Verifications/analytic_precision/new-run \
  --native-cpu-module build/py_hdfe_v11.cpython-312-x86_64-linux-gnu.so \
  --native-cuda-module build_cuda/py_hdfe_v11.cpython-312-x86_64-linux-gnu.so \
  --stata-package-root .
OPENBLAS_NUM_THREADS=1 python3 -B tests/analytic_precision/run.py run --output Verifications/analytic_precision/new-run --backend cpu
OPENBLAS_NUM_THREADS=1 python3 -B tests/analytic_precision/run.py run --output Verifications/analytic_precision/new-run --backend cuda
OPENBLAS_NUM_THREADS=1 python3 -B tests/analytic_precision/run.py report --output Verifications/analytic_precision/new-run
```

The example names the Linux CPython 3.12 modules; supply the exact filenames
of the builds being assessed. An old receipt lacking the complete parameter
block cannot pass the new full-parameter check and needs a new campaign.

Two separate regression guards cover the 14oct2026 audit findings:

```bash
python3 -B tests/analytic_precision/dkraay_iv_influence_contract.py \
  --backend cpu --package stata --output Verifications/dk_iv_cpu_new
python3 -B tests/analytic_precision/dkraay_iv_influence_contract.py \
  --backend cuda --package stata --output Verifications/dk_iv_cuda_new
```

The DK guard uses independent MP90/MP120 Walsh-score covariances, common and
opposite regressor means, both tolerance modes and the bandwidth-one cluster
identity. `tests/stata/iv_firststage_alias_contract.do cpu|cuda` exercises named
FE outputs, interactions, slopes, time-series terms, weights, DK, replay and
`nosample`; require `IV_FIRSTSTAGE_ALIAS_CONTRACT_PASS` with its backend. The
Stata suite runs both new CPU guards; CUDA needs the GPU-visible context below.

CUDA execution must run in a GPU-visible context. In a restricted Codex
sandbox, follow AGENTS.md's host-level GPU probe policy. The runner records
`nvidia-smi`, and a CUDA precision row requires actual `gpu_used == 1`.
Fresh processes, temporary files and CUDA caches stay within the campaign.

The 19 October CG regression guards are separate from that matrix:
`cg_restart_budget_contract.py`, `cg_projection_full_covariance_contract.py`
and `cg_final_step_contract.py` accept `--module PATH --backend cpu|cuda`
and an optional new `--output JSON`. The final-step fixture is public synthetic
data, checked against MP120 with complete OLS/IV covariance, three row orders
and distinct mode contracts. `cg_final_step_oracle.py` independently regenerates
the MP80/MP120 reference. `cg_cache_generation_contract.py --baseline-module
OLD_MODULE --module NEW_MODULE --output NEW_DIRECTORY` tests a genuinely
incorrect old cache without changing its bytes. Its default projection fixture
covers generation 4433; `--fixture final-step` covers generation 4434.
The old build must fail the independent contract and the new build must pass.

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

At execution, a filter/backend selecting no jobs is an error. The runner
reports selected, completed and pending counts; `ALREADY_COMPLETE` means a
valid selection needs no new attempts, while `PARTIAL` records a bounded run
with pending work. Neither state replaces `report` as the campaign verdict.
An interrupted attempt directory is preserved and requires a new campaign.
`runner_selection_contract.py --module PATH --output NEW_DIRECTORY` exercises
these behaviours through the real CLI with the synthetic OLS fixture,
including refusal of an empty selection and preservation of prior receipts.

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
- `iv_cache_contract.py --module MODULE --out NEW.json` (CPU only, 13oct2026)
  exercises the persistent absorption cache on an IV fit: the identical read
  must hit (`cache_hit=1` from the `XHDFE_DEBUG_SOLVER` stderr line) and be
  bit-identical to the writing fit; one-ULP changes in an instrument, the
  response or a regressor, a 256-row sample change and the other tolerance
  mode must miss and still match explicit 2SLS with FE dummies; the cache file
  is unchanged after reads.
- `iv_wide_omitted_contract.py --module MODULE --out NEW.json --mode
  xhdfe-fast|reghdfe-comparable` (13oct2026) adds an exogenous Walsh regressor
  and its exact multiple to the known-score IV design, so one column is
  omitted on the retained wide covariance path: 92 cases per mode (robust,
  cluster, multiway with equivalent and positive-definite distinct partitions,
  analytic and frequency weights, both column orders) grade the identified
  4x4 block against an MP90 reference and require exact zeros for the omitted
  row and column.
- `supplemental/weak_iv_dkraay_20261013/weak_iv_dk_oracle.py NEW_DIR
  [--backend cuda]` reuses the 11oct2026 Driscoll-Kraay oracle on a panel with
  a weak excluded instrument and grades `e(first)` (partial R2, F, df, df_r,
  p) against an explicit first-stage DK Wald oracle; allowances are
  predeclared from the 2SLS score Gram condition number.

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
  rule, followed by the D-C constant repair (policy of 10oct2026, applied to
  the reference on 13oct2026; see `docs/LICOES_INTEGRACAO_PR13_20261012.md`). At least 12 controls must detect an implementation that skips the
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

`pf3fe_covariance_contract.py` exercises the confidential PF three-FE input
in place, using a pinned independent aggregate Schur oracle. It grades the
complete covariance at the original diagonal-scaled `1e-8` limit and the
coefficients at `1e-9`, checks sample/backend/iteration accounting, and keeps
failed-refit cleanup and recovery separate from numerical passes. Pass the
exact module, data, oracle/hash and a new JSON output path. Non-nested
clusters, relabelled clusters, reversed FE order and the cluster-DoF toggle
require their matching independently constructed oracle. Short budgets may
require refusal (`--expect-refusal`) or allow a clean refusal while grading
any returned result (`--allow-refusal`). No observations or residual arrays
are written. This regression test is independent of the synthetic campaign
and is not a performance certificate.

`--phase-proof` also captures the CUDA correction's host preconditioner
trace. Each independent level pass must expose at least as many work units
as requested threads, form the real requested team and observe useful work
on multiple workers when more than one is requested. Active-worker counts
are recorded separately; dynamic scheduling need not give every worker a
task in every pass. The triangular solve reports its structural RHS width
separately. An aggregate `threads_used` value alone is insufficient.

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

## Approx-Cholesky adjacency ownership and factor equality

`ac_factor_storage_contract.py` copies the exact builder source bodies into
an isolated, synthetic-only harness. It checks every factor field against
`build_reference`, including 441 graph/seed/split cases, three graphs above
the public arena dispatch boundary and a rejected single-bit corruption.
It also compares complete factor archives against an explicitly selected
frozen arena source. It never serialises real-data graphs or changes sources.

For a permanent current-builder versus reference guard, run from the checkout:

```sh
python3 tests/analytic_precision/ac_factor_storage_contract.py \
  --output Verifications/ac_factor_storage_check \
  --frozen-source src/fe_absorption.cpp
```

Use a new output directory for each run. Passing a separately frozen
`fe_absorption.cpp` additionally checks historical arena equality; using the
current source checks reference equality without establishing historical
parity. `--compiler clang++ --sanitizers` enables ASan/UBSan and leak checks
where the local runtime supports them. The short `xhdfe_ac_arena_storage_contract`
CTest target independently checks allocation failure, relocation and slab
ownership. These are offline storage/factor gates, not estimator or runtime
certification. The main campaign manifest also pins `src/ac_arena_storage.hpp`.

## Shared Krylov scatter storage

`shared_scatter_storage_contract.py` extracts the exact common-store
preparation lambda from `fe_absorption.cpp`. It checks delayed allocation,
the unchanged group/chunk shape and zeros, owner-traversal exclusions,
idempotence and allocation failure before publishing the ready flag.
Negative controls reject premature publication and a missing consumer call.

```sh
python3 tests/analytic_precision/shared_scatter_storage_contract.py \
  --output Verifications/shared_scatter_storage_check
```

Use a new output directory. `--compiler clang++` checks the same body with
clang. This offline allocation guard uses a failure probe in place of Eigen;
actual estimator/oracle, cache, recovery, backend and memory receipts remain
separate requirements. It is not a numerical or performance certification.

## Portable lockstep memory routing and useful phase workers

The `xhdfe_lockstep_memory_plan_contract` CTest target checks a fixed-48,
data-only decision at simulated capacities 48/64/96. Scatter, contiguous
carries and row partials obey both `B=max(1 GiB,2R)` and `3R-Q`, where
`R=8*n*RHS` and `Q=56*levels*RHS`. The executor uses the same joint cap.
The decision prices its actual fixed-48 partition, not a hypothetical full
budget. Overflow or insufficient space has no executable partition.

At capacities up to 48, retained lockstep must keep the historical row
partition. The 266000-row/250-RHS counterexample that changed 48 to 47 chunks
is diverted. At larger capacities a memory-limited partition can differ;
trace fields declare that arithmetic class, and simulated capacities do not
certify numerical execution on 64/96-processor hardware. The general
deterministic header has no new fixed cap.

`lockstep_workspace_allocation_contract.py --output NEW_DIRECTORY` extracts
the actual core allocate/assign/row-partial expressions and records their
element requests. Bulk storage is intercepted: this tests sizes, not physical
RSS or the complete solver. Omitting carries in the model must be rejected.
Omitting the persistent row-partial term is also rejected. The separate
startup reduction partial, row vector, allocator overhead and other workspace
families are outside this selected model and still need physical receipts.
`lockstep_partition_boundary_oracle.py --module MODULE --output NEW_DIRECTORY`
checks the diverted boundary against independent sparse dummies and explicit
OLS, preserving the estimator's tolerances and separate offline reference.
It requires explicit dispatch (47 versus 48 historical chunks), batch-lanes
and useful-worker trace lines; absent evidence cannot pass by default.

`mlsmr_phase_trace_contract.py` compiles the exact opt-in phase trace class,
checks independent useful-worker slots and rejects team size reported as
active work. `XHDFE_MLSMR_TRACE=1` reports each phase's own units, team and
useful workers. The aggregate command diagnostic is not proof that every
solver phase used that team.
The observer test measures actual teams independently of requests; a runtime
limited to two workers remains testable. A strictly serial runtime leaves the
team-as-workers negative control explicitly `NOT_RUN`, not PASS.

These guards test structure and reporting. Physical memory, independent
projection/team-domain oracles, every changed core path, and the performance
matrix remain separate gates. A previously rejected routing trial is not
approved by these tests; current composite candidates require fresh receipts.

## Review closure contracts (10 October 2026)

`intercept_psd_contract.py --output NEW_DIRECTORY --captured SYNTHETIC_DTA`
checks the actual intercept covariance repair against independent spectral,
range and Schur-complement conditions. The optional input is the 40-seed
synthetic covariance capture, never confidential records. Slope blocks must
remain bit-identical; positive, singular, zero and rescaled cases are included.

`group_column_moment_contract.cpp` distinguishes a passing aggregate RMS from
a failing individual FE moment using analytic residuals. Link it against the
explicit CPU core under test. Both standard/individual dimension orders and
the unchanged Fast mode are covered.

`n1_continuation_context.cpp` checks thread isolation and restoration of the
private CUDA continuation predicate, including nested scopes and exceptions.
Compile with the repository `src` and Eigen include directories and `-pthread`.
Checks remain active with `-DNDEBUG`. The same executable must return 1 with
`--inject-context-error`, which corrupts the live scoped pointer; both builds
must reject that negative control. It is a lifetime contract; numerical CUDA
continuation is covered separately by the independent PF3FE tests and their
11 October receipts, not by this context test.

`n1_context_build_contract.py --include SOURCE_DIR --eigen EIGEN_DIR
--output NEW_DIRECTORY` compiles both configurations and exercises the positive
and injected-failure cases in each executable.

`stata_walsh_residual_contract.py --package PACKAGE --fixture SYNTHETIC_DTA
--backend cpu|cuda --output NEW_DIRECTORY` reproduces the 3,145,728-row Walsh
fixture in fresh Stata sessions (threads 1/2/8/16/24/48 by default). The package
must contain `stata/xhdfe.ado` and its matching plugin. `--version` selects the
expected package stamp when checking the frozen 2.26.2 baseline. CUDA needs a
GPU-visible context. `walsh_fixture_oracle.py` verifies the fixture's dyadic
construction, connected FE rank, exact coefficients and covariance independently.
The contract checks b, full V, FE moments, normal equations, sample/df and real
backend use. The older absolute pointwise diagnostic at 1e-7 is retained as a
separate value/pass flag: it fails for both the final 2.29 and frozen 2.26.2
CUDA fits, while their published contracts pass. It is not a universal solver
target. Injected FE and regressor residual errors must fail the moment checks.

`schur_gauge_contract.py --output NEW_JSON` checks the independent offline
three-FE oracle against dense synthetic OLS, including a structural null
direction missed by a two-anchor implementation and power-of-two rescaling.

`worker_allocation_contract.py --module MODULE --output NEW_DIRECTORY` is a
Linux/glibc synthetic fault-injection test. It builds a process-local malloc
shim, disables core dumps in its own children, verifies failure sites,
requires a propagated MemoryError and a non-converged state, and repeats the
fit in the same object. Coefficients, covariance, residuals and iterations must
match an uninjected run. It does not modify system settings or inject into
other processes. This test is separate from native Windows/macOS validation.

`strict_cuda_continuation_contract.py --module MODULE --fixture LAPLACIAN_DIR
--output NEW_JSON` checks the synthetic `laplacian_sum` strict oracle with
real CUDA use and unchanged tolerance, plus iteration budgets 1, 1128 and 1500.
All limited cases must refuse with empty estimates; the latter two also check
clearing a previous fit and subsequent recovery of the same object. The 1500
case verifies that refusal diagnostics describe the original model, rather
than an unaccepted auxiliary correction. Expected
budget refusals are lifecycle controls, separate from the positive numerical
strict result. The complete campaign supplies default/Fast controls.

`equivalent_cluster_partition_contract.py --module MODULE --backend cpu|cuda
--output NEW_JSON` tests the exact IV partition identity with independent
MP90/MP120 covariance. Distinct partitions with equal marginal counts guard
against an overbroad shortcut. Two unweighted distinct-partition cases per
backend have indefinite raw CGM covariance and instead check the authorised
PSD policy; they are not positive-covariance oracle passes.

`stata_iv_intercept_covariance_contract.py --package STATA_DIRECTORY
--stata STATA_EXECUTABLE --expected-version VERSION --output NEW_DIRECTORY`
checks twelve synthetic IV fits: unweighted/analytic/frequency weights,
Fast/Comparable and CPU/CUDA. It pins the loaded ado/plugin, requires real GPU
use and compares the full covariance, including the constant, against the
existing MP90/MP120 oracle. It does not install or modify a package.


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

## Ordinary CUDA weak-graph projection

`cuda_ordinary_weak_graph_contract.py --module MODULE --backend cpu|cuda
--output OUTPUT.json` checks the deterministic 200,000-row R audit design
against explicit sparse-dummy OLS with long-double refinement and an
independently bounded projection defect. It checks the existing coefficient
1e-9 and full-covariance 1e-8 contracts, exact sample/DoF, actual GPU use,
rescaled inputs, reversed FE order, explicit thread requests 1/2/8/16/24/48,
the cold restart after an explicit CG breakdown, the existing exact tree
shortcut, and separate iteration-budget refusals without stale estimates.
Frozen 2.29.1 CUDA fails the coefficient test; parity with it is not an
oracle. The unified core runs this regression alongside the analytic battery.

## Fast CG stopping policy

`fast_cg_mode_contract.py --module MODULE --backend cpu|cuda --output RESULT`
checks dyadic Walsh regressions on near-nested three-FE graphs. Supplying
`--github-data PATH` additionally checks the core24 `github` return regression
without copying individual data into the repository. The aggregate-only
reference pins all six numerical input columns. It uses independent sparse
dummy FWL by SciPy LSQR/LSMR, MP80 reduced OLS and a long-double cluster
sandwich. The two references agree to 0 in reported b and 3.93e-12 in scaled V.
`generate_fast_cg_reference.py --data PATH --output NEW_JSON` regenerates this
aggregate reference without exporting observations or overwriting old results.
The optional real-data control certifies return, sample, backend and b; V
deviations are recorded separately and do not redefine Fast's approximation.
Run `cg_final_step_contract.py` alongside it to retain the MP120 covariance
protection for Comparable, with the distinct existing Fast/Strict limits.
