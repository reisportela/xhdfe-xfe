# `xhdfegelbach` post-audit remediation report

**Date:** 23 July 2026 (Europe/Lisbon)
**Repository:** `/home/mangelo/Documents/GitHub/xhdfe`
**Branch / starting HEAD:** `main` / `d82c93243babaf050fdd43e9139b2ac880c53d8c`
**Audit remediated:** `Verifications/claude_xhdfegelbach_empirical_utility_audit_20260722/`
**Candidate versions:** shared package `2.20.0.20260723`; `xhdfegelbach`
`1.4.0`; Stata release date `23jul2026`

## 1. Executive status

The econometric and frontend remediation is implemented and the rebuilt local
Release artifacts pass the completed correctness, parity, guard, documentation,
artifact-hygiene, CPU/CUDA truthfulness, and standard-path non-regression gates.
The Gelbach estimand, decomposition identity, b1x2 stacked-GMM conventions,
classification boundary, solver tolerances, convergence criteria, and standard
non-Gelbach `xhdfe` path were not changed.

This document is not yet a final online-release certificate. The interleaved
30% catastrophic-performance-regression triage is complete and passes for CPU
and CUDA. Private/public byte propagation, commits, tags, release assets,
public CI, and live net-install verification remain the sole operational gate.

| Gate | Current status |
|---|---|
| Audit findings required by Gate 1 | **PASS locally**, with the F-03 scope qualification below |
| F-04 joint `share=base` inference | **PASS locally**, including corrected independent algebra and Monte Carlo |
| Standard path A/B | **PASS: 0/240 non-bit-identical output blocks** |
| CPU Release artifacts | **PASS** |
| CUDA `sm_90` Release artifacts | **PASS** |
| Python/R/Stata CPU parity | **PASS** |
| Python/R/Stata CUDA-use and hidden-device truthfulness | **PASS** |
| Full Stata suite | **PASS: 28 blocks** |
| Full R CPU/CUDA suites | **PASS: 0 failures** |
| Performance triage | **PASS at the pre-specified 30% catastrophic-regression margin** |
| Private/public propagation and online release | **PENDING** |

The appropriate present verdict is therefore:

> **TECHNICAL REMEDIATION AND 30% PERFORMANCE TRIAGE GREEN; ONLINE RELEASE
> CERTIFICATION PENDING PUBLICATION GATES.**

## 2. Scope, evidence discipline, and invariants

The audit trail under
`Verifications/claude_xhdfegelbach_empirical_utility_audit_20260722/` was read
as evidence and was not modified. Reproducers that needed adaptation were
copied under
`Remediation/xhdfegelbach_postaudit_20260723/repros/`. Nothing under
`literature/` was modified.

Every finding was treated as a hypothesis. The capability gap in F-04 was
reproduced, but one formula in the audit's newly proposed cross-covariance
oracle was not: it omitted a b1x2 finite-sample multiplier. That disagreement
was resolved algebraically and empirically without altering the audit trail;
details appear in §4.4 and §8.

The following invariants were held fixed:

- standard coefficient-movement and absorbed-target estimands;
- `delta`, `total`, and the exact Gelbach identity;
- b1x2 `gamma0` and `cov0` meanings;
- FE-collinearity classification at
  `||M_D x||² / ||x||² <= 1e-9`;
- absorption and convergence tolerances;
- rank and misuse guards;
- weighting and effective-N conventions;
- the conditional/`gamma0` treatment and labels for recovered-FE parcels;
- default CPU behavior and default output contracts;
- the standard non-Gelbach `xhdfe` estimation path.

The strongest direct non-regression evidence is the pre-change/current Release
worker comparison saved as:

- `Remediation/xhdfegelbach_postaudit_20260723/baseline/standard_path.pkl`;
- `Remediation/xhdfegelbach_postaudit_20260723/current_standard_path.pkl`.

It covers 24 specifications and 10 output blocks per specification:
`b_base`, `b_full`, `delta`, `cov`, `total`, `total_cov`, `identity_gap`,
`n_obs`, `df_full`, and `converged`. The result is:

```text
non-bit-identical output blocks: 0 / 240
```

The new FE-collinearity capture is explicitly default-off in the general
regressor (`include/hdfe/hdfe_regressor.hpp:63`) and enabled only by the
Gelbach full fit (`src/akm_kss.cpp:5092`). The existing standard `xhdfe` path
therefore performs no new diagnostic arithmetic.

## 3. Finding-level disposition

| Finding | Reproduced? | Disposition in this tranche |
|---|---|---|
| F-01 near-FE boundary | Yes | Fixed and recertified |
| F-02 Python/R backend diagnostics | Yes | Fixed and recertified |
| F-03 discarded metadata | Yes | Required subset (`G`, `df_base`, `gamma`) fixed; sample hash and generic per-fit iteration certificates remain deferred |
| F-04 base-share inference | Capability gap yes; audit oracle formula partly no | Implemented with corrected b1x2 algebra and recertified |
| F-05 multiway clustering | Yes | Explicitly deferred by work order |
| F-06 few-cluster silence | Yes | Fixed with `G < 30` warning and metadata |
| F-07 saturated full model | Yes | Fixed with catchable core error |
| F-08 warning retention | Yes | Fixed in all frontends |
| F-09 Stata tidy/export | Yes | Explicitly deferred |
| F-10 p-values/joint tests | Yes | Explicitly deferred |
| F-11 artifact provenance | Yes | Local build/provenance fixed; online release manifest pending |
| F-12 mobility-component flag | Yes | Explicitly deferred |
| F-13 coarser nested cluster certification | Yes | Explicitly deferred |
| F-14 Python absorbed-target names | Yes | Fixed and recertified |
| F-15 documentation/UX batch | Yes | Work-order batch fixed |
| F-16 “six real examples” wording | Yes | Shipped/active claims corrected |
| F-17 positives | Yes | Preserved; no redesign |
| OPP-1 broader estimators | Not defects | Explicitly not implemented |

## 4. Detailed remediation by finding

### 4.1 F-01 — near-boundary undeclared focal

**Reproduction.** The copied audit boundary attack reproduced an undeclared X1
column just above the `1e-9` omission boundary being accepted as identified:

```text
ratio = 4.0e-9
b_full = -4.8216e+02
total = +4.8244e+02
reported total SE = 8.754e-02
warnings = []
```

The copied Monte Carlo confirms that this was not merely a large point
estimate. At target ratios `4e-9`, `1e-6`, and `9e-5`, empirical movement SD
was respectively 3438, 217, and 22.9 times the mean reported SE.

**Change.**

- `include/hdfe/hdfe_regressor.hpp:63` adds an internal, default-off request to
  retain the FE-collinearity ratio already computed by the omission
  classifier.
- `src/hdfe_regressor_v11.cpp:7092-7107`,
  `src/hdfe_regressor_v11.cpp:7884-7889`,
  `src/hdfe_regressor_v11.cpp:9333-9347`, and
  `src/hdfe_regressor_v11.cpp:9642-9647` capture the ratio only when requested.
- `include/hdfe/akm_kss.hpp:244-245,262-264` defines the additive result
  fields and public thresholds.
- `src/akm_kss.cpp:5050-5057,5092,5105-5129` returns the per-X1 ratio, marks
  `(1e-9,1e-4]`, appends the required warning, and honors
  `XHDFE_GELBACH_NEAR_COLLINEAR_WARN=0` as an output-only kill switch.
- `python/py_hdfe_v11.cpp:1554-1555,1572-1576`,
  `xhdfe/gelbach.py:488-499`,
  `r/xhdfe/src/rcpp_xhdfe.cpp:1137-1144,1163-1166`,
  `r/xhdfe/R/gelbach.R:429-430`, and
  `stata/xhdfegelbach.ado:636-640,740-741` expose the same metadata.

No estimate, classification, covariance, or convergence decision depends on
the warning mask.

**Recertification.**

- `VALIDATE_GELBACH.py`: a `4e-9` straddle case warns and marks; a `2e-2`
  focal is silent; the absorbed-side ratio remains classified exactly as
  before.
- `VALIDATE_GELBACH_ADVERSARIAL.py`: default warning contract and kill-switch
  output-only invariance pass.
- Full Stata suite: the matrix fields and warning retained in `r(notes)` pass.
- Python/R/Stata frontend parity for the ratio and mask passes.
- `Remediation/.../repros/f01_near_fe_band_mc.json`: every mask and warning
  matches the contract in all 160 runs.

**Residual evidence, not hidden.** In the intentionally hostile MC, the
`1e-2` cell is outside the documented warning band but still has an empirical
SD / mean-SE ratio of 2.42. The implemented band closes the audit's
near-boundary silent trap and follows the work order's proposed `1e-4` upper
edge; it is not a claim that every design above `1e-4` has calibrated
conditional component inference. Researchers must still inspect the returned
ratio, the conditioning notes, and the substantive design.

### 4.2 F-02 — Python/R GPU and thread diagnostics

**Reproduction.** The audit's public Python result lacked all `gpu_*` keys and
had no GPU argument even when external monitoring showed CUDA activity. R had
the same public-contract gap. The audit attack recorded
`result keys contain gpu fields=False`.

**Change.**

- `include/hdfe/akm_kss.hpp:227,268-276` adds the opt-in request and full
  diagnostic result contract.
- `src/akm_kss.cpp:5058,5095-5100,5360-5380` scopes the CUDA request to the
  full-model absorption phase and reports requested/attempted/used/backend/
  status/convergence/iterations truthfully.
- When the device is not visible, `src/akm_kss.cpp:5366-5375` returns a CPU
  result labelled `status=unavailable`, `status_code=2`, never a false CUDA
  success.
- `python/py_hdfe_v11.cpp:1472-1477,1531-1532,1580-1590,1603` and
  `xhdfe/gelbach.py:174-177,203-207,390-395,513-525` add `gpu=False` and
  marshal the public fields.
- `r/xhdfe/R/RcppExports.R:28-29`,
  `r/xhdfe/src/RcppExports.cpp:109-129`,
  `r/xhdfe/src/rcpp_xhdfe.cpp:1059-1113,1171-1179`, and
  `r/xhdfe/R/gelbach.R:208-212,361-362,468,532-536` do the same for R.
- Stata already had the option; `stata/src/xhdfe_plugin.cpp:1745-1755` and
  `stata/xhdfegelbach.ado:447-479,648-653,716-717` now share the complete
  backend-classified contract.

**Recertification.**

- The CUDA frontend validator passed through Python, R, and Stata with
  `gpu_requested=1`, `gpu_used=1`, `gpu_backend=cuda`, and
  `gpu_status=used`.
- With `CUDA_VISIBLE_DEVICES=` it passed through all three frontends with
  `gpu_requested=1`, `gpu_used=0`, `gpu_backend=cpu`,
  `gpu_status=unavailable`, and code 2.
- The public R H100 smoke at 100,000 observations reports:

  ```text
  requested=TRUE used=TRUE backend=cuda status=used
  attempted=TRUE converged=TRUE iterations=3 threads=8
  ```

- Its CPU/CUDA maximum differences are `9.99e-16` for `b_full`,
  `3.33e-16` for `delta`, `1.02e-20` for `cov`, and `1.19e-20` for
  `total_cov`.

### 4.3 F-03 and F-06 — metadata and few-cluster guardrail

**Reproduction.** The pre-change public contract did not expose cluster count,
base-model residual degrees of freedom, or the full-model coefficients of
observed added blocks. A three-cluster design ran silently. The audit also
identified sample-signature and generic per-fit iteration metadata as absent.

**Change.**

- `include/hdfe/akm_kss.hpp:246,260-264` adds `gamma`, `df_base`,
  `n_clusters`, and the threshold.
- `src/akm_kss.cpp:5404-5413` returns the observed-block full-model
  coefficients, padded by block width.
- `src/akm_kss.cpp:5547-5573` independently counts retained-sample clusters
  and appends a warning when `G < 30`.
- `src/akm_kss.cpp:5651-5659` returns and guards `df_base`.
- `python/py_hdfe_v11.cpp:1556,1570-1576`,
  `xhdfe/gelbach.py:407-411,459,486-499`,
  `r/xhdfe/src/rcpp_xhdfe.cpp:1145,1160-1166`,
  `r/xhdfe/R/gelbach.R:378-389,524-529`, and
  `stata/xhdfegelbach.ado:634-640,736` expose the fields consistently.

**Recertification.**

- Independent full-model fits match returned `gamma`; the largest discrepancy
  in the final external grid is `2.78e-15`.
- Returned `G` equals an independent unique-count exactly.
- `G=3` warns and records the warning; `G=50` is silent.
- Frontend parity covers the new fields in the standard and absorbed-target
  designs.

**Scope qualification.** F-03's complete audit title also mentions a sample
mask/hash and generic per-fit iteration certificates. The post-audit work order
required `n_clusters`, `df_base`, and `gamma`; those are closed. It did not
authorize a new sample-hash or generic multi-phase iteration interface, and
those two broader metadata opportunities remain deferred. The GPU absorption
iteration count is surfaced, but it is not represented as a generic
all-phase certificate.

The warning is guidance, not a replacement estimator: it does not change
critical values, the VCE, or coefficients. Wild-cluster and other few-cluster
procedures remain separate, deferred estimators.

### 4.4 F-04 — `Cov(delta,b_base)` and joint `share=base` inference

#### Reproduction and audit-oracle reconciliation

The capability gap was reproduced: pre-change `share=base` point estimates
were available, but their SEs were deliberately missing, while
`base_fixed` held the realized denominator fixed.

The audit's broad inference diagnosis was correct, but the robust/cluster
cross-covariance formula in its copied oracle was not. It omitted the b1x2
`q_big` multiplier from the cross block and initially retained a full-model
coefficient term under `gamma0`. The correct b1x2-convention block is:

```text
Cov(delta_g, b_base) =
    q_big [ P M_(a_g,b) P
            + Gamma_g S_(J_g,:) M_(f,b) P ],
```

where the second term is absent for `gamma0` and for recovered-FE blocks.
`q_big` is `N/(N-1)` for robust inference and `G/(G-1)` for clustered
inference. `q_vu` belongs only to the separately added full/full block.
`cov0` removes robust full/aux cross terms from `Var(delta)`; it does not
remove `Cov(delta,b_base)`.

The audit directory was left untouched. A copy under
`Remediation/.../repros/ws4/oracles/gelbach_oracle.py` records the correction.
The file
`Remediation/.../repros/ws4/oracles/ws4_oracle_comparison_raw.json` predates
the final correction and is not used as certification evidence. The live
independent grid in `VALIDATE_GELBACH.py` is the authoritative final oracle
test.

#### Change

- `include/hdfe/akm_kss.hpp:249-251` adds `base_cov`,
  `cov_delta_bbase`, and `cov_total_bbase`.
- `src/akm_kss.cpp:5651-5677` builds the requested-VCE covariance of the base
  regression with its own finite-sample correction.
- `src/akm_kss.cpp:5722-5757` builds the homoskedastic component/base
  cross-block.
- `src/akm_kss.cpp:5855-5864,5867-5903,5908-5938` builds the robust or
  cluster stacked-score cross-block with weights and the correct b1x2
  multipliers.
- `src/akm_kss.cpp:5942-5966` aggregates
  `Cov(total,b_base)`. For every declared absorbed target it enforces the
  estimator identity `Cov(total_j,b_base)=Var(b_base_j)` on the authoritative
  total row.
- `python/py_hdfe_v11.cpp:1559-1561`,
  `r/xhdfe/src/rcpp_xhdfe.cpp:1148-1150`, and
  `stata/src/xhdfe_plugin.cpp:1718-1720` expose the matrices.
- Python's full delta method is at `xhdfe/gelbach.py:596-660` and the total-row
  analogue at `xhdfe/gelbach.py:760-805`.
- R's full delta method is at `r/xhdfe/R/gelbach.R:594-644` and the total-row
  analogue at `r/xhdfe/R/gelbach.R:768-792`.
- Stata's formula is at `stata/xhdfegelbach.ado:347-415`.
- `base_fixed` remains a separate, numerically unchanged
  `fixed_base_denominator_scaling` convention.

For a component share `s_g=delta_g/b_base`, all three frontends use:

```text
Var(s_g) =
    Var(delta_g) / b_base^2
  + delta_g^2 Var(b_base) / b_base^4
  - 2 delta_g Cov(delta_g,b_base) / b_base^3.
```

The label is `joint_base_covariance_delta_method`. Point shares remain
bit-identical. For a declared absorbed target, `total=b_base` is the same
estimator; its total base-share point is preserved and its SE is exactly zero.
The optional full-model-residual share row remains explicitly unavailable.

#### Recertification

`VALIDATE_GELBACH.py` exercises:

- three VCEs (`unadjusted`, `robust`, `cluster`);
- three weight contracts (unweighted, analytic weights, frequency weights);
- no-FE and one-FE designs;
- default, `gamma0`, and applicable `cov0` semantics;
- base covariance, every component/base block, total/base covariance,
  `gamma`, `df_base`, and cluster count.

All comparisons pass at tolerance `1e-14`. The observed final maxima are:

```text
max |Cov(delta,b_base) - oracle| = 6.94e-18
max |gamma - independent full fit| = 2.78e-15
df_base and cluster-count discrepancies = 0
```

The absorbed-target identity passes exactly:

```text
Cov(total_target,b_base) - Var(b_base) = 0
SE(total_target) - SE(b_base) = 0
```

The final 5,000-replication Monte Carlo is:

| Design | Share A coverage | Share B coverage | Mean-SE / empirical-SD |
|---|---:|---:|---:|
| iid, unadjusted, N=2400 | 0.9482 | 0.9428 | 1.006 / 0.983 |
| iid, robust, N=2400 | 0.9556 | 0.9486 | 1.003 / 1.002 |
| cluster, G=120, 20 rows/G | 0.9458 | 0.9468 | 0.995 / 0.994 |
| absorbed target, G=150 | 0.9540 | n/a | 1.003 |

These experiments meet the specified `>=0.94` gate for iid and an
adequately-clustered DGP. The `G=40` sensitivity is reported rather than
discarded: coverage is 0.934/0.939 under an independent seed and
0.9372/0.9330 under the audit seed. This is modest finite-cluster
undercoverage of normal-reference CR1 intervals, not a covariance-algebra
failure. No unauthorized t-reference, bootstrap, or wild-cluster default was
introduced.

### 4.5 F-07 — saturated full model

**Reproduction.** The copied pre-change reproducer returned
`df_full=0`, `converged=True`, empty notes, and `inf`/`NaN` SEs under
unadjusted/robust VCEs.

**Change.** `src/akm_kss.cpp:5417-5422` now rejects nonpositive or nonfinite
full-model residual degrees of freedom with:

```text
gelbach: insufficient full-model residual degrees of freedom
(df_full must be positive)
```

This is a real runtime guard, not an assertion, so it remains active in
Release/NDEBUG builds.

**Recertification.**

- Python friendly wrapper: catchable `RuntimeError`.
- Direct Release core: catchable `RuntimeError`.
- R testthat: catchable R error with the residual-df message.
- Stata suite: nonzero catchable return code, followed by successful later
  tests; process remains alive.
- `VALIDATE_GELBACH_ADVERSARIAL.py`: saturated guard and post-error healthy
  reuse pass.

### 4.6 F-08 — undefined-share warning retention

**Reproduction.** Before remediation, Stata printed the `sharetol()` warning
but omitted it from `r(notes)`; Python and R returned NaN/NA share fields
silently.

**Change.**

- `stata/xhdfegelbach.ado:600-604,618-624,729` appends the warning to the
  backend note string, prints it, and returns it in `r(notes)`.
- `xhdfe/gelbach.py:703-718` emits one `RuntimeWarning` per `tidy()` call for
  affected coefficients.
- `r/xhdfe/R/gelbach.R:717-728` emits one R warning per tidy call.

**Recertification.**

- Python `warnings.catch_warnings`: exactly one matching `RuntimeWarning`.
- R `withCallingHandlers`: visible matching warning.
- Stata: ``strpos("`r(notes)'","share denominator") > 0``.
- Undefined point/SE/CI fields remain missing; no threshold was loosened.

### 4.7 F-11 — artifact provenance and build hygiene

**Reproduction.** The audit correctly found a Debug `build/CMakeCache.txt`
residue and user-loadable `.so` hashes that were absent from the previous
release evidence.

**Change.**

- `CMakeLists.txt:8-28` makes Release the default and fails closed on a
  non-Release production configuration unless the explicit diagnostic-only
  opt-out is used.
- `CMakeLists.txt:97-106` pins Eigen to the in-repository 3.4.0 tree and
  refuses a missing vendored copy.
- `build/` and `build_cuda/` were reconfigured as true Release:
  `-O3 -DNDEBUG -march=native`; CUDA is off/on respectively and
  `build_cuda` targets architecture 90.
- The CPU module was copied byte-for-byte to the in-tree `xhdfe/*.so`, which
  is the artifact editable imports resolve first.
- R CPU and CUDA packages and both Stata plugins were rebuilt after the final
  source state.

The complete active-artifact manifest appears in §7.

**Recertification.**

- every active artifact has zero dynamic references to `__assert_fail`;
- every active artifact links `libgomp.so.1`;
- every CUDA artifact contains only `sm_90` cubins;
- the in-tree Python module is byte-identical to the CPU `build/` module;
- current CMake caches identify Release and the in-repository Eigen path.

The publication half of F-11 remains **PENDING** until these hashes are
included in the final private/public release evidence.

### 4.8 F-14 — Python name-based `absorbed_targets`

**Reproduction.** Python accepted zero-based indices or a mask, unlike Stata
and R, even though Python already had X1 names for `focal`.

**Change.** `xhdfe/gelbach.py:291-336` accepts names, zero-based indices, or a
Boolean mask; names are resolved against `x1_names`, mixed selector types,
duplicates, unknown names, and out-of-range indices fail closed.

**Recertification.** The absorbed frontend fixture calls Python with a name and
matches R's name-based and Stata's varname-based request while retaining the
unified public result:

```text
absorbed_targets = [0]
absorbed_target_names = ["focal"]
```

### 4.9 F-15 and F-16 — documentation and claim hygiene

**Reproduction.** The audit's listed documentation/UX gaps and the inaccurate
“six real examples” wording were found.

**Change.**

- Python: `xhdfe/help/gelbach.md:119-147,178-221,274-309,350-389`.
- Stata: `stata/xhdfegelbach.sthlp:186-210,275-299,399-420,437-449`.
- R: `r/xhdfe/man/xhdfe_gelbach.Rd:45-67,102-110,162-164,266-291`;
  tidy share contract in
  `r/xhdfe/man/xhdfe_gelbach_tidy.Rd:35-65`.
- Offline R dependency path: `r/README.md:45-59`.
- Frontend validator description:
  `VALIDATE_GELBACH_FRONTENDS.py:1-10`.
- Release summary:
  `RELEASE_NOTES_2.20.0.20260723.md`.

The three help surfaces now state:

- binary outcomes are linear probability models; a logit-scale decomposition
  is a separate estimator;
- factor/formula notation is not parsed; generate a full-rank numeric
  indicator/interactions matrix, remove the intercept and one reference
  category, and pass explicit named blocks;
- `gpu` accelerates only full-model FE absorption;
- `threads()` / `num_threads` is a per-phase cap and `threads_used` is the
  largest effective team;
- R's cluster and numeric-input errors use user-facing language;
- autonomous release media include the pinned official Rcpp 1.1.2 source
  archive and a network-disabled local-library installation route;
- examples are accurately described as the standard and absorbed-target
  examples executed in each of the three frontends.

`VALIDATE_GELBACH_HELP.py` passes all 57 checks. The historical wording inside
old remediation/audit evidence was not rewritten; those directories are
records, not shipped claims.

## 5. Findings deliberately not fixed

The work order explicitly deferred the following. They were not implemented,
relabelled, or smuggled in as switches:

- F-05 multiway/two-way CGM clustering;
- F-09 Stata tidy/contrast/export subcommand;
- F-10 native p-values, z statistics, and joint block tests;
- F-12 mobility-component count/flag for normalization-sensitive per-FE
  parcels;
- F-13 certification of clustering coarser than the absorbing FE;
- common HDFE in base and full models;
- nonconditional recovered-FE component inference;
- IV-Gelbach;
- dynamic/SPJ-corrected decomposition;
- pweights/IPW;
- wild-cluster bootstrap;
- KHB, MM-QR, RIF/distributional, Oaxaca, or mediation estimators;
- regex/factor block-selection helpers;
- curve/batch and nested-match-split helpers.

These are not silently approximated. Existing refusals and conditional labels
remain in force.

## 6. Completed validation matrix

All commands below exercised rebuilt Release artifacts, not the pre-build
audit binaries.

| Validation | Result |
|---|---|
| `python3 VALIDATE_GELBACH.py --module-dir build --external-absorbed-oracle require --stata stata-mp` | **PASS: all checks** |
| Same validator with local official `b1x2.ado` | **PASS**, maximum shared-domain difference about `4.44e-16` |
| External Stata-LSDV absorbed-target oracle | **PASS**; point max `2.83e-15`, target total-SE max `2.78e-17` |
| `python3 VALIDATE_GELBACH_FRONTENDS.py ... --module-dir build` | **PASS: all CPU frontend and example checks** |
| Frontend validator, `--gpu --require-gpu-used` | **PASS: CUDA used in Python/R/Stata** |
| Frontend validator with hidden device | **PASS: truthful CPU/unavailable in Python/R/Stata** |
| `python3 VALIDATE_GELBACH_ADVERSARIAL.py --module-dir build` | **PASS: all checks** |
| `python3 VALIDATE_GELBACH_HELP.py` | **PASS: 57 checks** |
| Full Stata certification suite | **PASS: 28 blocks** |
| Full R CPU suite | **PASS: 0 failures; 1 expected CUDA skip; 3 pre-existing AKM warnings** |
| Full R CUDA suite | **PASS: 0 failures, no GPU skip** |
| `python3 VALIDATE_AKM_KSS.py --module-dir build` | **PASS: all checks; expected CPU-build GPU skip** |
| `bash tools/check_cpp_core_alignment.sh` | **PASS** |
| `git diff --check` | **PASS** at the completed-code checkpoint |

The frontend parity validator covers both synthetic designs in each of Python,
R, and Stata. Representative maximum cross-frontend discrepancies are:

| Object | Standard design | Absorbed-target design |
|---|---:|---:|
| `delta` | Stata `1.67e-16`; R `4.16e-16` | Stata `1.80e-16`; R `4.44e-16` |
| `cov` | Stata `1.13e-17`; R `9.11e-17` | Stata `3.47e-17`; R `9.75e-17` |
| `cov_delta_bbase` | Stata `9.97e-18`; R `8.93e-17` | Stata `5.55e-17`; R `9.71e-17` |
| `cov_total_bbase` | Stata `1.47e-17`; R `9.56e-17` | Stata `4.51e-17`; R `8.53e-17` |
| `gamma` | Stata `1.11e-16`; R `4.16e-16` | Stata `0`; R `3.33e-16` |

The full Stata evidence is preserved at
`Remediation/xhdfegelbach_postaudit_20260723/evidence/stata_full_suite.log`;
its final marker is:

```text
XHDFE STATA CERTIFICATION TESTS COMPLETED SUCCESSFULLY
tests_run = 28
```

R build, test, and H100 evidence is summarized in
`Remediation/xhdfegelbach_postaudit_20260723/R_INTEGRATION_EVIDENCE.md`.

## 7. Release build and artifact manifest

### 7.1 Build configuration

| Build tree | Type | C++ Release flags | CUDA | Eigen |
|---|---|---|---|---|
| `build/` | Release | `-O3 -DNDEBUG -march=native` | OFF | `r/xhdfe/src/eigen` |
| `build_cuda/` | Release | `-O3 -DNDEBUG -march=native` | ON, arch 90 | `r/xhdfe/src/eigen` |

The CMake cache facts are:

```text
build/CMakeCache.txt:
  CMAKE_BUILD_TYPE=Release
  CMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=native
  XHDFE_ENABLE_CUDA=OFF

build_cuda/CMakeCache.txt:
  CMAKE_BUILD_TYPE=Release
  CMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=native
  CMAKE_CUDA_ARCHITECTURES=90
  XHDFE_ENABLE_CUDA=ON
```

### 7.2 Active/certified user-loadable artifacts

| Artifact | SHA256 | `U __assert_fail` | OpenMP | CUDA image |
|---|---|---:|---|---|
| `build/py_hdfe_v11.cpython-312-x86_64-linux-gnu.so` | `126e65b187fa017766ecdb43e70e34ad75ce8545722555bad8f525d115de31c8` | 0 | `libgomp.so.1` | CPU |
| `build_cuda/py_hdfe_v11.cpython-312-x86_64-linux-gnu.so` | `4e642d5eae1f2e43a5adcc898533019f9be398196c683e0dcef430b4d6fa12a1` | 0 | `libgomp.so.1` | 2 × `sm_90` |
| `xhdfe/py_hdfe_v11.cpython-312-x86_64-linux-gnu.so` | `126e65b187fa017766ecdb43e70e34ad75ce8545722555bad8f525d115de31c8` | 0 | `libgomp.so.1` | CPU |
| `r/Rlib/xhdfe/libs/xhdfe.so` | `b934a650013677261cbbcc7ac10303a4def184452e2bd6fee43113bed1563604` | 0 | `libgomp.so.1` | CPU |
| `Remediation/xhdfegelbach_postaudit_20260723/artifacts/r_cuda/xhdfe.so` | `6448036e9a14c0c7d59f08fa41422330faa6da56aa11e1d6848994a047584446` | 0 | `libgomp.so.1` | 2 × `sm_90` |
| `stata/xhdfe.plugin` | `a76ea1f92bd796048c6221576c1021b20c6273227d6b3a5910bdad4c4b07e7f3` | 0 | `libgomp.so.1` | 3 × `sm_90` |
| `stata/xfe.plugin` | `2fe4d0d55c4fed994b5dc9a7bd9c8283bdfb7067d8d945dd9060a4bc9cddeaea` | 0 | `libgomp.so.1` | 2 × `sm_90` |

The archived CPU R artifact at
`Remediation/.../artifacts/r_cpu/xhdfe.so` has the same
`b934a650...` hash as the active `r/Rlib` artifact.

`cuobjdump -lelf` lists only:

```text
build_cuda Python: .1.sm_90.cubin, .2.sm_90.cubin
R CUDA:            .1.sm_90.cubin, .2.sm_90.cubin
xhdfe.plugin:      .1.sm_90.cubin, .2.sm_90.cubin, .3.sm_90.cubin
xfe.plugin:        .1.sm_90.cubin, .2.sm_90.cubin
```

### 7.3 Non-active stale snapshots

Several historical artifacts remain under dated release/build/test
directories by repository policy. They are not on the certified import/library
path and must not enter the new release. In particular,
`build/r-lib/xhdfe` is a stale `2.18.1.20260711` test installation with hash
`a3d69d50353823c63362e097d17b0a5077fc5ab5d45730aaa971ea4c9200ef5a`;
the certified R load path is `r/Rlib/xhdfe`, not `build/r-lib/xhdfe`.
Likewise, `build/release_2_19_0_20260720_assets/` and other dated directories
are frozen prior-release evidence, not 2.20.0 artifacts.

The final release-staging script must select the explicit manifest above and
must fail if a stale build/test installation is collected by a broad glob.

### 7.4 Rcpp provenance and offline closure

The initially certified active R binaries resolved Rcpp `1.1.1-1.1` from the
repo-local validation library `r/Rlib/Rcpp`; their exact dependency record is
preserved in
`Remediation/xhdfegelbach_postaudit_20260723/RCPP_PROVENANCE.md`.

Release media now additionally contain the unmodified official CRAN source
archive `third_party/Rcpp_1.1.2.tar.gz`, SHA-256
`2746cf2fb188e5f0a84dbf5c8f68915b54564ed33e5754572f174e7b32e7f4f3`,
with URL/license metadata in `third_party/RCPP_SOURCE_PROVENANCE.md`. A fresh
network-disabled local-library install from that archive was used to compile
and test both CPU and CUDA xhdfe R builds:

| Offline rebuild | SHA-256 | `U __assert_fail` | OpenMP | CUDA |
|---|---|---:|---|---|
| CPU R `.so` | `f36f0a8125288b42aec9cf9fb6eca5cd4415647920a6ee5e812a733e3a43c1cd` | 0 | `libgomp.so.1` | none |
| CUDA R `.so` | `bc7830fda343e1dcf726addfb6009084ba1770fdc20370626a52ceb53d728675` | 0 | `libgomp.so.1` | exactly 2 × `sm_90` |

Both Gelbach test suites passed. The H100 smoke reported
`gpu_requested=TRUE`, `gpu_used=TRUE`, backend `cuda`, status `used`; CPU/GPU
differences were at most `6.66e-16` for coefficients and `6.78e-21` for the
covariance blocks. With the device hidden it reported backend `cpu`, status
`unavailable`, code 2, and no attempted GPU work. Full commands and evidence
are in `Remediation/.../RCPP_OFFLINE_SOURCE_VALIDATION.md`.

### 7.5 Autonomous bundle staging validation

`tools/package_xhdfe_distribution.py` was hardened to include the complete
Stata companion surface, R source, the pinned Rcpp source archive, examples,
validators, and selected certification evidence. Its Release workflow no
longer calls the artifact-dependent packager without the required arguments or
silently labels a compile-only result as an autonomous validated bundle.

The uncompressed stage
`dist/xhdfe_xfe_distribution_20260723_postaudit_220_stage_autonomous` was
created before any ZIP and validated in place:

- its complete `MANIFEST.sha256` initially verified;
- Python CPU/CUDA example validation passed, with CUDA `gpu_used=true`,
  seven iterations, maximum coefficient difference `5.995e-15`, and maximum
  SE difference `3.941e-16`;
- the Stata CPU/OpenMP bundle passed the raw-ID/group-ID parity and xfe/xhdfe
  coefficient checks;
- after switching only the staged plugins to their included CUDA-sm_90
  alternates, the Stata CUDA bundle passed with `e(gpu_used)==1`, backend
  `cuda`, and status `used`;
- the stage was restored to CPU/OpenMP default plugins afterward.

The staged plugin hashes are:

| Variant | xhdfe SHA-256 | xfe SHA-256 |
|---|---|---|
| CPU/OpenMP default | `55a980ba5c4865128cf740359a905eb724f6f020846046957c6e0d3e8711b03e` | `5ac4d248752c797473d601de8f8a706ddd04540c97d37243cce9885eb80e07fd` |
| CUDA H100 sm_90 | `a76ea1f92bd796048c6221576c1021b20c6273227d6b3a5910bdad4c4b07e7f3` | `2fe4d0d55c4fed994b5dc9a7bd9c8283bdfb7067d8d945dd9060a4bc9cddeaea` |

The final ZIP checksum will be inserted with the release-asset evidence in
§11 after the final report/evidence files are copied into the stage and its
manifest is regenerated.

## 8. Monte Carlo evidence and limits

Machine-readable final evidence:

- `Remediation/xhdfegelbach_postaudit_20260723/evidence/base_share_coverage_final.json`;
- `Remediation/xhdfegelbach_postaudit_20260723/evidence/absorbed_share_coverage_final.json`;
- `Remediation/xhdfegelbach_postaudit_20260723/repros/f01_near_fe_band_mc.json`.

The base-share study uses 5,000 replications per design. Monte Carlo standard
errors are about 0.003. The absorbed-target study also uses 5,000
replications; its mean reported SE / empirical SD ratio is 1.00289 and
coverage is 0.954.

Two limitations are intentionally visible:

1. normal-reference CR1 intervals at 40 clusters cover about 0.933-0.939 in
   this DGP; and
2. the adversarial near-FE MC shows continuing sensitivity outside the narrow
   `(1e-9,1e-4]` diagnostic band.

Neither is repaired by changing the estimand or silently substituting a new
inference procedure. The first motivates the new G metadata/note and a future
wild-cluster tranche. The second motivates returning the continuous ratio,
not merely a binary warning.

## 9. Version and documentation surface

The local version surface is internally coherent:

| Location | Version |
|---|---|
| `CMakeLists.txt:2` | `2.20.0.20260723` |
| `pyproject.toml:7` | `2.20.0.20260723` |
| `xhdfe/_version.py:1` | `2.20.0.20260723` |
| `r/xhdfe/DESCRIPTION:4` | `2.20.0.20260723` |
| `stata/xhdfe.pkg:1` | `2.20.0` |
| `stata/xhdfe.ado:1` | `2.20.0 23jul2026` |
| `stata/xhdfegelbach.ado:1` | `1.4.0 23jul2026` |
| `stata/xhdfegelbach.sthlp:2,12-13` | `1.4.0`, shared `2.20.0.20260723` |
| `stata/xfe.ado:1` / `stata/xfe.pkg:1` | `1.10.1 23jul2026` |
| `stata/xhdfeakm.ado:1` | `1.7.2 23jul2026` |
| `stata/xhdfeconnected.ado:1` | `1.2.1 23jul2026` |
| `CITATION.cff:15-16` | `2.20.0`, released `2026-07-23` |

`python -m xhdfe --version` reports `2.20.0.20260723`.
`stata/xhdfe.pkg` includes both `xhdfegelbach.ado` and
`xhdfegelbach.sthlp`. The help page visibly reports its own version and the
shared package version.

The version bump is prepared locally but is not an online release until §11 is
completed.

## 10. Performance gate — PASS at the 30% triage margin

The permitted host-level gate was the established 30% catastrophic-regression
triage, not fine-grained performance certification. The harness used Release
artefacts, deterministic 500,000-row fixtures, 16 threads, three inner
repetitions, alternating ABBA/BAAB order, and retained every completed pair.
It covered standard and absorbed-target designs, unadjusted and clustered VCE,
weighted and unweighted, on CPU and CUDA.

The host remained noisy: CPU load1 was 29.4–34.9 in the CPU runs and
29.1–33.2 in the CUDA run; median sampled background CPU busy was 56.2%,
57.7%, and 60.3% across the two CPU runs and the CUDA run. No quiet-host or
small-regression claim is made.

The first complete five-pair CPU run left `cluster/unweighted` inconclusive
because one candidate block was hit by an extreme host spike. A prospectively
sized complete eight-pair confirmation was therefore run. The final analysis
pooled all 13 pairs from both complete runs, excluded none, and used an exact
one-sided sign test against ratio 1.30:

| CPU standard cell | Median candidate/baseline | Bootstrap 95% CI | Pairs below 1.30 | one-sided p | Decision |
|---|---:|---:|---:|---:|---|
| unadjusted, unweighted | 1.0579 | [0.9506, 1.1196] | 10/13 | 0.0461 | rules out median regression ≥30% |
| unadjusted, weighted | 0.9998 | [0.9511, 1.1533] | 12/13 | 0.00171 | rules out |
| cluster, unweighted | 1.0079 | [0.9546, 1.1330] | 11/13 | 0.0112 | rules out |
| cluster, weighted | 0.9383 | [0.8696, 1.0508] | 13/13 | 0.000122 | rules out |

The independent eight-pair CUDA run passed in every cell:

| CUDA standard cell | Median candidate/baseline | Bootstrap 95% CI | Decision |
|---|---:|---:|---|
| unadjusted, unweighted | 1.0078 | [0.8235, 1.1172] | rules out median regression ≥30% |
| unadjusted, weighted | 1.0490 | [0.9207, 1.1287] | rules out |
| cluster, unweighted | 0.9410 | [0.8114, 0.9800] | rules out |
| cluster, weighted | 0.9646 | [0.9092, 1.0612] | rules out |

All 192 timed candidate CUDA calls (standard plus absorbed-target) reported
`gpu_requested=True`, `gpu_used=True`, backend `cuda`, status `used`.
Continuous NVIDIA logs contain 220 GPU and 140 compute-process samples.
Every timed fit converged and the identity checks remained within the existing
contract.

Evidence:

- `Remediation/.../performance_30pct_20260723/cpu_pooled_complete_runs.json`;
- `Remediation/.../performance_30pct_20260723/cuda8_confirmatory/summary.json`;
- raw pair JSON and telemetry in the two CPU and one CUDA run directories;
- reproducible harness `Remediation/.../repros/performance_ab_30pct.py`.

Verdict: a catastrophic median slowdown of 30% or more is excluded in every
tested standard CPU/CUDA cell. These data do not certify equality, a speedup,
or regressions smaller than 30%.

This targeted triage also does not replace the repository's full
`core23 × 8-cell` performance matrix for a general backend optimization. The
code change is Gelbach-scoped and the standard non-Gelbach numerical path is
bit-identical, but any broader performance claim remains bounded by the
measurement actually run.

## 11. Private/public propagation and release — PENDING

The following actions were not complete when this draft was written:

- selective private commit on `main`;
- private push and `v2.20.0.20260723` tag/release;
- propagation of production-facing source, Python, R, and Stata text files to
  the public `reisportela/xhdfe-xfe` repository;
- byte-identity comparison for every required production file;
- public commit/push/tag;
- CPU-only Linux/Windows/macOS public plugin CI;
- release asset and SHA256 verification;
- `gh-pages`/net-install publication;
- clean-install `help xhdfegelbach` version verification.

The public repo's distinct `xhdfe/__init__.py` must be patched rather than
overwritten. Private CUDA `sm_90` plugins and public CI-built CPU-only plugins
are the documented binary divergence; their source must be identical.

**PENDING FINAL INSERT:**

```text
private commit:
private tag/release URL:
public commit:
public tag/release URL:
production-file byte-identity count:
public CI runs and conclusions:
release asset list and checksums:
gh-pages/net-install verification:
installed help/version output:
```

Until this evidence is present, the local version bump must not be described as
the live public release.

## 12. Files changed by functional area

Canonical implementation:

- `include/hdfe/akm_kss.hpp`
- `include/hdfe/hdfe_regressor.hpp`
- `src/akm_kss.cpp`
- `src/hdfe_regressor_v11.cpp`
- `python/py_hdfe_v11.cpp`
- `xhdfe/gelbach.py`

R frontend:

- `r/xhdfe/src/rcpp_xhdfe.cpp`
- `r/xhdfe/src/RcppExports.cpp`
- `r/xhdfe/R/RcppExports.R`
- `r/xhdfe/R/gelbach.R`
- R help, DESCRIPTION, and tests

Stata frontend:

- `stata/src/xhdfe_plugin.cpp`
- `stata/xhdfegelbach.ado`
- `stata/xhdfegelbach.sthlp`
- Stata integration tests and version-stamped release files

Mirrors:

- `stata/src`, `stata/include`
- `share/xhdfe_estimation_cpp`
- `r/xhdfe/src/akm_kss.cpp`
- `r/xhdfe/src/hdfe_regressor_v11.cpp`
- `r/xhdfe/src/include/hdfe`

Validators and documentation:

- `VALIDATE_GELBACH.py`
- `VALIDATE_GELBACH_FRONTENDS.py`
- `VALIDATE_GELBACH_ADVERSARIAL.py`
- `VALIDATE_GELBACH_HELP.py`
- `README.md`, `xhdfe/help/*.md`, `r/README.md`, R `.Rd` files
- `RELEASE_NOTES_2.20.0.20260723.md`

`bash tools/check_cpp_core_alignment.sh` confirms byte identity of the
canonical core and all required mirrors.

## 13. Final handoff

The local implementation and 30% performance triage are complete. The
conditions for final online-release certification are:

1. re-run a short correctness smoke if any production artifact is rebuilt;
2. propagate exact production bytes to the public repository, respecting only
   the documented binary and packaging divergences;
3. obtain green public CI and verify the installed net-install help/version;
4. replace the remaining **PENDING** release block in this report with immutable commit, run,
   URL, and checksum evidence.

If that publication gate passes without changing the certified
source/artifact state, the post-audit candidate can move to final release
**GO**. If it fails, the correct outcome is to retain this report as a local
remediation record and not claim the 2.20.0 release complete.
