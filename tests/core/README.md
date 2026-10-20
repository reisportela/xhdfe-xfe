# Unified xhdfe test core

This core joins the unchanged **core24 × 8** acceptance matrix, adapted tests
from the public reghdfe repository, and independent numerical oracles. It is
offline: it does not install packages, change HOME, or modify the user's ado
directories. Test output must stay in a fresh directory in this checkout.

The operational performance and regression reference is the published
**2.29.1.20261014**. **2.28.1.20260929** remains the historical precision-contract
anchor. Neither reghdfe nor a second xhdfe binary is a mathematical oracle.

## Coverage

| Layer | Content | Interpretation |
| --- | --- | --- |
| core24 | The original 24 specifications, including the literal Marta `aggregation(sum)` detector; native/Stata × CPU/CUDA × Fast/Comparable | Three interleaved pairs against frozen 2.29.1, 1,152 fits; ordinary estimation without extra residual recovery or discarded warm-ups |
| Upstream compatibility | 55 applicable test files from reghdfe, pinned at `29b2b203f534413369d675d0c0f674d61291f536` | Adapted CPU and CUDA runs; public fields, sample, weights, inference, predictions, FE recovery, groups/individuals and options |
| Reference diagnostics | The original tests on reghdfe, plus literal xhdfe translations | Preserve strict failures and differences; do not count them as candidate numerical certification |
| Reference internals | Three files testing reghdfe's own Mata state or implementation | Reference-only on CPU; not applicable to CUDA/xhdfe |
| Infrastructure | Seven installers, generators, development examples or drivers | Inventoried, never executed as tests or counted as passes |
| Independent numerical battery | `tests/analytic_precision/run.py`, native/Stata and CPU/CUDA | Analytic/rational oracles, explicit dummy OLS, full covariance and negative controls; unsupported cases reported separately |
| Adapter controls | Snapshot mutations, real GPU routing and masked-device rejection | Detect false passes, lost results and silent CPU execution |
| Regression controls | Saturated adjusted R², loose-tolerance CG restart, ordinary CUDA weak graphs and cold CG recovery, independent MP80/MP120 DK | Public/synthetic data; native/Stata checks against independent mathematical references |

`manifest.json` enumerates all 65 upstream do-files. The 83 original test and
licence files are byte-exact and covered by `upstream/reghdfe/SOURCE.json`.
Their MIT licence is retained. Adaptations are generated in each campaign;
the original files are never edited and `ADAPTATION.diff` records every change.

CUDA cases can contain ordinary no-FE OLS controls, which are CPU-only in
xhdfe. Every call records its requested and actual backend. Successful absorbed
CUDA fits require `gpu_used=1`, backend `cuda` and status `used`; a mismatch
fails the outer Python runner even if a Stata test captures the error. Reports
list cases with no GPU execution rather than presenting them as CUDA successes.

## Running

Prerequisites are already-built candidate CPU/CUDA modules and Stata plugin,
Stata/MP, the existing Python scientific dependencies, and an isolated copy of
the Stata dependencies. A machine running CUDA stages needs a visible CUDA
device. There is no automatic software installation or build step.

First use `dependencies.py --plus PATH --output PATH` to copy the already
installed ftools, require, reghdfe, moremata, ivreg2, ranktest, ivreghdfe and
estout packages into a **new directory in this repository**. Source packages
are read-only; do not redistribute this local dependency copy with the core.

Prepare a campaign with the actual paths on the machine:

```sh
python3 -B tests/core/run.py prepare --output Verifications/core_candidate \
  --dependencies /path/to/isolated/dependencies \
  --baseline /path/to/frozen/2.29.1 \
  --registry /path/to/private/core24_registry.json \
  --scratch /path/to/authorised/private/data/temporaries
python3 -B tests/core/run.py run --output Verifications/core_candidate --stage controls
python3 -B tests/core/run.py run --output Verifications/core_candidate --stage reference
python3 -B tests/core/run.py run --output Verifications/core_candidate --stage literal
python3 -B tests/core/run.py run --output Verifications/core_candidate --stage compatibility_cpu
python3 -B tests/core/run.py run --output Verifications/core_candidate --stage compatibility_cuda
python3 -B tests/core/run.py run --output Verifications/core_candidate --stage analytic
python3 -B tests/core/run.py report --output Verifications/core_candidate
```

The example `/path/to/…` values are placeholders, not host defaults. Baseline,
registry and scratch may be omitted when preparing a functional-only campaign;
the performance layer then remains explicitly unconfigured.

After functional adjudication, run `--stage performance` with
`--authorised-scratch /path/to/authorised/private/data/temporaries`. Supplying a
flag records the selected path; it does not replace the human authorisation
required by the project's data policy. Real data are read in place and only
aggregate b/V, diagnostics and digests enter the repository. Stata temporary
data remain inside that separately authorised data directory. Nothing is
deleted from it. Do not run builds or another campaign alongside timings.
The performance runner currently targets Linux and reads `/proc` host counters.

`--stage functional` runs all layers except performance. `--stage all` includes
performance. Both run serially, stop after a failing required stage and return
a nonzero exit status. Existing stage receipts are never overwritten. Prepare a new campaign
after changing source, artifacts, adapters or dependencies. The lower-level
performance runner supports resuming unchanged input identities and a
`STOP_AFTER_PAIR` file for holding between complete pairs. A subset still
reports all missing matrix cells.

## Adaptation decisions

| Upstream expectation | xhdfe adaptation and evidence boundary |
| --- | --- |
| Private Mata objects, internal iteration counters, implementation headings | Keep public estimator results; internal tests remain reference-only |
| Legacy v3 two-FE recovery as the reference | Native explicit-dummy OLS supplies the FE-sum reference |
| `fvset` metadata affects Stata 19.5 `areg` setup | An identical-valued FE variable removes metadata interference in the native reference |
| Weighted Gaussian likelihood constant | Align only the data-dependent weight normalisation convention |
| Replicating clustered data leaves all results unchanged | Retain the literal expected failure; test invariant b, scaled N/RSS and the mathematically required finite-sample V correction |
| DK covariance parity at `1e-12` between estimators | Retain the strict difference, then apply the existing diagonal-scaled `1e-8` V contract in this declared case; independent MP80/120 observed-period Bartlett oracle is mandatory |
| Recovery defaults differ | Adapted tests request `fetolerance(1e-8)` explicitly; literal runs leave it unset; product defaults are unchanged |
| Tighter tolerance always means more iterations | Check finite positive cumulative work within the existing budget; record non-monotonic counts. Cold restarts need not preserve reghdfe's iteration ordering. Independent OLS checks both original tolerances |
| GPU repetition in `memory` and `singleton-and-fweights` | These strict failures also occur on published 2.29.1. Apply the existing scaled b `1e-9`, diagonal-scaled V `1e-8`, and scalar `1e-8` with floor 1 from `tests/stata/_helpers.do`; preserve strict differences and require half/twice-limit mutation controls. Counts, sample and DoF stay exact; no global relaxation |

The independent `_core_results.ado` replaces Stata's proprietary snapshot
utility in xhdfe translations. It tests long result names, scalar/matrix
mutations, absent unweighted macros, preserved r() results, and rejection of a
covariance perturbation above the declared contract.

The analytic stage also runs the CUDA04 weak-graph oracle on both native
backends, with explicit thread requests, rescaled/reordered inputs, the exact
tree shortcut and separate budget refusals. `cuda_regressions.py` runs its
Stata counterpart on CPU/CUDA and the loose-tolerance cold-restart regression;
each requires a terminal success marker and unchanged loaded artifacts.
An omitted or failed supplemental regression makes the stage fail even when
the original analytic battery passes.

CG05 preserves Fast's published CG stopping rule. The analytic stage runs
`fast_cg_mode_contract.py` on both backends, plus the MP120 CG02 checks in all
three tolerance modes. Its Walsh controls have exact independent b/V. When a
core24 registry is supplied, the `github` reproducer must return within its
budget and meet the existing Fast coefficient check against independent
sparse-dummy OLS. The reference contains aggregates and a numerical-input
digest only. Its full covariance differences remain explicit diagnostics:
the general entrywise synthetic-fixture V comparison also fails on published
2.29.1 and is not a new global Fast forward-error limit. Without a registry,
this real-data return regression is reported as `NOT_RUN`.

## Reading the report

The top-level report always lists every stage. It keeps numerical passes,
expected refusals, compatibility passes, reference-only diagnostics, missing
coverage and failures separate. `REVIEW_REQUIRED` is deliberate: numerical
parity is not performance adjudication, and this core is not a certificate for
unexecuted platforms, release assets, installation or repository alignment.

Classify a new failure before modifying the estimator: demonstrated estimator
error, undue verifier refusal, harness error, or missing coverage. Preserve the
literal reproducer, pre-fix evidence, final artifact identities and permanent
regression test. Do not change tolerances to manufacture a pass.
