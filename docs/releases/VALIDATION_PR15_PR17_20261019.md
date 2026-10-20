# PR #15 and PR #17 integration — 19 October 2026

Status: unpublished isolated candidate, based on private `ecc5b442` and
public PR #17 `7c974305`, with the private PR #15 follow-ups and two
independently reproduced CG fixes. Provisional identity: `2.30.0.20261016`;
the date must be stamped when the release is cut. Neither canonical checkout
nor online `main` has been replaced. Published net-install: 2.29.1.20261014.

This record supersedes the current-candidate scope of
`VALIDATION_2.30.0.20261016.md`, retained as the historical PR #15 record.
Local evidence is under `Verifications/integracao_pr15_pr17_20261019`
in the private repository.

Update, 20 October: expanded upstream coverage exposed CG-03, a CUDA
Comparable cold-restart convergence regression. Its fix retains the requested
precision mode through repair and honours the existing effective tolerance.
The native public-auto reproduction and independent MP80/MP120 b/V oracle
pass; complete validation after CG-03 is still in progress. The completed
results below describe the CG-02 milestone and do not certify later artifacts.
The permanent combined entry point is `tests/core/run.py`; its report keeps
core24, upstream compatibility, reference diagnostics and independent oracles
separate. Runtime comparison uses frozen published 2.29.1.20261014.

## Scope

PR #15 supplies macOS OpenMP export isolation, stored-version identity and
categorical absorb interactions. Private follow-ups retain k-way
cluster/xfepout interactions, factor-prefix parsing and stronger macOS gates.
PR #17 freezes solved CG right-hand sides and restarts from original input
after breakdown. CG-01 shares the iteration budget with that restart; CG-02
rejects a proposed final step inconsistent with the same public tolerance,
before applying it. Cache generation 4435 invalidates earlier faulty routes.
No public tolerance or alpha threshold was tuned.

Reproducers, classifications and permanent tests:
[`../LICOES_PR15_PR17_20261019.md`](../LICOES_PR15_PR17_20261019.md).

## Completed on Athena at the CG-02 milestone

| Check | Recorded outcome |
| --- | --- |
| CPU CTest | 15/15 passed |
| Complete analytic campaign, native/Stata CPU/CUDA | 664 xhdfe PASS, 223 expected unsupported, zero xhdfe failures; 1,029 total jobs including the comparator |
| Comparable T01 subset | 578 PASS, 219 expected unsupported, zero failures |
| MP120 final-step OLS/IV, orders and modes | 18 CPU and 18 CUDA passed |
| MP80/MP120 oracle regeneration | Passed; synthetic CSV identity checked |
| Original Tiago Stata CG contract | 135 CPU fits passed, unchanged limits |
| Complete Stata wrapper | 40 main tests, bootstrap, version/interactions/group/IV contracts and 72 DK/IV cases passed |
| Python frontend | 97/97, no skips, on CPU and in CPU mode on the CUDA-compiled module |
| Eligible CUDA formula oracle | 16/16 with actual GPU use |
| CUDA covariance and retry-budget guards | 16 and 4 passed |
| Cache migration | Incorrect 4433/4434 transforms recomputed; old cache bytes unchanged |
| Explicit 1, 2, 8, 16, 24, 48 threads | Python, Stata and R passed on CPU/CUDA |
| R testthat | CPU: 1,210 PASS, 6 warnings, 2 CUDA skips; CUDA: 1,229 PASS, 6 warnings, zero skips |
| C++ mirrors and provisional metadata | Passed |
| macOS packaging controls | 16/16 mock controls; not native Mac execution |
| Corresponding-source controls | 21/21 passed |
| Benchmark harness | 16 synthetic fits and 8 cross-interface workload checks passed |
| CI workflow repair | Candidate workflow uses the common 97-test validator instead of stale 83; normal CI runs CTest; YAML/shell syntax passed, online execution pending |

The separately graded reghdfe comparator has 10 false-convergence, 10
estimation-error and 13 covariance-failure results. These remain in the raw
report. Expected unsupported cases are refusals, not numerical successes.

R used existing dependencies. Strict `R CMD check` lacks optional
`nanoparquet` and `gt`. With `_R_CHECK_FORCE_SUGGESTS_=false`, CPU reports
one warning and one note; CUDA two warnings (source filename conventions and
compiled-code checks, including the CUDA object). This is qualified evidence,
not clean CRAN certification.

The analytic campaign preceded the extra generation-4434 cache control.
That extension was tested separately against the same unchanged estimator
binaries. A later help edit only wraps a long physical SMCL line.

## Pending performance and release gates

The immutable core24 manifest declares 1,152 fits: 24 specifications × eight
cells × baseline/candidate × three interleaved pairs. Both arms start fresh
processes without discarded warm-ups. Stata uses the ordinary workload,
without extra `residuals()` absorption. Following the maintainer's explicit
decision, the operational performance/non-regression reference is the
currently published **2.29.1.20261014**. Its frozen modules and plugins match
the sealed release hashes. The earlier 27 fits against 2.28.1 are retained
as historical diagnostics and cannot establish a gain over 2.29.1.
The historical precision contract remains anchored to 2.28.1.20260929.
Receipts include complete b/V, sample digests, DoF, iterations, actual backend
and workers, artifact hashes and host load. The exploratory synthetic pilot
and an incomplete matrix cannot establish performance acceptance.

Before publication:

- Complete and adjudicate core24, including noisy cells or local slowdowns.
- Reconcile the public production surface byte for byte, preserving its
  distinct Python packaging entry point.
- Stamp version/date and run mandatory OpenMP platform CI; validate the exact
  final assets. Local H100 builds do not replace those assets.
- Test final macOS assets in licensed Stata/MP on native Apple Silicon and
  native Intel; retain Rosetta evidence separately.
- Obtain the concrete commit/push/publication authorisation required by the
  workspace rules, reconcile both canonical local `main` checkouts and builds,
  publish and verify the net-install site and exact assets.

## Requested advisory debate

Two read-only rounds used Claude Fable 5.1 with maximum effort and public
source/report excerpts. The reviewer withdrew two lane/mask hypotheses after
the complete public code was supplied and supported a shared restart budget.
Root reproduced the budget and covariance defects independently. The debate
is not numerical or release acceptance. A private-code submission was rejected
by automatic approval review; the public-only alternative completed.
Decisions: `advisor/ACCEPTANCE.json` in the evidence root.
