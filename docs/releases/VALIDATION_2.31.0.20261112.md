# Validation of xhdfe 2.31.0.20261112

## Artefacts

The validated build is `cand_2310_lean_r2`: CUDA `sm_90`, OpenMP, native CPU tuning. Its sources and binaries are
recorded in `Verifications/perf_fereg_gap_20261106/p8/MANIFEST_cand_2310_lean_r2.json`. Later text-only re-freezes
(`MANIFEST_cand_2310_lean_r2_texts*.json`) changed help texts, notes, records and two CI workflow files. No `.ado`,
source or binary changed in them.

| Artefact | sha256 (prefix) | `libgomp` | `zmm` | `sm_90` |
|---|---|---|---|---|
| `stata/xhdfe.plugin` | `54bd8207…` | yes | 139,501 | yes |
| `stata/xfepout.plugin` | `bb9379dc…` | yes | 105,402 | yes |
| Python CPU module | `35662c92…` | yes | 137,244 | no |
| Python CUDA module | `1c2739dd…` | yes | 140,133 | yes |

Before the freeze, three gates passed: `tools/check_cpp_core_alignment.sh`, `tests/release_lint.sh` and
`tools/validate_release_metadata.py --expected-version 2.31.0.20261112`. All three passed again after each text
re-freeze. The seven Student-t CTests pass, including the x87 guard under an imposed 53-bit precision control word and
the host-regime test.

## Acceptance references

The references are those of the user decision of 11 Nov 2026:
- `reghdfe` 6.14.1 at `tolerance(1e-12)`, which converged in every case;
- the previously validated build `cand_p3fix_final`.

Working limits against `reghdfe` are |Δb_j| <= 1e-8·max(1,|b_j|), constant included, and |ΔSE_j|/SE_j <= 1e-7; for
residual vectors, a relative norm of 1e-7. Exact oracles arbitrate when the two disagree. The protocol was sealed
before any fit: `benchmarks/_out/perf_fereg_gap_20261106/p8/lean_tests_r2final/PROTOCOLO.md` and `RECIBO.json`. Each
case is one fit in a fresh process, with 16 threads and the absorption cache off.

The results below are bounded evidence: they hold for the cases listed and are not a universal guarantee.

## Component results (one fit per case)

### L1: weighted CPU retry, on by default

| Case | 2.31.0 against `reghdfe` | `cand_p3fix_final` against `reghdfe` |
|---|---|---|
| CPU, one regressor, fw (Python and Stata) | b 1.9e-13, SE 2.7e-8: **accepted** | b 2.5e-8, SE 2.6e-6: outside the limits; the oracle agrees with `reghdfe` |
| CPU, one regressor, aw | b 8e-15, SE 2.5e-9: **accepted** | b 3.3e-9, SE 6.7e-6: outside the limits |
| CPU, four regressors (192,000 and 240,000 rows), fw and aw | identical to the previous build, accepted | accepted |

- **Invariances.** Weights of ones against no weights, fw against literal replication, and rescaled aw agree to
  2e-16 or better.
- **Failures of the retry.** With an injected failure, or a budget of primary + 3 iterations, the primary is returned
  (status 3). The published iterations count the discarded retry and stay within `max_iter`.
- **Unweighted fits.** Bit for bit identical to the previous build in the 13 compared cases: b, V, e, t, p-values,
  intervals, iterations, method and retry status. These cover the three-way, path-like and small public fixtures;
  Fast, Comparable and strict, with and without `savefe`; and the components switched off, with and without weights.
  Unweighted CUDA fits kept the previous route and iteration counts in the compared cases.
- **Cost and effect.** aw U[0.5, 2), one fit per switch value:
  - on `pf_difficult_10m_3fe` the retry ran (828 + 40 MLSMR iterations) and cost +2.7% (316.2 s against 324.8 s);
    the effect was Δb/max(1,|b|) 5.0e-7, Δb/SE 1.4e-3 and ΔSE/SE 5.2e-6;
  - on `main_95_21_ready` the selector already chooses MLSMR, so the retry does not run.

### L2: weighted CUDA three-FE completion, off by default

- **Default route.** Weighted CUDA fits take the previous build's route (`cuda_tail` none). The iteration counts vary
  between runs in both builds. Against `reghdfe` they stay outside the limits: b up to 4.4e-7 and SE up to 2.3e-5
  (known limitation).
- **Switches.** `XHDFE_P8_CUDA_WEIGHTED_COMPLETION=1` enables the completion. With it, the weighted cases agree with
  `reghdfe` (b 2.4e-12 and 6.6e-12; SE 2.9e-8 and 7.1e-9), and a cold start agrees with the warm start to 1e-14.
  On large weighted data it ran more than 70 times (`main_95_21_ready`) and more than 150 times
  (`pf_difficult_10m_3fe`) longer than the default route and was stopped without completing, far above the 5% cost
  limit of this release, which is why it is off.
  `XHDFE_TEST_CUDA_COMPLETION_COLD` acts only when set, but then also on unweighted fits. L2 is therefore inert only
  without these two settings.

### L3: Stata `residuals()` from the fit

- **CPU.** In Fast and Comparable the stored vector equals the Python `residuals_` bit for bit. Against `reghdfe` it
  agrees to 1.5e-9 (Fast) and 6.4e-12 (Comparable) on the three-way fixture. On the path-like fixture `reghdfe` misses
  the vector by 5.4e-4; the oracle arbitrates for xhdfe (6e-12).
- **CUDA.** The stored vector has the accuracy of the CUDA fit. The path-like Comparable value, 1.3e-6, is a documented
  limitation against the historical 4.5e-3; it is not a pass of the 1e-7 limit.
- **Strict-residual** keeps the previous route. On CPU the vector is identical to the previous build. On CUDA it varies
  between runs in both builds (the P6 limitation): 45 iterations at 2.1e-6 or 136 iterations at 1.4e-13.
- **Large data.** On `main_95_21_ready`, Stata CUDA Comparable, one fit each, the fit took 14.9 s without
  `residuals()`, 16.5 s with it, and 147.3 s with the old route.

### Student-t correction

- The seven CTests pass.
- The final build is bit for bit identical to the first lean build in 7 of 7 cases (b, V, e, t, p-values, intervals).
  On this host the correction changes nothing.
- **Imposed 53-bit precision on Linux** (x87 control word 0x027F, the Windows x64 host default), independent driver
  and grader of the coordinator on the final Python CPU module:
  - no refusal in 410 fits (two designs of 205 residual-df values from 3 to about 5e7); on the 200 of those values
    audited with 2.30.1, that version refused 97;
  - the control word and MXCSR were restored after every fit;
  - p-values within 2.3e-15 (relative) and confidence limits within 7.3e-16 of the half-width against a 50-digit
    reference, at 53 and at 64 bits;
  - where b, SE and t coincide between the two regimes (150 fits), p-values and confidence limits are bit for bit
    identical.
- **Stata probe under the same imposed regime** (the probe later run on Windows): 2.30.1 refuses 28 of 100 and 4 of 9
  OLS fits and 3 of 6 `xhdfeakm` fits with `controls()`; the final plugin refuses none, and its KSS results agree
  with those at 64 bits in the nine digits displayed.
- **Real Windows.** The same probe with the Windows plugin built by the release workflow is a publication gate; its
  result is recorded with the executed release gates.

### Large-data CPU smoke after the plugin rebuild

On `main_95_21_ready` (46,156,187 rows; aggregates only), Stata CPU Comparable with 16 threads, one fit each:

| Build | Seconds | Iterations | Threads used |
|---|---|---|---|
| 2.31.0 | 32.97 | 23 | 16 |
| `cand_p3fix_final` | 34.38 | 23 | 16 |

## Test core (`tests/core`) on the final build

| Stage | Kind | Stage result | Counts | Non-pass files and classification |
|---|---|---|---|---|
| controls | required | `PASS_CONTROLS` | 11/11 PASS | — |
| reference | diagnostic | `DIAGNOSTIC_COMPLETE`, returncode 1 | 52 PASS, 6 FAIL | `cluster-after-expand`, `fvstrip`, `lagged-slope`, `predict-aweight`, `stdata2`, `stdata3` |
| literal | diagnostic | `DIAGNOSTIC_COMPLETE`, returncode 1 | 42 PASS, 13 FAIL, 3 REFERENCE_ONLY_PASS | `alphas`, `cluster-after-expand`, `cluster-equals-robust`, `dkraay`, `fvstrip`, `inconsistent-r2`, `lagged-slope`, `noabsorb`, `predict-aweight`, `report_constant`, `stdata2`, `stdata3`, `test-alphas` |
| compatibility_cpu | required | `PASS_COMPATIBILITY`, returncode 0 | 53 PASS, 1 PASS_EXPECTED_DIFFERENCE, 1 PASS_PUBLISHED_CONTRACT, 3 REFERENCE_ONLY_PASS | `cluster-after-expand` (expected difference), `dkraay` (published contract) |
| compatibility_cuda | required | `PASS_COMPATIBILITY`, returncode 0 | 50 PASS, 1 PASS_EXPECTED_DIFFERENCE, 4 PASS_PUBLISHED_CONTRACT | `cluster-after-expand` (expected difference); `dkraay`, `memory`, `options`, `singleton-and-fweights` (published contract) |
| analytic | required | `PASS_SCOPED_ORACLES`, returncode 0 | xhdfe: 664 PASS, 223 UNSUPPORTED_EXPECTED, 0 failures | supplemental contracts PASS (CUDA04, CG02, three-FE covariance, warm budget, Stata CUDA regressions; Fast CG `PASS_SCOPED_FAST_POLICY`) |

The campaign report (`p8/core_r2/REPORT_001.json`) has status `REVIEW_REQUIRED`, which is its design for every
campaign: performance and platform checks are separate. It lists no changed inputs. The performance stage was not
run.

Notes on the table:
- **Diagnostic stages.** `reference` and `literal` are diagnostic by definition (`diagnostic_stages` in `PLAN.json`).
  Their returncode 1 is the diagnostic result, not a harness failure.
- **`reference`** runs `reghdfe`'s own tests on this host. Its six failures are failures of `reghdfe` itself there.
- **`literal`** runs literal translations of upstream expectations, including `reghdfe` internals. Its failures are
  the documented strict differences, which are retained and not counted as certification.
- **`compatibility_*`.** The expected difference is the documented replication finite-sample correction; the
  published contracts are the scaled `1e-9`/`1e-8` contracts of `tests/stata/_helpers.do`.
- **Base comparison.** This harness accepts only the repository root as candidate, so `cand_p3fix_final` was not run
  through it. The analytic counts equal those recorded for 2.30.1 (664 PASS, 223 expected unsupported).

## Analytic battery (`tests/analytic_precision`), final build against the base

The final build ran inside the test core's analytic stage (report `p8/core_r2/analytic/report_001.json`):
- xhdfe: 664 PASS and 223 UNSUPPORTED_EXPECTED;
- T01: 578 PASS and 219 UNSUPPORTED_EXPECTED;
- `xhdfe_scoped_certified` and `xhdfe_defaults_certified` true; no custody mismatch.

The base `cand_p3fix_final` ran the same 1,029 jobs in `p8/battery_base_r2`. It gave the same counts in every
category, and the per-job verdicts are identical for all 1,029 jobs:
- xhdfe: 664 PASS and 223 UNSUPPORTED_EXPECTED;
- T01: 578 PASS and 219 UNSUPPORTED_EXPECTED;
- comparator rows: explicit OLS 25 PASS; reghdfe 77 PASS, 7 UNSUPPORTED_EXPECTED, 10 FALSE_CONVERGENCE,
  10 ESTIMATION_ERROR and 13 FAIL_COVARIANCE.

The reghdfe failures are failures of the comparator, not of xhdfe.

## Performance decisions

The maintainer accepted two measured losses of the P3 increment on 11 Nov 2026, under the performance trade-off
allowance:
- +3.09% (+0.104 s) in Stata CPU Fast on `pf_simple_10m_3fe`;
- +11.09% (+0.067 s) in Stata CUDA Comparable on `synthetic-uniform-harder`.

In both, the final method and the iterations are identical in the two arms; the mechanism is not identified. Twenty-one
cases of that comparison remain under review. The record is in
`Verifications/perf_fereg_gap_20261106/REPETICOES_REVIEW_MATRIZ3_20261110.md` §7.

## Not covered

- Windows and macOS host checks of the final plugins (CI).
- The full core24 performance campaign of the final build against 2.30.1. Its timings come from the 2.31-line matrix
  and its repetitions, with the chain caveat stated in the release notes.
