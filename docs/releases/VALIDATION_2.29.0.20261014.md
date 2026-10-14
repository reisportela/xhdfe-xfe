# Validation scope for 2.29.0.20261014

## Composition and reference

2.29.0 integrates the privately tested maximum-performance candidate
2.29.0.20261006 (rebuilt absorption engine, multiway-cluster intercept policy
D-C, Driscoll–Kraay, IV cluster partitions, PF3FE, upstream reghdfe lessons)
with public PR #13 (six commits over 2.28.1: IV first-stage diagnostics,
repeated-control and proportional-instrument fixes, wide multiway covariance,
precision fixes, restored fast-slope performance). The integration record is
`Verifications/pr13_integracao_2290_20261012/RELATORIO_INTEGRACAO_PR13_20261012.md`
(private repository); its reconciliation points are the IV covariance block
(the PR's wide path now runs under the D-C decision rule, and the retained path
accumulates the augmented meat again), the PR's PSD oracle (updated to policy
D-C; 9 of 44 cases change, all 12 positive controls kept), the pure-FE
certificate expectations in Fast, the Stata first stages under `vce(dkraay)`,
and `e(r2_a)` on saturated models (reghdfe convention restored).

The Precision Contract baseline is the current public release
2.28.1.20260929 (maintainer decision of 13oct2026, recorded in `CLAUDE.md` and
`AGENTS.md`); "never lower precision" and "never slower" are measured against
frozen 2.28.1 binaries built with the same native recipe.

## Precision contracts and analytic batteries

All on the CPU module and the CUDA sm_90 module of this composition, Fast and
Comparable unless noted (`RELATORIO` §3 and §7.4):

| Contract | CPU | CUDA |
| --- | --- | --- |
| `iv_covariance_psd_contract.py` (D-C oracle), both modes | 44 + 44 | 44 + 44 |
| `iv_proportional_instrument_contract.py`, `n1_iv_cache_contract.py` | pass | pass |
| `fast_slope_fit_contract.py` | 8/8 | 6/8: two Comparable refusals, adjudicated below |
| `fast_slope_policy_contract.cpp`, `slope_continuation_acceptance_contract.py` | 6, 32 | — |
| `iv_exact_proportional_contract` at 1, 2, 8, 16, 24, 48 threads | pass | — |
| `wide_cluster_covariance_reuse` (exact equality, error 2e-16) | 84 | — |
| `iv_intercept_covariance_contract.py`, both modes | 46 + 46 | pass |
| `iv_weak_distinct_covariance_contract.py` | 8 | 8 |
| `equivalent_cluster_partition_contract.py` (including the indefinite control) | 12 | 12 |
| `intercept_psd_contract.py`, `iv_augmented_meat_contract.py`, `group_column_moment_contract.cpp` | 8, 16, 8 | — |
| `pure_fe_certificate.cpp` (20 rows, two new negative controls) | 20/20 | — |
| New: `iv_cache_contract.py` (bit-identical hit; 1 ULP in Z, y, X, sample or mode misses) | 16/16 | — |
| New: `iv_wide_omitted_contract.py` (92 cases per mode, b ≤ 1.9e-15, V ≤ 1.6e-14) | 92 + 92 | 92 + 92 |
| New: weak IV with `vce(dkraay 3)` and `e(first)` against explicit oracles | 9/9 | 6/6 with real GPU use, 3 documented no-FE refusals |
| `tests/analytic_precision/run.py`, 25 fixtures, 984 jobs | 603 jobs: 371 pass, 90 expected refusals, 0 xhdfe failures | 381 jobs: 248 pass, 133 expected refusals, 0 xhdfe failures |
| ctest, CPU build, all targets | 10/10 | — |
| pytest `tests/` (without analytic, benchmarks, encompassing) | 159 passed, 17 skipped | — |

The 29 battery failures belong to the `reghdfe` comparator, not to xhdfe.

**CUDA Comparable refusals on heterogeneous slopes (2/8 rows).** A direct
absorption probe shows coefficients 9.3e-15 from the explicit-dummy oracle,
but a block residual of 2.8e-8 above the 8e-9 Comparable certificate that five
continuation stages down to 1e-15 do not move: an arithmetic floor of the CUDA
slope solver. The refusal is legitimate and conservative, identical in the
frozen 2.28.1 module, and stays documented as a limitation
(`etapa6_cuda_adjudicacao/`). Nothing in the verifier or the estimator was
changed for it.

## Stata

- `tests/stata/run_stata_tests.sh`: 40 tests, `COMPLETED SUCCESSFULLY`
  (13oct2026) after two documented changes: the two-way `se[_cons]`
  divergences versus reghdfe in `reghdfe-parity-dof.do` are policy D-C and are
  marked KNOWN; `e(r2_a)` on the saturated toy returns 1 again (reghdfe 6.14.1
  convention), removing a guard the candidate had added.
- PR #13 Stata batteries (first stage, repeated controls, proportional
  instruments, PSD; 88 fits) pass with the generic and the native plugin on
  CPU and CUDA; `pr13_firststage_dk_ts.do` (DK, time series in `absorb()`,
  `nosample`, restore) passes on both; `validate_server.do` passes its 8 fits
  on CPU and CUDA.
- Plugins: native recipe (`--linux --openmp --march-native`, CUDA sm_90,
  `XHDFE_FORCE_MARCH_NATIVE_WITH_CUDA=ON`), `libgomp` linked, FMA liveness
  gate; the campaign below is the large CPU smoke (ready, 16 real workers,
  23 iterations, Stata CPU comparable 74 s under the `residuals()` harness).

## R

Three isolated installs from the clone (`portable`, `native`, `cuda`) with
the new `Makevars` configuration stamp: 1210, 1210 and 1215 tests passed, 0
failures (2, 2 and 1 skipped); the CUDA arm reports `cuda_enabled` TRUE; the
Gelbach bootstrap gated on `XHDFE_TEST_CUDA=1` passes its 64 tests separately.
The R–Python parity fixture was regenerated for policy D-C (only
`fe3_cluster2` `se[_cons]` moved, slopes bit-identical; provenance recorded).
`Makevars.win` does not carry the stamp yet (follow-up).

## Performance: Precision Contract matrix

Frozen arms, interleaved pairs, 16 threads, `residuals()` harness, host load
recorded per leg and per minute
(`Verifications/pr13_integracao_2290_20261012/matriz_4bracos_20261013/` and
`core24_3bracos_20261013/`):

- Minimal cells (`main_95_21_ready`, `akm_v02_firstreg`, `simulated_panel`
  × Python/Stata × CPU/CUDA × fast/comparable), four arms, four pairs,
  13oct2026: the release is faster than 2.28.1 in 24/24 cells and in 4/4
  pairs of each, ratios 0.47–0.92; identical iteration counts; max |Δb|
  1.5e-12 (2.2e-9 of one standard error).
- Remaining core24 × 8 cells, three arms (release, pre-PR candidate, frozen
  2.28.1), three pairs, 13–14oct2026, 171/171 complete: on cells at or above
  1 s, 54 faster beyond the descriptive 7 % band, 14 inside it, 0 slower;
  101 sub-second cells reported, not adjudicated; the four CUDA comparable
  cells that 2.28.1 refuses (`directors`, `pf_difficult_10m_3fe`, Python and
  Stata) are estimated; max |Δb| 2.8e-11 outside the AKM second regression in
  Fast.
- Against the pre-PR candidate the release is inside the band in 55 of 70
  adjudicated cells, faster in 12 and slower in 3 (at most 11 % on 2–7 s
  legs); the three fast cells flagged by the minimal matrix were re-measured
  to ten pairs: ready Python CUDA fast 1.007, AKM Python CPU fast 1.031,
  `simulated_panel` Stata CUDA fast 1.058 with 0/10 pairs in favour of the
  release. The last one is recorded as an observation to investigate with its
  own hypothesis; against 2.28.1 that cell stands at 0.876.
- `akm_v02_secondreg` in `xhdfe-fast`: 30 iterations instead of 164 (CPU) and
  38 instead of about 1 940 (CUDA), 3.8–4.6× faster. The Fast coefficients sit
  1.1e-7 (6e-5 of one standard error) from Comparable, the order of the
  documented Fast approximation on ready; 2.28.1 spent 5× the work to sit at
  4e-11. Comparable is bit-identical between the release and the pre-PR
  candidate (472 iterations).

The 2.26.2 arm kept in the minimal matrix (the campaign was launched under
the previous baseline) serves only as a precision annex: the published 2.26.2
Stata plugin crashes with SIGSEGV on `akm_v02_firstreg` Comparable (the P0
of 22sep2026 fixed in 2.28.0).

## Known limitations

- The two CUDA Comparable refusals on heterogeneous slopes above.
- CUDA fits are not bit-reproducible between runs (coefficients move at
  1e-16 to 1e-8; iteration counts vary in Comparable on `directors` and
  `pf_difficult_10m_3fe`); refit assertions use 1e-8.
- Fast results on heterogeneous-slope designs changed at the 1e-7 level
  (see above); Fast and Comparable keep their distinct contracts.
- Sub-second cells and legs under foreign load (`stata-mp` of another user,
  8–20 cores, 13oct 13:20–17:24 WEST) are flagged in the records, not
  averaged.

## Artifact publication gates

The release sequence requires fresh CI builds, native OpenMP worker checks on
Linux/Windows/macOS, clean installed-wheel tests and source/runtime provenance.
The version tag creates a draft. Its exact Linux CUDA plugins must then pass
fresh-process H100 checks, and the exact Stata CPU/net-install artifacts must
load and pass numerical checks before the publish marker is pushed. Local
source tests and earlier release artifacts do not substitute for these
version-tag gates. See the [release procedure](../release-workflow.md).

## Executed release gates

Three preflight tags were needed. `preflight-v2.29.0.20261014` (run 38531061273)
failed on the Windows wheel and plugin probe because three top-level test
sources the CMake build references had not been propagated; `-r2`
(run 38531758954) failed on macOS arm64 because the new fused multiply-add
liveness probe expanded an empty bash array under `set -u` on bash 3.2;
`-r3` (run [38532492153](https://github.com/reisportela/xhdfe-xfe/actions/runs/38532492153))
passed all 11 build/validation/assembly jobs on public commit
`10a8adfa9a0ce7e686dddf333653c10c39241318` with the frontend contract at
97 tests, no skips, and the R OpenMP probe passing. Both fixes are in the
private and public sources (`docs/LICOES_INTEGRACAO_PR13_20261012.md` §12).

The version tag `v2.29.0.20261014` at the same commit ran
[38540846493](https://github.com/reisportela/xhdfe-xfe/actions/runs/38540846493): 12 jobs
succeeded and the publish job was skipped as designed; it staged draft release
412566854 with 13 assets. Every asset matched GitHub's SHA-256 digest, its
recorded size and `SHA256SUMS.txt` (`FINAL_ASSETS_AUTHENTICATED.json`):

| Asset | Bytes |
| --- | ---: |
| `SHA256SUMS.txt` | 1,257 |
| `xhdfe-2.29.0.20261014-corresponding-source.zip` | 273,122,751 |
| `xhdfe-2.29.0.20261014-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | 4,639,669 |
| `xhdfe-2.29.0.20261014-cp312-cp312-win_amd64.whl` | 5,431,614 |
| `xhdfe-2.29.0.20261014.tar.gz` | 4,791,832 |
| `xhdfe-src.zip` | 15,472,510 |
| `xhdfe_2.29.0.20261014.tar.gz` | 1,842,858 |
| `xhdfe_xfe-offline-source-and-platform-binaries.zip` | 349,424,354 |
| `xhdfe_xfe-stata-linux-cpu.zip` | 7,779,944 |
| `xhdfe_xfe-stata-linux-cuda.zip` | 11,008,779 |
| `xhdfe_xfe-stata-macos-universal.zip` | 8,220,536 |
| `xhdfe_xfe-stata-netinstall-site.zip` | 39,112,230 |
| `xhdfe_xfe-stata-windows-cpu.zip` | 8,041,533 |

Archive layout (`FINAL_ARCHIVE_LAYOUT.json`): the 20 Stata ado/help files
are byte-identical across the four platform ZIPs, the net-install snapshot and
the release sources; the offline bundle carries the same prebuilt plugins and
package files as the platform ZIPs and the Python/R assets; `xhdfe-src.zip`
carries the release `VERSION`, the frontend sources and the certified
`Rcpp_1.1.2.tar.gz`. Native receipts (`FINAL_NATIVE_PLATFORM_RECEIPTS.json`):
10 plugin receipts (Linux CPU, Linux CUDA host, Windows, macOS x86_64 and
arm64 × `xhdfe`/`xfepout`) tied to the final plugin bytes with one and two
useful OpenMP workers and numerical parity, 2 wheel receipts (Linux,
Windows) tied to the final module bytes, and 4 macOS serial negative controls
rejected as expected.

Runtime gates on the authenticated assets (`FINAL_RUNTIME_GATES.json`,
25 checks, all passed; no foreign GPU process and 0 % utilisation at start):

| Gate | Observed result |
| --- | --- |
| Corresponding-source bundle, four platform package sites, net-install site, Linux binary floor, `ldd` of the four Linux plugins | validators passed; every plugin links `libgomp`, none links `nvomp` |
| Downloaded manylinux wheel and sdist, isolated local installation | archive closure passed; the complete 97-test frontend gate passed without skips or source shadowing; the installed OpenMP probe passed |
| Exact Linux CPU ZIP, fresh licensed Stata: analytic b/full V in Fast and Comparable, `xfepout` within, documented auto-data example | PASS, package bytes unchanged (XHDFE_RELEASE_ANALYTIC_SMOKE_PASS, ISSUE11_AUTO_EXAMPLE_PASS) |
| Exact Linux CUDA ZIP on the H100 (`gpu_used==1`): the same analytic cases on CPU and CUDA | PASS, package bytes unchanged (XHDFE_RELEASE_ANALYTIC_SMOKE_PASS, ISSUE11_AUTO_EXAMPLE_PASS) |
| Exact Linux CUDA ZIP: `savefe` and reconstruction on CUDA | PASS, package bytes unchanged (XHDFE_SAVEFE_CUDA_CURRENT_BYTES_PASS) |
| Exact Linux CUDA ZIP: `xfepout` CUDA/CPU hint guard | PASS, package bytes unchanged (XFEPOUT_CUDA_CPU_HINT_GUARD_PASS) |
| Exact Linux CPU ZIP: `xfepout` cache hit receipt, signature-miss rewrite, cache profile | PASS, package bytes unchanged (XFEPOUT_CACHE_HIT_RECEIPT_PASS, XFEPOUT_CACHE_SIGNATURE_MISS_REWRITE_PASS, XFEPOUT_CACHE_PROFILE_PASS) |
| Exact Linux CUDA ZIP on CPU: the same `xfepout` cache cases | PASS, package bytes unchanged (XFEPOUT_CACHE_HIT_RECEIPT_PASS, XFEPOUT_CACHE_SIGNATURE_MISS_REWRITE_PASS, XFEPOUT_CACHE_PROFILE_PASS) |
| Large CPU smoke, exact 2.28.1 public CPU package vs exact 2.29.0 CPU package, fresh Stata processes, 16 workers | 46,156,187 observations, identical sample signature and degrees of freedom, 23 iterations in both; scaled coefficient error 5.1e-15, full-covariance scaled error 1.7e-12; fit times 72.38 s (2.28.1) and 49.24 s (2.29.0), ratio 0.68. One pair: an artifact smoke, not a performance campaign (the campaigns above are). |

The version-tag `R CMD check` reports `2 WARNINGs, 4 NOTEs`, the pattern already
recorded for 2.28.0 and 2.28.1 (compiled code (NOTE); for gnu extensions in makefiles (NOTE); if this is a source package (WARNING); installed package size (NOTE); package dependencies (NOTE); whether package ‘xhdfe’ can be installed (WARNING));
the native sources were not changed to suppress these diagnostics.

Both canonical local checkouts were brought to the release state: private `main`
fast-forwarded to the integration branch, `build/` and `build_cuda/` rebuilt
(Release, native tuning, required OpenMP, CUDA sm_90) and exercised with the
analytic Polars/FE check in Fast and Comparable on CPU and on the H100 with
actual GPU use (`canonical_cpu_native.json`, `canonical_cuda_native.json`); the
canonical CPU module is byte-identical to the module the promotion campaigns
measured (`74855148…`). The tracked Stata plugins are the native-recipe builds the
campaigns measured (`cd63b7aa…`, `7d379ec4…`); their sources did not change
after that build.
