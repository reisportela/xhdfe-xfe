# Validation record — xhdfe 2.30.0.20261016 (pre-publication cut)

Status: candidate on the private branch `integrate/pr15-issue14-20261016`
(worktree `build-worktrees/pr15-integration-20261016`), cut on 16 October 2026,
**not published**. The published release remains 2.29.1.20261014. The precision
and timing baseline of the Precision Contract remains 2.28.1.20260929. The cut
date is re-stamped if the public release lands on another day.

## Scope

- Integrates [PR #15](https://github.com/reisportela/xhdfe-xfe/pull/15) (three
  commits by Tiago Tavares, cherry-picked with their authorship) for
  [issue #14](https://github.com/reisportela/xhdfe-xfe/issues/14): macOS OpenMP
  runtime export isolation, shared `e(version)` identity, `absorb()` interactions
  with three or more categorical components.
- Follow-ups in this repository: `cluster()` and `xfepout` `absorb()`/`clustervars()`
  generalised to k components (`##` beyond two components refused in cluster
  options; the help now states that `reghdfe` treats `a##b` in `vce(cluster)` as
  one joint factor, which `xhdfe` never did); `_xfepout_package_version`;
  the documented `ib#.`/`b#.` prefixes finally parse in interaction terms (a
  double-escaped regular expression had made them fail with r(101) in every
  grammar since they were documented); help/README/overview corrections;
  portable `MH_WEAK_DEFINES` gate in `tools/validate_macos_openmp.py`; complete
  OpenMP-import classification with a self-test in
  `tools/check_macos_openmp_exports.py`; `build_macos.sh` one-command macOS
  build shipped in the source distribution; CUDA version tests reading the ado
  header; new `tests/stata/cluster_interaction_contract.do`; `version_contract.do`
  extended to `xfepout`.
- No file under `src/`, `include/` or `python/` changed. The Linux plugins in
  `stata/` are byte-identical to `main` (sha256 prefixes `f1880c48…` xhdfe,
  `7d379ec4…` xfepout). No absorption-path, tolerance, fast-mode or
  thread-selection change, hence no core24 performance campaign.

## Independent review (Codex Astra, gpt-6-astra, maximum effort, read-only)

Two rounds, memos and replies under `Verifications/issue14_pr15_20261016/astra_ronda{1,2}/`.
Round 1 returned eleven numbered findings: one blocking (the macOS build script
counted undefined weak imports as weak definitions), four to fix (relative
install path, incomplete import classification, `xfepout` missing from the
version contract, claims beyond the evidence and an unfixed archive identity),
one optional (the `##` policy), three without a finding (parsers, export list and
`MH_WEAK_DEFINES` gate, restamp) and one blocking only for public publication
(native-host validation of the final assets). Every fixable finding was addressed
in the commit "Address Astra round 1 …". Round 2 closed findings 3, 4, 6 and 7,
confirmed the prefix regex fix and the `##` decision, and raised one new harness
error (the Mac script grepped a full path that Stata wraps across log lines, a
false FAIL), fixed in "Address Astra round 2 …" by moving the identity check into
the Stata contract with short markers. Round 3 confirms closure (see below).

## Evidence carried by PR #15 (maintainer's Mac and this host)

Summarised from `candidates/ISSUE_14_STATA_FIXES_20261016.md`: on StataNow MP 19.5
(Apple Silicon native and the Intel package under Rosetta) the published 2.29.1
runtime crashed 5/5 eight-fit processes at four threads (SIGABRT), a relink of the
same objects without the export filter crashed 4/5, and the filtered runtime
completed 5/5; 60/60 real-host cases (1/2/4/automatic threads, both tolerance
modes, OLS, IV with first-stage diagnostics, `xfepout`); 26 OLS + 2 IV interaction
cases per configuration on ARM, Rosetta, Athena CPU and Athena CUDA; 72 timed
fits within noise (−7.1 % to +3.5 %). The raw receipts live on the maintainer's
Mac; attaching them to the PR or committing them under
`docs/releases/candidates/` was requested.

## Checks executed on this host (athena, Linux, Stata/MP 19.5, H100), final ados

Logs under `Verifications/issue14_pr15_20261016/gates_athena_20261016/` (the
`contracts/` folder holds the runs made before the restamp or before the
prefix fix; `contracts_final/` holds the runs on the committed ados).

| Check | Result |
| --- | --- |
| Published 2.29.1 `xhdfe_libomp.dylib` (`llvm-otool`, `llvm-nm`) | `WEAK_DEFINES BINDS_TO_WEAK`; 1724/1605 exports (x86_64/arm64), 22 weak C++ template exports including `__kmp_suspend_64<false,true>`; both plugins import 31 symbols, all C (`___kmpc_*`, `_omp_*`) |
| `tools/check_macos_openmp_exports.py --self-test`; then with `XHDFE_MACOS_NM=llvm-nm` against the published binaries | self-test OK (positive and negative controls); imports 31/31 covered for both plugins; runtime rejected with 843 (x86_64) / 834 (arm64) private exports, as intended |
| `tools/validate_macos_openmp.py inspect` against the published runtime / plugin | runtime rejected (`MH_WEAK_DEFINES`); plugin accepted with `weak_defines` recorded |
| `tools/validate_release_metadata.py --expected-version 2.30.0.20261016` | PASS (20 production Stata text files); fails closed against the pre-PR ado |
| `tools/check_cpp_core_alignment.sh`; 25 Stata text files compared with `share/` | OK; all identical |
| SMCL → text (`translate smcl2txt`) of `xhdfe.sthlp`, `xfepout.sthlp` | 0 leftover SMCL fragments; no "exactly two components" / "two-way" wording left |
| `tests/stata/version_contract.do` cpu, cuda (xhdfe and xfepout) | PASS; `e(version)` = `2.30.0 16oct2026` / `1.13.4 16oct2026` after fits and replay |
| `tests/stata/absorb_interaction_contract.do` cpu, cuda | PASS; max abs. difference vs reghdfe 1.887e-15 (b), 1.112e-13 (se); grouped-ID parity b = 0, V ≤ 1.7e-21 (CUDA) |
| `tests/stata/cluster_interaction_contract.do` cpu, cuda | PASS, 11 cases each (one-way and multiway clusters on three-component terms, `i./ib#./bn./ibn.` prefixes in cluster and absorb terms, `##` two-component expansion, negative controls, `xfepout` numeric and string parity); reghdfe comparator on the grouped cluster ≤ 6.7e-16 (b), ≤ 1.7e-14 (se) |
| `tests/stata/openmp_host_contract.do 4 100000 3` on Linux Stata/MP 19.5 | PASS, 15 cases (12 OLS/IV, 3 `xfepout`), 4 workers observed |
| `tests/stata/savefe-cuda.do`, `tests/stata/xfepout-cache-cuda.do` (H100) | PASS with header-derived version assertions (xfepout rerun after the restamp: 1.13.4) |
| Probes: `reghdfe` 6.14.1 `vce(cluster a##c)` and `a##c##firm`; factor prefixes before/after the regex fix | reghdfe: one joint factor in both cases (`N_clustervars = 1`); xhdfe `a##c`: three dimensions (documented); `ib2.a#bn.b#ibn.c` r(101) on 2.29.1 and on the branch before the fix, accepted after it (`probes/`) |
| `tests/test_macos_openmp.py --out …` (packaging controls) | 16/16 OK with the extended validator; the pre-PR test file fails against the current builders |
| `tests/test_corresponding_source_bundle.py` | 21 passed |
| `tests/stata/run_stata_tests.sh`, first run (ados after the restamp, before the prefix fix) | `testall.do` 40 tests COMPLETED SUCCESSFULLY; Gelbach bootstrap PASS; version, absorb-interaction, cluster-interaction, IV first-stage, alias, proportional-instrument and repeated-control contracts PASS; Driscoll–Kraay IV extended 72/72 PASS |
| `tests/stata/run_stata_tests.sh`, final run on the committed ados (commit `4b22a552`) | `testall.do` 40 tests COMPLETED SUCCESSFULLY; Gelbach bootstrap PASS; all eight standalone contracts PASS; Driscoll–Kraay IV extended 72/72 PASS (`gates_athena_20261016/suite_final/`) |
| Offline rebuild of the CUDA sm_90 plugins from the distribution (archive `eb483c5a…`, identical plugin sources to the final archive, see `dist_athena_20261016/final_zip_3403d844/SOURCE_IDENTITY.txt`) | fatbin present, libgomp linked; `version_contract.do`, `absorb_interaction_contract.do`, `cluster_interaction_contract.do` PASS on CUDA with `e(gpu_used) == 1` asserted (`dist_athena_20261016/final_zip/`) |
| Loaded-ado identity in `version_contract.do` (`VERSION_CONTRACT_SOURCE_OK` markers) | positive control on the long worktree path PASS; negative control (package directory without the ados) fails with r(9) and no PASS marker |
| Codex Astra round 3 (`Verifications/issue14_pr15_20261016/astra_ronda3/`) | favourable to fast-forwarding the private `main` and to delivering the archive as a macOS test candidate; findings 1, 2/12, 5 and the `xfepout` prefix coverage closed; still open and recorded here: abbreviation and `group()`/`individual()` coverage, the positive control of the filtered runtime per architecture, and the local `CLAUDE.md` correction to preserve at the closing commit |

## Source distribution for macOS builds

`dist/xhdfe-src-2.30.0.20261016.zip` (16 MB, committed a80774db) is built by
`tools/make_source_dist_zip.sh` from `git archive` of the committed tree (an
earlier build from the working tree had shipped 164 test logs and was replaced),
with the pinned LLVM OpenMP 20.1.8 source tarball (`87f0227c…`, taken from the
published 2.29.1 `xhdfe-src.zip`) and the new `build_macos.sh`. Checksums:
`dist/DIST_2300_20261016.sha256`. The builder's own validation passed (archive
closure, 21 corresponding-source tests, checker self-test, CMake configure and
non-finite guard build). Archive SHA-256
`3403d8441ff4c1ce7334e87511846f91278f7d9299f856682178a237e3586035`. Offline rebuild on this host
from this archive: Linux CPU plugins (0 downloads, libgomp linked) and the three
contracts PASS on CPU (`dist_athena_20261016/final_zip_3403d844/`); the CUDA sm_90
plugins were rebuilt offline from the previous archive, whose plugin sources are
byte-identical (`SOURCE_IDENTITY.txt`), and passed the three contracts on the
H100 (`dist_athena_20261016/final_zip/`). `build_macos.sh --dry-run` prints the
36 commands it would run on a Mac. Guide: `TESTAR_MAC_ARM_2300_20261016.md`.

## Not executed here; required before publication

- GitHub Actions on the public repository: `macos-openmp.yml` with the export
  checker, plugin-import gate and `MH_WEAK_DEFINES` gate, and the `release.yml`
  revalidation, have never run (PR #15 carried `[skip ci]`). The version-tag
  preflight is their first execution. The `MH_WEAK_DEFINES` gate needs a
  positive control on a correctly filtered runtime of each architecture (the
  Mac build from the distribution provides it) before it can be trusted not to
  block a correct release.
- Real Stata/MP on macOS with the rebuilt release assets:
  `tools/run_stata_openmp_host.py --threads 1 2 4 0` on Apple Silicon (native)
  and Intel (Rosetta) by the maintainer, and native Intel by the reporter.
- Windows and the Python/R frontends were not rerun (version strings only).
- Coverage not added: `group()`/`individual()` with k-way interaction terms and
  variable abbreviations in interaction components.
- Public propagation with byte identity, H100 validation of the exact assets,
  gh-pages/net-install readback, and the closing commit (`CLAUDE.md` with its
  local core24 correction, `AGENTS.md`, sealed checkouts manifest) follow the
  release workflow.
- Follow-up: a CI-runnable collision fixture that pre-loads a foreign LLVM
  runtime into the probe host (`docs/LICOES_ISSUE14_20261016.md`, lesson 2); a
  decision on aligning `vce(cluster a##b)` with reghdfe's joint-factor meaning
  (lesson 10); the share-mirror validation debt (lesson 8).
