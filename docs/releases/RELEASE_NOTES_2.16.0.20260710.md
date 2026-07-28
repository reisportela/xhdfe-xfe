# xhdfe 2.16.0 — AKM memory/CPU/CUDA optimization and companion hardening

Released 10 July 2026.

This release ships the adversarially reviewed optimization round for
`xhdfeakm` while preserving the estimator, convergence criteria, interfaces
and FP64 precision. It also hardens `xhdfegelbach` and the public release
pipeline.

## Main changes

- Reduced AKM CPU staging and copies substantially: direct writes into packed
  multi-RHS buffers, lazy weight/square-root materialization and move-based
  transfer of row-level outputs. At 5 million workers and 24 lanes, the main
  staging change alone avoids roughly 960 MB per solve; the common unweighted
  path also avoids a full vector of ones.
- Reduced CUDA synchronization with a fused PCG update, change-only active
  masks and paired CUB reductions returned in one synchronization.
- Corrected cumulative CUDA `solver_iterations` after SE/lincom work.
- Removed repeated sorting/indexing work from Python's
  `subsampling_diagnostic` helper.
- Made Stata `xhdfegelbach` exactly compact FE/cluster codes outside signed
  int32 (and non-integer numeric labels) before plugin transport. Added a
  fail-closed stale-plugin path guard and an int32 range check in the plugin.
- Added companion-command certification for exact categorical relabelling and
  plugin binding.

## Correctness and backend gates

- Canonical `core23 × 8`: 184/184 successful runs, 704/704 coefficient
  records, identical observation counts, CPU rows reporting CPU and all 92/92
  CUDA rows reporting real GPU use.
- AKM CPU and CUDA oracle suites: all checks passed; maximum CPU/GPU
  difference `1.11e-16`; deterministic repeated GPU output.
- Stata certification: 26/26 tests passed, including large/non-integer
  FE/cluster IDs and stale-plugin rejection.
- R AKM/Gelbach tests: 63/63 passed.
- Local H100 Stata plugin: OpenMP-linked, exactly three `sm_90` cubins.

No tolerance, stopping criterion, estimator definition or output interface was
relaxed.

## Performance evidence and caveat

Interleaved incremental probes observed approximately 3.3% on a 1M-row CPU
PCG case, 5–11% on relevant GPU cases, 1–2% on the large real QP panel, and
substantial peak-memory reductions. A tiny forced-PCG GPU probe can cost about
20 ms more; this documented localized trade-off is outweighed by the larger
workload gains.

The full `core23 × 8` timing pass ran while host load averages were roughly
57–80, so it certifies correctness, convergence, feature coverage and backend
selection, but not clean-machine runtime deltas. The changed AKM translation
unit is not called by ordinary core23 estimation paths, and the release makes
no claim based on those contaminated aggregate timings. A quiet-window,
multi-repetition A-B-B-A performance rerun remains useful follow-up evidence.

## Versions

- Shared C++/Python/R package: `2.16.0.20260710`
- Stata `xhdfe`: `2.16.0` (10jul2026)
- Stata `xhdfeakm`: `1.6.0`
- Stata `xhdfegelbach`: `1.0.2`
- `xhdfeconnected`: `1.1.1`; `xfe`: `1.10.0` (date stamps refreshed)

The public release provides CPU plugin bundles for Linux, Windows and macOS,
the Stata net-install site, offline sources (including CUDA build inputs), and
an autonomous offline bundle combining installable binaries with sources.
