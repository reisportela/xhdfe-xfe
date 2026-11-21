# Validation — 2.34.1.20261121

Scope: the first public release of the 2.32 to 2.34 line (reduced-system route
for `reghdfe-comparable` fits on CPU, corrections to `savefe`, to
heterogeneous slopes and to weighted projections, CUDA libraries loaded on
demand). This record lists what was run for it on 21 November 2026 and what
was not. The numerical and performance results identify their scope; this is
not whole-package certification. The comparison version is the published
2.31.1.20261112.

Private evidence is kept in the development repository. The estimator code of
this release is that of development commit `a0c1d8b6f`; the commits after it
change help files and documentation, release metadata, packaging lists and
release scripts.

## Agreement with the published 2.31.1

`core24s1` suite, Stata, `reghdfe-comparable`, CPU, tolerance 1e-8, the
default routes of each version, at 4 and at 16 processors (`set processors`
and `numthreads()`), two runs per cell, one new Stata process per run. The 24
specifications completed: 22 in the main measurement and
`akm_v02_secondreg_small` and `marta_group_individual_3m` in a complement
with the two xhdfe versions only.

- Coefficients: largest `|Δb|/max(1,|b|)` 7.4e-13 over the 24 specifications
  and both processor settings.
- Standard errors: largest relative difference 4.5e-10.
- The estimation sample, by count and fingerprint, is the same in every cell.
- Every run converged and was certified.
- With the experimental firm-slope solver switched on (see the release
  notes), `akm_v02_secondreg_small` agrees with 2.31.1 to 3.7e-13 in the
  coefficients and 1.8e-9 in the standard errors.

## Time against the published 2.31.1

Same runs; the clock starts with the data in memory and covers the fit and the
printed output. The server was in shared use and each cell is the median of
two runs, so small differences are noise: in 29 of 152 cells of the main
measurement the two runs differ by more than 15 %.

- 2.34.1 is faster than 2.31.1 in 20 of the 22 specifications of the main
  measurement at both processor settings; the ratio of the medians goes from
  0.17 to 0.95.
- It is slower in two, the ones on which the reduced-system route gives up
  and the fit continues on the route of 2.31.1 after paying for the attempt:
  `pf_difficult_10m_3fe` (17.6 s against 12.0 s at 4 processors, 8.1 s
  against 6.5 s at 16) and `synthetic-zigzag` (about 0.03 s more).
- On the 46-million-row panel `main_95_21_ready`: 49.8 s against 52.6 s at 4
  processors, 21.9 s against 35.7 s at 16.
- `akm_v02_secondreg_small` (firm-specific slopes) takes the same route in
  both versions by default, with the same 3,013 iterations and identical
  results: 34.4 s against 39.9 s at 16 processors. At 4 processors the
  two-run medians, 80.3 s and 75.5 s, were not measured in alternation and
  their ranges overlap. With the experimental firm-slope solver switched on,
  2.34.1 takes 2.2 s at 4 processors and 1.2 s at 16.
- `marta_group_individual_3m` (`group()` and `individual()`) takes the same
  route in both versions, with identical results: 7.7 s against 8.3 s at 4
  processors. At 16 processors, ten alternated pairs give a median ratio of
  1.04, with 2.34.1 slower in six of them, which is within the noise of the
  server.

## 2.34.1 against 2.34.0 (CUDA libraries on demand)

- Fifteen public datasets, Stata plugin and CPU module, 24 threads: results
  identical bit for bit. The CPU module is the same binary.
- Three large panels (`main_95_21_ready`, `akm_v02_firstreg`,
  `simulated_panel`), `xhdfe-fast` and `reghdfe-comparable`: on CPU in Stata,
  coefficients, standard errors and iterations identical bit for bit; on the
  H100 in Stata and Python, coefficients within 1.1e-12 and variances within
  1.2e-10, with real GPU use asserted in every run.
- The plugins and the hybrid module list no cuBLAS or cuSPARSE dependency and
  load without the NVIDIA libraries on the loader path. With the libraries
  hidden, the first use of the network route fails with an error and leaves no
  estimates, later attempts fail at once, and a CPU fit in the same process
  runs.
- `xfepout` on CPU and on the H100; the R package built from a clean copy,
  with one CPU and one CUDA fit agreeing with Python to 2.3e-14; CTest 25 of
  25 on the CPU build and 27 of 27 on the CUDA build.

## Offline analytic battery

The main matrix of the permanent battery (`tests/analytic_precision/`) ran on
the binaries of this release: 887 xhdfe rows across the Stata and native
interfaces, on CPU and on CUDA.

- 664 rows pass (371 on CPU, 293 on CUDA) and none fails.
- 223 rows are the refusals the catalogue expects (90 on CPU, 133 on CUDA):
  explicit methods or backends that a design does not support, invalid
  `strict-residual` requests with slopes, and the documented restriction of
  `savefe` with `group()` and `individual()`. They are not passes.
- The 25 explicit-dummy OLS rows pass. The `reghdfe` rows of the battery are
  graded separately and are not part of this result.
- The four permanent CPU contracts added on 19 November (projection,
  weighted `savefe` with four effects, weighted `savefe` with instruments and
  the Strict iteration budget) pass.

## Help files and release checks

- The 21 example blocks of `help xhdfe_examples` run on this package with
  public Stata Press data, including the CUDA block.
- The options, changed defaults, stored-result names, cross-references and
  version lines of the nine Stata help files were checked against the ado
  files; the "Known limitations" of `help xhdfe` match the release notes.
- The file lists of the net-install site and of the platform bundles were
  corrected to carry `xhdfe_examples.sthlp`, and the site validator now
  compares the staged package with the package file of the repository.
- The source archive that `xhdfegpu` downloads, built from a clean export,
  carries the sources added since 2.31.1 and passes its validator.
- `tests/release_lint.sh` and `tools/validate_release_metadata.py` pass.

## Builds on other platforms

A trial run of the release workflow's build-and-test matrix on the estimator
code of this release, in the development repository, built and tested the
Windows Python wheel, the Windows plugins in a Stata installation layout,
macOS arm64 and x86_64 and the universal macOS binaries. Its Linux job did not
complete: the build container lacked the cuBLAS and cuSPARSE headers that
`src/network_cuda.cu` includes since 2.34.0, which the release build script
now installs. The matrix runs of the public repository for the preflight and
version tags of this release, which build the published binaries on all three
platforms, are recorded on the release page.

## Not executed for this release

- The full `core24` matrix on the eight execution variants. Python, the
  `xhdfe-fast` mode and GPU were not timed against 2.31.1 today, beyond the
  three large panels above.
- The R matrix, the supplemental cases and the audit-mode campaign of the
  offline analytic battery.
- Windows: the nearly-perfect-fit limitation was not re-examined; it is as in
  2.31.1.
- Native Stata/MP hosts on macOS.
