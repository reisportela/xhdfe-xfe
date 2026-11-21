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

## Published artifacts

The release was published on 21 November 2026 at 17:50:39 UTC.
[Preflight run 44923004907](https://github.com/reisportela/xhdfe-xfe/actions/runs/44923004907)
and [version-tag run 44927844827](https://github.com/reisportela/xhdfe-xfe/actions/runs/44927844827)
passed on Linux, Windows and macOS, including the assembly of the release
assets, and [publication run 44932190374](https://github.com/reisportela/xhdfe-xfe/actions/runs/44932190374)
published the draft of the version-tag run and its net-install site.

Before publication, the exact assets of that draft were checked on a Linux
host with an NVIDIA H100:

- All thirteen assets match `SHA256SUMS.txt` and their GitHub digests; the
  five that were executed were also hashed locally.
- The Linux CPU and CUDA Stata plugins ran in licensed Stata from an adopath
  holding only the unpacked bundle, without NVIDIA libraries on the loader
  path: ordinary fits in both tolerance modes against a closed-form oracle,
  the changes since 2.31.1 (the reduced-system route and `reducedpcg(off)`,
  `savefe`, an absorbed regressor with slopes, a weighted fit), `xfepout` and
  an eight-million-row CPU smoke.
- On the 46-million-row panel `main_95_21_ready` at 16 processors, the CPU
  plugin reproduces the result of the development build (coefficients within
  2.1e-16, standard errors within 1.1e-15, same sample, route and iterations)
  in 28.2 s. It is the portable build; the development build, compiled for
  the host's instruction set, takes 21 to 22 s. The CUDA plugin, with real GPU
  use asserted, agrees with the CPU plugin to 8.7e-18 and 8.1e-15 and takes
  8.9 s.
- The CUDA plugin lists no cuBLAS or cuSPARSE dependency. The optional
  firm-slope route ran with the NVIDIA libraries on the loader path
  (coefficients within 3.3e-13 and standard errors within 1.8e-9 of the
  default route) and, without them, stopped with the documented error and
  left no estimates.
- The manylinux wheel, installed in a new environment, reproduces the Stata
  result.

After publication, a fresh `net install` of `xhdfe` and `xfepout` from the
public address into an empty folder matched the published site byte for byte
in its 38 installed and ancillary files, `xhdfe_examples.sthlp` among them,
and the installed package passed the ordinary fit.

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
  results: 34.4 s against 39.9 s at 16 processors. At 4 processors, five
  alternated pairs give medians of 75.6 s and 79.0 s and a median ratio of
  0.97, with 2.34.1 slower in two of them. With the experimental firm-slope
  solver switched on, 2.34.1 takes 2.2 s at 4 processors and 1.2 s at 16.
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

- The 21 code blocks of the 20 examples of `help xhdfe_examples` run on this
  package with public Stata Press data, including the CUDA block.
- The options, changed defaults, stored-result names, cross-references and
  version lines of the nine Stata help files were checked against the ado
  files; the "Known limitations" of `help xhdfe` match the release notes.
- The file lists of the net-install site and of the platform bundles were
  corrected to carry `xhdfe_examples.sthlp`, and the site validator now
  compares the staged package with the package file of the repository.
- The source archive that `xhdfegpu` downloads, built from a clean export,
  carries the sources added since 2.31.1 and passes its validator. The CUDA
  plugins built from that archive with its own scripts (`sm_90`, OpenMP) load
  from an isolated adopath: on a public dataset, a GPU fit with real GPU use
  asserted agrees with the CPU fit to 3.3e-14 in the coefficients and 1.6e-15
  in the standard errors, and `xfepout` runs.
- The R package, built and checked from the committed tree with the commands
  of the release workflow (`R CMD build` and `R CMD check`, R 4.6.0, CPU): no
  error, one warning (the source-file extensions, as in 2.31.1) and one note
  (the compiled code writes to the standard output and error streams). Its
  tests give 1,192 passes and no failure; 8 are skipped for inputs or builds
  that check does not have.
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
platforms, are listed under "Published artifacts" above.

## Not executed for this release

- The full `core24` matrix on the eight execution variants. Python, the
  `xhdfe-fast` mode and GPU were not timed against 2.31.1 today, beyond the
  three large panels above.
- The R matrix, the supplemental cases and the audit-mode campaign of the
  offline analytic battery.
- Windows: the nearly-perfect-fit limitation was not re-examined; it is as in
  2.31.1.
- Native Stata/MP hosts on macOS.
