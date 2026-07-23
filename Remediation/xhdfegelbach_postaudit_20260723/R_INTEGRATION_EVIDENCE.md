# R integration evidence

Date: 23 July 2026

## Dependency provenance

The Release builds resolved `Rcpp` from `r/Rlib/Rcpp`, not from the user's
default R library. That directory is a byte-for-byte copy of the already
installed, repo-local `benchmarks/_r_lib_latest/Rcpp`. It is Rcpp
`1.1.1-1.1`, built for the active R `4.6.0` runtime and
`x86_64-redhat-linux-gnu` target. Full hashes and the source/destination
comparison are recorded in `RCPP_PROVENANCE.md`.

This is an ignored local validation library, not a file tracked in an ordinary
public clone. Documentation must therefore qualify the offline command with
"when the validation bundle/check-out includes `r/Rlib/Rcpp`"; it must not
claim that every public source clone contains the directory.

The full test suite used testthat `3.3.2` and its test-only dependencies from
the existing read-only local library
`/home/mangelo/R/x86_64-pc-linux-gnu-library/4.3`, because testthat is not
present in `r/Rlib`. Library precedence remained:

1. the R build under test (`r/Rlib` or `r/Rlib_cuda`);
2. repo-local `r/Rlib`, including Rcpp;
3. the read-only test-dependency library.

Every build and test set `R_PROFILE_USER=/dev/null`.

## Release builds

CPU:

```text
R_PROFILE_USER=/dev/null
R_LIBS_USER=<repo>/r/Rlib
XHDFE_ENABLE_CUDA=OFF
XHDFE_MARCH=native
R CMD INSTALL --preclean --clean --library=<repo>/r/Rlib r/xhdfe
```

CUDA:

```text
R_PROFILE_USER=/dev/null
R_LIBS_USER=<repo>/r/Rlib_cuda:<repo>/r/Rlib
XHDFE_ENABLE_CUDA=ON
XHDFE_CUDA_ARCH=90
XHDFE_MARCH=native
R CMD INSTALL --preclean --clean --library=<repo>/r/Rlib_cuda r/xhdfe
```

Both builds installed package version `2.20.0.20260723`, compiled with
`-O3 -DNDEBUG -march=native` and OpenMP, and resolved the Rcpp headers from
`<repo>/r/Rlib/Rcpp/include`.

| Artifact | SHA256 | `__assert_fail` | OpenMP | CUDA image |
|---|---|---:|---|---|
| `artifacts/r_cpu/xhdfe.so` | `b934a650013677261cbbcc7ac10303a4def184452e2bd6fee43113bed1563604` | 0 | `libgomp.so.1` | not compiled |
| `artifacts/r_cuda/xhdfe.so` | `6448036e9a14c0c7d59f08fa41422330faa6da56aa11e1d6848994a047584446` | 0 | `libgomp.so.1` | two `sm_90` cubins, no other cubin |

The installed `DESCRIPTION` files are preserved beside each shared object.
The CUDA build also links `libcudart.so.12`.

Build logs:

- `r_install_release_cpu_v220_final.log`
- `r_install_release_cuda_sm90_final_rerun.log`
- `r_cuda_cuobjdump_list_elf.log`

## Tests

The final CPU Gelbach suite passed with no failures:

```text
r_test_gelbach_cpu_final_rerun.log
```

The final complete CPU suite passed with zero failures. It contains one
expected skip for the real-GPU interface test because this is the CPU-only
build, and three existing AKM weak-identification warnings:

```text
r_test_full_cpu_v220_final.log
```

The complete CUDA suite passed with zero failures and no skip; the real-GPU
interface test executed on the H100:

```text
r_test_full_cuda_sm90_v220_final.log
```

No tolerance or convergence criterion was changed.

## Gelbach CUDA public-contract smoke

`r_gpu_gelbach_smoke.R` runs the public `xhdfe_gelbach()` interface with two
fixed-effect dimensions, a clustered VCE, and 100,000 observations.

On the H100:

```text
gpu_requested=TRUE
gpu_used=TRUE
backend=cuda
status=used
attempted=TRUE
converged=TRUE
iterations=3
threads=8
```

CPU/CUDA maximum absolute differences were:

```text
b_full       9.992007221626409e-16
delta        3.330669073875470e-16
cov          1.016439536705160e-20
total_cov    1.185846126156021e-20
```

The scoped request restored the pre-call `XHDFE_GPU_BACKEND=cpu`.

With `CUDA_VISIBLE_DEVICES=` the same public option returned normally and
reported truthful fallback:

```text
gpu_requested=TRUE
gpu_used=FALSE
backend=cpu
status=unavailable
status_code=2
attempted=FALSE
converged=TRUE
iterations=0
```

Logs:

- `r_gpu_gelbach_smoke_used.log`
- `r_gpu_gelbach_smoke_fallback.log`

## Reproduced integration-test defects

Two failures were in newly added tests, not in the estimator:

1. The independent gamma check called `lm.fit()` without its required `y`
   argument. Supplying the already defined outcome made the intended oracle
   executable.
2. The exact-zero absorbed-target share assertion selected total rows for both
   an absorbed target and a separate identified focal. Restricting the
   assertion to `absorbed_target_names` verified the intended invariant:
   the absorbed target's `share = "base"` total-row SE is numeric zero exactly.

No production behavior was changed to accommodate either test.

An interrupted stale `r/Rlib_cuda/00LOCK-xhdfe` was moved, rather than
deleted, to `stale_00LOCK_xhdfe_interrupted_cuda_build/`.
