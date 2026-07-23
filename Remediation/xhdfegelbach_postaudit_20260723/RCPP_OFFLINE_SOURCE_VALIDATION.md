# Rcpp offline source validation

Date: 23 July 2026

## Source provenance

- Official source:
  `https://cran.r-project.org/src/contrib/Rcpp_1.1.2.tar.gz`
- Version: Rcpp 1.1.2
- CRAN publication date: 5 July 2026
- Stored release input: `third_party/Rcpp_1.1.2.tar.gz`
- SHA-256:
  `2746cf2fb188e5f0a84dbf5c8f68915b54564ed33e5754572f174e7b32e7f4f3`
- License and provenance record:
  `third_party/RCPP_SOURCE_PROVENANCE.md`

The archive was downloaded once from CRAN. Every installation and test below
then used the local archive and a new library under `/tmp`; no package manager
or network dependency was used.

## Offline installation

The following sequence completed successfully:

```bash
R_PROFILE_USER=/dev/null \
R_LIBS_USER=/tmp/xhdfe_rcpp112_validation_20260723/lib \
R CMD INSTALL \
  --library=/tmp/xhdfe_rcpp112_validation_20260723/lib \
  third_party/Rcpp_1.1.2.tar.gz
```

The install reported package `Rcpp` version `1.1.2`, verified the upstream MD5
manifest, built its native library, and passed its temporary/final load checks.

Both xhdfe builds then resolved
`/tmp/xhdfe_rcpp112_validation_20260723/lib/Rcpp/include`:

```bash
# CPU
R_PROFILE_USER=/dev/null \
R_LIBS_USER=/tmp/xhdfe_rcpp112_validation_20260723/lib \
XHDFE_ENABLE_CUDA=OFF XHDFE_MARCH=native \
R CMD INSTALL --preclean --clean \
  --library=/tmp/xhdfe_rcpp112_validation_20260723/lib r/xhdfe

# CUDA H100
R_PROFILE_USER=/dev/null \
R_LIBS_USER=/tmp/xhdfe_rcpp112_validation_20260723/lib_cuda:/tmp/xhdfe_rcpp112_validation_20260723/lib \
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 XHDFE_MARCH=native \
R CMD INSTALL --preclean --clean \
  --library=/tmp/xhdfe_rcpp112_validation_20260723/lib_cuda r/xhdfe
```

## Artifact hygiene

| Build | SHA-256 | `U __assert_fail` | OpenMP | CUDA image |
|---|---|---:|---|---|
| CPU | `f36f0a8125288b42aec9cf9fb6eca5cd4415647920a6ee5e812a733e3a43c1cd` | 0 | `libgomp.so.1` | none |
| CUDA | `bc7830fda343e1dcf726addfb6009084ba1770fdc20370626a52ceb53d728675` | 0 | `libgomp.so.1` | exactly two `sm_90` cubins |

The CUDA artifact also linked `libcudart.so.12`.

## Recertification

- `r/xhdfe/tests/testthat/test-gelbach.R`: all expectations passed on the CPU
  build and all expectations passed on the CUDA build.
- H100 public-interface smoke:
  `gpu_requested=TRUE`, `gpu_used=TRUE`, backend `cuda`, status `used`,
  attempted/converged true, three absorption iterations, eight threads.
- CPU/CUDA maximum absolute differences in that smoke:
  `b_full=6.66e-16`, `delta=2.50e-16`, `cov=6.78e-21`,
  `total_cov=6.78e-21`.
- With `CUDA_VISIBLE_DEVICES=`:
  `gpu_requested=TRUE`, `gpu_used=FALSE`, backend `cpu`,
  status `unavailable`, code 2, attempted false, converged true.

This closes portable Rcpp source provenance and the tested network-disabled
R installation route. It does not claim that the generated local R library is
portable; only the upstream source archive is distributed.
