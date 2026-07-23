# Repo-local Rcpp provenance

This record applies to the initially certified in-tree R binaries. Portable
offline release closure was subsequently validated from the official Rcpp
1.1.2 source archive; see `RCPP_OFFLINE_SOURCE_VALIDATION.md`. The records
remain separate so that the dependency of each artifact is explicit.

- Destination used for the post-audit R build: `r/Rlib/Rcpp`
- Source: `benchmarks/_r_lib_latest/Rcpp` in the same private `xhdfe`
  checkout; no download or external installation was used.
- Package: `Rcpp` version `1.1.1-1.1`
- Source package metadata: CRAN, publication `2026-04-24`
- Installed build metadata: `R 4.6.0`, `x86_64-redhat-linux-gnu`,
  built `2026-06-15 10:01:35 UTC`
- Current runtime used for certification: `R 4.6.0`,
  `x86_64-redhat-linux-gnu`
- `DESCRIPTION` SHA256:
  `9d523ae0e1b185d5eb28f7333219b7d13e1ae65a7d42c7a986f16f4ca5fedabf`
- `libs/Rcpp.so` SHA256:
  `e9238289491d0ac124a59df840148da8a772e14578c27c00cda92686d123b677`
- `include/Rcpp.h` SHA256:
  `7e5978de43a37d44cfc05bbdd670a652e5df83a2b8a112f46356f3201f0acdc0`

The source and destination `DESCRIPTION` and `libs/Rcpp.so` files were
verified byte-identical with `cmp`. This installed package is technically
appropriate for the local Release build because its R runtime and target
architecture exactly match the active runtime. The build was invoked with
`R_PROFILE_USER=/dev/null` and `R_LIBS_USER` pointing first to `r/Rlib`, so
the Rcpp headers and runtime library did not come from the user's default R
library.
