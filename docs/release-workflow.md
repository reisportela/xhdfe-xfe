# xhdfe-xfe release workflow

Development happens in `xhdfe`.  This public repository is synchronized only
when a version is ready to publish.

## Release sequence

1. Validate the candidate in the private repository and freeze its sources and
   local artifact hashes. Preserve the two tolerance modes and record numerical
   exceptions and measured performance costs.
2. Synchronize production sources, package metadata, help, tests and release
   tooling into the public repository. Confirm exact text/source identity;
   private local CUDA binaries and public CI binaries have different roles.
3. Push `preflight-v<full-version>` (or `-rN` for a new attempt). It runs all
   platform gates and assembles the release without publishing. For example,
   the full version here is `2.28.0.20260924`, including the date.
4. Push `v<full-version>` at the reviewed commit. This builds and stages a
   **draft** release. Download and authenticate the artifacts from that exact
   version-tag run. Preflight or local binaries cannot substitute for them.
5. Exercise the exact Linux CUDA plugins on the maintainer's H100, checking
   actual GPU use and numerical results. Check Linux Stata loading, the
   packaged source closure and the intended net-install snapshot as well.
6. Push `publish-v<full-version>` at the same commit only after validation.
   This publishes the existing draft and its existing net-install snapshot;
   it does not rebuild the assets. Verify a fresh public install and reconcile
   both canonical local checkouts before declaring the release aligned.

OpenMP is mandatory for every native release target. Linux plugins use the
portable toolchain recorded in their provenance. Windows Stata plugins embed
the GNU/OpenMP runtimes: the native gate loads them with ordinary `LoadLibrary`
from Stata-style letter directories, outside the executable directory and
without compiler paths, and checks two real estimator workers. Python wheels
retain their separately validated DLL closure. macOS universal plugins package
a compatible runtime and are tested natively on both Intel and Apple Silicon.

The online Stata site is:

```stata
net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
net install xfepout, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
```

Restart Stata after replacing native plugins. Generated `.pkg` files use
platform-specific lowercase `g` entries for plugins; external runtime entries,
when needed, use uppercase `G`. Self-contained Windows Stata plugins do not
need DLL entries. CPU plugins are the net-install default; Linux CUDA bundles
are separate assets. See the version's validation record for numerical limits.

## Autonomous source and offline bundles

The release workflow publishes `xhdfe-src.zip` and
`xhdfe_xfe-offline-source-and-platform-binaries.zip`. The source-only archive
is also copied to the Stata net-install site for `xhdfegpu`. The larger bundle
is assembled from the exact tagged Git tree and adds release package artifacts
and platform-specific Stata binaries. Both contain the shared C++ core, all
three frontend sources, Stata plugin build inputs, Eigen, pybind11, and the
unmodified official `Rcpp_1.1.2.tar.gz` CRAN source archive. The Rcpp upstream
URL, license, version, and certified SHA-256 are recorded in
`third_party/RCPP_SOURCE_PROVENANCE.md`.

`tools/make_source_dist_zip.sh` builds `xhdfe-src.zip` and invokes
`tools/validate_source_dist_zip.sh`. The validator checks ZIP integrity,
required dependency entries, the Rcpp SHA-256, and the package name and version
inside the Rcpp tarball.
