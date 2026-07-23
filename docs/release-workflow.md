# xhdfe-xfe release workflow

Development happens in `xhdfe`.  This public repository is synchronized only
when a version is ready to publish.

## Release sequence

1. Validate the release in `xhdfe`.
2. Sync the release-owned source, package, help, CI, and documentation files
   into `xhdfe-xfe`.
3. Commit and push `xhdfe-xfe` on `main`.
4. Create or publish the GitHub release.

The release workflow builds native Stata plugin bundles for Linux, Windows, and
macOS. Linux release plugins are compiled inside a `manylinux2014` container
with static GNU C++ runtime libraries so the online install remains compatible
with older enterprise Linux systems. It also publishes an online Stata
net-install site to:

```text
https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata
```

Users can install from Stata with:

```stata
net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
net install xfe,   from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
```

The generated `.pkg` files use Stata `g` lines so Stata downloads the plugin
matching the user's OS.

## Autonomous source and offline bundles

The release workflow also publishes `xhdfe-src.zip` and
`xhdfe-offline-bundle.zip`. These archives contain the shared C++ core, all
three frontend sources, Stata plugin build inputs, Eigen, pybind11, and the
unmodified official `Rcpp_1.1.2.tar.gz` CRAN source archive. The latter lets an
R user install the package into a local library with networking disabled; its
upstream URL, license, version, and certified SHA-256 are recorded in
`third_party/RCPP_SOURCE_PROVENANCE.md`.

`tools/make_source_dist_zip.sh` fails if the resulting archive is not closed.
Its independent `tools/validate_source_dist_zip.sh` check verifies ZIP
integrity, required dependency entries, the Rcpp SHA-256, and the package name
and version inside the Rcpp tarball. The release workflow runs that validator
again and confirms that both the archive and provenance survive nesting inside
the autonomous offline bundle.
