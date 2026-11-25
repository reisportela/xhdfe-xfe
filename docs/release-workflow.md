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
   the full version here is `2.28.1.20260929`, including the date.
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

## SSC submission

Use the authenticated version-tag CI assets. The SSC ZIP takes `xhdfe.pkg`,
`xfepout.pkg`, `stata.toc` and the union of the files declared in both package
descriptors from `xhdfe_xfe-stata-netinstall-site.zip`. All members are at the
ZIP root and retain their source names and bytes. They include the runtimes,
notices, licences and the two `.do` examples. CI checks the catalogue, both
package versions, the complete file lists and the plug-in mappings. Verify
`xhdfe-ssc-<full-version>.zip` against `SHA256SUMS.txt`.

To reproduce the assembly, run the builder from the repository root with the
authenticated site ZIP and a new output path. Set `EXPECTED_VERSION` to the
dated package version of the new tag, omitting the leading `v`.
[SSC ZIP builder](../tools/build_stata_ssc_zip.py)

```bash
python3 -B tools/build_stata_ssc_zip.py \
  release_assets/xhdfe_xfe-stata-netinstall-site.zip "$EXPECTED_VERSION" \
  "release_assets/xhdfe-ssc-${EXPECTED_VERSION}.zip"
```

The `xhdfe` descriptor maps both plug-ins. The `xfepout` descriptor maps its
own plug-in and carries its own version. Each `g <selector> <source> <alias>`
mapping installs one of `xhdfe.plugin` and `xfepout.plugin`.

| Stata selectors | xhdfe source | xfepout source |
| --- | --- | --- |
| `LINUX64`, `LINUX64P` | `xhdfe.linux64.plugin` | `xfepout.linux64.plugin` |
| `MACARM64`, `OSX.ARM64`, `MACINTEL64`, `OSX.X8664` | `xhdfe.macos-universal.plugin` | `xfepout.macos-universal.plugin` |
| `WIN64` | `xhdfe.win64.plugin` | `xfepout.win64.plugin` |

Each descriptor requires its mapped aliases through `h xhdfe.plugin` or
`h xfepout.plugin` entries as applicable. Both descriptors use
`G <selector> xhdfe_libomp.dylib xhdfe_libomp.dylib` for each macOS selector.
Any Windows runtime DLL declared in the CI manifest uses an uppercase
`G WIN64` mapping with the same source and destination name.
[Stata package directives](https://www.stata.com/manuals/rnet.pdf)

Extract the ZIP and test each package in a fresh isolated PLUS. Start a new
Stata session in a separate working directory so Stata resolves the installed
ado-files and plug-ins. Use the extracted root as the installation source.

```stata
net install xhdfe, from("/path/to/unpacked-directory") replace
net install xfepout, from("/path/to/unpacked-directory") replace
```

Repeat the install in an isolated PLUS seeded with older plug-in aliases. Check
that both aliases resolve beside the installed `.ado` files and match their
platform members by SHA-256. Run the regression and
partialling-out check on the official Grunfeld data.

```stata
use "https://www.stata-press.com/data/r19/grunfeld.dta", clear
xhdfe invest mvalue kstock, absorb(company year) vce(cluster company)
xfepout invest mvalue kstock, absorb(company year) generate(within_)
```

Use `net get xhdfe, from("/path/to/unpacked-directory")` to download the examples.
The two example do-files use lowercase `f` in `xhdfe.pkg`. The installer
helpers use uppercase `F` to install them alongside `xhdfegpu.ado` during
ordinary `net install`. Their sources are directly in `stata/`.
Then run `do xhdfe_example_cluster.do` and `do xhdfe_example_partial_out.do` from
the download directory. Send Kit Baum the single ZIP or its versioned release URL,
with the archive SHA-256. After SSC publishes the update, check the live
descriptor, repeat the isolated install with `ssc install xhdfe` and verify
the aliases and both examples again.

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
