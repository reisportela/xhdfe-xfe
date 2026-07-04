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
macOS.  It also publishes an online Stata net-install site to:

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
