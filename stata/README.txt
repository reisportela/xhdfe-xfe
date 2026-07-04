xhdfe / xfe Stata net-install site

Install from Stata with:

  net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
  net install xfe,   from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace

The package manifests use Stata's platform-specific g lines:
LINUX64/LINUX64P, MACARM64/OSX.ARM64, MACINTEL64/OSX.X8664, and WIN64 when
the corresponding release binary was built.  Each platform-specific server
file is installed under the canonical runtime name xhdfe.plugin or xfe.plugin.
