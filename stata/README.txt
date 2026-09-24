xhdfe / xfepout Stata net-install site

Install from Stata with:

  net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
  net install xfepout,   from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace

The package manifests use Stata's platform-specific g lines:
LINUX64/LINUX64P, MACARM64/OSX.ARM64, MACINTEL64/OSX.X8664, and WIN64 when
the corresponding release binary was built.  Each platform-specific server
file is installed under the canonical runtime name xhdfe.plugin or xfepout.plugin.
Windows plugins embed their GNU/OpenMP runtimes and need no external runtime DLLs.
windows-stata-provider-ledger.json records the static link inputs and plugin hashes;
windows-stata-runtime-ledger.json records their system-only PE dependencies.
Every package installs the exact GNU/MinGW and Eigen license texts. When the
two NVIDIA license inputs are supplied, their CUDA 12.6/CCCL 2.5.0 materials
are also listed in both package manifests.
