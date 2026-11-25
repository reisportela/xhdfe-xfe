# xhdfe 2.35.1.20261125 and xfepout 1.14.7

- `xhdfegpu` defaults to a hybrid build with a portable CPU target in a
  clean environment. Set `cputarget(native)` to tune the CPU instructions to
  the build host. CUDA architecture is selected independently with `arch()`.
- `xhdfegpu, status` shows the resolved plugin paths and the build metadata
  returned by each module. Restart Stata after replacement to associate a
  query with the file currently in the installation directory. Older modules
  can lack build metadata.
- The Linux installer stages the active plugins and checks CUDA execution
  before replacement. The package includes the two installer helpers as
  runtime files.

The published Stata performance measurements in the README refer to
xhdfe 2.35.0.20261125. This update changes the installer and adds native build
metadata. The regression specifications and numerical tolerances used in the
historical comparison are unchanged.

[Validation and scope](VALIDATION_2.35.1.20261125.md)
