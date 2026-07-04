# xhdfe (Stata wrapper)

This folder is self-contained: it includes the Stata wrapper (`xhdfe.ado`, `xhdfe.sthlp`), the C++ plugin source, and all dependencies needed to build the plugin.

## Build the plugin

From this folder, run:

```bash
bash tools/build-plugin.sh
```

This produces `xhdfe.plugin` in the same directory.

To build with CUDA support (GPU backend), set `XHDFE_ENABLE_CUDA=1` and ensure `nvcc` is on PATH:

```bash
# This workstation (H100 / sm_90)
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash tools/build-plugin.sh --linux --openmp
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash tools/build-xfe-plugin.sh --linux --openmp

# Multi-architecture fatbin for redistribution
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCHS="75,80,86,89,90" bash tools/build-plugin.sh --linux --openmp
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCHS="75,80,86,89,90" bash tools/build-xfe-plugin.sh --linux --openmp
```

To confirm Stata is actually using CUDA instead of silently falling back to CPU:

```stata
discard
which xhdfe
which xfe

webuse nlswork, clear
xhdfe ln_wage ttl_exp tenure, absorb(idcode year) gpubackend(cuda) numthreads(8) tolerance(1e-8)
di e(gpu_used)
di "`e(gpu_backend)'"

webuse nlswork, clear
xfe ln_wage ttl_exp tenure, absorb(idcode year) gpubackend(cuda) numthreads(8) tolerance(1e-8) clear
di e(gpu_used)
di "`e(gpu_backend)'"
```

Expected result: `e(gpu_used) == 1` and `e(gpu_backend) == "cuda"`.
If `gpubackend(cuda)` was requested but CUDA is unavailable, the commands now stop with an error instead of silently
returning CPU output.
If you rebuilt or switched plugin binaries in the same Stata session, run `discard` with no arguments before rerunning the command. Do not use `discard xhdfe` or `discard xfe`. Ordinary repeated `xhdfe`/`xfe` calls in the same session do not require `discard` when the plugin binary itself has not changed.

For a more detailed local build checklist, see `BUILD_CUDA.md` in this folder.

## Install / use in Stata

For a released online install from `xhdfe-xfe`, use the generated Stata
net-install site:

```stata
net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
net install xfe,   from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
```

That site uses Stata platform-specific `g` lines so Linux, macOS, and Windows
users receive the matching plugin binary when it exists in the release. For a
local development checkout or an unzipped release bundle, point `net install`
at the folder containing `stata.toc` and the `.pkg` files:

```stata
net install xhdfe, from("/path/to/xhdfe/stata") replace
net install xfe,   from("/path/to/xhdfe/stata") replace
```

Alternatively, make sure this folder is on your Stata `adopath` (or copy
`xhdfe.ado`, `xhdfe.sthlp`, and `xhdfe.plugin` to your personal ado folder).

Example (reghdfe-style):

```stata
webuse nlswork, clear
xhdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year) vce(robust)
xhdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year) vce(cluster idcode)
xhdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year, savefe) vce(cluster idcode)
```

For the full option set (savefe/savefes, fetolerance/ferecoverymethod, caching/mobility profiles,
and GPU backend selection), see `help xhdfe` in Stata.
