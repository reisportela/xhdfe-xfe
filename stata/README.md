# xhdfe (Stata wrapper)

This folder is self-contained: it includes the Stata wrapper (`xhdfe.ado`, `xhdfe.sthlp`), the C++ plugin source, and all dependencies needed to build the plugin.

## Build the plugin

From this folder, run:

```bash
bash tools/build-plugin.sh --openmp
bash tools/build-xfepout-plugin.sh --openmp
```

This produces `xhdfe.plugin` and `xfepout.plugin` in the same directory.
Production builds require OpenMP. Windows builds link the GNU/OpenMP runtimes
statically; macOS builds need a compatible OpenMP runtime for each architecture.

To build with CUDA support (GPU backend), set `XHDFE_ENABLE_CUDA=ON`, ensure `nvcc`
is on PATH, and target your GPU's compute capability — read it with
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` and drop the dot
(`9.0` → `90`, `8.6` → `86`; minimum `75`):

```bash
# example: compute capability 9.0 (H100 -> sm_90); use your own value
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash tools/build-plugin.sh --linux --openmp
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash tools/build-xfepout-plugin.sh --linux --openmp

# Multi-architecture fatbin for redistribution
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCHS="75,80,86,89,90" bash tools/build-plugin.sh --linux --openmp
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCHS="75,80,86,89,90" bash tools/build-xfepout-plugin.sh --linux --openmp
```

To confirm Stata is actually using CUDA instead of silently falling back to CPU:

```stata
discard
which xhdfe
which xfepout

webuse nlswork, clear
xhdfe ln_wage ttl_exp tenure, absorb(idcode year) gpubackend(cuda) numthreads(8) tolerance(1e-8)
di e(gpu_used)
di "`e(gpu_backend)'"

webuse nlswork, clear
xfepout ln_wage ttl_exp tenure, absorb(idcode year) gpubackend(cuda) numthreads(8) tolerance(1e-8) clear
di e(gpu_used)
di "`e(gpu_backend)'"
```

Expected result: `e(gpu_used) == 1` and `e(gpu_backend) == "cuda"`.
If `gpubackend(cuda)` was requested but CUDA is unavailable, the commands now stop with an error instead of silently
returning CPU output.
After rebuilding or switching plugin binaries, restart Stata to ensure the new
native code is loaded. `discard` takes no command-name argument. Ordinary
repeated calls do not require a restart when the plugin binary is unchanged.

For a more detailed local build checklist, see `BUILD_CUDA.md` in this folder.

## Install / use in Stata

For a released online install from `xhdfe-xfe`, use the generated Stata
net-install site:

```stata
net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
net install xfepout,   from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
```

`xfepout` replaces the former `xfe` command. Existing standalone `xfe`
installations can be removed with `ado uninstall xfe` before installing
`xfepout`; no compatibility alias is shipped.

That site uses Stata platform-specific `g` lines for plugins, and uppercase
`G` lines for external platform runtimes, so Linux, macOS, and Windows
users receive the matching CPU plugin binary when it exists in the release.
The Windows plugins are self-contained with respect to GNU/OpenMP runtimes;
the loader does not depend on DLLs installed under Stata's `plus/l` directory.
Linux CUDA fatbin plugins are distributed as separate release assets; install
one of those bundles explicitly or build locally with `XHDFE_ENABLE_CUDA=ON`
as shown above. For a local development checkout or an unzipped release
bundle, point `net install` at the folder containing `stata.toc` and the
`.pkg` files:

```stata
net install xhdfe, from("/path/to/xhdfe/stata") replace
net install xfepout,   from("/path/to/xhdfe/stata") replace
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
