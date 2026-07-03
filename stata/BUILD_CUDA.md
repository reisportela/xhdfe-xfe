# Stata CUDA Build Instructions

This note is for the Stata plugins in this folder: `xhdfe.plugin` and `xfe.plugin`.

On this workstation the GPU is an `H100`, so the local CUDA target should be `sm_90`.

## Build commands

From the repository root:

```bash
cd stata/tools

# Local H100 build
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash build-plugin.sh --linux --openmp
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash build-xfe-plugin.sh --linux --openmp

# Multi-architecture distribution build
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCHS="75,80,86,89,90" bash build-plugin.sh --linux --openmp
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCHS="75,80,86,89,90" bash build-xfe-plugin.sh --linux --openmp
```

Outputs:

- `stata/xhdfe.plugin`
- `stata/xfe.plugin`

## Stata-side verification

Always check which ado/plugin is being used:

```stata
discard
adopath ++ "/PATH/TO/xhdfe/stata"
which xhdfe
which xfe
```

`which` should point to this repository's `stata/` folder.

If you rebuilt or switched plugin binaries in the same Stata session, run:

```stata
discard
```

before rerunning `xhdfe` or `xfe`.

Then verify CUDA usage:

```stata
webuse nlswork, clear
xhdfe ln_wage ttl_exp tenure, absorb(idcode year) gpubackend(cuda) numthreads(8) tolerance(1e-8)
di e(gpu_used)
di "`e(gpu_backend)'"
di "`e(gpu_backend_requested)'"

webuse nlswork, clear
xfe ln_wage ttl_exp tenure, absorb(idcode year) gpubackend(cuda) numthreads(8) tolerance(1e-8) clear
di e(gpu_used)
di "`e(gpu_backend)'"
di "`e(gpu_backend_requested)'"
```

Expected result:

- `e(gpu_used) == 1`
- `e(gpu_backend) == "cuda"`
- `e(gpu_backend_requested) == "cuda"`

If `gpubackend(cuda)` was requested but CUDA is unavailable, `xhdfe`/`xfe` now stop with an error instead of
silently returning CPU output.

Typical causes:

- the plugin was rebuilt without `XHDFE_ENABLE_CUDA=ON`
- the wrong plugin is being picked up on the `adopath`
- Stata is still holding an older plugin in memory after a rebuild or plugin-path switch; run `discard` with no arguments
- CUDA is unavailable at runtime

## Notes

- `xhdfe` and `xfe` have separate plugin binaries; rebuild both if you want both commands to use CUDA.
- For local H100-only use, prefer `XHDFE_CUDA_ARCH=90`.
- For shared distribution bundles, prefer `XHDFE_CUDA_ARCHS="75,80,86,89,90"`.
