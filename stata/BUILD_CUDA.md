# Stata CUDA Build Instructions

This note is for the Stata plugins in this folder: `xhdfe.plugin` and `xfe.plugin`.
For the cross-language walkthrough (Stata, Python, and R), see
[`../docs/gpu.md`](../docs/gpu.md).

GPU support is not available from the online net-install or the release ZIPs
(those ship CPU-only plugins). It must be built from source, as below.

The recommended build uses `--cuda auto`, which detects the local NVIDIA GPU
architecture with `nvidia-smi`. To inspect the value yourself:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. 9.0, 8.6, 7.5
```

Drop the dot to get `XHDFE_CUDA_ARCH` (`9.0` → `90`, `8.6` → `86`; minimum `75`).
You only need this manual value when passing an explicit `--cuda 90` target.

## Build commands

From the repository root:

```bash
cd stata/tools

# Local build: auto-detect the GPU architecture
bash build-plugin.sh --linux --openmp --cuda auto
bash build-xfe-plugin.sh --linux --openmp --cuda auto

# Explicit single-architecture build, e.g. H100 / sm_90
bash build-plugin.sh --linux --openmp --cuda 90
bash build-xfe-plugin.sh --linux --openmp --cuda 90

# Multi-architecture distribution build
bash build-plugin.sh --linux --openmp --cuda-archs "75,80,86,89,90"
bash build-xfe-plugin.sh --linux --openmp --cuda-archs "75,80,86,89,90"
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

- the plugin was rebuilt without `--cuda auto` or another CUDA enable flag
- the wrong plugin is being picked up on the `adopath`
- Stata is still holding an older plugin in memory after a rebuild or plugin-path switch; run `discard` with no arguments
- CUDA is unavailable at runtime

## Notes

- `xhdfe` and `xfe` have separate plugin binaries; rebuild both if you want both commands to use CUDA.
- For local use, prefer `--cuda auto`.
- For explicit H100-only use, pass `--cuda 90`.
- For shared distribution bundles, prefer `--cuda-archs "75,80,86,89,90"`.
