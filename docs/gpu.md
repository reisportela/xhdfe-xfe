# Using the GPU (CUDA)

xhdfe can offload the fixed-effect absorber to an NVIDIA GPU in **every version
of the package — Stata, Python, and R**. This guide is self-contained: it takes
you from a bare machine to a verified GPU run in whichever language you use.

Key facts, true for all three versions:

- The GPU is **optional and opt-in per call**. CPU is the reference backend and
  defines the numbers; the GPU produces the same results, only faster on large
  problems.
- **A GPU build is never distributed as a prebuilt binary.** The online
  net-install (Stata), PyPI-style wheels (Python), and any CPU package are
  CPU-only. To use the GPU you **build/install the package from source with CUDA
  enabled** — there is no way to "turn on" the GPU in an already-installed CPU
  build.
- The three steps are the same everywhere: **(1)** install/build with CUDA for
  your GPU, **(2)** request the GPU on a call, **(3)** verify it was actually
  used.

## Requirements

- An NVIDIA GPU of **compute capability 7.5 or newer** (T4, RTX 20xx, or later).
- **Linux x86-64** (CUDA builds are Linux + NVIDIA only).
- The **CUDA toolkit** (`nvcc`) and a working driver. Check with
  `nvcc --version` and `nvidia-smi`.

## Step 1 — let the installer detect your GPU

The CUDA build must target your card. The recommended commands below use
`auto` mode, which calls `nvidia-smi` and selects the local compute capability.
If you need to set the target manually, read it with:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# e.g. 9.0 (H100), 8.9 (L40 / RTX 4090), 8.6 (A10 / RTX 30xx), 8.0 (A100), 7.5 (T4)
```

Drop the dot to get the **arch value** used below: `9.0 → 90`, `8.6 → 86`,
`7.5 → 75`. The minimum supported is `75`.

## Step 2 — install/build with CUDA

Use `auto` unless you are cross-compiling or building a multi-architecture
binary.

### Stata

The net-install site ships CPU-only plugins. The easiest way to get CUDA is the
companion command **`xhdfegpu`** — run it once after `net install`:

```stata
xhdfegpu
```

It gates on `nvidia-smi` (so it only builds when a GPU is present), compiles a
plugin for the local architecture, and installs it *over* the CPU plugin in
place (same `xhdfe.plugin` / `xfe.plugin`). On a machine without internet, fetch
the source zip elsewhere and pass it in: `xhdfegpu, zip("/path/to/xhdfe-src.zip")`.
See `help xhdfegpu`. Then `discard` and use `gpubackend(cuda)` (Step 3).

To build the plugin by hand instead, from a clone:

```bash
git clone https://github.com/reisportela/xhdfe-xfe.git && cd xhdfe-xfe
bash stata/tools/build-plugin.sh     --linux --openmp --cuda auto
bash stata/tools/build-xfe-plugin.sh --linux --openmp --cuda auto
```

Then add the folder to your adopath (this writes nothing outside it):

```stata
adopath + "/path/to/xhdfe-xfe/stata"
```

For an explicit target, use `--cuda 90`. For a single binary that runs on
several GPU generations, use `--cuda-archs "75,80,86,89,90"`.

### Python

Always builds from source; `XHDFE_ENABLE_CUDA=auto` detects the local GPU
architecture. The build also needs Python development headers for the active
Python interpreter (`Python.h`). Install `python3-dev` on Debian/Ubuntu or
`python3-devel` on Fedora/RHEL/Rocky; on clusters without sudo, use a
conda/mamba environment or a Python module that includes headers.

```bash
# from a clone:
XHDFE_ENABLE_CUDA=auto python -m pip install .
# or straight from GitHub:
XHDFE_ENABLE_CUDA=auto python -m pip install "git+https://github.com/reisportela/xhdfe-xfe.git"
```

For an explicit target, set `XHDFE_CUDA_ARCH=90` or
`CMAKE_CUDA_ARCHITECTURES=90`.

### R

Set the build variable in R, then install from GitHub:

```r
Sys.setenv(XHDFE_ENABLE_CUDA = "auto")
remotes::install_github("reisportela/xhdfe-xfe", subdir = "r/xhdfe")
```

or, from a clone: `XHDFE_ENABLE_CUDA=auto R CMD INSTALL r/xhdfe`.
For an explicit target, set `XHDFE_CUDA_ARCH=90`.

## Step 3 — request the GPU on a call

| Version | How to request the GPU |
|---------|------------------------|
| Stata   | add the option `gpubackend(cuda)` to the `xhdfe` / `xfe` command |
| Python  | set `os.environ["XHDFE_GPU_BACKEND"] = "cuda"` before `.fit()` |
| R       | pass `backend = "cuda"` to `xhdfe(...)` |

## Step 4 — verify the GPU was actually used

The CPU and GPU produce identical numbers, so a silent CPU run is easy to miss.
**Always confirm** with the status fields.

Stata:

```stata
xhdfe y x1 x2, absorb(id year) gpubackend(cuda)
di e(gpu_used)          // 1
di "`e(gpu_backend)'"   // cuda
```

Python:

```python
import os
os.environ["XHDFE_GPU_BACKEND"] = "cuda"
reg = xhdfe.HdfeRegressor()
reg.fit(y, X, fes=[id, year])
assert reg.gpu_used_ == 1          # 1 = GPU was used
assert reg.gpu_status_code_ == 1
```

R:

```r
m <- xhdfe(y ~ x1 + x2 | id + year, data = d, backend = "cuda")
stopifnot(m$gpu_used == 1, m$gpu_status == "used")
```

## What happens when the GPU cannot be used

If you request the GPU but it can't be honored (you installed a CPU build, no
device is present, or the GPU path fails), the behavior differs by version:

- **Stata and R are fail-closed**: the call **stops with an error** rather than
  quietly returning CPU results (Stata error 498; R `stop()`).
- **Python does not auto-error**: `.fit()` returns a CPU result with
  `reg.gpu_used_ == 0`. In Python you must **check `reg.gpu_used_` yourself** —
  it will not raise.

`gpu_status_code` values (Stata `e(gpu_status_code)`, Python
`reg.gpu_status_code_`): `0` not requested, `1` used, `2` unavailable,
`3` did not converge on GPU, `4` failed, `5` CPU cache/profile result.

## Confirm a build actually has CUDA

- **Python / R**: call `xhdfe_info()` — its CUDA-arch field reports e.g. `sm_90`
  for a CUDA build and is empty for a CPU build.
- **Stata**: run once with `gpubackend(cuda)` and check `e(gpu_used) == 1`, or
  inspect the plugin (`readelf -S stata/xhdfe.plugin | grep nv_fatb` is present
  only in a CUDA build).

## Troubleshooting

- **"backend cuda was requested but the GPU was not used"** (R / Stata error) —
  you installed a CPU build or no device is available. Rebuild with
  `--cuda auto` for Stata or `XHDFE_ENABLE_CUDA=auto` for Python/R.
- **Wrong architecture / no kernels run** — the build arch must match your card;
  below `sm_75` is unsupported. Re-check Step 1 and rebuild.
- **Stata still runs on CPU after a rebuild** — run `discard` (no argument) so
  Stata reloads the new plugin, and confirm `which xhdfe` points to your
  `stata/` folder. Do not use `discard xhdfe`.
- **`nvcc: command not found`** — install the CUDA toolkit and put `nvcc` on
  `PATH` (or set `CUDA_HOME`).

For the detailed Stata build and verification checklist, see
[`../stata/BUILD_CUDA.md`](../stata/BUILD_CUDA.md).
