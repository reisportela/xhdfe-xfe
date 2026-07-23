# Compilation notes (CPU + GPU + Stata)

This package builds on Linux, macOS (Apple Silicon), and Windows 10+.
Prebuilt artifacts may be included in distribution bundles, but clean rebuilds
are supported on each platform.

## C++ / Python build (CPU)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Platform notes:
- macOS Apple Silicon: add `-DCMAKE_OSX_ARCHITECTURES=arm64` if needed.
- macOS Apple Silicon running an x86_64 Python/Rosetta target: keep
  `XHDFE_ENABLE_MARCH_NATIVE=OFF` so AppleClang does not try to apply an
  Apple Silicon CPU name to an x86_64 target.
- Windows (Visual Studio): `cmake -S . -B build -A x64` then
  `cmake --build build --config Release --parallel`.
- Windows (MSYS2/MinGW): use `-G "MinGW Makefiles"` and build with
  `cmake --build build --parallel`.

`-march=native` is local tuning, not a portability requirement. CMake disables
it by default on Apple platforms and only applies it when the active
compiler/target accepts the flag. To force the portable path on any platform:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DXHDFE_ENABLE_MARCH_NATIVE=OFF
```

To build only the Python module:

```bash
cmake --build build --target py_hdfe_v11 --parallel
```

The Python extension is built as `py_hdfe_v11*` in `build/`.

## C++ / Python build (CUDA)

```bash
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release \
  -DXHDFE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build_cuda --parallel
```

Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU (minimum supported: sm_75).
On this H100 server, use `90`.
For multi-GPU compatibility in one build, pass a semicolon-separated list, e.g.
`-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"`.

## C++ / Python build (Metal)

```bash
cmake -S . -B build_metal -DCMAKE_BUILD_TYPE=Release \
  -DXHDFE_ENABLE_METAL=ON
cmake --build build_metal --parallel
```

Metal builds are supported on macOS with clang.

## Runtime backend selection (Python)

- Default backend is CPU unless `XHDFE_GPU_BACKEND_DEFAULT` was set at build.
- Override at runtime with:

```bash
XHDFE_GPU_BACKEND=cpu|cuda|metal
```

If CUDA/Metal is not available at runtime, the code falls back to CPU.

## Stata plugin build

From the repository root:

```bash
cd stata/tools
./build-plugin.sh
```

Output: `stata/xhdfe.plugin`

Optional GPU builds:

```bash
# CUDA (Linux native only)
# Single target for this H100 server
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 ./build-plugin.sh
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 ./build-xfe-plugin.sh

# Multi-target fatbin
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCHS="75,80,86,89,90" ./build-plugin.sh
XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCHS="75,80,86,89,90" ./build-xfe-plugin.sh

# Metal (macOS only)
XHDFE_ENABLE_METAL=ON CXX=clang++ ./build-plugin.sh
```

The script uses `stata/tools/_deps` and `stata/tools/_build`. If the bundled
`stplugin.*` and `eigen-3.4.0.tar.gz` are present, no network access is needed.

## Dependencies

- CMake >= 3.16
- C++17 compiler (g++/clang++/MSVC)
- Python 3 development headers (for `py_hdfe_v11`)
- CUDA Toolkit (only when `XHDFE_ENABLE_CUDA=ON`)

The autonomous release bundle vendors Eigen3 and pybind11 and therefore does
not need CMake network fetches. It also includes
`third_party/Rcpp_1.1.2.tar.gz`; install that archive into a local R library
before `R CMD INSTALL r/xhdfe` (see `r/README.md`). External prerequisites
that are not bundled are the platform toolchains themselves: CMake, a C++17
compiler, Python development headers, R, Stata for Stata execution, and the
CUDA toolkit/driver for CUDA builds.

## Help files

- C++/Python help: `xhdfe_py_hdfe_v11_help.html`
- Stata help: `stata/xhdfe.sthlp`
