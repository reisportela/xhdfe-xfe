#!/usr/bin/env bash
# Build the Linux release artifacts inside quay.io/pypa/manylinux_2_28_x86_64
# (AlmaLinux 8, glibc 2.28), so the platform floor of the artifacts is
# GLIBC_2.28 / GLIBCXX_3.4.25 instead of whatever Ubuntu the CI runner ships.
#
# Why: release 2.21.0's Linux binaries were built on Ubuntu 24.04 and require
# GLIBC_2.38/GLIBCXX_3.4.32, which does not load on RHEL/AlmaLinux/Rocky 9
# (glibc 2.34), Ubuntu 22.04, Debian 12 or SLES 15 — the platforms the stated
# audience (statistical institutes, central banks) actually runs (F-01,
# Verifications/codex_gpu_native_20260728/FINDINGS.md). The gcc-toolset model
# on EL8 links new C++ runtime parts statically (libstdc++_nonshared), so a
# modern compiler still yields the old floor.
#
# Run from the repository root, inside the container:
#   docker run --rm -v "$PWD:/w" -w /w -e XHDFE_CUDA_ARCHS \
#     quay.io/pypa/manylinux_2_28_x86_64 bash ci/build_linux_release_manylinux.sh
#
# Produces (in artifacts/): xhdfe.plugin.linux-cpu, xfe.plugin.linux-cpu,
# xhdfe.plugin.linux-cuda, xfe.plugin.linux-cuda, xhdfe-*.whl, xhdfe-*.tar.gz.
# Every ELF is gated by tools/check_binary_floor.sh before this script exits;
# a floor violation is a build FAILURE, not a warning.
set -euo pipefail

FLOOR_GLIBC="${XHDFE_FLOOR_GLIBC:-2.28}"
FLOOR_GLIBCXX="${XHDFE_FLOOR_GLIBCXX:-3.4.25}"
CUDA_MM="${XHDFE_CUDA_VERSION:-12-6}"      # dnf package suffix, e.g. 12-6
PYBIN="${XHDFE_PYBIN:-/opt/python/cp312-cp312/bin/python}"

echo "== toolchain =="
dnf -y install gcc-toolset-13-gcc-c++ cmake git zip >/dev/null
source /opt/rh/gcc-toolset-13/enable
g++ --version | head -1

echo "== CUDA toolkit (nvcc + static cudart; no driver needed to compile) =="
dnf -y config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo >/dev/null
dnf -y install "cuda-nvcc-${CUDA_MM}" "cuda-cudart-devel-${CUDA_MM}" \
  "libcurand-devel-${CUDA_MM}" >/dev/null
export PATH="/usr/local/cuda/bin:${PATH}"
nvcc --version | tail -2

mkdir -p artifacts

echo "== linux-cpu plugins =="
bash stata/tools/build-plugin.sh --linux --openmp
cp stata/xhdfe.plugin artifacts/xhdfe.plugin.linux-cpu
bash stata/tools/build-xfe-plugin.sh --linux --openmp
cp stata/xfe.plugin artifacts/xfe.plugin.linux-cpu
bash tools/check_binary_floor.sh --max-glibc "$FLOOR_GLIBC" --max-glibcxx "$FLOOR_GLIBCXX" \
  artifacts/xhdfe.plugin.linux-cpu artifacts/xfe.plugin.linux-cpu

echo "== linux-cuda fatbin plugins =="
XHDFE_ENABLE_CUDA=ON bash stata/tools/build-plugin.sh --linux --openmp
cp stata/xhdfe.plugin artifacts/xhdfe.plugin.linux-cuda
XHDFE_ENABLE_CUDA=ON bash stata/tools/build-xfe-plugin.sh --linux --openmp
cp stata/xfe.plugin artifacts/xfe.plugin.linux-cuda
bash tools/check_binary_floor.sh --max-glibc "$FLOOR_GLIBC" --max-glibcxx "$FLOOR_GLIBCXX" \
  artifacts/xhdfe.plugin.linux-cuda artifacts/xfe.plugin.linux-cuda

echo "== python wheel + sdist (cp312, march-native OFF, CUDA OFF) =="
"$PYBIN" -m pip install --quiet numpy setuptools wheel
XHDFE_ENABLE_CUDA=OFF XHDFE_ENABLE_MARCH_NATIVE=OFF \
  "$PYBIN" -m pip wheel . --no-build-isolation --no-deps --wheel-dir artifacts
XHDFE_ENABLE_CUDA=OFF XHDFE_ENABLE_MARCH_NATIVE=OFF \
  "$PYBIN" setup.py --quiet sdist --dist-dir artifacts

echo "== floor-gate the wheel's extension module =="
WHEELS=(artifacts/xhdfe-*.whl)
test "${#WHEELS[@]}" -eq 1
WHEEL_STAGE="$(mktemp -d)"
"$PYBIN" -m zipfile -e "${WHEELS[0]}" "$WHEEL_STAGE"
mapfile -t WHEEL_SOS < <(find "$WHEEL_STAGE" -name '*.so')
test "${#WHEEL_SOS[@]}" -ge 1
bash tools/check_binary_floor.sh --max-glibc "$FLOOR_GLIBC" --max-glibcxx "$FLOOR_GLIBCXX" \
  "${WHEEL_SOS[@]}"
rm -rf "$WHEEL_STAGE"

echo "MANYLINUX BUILD OK (floor GLIBC_${FLOOR_GLIBC} / GLIBCXX_${FLOOR_GLIBCXX})"
