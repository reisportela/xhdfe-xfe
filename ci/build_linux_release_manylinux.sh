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
# xhdfe.plugin.linux-cuda, xfe.plugin.linux-cuda, xhdfe-*.whl, xhdfe-*.tar.gz,
# CUDA/wheel provenance ledgers, exact nvcc link traces, license copies, and
# the exact libgomp source RPM required by the repaired wheel.
# Every ELF is gated by tools/check_binary_floor.sh before this script exits;
# a floor violation is a build FAILURE, not a warning.
set -euo pipefail
shopt -s nullglob

FLOOR_GLIBC="${XHDFE_FLOOR_GLIBC:-2.28}"
FLOOR_GLIBCXX="${XHDFE_FLOOR_GLIBCXX:-3.4.25}"
CUDA_MM="${XHDFE_CUDA_VERSION:-12-6}"      # dnf package suffix, e.g. 12-6
PYBIN="${XHDFE_PYBIN:-/opt/python/cp312-cp312/bin/python}"
CUDA_EULA_INPUT="${XHDFE_CUDA_EULA:?set XHDFE_CUDA_EULA to the pinned CUDA 12.6 EULA input}"
test -f "$CUDA_EULA_INPUT"

echo "== toolchain =="
dnf -y install gcc-toolset-13-gcc-c++ cmake dnf-plugins-core git zip >/dev/null
source /opt/rh/gcc-toolset-13/enable
g++ --version | head -1

echo "== CUDA toolkit (nvcc + static cudart; no driver needed to compile) =="
dnf -y config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo >/dev/null
# cuda-cuobjdump: required by the fail-closed verifier FMA gate that
# build-plugin.sh runs on the verifier object (§7.6/R-04); nvcc alone does
# not ship it and the gate refuses to certify what it cannot see.
dnf -y install "cuda-nvcc-${CUDA_MM}" "cuda-cudart-devel-${CUDA_MM}" \
  "cuda-cccl-${CUDA_MM}" \
  "cuda-cuobjdump-${CUDA_MM}" "cuda-nvdisasm-${CUDA_MM}" \
  >/dev/null
# cuda-nvdisasm: cuobjdump --dump-sass DELEGATES disassembly to nvdisasm;
# without it the dump emits fatbin headers but zero "Function :" lines and
# the FMA gate fail-closes on "no hdfe_cert functions found in SASS".
export PATH="/usr/local/cuda/bin:${PATH}"
nvcc --version | tail -2

mkdir -p artifacts

# Preserve the exact nvcc argv and the expanded --dryrun link commands without
# changing either production build script. Exported Bash functions survive the
# two child `bash` invocations and are accepted by their command -v gate.
XHDFE_REAL_NVCC="$(readlink -e "$(command -v nvcc)")"
XHDFE_NVCC_INVOCATIONS="${PWD}/artifacts/cuda-nvcc-invocations.jsonl"
XHDFE_NVCC_LINK_DRYRUN="${PWD}/artifacts/cuda-nvcc-link-dryrun.log"
XHDFE_PROVENANCE_PYTHON="$PYBIN"
export XHDFE_REAL_NVCC XHDFE_NVCC_INVOCATIONS XHDFE_NVCC_LINK_DRYRUN
export XHDFE_PROVENANCE_PYTHON
: > "$XHDFE_NVCC_INVOCATIONS"
: > "$XHDFE_NVCC_LINK_DRYRUN"
xhdfe_release_nvcc() {
  "$XHDFE_PROVENANCE_PYTHON" tools/record_linux_release_provenance.py \
    nvcc-wrapper -- "$@"
}
export -f xhdfe_release_nvcc
export NVCC=xhdfe_release_nvcc

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

echo "== CUDA static-link provenance gate =="
"$PYBIN" tools/record_linux_release_provenance.py cuda-ledger \
  --nvcc "$XHDFE_REAL_NVCC" \
  --trace-jsonl "$XHDFE_NVCC_INVOCATIONS" \
  --link-dryrun "$XHDFE_NVCC_LINK_DRYRUN" \
  --plugin artifacts/xhdfe.plugin.linux-cuda \
  --plugin artifacts/xfe.plugin.linux-cuda \
  --toolkit-eula "$CUDA_EULA_INPUT" \
  --license-dir artifacts/cuda-license-files \
  --output artifacts/linux-cuda-provenance.json

echo "== python wheel + sdist (cp312, march-native OFF, CUDA OFF) =="
"$PYBIN" -m pip install --quiet numpy setuptools wheel
RAW_WHEEL_DIR="$(mktemp -d)"
XHDFE_ENABLE_CUDA=OFF XHDFE_ENABLE_MARCH_NATIVE=OFF \
  "$PYBIN" -m pip wheel . --no-build-isolation --no-deps --wheel-dir "$RAW_WHEEL_DIR"
RAW_WHEELS=("$RAW_WHEEL_DIR"/xhdfe-*.whl)
test "${#RAW_WHEELS[@]}" -eq 1
auditwheel repair --plat manylinux_2_28_x86_64 \
  --wheel-dir artifacts "${RAW_WHEELS[0]}"
XHDFE_ENABLE_CUDA=OFF XHDFE_ENABLE_MARCH_NATIVE=OFF \
  "$PYBIN" setup.py --quiet sdist --dist-dir artifacts

echo "== exact libgomp provider + corresponding source RPM =="
LIBGOMP_PROVIDER="$(readlink -e "$(g++ -print-file-name=libgomp.so.1)")"
test -f "$LIBGOMP_PROVIDER"
LIBGOMP_NEVRA="$(rpm -qf --qf '%{NAME}-%{VERSION}-%{RELEASE}.%{ARCH}' "$LIBGOMP_PROVIDER")"
LIBGOMP_SOURCE_RPM="$(rpm -qf --qf '%{SOURCERPM}' "$LIBGOMP_PROVIDER")"
test -n "$LIBGOMP_NEVRA"
test -n "$LIBGOMP_SOURCE_RPM"
mkdir -p artifacts/corresponding-source
PREEXISTING_SOURCE_RPMS=(artifacts/corresponding-source/*.src.rpm artifacts/corresponding-source/*.nosrc.rpm)
test "${#PREEXISTING_SOURCE_RPMS[@]}" -eq 0
dnf -y download --source --destdir artifacts/corresponding-source "$LIBGOMP_NEVRA"
LIBGOMP_SOURCE_RPMS=(artifacts/corresponding-source/*.src.rpm artifacts/corresponding-source/*.nosrc.rpm)
test "${#LIBGOMP_SOURCE_RPMS[@]}" -eq 1
test "$(basename "${LIBGOMP_SOURCE_RPMS[0]}")" = "$LIBGOMP_SOURCE_RPM"

WHEELS=(artifacts/xhdfe-*.whl)
test "${#WHEELS[@]}" -eq 1
"$PYBIN" tools/record_linux_release_provenance.py wheel-ledger \
  --raw-wheel "${RAW_WHEELS[0]}" \
  --repaired-wheel "${WHEELS[0]}" \
  --libgomp-provider "$LIBGOMP_PROVIDER" \
  --libgomp-source-rpm "${LIBGOMP_SOURCE_RPMS[0]}" \
  --output artifacts/linux-wheel-runtime-ledger.json

# Export the two exact byte sequences named in the ledger so the release
# assembler can prove packaged-runtime -> provider-binary -> source-RPM without
# relying on container-only absolute paths.
mkdir -p artifacts/linux-wheel-provider-runtime \
  artifacts/linux-wheel-packaged-runtime
cp "$LIBGOMP_PROVIDER" artifacts/linux-wheel-provider-runtime/
"$PYBIN" - artifacts/linux-wheel-runtime-ledger.json "${WHEELS[0]}" <<'PY'
import json
from pathlib import Path
import sys
import zipfile

ledger = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
private = ledger["repaired_wheel"]["private_libraries"]
if len(private) != 1:
    raise SystemExit(f"expected one private wheel runtime, found {len(private)}")
member = private[0]["member"]
with zipfile.ZipFile(sys.argv[2]) as archive:
    payload = archive.read(member)
destination = Path("artifacts/linux-wheel-packaged-runtime") / Path(member).name
destination.write_bytes(payload)
PY

echo "== floor-gate the wheel's extension module =="
WHEEL_STAGE="$(mktemp -d)"
"$PYBIN" -m zipfile -e "${WHEELS[0]}" "$WHEEL_STAGE"
mapfile -t WHEEL_SOS < <(find "$WHEEL_STAGE" -name '*.so')
test "${#WHEEL_SOS[@]}" -ge 1
bash tools/check_binary_floor.sh --max-glibc "$FLOOR_GLIBC" --max-glibcxx "$FLOOR_GLIBCXX" \
  "${WHEEL_SOS[@]}"
rm -rf "$WHEEL_STAGE"

echo "MANYLINUX BUILD OK (floor GLIBC_${FLOOR_GLIBC} / GLIBCXX_${FLOOR_GLIBCXX})"
