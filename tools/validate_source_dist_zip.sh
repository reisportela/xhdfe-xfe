#!/usr/bin/env bash
# Validate the autonomous xhdfe source archive, including the pinned Rcpp
# source needed for an R installation with networking disabled.
set -euo pipefail

ARCHIVE="${1:?usage: validate_source_dist_zip.sh XHDFE-SRC.ZIP}"
[[ -f "${ARCHIVE}" ]] || {
  echo "missing source archive: ${ARCHIVE}" >&2
  exit 1
}

EXPECTED_RCPP_VERSION="1.1.2"
EXPECTED_RCPP_SHA256="2746cf2fb188e5f0a84dbf5c8f68915b54564ed33e5754572f174e7b32e7f4f3"
RCPP_ARCHIVE="xhdfe-src/third_party/Rcpp_${EXPECTED_RCPP_VERSION}.tar.gz"
RCPP_PROVENANCE="xhdfe-src/third_party/RCPP_SOURCE_PROVENANCE.md"

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${path}" | awk '{print $1}'
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "${path}" | awk '{print $NF}'
  else
    echo "need sha256sum, shasum, or openssl to verify the source archive" >&2
    return 127
  fi
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

unzip -tq "${ARCHIVE}" >/dev/null
unzip -Z1 "${ARCHIVE}" > "${TMP_DIR}/listing.txt"

required_entries=(
  "xhdfe-src/BUILD_OFFLINE.md"
  "xhdfe-src/CMakeLists.txt"
  "xhdfe-src/LICENSE"
  "xhdfe-src/MANIFEST.in"
  "xhdfe-src/NOTICE"
  "xhdfe-src/py_hdfe_v11.py"
  "xhdfe-src/third_party/eigen-3.4.0/Eigen/Core"
  "xhdfe-src/third_party/pybind11-2.11.1/include/pybind11/pybind11.h"
  "xhdfe-src/stata/tools/build-plugin.sh"
  "xhdfe-src/stata/tools/build-xfe-plugin.sh"
  "xhdfe-src/stata/tools/cuda-common.sh"
  "xhdfe-src/stata/tools/_deps/eigen-3.4.0.tar.gz"
  "xhdfe-src/stata/tools/_deps/stplugin.h"
  "xhdfe-src/tools/check_no_raw_isfinite.cmake"
  "xhdfe-src/tools/check_verifier_device_fma.sh"
  "xhdfe-src/tests/ieee_bits_liveness.cpp"
  "xhdfe-src/tests/fail_closed_entrypoints.cpp"
  "xhdfe-src/tests/audit_20260804_contracts.py"
  "${RCPP_ARCHIVE}"
  "${RCPP_PROVENANCE}"
)
for entry in "${required_entries[@]}"; do
  grep -Fxq "${entry}" "${TMP_DIR}/listing.txt" || {
    echo "source archive is not closed: missing ${entry}" >&2
    exit 1
  }
done

unzip -q "${ARCHIVE}" -d "${TMP_DIR}/unpacked"

rcpp_path="${TMP_DIR}/unpacked/${RCPP_ARCHIVE}"
provenance_path="${TMP_DIR}/unpacked/${RCPP_PROVENANCE}"
offline_doc="${TMP_DIR}/unpacked/xhdfe-src/BUILD_OFFLINE.md"
notice="${TMP_DIR}/unpacked/xhdfe-src/NOTICE"
source_root="${TMP_DIR}/unpacked/xhdfe-src"

bash -n \
  "${source_root}/stata/tools/build-plugin.sh" \
  "${source_root}/stata/tools/build-xfe-plugin.sh" \
  "${source_root}/stata/tools/cuda-common.sh" \
  "${source_root}/tools/check_verifier_device_fma.sh"

command -v cmake >/dev/null 2>&1 || {
  echo "cmake is required to validate the autonomous source archive" >&2
  exit 1
}
cmake -S "${source_root}" -B "${TMP_DIR}/cmake-build" \
  -DBUILD_TESTING=ON \
  -DXHDFE_BUILD_PYTHON=OFF \
  -DXHDFE_ENABLE_CUDA=OFF \
  -DXHDFE_ENABLE_MARCH_NATIVE=OFF >/dev/null
cmake --build "${TMP_DIR}/cmake-build" \
  --target xhdfe_nonfinite_guard --parallel 2 >/dev/null

actual_sha256="$(sha256_file "${rcpp_path}")"
[[ "${actual_sha256}" == "${EXPECTED_RCPP_SHA256}" ]] || {
  echo "Rcpp archive SHA-256 mismatch: ${actual_sha256}" >&2
  exit 1
}

tar -xOzf "${rcpp_path}" Rcpp/DESCRIPTION > "${TMP_DIR}/Rcpp-DESCRIPTION"
actual_package="$(awk -F ': *' '$1 == "Package" { print $2; exit }' \
  "${TMP_DIR}/Rcpp-DESCRIPTION")"
actual_version="$(awk -F ': *' '$1 == "Version" { print $2; exit }' \
  "${TMP_DIR}/Rcpp-DESCRIPTION")"
[[ "${actual_package}" == "Rcpp" && "${actual_version}" == "${EXPECTED_RCPP_VERSION}" ]] || {
  echo "unexpected bundled R package: ${actual_package} ${actual_version}" >&2
  exit 1
}

grep -Fq "${EXPECTED_RCPP_SHA256}" "${provenance_path}" || {
  echo "Rcpp provenance does not record the certified SHA-256" >&2
  exit 1
}
grep -Fq "Rcpp_${EXPECTED_RCPP_VERSION}.tar.gz" "${offline_doc}" || {
  echo "BUILD_OFFLINE.md does not give the pinned Rcpp install path" >&2
  exit 1
}
grep -Fq "Rcpp ${EXPECTED_RCPP_VERSION}" "${notice}" || {
  echo "NOTICE does not disclose the bundled Rcpp source" >&2
  exit 1
}

echo "Autonomous source archive closure OK: ${ARCHIVE}"
echo "CMake configure and non-finite guard build: OK"
echo "Rcpp ${EXPECTED_RCPP_VERSION}: ${EXPECTED_RCPP_SHA256}"
