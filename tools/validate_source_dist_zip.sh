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
  "xhdfe-src/pyproject.toml"
  "xhdfe-src/setup.py"
  "xhdfe-src/NOTICE"
  "xhdfe-src/third_party/licenses/GCC-13.2.0-COPYING3"
  "xhdfe-src/third_party/licenses/GCC-13.2.0-COPYING.RUNTIME"
  "xhdfe-src/third_party/licenses/mingw-w64-11.0.1-winpthreads-COPYING"
  "xhdfe-src/third_party/licenses/dlfcn-win32-1.4.1-COPYING"
  "xhdfe-src/py_hdfe_v11.py"
  "xhdfe-src/xhdfe/_formula.py"
  "xhdfe-src/xhdfe/_maketables.py"
  "xhdfe-src/xhdfe/help/xhdfe.md"
  "xhdfe-src/tests/test_formula_frontend.py"
  "xhdfe-src/tests/test_maketables_integration.py"
  "xhdfe-src/tests/test_windows_runtime_packaging.py"
  "xhdfe-src/third_party/eigen-3.4.0/Eigen/Core"
  "xhdfe-src/third_party/pybind11-2.11.1/include/pybind11/pybind11.h"
  "xhdfe-src/stata/tools/build-plugin.sh"
  "xhdfe-src/stata/tools/build-xfe-plugin.sh"
  "xhdfe-src/stata/tools/cuda-common.sh"
  "xhdfe-src/stata/tools/_deps/eigen-3.4.0.tar.gz"
  "xhdfe-src/stata/tools/_deps/stplugin.h"
  "xhdfe-src/stata/LICENSE"
  "xhdfe-src/stata/NOTICE"
  "xhdfe-src/tools/check_no_raw_isfinite.cmake"
  "xhdfe-src/tools/check_verifier_device_fma.sh"
  "xhdfe-src/tools/validate_python_release_artifacts.py"
  "xhdfe-src/tools/validate_release_metadata.py"
  "xhdfe-src/tools/build_corresponding_source_bundle.py"
  "xhdfe-src/tools/record_linux_release_provenance.py"
  "xhdfe-src/tools/validate_corresponding_source_bundle.py"
  "xhdfe-src/tests/ieee_bits_liveness.cpp"
  "xhdfe-src/tests/fail_closed_entrypoints.cpp"
  "xhdfe-src/tests/audit_20260804_contracts.py"
  "xhdfe-src/tests/test_corresponding_source_bundle.py"
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

command -v python3 >/dev/null 2>&1 || {
  echo "python3 is required to validate the Python source closure" >&2
  exit 1
}
python3 -m py_compile \
  "${source_root}/xhdfe/_formula.py" \
  "${source_root}/xhdfe/_maketables.py" \
  "${source_root}/tests/test_formula_frontend.py" \
  "${source_root}/tests/test_maketables_integration.py" \
  "${source_root}/tests/test_windows_runtime_packaging.py" \
  "${source_root}/tests/test_corresponding_source_bundle.py" \
  "${source_root}/tools/validate_python_release_artifacts.py" \
  "${source_root}/tools/validate_release_metadata.py" \
  "${source_root}/tools/build_corresponding_source_bundle.py" \
  "${source_root}/tools/record_linux_release_provenance.py" \
  "${source_root}/tools/validate_corresponding_source_bundle.py"

(
  cd "${source_root}"
  PYTHONPATH=. python3 -m unittest -v tests.test_corresponding_source_bundle
)

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
grep -Fq "Formulaic >= 1.2.1,<2" "${offline_doc}" || {
  echo "BUILD_OFFLINE.md does not delimit the optional formula dependency" >&2
  exit 1
}
grep -Fq "Rcpp ${EXPECTED_RCPP_VERSION}" "${notice}" || {
  echo "NOTICE does not disclose the bundled Rcpp source" >&2
  exit 1
}

license_checks=(
  "GCC-13.2.0-COPYING3:8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"
  "GCC-13.2.0-COPYING.RUNTIME:9d6b43ce4d8de0c878bf16b54d8e7a10d9bd42b75178153e3af6a815bdc90f74"
  "mingw-w64-11.0.1-winpthreads-COPYING:63263614cdd29f2f93cba85e992f041b31f9fc7b4033692f31269489a8a1b177"
  "dlfcn-win32-1.4.1-COPYING:4cc7ac997b9293db5919baf630100cc09b3508efdfe6a6611c95511fb863b3c7"
)
for item in "${license_checks[@]}"; do
  name="${item%%:*}"
  expected="${item#*:}"
  actual="$(sha256_file "${source_root}/third_party/licenses/${name}")"
  [[ "$actual" == "$expected" ]] || {
    echo "${name}: license SHA-256 mismatch: ${actual}" >&2
    exit 1
  }
done

echo "Autonomous source archive closure OK: ${ARCHIVE}"
echo "CMake configure and non-finite guard build: OK"
echo "Rcpp ${EXPECTED_RCPP_VERSION}: ${EXPECTED_RCPP_SHA256}"
