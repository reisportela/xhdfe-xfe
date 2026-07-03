#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATA_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${SCRIPT_DIR}/_build"
DEPS_DIR="${SCRIPT_DIR}/_deps"

EIGEN_VER="3.4.0"
EIGEN_TGZ="${DEPS_DIR}/eigen-${EIGEN_VER}.tar.gz"
EIGEN_DIR="${BUILD_DIR}/eigen-${EIGEN_VER}"

STPLUGIN_H="${DEPS_DIR}/stplugin.h"
STPLUGIN_C="${DEPS_DIR}/stplugin.c"

mkdir -p "${BUILD_DIR}" "${DEPS_DIR}"

download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$out"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO "$out" "$url"
    return
  fi
  echo "Error: neither curl nor wget is available to download $url" >&2
  exit 1
}

if [[ ! -f "${STPLUGIN_H}" ]]; then
  echo "Downloading stplugin.h..."
  download "https://www.stata.com/plugins/stplugin.h" "${STPLUGIN_H}"
fi

if [[ ! -f "${STPLUGIN_C}" ]]; then
  echo "Downloading stplugin.c..."
  download "https://www.stata.com/plugins/stplugin.c" "${STPLUGIN_C}"
fi

if [[ ! -f "${EIGEN_TGZ}" ]]; then
  echo "Downloading Eigen ${EIGEN_VER}..."
  download "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VER}/eigen-${EIGEN_VER}.tar.gz" "${EIGEN_TGZ}"
fi

if [[ ! -f "${EIGEN_DIR}/Eigen/Dense" ]]; then
  echo "Extracting Eigen..."
  rm -rf "${EIGEN_DIR}"
  tar -xzf "${EIGEN_TGZ}" -C "${BUILD_DIR}"
fi

OUT_PLUGIN="${STATA_DIR}/xhdfe.plugin"

usage() {
  cat <<'EOF'
Usage: build-plugin.sh [--windows|--linux] [--openmp|--no-openmp]

Builds the Stata plugin `xhdfe.plugin` in the repository root.

Targets:
  --windows    Build a Windows (PE/DLL) plugin using mingw-w64.
  --linux      Build a Linux (ELF) plugin using the native toolchain.

OpenMP:
  --openmp     Enable OpenMP (may require shipping libgomp on Windows).
  --no-openmp  Disable OpenMP (default on Windows).

GPU (environment variables):
  XHDFE_ENABLE_CUDA=ON   Enable CUDA (Linux native builds only).
  XHDFE_CUDA_ARCH=90     CUDA SM architecture (default: 90).
  XHDFE_ENABLE_METAL=ON  Enable Metal (macOS native builds only).
EOF
}

TARGET="${XHDFE_TARGET:-}"
OPENMP_MODE="${XHDFE_OPENMP:-}"
USE_METAL=0
USE_CUDA=0

for arg in "$@"; do
  case "$arg" in
    --windows|--win|--win64)
      TARGET="windows"
      ;;
    --linux)
      TARGET="linux"
      ;;
    --openmp)
      OPENMP_MODE="on"
      ;;
    --no-openmp)
      OPENMP_MODE="off"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

UNAME_S="$(uname -s)"
if [[ -z "${TARGET}" ]]; then
  if [[ "${UNAME_S}" == "Darwin" ]]; then
    TARGET="linux" # macOS builds are handled as Unix (APPLEMAC SYSTEM) below.
  elif [[ "${UNAME_S}" == "Linux" && -n "${WSL_DISTRO_NAME:-}" ]]; then
    TARGET="windows"
  else
    TARGET="linux"
  fi
fi

SYSTEM_DEF="OPUNIX"
if [[ "${UNAME_S}" == "Darwin" ]]; then
  SYSTEM_DEF="APPLEMAC"
fi

if [[ "${TARGET}" == "windows" ]]; then
  CXX="${CXX:-x86_64-w64-mingw32-g++}"
  STRIP_BIN="${STRIP_BIN:-x86_64-w64-mingw32-strip}"
  SYSTEM_DEF="STWIN32"
  if [[ -z "${OPENMP_MODE}" ]]; then
    OPENMP_MODE="off"
  fi
else
  if [[ "${UNAME_S}" == "Darwin" ]]; then
    CXX="${CXX:-clang++}"
  else
    CXX="${CXX:-g++}"
  fi
  STRIP_BIN="${STRIP_BIN:-strip}"
  if [[ -z "${OPENMP_MODE}" ]]; then
    if [[ "${UNAME_S}" == "Darwin" ]]; then
      OPENMP_MODE="off"
    else
      OPENMP_MODE="auto"
    fi
  fi
fi

if ! command -v "${CXX}" >/dev/null 2>&1; then
  echo "Error: compiler not found: ${CXX}" >&2
  if [[ "${TARGET}" == "windows" ]]; then
    echo "Install mingw-w64 (Ubuntu/Debian): apt-get install -y g++-mingw-w64-x86-64" >&2
  fi
  exit 1
fi

link_flag="-shared"
if [[ "${UNAME_S}" == "Darwin" && "${TARGET}" != "windows" ]]; then
  link_flag="-bundle"
fi

common_compile_flags=( -std=c++17 -O3 "-DSYSTEM=${SYSTEM_DEF}" )
if [[ "${TARGET}" != "windows" ]]; then
  common_compile_flags+=( -fPIC -pthread )
  pthread_flag=( -pthread )
else
  common_compile_flags+=( -static-libgcc -static-libstdc++ )
  pthread_flag=()
fi
common_compile_flags+=( -I"${STATA_DIR}/include" -I"${EIGEN_DIR}" -I"${DEPS_DIR}" )
common_compile_flags+=( -ffast-math -funroll-loops )
common_link_flags=( "${link_flag}" )

stplugin_src="${STPLUGIN_C}"
srcs=( "${STATA_DIR}/src/xhdfe_plugin.cpp" )
srcs+=( "${STATA_DIR}/src/hdfe_regressor_v11.cpp" )
srcs+=( "${STATA_DIR}/src/fe_absorption.cpp" )
srcs+=( "${STATA_DIR}/src/ols.cpp" )
srcs+=( "${STATA_DIR}/src/iv.cpp" )

if [[ "${XHDFE_ENABLE_METAL:-0}" =~ ^(1|ON|on|true|yes)$ ]]; then
  USE_METAL=1
fi

if [[ "${XHDFE_ENABLE_CUDA:-0}" =~ ^(1|ON|on|true|yes)$ ]]; then
  USE_CUDA=1
fi

if [[ "${USE_CUDA}" -eq 1 && "${USE_METAL}" -eq 1 ]]; then
  echo "Error: XHDFE_ENABLE_CUDA and XHDFE_ENABLE_METAL cannot both be enabled." >&2
  exit 1
fi

metal_flags=()
if [[ "${USE_METAL}" -eq 1 ]]; then
  if [[ "${UNAME_S}" != "Darwin" || "${TARGET}" == "windows" ]]; then
    echo "Error: XHDFE_ENABLE_METAL=ON requires a macOS native build target." >&2
    exit 1
  fi
  srcs+=( "${STATA_DIR}/src/fe_absorption_metal.mm" )
  metal_flags+=( -DHDFE_USE_METAL -fobjc-arc -framework Metal -framework Foundation )
fi

compile_plugin() {
  local out="$1"
  shift
  local extra_flags=( "$@" )

  if [[ "${USE_CUDA}" -eq 1 ]]; then
    if [[ "${UNAME_S}" == "Darwin" ]]; then
      echo "Error: CUDA builds are not supported on macOS." >&2
      exit 1
    fi
    if [[ "${TARGET}" == "windows" ]]; then
      echo "Error: CUDA builds are only supported for native Linux targets." >&2
      exit 1
    fi
    NVCC="${NVCC:-nvcc}"
    CUDA_ARCH="${XHDFE_CUDA_ARCH:-90}"
    if ! command -v "${NVCC}" >/dev/null 2>&1; then
      echo "Error: nvcc not found; install CUDA or disable XHDFE_ENABLE_CUDA." >&2
      exit 1
    fi

    OBJ_DIR="${BUILD_DIR}/obj"
    rm -rf "${OBJ_DIR}"
    mkdir -p "${OBJ_DIR}"
    cuda_src="${STATA_DIR}/src/fe_absorption_cuda.cu"

    build_openmp=1
    if [[ "${OPENMP_MODE}" == "off" ]]; then
      build_openmp=0
    fi

    while true; do
      objs=()
      cxx_flags=( "${common_compile_flags[@]}" "${extra_flags[@]}" -DHDFE_USE_CUDA )
      nvcc_flags=( -std=c++17 -O3 --expt-relaxed-constexpr -DHDFE_USE_CUDA "-DSYSTEM=${SYSTEM_DEF}" )
      nvcc_flags+=( -I"${STATA_DIR}/include" -I"${EIGEN_DIR}" -I"${DEPS_DIR}" )
      nvcc_flags+=( -Xcompiler "-fPIC" )
      if [[ "${build_openmp}" -eq 1 ]]; then
        cxx_flags+=( -DHDFE_USE_OPENMP -fopenmp )
        nvcc_flags+=( -Xcompiler "-fopenmp" )
      fi
      nvcc_flags+=( -gencode "arch=compute_${CUDA_ARCH},code=[compute_${CUDA_ARCH},sm_${CUDA_ARCH}]" )

      set +e
      obj="${OBJ_DIR}/stplugin.o"
      "${CXX}" "${cxx_flags[@]}" -x c++ -c "${stplugin_src}" -o "${obj}"
      rc=$?
      if [[ $rc -ne 0 ]]; then
        break
      fi
      objs+=( "${obj}" )
      for src in "${srcs[@]}"; do
        obj="${OBJ_DIR}/$(basename "${src%.*}").o"
        "${CXX}" "${cxx_flags[@]}" -c "${src}" -o "${obj}"
        rc=$?
        if [[ $rc -ne 0 ]]; then
          break
        fi
        objs+=( "${obj}" )
      done
      if [[ $rc -eq 0 ]]; then
        cuda_obj="${OBJ_DIR}/fe_absorption_cuda.o"
        "${NVCC}" "${nvcc_flags[@]}" -c "${cuda_src}" -o "${cuda_obj}"
        rc=$?
        if [[ $rc -eq 0 ]]; then
          objs+=( "${cuda_obj}" )
          set -e
          break
        fi
      fi
      set -e
      if [[ "${build_openmp}" -eq 1 && "${OPENMP_MODE}" != "on" ]]; then
        echo "OpenMP build failed; retrying without OpenMP..."
        build_openmp=0
        rm -f "${OBJ_DIR}"/*.o
        continue
      fi
      exit $rc
    done

    link_flags=( "${common_link_flags[@]}" -Xcompiler "-pthread" )
    if [[ "${build_openmp}" -eq 1 ]]; then
      link_flags+=( -Xcompiler "-fopenmp" )
    fi
    "${NVCC}" "${link_flags[@]}" "${objs[@]}" -o "${out}"
  else
    if [[ "${OPENMP_MODE}" == "on" ]]; then
      "${CXX}" "${common_compile_flags[@]}" "${common_link_flags[@]}" "${extra_flags[@]}" "${pthread_flag[@]}" \
        -DHDFE_USE_OPENMP -fopenmp -x c++ "${stplugin_src}" -x none "${srcs[@]}" -o "${out}"
    elif [[ "${OPENMP_MODE}" == "off" ]]; then
      "${CXX}" "${common_compile_flags[@]}" "${common_link_flags[@]}" "${extra_flags[@]}" "${pthread_flag[@]}" \
        -x c++ "${stplugin_src}" -x none "${srcs[@]}" -o "${out}"
    else
      set +e
      "${CXX}" "${common_compile_flags[@]}" "${common_link_flags[@]}" "${extra_flags[@]}" "${pthread_flag[@]}" \
        -DHDFE_USE_OPENMP -fopenmp -x c++ "${stplugin_src}" -x none "${srcs[@]}" -o "${out}"
      rc=$?
      set -e
      if [[ $rc -ne 0 ]]; then
        echo "OpenMP build failed; retrying without OpenMP..."
        "${CXX}" "${common_compile_flags[@]}" "${common_link_flags[@]}" "${extra_flags[@]}" "${pthread_flag[@]}" \
          -x c++ "${stplugin_src}" -x none "${srcs[@]}" -o "${out}"
      fi
    fi
  fi
}

if [[ "${UNAME_S}" == "Darwin" && "${TARGET}" != "windows" ]]; then
  echo "Building ${OUT_PLUGIN} (universal: x86_64 + arm64)"
  tmp_x86="${BUILD_DIR}/xhdfe.plugin.x86_64"
  tmp_arm="${BUILD_DIR}/xhdfe.plugin.arm64"
  compile_plugin "${tmp_x86}" -target x86_64-apple-macos10.12 "${metal_flags[@]}"
  compile_plugin "${tmp_arm}" -target arm64-apple-macos11 "${metal_flags[@]}"
  lipo -create -output "${OUT_PLUGIN}" "${tmp_x86}" "${tmp_arm}"
else
  echo "Building ${OUT_PLUGIN}"
  compile_plugin "${OUT_PLUGIN}" "${metal_flags[@]}"
fi

if command -v "${STRIP_BIN}" >/dev/null 2>&1; then
  if [[ "${UNAME_S}" == "Darwin" && "${TARGET}" != "windows" ]]; then
    "${STRIP_BIN}" -x "${OUT_PLUGIN}" || true
  else
    "${STRIP_BIN}" "${OUT_PLUGIN}" || true
  fi
fi

echo "Done: ${OUT_PLUGIN}"
