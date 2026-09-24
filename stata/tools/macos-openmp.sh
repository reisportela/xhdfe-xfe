#!/usr/bin/env bash
# Sourced by both plugin builders; compatible with macOS Bash 3.2.

xhdfe_macos_architectures() {
  read -r -a macos_archs <<< "${XHDFE_MACOS_ARCHS:-x86_64 arm64}"
  [[ ${#macos_archs[@]} -ge 1 && ${#macos_archs[@]} -le 2 ]] || {
    echo "Error: XHDFE_MACOS_ARCHS must select x86_64, arm64, or both." >&2
    return 1
  }
  local previous="" arch
  for arch in "${macos_archs[@]}"; do
    case "$arch" in x86_64|arm64) ;; *) echo "Unsupported macOS architecture: $arch" >&2; return 1 ;; esac
    [[ "$arch" != "$previous" ]] || { echo "Duplicate macOS architecture: $arch" >&2; return 1; }
    previous="$arch"
  done
}

xhdfe_macos_openmp_flags() {
  local arch="$1" target="$2"
  openmp_compile_flags=()
  openmp_link_flags=()
  [[ "$OPENMP_MODE" == "on" ]] || return 0
  [[ -n "${XHDFE_OPENMP_ROOT:-}" ]] || {
    echo "Error: macOS OpenMP requires XHDFE_OPENMP_ROOT with an SDK for each requested architecture." >&2
    echo "Build the pinned SDK with tools/build_macos_openmp_runtime.sh." >&2
    return 1
  }
  local prefix="${XHDFE_OPENMP_ROOT}/${arch}"
  local runtime="${prefix}/lib/xhdfe_libomp.dylib"
  [[ -f "${prefix}/include/omp.h" && -f "$runtime" ]] || {
    echo "Error: missing macOS OpenMP header/runtime for ${arch} under ${prefix}." >&2
    return 1
  }
  python3 "${STATA_DIR}/../tools/validate_macos_openmp.py" inspect \
    --file "$runtime" --arch "$arch" --kind runtime
  openmp_compile_flags=( -DHDFE_USE_OPENMP -Xpreprocessor -fopenmp "-I${prefix}/include" )
  openmp_link_flags=( "$runtime" '-Wl,-rpath,@loader_path' )
  printf 'OpenMP compile flags (%s):' "$arch"
  printf ' %q' "${openmp_compile_flags[@]}"
  printf '\nOpenMP link flags (%s):' "$arch"
  printf ' %q' "${openmp_link_flags[@]}"
  printf '\n'
  local probe="${BUILD_DIR}/openmp-toolchain-${arch}"
  cat > "${probe}.cpp" <<'CPP'
#include <omp.h>
#ifndef _OPENMP
#error OpenMP was requested but is not enabled by the compiler
#endif
int main() {
    omp_set_dynamic(0);
    int workers = 0;
#pragma omp parallel num_threads(2) reduction(+:workers)
    workers += 1;
    return workers == 2 ? 0 : 1;
}
CPP
  "${CXX}" -std=c++17 -target "$target" "${openmp_compile_flags[@]}" \
    "${probe}.cpp" "$runtime" "-Wl,-rpath,${prefix}/lib" -o "$probe"
  if [[ "$(uname -m)" == "$arch" ]]; then
    "$probe"
  fi
}

xhdfe_macos_bundle_runtime() {
  [[ "$OPENMP_MODE" == "on" ]] || return 0
  local arch
  local inputs=()
  for arch in "${macos_archs[@]}"; do
    inputs+=( "${XHDFE_OPENMP_ROOT}/${arch}/lib/xhdfe_libomp.dylib" )
  done
  local staged="${BUILD_DIR}/xhdfe_libomp.dylib"
  lipo -create -output "$staged" "${inputs[@]}"
  codesign --force --sign - --timestamp=none --identifier xhdfe_libomp "$staged"
  local destination="${STATA_DIR}/xhdfe_libomp.dylib"
  if [[ -e "$destination" ]]; then
    cmp -s "$staged" "$destination" || {
      echo "Error: an existing xhdfe_libomp.dylib differs; use a fresh build staging tree." >&2
      return 1
    }
  else
    cp "$staged" "$destination"
  fi
}
