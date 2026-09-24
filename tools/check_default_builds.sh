#!/usr/bin/env bash
# Refuse to benchmark the mutable default build directories unless they are
# release artifacts.  Dated build_* directories are frozen A/B controls and
# intentionally are not inspected here.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
allow_nonstandard="${XHDFE_BENCH_ALLOW_NONSTANDARD:-0}"
failed=0

violation() {
    printf 'XHDFE BENCHMARK INTEGRITY VIOLATION: %s\n' "$*" >&2
    failed=1
}

cache_value() {
    local cache="$1" key="$2"
    sed -n "s/^${key}:[^=]*=//p" "$cache" | head -n 1
}

# CMakeLists.txt adds -march=native per target (target_compile_options), not
# through CMAKE_CXX_FLAGS_RELEASE, so read the flags each target is actually
# compiled with.
target_has_flag() {
    local dir="$1" target="$2" flag="$3"
    local flags_make="$repo_root/$dir/CMakeFiles/$target.dir/flags.make"
    if [[ -f "$flags_make" ]]; then
        grep -E '^CXX_FLAGS' "$flags_make" | grep -q -- " $flag"
        return
    fi
    local commands="$repo_root/$dir/compile_commands.json"
    [[ -f "$commands" ]] && grep -F "CMakeFiles/$target.dir/" "$commands" | grep -q -- " $flag"
}

# ldd | grep libgomp also matches NVIDIA HPC SDK's libnvomp, which that SDK
# installs as libgomp.so.1 and which LD_LIBRARY_PATH can put first. Require the
# GNU runtime that users load.
check_openmp_runtime() {
    local binary="$1" resolved
    command -v ldd >/dev/null 2>&1 || return 0
    resolved="$(ldd "$binary" 2>/dev/null | awk '/libgomp/ {print $3; exit}')"
    if [[ -z "$resolved" ]]; then
        violation "${binary#$repo_root/} does not link libgomp (OpenMP required)"
    elif [[ "$(readlink -f "$resolved")" == *nvomp* ]]; then
        violation "${binary#$repo_root/} resolves libgomp to NVIDIA libnvomp ($resolved); rerun with 'env -u LD_LIBRARY_PATH' so the GNU OpenMP runtime is measured"
    fi
}

check_build() {
    local name="$1" dir="$2" require_cuda="$3"
    local cache="$repo_root/$dir/CMakeCache.txt"
    local build_type flags cuda_arch marker_count
    local -a modules=()

    if [[ ! -f "$cache" ]]; then
        violation "$dir has no CMakeCache.txt"
        return
    fi

    build_type="$(cache_value "$cache" CMAKE_BUILD_TYPE)"
    flags="$(cache_value "$cache" CMAKE_CXX_FLAGS_RELEASE)"
    [[ "$build_type" == "Release" ]] || violation "$dir CMAKE_BUILD_TYPE='$build_type' (need Release)"
    for required in -O3 -DNDEBUG; do
        [[ " $flags " == *" $required "* ]] ||
            violation "$dir CMAKE_CXX_FLAGS_RELEASE lacks $required: '$flags'"
    done
    for target in xhdfe py_hdfe_v11; do
        target_has_flag "$dir" "$target" -march=native ||
            violation "$dir target $target is not compiled with -march=native"
    done
    if [[ "$require_cuda" == 1 ]]; then
        cuda_arch="$(cache_value "$cache" CMAKE_CUDA_ARCHITECTURES)"
        [[ "$cuda_arch" == 90 ]] ||
            violation "$dir CMAKE_CUDA_ARCHITECTURES='$cuda_arch' (need 90 for local sm_90)"
    fi

    # The benchmark loader imports the top-level extension.  Nested CMake
    # test/helper targets may legitimately share the stem, so do not confuse
    # those with the default artifact.
    mapfile -t modules < <(find "$repo_root/$dir" -maxdepth 1 -type f -name 'py_hdfe_v11*.so' -print | sort)
    if [[ ${#modules[@]} -ne 1 ]]; then
        violation "$dir has ${#modules[@]} py_hdfe_v11*.so modules (need exactly one)"
        return
    fi
    marker_count="$(strings "${modules[0]}" | grep -c XHDFE_UNCAP_LARGE_N || true)"
    [[ "$marker_count" -ge 1 ]] ||
        violation "${modules[0]#$repo_root/} lacks XHDFE_UNCAP_LARGE_N provenance marker"
    check_openmp_runtime "${modules[0]}"
    printf '%s module sha256=%s path=%s\n' "$name" \
        "$(sha256sum "${modules[0]}" | cut -c1-12)" "${modules[0]#$repo_root/}"
}

check_build CPU build 0
check_build CUDA build_cuda 1

check_plugin() {
    local name="$1" plugin="$2"
    local marker_count
    if [[ ! -f "$plugin" ]]; then
        violation "${plugin#$repo_root/} is missing"
        return
    fi
    marker_count="$(strings "$plugin" | grep -c XHDFE_UNCAP_LARGE_N || true)"
    [[ "$marker_count" -ge 1 ]] ||
        violation "${plugin#$repo_root/} lacks XHDFE_UNCAP_LARGE_N provenance marker"
    check_openmp_runtime "$plugin"
    printf '%s plugin sha256=%s path=%s\n' "$name" \
        "$(sha256sum "$plugin" | cut -c1-12)" "${plugin#$repo_root/}"
}

check_plugin xhdfe "$repo_root/stata/xhdfe.plugin"
check_plugin xfepout "$repo_root/stata/xfepout.plugin"

if [[ "$failed" -ne 0 ]]; then
    if [[ "$allow_nonstandard" == 1 ]]; then
        printf 'XHDFE_BENCH_ALLOW_NONSTANDARD=1: continuing despite the violations above.\n' >&2
        exit 0
    fi
    printf 'Refusing benchmark run. Rebuild build/ and build_cuda/ as Release artifacts, or set XHDFE_BENCH_ALLOW_NONSTANDARD=1 only for a deliberate diagnostic experiment.\n' >&2
    exit 1
fi

printf 'XHDFE default build integrity OK.\n'
