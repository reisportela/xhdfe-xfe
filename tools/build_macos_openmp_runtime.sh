#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || { echo "Usage: build_macos_openmp_runtime.sh LLVM_SOURCE SDK_ROOT ARCH" >&2; exit 2; }
source_root="$1"
sdk_root="$2"
arch="$3"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == "$arch" ]] || {
  echo "Build each OpenMP SDK on its native macOS architecture." >&2; exit 1;
}
case "$arch" in
  x86_64) deployment=10.12 ;;
  arm64) deployment=11.0 ;;
  *) echo "Unsupported architecture: $arch" >&2; exit 2 ;;
esac
[[ -f "$source_root/openmp/CMakeLists.txt" && -d "$source_root/cmake/Modules" ]] || {
  echo "LLVM source must include openmp/ and cmake/ from the pinned release." >&2; exit 1;
}
prefix="${sdk_root}/${arch}"
[[ ! -e "$prefix" ]] || { echo "SDK output already exists: $prefix" >&2; exit 1; }
mkdir -p "$prefix"
cmake -S "$source_root/openmp" -B "$prefix/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib \
  -DOPENMP_INSTALL_LIBDIR=lib -DCMAKE_OSX_ARCHITECTURES="$arch" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment" -DCMAKE_MACOSX_RPATH=ON \
  -DLIBOMP_ENABLE_SHARED=ON -DLIBOMP_ENABLE_ASSERTIONS=OFF \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF -DOPENMP_ENABLE_OMPT_TOOLS=OFF
cmake --build "$prefix/build" --parallel 2
cmake --install "$prefix/build"
cp -L "$prefix/lib/libomp.dylib" "$prefix/lib/xhdfe_libomp.dylib"
install_name_tool -id '@rpath/xhdfe_libomp.dylib' "$prefix/lib/xhdfe_libomp.dylib"
codesign --force --sign - --timestamp=none --identifier xhdfe_libomp "$prefix/lib/xhdfe_libomp.dylib"
cp "$source_root/openmp/LICENSE.TXT" "$prefix/LLVM-OpenMP-LICENSE.txt"
python3 "$(dirname "$0")/validate_macos_openmp.py" inspect \
  --file "$prefix/lib/xhdfe_libomp.dylib" --arch "$arch" --kind runtime
