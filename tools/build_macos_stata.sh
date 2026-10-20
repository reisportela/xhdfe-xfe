#!/usr/bin/env bash
# build_macos.sh: one-command macOS build of the isolated OpenMP runtime and
# both Stata plugins from this offline source distribution, followed by the
# export/Mach-O verifications and an installable staging folder. Runs with the
# macOS system /bin/bash 3.2.
#
# Usage: bash build_macos.sh [--arch arm64|x86_64] [--install-dir DIR]
#                            [--stata /path/to/StataMP] [--dry-run]
#
#   --arch         Slice to build; default: this host (uname -m). The OpenMP
#                  runtime must be built on a native host for the slice; on
#                  Apple Silicon an Intel slice is only a Rosetta diagnostic
#                  (run under `arch -x86_64 bash build_macos.sh --arch x86_64`).
#   --install-dir  Folder receiving the ados, help files, plugins and the
#                  runtime; default _install_macos_<arch> in the source root.
#   --stata        Stata executable (e.g.
#                  /Applications/Stata/StataMP.app/Contents/MacOS/StataMP).
#                  When given, the real-host OpenMP contract and the three
#                  functional contracts run against the staged folder.
#   --dry-run      Print every command without executing it.
#
# Requirements (not bundled): Xcode Command Line Tools (clang++, lipo,
# codesign, nm, otool), CMake >= 3.20, Python 3; for --stata also Stata with
# reghdfe and ivreghdfe installed.
set -euo pipefail

ARCH=""; INSTALL_DIR=""; STATA=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --arch|--install-dir|--stata)
      [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }
      case "$1" in
        --arch) ARCH="$2" ;;
        --install-dir) INSTALL_DIR="$2" ;;
        --stata) STATA="$2" ;;
      esac
      shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; sed -n '7,8p' "$0"; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT=""
for candidate in "$SCRIPT_DIR" "$SCRIPT_DIR/.."; do
  if [ -f "$candidate/stata/tools/build-plugin.sh" ]; then
    ROOT="$(cd "$candidate" && pwd)"
    break
  fi
done
[ -n "$ROOT" ] || { echo "cannot find the source root (stata/tools/build-plugin.sh) near $SCRIPT_DIR" >&2; exit 1; }

run() {
  echo "+ $*"
  if [ "$DRY" -eq 0 ]; then "$@"; fi
}

if [ "$DRY" -eq 0 ]; then
  [ "$(uname -s)" = "Darwin" ] || { echo "this script builds macOS artifacts; run it on a Mac" >&2; exit 1; }
  for tool in clang++ cmake python3 lipo codesign nm otool tar shasum; do
    command -v "$tool" >/dev/null 2>&1 || {
      echo "missing tool: $tool (xcode-select --install; CMake >= 3.20 from cmake.org or Homebrew)" >&2
      exit 1
    }
  done
fi
[ -n "$ARCH" ] || ARCH="$(uname -m)"
case "$ARCH" in
  arm64|x86_64) ;;
  *) echo "unsupported --arch $ARCH (arm64 or x86_64)" >&2; exit 2 ;;
esac
if [ "$DRY" -eq 0 ] && [ "$(uname -m)" != "$ARCH" ]; then
  echo "the OpenMP runtime must be built on a native $ARCH host (or under 'arch -$ARCH bash' as a Rosetta diagnostic)" >&2
  exit 1
fi

BUILD="$ROOT/_build_macos_$ARCH"
LLVM_TGZ="$ROOT/third_party/LLVM-OpenMP-20.1.8-source.tar.gz"
LLVM_SRC="$BUILD/llvm-openmp-20.1.8-source"
SDK="$BUILD/macos-openmp-sdk"
[ -n "$INSTALL_DIR" ] || INSTALL_DIR="$ROOT/_install_macos_$ARCH"
case "$INSTALL_DIR" in
  /*) ;;
  *) INSTALL_DIR="$PWD/$INSTALL_DIR" ;;
esac
[ -f "$LLVM_TGZ" ] || { echo "missing bundled LLVM OpenMP source: $LLVM_TGZ" >&2; exit 1; }

echo "== xhdfe macOS build: arch=$ARCH root=$ROOT"
run mkdir -p "$BUILD"
if [ ! -f "$LLVM_SRC/openmp/CMakeLists.txt" ]; then
  run mkdir -p "$LLVM_SRC"
  run tar -xzf "$LLVM_TGZ" -C "$LLVM_SRC"
fi

echo "== 1/4 OpenMP runtime SDK (pinned LLVM 20.1.8, export list tools/macos_openmp.exports)"
if [ -f "$SDK/$ARCH/lib/xhdfe_libomp.dylib" ]; then
  echo "reusing the existing SDK slice $SDK/$ARCH"
else
  if [ -e "$SDK/$ARCH" ]; then
    echo "incomplete SDK slice at $SDK/$ARCH; move it away and rerun" >&2
    exit 1
  fi
  run bash "$ROOT/tools/build_macos_openmp_runtime.sh" "$LLVM_SRC" "$SDK" "$ARCH"
fi

echo "== 2/4 Stata plugins"
# The plugin builder refuses to overwrite a bundled runtime with different
# bytes; keep any previous copy aside instead of deleting it.
if [ -f "$ROOT/stata/xhdfe_libomp.dylib" ]; then
  run mkdir -p "$BUILD/previous"
  run mv "$ROOT/stata/xhdfe_libomp.dylib" "$BUILD/previous/xhdfe_libomp.dylib.$(date +%Y%m%dT%H%M%S)"
fi
run env XHDFE_MACOS_ARCHS="$ARCH" XHDFE_OPENMP_ROOT="$SDK" bash "$ROOT/stata/tools/build-plugin.sh" --openmp
run env XHDFE_MACOS_ARCHS="$ARCH" XHDFE_OPENMP_ROOT="$SDK" bash "$ROOT/stata/tools/build-xfepout-plugin.sh" --openmp

echo "== 3/4 Verification (export isolation, Mach-O layout, weak definitions)"
run python3 "$ROOT/tools/check_macos_openmp_exports.py" --runtime "$ROOT/stata/xhdfe_libomp.dylib" --arch "$ARCH" \
  --plugin "$ROOT/stata/xhdfe.plugin" --plugin "$ROOT/stata/xfepout.plugin"
run python3 "$ROOT/tools/validate_macos_openmp.py" inspect --file "$ROOT/stata/xhdfe_libomp.dylib" --arch "$ARCH" --kind runtime
run python3 "$ROOT/tools/validate_macos_openmp.py" inspect --file "$ROOT/stata/xhdfe.plugin" --arch "$ARCH" --kind plugin
run python3 "$ROOT/tools/validate_macos_openmp.py" inspect --file "$ROOT/stata/xfepout.plugin" --arch "$ARCH" --kind plugin
if [ "$DRY" -eq 0 ]; then
  # Defined external weak symbols only (-g -U): undefined weak imports from
  # libSystem, such as _pthread_mutexattr_settype, are legitimate.
  weak="$(nm -m -g -U -arch "$ARCH" "$ROOT/stata/xhdfe_libomp.dylib" | grep -c 'weak external' || true)"
  echo "runtime defined weak external symbols: $weak (expected 0)"
  [ "$weak" = "0" ] || { echo "the runtime still exports weak definitions; do not install it" >&2; exit 1; }
  otool -hv "$ROOT/stata/xhdfe_libomp.dylib"
fi

echo "== 4/4 Staging $INSTALL_DIR"
run mkdir -p "$INSTALL_DIR"
for f in "$ROOT"/stata/*.ado "$ROOT"/stata/*.sthlp "$ROOT/stata/xhdfe.plugin" "$ROOT/stata/xfepout.plugin" "$ROOT/stata/xhdfe_libomp.dylib"; do
  run cp "$f" "$INSTALL_DIR/"
done
if [ "$DRY" -eq 0 ]; then
  (cd "$INSTALL_DIR" && shasum -a 256 xhdfe.plugin xfepout.plugin xhdfe_libomp.dylib xhdfe.ado xfepout.ado > INSTALL_MANIFEST.sha256)
  cat "$INSTALL_DIR/INSTALL_MANIFEST.sha256"
fi

if [ -n "$STATA" ]; then
  echo "== Stata validation with $STATA"
  STAMP="$(date +%Y%m%dT%H%M%S)"
  run python3 "$ROOT/tools/run_stata_openmp_host.py" --stata "$STATA" --adopath "$INSTALL_DIR" \
    --arch "$ARCH" --threads 1 2 4 0 --out "$BUILD/host-receipt-$STAMP"
  WORK="$BUILD/contracts-$STAMP"
  run mkdir -p "$WORK"
  for t in version_contract absorb_interaction_contract cluster_interaction_contract; do
    echo "+ (cd $WORK && XHDFE_STATA_ADOPATH=$INSTALL_DIR $STATA -b do $ROOT/tests/stata/$t.do cpu)"
    if [ "$DRY" -eq 0 ]; then
      (cd "$WORK" && env XHDFE_STATA_ADOPATH="$INSTALL_DIR" "$STATA" -b do "$ROOT/tests/stata/$t.do" cpu)
      if [ "$t" = "version_contract" ]; then
        for cmd in xhdfe xfepout; do
          grep -Fxq "VERSION_CONTRACT_SOURCE_OK $cmd" "$WORK/$t.log" || {
            echo "$t: $cmd.ado was not loaded from $INSTALL_DIR (see $WORK/$t.log)" >&2
            exit 1
          }
        done
      fi
      if grep -q "CONTRACT_PASS cpu" "$WORK/$t.log"; then
        echo "$t: PASS"
      else
        echo "$t: FAIL (see $WORK/$t.log)" >&2
        exit 1
      fi
    fi
  done
  echo "receipts: $BUILD/host-receipt-$STAMP/RESULT.json and $WORK/*.log"
fi

cat <<MSG

Done. In Stata:   adopath ++ "$INSTALL_DIR"
                  xhdfe, version        (expected: the version in stata/xhdfe.ado)
Report back: INSTALL_MANIFEST.sha256, the verification output above, sw_vers,
and (with --stata) the RESULT.json and contract logs.
MSG
