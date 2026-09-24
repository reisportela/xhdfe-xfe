#!/usr/bin/env bash
# Build xhdfe-src.zip: a complete, self-contained source distribution that
# compiles ALL xhdfe/xfepout components --- the C++ core, the Stata plugins, the
# Python package, and the R package --- for CPU and GPU (CUDA), on a machine
# WITHOUT internet access.
#
# Native build dependencies are vendored (Eigen, pybind11, Rcpp, and Stata's
# stplugin inputs), so no network download is needed for those build paths.
# Python runtime packages such as NumPy, and the optional Formulaic stack, must
# already be available in the offline Python environment.
#
# The archive is published to the gh-pages net-install site and fetched by the
# `xhdfegpu` Stata command, which builds a CUDA plugin for the local GPU and
# installs it over the CPU plugin obtained from `net install`. The same archive
# lets anyone rebuild any component offline (see BUILD_OFFLINE.md inside it).
#
# Usage: bash tools/make_source_dist_zip.sh [OUT_ZIP]
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ZIP="${1:-${ROOT_DIR}/xhdfe-src.zip}"
# Resolve OUT_ZIP to an absolute path: the archive is written after a
# `cd "${STAGE}"`, so a relative path (as passed by CI) would otherwise resolve
# against the wrong directory. Also ensure its parent directory exists.
mkdir -p "$(dirname -- "${OUT_ZIP}")"
OUT_ZIP="$(cd -- "$(dirname -- "${OUT_ZIP}")" && pwd)/$(basename -- "${OUT_ZIP}")"

VERSION="$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "${ROOT_DIR}/xhdfe/_version.py")"
[[ -n "${VERSION}" ]] || { echo "could not read version from xhdfe/_version.py" >&2; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
PKG="${STAGE}/xhdfe-src"
mkdir -p "${PKG}"

# rsync-style copy that skips build artifacts and VCS/cache dirs.
copy_tree() {
  local src="$1" dst="$2"
  mkdir -p "${dst}"
  tar -C "${src}" \
      --exclude='.git' --exclude='_build' --exclude='__pycache__' \
      --exclude='*.o' --exclude='*.so' --exclude='*.dll' --exclude='*.dylib' \
      --exclude='*.plugin' --exclude='*.a' --exclude='*.pyc' \
      -cf - . | tar -C "${dst}" -xf -
}

# ---- Shared C++ core ------------------------------------------------------
copy_tree "${ROOT_DIR}/src"     "${PKG}/src"
copy_tree "${ROOT_DIR}/include" "${PKG}/include"

# CMake and the CUDA plugin builders execute repository-level scientific
# gates, and the default CMake configuration also compiles the test probes.
# Keep those inputs in the source archive instead of validating only the
# vendored third-party dependencies.
copy_tree "${ROOT_DIR}/tools" "${PKG}/tools"
copy_tree "${ROOT_DIR}/tests" "${PKG}/tests"

# The included R/Stata certification suites and the documented cross-frontend
# examples depend on these small tracked fixtures. Keep them in the autonomous
# archive so its test surface is as complete as the repository checkout.
copy_tree "${ROOT_DIR}/data"     "${PKG}/data"
copy_tree "${ROOT_DIR}/examples" "${PKG}/examples"

# ---- Stata plugin sources + build scripts + vendored deps -----------------
copy_tree "${ROOT_DIR}/stata/src"     "${PKG}/stata/src"
copy_tree "${ROOT_DIR}/stata/include" "${PKG}/stata/include"
mkdir -p "${PKG}/stata/tools/_deps"
for f in build-plugin.sh build-xfepout-plugin.sh cuda-common.sh macos-openmp.sh mingw_stdio_shim.h; do
  cp -a "${ROOT_DIR}/stata/tools/${f}" "${PKG}/stata/tools/${f}"
done
# Vendored Stata plugin dependencies (offline): Eigen tarball + stplugin SDK.
cp -a "${ROOT_DIR}/stata/tools/_deps/eigen-3.4.0.tar.gz" "${PKG}/stata/tools/_deps/"
cp -a "${ROOT_DIR}/stata/tools/_deps/stplugin.h" "${PKG}/stata/tools/_deps/"
cp -a "${ROOT_DIR}/stata/tools/_deps/stplugin.c" "${PKG}/stata/tools/_deps/"

# Installable Stata surface. Plugins are intentionally excluded here because
# the autonomous release bundle combines this source tree with the native
# Linux/Windows/macOS binaries from the generated net-install site.
for f in "${ROOT_DIR}"/stata/*.ado "${ROOT_DIR}"/stata/*.sthlp \
         "${ROOT_DIR}"/stata/*.pkg "${ROOT_DIR}"/stata/stata.toc; do
  cp -a "$f" "${PKG}/stata/"
done
cp -a "${ROOT_DIR}/stata/LICENSE" "${ROOT_DIR}/stata/NOTICE" "${PKG}/stata/"

# ---- Python package -------------------------------------------------------
copy_tree "${ROOT_DIR}/python" "${PKG}/python"
copy_tree "${ROOT_DIR}/xhdfe"  "${PKG}/xhdfe"
cp -a "${ROOT_DIR}/setup.py" "${ROOT_DIR}/pyproject.toml" \
      "${ROOT_DIR}/MANIFEST.in" "${ROOT_DIR}/py_hdfe_v11.py" \
      "${ROOT_DIR}/CMakeLists.txt" "${PKG}/"

# ---- Project documentation and notices -----------------------------------
cp -a "${ROOT_DIR}/README.md" "${ROOT_DIR}/LICENSE" "${ROOT_DIR}/NOTICE" \
      "${ROOT_DIR}/CITATION.cff" "${PKG}/"

# ---- R package (self-contained: src mirror + its own vendored Eigen) ------
copy_tree "${ROOT_DIR}/r/xhdfe" "${PKG}/r/xhdfe"

# ---- Vendored third-party for CMake/Python (offline) ----------------------
copy_tree "${ROOT_DIR}/third_party/eigen-3.4.0"      "${PKG}/third_party/eigen-3.4.0"
copy_tree "${ROOT_DIR}/third_party/pybind11-2.11.1"  "${PKG}/third_party/pybind11-2.11.1"
copy_tree "${ROOT_DIR}/third_party/licenses"          "${PKG}/third_party/licenses"

# ---- Vendored R dependency (official CRAN source, offline) ----------------
# Keep Rcpp as its upstream source archive rather than unpacking or modifying
# it. BUILD_OFFLINE.md installs that exact archive into a local R library.
cp -a "${ROOT_DIR}/third_party/Rcpp_1.1.2.tar.gz" \
      "${ROOT_DIR}/third_party/RCPP_SOURCE_PROVENANCE.md" \
      "${PKG}/third_party/"

MACOS_OPENMP_SOURCE="${XHDFE_MACOS_OPENMP_SOURCE:-}"
if [[ "${XHDFE_RELEASE_BUILD:-0}" == "1" && -z "${MACOS_OPENMP_SOURCE}" ]]; then
  echo "XHDFE_RELEASE_BUILD=1 requires XHDFE_MACOS_OPENMP_SOURCE" >&2
  exit 1
fi
if [[ -n "${MACOS_OPENMP_SOURCE}" ]]; then
  [[ -f "${MACOS_OPENMP_SOURCE}" && ! -L "${MACOS_OPENMP_SOURCE}" ]] || {
    echo "invalid XHDFE_MACOS_OPENMP_SOURCE: ${MACOS_OPENMP_SOURCE}" >&2
    exit 1
  }
  tar -tzf "${MACOS_OPENMP_SOURCE}" > "${STAGE}/macos-openmp-source.list"
  grep -Eq '(^|/)openmp/' "${STAGE}/macos-openmp-source.list" || {
    echo "macOS OpenMP source archive lacks pinned openmp sources" >&2
    exit 1
  }
  grep -Eq '(^|/)cmake/' "${STAGE}/macos-openmp-source.list" || {
    echo "macOS OpenMP source archive lacks the LLVM cmake support tree" >&2
    exit 1
  }
  cp -a "${MACOS_OPENMP_SOURCE}" \
    "${PKG}/third_party/LLVM-OpenMP-20.1.8-source.tar.gz"
fi

printf '%s\n' "${VERSION}" > "${PKG}/VERSION"

cat > "${PKG}/BUILD_OFFLINE.md" <<EOF
# xhdfe / xfepout --- offline source distribution (version ${VERSION})

Self-contained sources to build every native component for CPU and GPU with
**no internet access**. Native package dependencies are vendored (Eigen,
pybind11, the official Rcpp 1.1.2 CRAN source archive, and Stata \`stplugin\`
inputs). Python runtime dependencies must be present in the local environment.

## System requirements (not bundled)

- A C++17 compiler (GCC/Clang; MSVC on Windows).
- For GPU: the NVIDIA CUDA toolkit (\`nvcc\`) and a compatible driver.
- Python builds: Python >= 3.9 with development headers, CMake >= 3.18, and
  NumPy >= 1.23 already installed.
- Optional Python formula interface: Formulaic >= 1.2.1,<2 and its pandas/SciPy
  dependencies already installed; these optional runtime packages are not
  vendored in this archive.
- R builds: R >= 4.0, its source-package build toolchain, and OpenMP support.
  Rcpp itself is bundled and does not need to be downloaded or preinstalled.
  \`XHDFE_ALLOW_SERIAL_BUILD=1\` is reserved for explicit diagnostic builds and
  is forbidden for releases.
- Stata builds: none beyond the C++ (and CUDA) toolchain.
- macOS Stata release builds: CMake >= 3.20, Python 3, the native Apple build
  tools, and the pinned LLVM OpenMP source archive in
  \`third_party/LLVM-OpenMP-20.1.8-source.tar.gz\` on a native macOS build host.

## Stata plugin

\`\`\`bash
# CPU (OpenMP required for production)
bash stata/tools/build-plugin.sh     --linux --openmp
bash stata/tools/build-xfepout-plugin.sh --linux --openmp
# GPU (auto-detect the local NVIDIA architecture)
XHDFE_ENABLE_CUDA=auto bash stata/tools/build-plugin.sh     --linux --openmp
XHDFE_ENABLE_CUDA=auto bash stata/tools/build-xfepout-plugin.sh --linux --openmp
\`\`\`

Produces \`stata/xhdfe.plugin\` and \`stata/xfepout.plugin\`. Put them on the Stata
adopath (next to \`xhdfe.ado\`). The \`xhdfegpu\` command automates the GPU case.

On a native macOS host, extract the bundled source, build the SDK for that
host's architecture, and build both plugins against its absolute SDK path:

\`\`\`bash
mkdir llvm-openmp-20.1.8-source
tar -xzf third_party/LLVM-OpenMP-20.1.8-source.tar.gz \\
  -C llvm-openmp-20.1.8-source
sdk_root="\$PWD/macos-openmp-sdk"
arch="\$(uname -m)"
bash tools/build_macos_openmp_runtime.sh \\
  "\$PWD/llvm-openmp-20.1.8-source" "\$sdk_root" "\$arch"
XHDFE_MACOS_ARCHS="\$arch" XHDFE_OPENMP_ROOT="\$sdk_root" \\
  bash stata/tools/build-plugin.sh --openmp
XHDFE_MACOS_ARCHS="\$arch" XHDFE_OPENMP_ROOT="\$sdk_root" \\
  bash stata/tools/build-xfepout-plugin.sh --openmp
\`\`\`

The builders run \`validate_macos_openmp.py inspect\` on each native file. A
universal release requires separate native arm64 and x86_64 SDKs, followed by
the release manifest assembly and native validation on both architectures;
the SDK directory alone is not an input to the final \`verify\` command.

## Python package

\`\`\`bash
python -m pip install .                      # CPU
XHDFE_ENABLE_CUDA=auto python -m pip install .   # GPU
# Formula frontend, when its optional dependencies are available locally:
python -m pip install --no-index --no-deps --no-build-isolation '.[formula]'
\`\`\`

## R package

\`\`\`bash
mkdir -p r/Rlib

# Install the pinned dependency locally, with user/site startup files disabled.
R_PROFILE_USER=/dev/null R_ENVIRON_USER=/dev/null \\
  R_LIBS_USER="\$PWD/r/Rlib" \\
  R CMD INSTALL --library="\$PWD/r/Rlib" third_party/Rcpp_1.1.2.tar.gz

# CPU
R_PROFILE_USER=/dev/null R_ENVIRON_USER=/dev/null \\
  R_LIBS_USER="\$PWD/r/Rlib" XHDFE_ENABLE_CUDA=OFF \\
  R CMD INSTALL --library="\$PWD/r/Rlib" r/xhdfe

# GPU (optional; install into a separate local library if both builds are kept)
mkdir -p r/Rlib_cuda
R_PROFILE_USER=/dev/null R_ENVIRON_USER=/dev/null \\
  R_LIBS_USER="\$PWD/r/Rlib_cuda" \\
  R CMD INSTALL --library="\$PWD/r/Rlib_cuda" third_party/Rcpp_1.1.2.tar.gz
R_PROFILE_USER=/dev/null R_ENVIRON_USER=/dev/null \\
  R_LIBS_USER="\$PWD/r/Rlib_cuda" XHDFE_ENABLE_CUDA=auto \\
  R CMD INSTALL --library="\$PWD/r/Rlib_cuda" r/xhdfe
\`\`\`

The unmodified Rcpp archive's URL, license, version, and SHA-256 are recorded
in \`third_party/RCPP_SOURCE_PROVENANCE.md\`.

CPU is the reference backend. GPU builds auto-detect the local compute
capability via \`nvidia-smi\`; set \`XHDFE_CUDA_ARCH=90\` to force a target.
EOF

rm -f "${OUT_ZIP}"
( cd "${STAGE}" && zip -q -r -X "${OUT_ZIP}" xhdfe-src )
bash "${ROOT_DIR}/tools/validate_source_dist_zip.sh" "${OUT_ZIP}"
echo "Wrote ${OUT_ZIP} (version ${VERSION}); $(du -h "${OUT_ZIP}" | cut -f1)"
