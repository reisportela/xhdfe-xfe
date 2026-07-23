#!/usr/bin/env bash
# Build xhdfe-src.zip: a complete, self-contained source distribution that
# compiles ALL xhdfe/xfe components --- the C++ core, the Stata plugins, the
# Python package, and the R package --- for CPU and GPU (CUDA), on a machine
# WITHOUT internet access.
#
# Every package-side dependency is vendored (Eigen, pybind11, Rcpp, and Stata's
# stplugin inputs), so no network download is needed. Only a system toolchain
# is required at build time (C++ compiler, CMake, Python + dev headers, R, and
# the CUDA toolkit/driver for GPU builds).
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

VERSION="$(grep -m1 '^\*! version' "${ROOT_DIR}/stata/xhdfe.ado" | sed -E 's/^\*! version[[:space:]]+//')"
[[ -n "${VERSION}" ]] || { echo "could not read version from stata/xhdfe.ado" >&2; exit 1; }

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

# ---- Stata plugin sources + build scripts + vendored deps -----------------
copy_tree "${ROOT_DIR}/stata/src"     "${PKG}/stata/src"
copy_tree "${ROOT_DIR}/stata/include" "${PKG}/stata/include"
mkdir -p "${PKG}/stata/tools/_deps"
for f in build-plugin.sh build-xfe-plugin.sh cuda-common.sh mingw_stdio_shim.h; do
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

# ---- Python package -------------------------------------------------------
copy_tree "${ROOT_DIR}/python" "${PKG}/python"
copy_tree "${ROOT_DIR}/xhdfe"  "${PKG}/xhdfe"
cp -a "${ROOT_DIR}/setup.py" "${ROOT_DIR}/pyproject.toml" "${ROOT_DIR}/CMakeLists.txt" "${PKG}/"

# ---- Project documentation and notices -----------------------------------
cp -a "${ROOT_DIR}/README.md" "${ROOT_DIR}/LICENSE" "${ROOT_DIR}/NOTICE" \
      "${ROOT_DIR}/CITATION.cff" "${PKG}/"

# ---- R package (self-contained: src mirror + its own vendored Eigen) ------
copy_tree "${ROOT_DIR}/r/xhdfe" "${PKG}/r/xhdfe"

# ---- Vendored third-party for CMake/Python (offline) ----------------------
copy_tree "${ROOT_DIR}/third_party/eigen-3.4.0"      "${PKG}/third_party/eigen-3.4.0"
copy_tree "${ROOT_DIR}/third_party/pybind11-2.11.1"  "${PKG}/third_party/pybind11-2.11.1"

# ---- Vendored R dependency (official CRAN source, offline) ----------------
# Keep Rcpp as its upstream source archive rather than unpacking or modifying
# it. BUILD_OFFLINE.md installs that exact archive into a local R library.
cp -a "${ROOT_DIR}/third_party/Rcpp_1.1.2.tar.gz" \
      "${ROOT_DIR}/third_party/RCPP_SOURCE_PROVENANCE.md" \
      "${PKG}/third_party/"

printf '%s\n' "${VERSION}" > "${PKG}/VERSION"

cat > "${PKG}/BUILD_OFFLINE.md" <<EOF
# xhdfe / xfe --- offline source distribution (version ${VERSION})

Self-contained sources to build every component for CPU and GPU with **no
internet access**. All package-side dependencies are vendored (Eigen,
pybind11, the official Rcpp 1.1.2 CRAN source archive, and Stata
\`stplugin\` inputs).

## System requirements (not bundled)

- A C++17 compiler (GCC/Clang; MSVC on Windows).
- For GPU: the NVIDIA CUDA toolkit (\`nvcc\`) and a compatible driver.
- Python builds: Python >= 3.9 with development headers and CMake >= 3.18.
- R builds: R >= 4.0 and its source-package build toolchain. Rcpp itself is
  bundled and does not need to be downloaded or preinstalled.
- Stata builds: none beyond the C++ (and CUDA) toolchain.

## Stata plugin

\`\`\`bash
# CPU (OpenMP recommended)
bash stata/tools/build-plugin.sh     --linux --openmp
bash stata/tools/build-xfe-plugin.sh --linux --openmp
# GPU (auto-detect the local NVIDIA architecture)
XHDFE_ENABLE_CUDA=auto bash stata/tools/build-plugin.sh     --linux --openmp
XHDFE_ENABLE_CUDA=auto bash stata/tools/build-xfe-plugin.sh --linux --openmp
\`\`\`

Produces \`stata/xhdfe.plugin\` and \`stata/xfe.plugin\`. Put them on the Stata
adopath (next to \`xhdfe.ado\`). The \`xhdfegpu\` command automates the GPU case.

## Python package

\`\`\`bash
python -m pip install .                      # CPU
XHDFE_ENABLE_CUDA=auto python -m pip install .   # GPU
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
