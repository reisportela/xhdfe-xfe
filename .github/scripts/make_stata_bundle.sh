#!/usr/bin/env bash
# Assemble a Stata net-install bundle ZIP for one platform.
# Usage: make_stata_bundle.sh <platform-tag>   e.g. linux-x86_64-cpu
# Expects stata/xhdfe.plugin and stata/xfe.plugin to already be built.
set -euo pipefail

PLATFORM="${1:?platform tag required, e.g. linux-x86_64-cpu}"
NAME="xhdfe-stata-${PLATFORM}"
OUT="${NAME}.zip"
STAGE="_bundle/${NAME}"

rm -rf "_bundle" "${OUT}"
mkdir -p "${STAGE}"

cp stata/xhdfe.ado stata/xhdfe_estat.ado stata/xhdfe_p.ado stata/xhdfe.sthlp \
   stata/xhdfeakm.ado stata/xhdfeakm.sthlp stata/xhdfeconnected.ado \
   stata/xhdfeconnected.sthlp stata/xhdfegelbach.ado stata/xhdfegelbach.sthlp \
   stata/xhdfegpu.ado stata/xhdfegpu.sthlp \
   stata/xhdfe.pkg stata/xfe.ado stata/xfe.sthlp stata/xfe.pkg stata/stata.toc \
   "${STAGE}/"
cp stata/xhdfe_hetero.ado stata/xhdfe_hetero.sthlp "${STAGE}/" 2>/dev/null || true
cp stata/xhdfe.plugin "${STAGE}/xhdfe.plugin"
cp stata/xfe.plugin "${STAGE}/xfe.plugin"

cat > "${STAGE}/INSTALL.txt" <<EOF
xhdfe / xfe -- Stata package (${PLATFORM})

From Stata, point net install at this unzipped folder:

    net install xhdfe, from("/path/to/this/folder") replace
    net install xfe,   from("/path/to/this/folder") replace

This is the CPU reference build. For GPU (NVIDIA/CUDA) acceleration on a
machine with an NVIDIA GPU, just run the companion command once after install:

    xhdfegpu

It detects the GPU, compiles a CUDA plugin for the local architecture, and
installs it over this CPU plugin. On an offline machine, download the source
zip and pass it in: xhdfegpu, zip("/path/to/xhdfe-src.zip"). To build by hand:

    XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=<your card, e.g. 90> \\
      bash stata/tools/build-plugin.sh --linux --openmp
EOF

( cd _bundle && zip -qr "../${OUT}" "${NAME}" )
echo "built ${OUT}"
ls -la "${OUT}"
