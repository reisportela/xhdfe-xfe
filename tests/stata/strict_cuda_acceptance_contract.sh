#!/bin/bash
# Package B (18nov2026): the Stata leg of the strict-residual CUDA acceptance contract
# (tests/stata/strict_cuda_acceptance_contract.do), driven by
# tests/analytic_precision/strict_cuda_acceptance_contract.py --only stata: it writes the synthetic fixtures and
# their oracles, runs one fresh stata-mp session per leg (each with its own STATATMP, checked empty afterwards)
# and grades the logs and the trace. A visible CUDA device and the public cov240k oracle file are required; the
# reference package adds the comparisons with today's numbers and the excluded fits (CPU bit for bit, CUDA A/A
# against A/B); a reghdfe adopath adds the four-effect covariance against reghdfe at tolerance 1e-12.
# usage: strict_cuda_acceptance_contract.sh <package stata dir> <new scratch dir> <CUDA python module>
#        <oracle_cov240k.npz> [<reference package stata dir>] [<reghdfe adopath>]
set -u
PKG=${1:?package stata directory}
OUT=${2:?new scratch directory}
MOD=${3:?CUDA python module}
NPZ=${4:?public cov240k oracle}
REF=${5:-}
RGH=${6:-}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
args=(--module "$MOD" --v3-npz "$NPZ" --stata-package "$PKG" --only stata --keep "$OUT" --output "$OUT.json")
[ -n "$REF" ] && args+=(--reference-stata-package "$REF")
[ -n "$RGH" ] && args+=(--reghdfe-adopath "$RGH")
exec env -u LD_LIBRARY_PATH python3 -B "$HERE/../analytic_precision/strict_cuda_acceptance_contract.py" "${args[@]}"
