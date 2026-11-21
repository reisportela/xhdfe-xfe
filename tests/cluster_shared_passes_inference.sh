#!/bin/bash
# Phase 5b of the Stata-path plan: builds tests/cluster_shared_passes_inference.cpp
# against the branch's core library (with -DCSP_NEW_API when the branch has the
# carrier) and against the frozen one (with the frozen headers, FROZEN_INCLUDE,
# ahead of the worktree's), runs new (XHDFE_CLUSTER_SHARED_PASSES unset), off (=0,
# a fresh process) and frozen (=0 as well), all with XHDFE_PROFILE_CPU=1, and
# requires: every CMP line identical in the three arms, every CHECK line ok, and the
# controls on stderr (off and frozen: none reading a carrier; new with the carrier:
# the two failure cases end in the wide route with the carrier empty). The frozen
# library is a previous build that already has phase 5b (17nov2026, point 27: a
# build from before it is no longer comparable in bits with today's code); with the
# switch at 0 it takes today's path, as the off arm does.
#
#   usage: cluster_shared_passes_inference.sh <new libxhdfe.a> <frozen libxhdfe.a> <new scratch dir> [api 1|0]
#   api 0           the branch library has no carrier (a build from before phase 5b):
#                   its new arm must then print no cluster_passes line
#   FROZEN_INCLUDE  a directory with the frozen build's include/ (ols.hpp, fe_absorption.hpp,
#                   hdfe/), from git show or git archive; needed whenever the headers changed
#                   since that build, so that the frozen arm never mixes headers and library
#   EXTRA_LIBS      extra link flags (the CUDA runtime for a CUDA build's library)
#   DUMP_BACKEND    cuda: the post-absorption inputs of part H come from a GPU absorption
# Never set OMP_THREAD_LIMIT here.
set -u
NEW=${1:?new core library}
BASE=${2:?frozen core library}
OUT=${3:?new scratch directory}
API=${4:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" || exit 2
F="-std=gnu++17 -O2 -march=native -fopenmp -DHDFE_USE_OPENMP -isystem $ROOT/r/xhdfe/src/eigen"
newflags="$F -I$ROOT/include -I$ROOT/src"
[ "$API" = 1 ] && newflags="$newflags -DCSP_NEW_API"
baseflags="$F ${FROZEN_INCLUDE:+-I$FROZEN_INCLUDE} -I$ROOT/include -I$ROOT/src"
c++ $newflags "$HERE/cluster_shared_passes_inference.cpp" "$NEW" ${EXTRA_LIBS:-} -o "$OUT/csp_new" \
  > "$OUT/compile_new.log" 2>&1 || { echo "FAIL compile new"; tail -n 20 "$OUT/compile_new.log"; exit 1; }
c++ $baseflags "$HERE/cluster_shared_passes_inference.cpp" "$BASE" ${EXTRA_LIBS:-} -o "$OUT/csp_frozen" \
  > "$OUT/compile_frozen.log" 2>&1 || { echo "FAIL compile frozen"; tail -n 20 "$OUT/compile_frozen.log"; exit 1; }
clean=(env -u XHDFE_CLUSTER_SHARED_PASSES -u OMP_THREAD_LIMIT -u OMP_NUM_THREADS -u XHDFE_CERTIFY
       -u XHDFE_SPARSE_RANK -u XHDFE_REDUCED_PCG -u XHDFE_GPU_BACKEND XHDFE_PROFILE_CPU=1)
status=0
# H: one absorption (DUMP_BACKEND=cuda for a CUDA build: real GPU use required),
# written once and read by every arm.
"${clean[@]}" XHDFE_GPU_BACKEND="${DUMP_BACKEND:-cpu}" CSP_DUMP="$OUT/frozen_input.bin" "$OUT/csp_new" \
  > "$OUT/dump.txt" 2> "$OUT/dump_err.txt"; echo "dump exit=$? $(cat "$OUT/dump.txt")"
if [ "${DUMP_BACKEND:-cpu}" = cuda ] && ! grep -q 'gpu_used 1' "$OUT/dump.txt"; then
  echo "FAIL dump: no real GPU use"; status=1
fi
clean+=("CSP_FROZEN_INPUT=$OUT/frozen_input.bin")
"${clean[@]}" "$OUT/csp_new" > "$OUT/new.txt" 2> "$OUT/new_err.txt"; echo "new exit=$?"
"${clean[@]}" XHDFE_CLUSTER_SHARED_PASSES=0 "$OUT/csp_new" > "$OUT/off.txt" 2> "$OUT/off_err.txt"; echo "off exit=$?"
"${clean[@]}" XHDFE_CLUSTER_SHARED_PASSES=0 "$OUT/csp_frozen" > "$OUT/frozen.txt" 2> "$OUT/frozen_err.txt"; echo "frozen exit=$?"
for arm in new off; do
  if diff <(grep '^CMP ' "$OUT/$arm.txt") <(grep '^CMP ' "$OUT/frozen.txt") > "$OUT/diff_${arm}_frozen.txt" \
     && [ "$(grep -c '^CMP ' "$OUT/frozen.txt")" -gt 10 ]; then
    echo "PASS ${arm}_vs_frozen ($(grep -c '^CMP ' "$OUT/frozen.txt") CMP lines)"
  else
    echo "FAIL ${arm}_vs_frozen"; head -n 10 "$OUT/diff_${arm}_frozen.txt"; status=1
  fi
done
for arm in new off frozen; do
  bad=$(grep -c '^CHECK .* FAIL$' "$OUT/$arm.txt")
  oks=$(grep -c '^CHECK .* ok$' "$OUT/$arm.txt")
  grep -q '^DONE failures 0$' "$OUT/$arm.txt" && [ "$bad" -eq 0 ] && echo "PASS checks $arm ($oks ok)" \
    || { echo "FAIL checks $arm"; grep '^CHECK .* FAIL$\|^DONE' "$OUT/$arm.txt"; status=1; }
done
# Controls on stderr.
for arm in off frozen; do
  grep 'label=cluster_passes' "$OUT/${arm}_err.txt" | grep -q 'counts=carrier\|pair=shared' \
    && { echo "FAIL control: the switch off read a carrier ($arm)"; status=1; }
done
if [ "$API" = 1 ]; then
  wide=$(grep -c 'label=cluster_passes site=ols .*route=wide ready=0 pair_rows=0' "$OUT/new_err.txt")
  [ "$wide" -ge 2 ] && echo "PASS control: wide route after the failures, carrier empty ($wide lines)" \
    || { echo "FAIL control: $wide wide-route lines with an empty carrier in new (want >= 2)"; status=1; }
  grep -q 'label=cluster_passes site=recon counts=carrier pair=shared' "$OUT/new_err.txt" \
    && echo "PASS control: the reconstruction read the shared pair" \
    || { echo "FAIL control: no reconstruction reading the shared pair in new"; status=1; }
  # Part D: the reconstruction's lines for no carrier, a foreign, an incomplete and a
  # not-ready carrier (today's code), a wrong-size pair (the counts only) and a good one.
  seq=$(sed -n '/^CSP_PART_D_BEGIN$/,/^CSP_PART_D_END$/p' "$OUT/new_err.txt" | grep 'site=recon' |
        sed -E 's/.*counts=([a-z]+) pair=([a-z]+).*/\1\/\2/' | tr '\n' ' ')
  want="recount/built recount/built recount/built recount/built carrier/built carrier/shared "
  [ "$seq" = "$want" ] && echo "PASS control: part D took today's path for every bad carrier ($seq)" \
    || { echo "FAIL control: part D sequence '$seq' != '$want'"; status=1; }
else
  grep -q 'label=cluster_passes' "$OUT/new_err.txt" && { echo "FAIL control: cluster_passes lines in a frozen new arm"; status=1; }
  echo "INFO controls: today's path in every arm (no cluster_passes line)"
fi
if [ "$status" -eq 0 ]; then echo "CSP_INFERENCE_PASS"; else echo "CSP_INFERENCE_FAIL"; fi
exit "$status"
