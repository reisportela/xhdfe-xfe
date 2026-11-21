#!/bin/bash
# Package 3-lite (17 November 2026) through Stata (tests/stata/reduced_cuda_replacement_contract.do): every
# mode in its own fresh stata-mp process (the switch XHDFE_REDUCED_PCG_CUDA_REPLACEMENT, on unless 0, is read once per
# process), in one new scratch directory, with stderr kept (the tier's trace). Synthetic data only; a CUDA
# plugin and a visible device are required. The do-file is copied once per leg under the leg's name, so
# that each batch log carries that name (nothing is moved or renamed).
# usage: reduced_cuda_replacement_contract.sh <package stata dir> <build_p2b's package stata dir> <new scratch dir>
set -u
PKG=${1:?package stata directory}
REF=${2:?reference package stata directory}
OUT=${3:?new scratch directory}
STATA=${STATA:-stata-mp}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" || exit 2
clean=(env -u LD_LIBRARY_PATH -u XHDFE_REDUCED_PCG -u XHDFE_REDUCED_PCG_CUDA -u XHDFE_REDUCED_PCG_TRACE
       -u XHDFE_TEST_REDUCED_PCG_DECLINE -u XHDFE_TEST_REDUCED_PCG_CUDA_FATAL -u XHDFE_TEST_REDUCED_PCG_CUDA_FAIL
       -u XHDFE_REDUCED_PCG_CUDA_REPLACEMENT -u XHDFE_TEST_REDUCED_PCG_CUDA_REPLACEMENT_COMPARE
       -u XHDFE_TEST_REDUCED_PCG_CUDA_RECURRENCE_CHAT -u XHDFE_REDUCED_PCG_CHAT_STOP
       -u XHDFE_ABSORPTION_CACHE -u XHDFE_ABSORPTION_CACHE_MODE -u XHDFE_FE_STRUCTURE_CACHE
       -u XHDFE_FE_STRUCTURE_MODE -u XHDFE_MOBILITY_MODE -u XHDFE_MOBILITY_PROFILE
       -u XHDFE_GPU_BACKEND -u XHDFE_PROFILE_CPU -u XHDFE_REDUCED_PCG_COMPANION
       -u XHDFE_AUTO_ROUTING_RETRY_DIAG -u XHDFE_SAVEFE_ACCURATE_PROJECTION
       "CUDA_CACHE_PATH=$OUT/cuda_cache")
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$OUT/gpu_apps_before.txt" 2>&1
fi
status=0
# leg <name> <package dir> <env assignment>... -- <do-file arguments>...
leg() {
  local name=$1 pkg=$2; shift 2
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  cp "$HERE/reduced_cuda_replacement_contract.do" "$OUT/rp_${name}.do" || { status=1; return; }
  ( cd "$OUT" && "${clean[@]}" "XHDFE_STATA_ADOPATH=$pkg" "${vars[@]}" "$STATA" -q -b do "rp_${name}.do" "$@" \
      2> "trace_${name}.txt" )
  if grep -q "^REPLACEMENT_PASS" "$OUT/rp_${name}.log"; then
    echo "PASS $name"
  else
    echo "FAIL $name"; tail -n 25 "$OUT/rp_${name}.log"; status=1
  fi
}
count() { grep -c "$2" "$OUT/trace_$1.txt" || true; }
value() { grep "event=replacement " "$OUT/trace_$1.txt" | head -1 | tr ' ' '\n' | grep "^$2=" | cut -d= -f2; }
expect() {
  local name=$1 what=$2 got=$3 want=$4
  if [ "$got" != "$want" ]; then echo "FAIL $name trace: $what $got, expected $want"; status=1; fi
}
T=XHDFE_REDUCED_PCG_TRACE=1
D=XHDFE_AUTO_ROUTING_RETRY_DIAG=1
# The device replacement is the default since 17 November 2026: no variable is "on"; the variable at 0 is
# "off" (the kill switch); the variable at 1 (onexplicit) must give the bits of the default.
ON=XHDFE_REDUCED_PCG_CUDA_REPLACEMENT=1
OFF=XHDFE_REDUCED_PCG_CUDA_REPLACEMENT=0
CMP=XHDFE_TEST_REDUCED_PCG_CUDA_REPLACEMENT_COMPARE=1
leg off "$PKG" $T $D $OFF -- off
leg p2boff "$REF" $T $D -- p2boff
leg on "$PKG" $T $D -- on
leg offcompare "$PKG" $T $OFF $CMP -- offcompare
leg oncompare "$PKG" $T $CMP -- oncompare
leg fatal "$PKG" $T XHDFE_TEST_REDUCED_PCG_CUDA_FAIL=replacement -- fatal
leg onexplicit "$PKG" $T $ON -- onexplicit
leg compare "$PKG" -- compare
# The trace: with the switch on every replacement on the device; off, on the CPU; the comparison lines
# in both qualification legs; the fatal leg's first fit leaves no accepted final line before its second.
expect on "in force (first attempt)" "$(value on in_force)" 1
expect on "CPU replacements (first attempt)" "$(value on replacement_cpu)" 0
expect onexplicit "in force (first attempt)" "$(value onexplicit in_force)" 1
expect off "in force (first attempt)" "$(value off in_force)" 0
expect off "device replacements (first attempt)" "$(value off replacement_device)" 0
expect p2boff "event=replacement lines (build_p2b)" "$(count p2boff 'event=replacement ')" 0
for leg_name in offcompare oncompare; do
  n=$(count $leg_name "event=replacement_compare ")
  if [ "$n" -lt 1 ]; then echo "FAIL $leg_name trace: no comparison line"; status=1; fi
  echo "comparison lines $leg_name: $n; decisions differing: $(count $leg_name 'decision_differs=1');" \
       "chat unverified: $(count $leg_name ' chat_verified=0'); layer 2 margin: $(count $leg_name ' layer2=margin');" \
       "layer 2 unverified: $(count $leg_name ' layer2=unverified'); divergent: $(count $leg_name ' divergent=1')"
  expect $leg_name "divergent attempts" "$(count $leg_name ' divergent=1')" 0
done
expect off "projection published" "$(count off 'savefe_accurate_projection published=1')" 1
expect on "projection published" "$(count on 'savefe_accurate_projection published=1')" 1
expect fatal "accepted CUDA fits" "$(grep 'event=final status=accepted' "$OUT/trace_fatal.txt" | grep -c ' backend=cuda ' || true)" 1
if [ "$status" -eq 0 ]; then echo "REPLACEMENT_CONTRACT_PASS"; else echo "REPLACEMENT_CONTRACT_FAIL"; fi
exit "$status"
