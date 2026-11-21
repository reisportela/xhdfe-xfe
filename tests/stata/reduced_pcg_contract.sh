#!/bin/bash
# The Stata contract of the reduced-system tier (tests/stata/reduced_pcg_contract.do):
# every mode in its own fresh stata-mp process, in a new scratch directory, with
# stderr kept (the tier's trace). Synthetic data only, CPU; with "cuda" the battery
# adds one fit with gpubackend(cuda).
# usage: reduced_pcg_contract.sh <package stata dir> <new scratch dir> [cuda]
set -u
PKG=${1:?package stata directory}
OUT=${2:?new scratch directory}
CUDA=${3:-}
STATA=${STATA:-stata-mp}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" && cp "$HERE/reduced_pcg_contract.do" "$OUT/" && cd "$OUT" || exit 2
clean=(env -u XHDFE_REDUCED_PCG -u XHDFE_REDUCED_PCG_TRACE -u XHDFE_TEST_REDUCED_PCG_DECLINE
       -u XHDFE_ABSORPTION_CACHE -u XHDFE_ABSORPTION_CACHE_MODE -u XHDFE_STATA_LEGACY_RESIDUAL_REFINE
       -u XHDFE_GPU_BACKEND -u XHDFE_RPCG_TEST_CUDA -u XHDFE_PROFILE_CPU -u XHDFE_REDUCED_PCG_COMPANION
       "XHDFE_STATA_ADOPATH=$PKG")
gpu=()
[ "$CUDA" = cuda ] && gpu=(XHDFE_RPCG_TEST_CUDA=1)
status=0
# leg <name> <env assignment>... -- <do-file arguments>...
leg() {
  local name=$1; shift
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  "${clean[@]}" "${vars[@]}" "$STATA" -q -b do reduced_pcg_contract.do "$@" 2> "trace_${name}.txt"
  mv reduced_pcg_contract.log "log_${name}.log"
  if grep -q "^RPCG_PASS" "log_${name}.log"; then
    echo "PASS $name"
  else
    echo "FAIL $name"; tail -n 25 "log_${name}.log"; status=1
  fi
}
leg battery_on XHDFE_REDUCED_PCG_TRACE=1 "${gpu[@]}" -- battery on
leg battery_off XHDFE_REDUCED_PCG=0 "${gpu[@]}" -- battery off
leg battery_declined XHDFE_REDUCED_PCG_TRACE=1 XHDFE_TEST_REDUCED_PCG_DECLINE=iterate "${gpu[@]}" -- battery declined
leg compare -- compare
leg firststage_on XHDFE_REDUCED_PCG_TRACE=1 -- firststage on
leg firststage_off XHDFE_REDUCED_PCG=0 -- firststage off
leg firstcompare -- firstcompare
leg cache_write XHDFE_REDUCED_PCG_TRACE=1 "XHDFE_ABSORPTION_CACHE=$PWD/abs.cache" XHDFE_ABSORPTION_CACHE_MODE=write -- cache write
leg cache_read XHDFE_REDUCED_PCG_TRACE=1 "XHDFE_ABSORPTION_CACHE=$PWD/abs.cache" XHDFE_ABSORPTION_CACHE_MODE=read -- cache read
leg cache_write_legacy XHDFE_REDUCED_PCG_TRACE=1 "XHDFE_ABSORPTION_CACHE=$PWD/abs_legacy.cache" \
    XHDFE_ABSORPTION_CACHE_MODE=write XHDFE_STATA_LEGACY_RESIDUAL_REFINE=1 -- cache write legacy
leg cache_read_legacy XHDFE_REDUCED_PCG_TRACE=1 "XHDFE_ABSORPTION_CACHE=$PWD/abs_legacy.cache" \
    XHDFE_ABSORPTION_CACHE_MODE=read XHDFE_STATA_LEGACY_RESIDUAL_REFINE=1 -- cache read legacy
leg cachecompare -- cachecompare
if [ "$status" -eq 0 ]; then echo "RPCG_CONTRACT_PASS"; else echo "RPCG_CONTRACT_FAIL"; fi
exit "$status"
