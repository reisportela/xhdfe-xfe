#!/bin/bash
# The Stata contract of phase 1 of the Stata-path plan (sparse identifiers ranked
# on the team; tests/stata/sparse_rank_contract.do): every leg in its own fresh
# stata-mp process, in a new scratch directory, with stderr kept. Synthetic data.
#
#   new        the new package, XHDFE_SPARSE_RANK unset
#   off        the new package, XHDFE_SPARSE_RANK=0
#   base       the frozen package of dea5fa00 (packages/cand_c232b/stata)
#   control    the new package with XHDFE_PROFILE_CPU=1 (path=bitmap lines)
#   offcontrol the new package, switch off, XHDFE_PROFILE_CPU=1 (path=serial reason=off)
#   verify     the new package with XHDFE_SPARSE_RANK_VERIFY=1 and XHDFE_PROFILE_CPU=1:
#              every site that took the index recomputes today's serial result
#              (no ok=0 line; an ok=1 line for every site)
# then new against base, against off and against verify, value by value.
# With "cuda" the fit legs add one gpubackend(cuda) fit (compared within 1e-7).
# Never set OMP_THREAD_LIMIT for Stata legs: it breaks Stata-MP itself.
# usage: sparse_rank_contract.sh <new package stata dir> <frozen package stata dir> <new scratch dir> [cuda]
set -u
NEW=${1:?new package stata directory}
BASE=${2:?frozen package stata directory}
OUT=${3:?new scratch directory}
CUDA=${4:-}
STATA=${STATA:-stata-mp}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" && cp "$HERE/sparse_rank_contract.do" "$OUT/" && cd "$OUT" || exit 2
# The frozen package predates D10, so every leg runs with D10 off
# (XHDFE_REDUCED_PCG_CHAT_STOP=0); D10 has its own contracts.
clean=(env -u XHDFE_SPARSE_RANK -u XHDFE_PROFILE_CPU -u XHDFE_REDUCED_PCG -u XHDFE_REDUCED_PCG_TRACE
       -u XHDFE_TEST_REDUCED_PCG_DECLINE -u XHDFE_ABSORPTION_CACHE -u XHDFE_ABSORPTION_CACHE_MODE
       -u XHDFE_GPU_BACKEND -u XHDFE_SR_TEST_CUDA -u OMP_THREAD_LIMIT XHDFE_REDUCED_PCG_CHAT_STOP=0)
gpu=()
[ "$CUDA" = cuda ] && gpu=(XHDFE_SR_TEST_CUDA=1)
status=0
# leg <name> <package> <env assignment>... -- <do-file arguments>...
leg() {
  local name=$1 pkg=$2; shift 2
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  "${clean[@]}" "XHDFE_STATA_ADOPATH=$pkg" "${vars[@]}" "$STATA" -q -b do sparse_rank_contract.do "$@" \
    2> "stderr_${name}.txt"
  mv sparse_rank_contract.log "log_${name}.log"
  if grep -q "^SR_PASS" "log_${name}.log"; then
    echo "PASS $name"
  else
    echo "FAIL $name"; tail -n 25 "log_${name}.log"; status=1
  fi
}
leg new "$NEW" "${gpu[@]}" -- fit new
leg off "$NEW" XHDFE_SPARSE_RANK=0 "${gpu[@]}" -- fit off
leg base "$BASE" "${gpu[@]}" -- fit base
leg control "$NEW" XHDFE_PROFILE_CPU=1 -- control control
leg offcontrol "$NEW" XHDFE_PROFILE_CPU=1 XHDFE_SPARSE_RANK=0 -- control offcontrol
# The in-situ verifier (CPU only: profiling changes a CUDA routing condition).
leg verify "$NEW" XHDFE_SPARSE_RANK_VERIFY=1 XHDFE_PROFILE_CPU=1 -- fit verify
leg new_vs_base "$NEW" -- compare new base
leg new_vs_off "$NEW" -- compare new off
leg new_vs_verify "$NEW" -- compare new verify
if grep -q "label=sparse_rank_verify.* ok=0" stderr_verify.txt; then
  echo "FAIL verify: the verifier found a difference"; grep "label=sparse_rank_verify.* ok=0" stderr_verify.txt; status=1
fi
for site in tier_codes indexer singleton_lite lookup components; do
  if ! grep -q "label=sparse_rank_verify site=$site ok=1" stderr_verify.txt; then
    echo "FAIL verify: site $site was never verified"; status=1
  fi
done
# Positive controls: the new code ran in the control leg, and only the serial
# branches with the switch off.
# (The nested test keeps today's hash: bench decision of 14 Nov.)
for site in tier_codes indexer singleton_lite lookup; do
  if ! grep -q "label=sparse_rank site=$site path=bitmap" stderr_control.txt; then
    echo "FAIL control: no site=$site path=bitmap line"; status=1
  fi
done
if ! grep -q "label=sparse_rank site=components path=marking base=rows" stderr_control.txt; then
  echo "FAIL control: no component marking with a Rows lookup (cbig)"; status=1
fi
if grep "label=sparse_rank" stderr_offcontrol.txt | grep -v "site=components" | grep -q "path=bitmap"; then
  echo "FAIL offcontrol: the index ran with XHDFE_SPARSE_RANK=0"; status=1
fi
if ! grep -q "path=serial reason=off" stderr_offcontrol.txt; then
  echo "FAIL offcontrol: no path=serial reason=off line"; status=1
fi
if [ "$status" -eq 0 ]; then echo "SR_CONTRACT_PASS"; else echo "SR_CONTRACT_FAIL"; fi
exit "$status"
