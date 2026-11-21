#!/bin/bash
# The Stata contract of phase 5a of the Stata-path plan (the identifier builds of
# the route with heterogeneous slopes on the team;
# tests/stata/slope_team_indexers_contract.do): every leg in its own fresh stata-mp
# process, in a new scratch directory, with XHDFE_PROFILE_CPU=1 and stderr kept.
# Synthetic data.
#
#   new   the new package, XHDFE_SLOPE_TEAM_INDEXERS unset
#   off   the new package, XHDFE_SLOPE_TEAM_INDEXERS=0 (today's builders)
#   base  the frozen package (the phase chain, packages/cand_sp2/stata)
# then new against base and against off, line for line (b, the lower triangle of
# V, every scalar, the return code of xfepout with slopes, which the ado refuses),
# and the positive controls: in new, the lines "cpu_profile label=mixed_indexers
# ... team=8" and "label=slope_rank_indexer team=8" (fits with numthreads(8)) and
# team=1 lines (the automatic thread count, below the gate); in off, team=1 only.
# Never set OMP_THREAD_LIMIT for Stata legs: it breaks Stata-MP itself.
# usage: slope_team_indexers_contract.sh <new package stata dir> <frozen package stata dir> <new scratch dir>
set -u
NEW=${1:?new package stata directory}
BASE=${2:?frozen package stata directory}
OUT=${3:?new scratch directory}
STATA=${STATA:-stata-mp}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" && cp "$HERE/slope_team_indexers_contract.do" "$OUT/" && cd "$OUT" || exit 2
clean=(env -u XHDFE_SLOPE_TEAM_INDEXERS -u XHDFE_SPARSE_RANK -u XHDFE_SPARSE_RANK_VERIFY
       -u XHDFE_PROFILE_CPU -u XHDFE_REDUCED_PCG -u XHDFE_ABSORPTION_CACHE
       -u XHDFE_ABSORPTION_CACHE_MODE -u XHDFE_GPU_BACKEND -u OMP_THREAD_LIMIT)
status=0
# leg <name> <package> <env assignment>...
leg() {
  local name=$1 pkg=$2; shift 2
  "${clean[@]}" "XHDFE_STATA_ADOPATH=$pkg" XHDFE_PROFILE_CPU=1 "$@" "$STATA" -q -b do \
    slope_team_indexers_contract.do fit "$name" 2> "stderr_${name}.txt"
  mv slope_team_indexers_contract.log "log_${name}.log"
  if grep -q "^STI_PASS fit" "log_${name}.log"; then
    echo "PASS $name"
  else
    echo "FAIL $name"; tail -n 25 "log_${name}.log"; status=1
  fi
}
leg new "$NEW"
leg off "$NEW" XHDFE_SLOPE_TEAM_INDEXERS=0
leg base "$BASE"
for other in base off; do
  if cmp -s st_new.txt "st_${other}.txt" && [ -s st_new.txt ]; then
    echo "PASS new_vs_${other} ($(wc -l < st_new.txt) lines)"
  else
    echo "FAIL new_vs_${other}"; diff st_new.txt "st_${other}.txt" | head -n 10; status=1
  fi
done
teams() {
  grep -h "label=mixed_indexers \|label=slope_rank_indexer " "$1" |
    sed -E 's/.*label=([a-z_]+) .*team=([0-9]+) .*/\1:team=\2/' | sort | uniq -c | tr -s ' ' | tr '\n' ' '
}
team_new=$(teams stderr_new.txt)
team_off=$(teams stderr_off.txt)
echo "teams new: ${team_new:-none}; off: ${team_off:-none}"
grep -q "label=mixed_indexers .*team=8 " stderr_new.txt || { echo "FAIL control: no mixed_indexers team=8 line in new"; status=1; }
grep -q "label=slope_rank_indexer team=8 " stderr_new.txt || { echo "FAIL control: no slope_rank_indexer team=8 line in new"; status=1; }
grep -q "label=mixed_indexers .*team=1 " stderr_new.txt || { echo "FAIL control: no team=1 line (automatic count) in new"; status=1; }
# The sparse identifier takes the phase-1 index in both builders (not the serial
# fallback for its budget), and the wide one the phase-3 read.
grep -q "label=sparse_rank site=indexer path=bitmap " stderr_new.txt || { echo "FAIL control: no indexer path=bitmap line in new"; status=1; }
grep -q "label=sparse_rank site=indexer_lite path=bitmap " stderr_new.txt || { echo "FAIL control: no indexer_lite path=bitmap line in new"; status=1; }
grep -q "label=wide_id_read what=fes .*path=team " stderr_new.txt || { echo "FAIL control: no wide_id_read path=team line in new"; status=1; }
echo "index lines new: $(grep -o 'label=sparse_rank site=indexer[a-z_]* path=[a-z]*' stderr_new.txt | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
if grep -q "label=\(mixed_indexers\|slope_rank_indexer\) .*team=\([2-9]\|[1-9][0-9]\)" stderr_off.txt; then
  echo "FAIL control: the switch off ran a team"; status=1
fi
if [ "$status" -eq 0 ]; then echo "STI_CONTRACT_PASS"; else echo "STI_CONTRACT_FAIL"; fi
exit "$status"
