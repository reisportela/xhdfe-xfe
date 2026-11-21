#!/bin/bash
# The Stata contract of phase 5b of the Stata-path plan (multi-way clustering without
# the repeated passes; tests/stata/cluster_shared_passes_contract.do): every leg in
# its own fresh stata-mp process, in a new scratch directory, with
# XHDFE_PROFILE_CPU=1 and stderr kept. Synthetic data.
#
#   new     the new package, XHDFE_CLUSTER_SHARED_PASSES unset
#   off     the new package, XHDFE_CLUSTER_SHARED_PASSES=0 (today's code)
#   frozen  the frozen package, also with XHDFE_CLUSTER_SHARED_PASSES=0: a previous
#           build that already has phase 5b (17nov2026, point 27: a build from before
#           it is no longer comparable in bits with today's code)
# then new against frozen and against off, line for line (b, the lower triangle of
# V, every scalar, the counts against distinct counts made in Stata, the sums of the
# saved and partialled variables), and the positive controls on stderr: off and
# frozen print counts=recount and never read a carrier (counts=carrier, pair=shared);
# new reads the carrier (counts=carrier at the counts site, pair=shared in the
# intercept and the reconstruction). With "proof" as the fourth argument every leg
# runs the frozen package (the first argument is not read) under the same checks:
# the run proves that the comparisons and the controls hold on the reference build
# against itself (its switch on in new, off in off and frozen), so that a failure
# of a real run comes from the package under test, not from the harness.
# Never set OMP_THREAD_LIMIT for Stata legs: it breaks Stata-MP itself.
# usage: cluster_shared_passes_contract.sh <new package stata dir> <frozen package stata dir> <new scratch dir> [proof]
set -u
NEW=${1:?new package stata directory}
BASE=${2:?frozen package stata directory}
OUT=${3:?new scratch directory}
MODE=${4:-gate}
[ "$MODE" = proof ] && NEW=$BASE
STATA=${STATA:-stata-mp}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT/stata_tmp" && cp "$HERE/cluster_shared_passes_contract.do" "$OUT/" && cd "$OUT" || exit 2
clean=(env -u XHDFE_CLUSTER_SHARED_PASSES -u XHDFE_SLOPE_TEAM_INDEXERS -u XHDFE_SPARSE_RANK
       -u XHDFE_SPARSE_RANK_VERIFY -u XHDFE_PROFILE_CPU -u XHDFE_REDUCED_PCG -u XHDFE_ABSORPTION_CACHE
       -u XHDFE_ABSORPTION_CACHE_MODE -u XHDFE_GPU_BACKEND -u XHDFE_CERTIFY -u OMP_THREAD_LIMIT
       "STATATMP=$OUT/stata_tmp")
status=0
leg() {
  local name=$1 pkg=$2; shift 2
  "${clean[@]}" "XHDFE_STATA_ADOPATH=$pkg" XHDFE_PROFILE_CPU=1 "$@" "$STATA" -q -b do \
    cluster_shared_passes_contract.do fit "$name" 2> "stderr_${name}.txt"
  mv cluster_shared_passes_contract.log "log_${name}.log"
  if grep -q "^CSP_PASS fit" "log_${name}.log"; then
    echo "PASS $name ($(grep -c ' counts ok ' "csp_${name}.txt") count checks)"
  else
    echo "FAIL $name"; tail -n 25 "log_${name}.log"; status=1
  fi
  local left; left=$(ls -A stata_tmp)
  if [ -n "$left" ]; then echo "STOP: files left in $OUT/stata_tmp after $name: $left"; status=1; fi
}
leg new "$NEW"
leg off "$NEW" XHDFE_CLUSTER_SHARED_PASSES=0
leg frozen "$BASE" XHDFE_CLUSTER_SHARED_PASSES=0
for other in frozen off; do
  if cmp -s csp_new.txt "csp_${other}.txt" && [ -s csp_new.txt ]; then
    echo "PASS new_vs_${other} ($(wc -l < csp_new.txt) lines)"
  else
    echo "FAIL new_vs_${other}"; diff csp_new.txt "csp_${other}.txt" | head -n 10; status=1
  fi
done
summary() {
  grep -h "label=cluster_passes" "$1" | sed -E 's/.*site=([a-z]+) ?(counts=[a-z]+)? ?(pair=[a-z]+)?.*/\1 \2 \3/' |
    sort | uniq -c | tr -s ' ' | tr '\n' ';'
}
for name in new off frozen; do echo "controls $name: $(summary "stderr_${name}.txt")"; done
for name in off frozen; do
  if grep "label=cluster_passes" "stderr_${name}.txt" | grep -q "counts=carrier\|pair=shared"; then
    echo "FAIL control: the switch off read a carrier ($name)"; status=1
  fi
  grep -q "label=cluster_passes site=counts counts=recount" "stderr_${name}.txt" \
    || { echo "FAIL control: $name shows no counts site with today's recount"; status=1; }
done
grep -q "label=cluster_passes site=counts counts=carrier" stderr_new.txt \
  || { echo "FAIL control: no counts=carrier at the counts site in new"; status=1; }
grep -q "label=cluster_passes site=intercept counts=carrier pair=shared" stderr_new.txt \
  || { echo "FAIL control: no intercept reading the shared pair in new"; status=1; }
grep -q "label=cluster_passes site=recon counts=carrier pair=shared" stderr_new.txt \
  || { echo "FAIL control: no reconstruction reading the shared pair in new"; status=1; }
[ "$MODE" = proof ] && echo "INFO proof: every leg ran the frozen package $BASE"
if [ "$status" -eq 0 ]; then echo "CSP_CONTRACT_PASS"; else echo "CSP_CONTRACT_FAIL"; fi
exit "$status"
