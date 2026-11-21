#!/bin/bash
# The Stata contract of phase 2 of the Stata-path plan
# (tests/stata/shared_fe_coding_contract.do): the old package against the new
# one, every leg in a fresh stata-mp process and its own folder, with
# XHDFE_PROFILE_CPU=1 and stderr kept. Synthetic data only.
#
#   old, new     the package of the previous phase and the new one
#   off          the new package with XHDFE_SHARED_FE_CODING=0 (the certificate
#                numbers the identifiers again)
#   verify       the new package with XHDFE_SPARSE_RANK_VERIFY=1 (every indexer
#                taken over is compared with build_indexer_team())
#   old_<stage>, new_<stage>
#                the tier forced to decline at each stage
#                (XHDFE_TEST_REDUCED_PCG_DECLINE), on the tier_t16 case
# then every leg against old (and new_<stage> against old_<stage>), value by
# value, threads_used and parallel_workers_active included; and the positive
# controls: one "site=cert_shared" line per effect of every tier certificate,
# path=codes or path=build reason=holes by range and levels, none in old, off
# and the fits without the tier, an ok=1 verify line per taken indexer.
# Never set OMP_THREAD_LIMIT for Stata legs: it breaks Stata-MP itself.
# usage: shared_fe_coding_contract.sh <old package stata dir> <new package stata dir> <new scratch dir>
set -u
OLD=${1:?old package stata directory}
NEW=${2:?new package stata directory}
OUT=${3:?new scratch directory}
STATA=${STATA:-stata-mp}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" && cp "$HERE/shared_fe_coding_contract.do" "$OUT/" && cd "$OUT" || exit 2
# The old package predates D10, so every leg runs with D10 off
# (XHDFE_REDUCED_PCG_CHAT_STOP=0); D10 has its own contracts.
clean=(env -u XHDFE_SPARSE_RANK -u XHDFE_SPARSE_RANK_VERIFY -u XHDFE_PROFILE_CPU
       -u XHDFE_REDUCED_PCG -u XHDFE_REDUCED_PCG_TRACE -u XHDFE_TEST_REDUCED_PCG_DECLINE
       -u XHDFE_ABSORPTION_CACHE -u XHDFE_ABSORPTION_CACHE_MODE -u XHDFE_GPU_BACKEND
       -u XHDFE_SHARED_FE_CODING -u OMP_THREAD_LIMIT XHDFE_REDUCED_PCG_CHAT_STOP=0)
status=0
fail() { echo "FAIL $*"; status=1; }
# leg <folder> <package> <env assignment>... -- <do-file arguments>...
leg() {
  local dir=$1 pkg=$2; shift 2
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  mkdir -p "$dir"
  ( cd "$dir" && "${clean[@]}" "XHDFE_STATA_ADOPATH=$pkg" XHDFE_PROFILE_CPU=1 XHDFE_REDUCED_PCG_TRACE=1 \
      "${vars[@]}" "$STATA" -q -b do ../shared_fe_coding_contract.do "$@" 2> stderr.txt )
  if grep -q "^SF_PASS" "$dir/shared_fe_coding_contract.log"; then
    echo "PASS $dir"
  else
    fail "$dir"; tail -n 30 "$dir/shared_fe_coding_contract.log"
  fi
}
stages="geometry iterate residuals certificate bad_alloc"
leg old "$OLD" -- fit
leg new "$NEW" -- fit
leg off "$NEW" XHDFE_SHARED_FE_CODING=0 -- fit
leg verify "$NEW" XHDFE_SPARSE_RANK_VERIFY=1 -- fit
for s in $stages; do
  leg "old_$s" "$OLD" XHDFE_TEST_REDUCED_PCG_DECLINE=$s -- decline
  leg "new_$s" "$NEW" XHDFE_TEST_REDUCED_PCG_DECLINE=$s -- decline
done
for l in new off verify; do
  leg "cmp_$l" "$NEW" -- compare "$l" old
done
for s in $stages; do
  leg "cmp_$s" "$NEW" -- compare "new_$s" "old_$s"
done

# The tier's decision in every leg: accepted in the tier fits of fit mode,
# declined at the forced stage in the decline legs.
for l in old new off verify; do
  na=$(grep -c "reduced_pcg event=final status=accepted" "$l/stderr.txt")
  [ "$na" -eq 8 ] || fail "tier: $na accepted fits in $l (want 8: six teams, sorted, boundary)"
done
for s in $stages; do
  for l in old new; do
    grep -q "reduced_pcg event=final status=declined" "${l}_$s/stderr.txt" || fail "tier: ${l}_$s did not decline"
  done
done

# Positive controls: one line per effect of each tier certificate.
lines() { grep -o "site=cert_shared path=[a-z]* reason=[a-z]* n=[0-9]* range=[0-9]* levels=[0-9]*" "$1/stderr.txt" |
          sed 's/ n=[0-9]*//' | sort | uniq -c | sed 's/^ *//'; }
expect_new=$(printf '%s\n' \
  "1 site=cert_shared path=build reason=holes range=101 levels=91" \
  "7 site=cert_shared path=build reason=holes range=1000 levels=900" \
  "1 site=cert_shared path=codes reason=rank range=101 levels=90" \
  "7 site=cert_shared path=codes reason=rank range=1000 levels=890" \
  "7 site=cert_shared path=codes reason=rank range=3479914 levels=40000" \
  "1 site=cert_shared path=codes reason=rank range=400000 levels=40000" \
  "1 site=cert_shared path=codes reason=rank range=400001 levels=40000" \
  "7 site=cert_shared path=codes reason=rank range=500 levels=500" | sort)
for l in new verify; do
  got=$(lines "$l" | sort)
  [ "$got" = "$expect_new" ] || { fail "control $l: cert_shared lines differ"; diff <(echo "$expect_new") <(echo "$got"); }
done
for l in old off; do
  [ -z "$(lines "$l")" ] || fail "control $l: cert_shared lines in $l"
done
nv=$(grep -c "label=sparse_rank_verify site=cert_shared ok=1" verify/stderr.txt)
[ "$nv" -eq 24 ] || fail "control verify: $nv cert_shared ok=1 lines (want 24, one per indexer taken over)"
grep -q "label=sparse_rank_verify.* ok=0" verify/stderr.txt && fail "verify: the verifier found a difference"
for s in $stages; do
  n=$(lines "new_$s" | awk '{n += $1} END {print n + 0}')
  want=0; [ "$s" = certificate ] && want=4
  [ "$n" -eq "$want" ] || fail "control new_$s: $n cert_shared lines (want $want)"
  [ -z "$(lines "old_$s")" ] || fail "control old_$s: cert_shared lines"
done
echo "controls: new $(lines new | awk '{n += $1} END {print n + 0}') lines, verify ok=1 $nv"
if [ "$status" -eq 0 ]; then echo "SF_CONTRACT_PASS"; else echo "SF_CONTRACT_FAIL"; fi
exit "$status"
