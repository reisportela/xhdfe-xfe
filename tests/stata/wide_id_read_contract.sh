#!/bin/bash
# The Stata contract of phase 3 of the Stata-path plan
# (tests/stata/wide_id_read_contract.do): the old package against the new one,
# every leg in a fresh stata-mp process and its own folder, with
# XHDFE_PROFILE_CPU=1 and stderr kept. Synthetic data only.
#
#   old        the package of the previous phase
#   new        the new package
#   off        the new package with XHDFE_WIDE_ID_READ=0 (today's column reader)
#   anomaly    the new package with XHDFE_TEST_WIDE_ID_READ_ANOMALY=1: the team
#              read marks an anomaly after writing block 1, and is discarded
#   fusedoff   the new package with XHDFE_FUSED_READ=0
#   paroff     the new package with XHDFE_PARALLEL_READ=0
#   nopretouch the new package with XHDFE_PARALLEL_READ_PRETOUCH=0 (the workers
#              touch the pages of the pairs and codes first)
# then every leg against old, value by value; the input fingerprint of every
# fit identical in every leg (it covers the codes of every identifier); the
# output of every case (the refusals' messages) line by line; and the positive
# controls: which reader took each wide column in each leg.
# Never set OMP_THREAD_LIMIT for Stata legs: it breaks Stata-MP itself.
# usage: wide_id_read_contract.sh <old package stata dir> <new package stata dir> <new scratch dir>
set -u
OLD=${1:?old package stata directory}
NEW=${2:?new package stata directory}
OUT=${3:?new scratch directory}
STATA=${STATA:-stata-mp}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" && cp "$HERE/wide_id_read_contract.do" "$OUT/" && cd "$OUT" || exit 2
clean=(env -u XHDFE_SPARSE_RANK -u XHDFE_SPARSE_RANK_VERIFY -u XHDFE_PROFILE_CPU
       -u XHDFE_REDUCED_PCG -u XHDFE_REDUCED_PCG_TRACE -u XHDFE_TEST_REDUCED_PCG_DECLINE
       -u XHDFE_ABSORPTION_CACHE -u XHDFE_ABSORPTION_CACHE_MODE -u XHDFE_GPU_BACKEND
       -u XHDFE_SORTPRESERVE -u XHDFE_PARALLEL_READ -u XHDFE_PARALLEL_READ_MIN_VALUES
       -u XHDFE_FUSED_READ -u XHDFE_WIDE_ID_READ -u XHDFE_TEST_WIDE_ID_READ_ANOMALY
       -u XHDFE_PARALLEL_READ_PRETOUCH -u OMP_THREAD_LIMIT)
status=0
fail() { echo "FAIL $*"; status=1; }
# leg <folder> <package> <env assignment>... -- <do-file arguments>...
leg() {
  local dir=$1 pkg=$2; shift 2
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  mkdir -p "$dir"
  ( cd "$dir" && "${clean[@]}" "XHDFE_STATA_ADOPATH=$pkg" XHDFE_PROFILE_CPU=1 "${vars[@]}" \
      "$STATA" -q -b do ../wide_id_read_contract.do "$@" 2> stderr.txt )
  if grep -q "^WR_PASS" "$dir/wide_id_read_contract.log"; then
    echo "PASS $dir"
  else
    fail "$dir"; tail -n 30 "$dir/wide_id_read_contract.log"
  fi
}
legs="new off anomaly fusedoff paroff nopretouch"
leg old "$OLD" -- fit
leg new "$NEW" -- fit
leg off "$NEW" XHDFE_WIDE_ID_READ=0 -- fit
leg anomaly "$NEW" XHDFE_TEST_WIDE_ID_READ_ANOMALY=1 -- fit
leg fusedoff "$NEW" XHDFE_FUSED_READ=0 -- fit
leg paroff "$NEW" XHDFE_PARALLEL_READ=0 -- fit
leg nopretouch "$NEW" XHDFE_PARALLEL_READ_PRETOUCH=0 -- fit
for l in $legs; do
  leg "cmp_$l" "$NEW" -- compare "$l" old
done

# Every input as the estimator received it, codes included, fit by fit.
grep -o 'label=input_digest value=[0-9]*' old/stderr.txt > old/digests.txt
nd=$(wc -l < old/digests.txt)
[ "$nd" -ge 14 ] || fail "input digests: only $nd in the old leg"
for l in $legs; do
  grep -o 'label=input_digest value=[0-9]*' "$l/stderr.txt" > "$l/digests.txt"
  cmp -s old/digests.txt "$l/digests.txt" || fail "input digests of $l differ from old"
done
echo "input digests: $nd fits, identical in every leg"

# What each case displays (the refusals' messages), line by line, without the
# ado's profile lines.
segments() {
  awk '/^WR_CASE_BEGIN /{on=1} on{print} /^WR_CASE_END /{on=0}' "$1" | grep -v '^ado_profile '
}
for l in $legs; do
  diff <(segments old/wide_id_read_contract.log) <(segments "$l/wide_id_read_contract.log") \
    > "output_${l}_vs_old.diff" || fail "output of $l differs from old (output_${l}_vs_old.diff)"
done
echo "displayed output compared: $(segments old/wide_id_read_contract.log | grep -c '^WR_CASE_END') cases"

# Positive controls. The wide columns of the large fits at 16 and 24 threads and
# in automatic mode, read on the team in the new leg: fe_wide, cl_wide,
# fe_cl_wide (two columns), dup, float, negative, fraction, late, ifin, outputs
# (two columns), tauto, t24 and the individual() column of ind give 15. A
# serial first pass leaves the column to its reader: t1, small (two columns)
# and the four refusals (the parallel pass meets the invalid identifier, the
# serial one stops reading the column at its first wide value) give 7.
count() { grep -c "label=wide_id_read .*$2" "$1/stderr.txt"; }
nt=$(count new 'path=team reason=done ')
[ "$nt" -eq 15 ] || fail "control new: $nt columns read on the team (want 15)"
# The ind fit is the only one with more than 600 000 rows.
nind=$(grep "label=wide_id_read what=fes .*path=team reason=done " new/stderr.txt |
       awk -F'rows=' '{split($2, a, " "); if (a[1] + 0 > 600000) n++} END {print n + 0}')
[ "$nind" -eq 1 ] || fail "control new: the individual() column of ind was not read on the team"
ns=$(count new 'path=column_reader reason=serial_first_pass ')
[ "$ns" -eq 7 ] || fail "control new: $ns columns left after a serial first pass (want 7)"
[ "$(count new 'path=column_reader reason=\(anomaly\|exception\|empty\|serial_team\) ')" -eq 0 ] \
  || fail "control new: a team read failed"
[ "$(count off 'path=team ')" -eq 0 ] || fail "control off: a column was read on the team"
[ "$(count off 'reason=switch_off ')" -eq 15 ] || fail "control off: switch_off on $(count off 'reason=switch_off ') columns (want 15)"
[ "$(count nopretouch 'path=team reason=done ')" -eq 15 ] || fail "control nopretouch: team reads"
# The four refusals went through the parallel pass, which met the invalid
# identifier and was discarded (then the serial pass, then the column reader).
for l in old new; do
  nd=$(grep -c "label=read_parallel_mode .* read=0 .*fallback=1 " "$l/stderr.txt")
  [ "$nd" -eq 4 ] || fail "control $l: $nd discarded parallel passes (want 4: the refusals)"
done
[ "$(count anomaly 'path=team ')" -eq 0 ] || fail "control anomaly: a column was read on the team"
[ "$(count anomaly 'reason=test_anomaly ')" -eq 15 ] \
  || fail "control anomaly: test_anomaly on $(count anomaly 'reason=test_anomaly ') columns (want 15)"
for l in fusedoff paroff; do
  [ "$(count "$l" 'path=team ')" -eq 0 ] || fail "control $l: a column was read on the team"
done
grep -q "label=wide_id_read" old/stderr.txt && fail "control old: the old plugin printed wide_id_read"
echo "controls: new team=$nt serial_first_pass=$ns; off switch_off=$(count off 'reason=switch_off '); anomaly test_anomaly=$(count anomaly 'reason=test_anomaly ')"
if [ "$status" -eq 0 ]; then echo "WR_CONTRACT_PASS"; else echo "WR_CONTRACT_FAIL"; fi
exit "$status"
