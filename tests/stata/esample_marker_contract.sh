#!/bin/bash
# The Stata contract of phase 4 of the Stata-path plan
# (tests/stata/esample_marker_contract.do): the old package against the new one,
# every leg in a fresh stata-mp process and its own folder, stderr kept.
# Synthetic data only; every fit at 16 threads.
#
#   fit_old, fit_new   every case of the fit mode
#   fit_old2           the old package again: a difference from fit_old is noise
#   write_old/_new     the write through the marker on large data, with
#                      XHDFE_PROFILE_CPU=1 (parallel selection of the sample)
#   wser_old/_new      the same with XHDFE_PARALLEL_READ=0 (serial selection)
#   d12_old, d12_new   the validations moved before the outputs
# then, value by value, new against old (fit, write, wser, d12) and fit_old2
# against fit_old; the output each fit case displays, line by line; the D12
# error messages; and the positive controls (the marker line of the new ado,
# the parallel and serial selections of the sample).
# Never set OMP_THREAD_LIMIT for Stata legs: it breaks Stata-MP itself.
# usage: esample_marker_contract.sh <old package stata dir> <new package stata dir> <new scratch dir>
set -u
OLD=${1:?old package stata directory}
NEW=${2:?new package stata directory}
OUT=${3:?new scratch directory}
STATA=${STATA:-stata-mp}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" && cp "$HERE/esample_marker_contract.do" "$OUT/" && cd "$OUT" || exit 2
clean=(env -u XHDFE_SPARSE_RANK -u XHDFE_SPARSE_RANK_VERIFY -u XHDFE_PROFILE_CPU
       -u XHDFE_REDUCED_PCG -u XHDFE_REDUCED_PCG_TRACE -u XHDFE_TEST_REDUCED_PCG_DECLINE
       -u XHDFE_ABSORPTION_CACHE -u XHDFE_ABSORPTION_CACHE_MODE -u XHDFE_GPU_BACKEND
       -u XHDFE_SORTPRESERVE -u XHDFE_PARALLEL_READ -u XHDFE_FUSED_READ -u OMP_THREAD_LIMIT)
status=0
fail() { echo "FAIL $*"; status=1; }
# leg <folder> <package> <env assignment>... -- <do-file arguments>...
leg() {
  local dir=$1 pkg=$2; shift 2
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  mkdir -p "$dir"
  ( cd "$dir" && "${clean[@]}" "XHDFE_STATA_ADOPATH=$pkg" "${vars[@]}" "$STATA" -q -b do \
      ../esample_marker_contract.do "$@" 2> stderr.txt )
  if grep -q "^EM_PASS" "$dir/esample_marker_contract.log"; then
    echo "PASS $dir"
  else
    fail "$dir"; tail -n 30 "$dir/esample_marker_contract.log"
  fi
}
leg fit_old "$OLD" -- fit
leg fit_new "$NEW" -- fit
leg fit_old2 "$OLD" -- fit
leg write_old "$OLD" XHDFE_PROFILE_CPU=1 -- write
leg write_new "$NEW" XHDFE_PROFILE_CPU=1 -- write
leg wser_old "$OLD" XHDFE_PROFILE_CPU=1 XHDFE_PARALLEL_READ=0 -- write
leg wser_new "$NEW" XHDFE_PROFILE_CPU=1 XHDFE_PARALLEL_READ=0 -- write
leg d12_old "$OLD" -- d12 altered
leg d12_new "$NEW" -- d12 kept
leg cmp_fit "$NEW" -- compare fit_new fit_old
leg cmp_noise "$OLD" -- compare fit_old2 fit_old
leg cmp_write "$NEW" -- compare write_new write_old
leg cmp_wser "$NEW" -- compare wser_new wser_old
leg cmp_d12 "$NEW" -- d12compare d12_new d12_old

# What each fit case displays, line by line (the program's own commands are not
# echoed inside a program, so a segment holds the case's output only).
segments() { awk '/^EM_CASE_BEGIN /{on=1} on{print} /^EM_CASE_END /{on=0}' "$1"; }
for pair in "fit_new fit_old" "fit_old2 fit_old"; do
  set -- $pair
  if diff <(segments "$1/esample_marker_contract.log") <(segments "$2/esample_marker_contract.log") \
       > "output_$1_vs_$2.diff"; then
    echo "PASS output $1 = $2 ($(segments "$1/esample_marker_contract.log" | grep -c '^EM_CASE_END') cases)"
  else
    fail "output $1 != $2 (output_$1_vs_$2.diff)"; head -n 20 "output_$1_vs_$2.diff"
  fi
done

# D12 messages: the same text and rc in both packages, once the lines of the
# outputs the old package created ("missing values generated") are set aside;
# the two declared changes are written out.
d12seg() {
  awk -v t="$2" '$0 == "EM_D12_BEGIN " t {on=1} on {print} index($0, "EM_D12_END " t " ") == 1 {on=0}' \
    "$1/esample_marker_contract.log" | grep -v "missing values generated)$"
}
for t in c01 c02 c03 c04 c05 c06 c07 c08 c09 c10 c11 c12 c13 c14 c15 c16 col; do
  a=$(d12seg d12_new "$t"); b=$(d12seg d12_old "$t")
  if [ -z "$a" ] || [ "$a" != "$b" ]; then
    fail "d12 $t message"; echo "new: $a"; echo "old: $b"
  fi
done
# Each case reaches its own guard (not, say, a repeated option).
while IFS='|' read -r t msg; do
  grep -qF "$msg" <(d12seg d12_new "$t") || fail "d12 $t: expected the message: $msg"
done <<'EOF'
c01|convergence() must be auto, normchange, reghdfe, or both
c02|numthreads() must be >= 0
c03|defaultthreads() must be >= 0
c04|maxthreads() must be >= 0
c05|minparallelrows() must be positive
c06|targetrowsperthread() must be >= 0
c07|statstyle() must be reghdfe or legacy
c08|tolerancemode() must be xhdfe-fast, reghdfe-comparable, or strict-residual
c09|gpubackend() must be one of cpu, cuda, or metal
c10|mobilityprofile path cannot contain ';'
c11|absorptioncachemode() must be off, auto, read, or write
c12|absorptioncache path cannot contain ';'
c13|fescachemode() must be off, auto, read, or write
c14|festructurecache cannot be combined with fescache()
c15|fescachemode() requires fescache() or festructurecache
c16|festructurecache path cannot contain ';'
EOF
: > d12_declared.txt
for t in dbl kws; do
  { echo "== $t new"; d12seg d12_new "$t"; echo "== $t old"; d12seg d12_old "$t"; } >> d12_declared.txt
done
grep -q "WARNING: Singleton observations not dropped" <(d12seg d12_old kws) \
  || fail "d12 kws: the old package should print the keepsingletons warning first"
grep -q "WARNING: Singleton observations not dropped" <(d12seg d12_new kws) \
  && fail "d12 kws: the new package printed the keepsingletons warning before the option error"
grep -q "rc=110" <(d12seg d12_old dbl) || fail "d12 dbl: the old package should stop with rc 110"
grep -q "rc=198" <(d12seg d12_new dbl) || fail "d12 dbl: the new package should stop with rc 198"
echo "d12 messages compared; declared changes in d12_declared.txt"

# Positive controls: the new ado posts touse itself on the three ordinary fits
# and keeps a separate marker in group() mode; the old ado prints no such line.
n1=$(grep -c "^ado_profile label=esample_marker same_as_touse=1$" write_new/esample_marker_contract.log)
n0=$(grep -c "^ado_profile label=esample_marker same_as_touse=0$" write_new/esample_marker_contract.log)
[ "$n1" -eq 3 ] && [ "$n0" -eq 1 ] || fail "control: same_as_touse=1 x$n1 (want 3), =0 x$n0 (want 1)"
grep -q "label=esample_marker" write_old/esample_marker_contract.log \
  && fail "control: the old package printed an esample_marker line"
# The sample selected in parallel (16 workers, no fallback) on the three large
# fits by default, and never with XHDFE_PARALLEL_READ=0.
for l in write_old write_new; do
  np=$(grep -c "label=read_parallel_mode allowed=1 .*select_team=16 select_fallback=0 " "$l/stderr.txt")
  [ "$np" -ge 3 ] || fail "control $l: parallel selection on $np large fits (want 3)"
done
for l in wser_old wser_new; do
  grep -q "label=read_parallel_mode allowed=1" "$l/stderr.txt" \
    && fail "control $l: parallel read allowed with XHDFE_PARALLEL_READ=0"
done
grep -h "^EM_WRITE " write_new/esample_marker_contract.log wser_new/esample_marker_contract.log
if [ "$status" -eq 0 ]; then echo "EM_CONTRACT_PASS"; else echo "EM_CONTRACT_FAIL"; fi
exit "$status"
