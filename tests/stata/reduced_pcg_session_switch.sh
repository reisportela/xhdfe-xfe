#!/bin/bash
# The session switch of the reduced-system tier through Stata
# (tests/stata/reduced_pcg_session_switch.do): every mode in its own fresh stata-mp
# process, in a new scratch directory, with stderr kept (the tier's trace).
# Synthetic data only, CPU.
# usage: reduced_pcg_session_switch.sh <package stata dir> <new scratch dir>
set -u
PKG=${1:?package stata directory}
OUT=${2:?new scratch directory}
STATA=${STATA:-stata-mp}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" && cp "$HERE/reduced_pcg_session_switch.do" "$OUT/" && cd "$OUT" || exit 2
clean=(env -u XHDFE_REDUCED_PCG -u XHDFE_REDUCED_PCG_TRACE -u XHDFE_TEST_REDUCED_PCG_DECLINE
       -u XHDFE_ABSORPTION_CACHE -u XHDFE_ABSORPTION_CACHE_MODE -u XHDFE_GPU_BACKEND
       -u XHDFE_PROFILE_CPU -u XHDFE_REDUCED_PCG_COMPANION "XHDFE_STATA_ADOPATH=$PKG")
status=0
# leg <name> <env assignment>... -- <do-file arguments>...
leg() {
  local name=$1; shift
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  "${clean[@]}" "${vars[@]}" "$STATA" -q -b do reduced_pcg_session_switch.do "$@" 2> "trace_${name}.txt"
  mv reduced_pcg_session_switch.log "log_${name}.log"
  if grep -q "^SWITCH_PASS" "log_${name}.log"; then
    echo "PASS $name"
  else
    echo "FAIL $name"; tail -n 25 "log_${name}.log"; status=1
  fi
}
finals() { grep -c "reduced_pcg event=final" "$1" || true; }
leg session XHDFE_REDUCED_PCG_TRACE=1 -- session
leg envoff XHDFE_REDUCED_PCG=0 XHDFE_REDUCED_PCG_TRACE=1 -- envoff
leg first0 XHDFE_REDUCED_PCG_TRACE=1 -- first0
leg firstdefault XHDFE_REDUCED_PCG_TRACE=1 -- firstdefault
leg compare -- compare
# The session: one final line per fit with the tier on, none for the 3 off fits and none
# for the two refused calls: 7 final lines. The 6 fits without saved effects are accepted;
# the savefe fit's line is the one its report gives (SWITCH_SAVEFE_REPORT in the log):
# accepted when the tier publishes its projection (reason none), ineligible by savefe when
# the capturing sweep is published as it is (reason savefe).
r=$(sed -n 's/^SWITCH_SAVEFE_REPORT status=\([0-9]*\) reason=\([A-Za-z_]*\)[[:space:]]*$/\1 \2/p' log_session.log)
case "${r#* }" in
  none) want_n=7; want_s=0 ;;
  savefe) want_n=6; want_s=1 ;;
  *) want_n=-1; want_s=-1; echo "FAIL session report of the savefe fit: '$r'"; status=1 ;;
esac
n=$(grep -c "event=final status=accepted" trace_session.txt || true)
s=$(grep -c "event=final status=ineligible reason=savefe" trace_session.txt || true)
m=$(finals trace_session.txt)
if [ "$n" -ne "$want_n" ] || [ "$s" -ne "$want_s" ] || [ "$m" -ne 7 ]; then
  echo "FAIL session trace: $n accepted, $s savefe of $m final lines (the savefe fit reports status/reason '$r')"; status=1
fi
# envoff: the off fit prints nothing, the two on fits one accepted line each.
n=$(grep -c "event=final status=accepted" trace_envoff.txt || true)
m=$(finals trace_envoff.txt)
if [ "$n" -ne 2 ] || [ "$m" -ne 2 ]; then echo "FAIL envoff trace: $n accepted of $m final lines"; status=1; fi
# first0: with the global at 0, the first fit and its auxiliary fits print nothing;
# the second IV fit (reducedpcg(on), ineligible by iv) prints its own line only, and
# its auxiliary fits and estat firststage's follow the global: nothing.
n=$(grep -c "event=final status=ineligible reason=iv" trace_first0.txt || true)
m=$(finals trace_first0.txt)
if [ "$n" -ne 1 ] || [ "$m" -ne 1 ]; then echo "FAIL first0 trace: $n iv lines of $m final lines"; status=1; fi
# firstdefault: the auxiliary fits enter the tier (positive control).
n=$(grep -c "event=final status=accepted" trace_firstdefault.txt || true)
if [ "$n" -lt 2 ]; then echo "FAIL firstdefault trace: $n accepted auxiliary lines"; status=1; fi
if [ "$status" -eq 0 ]; then echo "SWITCH_CONTRACT_PASS"; else echo "SWITCH_CONTRACT_FAIL"; fi
exit "$status"
