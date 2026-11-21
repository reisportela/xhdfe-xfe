#!/bin/bash
# Point 13 (17nov2026): the Stata leg of tests/reduced_pcg_trace_final_contract.py. Every mode of
# tests/stata/reduced_pcg_trace_final_contract.do in its own fresh stata-mp process, for the new and
# the frozen package, in a new scratch directory, with XHDFE_REDUCED_PCG_TRACE=1 and
# XHDFE_AUTO_ROUTING_RETRY_DIAG=1 and stderr kept (the trace), each leg with its own STATATMP,
# checked empty afterwards. Synthetic data, CPU, numthreads(4). Then, per mode:
#   - one "xhdfe reduced_pcg event=final" line per fit, in order, agreeing with the fit's
#     e(reduced_pcg_status) and e(reduced_pcg_reason) (new package);
#   - every coefficient and every element of e(V) bit for bit the frozen package's, and so the
#     tier's status, reason and e(iterations); e(absorption_method_used) the frozen package's
#     (or 9 under the forced Krylov route, with point 14);
#   - untouched modes (no projection attempted, no saved effects, weights): the tier's trace lines
#     byte for byte the frozen package's, seconds= masked;
#   - published modes (E1 publishes the projection): the E1 diagnostic says so, and the tier's lines
#     are those of the same fit without saved effects (lowmob: the tier publishes the projection;
#     declined, with XHDFE_TEST_REDUCED_PCG_DECLINE=iterate: the tier declines it and another route
#     publishes it), or the line is the projection's "status=ineligible reason=forced_solver"
#     (krylov).
# usage: reduced_pcg_trace_final_contract.sh <new package stata dir> <frozen package stata dir> <new scratch dir>
# Never set OMP_THREAD_LIMIT for Stata legs.
set -u
NEW=${1:?new package stata directory}
BASE=${2:?frozen package stata directory}
OUT=${3:?new scratch directory}
STATA=${STATA:-stata-mp}
PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" && cp "$HERE/reduced_pcg_trace_final_contract.do" "$OUT/" && cd "$OUT" || exit 2
clean=(env -u XHDFE_REDUCED_PCG -u XHDFE_REDUCED_PCG_TRACE -u XHDFE_SAVEFE_ACCURATE_PROJECTION -u XHDFE_USE_KRYLOV
       -u XHDFE_ABSORPTION_CACHE -u XHDFE_ABSORPTION_CACHE_MODE -u XHDFE_GPU_BACKEND -u XHDFE_PROFILE_CPU
       -u XHDFE_MOBILITY_MODE -u XHDFE_MOBILITY_PROFILE -u XHDFE_REDUCED_PCG_CUDA -u XHDFE_CERTIFY
       -u XHDFE_TEST_REDUCED_PCG_DECLINE -u OMP_THREAD_LIMIT XHDFE_AUTO_TEAM_COORDINATION=0
       XHDFE_REDUCED_PCG_TRACE=1 XHDFE_AUTO_ROUTING_RETRY_DIAG=1)
status=0
modes="lowmob_savefe lowmob_plain declined_savefe declined_plain weighted_savefe below_savefe below_plain
       e1off_savefe krylov_savefe krylov_plain sequence"
for arm in new frozen; do
  pkg=$NEW; [ "$arm" = frozen ] && pkg=$BASE
  for mode in $modes; do
    leg="${arm}_${mode}"
    extra=()
    case "$mode" in
      e1off_*) extra=(XHDFE_SAVEFE_ACCURATE_PROJECTION=0) ;;
      krylov_*) extra=(XHDFE_USE_KRYLOV=1) ;;
      declined_*) extra=(XHDFE_TEST_REDUCED_PCG_DECLINE=iterate) ;;
    esac
    mkdir -p "stata_tmp_${leg}"
    "${clean[@]}" "STATATMP=$OUT/stata_tmp_${leg}" "XHDFE_STATA_ADOPATH=$pkg" "${extra[@]}" \
      "$STATA" -q -b do reduced_pcg_trace_final_contract.do "$mode" "$leg" 2> "trace_${leg}.txt"
    mv reduced_pcg_trace_final_contract.log "log_${leg}.log"
    if grep -q "^RPCGF_PASS $mode" "log_${leg}.log" && ! grep -q "^r([0-9]*);" "log_${leg}.log"; then
      echo "RAN $leg"
    else
      echo "FAIL $leg"; tail -n 20 "log_${leg}.log"; status=1
    fi
    left=$(ls -A "stata_tmp_${leg}")
    [ -n "$left" ] && { echo "STOP: files left in $OUT/stata_tmp_${leg}: $left"; status=1; }
  done
done
"$PYTHON" -B -I - "$OUT" <<'PY' || status=1
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
FINAL = "xhdfe reduced_pcg event=final "
SECONDS = re.compile(r"seconds=\S+")
UNTOUCHED = {"lowmob_plain", "declined_plain", "weighted_savefe", "below_savefe", "below_plain", "e1off_savefe",
             "krylov_plain"}
PUBLISHED = {"lowmob_savefe": "lowmob_plain", "declined_savefe": "declined_plain", "krylov_savefe": None}
failures = []


def fail(msg):
    failures.append(msg)
    print("FAIL", msg)


def trace(leg):
    return (out / f"trace_{leg}.txt").read_text(errors="replace").splitlines()


def tier(lines):
    return [SECONDS.sub("seconds=*", l) for l in lines if l.startswith("xhdfe reduced_pcg ")]


def receipt(leg):
    fits, values = [], []
    for line in (out / f"rpcgf_{leg}.txt").read_text().splitlines():
        if line.startswith("FIT "):
            parts = line.split()
            fits.append(dict(tag=parts[1], rc=parts[2], status=parts[3], reason=parts[4], method=parts[5],
                             iterations=parts[6]))
            values.append([])
        else:
            values[-1].append(line)
    return fits, values


def agrees(line, fit):
    f = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
    want = {"accepted": "1", "declined": "2", "ineligible": "0"}.get(f.get("status"))
    reason_ok = fit["reason"] == f.get("reason") or (f.get("status") == "accepted" and fit["reason"] == "none")
    return want is not None and fit["status"] == want and reason_ok


for mode in sorted({p.name[len("rpcgf_new_"):-4] for p in out.glob("rpcgf_new_*.txt")}):
    new_fits, new_vals = receipt(f"new_{mode}")
    old_fits, old_vals = receipt(f"frozen_{mode}")
    new_lines, old_lines = trace(f"new_{mode}"), trace(f"frozen_{mode}")
    finals = [l[len(FINAL):] for l in new_lines if l.startswith(FINAL)]
    old_finals = [l[len(FINAL):] for l in old_lines if l.startswith(FINAL)]
    if len(finals) != len(new_fits) or len(old_finals) != len(old_fits):
        fail(f"{mode}: {len(finals)} final lines for {len(new_fits)} fits (frozen {len(old_finals)}/{len(old_fits)})")
        continue
    for line, fit in zip(finals, new_fits):
        if fit["rc"] != "0":
            fail(f"{mode}/{fit['tag']}: rc {fit['rc']}")
        elif not agrees(line, fit):
            fail(f"{mode}/{fit['tag']}: line '{line}' disagrees with e() {fit['status']}/{fit['reason']}")
    for nf, of, nv, ov in zip(new_fits, old_fits, new_vals, old_vals):
        if nv != ov:
            fail(f"{mode}/{nf['tag']}: b or V differ from frozen")
        keys = ("rc", "status", "reason", "iterations")
        if any(nf[k] != of[k] for k in keys):
            fail(f"{mode}/{nf['tag']}: e() differs from frozen: {nf} / {of}")
        # Under the forced Krylov route a package with code 9 (point 14, a separate patch, whose own
        # contract checks it strictly) reports 9; otherwise the frozen package's code.
        allowed = {of["method"], "9"} if mode.startswith("krylov_") else {of["method"]}
        if nf["method"] not in allowed:
            fail(f"{mode}/{nf['tag']}: e(absorption_method_used) {nf['method']}, want {sorted(allowed)}")
    if mode in UNTOUCHED and tier(new_lines) != tier(old_lines):
        fail(f"{mode}: the tier's lines differ from frozen: {tier(new_lines)} / {tier(old_lines)}")
    if mode in PUBLISHED:
        if not any(l.startswith("xhdfe savefe_accurate_projection published=1") for l in new_lines):
            fail(f"{mode}: E1 did not publish the projection")
        plain = PUBLISHED[mode]
        if plain and tier(new_lines) != tier(trace(f"new_{plain}")):
            fail(f"{mode}: the tier's lines differ from {plain}: {tier(new_lines)} / {tier(trace(f'new_{plain}'))}")
        if mode == "krylov_savefe" and finals != ["status=ineligible reason=forced_solver"]:
            fail(f"{mode}: final line {finals}")
    if mode == "sequence":
        published = sum(1 for l in new_lines if l.startswith("xhdfe savefe_accurate_projection published=1"))
        print(f"INFO sequence: {len(finals)} final lines for {len(new_fits)} fits in order; {published} projections "
              f"published")
    print(f"INFO {mode}: final {[SECONDS.sub('seconds=*', f) for f in finals]} (frozen "
          f"{[SECONDS.sub('seconds=*', f) for f in old_finals]}); e() "
          f"{[(f['status'], f['reason'], f['method'], f['iterations']) for f in new_fits]}")
print("RPCGF_COMPARE_PASS" if not failures else f"RPCGF_COMPARE_FAIL {len(failures)}")
sys.exit(1 if failures else 0)
PY
if [ "$status" -eq 0 ]; then echo "RPCGF_CONTRACT_PASS"; else echo "RPCGF_CONTRACT_FAIL"; fi
exit "$status"
