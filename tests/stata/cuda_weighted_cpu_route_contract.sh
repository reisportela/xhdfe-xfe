#!/bin/bash
# Package C (18nov2026): the Stata leg of the weighted CPU route (tests/stata/cuda_weighted_cpu_route_contract.do).
# Writes the synthetic fixtures (tests/cuda_weighted_cpu_route_contract.py --write-stata-data), runs one fresh
# stata-mp session per leg (each from its own folder, with its own STATATMP, checked empty afterwards, every XHDFE_
# variable purged and the leg's own set), and grades: each log ends without an error (r() lines), has no CWR_FAIL
# and closes with CWR_PASS; the routed and cpu fits agree bit for bit (est_<name>.txt equal; residuals and saved effects
# equal value for value); the partial-out is compared numerically across backends and each backend shows both expected
# refusals; the routed leg prints one note per routed fit and the cpu leg none. A visible CUDA device is
# required. The reghdfe leg runs when reghdfe is found (the optional adopath, appended, else the default one) and says
# so when it does not; the gelbach leg checks the r() status of xhdfegelbach.
# usage: cuda_weighted_cpu_route_contract.sh <package stata dir> <new scratch dir> [<reghdfe adopath>] [power]
set -u
PKG=${1:?package stata directory}
OUT=${2:?new scratch directory}
RGH=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
DO="$HERE/cuda_weighted_cpu_route_contract.do"
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT/data"
status=0
fail() { echo "FAIL: $*"; status=1; }
env -u LD_LIBRARY_PATH python3 -B "$HERE/../cuda_weighted_cpu_route_contract.py" --write-stata-data "$OUT/data" \
  || { echo "FAIL: fixtures not written"; exit 1; }
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$OUT/gpu_apps_at_start.txt" 2>&1
clean=$(env | sed -n 's/^\(XHDFE_[A-Za-z0-9_]*\)=.*/-u \1/p' | tr '\n' ' ')
# Batch Stata names the log after its last argument (the reghdfe folder, or an empty word), so a leg's log is the
# only .log file in its folder.
leg_log() {
  local logs=("$OUT/$1"/*.log)
  if [ ${#logs[@]} -eq 1 ] && [ -f "${logs[0]}" ]; then echo "${logs[0]}"; fi
}
run_leg() {
  local kind="$1"; shift
  local leg="$OUT/$kind"
  mkdir -p "$leg/tmp"
  # shellcheck disable=SC2086
  (cd "$leg" && env $clean -u LD_LIBRARY_PATH STATATMP="$leg/tmp" "$@" \
     stata-mp -q -b do "$DO" "$PKG" "$OUT/data" "$kind" "$leg" "$RGH")
  local log; log=$(leg_log "$kind")
  if [ -z "$log" ]; then fail "$kind: not exactly one log in $leg"; return; fi
  if grep -nE '^r\([0-9]+\);' "$log" > /dev/null; then fail "$kind: the session ended with an error: $(grep -nE '^r\([0-9]+\);' "$log" | tail -1)"; fi
  if grep -q '^CWR_FAIL' "$log"; then grep -n '^CWR_FAIL' "$log" | sed "s/^/FAIL: $kind: /"; status=1; fi
  grep -q '^CWR_PASS' "$log" || fail "$kind: no CWR_PASS"
  if [ -n "$(ls -A "$leg/tmp")" ]; then fail "$kind: STATATMP not empty: $(ls "$leg/tmp" | head -3)"; fi
  echo "leg $kind: $(grep -c '^CWR_OK' "$log") fits ok, $(grep -c '^CWR_FAIL' "$log") failures"
}
if [ "${4:-}" = power ]; then
  run_leg power
  grep -q '^CWR_OK name=power_baseline cmd=xhdfe gpu=1 code=1 st=used' "$(leg_log power)" || fail "the baseline must complete a weighted fit on CUDA"
  echo "BASELINE_STATUS $([ "$status" -eq 0 ] && echo PASS || echo FAIL)"
  exit "$status"
fi
run_leg routed
run_leg cpu
run_leg env XHDFE_GPU_BACKEND=cuda
run_leg boundaries
run_leg session XHDFE_GPU_BACKEND=cuda
run_leg switch0
run_leg reghdfe
run_leg gelbach
# The routed fits print one note each; the CPU ones none.
notes_routed=$(grep -c '^note: weighted fit absorbed on the CPU (e(gpu_status) = weighted_cpu_route)' "$(leg_log routed)")
notes_cpu=$(grep -c '^note: weighted \(fit\|partial-out\) absorbed on the CPU' "$(leg_log cpu)")
[ "$notes_routed" -eq 11 ] || fail "routed leg: $notes_routed note lines (expected 11)"
[ "$notes_cpu" -eq 0 ] || fail "cpu leg: $notes_cpu note lines (expected 0)"
if grep -q '^note: weighted partial-out absorbed on the CPU' "$OUT"/*/*.log; then fail "partial-out routing note"; fi
grep -q '^GPU use was required.*runs on the CPU by design (r(gpu_status) = weighted_cpu_route)' "$(leg_log gelbach)" \
  || fail "gelbach: missing requiregpu refusal reason"
grep -h '^CWR_SKIP\|^CWR_R ' "$(leg_log reghdfe)" || true
# The identity of routed fits and the CPU reference, and the partial-out observations on both backends.
env -u LD_LIBRARY_PATH python3 -B - "$OUT" "$(leg_log routed)" "$(leg_log cpu)" <<'EOF' || status=1
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
out = Path(sys.argv[1])
bad = 0
fits = ("aw3", "fw3", "pw3", "iw3", "aw2", "aw4", "aw3_strict", "aw3_savefe", "aw2_copy", "xfe_aw23",
        "iv_first", "grp_aw2")
variables = ("aw3", "aw3_strict", "aw3_savefe", "xfe_aw23")
for leg in ("routed", "cpu"):
    for prefix, suffix, names in (("est", ".txt", fits), ("vars", ".dta", variables)):
        expected = {f"{prefix}_{name}{suffix}" for name in names}
        actual = {file.name for file in (out / leg).glob(f"{prefix}_*{suffix}")}
        if actual != expected:
            print(f"FAIL: {leg}: missing {sorted(expected - actual)}; extra {sorted(actual - expected)}")
            bad += 1


def refusals(log):
    # The message a refused fit printed: the last line starting with "HDFE " between the previous CWR_ line and
    # its CWR_OK ... refused line, with the wrapped continuations ("> ") joined.
    lines, found = Path(log).read_text(errors='replace').splitlines(), {}
    for i, line in enumerate(lines):
        m = re.match(r'^CWR_OK name=(\S+) refused rc=(\d+)', line)
        if not m:
            continue
        start = max((k for k in range(i) if lines[k].startswith('CWR_')), default=-1) + 1
        j = max((k for k in range(start, i) if lines[k].startswith('HDFE ')), default=None)
        text = None if j is None else ''.join([lines[j]] + [x[2:] for x in lines[j + 1:i] if x.startswith('> ')])
        found[m.group(1)] = (m.group(2), text)
    return found


for leg, path in zip(('cuda', 'cpu'), sys.argv[2:]):
    refused = refusals(path)
    for name in ('xfe_aw3_refused', 'xfe_aw12_refused'):
        rc, reason = refused.get(name, (None, None))
        if rc != '198' or reason is None or 'precision certification failed' not in reason:
            print(f'FAIL: {leg} refusal {name}: rc={rc}, reason={reason}')
            bad += 1
        else:
            print(f'{leg}: {name} refused on precision grounds (rc {rc})')
for est in sorted((out / 'routed').glob('est_*.txt')):
    other = out / 'cpu' / est.name
    if not other.is_file() or other.read_text() != est.read_text():
        print(f'FAIL: identity: {est.name} differs between the routed and the cpu legs'); bad += 1
    else:
        print(f'identity: {est.name} equal')
for vars_ in sorted((out / 'routed').glob('vars_*.dta')):
    a = pd.read_stata(vars_, convert_categoricals=False)
    b = pd.read_stata(out / 'cpu' / vars_.name, convert_categoricals=False)
    if vars_.name == 'vars_xfe_aw23.dta':
        same = a.columns.equals(b.columns) and a.shape == b.shape and np.allclose(
            a.to_numpy(), b.to_numpy(), rtol=1e-8, atol=1e-9, equal_nan=True)
        print(f"partial-out: {vars_.name} {'within tolerance' if same else 'DIFFERS'}")
        bad += 0 if same else 1
        continue
    same = list(a.columns) == list(b.columns) and all(
        np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True) and
        a[c].to_numpy().tobytes() == b[c].to_numpy().tobytes() for c in a.columns)
    print(f"identity: {vars_.name} {'equal' if same else 'DIFFERS'} ({', '.join(a.columns)})")
    bad += 0 if same else 1
sys.exit(1 if bad else 0)
EOF
echo "STATUS $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
