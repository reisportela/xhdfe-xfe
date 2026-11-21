#!/bin/bash
# Point 14 (17nov2026): the Stata leg of tests/absorption_method_declared_contract.py. Every mode of
# tests/stata/absorption_method_declared_contract.do in its own fresh stata-mp process, for the new
# and the frozen package, in a new scratch directory, each leg with its own STATATMP (checked empty
# afterwards). Synthetic data, CPU, numthreads(4). Then, per mode, new against frozen:
#   - every coefficient and element of e(V) (xhdfe), and the signature and sums of the partialled
#     variables (xfepout), bit for bit; e(iterations) and the reduced tier's status and reason equal;
#   - e(absorption_method_used) (xhdfe) or e(method_used) (xfepout): 9 where the Krylov-PCG route is
#     forced and the Krylov-PCG absorber produced the transform (XHDFE_USE_KRYLOV=1 for xhdfe; a
#     mobility profile that suggests the route for xhdfe and xfepout), the frozen package's code
#     elsewhere, including xfepout under XHDFE_USE_KRYLOV=1, which the partialling-out entry does not
#     read;
#   - krylov-pcg and krylov_pcg as absorptionmethod() rejected with the return code of reduced-pcg
#     and a message that says the method is reported, never requested;
#   - the profile xfepout writes identical in both packages.
# usage: absorption_method_declared_contract.sh <new package stata dir> <frozen package stata dir> <new scratch dir>
# Never set OMP_THREAD_LIMIT for Stata legs.
set -u
NEW=${1:?new package stata directory}
BASE=${2:?frozen package stata directory}
OUT=${3:?new scratch directory}
STATA=${STATA:-stata-mp}
PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" && cp "$HERE/absorption_method_declared_contract.do" "$OUT/" && cd "$OUT" || exit 2
clean=(env -u XHDFE_REDUCED_PCG -u XHDFE_REDUCED_PCG_TRACE -u XHDFE_SAVEFE_ACCURATE_PROJECTION -u XHDFE_USE_KRYLOV
       -u XHDFE_ABSORPTION_CACHE -u XHDFE_ABSORPTION_CACHE_MODE -u XHDFE_GPU_BACKEND -u XHDFE_PROFILE_CPU
       -u XHDFE_MOBILITY_MODE -u XHDFE_MOBILITY_PROFILE -u XHDFE_REDUCED_PCG_CUDA -u XHDFE_CERTIFY
       -u AMD_PROFILE -u OMP_THREAD_LIMIT XHDFE_AUTO_TEAM_COORDINATION=0)
status=0
modes="forced_fast forced_comparable plain_fast plain_comparable forced_savefe plain_savefe request xfepout_env
       xfepout_plain profile_make xfepout_profile xhdfe_profile"
for arm in new frozen; do
  pkg=$NEW; [ "$arm" = frozen ] && pkg=$BASE
  for mode in $modes; do
    leg="${arm}_${mode}"
    extra=()
    case "$mode" in
      forced_*|xfepout_env) extra=(XHDFE_USE_KRYLOV=1) ;;
      xfepout_profile|xhdfe_profile) extra=("AMD_PROFILE=$OUT/amd_profile_krylov_${arm}_profile_make.txt") ;;
    esac
    mkdir -p "stata_tmp_${leg}"
    "${clean[@]}" "STATATMP=$OUT/stata_tmp_${leg}" "XHDFE_STATA_ADOPATH=$pkg" "${extra[@]}" \
      "$STATA" -q -b do absorption_method_declared_contract.do "$mode" "$leg" 2> "stderr_${leg}.txt"
    mv absorption_method_declared_contract.log "log_${leg}.log"
    if grep -q "^AMD_PASS $mode" "log_${leg}.log" && ! grep -q "^r([0-9]*);" "log_${leg}.log"; then
      echo "RAN $leg"
    else
      echo "FAIL $leg"; tail -n 20 "log_${leg}.log"; status=1
    fi
    left=$(ls -A "stata_tmp_${leg}")
    [ -n "$left" ] && { echo "STOP: files left in $OUT/stata_tmp_${leg}: $left"; status=1; }
  done
done
"$PYTHON" -B -I - "$OUT" <<'PY' || status=1
import sys
from pathlib import Path

out = Path(sys.argv[1])
NINE = {"forced_fast", "forced_comparable", "forced_savefe", "xfepout_profile", "xhdfe_profile"}
failures = []


def fail(msg):
    failures.append(msg)
    print("FAIL", msg)


def receipt(leg):
    fits, values = [], []
    for line in (out / f"amd_{leg}.txt").read_text().splitlines():
        if line.startswith("FIT "):
            p = line.split()
            fits.append(dict(tag=p[1], rc=p[2], method=p[3], iterations=p[4], status=p[5], reason=p[6]))
            values.append([])
        elif line.startswith("REQUEST "):
            fits.append(dict(tag=line.split()[1], request=line.split()[2]))
            values.append([])
        elif values:
            values[-1].append(line)
    return fits, values


modes = sorted({p.name[len("amd_new_"):-4] for p in out.glob("amd_new_*.txt")})
for mode in modes:
    nf, nv = receipt(f"new_{mode}")
    of, ov = receipt(f"frozen_{mode}")
    if len(nf) != len(of) or not nf:
        fail(f"{mode}: {len(nf)} records, frozen {len(of)}")
        continue
    if mode == "request":
        rc = {f["tag"]: f["request"] for f in nf}
        log = (out / f"log_new_{mode}.log").read_text(errors="replace")
        if rc.get("reduced-pcg") in (None, "rc=0") or rc.get("krylov-pcg") != rc.get("reduced-pcg") or \
                rc.get("krylov_pcg") != rc.get("reduced-pcg"):
            fail(f"request: return codes {rc}")
        if "krylov-pcg is reported when the Krylov-PCG route" not in log:
            fail("request: the rejection message of krylov-pcg is not in the log")
        print(f"INFO request: new {rc}; frozen {[f['request'] for f in of]}")
        continue
    for a, b, va, vb in zip(nf, of, nv, ov):
        if va != vb:
            fail(f"{mode}/{a['tag']}: the published numbers differ from frozen")
        if any(a[k] != b[k] for k in ("rc", "iterations", "status", "reason")):
            fail(f"{mode}/{a['tag']}: e() differs from frozen: {a} / {b}")
        want = "9" if mode in NINE else b["method"]
        if a["method"] != want or (mode in NINE and b["method"] == "9"):
            fail(f"{mode}/{a['tag']}: declared {a['method']}, want {want} (frozen {b['method']})")
        print(f"INFO {mode}: declared {a['method']} (frozen {b['method']}); iterations {a['iterations']}; "
              f"tier {a['status']}/{a['reason']}; numbers {'bit-identical' if va == vb else 'DIFFER'}")
pn, pf = out / "amd_profile_new_profile_make.txt", out / "amd_profile_frozen_profile_make.txt"
if not pn.exists() or pn.read_bytes() != pf.read_bytes():
    fail("profile_make: the profiles differ between the packages")
kn = out / "amd_profile_krylov_new_profile_make.txt"
if not kn.exists() or "suggest_krylov=1" not in kn.read_text():
    fail("profile_make: no profile suggesting the Krylov-PCG route")
print("AMD_COMPARE_PASS" if not failures else f"AMD_COMPARE_FAIL {len(failures)}")
sys.exit(1 if failures else 0)
PY
if [ "$status" -eq 0 ]; then echo "AMD_CONTRACT_PASS"; else echo "AMD_CONTRACT_FAIL"; fi
exit "$status"
