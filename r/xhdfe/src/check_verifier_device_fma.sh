#!/usr/bin/env bash
# §7.6 device-side flag gate (audit A-02, hardened per re-audit R-04). The
# host-side §15.3 gate proves the HOST switches from .GCC.command.line, but
# NVHPC 25.3's nvcc was caught contracting a*b+c into DFMA in the verifier
# kernels DESPITE --fmad=false — invisible to every host-side check. After
# the A-02 fix the verifier's accumulation and decision arithmetic is written
# exclusively with explicit __dmul_rn/__dadd_rn/__d*_rd/__d*_ru intrinsics,
# which nvcc never contracts.
#
# R-04 hardening over the original gate:
#   - PTX regex also catches PREDICATED forms (@%p1 fma.rn.f64 ...);
#   - SASS is checked too (DFMA/FFMA/HFMA), not just PTX;
#   - analysis is restricted to hdfe_cert functions, so the gate also works
#     on LINKED artifacts (.so/.plugin) whose absorber kernels legitimately
#     use FMA;
#   - allow-list: bracket_decide/gamma_of SASS may contain DFMA inside the
#     __ddiv_ru/__drcp Newton expansions (IEEE-correct division is
#     implemented WITH fma by design; the intrinsic itself is correctly
#     rounded). PTX-level fma remains forbidden even there. The allowed SASS
#     count is reported for the record.
#
# Fail-closed: no dump extractable => FAIL.
# Usage: check_verifier_device_fma.sh <verifier .o | linked .so/.plugin>
set -euo pipefail

OBJ="${1:?usage: check_verifier_device_fma.sh <object>}"
# cuobjdump lookup: env override, PATH, next to nvcc (network/minimal CUDA
# installs put both in the same bindir when the cuobjdump package is
# present), then the local HPC-SDK path. Still fail-closed when absent.
CUOBJDUMP="${CUOBJDUMP:-$(command -v cuobjdump || true)}"
if [[ -z "$CUOBJDUMP" || ! -x "$CUOBJDUMP" ]]; then
  _nvcc="$(command -v nvcc || true)"
  if [[ -n "$_nvcc" && -x "$(dirname "$_nvcc")/cuobjdump" ]]; then
    CUOBJDUMP="$(dirname "$_nvcc")/cuobjdump"
  fi
fi
if [[ -z "$CUOBJDUMP" || ! -x "$CUOBJDUMP" ]]; then
  for _cand in /usr/local/cuda/bin/cuobjdump                /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/bin/cuobjdump; do
    [[ -x "$_cand" ]] && { CUOBJDUMP="$_cand"; break; }
  done
fi
if [[ -z "$CUOBJDUMP" || ! -x "$CUOBJDUMP" ]]; then
  echo "FAIL: cuobjdump not found — the gate cannot certify what it cannot see" >&2
  exit 1
fi

# Extract only hdfe_cert function bodies from a dump. Function headers:
#   PTX:  .visible .entry _ZN9hdfe_cert...  /  .func ... _ZN9hdfe_cert...
#   SASS: "Function : _ZN9hdfe_cert..."
filter_ptx() {
  awk '
    /\.(entry|func)/ { infun = ($0 ~ /9hdfe_cert/) }
    infun { print }
  '
}
filter_sass() {
  awk '
    /^[[:space:]]*Function[[:space:]]*:/ { infun = ($0 ~ /9hdfe_cert/) }
    infun { print }
  '
}

ptx_all="$("$CUOBJDUMP" --dump-ptx "$OBJ" 2>/dev/null || true)"
if [[ -z "$ptx_all" ]]; then
  echo "FAIL: no PTX extractable from $OBJ (gate is fail-closed)" >&2
  exit 1
fi
ptx="$(filter_ptx <<<"$ptx_all")"
if [[ -z "$ptx" ]]; then
  echo "FAIL: no hdfe_cert functions found in PTX of $OBJ (wrong object, or namespace renamed — gate is fail-closed)" >&2
  exit 1
fi

# PTX: any fma, predicated or not, anywhere in verifier functions => FAIL.
fma_lines="$(grep -nE '^\s*(@!?%p[0-9]+\s+)?fma\.' <<<"$ptx" || true)"
if [[ -n "$fma_lines" ]]; then
  echo "FAIL: verifier PTX contains fma instructions (§7.6 forbids implicit contraction):" >&2
  head -10 <<<"$fma_lines" >&2
  echo "count: $(wc -l <<<"$fma_lines")" >&2
  exit 1
fi
scalar_n="$(grep -cE '^\s*(add|mul)\.rn\.f64|^\s*(add|mul)\.f64' <<<"$ptx" || true)"

# SASS: DFMA forbidden outside the bracket_decide/decide div-expansion
# allow-list; predicated (@P0 DFMA) caught by matching the opcode anywhere.
sass_all="$("$CUOBJDUMP" --dump-sass "$OBJ" 2>/dev/null || true)"
if [[ -z "$sass_all" ]]; then
  echo "FAIL: no SASS extractable from $OBJ (gate is fail-closed; PTX alone is not proof of what the GPU executes)" >&2
  exit 1
fi
sass="$(filter_sass <<<"$sass_all")"
if [[ -z "$sass" ]]; then
  echo "FAIL: no hdfe_cert functions found in SASS of $OBJ (gate is fail-closed)" >&2
  exit 1
fi
sass_viol="$(awk '
  /^[[:space:]]*Function[[:space:]]*:/ {
    fn = $0
    allowed = ($0 ~ /bracket_decide|decide_one|xhdfe_cert_decide_device/)
  }
  # DFMA = double-precision contraction, the §7.6 subject — strict everywhere
  # outside the div/sqrt-expansion allow-list.
  /\yDFMA\y/ {
    if (allowed) { allow_n++ } else { print fn; print; viol = 1 }
    next
  }
  # FFMA/HFMA on real data would also be a bug (the verifier is pure f64),
  # BUT compilers emit "HFMA2.MMA Rn, -RZ, RZ, imm, imm" as a constant-
  # materialization idiom (-0*0+imm == MOV) with no data arithmetic — skip
  # exactly that pattern, flag everything else.
  /\y(FFMA|HFMA2?)\y/ {
    if ($0 ~ /-?RZ,[[:space:]]*RZ,/) next
    if (allowed) { allow_n++ } else { print fn; print; viol = 1 }
  }
  END { printf "ALLOWED_COUNT=%d\n", allow_n > "/dev/stderr"; exit viol ? 0 : 0 }
' <<<"$sass" 2>/tmp/fma_gate_allowed.$$)"
allowed_count="$(sed -n 's/^ALLOWED_COUNT=//p' /tmp/fma_gate_allowed.$$ 2>/dev/null || echo '?')"
rm -f /tmp/fma_gate_allowed.$$
if [[ -n "$sass_viol" ]]; then
  echo "FAIL: SASS fma in verifier accumulation/decision functions (predication included):" >&2
  head -12 <<<"$sass_viol" >&2
  exit 1
fi
echo "OK: verifier device code clean — PTX zero fma (${scalar_n} scalar f64 add/mul); SASS zero fma outside the div/sqrt expansion allow-list (${allowed_count} allowed DFMA in bracket_decide/decide)"
