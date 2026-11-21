#!/bin/bash
# The per-fit request of the experimental CUDA reduced tier through Stata
# (tests/stata/reduced_cuda_request_contract.do, package 1 with its Amendment 1): every mode in its
# own fresh stata-mp process, in a new scratch directory, with stderr kept (the tier's trace).
# Synthetic data only; CPU and CUDA (a CUDA plugin and a visible device are required). The do-file
# is copied once per leg under the leg's name, so that each batch log carries that name (nothing is
# moved or renamed).
# usage: reduced_cuda_request_contract.sh <package stata dir> <new scratch dir>
set -u
PKG=${1:?package stata directory}
OUT=${2:?new scratch directory}
STATA=${STATA:-stata-mp}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -e "$OUT" ]; then echo "refusing to reuse $OUT" >&2; exit 2; fi
mkdir -p "$OUT" || exit 2
clean=(env -u LD_LIBRARY_PATH -u XHDFE_REDUCED_PCG -u XHDFE_REDUCED_PCG_CUDA -u XHDFE_REDUCED_PCG_TRACE
       -u XHDFE_TEST_REDUCED_PCG_DECLINE -u XHDFE_TEST_REDUCED_PCG_CUDA_FATAL
       -u XHDFE_ABSORPTION_CACHE -u XHDFE_ABSORPTION_CACHE_MODE -u XHDFE_FE_STRUCTURE_CACHE
       -u XHDFE_FE_STRUCTURE_MODE -u XHDFE_MOBILITY_MODE -u XHDFE_MOBILITY_PROFILE
       -u XHDFE_GPU_BACKEND -u XHDFE_PROFILE_CPU -u XHDFE_REDUCED_PCG_COMPANION
       -u XHDFE_AUTO_ROUTING_RETRY_DIAG -u XHDFE_SAVEFE_ACCURATE_PROJECTION
       "XHDFE_STATA_ADOPATH=$PKG" "CUDA_CACHE_PATH=$OUT/cuda_cache")
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$OUT/gpu_apps_before.txt" 2>&1
fi
status=0
# leg <name> <env assignment>... -- <do-file arguments>...
leg() {
  local name=$1; shift
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  cp "$HERE/reduced_cuda_request_contract.do" "$OUT/rq_${name}.do" || { status=1; return; }
  ( cd "$OUT" && "${clean[@]}" "${vars[@]}" "$STATA" -q -b do "rq_${name}.do" "$@" \
      2> "trace_${name}.txt" )
  if grep -q "^REQUEST_PASS" "$OUT/rq_${name}.log"; then
    echo "PASS $name"
  else
    echo "FAIL $name"; tail -n 25 "$OUT/rq_${name}.log"; status=1
  fi
}
# Lines of the trace: final lines (CUDA tier accepted, CPU tier accepted, by reason), CUDA operator
# lines with and without products, and the saved-effects projection's diagnostic line.
cuda_ok() { grep "event=final status=accepted" "$OUT/trace_$1.txt" | grep -c " backend=cuda " || true; }
cpu_ok() { grep "event=final status=accepted" "$OUT/trace_$1.txt" | grep -vc " backend=cuda " || true; }
why() { grep -c "event=final status=ineligible reason=$2\$" "$OUT/trace_$1.txt" || true; }
finals() { grep -c "reduced_pcg event=final" "$OUT/trace_$1.txt" || true; }
calls() { grep "reduced_pcg_cuda event=operator calls=" "$OUT/trace_$1.txt" | grep -vc " calls=0 " || true; }
calls0() { grep -c "reduced_pcg_cuda event=operator calls=0 " "$OUT/trace_$1.txt" || true; }
published() { grep -c "savefe_accurate_projection published=1" "$OUT/trace_$1.txt" || true; }
expect() {
  local name=$1 what=$2 got=$3 want=$4
  if [ "$got" -ne "$want" ]; then echo "FAIL $name trace: $what $got, expected $want"; status=1; fi
}
leg session XHDFE_REDUCED_PCG_TRACE=1 -- session
leg reverse XHDFE_REDUCED_PCG_TRACE=1 -- reverse
leg envon XHDFE_REDUCED_PCG_CUDA=1 XHDFE_REDUCED_PCG_TRACE=1 -- envon
leg envoff XHDFE_REDUCED_PCG_CUDA=0 XHDFE_REDUCED_PCG_TRACE=1 -- envoff
leg profile XHDFE_REDUCED_PCG_TRACE=1 -- profile
leg invalid XHDFE_REDUCED_PCG_TRACE=1 -- invalid
leg savefe XHDFE_REDUCED_PCG_TRACE=1 XHDFE_AUTO_ROUTING_RETRY_DIAG=1 -- savefe
leg savefefatal XHDFE_REDUCED_PCG_TRACE=1 XHDFE_AUTO_ROUTING_RETRY_DIAG=1 \
    XHDFE_TEST_REDUCED_PCG_CUDA_FATAL=2 -- savefefatal
leg firstoption XHDFE_REDUCED_PCG_TRACE=1 -- firstoption
leg firstglobal XHDFE_REDUCED_PCG_TRACE=1 -- firstglobal
leg cpufresh -- cpufresh
leg compare -- compare
# session: 5 CUDA tier fits, 3 CPU tier fits, 5 CUDA fits without the request (gpu), one
# xhdfe-fast fit (tolerance_mode), none for the 3 fits with the tier off: 14 final lines; every
# CUDA operator line with products; no cache or profile reason.
expect session "CUDA accepted" "$(cuda_ok session)" 5
expect session "CPU accepted" "$(cpu_ok session)" 3
expect session "reason gpu" "$(why session gpu)" 5
expect session "reason tolerance_mode" "$(why session tolerance_mode)" 1
for r in mobility_profile absorption_cache fe_structure_cache; do
  expect session "reason $r" "$(why session $r)" 0
done
expect session "final lines" "$(finals session)" 14
expect session "operator lines with products" "$(calls session)" 5
expect session "operator lines with calls=0" "$(calls0 session)" 0
# reverse: 2 CUDA tier fits, 2 without the request, the CPU default; the CPU fit with the tier
# off prints nothing.
expect reverse "CUDA accepted" "$(cuda_ok reverse)" 2
expect reverse "reason gpu" "$(why reverse gpu)" 2
expect reverse "CPU accepted" "$(cpu_ok reverse)" 1
expect reverse "final lines" "$(finals reverse)" 5
# envon: two CUDA tier fits, two CUDA fits without it, one CPU tier fit.
expect envon "CUDA accepted" "$(cuda_ok envon)" 2
expect envon "reason gpu" "$(why envon gpu)" 2
expect envon "CPU accepted" "$(cpu_ok envon)" 1
expect envon "final lines" "$(finals envon)" 5
# envoff: two CUDA fits without the tier, two CUDA tier fits.
expect envoff "reason gpu" "$(why envoff gpu)" 2
expect envoff "CUDA accepted" "$(cuda_ok envoff)" 2
expect envoff "final lines" "$(finals envoff)" 4
# profile: the implicit profile keeps the request out (mobility_profile), the fit without the
# request says gpu, and the request enters once the working directory has no profile.
expect profile "reason mobility_profile" "$(why profile mobility_profile)" 1
expect profile "reason gpu" "$(why profile gpu)" 1
expect profile "CUDA accepted" "$(cuda_ok profile)" 1
expect profile "final lines" "$(finals profile)" 3
# invalid: the fit with saved effects prints its own line; the refused call prints nothing.
expect invalid "reason savefe" "$(why invalid savefe)" 1
expect invalid "final lines" "$(finals invalid)" 1
# savefe: the twin's CUDA tier fit, and the fit with saved effects, whose projection is published
# by the CUDA tier: the fit's final line is the projection's (accepted on the device), as it reports.
expect savefe "CUDA accepted" "$(cuda_ok savefe)" 2
expect savefe "reason savefe" "$(why savefe savefe)" 0
expect savefe "operator lines with products" "$(calls savefe)" 2
expect savefe "projection published" "$(published savefe)" 1
# savefefatal: three fits with saved effects; the second one's projection meets the injected fatal
# error (no operator line, no published projection, no estimates: its line stays savefe); the first
# and the third publish it, and their final lines are the projection's (accepted on the device).
expect savefefatal "reason savefe" "$(why savefefatal savefe)" 1
expect savefefatal "CUDA accepted" "$(cuda_ok savefefatal)" 2
expect savefefatal "final lines" "$(finals savefefatal)" 3
expect savefefatal "operator lines with products" "$(calls savefefatal)" 2
expect savefefatal "projection published" "$(published savefefatal)" 2
# firstoption: the auxiliary fits of estat firststage follow the (absent) global and environment.
expect firstoption "CUDA accepted" "$(cuda_ok firstoption)" 0
n=$(why firstoption gpu)
if [ "$n" -lt 2 ]; then echo "FAIL firstoption trace: $n auxiliary lines with reason gpu"; status=1; fi
# firstglobal: with the global at 1 they enter the CUDA tier.
n=$(cuda_ok firstglobal)
if [ "$n" -lt 2 ]; then echo "FAIL firstglobal trace: $n CUDA accepted auxiliary lines"; status=1; fi
if [ "$status" -eq 0 ]; then echo "REQUEST_CONTRACT_PASS"; else echo "REQUEST_CONTRACT_FAIL"; fi
exit "$status"
