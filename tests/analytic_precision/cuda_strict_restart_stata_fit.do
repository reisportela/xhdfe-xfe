* P3-fix (3'): one strict-residual CUDA fit of the cov240k fixture through the Stata plugin, for
* the strict restart gate (Verifications/perf_fereg_gap_20261106/p3fix/p3fix_strict_stata_gate.py,
* which writes the fixture with the oracle's within residuals e_oracle and the oracle b and V, and
* runs one fresh Stata process per fit with XHDFE_TEST_CUDA_CG_BREAKDOWN, XHDFE_GI_LSMR_TRACE=1,
* XHDFE_ACCEL_DEBUG=1 and XHDFE_CUDA_ACCEL_TRACE=1, so the first CG stage of the session breaks
* down and the strict device LSMR restart runs). Prints one P3S_STATA line with e() diagnostics,
* the distances to the oracle (b, V, SE, stored residuals), the largest absolute group mean of the
* stored residuals over the three FEs (D) and 1 + the l1 norm of the slopes (B1); the driver grades
* them. A refusal prints P3S_STATA_REFUSED with the return code and whether e() was left empty.
* Optional sixth argument: the tolerance mode (default strict-residual; the Fast gate passes xhdfe-fast);
* optional seventh: the convergence criterion (default auto; scope addendum 2 passes reghdfe);
* optional eighth: the absorption method (default auto); optional ninth: the backend (default cuda).
version 18
clear all
set more off
set linesize 255
args package fixture oracle tolerance maxiter mode convergence method backend
if "`mode'" == "" local mode "strict-residual"
if "`convergence'" == "" local convergence "auto"
if "`method'" == "" local method "auto"
if "`backend'" == "" local backend "cuda"
adopath ++ "`package'"
discard
findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
use "`fixture'", clear
do "`oracle'"
local k = colsof(oracle_b)
capture noisily xhdfe y x1 x2 x3 x4, absorb(worker firm year) vce(cluster worker) ///
    tolerance(`tolerance') tolerancemode(`mode') absorptioncachemode(off) ///
    numthreads(16) gpubackend(`backend') maxiter(`maxiter') convergence(`convergence') ///
    absorptionmethod(`method') residuals(r_strict)
local rc = _rc
if `rc' != 0 {
    local empty = ("`e(cmd)'" == "")
    capture matrix stale_b = e(b)
    local nob = (_rc != 0)
    display "P3S_STATA_REFUSED rc=`rc' ecmd_empty=`empty' no_eb=`nob'"
    exit 0
}
generate byte s_sample = e(sample)
quietly count if s_sample & (missing(r_strict) | missing(e_oracle))
local missing = r(N)
mata: st_numscalar("db", max(abs(st_matrix("e(b)")-st_matrix("oracle_b")) :/ rowmax((J(`k',1,1),abs(st_matrix("oracle_b")')))'))
mata: st_numscalar("dv", max(abs(st_matrix("e(V)")-st_matrix("oracle_V")) :/ sqrt(diagonal(st_matrix("oracle_V"))*diagonal(st_matrix("oracle_V"))')))
mata: st_numscalar("dse", max(abs(sqrt(diagonal(st_matrix("e(V)"))) :/ sqrt(diagonal(st_matrix("oracle_V"))) :- 1)))
mata: st_numscalar("dr", norm(st_data(., "r_strict", "s_sample") - st_data(., "e_oracle", "s_sample")) / norm(st_data(., "e_oracle", "s_sample")))
mata: st_numscalar("b1", 1 + sum(abs(st_matrix("e(b)")[1, 1..4])))
scalar dmax = 0
foreach fe in worker firm year {
    tempvar m
    quietly bysort `fe': egen double `m' = mean(r_strict) if s_sample
    quietly summarize `m', meanonly
    scalar dmax = max(dmax, abs(r(min)), abs(r(max)))
    drop `m'
}
* Two short lines (the batch log wraps long lines): diagnostics, then the distances.
display "P3S_STATA iterations=" e(iterations) " converged=" e(converged) ///
    " certified=" e(precision_certified) " gpu_used=" e(gpu_used) ///
    " gpu_backend=`e(gpu_backend)' gpu_status=`e(gpu_status)' missing=`missing'"
display "P3S_STATA_M b=" %21.14e db " V=" %21.14e dv " se=" %21.14e dse " resid=" %21.14e dr ///
    " D=" %21.14e dmax " B1=" %21.14e b1 " arr=" %21.14e e(abs_residual_rel)
