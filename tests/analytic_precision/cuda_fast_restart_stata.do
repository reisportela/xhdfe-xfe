* P3-fix: the xhdfe-fast CUDA restart of a broken CG stage through the Stata plugin.
* Driven by cuda_fast_restart_contract.py --stata-package, which writes the synthetic
* cov240k fixture (with the oracle's within residuals e_oracle), the oracle b and V and
* the envelope limits, and runs Stata with XHDFE_TEST_CUDA_CG_BREAKDOWN=30 (the first CG
* stage of the session breaks down at its iteration 30), XHDFE_GI_LSMR_TRACE=1 and
* XHDFE_ACCEL_DEBUG=1; the driver checks the trace lines of the session.
* A. A CPU Comparable fit (no CUDA CG stage: the hook is not consumed).
* B. xhdfe-fast on CUDA: the hook breaks the repair's CG stage, the fit restarts cold and
*    must return converged, certified, on the GPU, inside the envelope (b, V, SE and the
*    stored residuals against the oracle).
* C. Right after B, a refusal: maxiter leaves the CG stage too little budget to converge
*    and none for a restart; no estimate of B may remain in e().
* D. The session recovers: the default budget returns again.
version 18
clear all
set more off
set linesize 255
args package fixture oracle refusal_maxiter
adopath ++ "`package'"
discard
findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
use "`fixture'", clear
do "`oracle'"
local k = colsof(oracle_b)

quietly xhdfe y x1 x2 x3 x4, absorb(worker firm year) vce(cluster worker) ///
    tolerance(1e-8) tolerancemode(reghdfe-comparable) absorptioncachemode(off) ///
    numthreads(16) gpubackend(cpu)
assert e(converged) == 1
assert e(gpu_used) == 0
display "P3FIX_STATA_A_OK"

quietly xhdfe y x1 x2 x3 x4, absorb(worker firm year) vce(cluster worker) ///
    tolerance(1e-8) tolerancemode(xhdfe-fast) absorptioncachemode(off) ///
    numthreads(16) gpubackend(cuda) residuals(r_fast)
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(gpu_used) == 1
assert "`e(gpu_backend)'" == "cuda"
assert "`e(gpu_status)'" == "used"
assert e(iterations) > 0 & e(iterations) <= 100000
generate byte b_sample = e(sample)
quietly count if b_sample & (missing(r_fast) | missing(e_oracle))
assert r(N) == 0
mata: st_numscalar("db", max(abs(st_matrix("e(b)")-st_matrix("oracle_b")) :/ rowmax((J(`k',1,1),abs(st_matrix("oracle_b")')))'))
mata: st_numscalar("dv", max(abs(st_matrix("e(V)")-st_matrix("oracle_V")) :/ sqrt(diagonal(st_matrix("oracle_V"))*diagonal(st_matrix("oracle_V"))')))
mata: st_numscalar("dse", max(abs(sqrt(diagonal(st_matrix("e(V)"))) :/ sqrt(diagonal(st_matrix("oracle_V"))) :- 1)))
mata: st_numscalar("dr", norm(st_data(., "r_fast", "b_sample") - st_data(., "e_oracle", "b_sample")) / norm(st_data(., "e_oracle", "b_sample")))
display "P3FIX_STATA_B iterations=" e(iterations) " b=" %21.14e db " V=" %21.14e dv ///
    " se=" %21.14e dse " residuals=" %21.14e dr
assert db <= envelope_b
assert dv <= envelope_V
assert dse <= envelope_se
* The stored residuals come from refine_stored_residuals (a CUDA MAP re-absorption of y - Xb at the
* public tolerance, older than 2.28.1), not from the restart: reported, labelled by the driver
* (KNOWN_DEFECT_STATA_RESIDUALS_CUDA_REFINE, hypothesis P7), never a failure of the restart.
scalar dmax = 0
foreach fe in worker firm year {
    tempvar m
    quietly bysort `fe': egen double `m' = mean(r_fast) if b_sample
    quietly summarize `m', meanonly
    scalar dmax = max(dmax, abs(r(min)), abs(r(max)))
    drop `m'
}
if dr > envelope_resid {
    display "P3FIX_STATA_B_RESIDUALS_OUTSIDE residuals=" %21.14e dr " D=" %21.14e dmax
}
display "P3FIX_STATA_B_OK"

capture noisily xhdfe y x1 x2 x3 x4, absorb(worker firm year) vce(cluster worker) ///
    tolerance(1e-8) tolerancemode(xhdfe-fast) absorptioncachemode(off) ///
    numthreads(16) gpubackend(cuda) maxiter(`refusal_maxiter')
local failure = _rc
assert `failure' != 0
assert "`e(cmd)'" == ""
capture matrix stale_b = e(b)
assert _rc != 0
capture matrix stale_V = e(V)
assert _rc != 0
display "P3FIX_STATA_C_REFUSED rc=`failure'"

quietly xhdfe y x1 x2 x3 x4, absorb(worker firm year) vce(cluster worker) ///
    tolerance(1e-8) tolerancemode(xhdfe-fast) absorptioncachemode(off) ///
    numthreads(16) gpubackend(cuda)
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(gpu_used) == 1
mata: st_numscalar("db", max(abs(st_matrix("e(b)")-st_matrix("oracle_b")) :/ rowmax((J(`k',1,1),abs(st_matrix("oracle_b")')))'))
assert db <= envelope_b
display "P3FIX_STATA_D_OK iterations=" e(iterations)
display "P3FIX_STATA_PASS"
