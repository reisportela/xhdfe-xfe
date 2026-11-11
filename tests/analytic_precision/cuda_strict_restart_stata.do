* P3-fix (3'): the strict-residual CUDA recovery of a broken CG stage through the Stata plugin, in one
* session. Driven by cuda_strict_restart_contract.py --stata-package, which writes the cov240k fixture
* (with the P3 v3 oracle's within residuals e_oracle), the oracle b and V and the strict envelope at
* 1e-8, and runs Stata with XHDFE_TEST_CUDA_CG_BREAKDOWN=30:all (every CG stage of the session breaks
* down at its iteration 30) and the trace variables; the driver checks the session's
* cuda_strict_restart outcomes (lsmr-strict, lsmr-stopped, lsmr-strict) and that no cold MAP ran.
* A. strict-residual on CUDA: the native attempt breaks down, the device LSMR restart is approved:
*    converged, certified, on the GPU, 30 + 90 iterations, inside the strict envelope (b, V, SE and the
*    stored residuals against the oracle).
* B. Right after A, maxiter(31): the LSMR restart gets one iteration and stops; the fit is refused with
*    iterations=31/31 and no estimate of A may remain in e().
* C. The session recovers: A again.
version 18
clear all
set more off
set linesize 255
args package fixture oracle
adopath ++ "`package'"
discard
findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
use "`fixture'", clear
do "`oracle'"
local k = colsof(oracle_b)

foreach step in A C {
    capture drop r_strict
    capture drop s_sample
    quietly xhdfe y x1 x2 x3 x4, absorb(worker firm year) vce(cluster worker) ///
        tolerance(1e-8) tolerancemode(strict-residual) absorptioncachemode(off) ///
        numthreads(16) gpubackend(cuda) residuals(r_strict)
    assert e(converged) == 1
    assert e(precision_certified) == 1
    assert e(gpu_used) == 1
    assert "`e(gpu_backend)'" == "cuda"
    assert "`e(gpu_status)'" == "used"
    assert e(iterations) == 120
    generate byte s_sample = e(sample)
    quietly count if s_sample & (missing(r_strict) | missing(e_oracle))
    assert r(N) == 0
    mata: st_numscalar("db", max(abs(st_matrix("e(b)")-st_matrix("oracle_b")) :/ rowmax((J(`k',1,1),abs(st_matrix("oracle_b")')))'))
    mata: st_numscalar("dv", max(abs(st_matrix("e(V)")-st_matrix("oracle_V")) :/ sqrt(diagonal(st_matrix("oracle_V"))*diagonal(st_matrix("oracle_V"))')))
    mata: st_numscalar("dse", max(abs(sqrt(diagonal(st_matrix("e(V)"))) :/ sqrt(diagonal(st_matrix("oracle_V"))) :- 1)))
    mata: st_numscalar("dr", norm(st_data(., "r_strict", "s_sample") - st_data(., "e_oracle", "s_sample")) / norm(st_data(., "e_oracle", "s_sample")))
    display "P3S_STATA_`step' iterations=" e(iterations) " b=" %21.14e db " V=" %21.14e dv ///
        " se=" %21.14e dse " residuals=" %21.14e dr
    assert db <= envelope_b_scaled
    assert dv <= envelope_V_scaled
    assert dse <= envelope_se_rel
    assert dr <= envelope_resid_rel
    display "P3S_STATA_`step'_OK"
    if "`step'" == "A" {
        capture noisily xhdfe y x1 x2 x3 x4, absorb(worker firm year) vce(cluster worker) ///
            tolerance(1e-8) tolerancemode(strict-residual) absorptioncachemode(off) ///
            numthreads(16) gpubackend(cuda) maxiter(31)
        local failure = _rc
        assert `failure' != 0
        assert "`e(cmd)'" == ""
        capture matrix stale_b = e(b)
        assert _rc != 0
        capture matrix stale_V = e(V)
        assert _rc != 0
        display "P3S_STATA_B_REFUSED rc=`failure'"
    }
}
display "P3S_STATA_PASS"
