* Uses the synthetic cov240k.dta from tests/cuda_weighted_cpu_route_contract.py.
version 16
clear all
set more off
set linesize 255
args package fixture
adopath ++ "`package'"
quietly findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
quietly findfile xhdfe.plugin
assert "`r(fn)'" == "`package'/xhdfe.plugin"
which reghdfe

quietly use "`fixture'", clear
xhdfe y x1 x2 x3 x4 [aw=w_aw], absorb(fe1 fe2 fe3) vce(cluster fe1) ///
    gpubackend(cuda) numthreads(8) tolerancemode(strict-residual) ///
    ferecoverymethod(map) savefes(saved_) residuals(checked_residual)
assert e(gpu_used)==1 & e(gpu_status_code)==1 & e(precision_certified)==1
assert "`e(gpu_status)'" == "used" & e(iterations) <= 100000
matrix estimate_b = e(b)
matrix estimate_V = e(V)
generate byte estimate_sample = e(sample)
egen double summed_effects = rowtotal(saved_*)
assert !missing(checked_residual, summed_effects)
generate double reconstruction = abs(y-(_b[_cons]+_b[x1]*x1+_b[x2]*x2+_b[x3]*x3+_b[x4]*x4 ///
    +summed_effects+checked_residual))
quietly summarize reconstruction, meanonly
display "XHDFE_WEIGHTED_CUDA_RECONSTRUCTION " %12.5e r(max)
assert r(max) <= 7.3e-5
foreach effect in fe1 fe2 fe3 {
    tempvar score mass
    bysort `effect': egen double `score' = total(w_aw*checked_residual)
    bysort `effect': egen double `mass' = total(w_aw)
    assert abs(`score'/`mass') <= 1e-8
}
quietly reghdfe y x1 x2 x3 x4 [aw=w_aw], absorb(fe1 fe2 fe3) vce(cluster fe1) ///
    tolerance(1e-12) technique(lsmr)
assert e(sample) == estimate_sample
mata:
b_reference = st_matrix("e(b)")
V_reference = st_matrix("e(V)")
sd = sqrt(diagonal(V_reference))
st_numscalar("b_diff", max(abs(st_matrix("estimate_b")-b_reference):/colmax((J(1,cols(b_reference),1) \ abs(b_reference)))))
st_numscalar("V_diff", max(abs(st_matrix("estimate_V")-V_reference):/(sd*sd')))
end
display "XHDFE_WEIGHTED_CUDA_DIFF b=" %12.5e b_diff " V=" %12.5e V_diff
assert b_diff <= 1e-8 & V_diff <= 1e-8
display "XHDFE_WEIGHTED_CUDA_PASS"
