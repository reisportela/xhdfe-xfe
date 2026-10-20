* A failed GPU primary and its repair share the requested iteration budget.
version 18
clear all
set more off
args package backend fixture oracle
adopath ++ "`package'"
discard
findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
use "`fixture'", clear
do "`oracle'"

quietly xhdfe y x1 x2 x3 x4, absorb(worker firm year) vce(cluster worker) ///
    tolerance(1e-8) tolerancemode(reghdfe-comparable) maxiter(1000) ///
    absorptionmethod(auto) absorptioncachemode(off) numthreads(1) gpubackend(cuda)
assert e(converged) == 1
assert e(gpu_used) == 1
assert "`e(gpu_backend)'" == "cuda"
assert "`e(gpu_status)'" == "used"
assert e(iterations) <= 1000
matrix actual_b = e(b)
matrix actual_V = e(V)
mata: st_numscalar("db", max(abs(st_matrix("actual_b")-st_matrix("oracle_b")) :/ rowmax((J(5,1,1),abs(st_matrix("oracle_b")')))'))
mata: st_numscalar("dv", max(abs(st_matrix("actual_V")-st_matrix("oracle_V")) :/ sqrt(diagonal(st_matrix("oracle_V"))*diagonal(st_matrix("oracle_V"))')))
assert db <= 1e-9
assert dv <= 1e-8
display "CUDA_WARM_BUDGET_NUMERICAL_PASS"

capture noisily xhdfe y x1 x2 x3 x4, absorb(worker firm year) vce(cluster worker) ///
    tolerance(1e-8) tolerancemode(reghdfe-comparable) maxiter(150) ///
    absorptionmethod(auto) absorptioncachemode(off) numthreads(1) gpubackend(cuda)
local failure = _rc
assert `failure' != 0
assert "`e(cmd)'" == ""
capture matrix stale_b = e(b)
assert _rc != 0
display "CUDA_WARM_BUDGET_EXPECTED_REFUSAL"
display "CUDA_WARM_BUDGET_PASS"
