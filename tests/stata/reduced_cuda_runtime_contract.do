version 16.0
clear all
set more off
args package mode
adopath ++ "`package'"
set seed 2026111502
set obs 49152
generate long worker = floor((_n-1)/24)+1
generate int firm = ceil(83*runiform())
generate byte year = ceil(13*runiform())
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double x3 = rnormal()
generate double y = .5*x1-.25*x2+.01*x3+sin(worker)+cos(firm)+year/10+rnormal()

if "`mode'" == "threads" {
    foreach team in 1 2 8 16 24 48 {
        quietly xhdfe y x1 x2 x3, absorb(worker firm year) vce(cluster worker) ///
            gpubackend(cuda) numthreads(`team') tolerancemode(reghdfe-comparable)
        assert e(converged)==1 & e(precision_certified)==1
        assert e(absorption_method_used)==8 & e(reduced_pcg_status)==1
        assert e(gpu_used)==1 & "`e(gpu_backend)'"=="cuda" & "`e(gpu_status)'"=="used"
        assert e(threads_used)==`team'
        if `team'>1 assert e(parallel_workers_active)>1
        display "REDUCED_CUDA_STATA_TEAM " `team' " used=" e(threads_used) " active=" e(parallel_workers_active)
    }
}
if "`mode'" == "fatal" {
    quietly xhdfe y x1 x2 x3, absorb(worker firm year) vce(cluster worker) ///
        gpubackend(cuda) numthreads(4) tolerancemode(reghdfe-comparable)
    assert e(absorption_method_used)==8 & e(reduced_pcg_status)==1
    matrix B0=e(b)
    matrix V0=e(V)
    capture noisily xhdfe y x1 x2 x3, absorb(worker firm year) vce(cluster worker) ///
        gpubackend(cuda) numthreads(4) tolerancemode(reghdfe-comparable)
    assert _rc!=0
    assert "`e(cmd)'"==""
    capture matrix STALE=e(b)
    assert _rc!=0
    quietly xhdfe y x1 x2 x3, absorb(worker firm year) vce(cluster worker) ///
        gpubackend(cuda) numthreads(4) tolerancemode(reghdfe-comparable)
    assert e(absorption_method_used)==8 & e(reduced_pcg_status)==1 & e(gpu_used)==1
    mata: assert(mreldif(st_matrix("B0"),st_matrix("e(b)"))==0)
    mata: assert(mreldif(st_matrix("V0"),st_matrix("e(V)"))==0)
}
display "REDUCED_CUDA_STATA_RUNTIME_PASS `mode'"
