version 16.0
clear all
set more off

local ROOT "/home/mangelo/Documents/GitHub/xhdfe"
local ROOT_ENV : environment XHDFE_REPO_ROOT
if ("`ROOT_ENV'" != "") local ROOT "`ROOT_ENV'"
adopath ++ "`ROOT'/stata"
discard

set obs 600000
generate long row = _n - 1
generate int fe1 = mod(row, 1200)
generate int fe2 = mod(floor(row / 1200), 500)
generate double x1 = sin(row * .001) + mod(row * 17, 101) / 101
generate double x2 = cos(row * .0013) + mod(row * 29, 103) / 103
generate double y = .6*x1 - .25*x2 + .001*fe1 - .0005*fe2 + sin(row * .0021)

foreach threads in 1 2 8 16 24 48 {
    quietly xhdfe y x1 x2, absorb(fe1 fe2) vce(cluster fe1) keepsingletons ///
        tolerance(1e-10) tolerancemode(reghdfe-comparable) ///
        numthreads(`threads') gpubackend(cpu)
    assert e(converged) == 1
    assert e(threads_requested) == `threads'
    assert e(threads_effective) == `threads'
    assert e(threads_used) == `threads'
    assert e(gpu_used) == 0

    if (`threads' == 1) {
        matrix b_ref = e(b)
        matrix V_ref = e(V)
        scalar iterations_ref = e(iterations)
    }
    else {
        mata: st_numscalar("threads_b_diff", ///
            max(abs(st_matrix("e(b)") :- st_matrix("b_ref"))))
        mata: st_numscalar("threads_V_diff", ///
            max(abs(st_matrix("e(V)") :- st_matrix("V_ref"))))
        assert scalar(threads_b_diff) <= 1e-13
        assert scalar(threads_V_diff) <= 1e-13
        assert e(iterations) == scalar(iterations_ref)
    }
    display as result "XHDFE_THREADS_CONTRACT|cpu|requested=`threads'|used=" ///
        e(threads_used)
}

local TEST_CUDA : environment XHDFE_THREADS_TEST_CUDA
if ("`TEST_CUDA'" == "1") {
    foreach threads in 1 2 8 16 24 48 {
        quietly xhdfe y x1 x2, absorb(fe1 fe2) vce(cluster fe1) keepsingletons ///
            tolerance(1e-10) tolerancemode(reghdfe-comparable) ///
            numthreads(`threads') gpubackend(cuda)
        assert e(converged) == 1
        assert e(threads_requested) == `threads'
        assert e(threads_effective) == `threads'
        assert e(threads_used) == `threads'
        assert e(gpu_used) == 1
        assert "`e(gpu_backend)'" == "cuda"
        assert "`e(gpu_status)'" == "used"
        mata: st_numscalar("threads_b_diff", ///
            max(abs(st_matrix("e(b)") :- st_matrix("b_ref"))))
        mata: st_numscalar("threads_V_diff", ///
            max(abs(st_matrix("e(V)") :- st_matrix("V_ref"))))
        assert scalar(threads_b_diff) <= 1e-9
        assert scalar(threads_V_diff) <= 1e-9
        display as result "XHDFE_THREADS_CONTRACT|cuda|requested=`threads'|used=" ///
            e(threads_used)
    }
}

display as result "PASS: Stata explicit thread contract"
