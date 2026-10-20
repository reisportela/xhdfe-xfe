* CUDA04: public synthetic R-audit design; independent sparse-dummy OLS.
* Reference construction: tests/analytic_precision/cuda_ordinary_weak_graph_contract.py.
* Constants are MP80/long-double OLS results, not reghdfe/xhdfe estimates.
version 16
clear all
set more off
args backend
if "`backend'" == "" local backend cpu
local package : env XHDFE_STATA_ADOPATH
adopath ++ "`package'"
discard
set obs 200000
generate long f1 = mod(_n-1, 20000)
generate long f2 = mod((_n-1)*37, 17003)
generate double x1 = sin((_n-1)*.001)
generate double x2 = cos((_n-1)*.0013)
generate double y = .6*x1 - .25*x2 + .001*f1 - .0005*f2 + sin((_n-1)*.0021)
matrix oracle_b = (.6022947169901182, -.23636327682008004, 5.751628685867754)
matrix oracle_V = (6.215599933713912e-6, 1.8486498877792318e-7, -1.6437346206169316e-8 \ 1.8486498877792318e-7, 6.454975482370246e-6, -1.746500244371731e-8 \ -1.6437346206169316e-8, -1.746500244371731e-8, 3.051177166467282e-6)

foreach criterion in auto reghdfe {
    xhdfe y x1 x2, absorb(f1 f2) keepsingletons tolerance(1e-8) ///
        tolerancemode(reghdfe-comparable) convergence(`criterion') ///
        gpubackend(`backend') numthreads(16)
    assert e(N) == 200000
    assert e(df_r) == 162996
    assert e(converged) == 1
    assert e(iterations) <= 100000
    if "`backend'" == "cuda" {
        assert e(gpu_used) == 1
        assert "`e(gpu_backend)'" == "cuda"
        assert "`e(gpu_status)'" == "used"
    }
    mata: st_numscalar("b_error", max(abs(st_matrix("e(b)")-st_matrix("oracle_b")) :/ rowmax((J(3,1,1),abs(st_matrix("oracle_b")')))'))
    mata: st_numscalar("V_error", max(abs(st_matrix("e(V)")-st_matrix("oracle_V")) :/ sqrt(diagonal(st_matrix("oracle_V"))*diagonal(st_matrix("oracle_V"))')))
    assert b_error <= 1e-9
    assert V_error <= 1e-8
    display "CUDA04 backend=`backend' criterion=`criterion' b_error=" %21.14e b_error " V_error=" %21.14e V_error
}
display "CUDA_ORDINARY_WEAK_GRAPH_PASS `backend'"
