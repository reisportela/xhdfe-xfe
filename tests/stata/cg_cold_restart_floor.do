* CG-03: a cold CUDA restart must honour the existing Comparable floor.
* Public auto data; independent native dummy OLS on the identical sample.
version 16
clear all
set more off
set seed 123456789
set sortseed 123456789
local package : env XHDFE_STATA_ADOPATH
adopath ++ "`package'"
discard
sysuse auto, clear
bysort turn: generate t = _n
tsset turn t
drop if missing(rep78)

mata:
void core_cold_floor_compare()
{
    real matrix b, bref, V, Vref, bscale, vscale
    b = st_matrix("candidate_b")[1, 1..2]
    bref = st_matrix("oracle_b")[1, 1..2]
    V = st_matrix("candidate_V")[1..2, 1..2]
    Vref = st_matrix("oracle_V")[1..2, 1..2]
    bscale = rowmax((J(2,1,1), abs(bref')))'
    st_numscalar("b_error", max(abs(b-bref):/bscale))
    vscale = sqrt(diagonal(Vref)*diagonal(Vref)')
    st_numscalar("V_error", max(abs(V-Vref):/vscale))
}
end

foreach tolerance in 1e-5 1e-10 {
foreach backend in cpu cuda {
    xhdfe price weight gear, absorb(turn trunk) tolerance(`tolerance') ///
        tolerancemode(reghdfe-comparable) gpubackend(`backend') numthreads(2)
    assert e(converged) == 1
    assert e(iterations) <= 100000
    if ("`backend'" == "cuda") {
        assert e(gpu_used) == 1
        assert "`e(gpu_backend)'" == "cuda"
        assert "`e(gpu_status)'" == "used"
    }
    matrix candidate_b = e(b)
    matrix candidate_V = e(V)
    scalar candidate_df = e(df_r)
    generate byte candidate_sample = e(sample)
    regress price weight gear i.turn i.trunk if candidate_sample
    assert e(df_r) == candidate_df
    matrix oracle_b = e(b)
    matrix oracle_V = e(V)
    mata: core_cold_floor_compare()
    assert b_error <= 1e-9
    assert V_error <= 1e-8
    display "CG_COLD_FLOOR backend=`backend' tolerance=`tolerance' b_error=" %21.14e b_error " V_error=" %21.14e V_error
    drop candidate_sample
}
}
display "CG_COLD_FLOOR_PASS"
