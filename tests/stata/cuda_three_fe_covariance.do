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
foreach threads in 1 2 8 16 24 48 {
    foreach variant in original balanced reordered {
        preserve
        local absorb worker firm year
        generate double ys = y
        generate double z1 = x1
        generate double z2 = x2
        generate double z3 = x3
        generate double z4 = x4
        matrix undo = I(5)
        if "`variant'" == "balanced" {
            replace ys = y*.5
            replace z1 = x1*2^(-11)
            replace z2 = x2*2^(-4)
            replace z3 = x3*2^(-8)
            replace z4 = x4*2^(-3)
            matrix undo[1,1] = 2^(-10)
            matrix undo[2,2] = 2^(-3)
            matrix undo[3,3] = 2^(-7)
            matrix undo[4,4] = 2^(-2)
            matrix undo[5,5] = 2
        }
        if "`variant'" == "reordered" {
            sort permutation
            local absorb year worker firm
            replace ys = y*8
            replace z1 = x1*.25
            replace z2 = x2*16
            replace z3 = x3*2^(-6)
            replace z4 = x4*32
            matrix undo[1,1] = 2^(-5)
            matrix undo[2,2] = 2
            matrix undo[3,3] = 2^(-9)
            matrix undo[4,4] = 4
            matrix undo[5,5] = .125
        }
        quietly xhdfe ys z1 z2 z3 z4, absorb(`absorb') vce(cluster worker) ///
            tolerance(1e-8) tolerancemode(reghdfe-comparable) ///
            absorptionmethod(auto) absorptioncachemode(off) ///
            numthreads(`threads') gpubackend(`backend')
        assert e(N) == 240000
        assert e(df_a) == 118
        assert e(df_r) == 5999
        assert e(converged) == 1
        assert e(threads_used) == `threads'
        if "`backend'" == "cuda" {
            assert e(gpu_used) == 1
            assert "`e(gpu_backend)'" == "cuda"
            assert "`e(gpu_status)'" == "used"
        }
        matrix actual_b = e(b)*undo
        matrix actual_V = undo*e(V)*undo
        mata: st_numscalar("db", max(abs(st_matrix("actual_b")-st_matrix("oracle_b")) :/ rowmax((J(5,1,1),abs(st_matrix("oracle_b")')))'))
        mata: st_numscalar("dv", max(abs(st_matrix("actual_V")-st_matrix("oracle_V")) :/ sqrt(diagonal(st_matrix("oracle_V"))*diagonal(st_matrix("oracle_V"))')))
        assert db <= 1e-9
        assert dv <= 1e-8
        display "THREE_FE `backend' `threads' `variant' b=" %21.14e db " V=" %21.14e dv
        restore
    }
}
display "THREE_FE_COVARIANCE_STATA_PASS"
