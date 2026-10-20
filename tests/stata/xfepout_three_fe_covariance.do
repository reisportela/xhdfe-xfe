version 18
clear all
set more off
set linesize 255
args package backend fixture oracle
adopath ++ "`package'"
discard
findfile xfepout.ado
assert "`r(fn)'" == "`package'/xfepout.ado"
use "`fixture'", clear
do "`oracle'"

mata:
void xfepout_three_fe_inference()
{
    real matrix X, Xs, bread, scores, info, sums, T, V
    real colvector y, b, u, w, full_b
    real rowvector scales, mx
    real scalar n, g, j, my
    y = st_data(., "po_ys")
    X = st_data(., "po_z1 po_z2 po_z3 po_z4")
    w = st_data(., "worker")
    n = rows(y)
    scales = sqrt(colsum(X:^2))
    Xs = X :/ scales
    b = qrsolve(Xs, y) :/ scales'
    bread = diag(1:/scales) * invsym(quadcross(Xs, Xs)) * diag(1:/scales)
    u = y-X*b
    scores = (X,J(n,1,1))
    for (j=1; j<=5; j++) scores[.,j] = scores[.,j] :* u
    info = panelsetup(w,1)
    sums = panelsum(scores,info)
    g = rows(sums)
    mx = mean(st_data(., "z1 z2 z3 z4"))
    my = mean(st_data(., "ys"))
    full_b = (b \ my-mx*b)
    T = J(5,5,0)
    T[|1,1 \ 4,4|] = bread
    T[5,1..4] = -mx*bread
    T[5,5] = 1/n
    // The synthetic oracle has 118 non-nested absorbed DoF and 6000 clusters.
    assert(n==240000 & g==6000)
    V = T*quadcross(sums,sums)*T' * (g/(g-1)) * ((n-1)/(n-4-118-1))
    st_matrix("within_b",full_b')
    st_matrix("within_V",V)
}
end

foreach threads in 1 16 {
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
        capture noisily xfepout ys z1 z2 z3 z4, absorb(`absorb') generate(po_) ///
            tolerance(1e-8) ///
            absorptionmethod(auto) absorptioncachemode(off) ///
            numthreads(`threads') gpubackend(`backend')
        local result = _rc
        if `result' {
            // These exact three CUDA rows also refuse in the frozen release.
            // The driver additionally checks the necessary-check diagnostic.
            assert "`backend'" == "cuda" & `threads' == 16
            assert `result' == 198
            assert "`e(cmd)'" == ""
            assert missing(e(N))
            capture matrix stale = e(b)
            assert _rc != 0
            display "XFEPOUT_THREE_FE_EXPECTED_REFUSAL `backend' `threads' `variant'"
            restore
            continue
        }
        assert e(N) == 240000
        assert e(converged) == 1
        assert e(threads_used) == `threads'
        assert e(gpu_used) == ("`backend'" == "cuda")
        if "`backend'" == "cuda" {
            assert "`e(gpu_backend)'" == "cuda"
            assert "`e(gpu_status)'" == "used"
        }
        sort worker
        mata: xfepout_three_fe_inference()
        matrix actual_b = within_b*undo
        matrix actual_V = undo*within_V*undo
        mata: st_numscalar("db", max(abs(st_matrix("actual_b")-st_matrix("oracle_b")) :/ rowmax((J(5,1,1),abs(st_matrix("oracle_b")')))'))
        mata: st_numscalar("dv", max(abs(st_matrix("actual_V")-st_matrix("oracle_V")) :/ sqrt(diagonal(st_matrix("oracle_V"))*diagonal(st_matrix("oracle_V"))')))
        assert db <= 1e-9
        assert dv <= 1e-8
        display "XFEPOUT_THREE_FE `backend' `threads' `variant' b=" %21.14e db " V=" %21.14e dv
        restore
    }
}
display "XFEPOUT_THREE_FE_CONTRACT_COMPLETE"
