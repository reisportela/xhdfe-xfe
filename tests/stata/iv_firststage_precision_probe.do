* Targeted numerical probe for the probability-weighted IV first stage.
* Runs CPU and actual CUDA at 1e-8, 1e-12 and 1e-14. Requires ivreghdfe/reghdfe.
version 16
clear all
set more off
set linesize 255
local adopath : env XHDFE_STATA_ADOPATH
if (`"`adopath'"' != "") adopath ++ `"`adopath'"'
di "Stata version: " c(stata_version) "; OS: " c(os) "; MP: " c(MP)
which xhdfe
which ivreghdfe
which reghdfe

set seed 71903
set obs 6000
generate long firm = 1 + mod(_n,120)
generate long year = 1 + mod(floor((_n-1)/120),10)
generate long state = 1 + mod(_n,29)
generate long county = 1 + mod(floor((_n-1)/29),37)
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double z1 = rnormal()
generate double z2 = rnormal()
generate double z3 = rnormal()
generate double v = rnormal()
generate double d1 = .6*z1 + .3*z2 + .2*x1 + firm/120 + v
generate double d2 = .5*z2 + .7*z3 - .2*x2 + year/20 + v + rnormal()
generate double weak = .01*z1 + .02*z2 + .2*x1 + firm/120 + v
generate double y = 1.5*d1 - .8*d2 + .5*x1 - .2*x2 + firm/30 + year/10 + v + rnormal()
generate double aw = .5 + runiform()
generate byte fw = 1 + mod(_n,3)
generate double zdup = 2*z1
generate double zzero = 0

* All stochastic draws above match iv_firststage_contract.do, including those
* for variables not used here, so this reproduces its worst-case pweight row.
quietly regress d1 x1 x2 z1 z2 i.firm i.year [pw=aw], vce(cluster state)
assert e(N) == 6000
assert e(N_clust) == 29
matrix dense_fullb = e(b)
matrix dense_fullV = e(V)
matrix dense_b = dense_fullb[1,1..4]
matrix dense_V = dense_fullV[1..4,1..4]
quietly test z1 z2
scalar dense_F = r(F)
scalar dense_df = r(df)
scalar dense_df_r = r(df_r)
di "EXPLICIT_DUMMY_F " %24.17g dense_F

tempname summary matrices
postfile `summary' str10 engine double(tolerance F F_ivreghdfe F_dense F_diagnostic ///
    abs_F_ref abs_F_ivref abs_F_dense abs_F_diagnostic max_abs_b_ref max_abs_V_ref ///
    max_abs_b_dense max_abs_V_dense b_mreldif_ref V_mreldif_ref N df df_r ///
    df_a N_clust iterations method_used abs_residual_rel precision_certified gpu_used) ///
    using "precision_summary.dta", replace
postfile `matrices' str10 engine double tolerance str1 kind str4 row str4 col ///
    double value using "precision_matrices.dta", replace
local vars x1 x2 z1 z2
forvalues i=1/4 {
    local vi : word `i' of `vars'
    post `matrices' ("dense") (.) ("b") ("b") ("`vi'") (dense_b[1,`i'])
    forvalues j=1/4 {
        local vj : word `j' of `vars'
        post `matrices' ("dense") (.) ("V") ("`vi'") ("`vj'") (dense_V[`i',`j'])
    }
}
foreach tol in 1e-8 1e-12 1e-14 {
    quietly ivreghdfe y x1 x2 (d1=z1 z2) [pw=aw], absorb(firm year) ///
        cluster(state) small ffirst tolerance(`tol')
    matrix iv_first = e(first)
    assert e(N) == 6000
    assert e(sample) == 1
    scalar iv_F = iv_first["F","d1"]
    foreach engine in reghdfe cpu cuda {
        scalar diagnostic_F = .
        if ("`engine'" == "reghdfe") {
            quietly reghdfe d1 x1 x2 z1 z2 [pw=aw], absorb(firm year) ///
                vce(cluster state) tolerance(`tol')
        }
        else {
            quietly xhdfe y x1 x2 d1 [pw=aw], absorb(firm year) ///
                endogenous(d1) instruments(z1 z2) vce(cluster state) ///
                gpubackend(`engine') tolerance(`tol') ffirst
            assert e(N) == 6000
            assert e(sample) == 1
            assert e(gpu_used) == ("`engine'" == "cuda")
            if ("`engine'" == "cuda") {
                assert "`e(gpu_backend)'" == "cuda"
                assert "`e(gpu_status)'" == "used"
            }
            matrix diagnostic = e(first)
            matrix diag_gpu = e(first_gpu_used)
            assert diag_gpu[1,1] == ("`engine'" == "cuda")
            assert diag_gpu[2,1] == ("`engine'" == "cuda")
            scalar diagnostic_F = diagnostic["F","d1"]
            assert diagnostic["df","d1"] == dense_df
            assert diagnostic["df_r","d1"] == dense_df_r
            quietly xhdfe d1 x1 x2 z1 z2 [pw=aw], absorb(firm year) ///
                vce(cluster state) gpubackend(`engine') tolerance(`tol')
            assert e(gpu_used) == ("`engine'" == "cuda")
            assert e(converged) == 1
            assert e(precision_certified) == 1
        }
        assert e(N) == 6000
        assert e(N_clust) == 29
        assert e(sample) == 1
        matrix fullb = e(b)
        matrix fullV = e(V)
        matrix b = fullb[1,1..4]
        matrix V = fullV[1..4,1..4]
        local names : colnames b
        assert "`names'" == "x1 x2 z1 z2"
        scalar iterations = e(iterations)
        if ("`engine'" == "reghdfe") scalar iterations = e(ic)
        scalar method_used = e(absorption_method_used)
        scalar abs_residual_rel = e(abs_residual_rel)
        scalar precision_certified = e(precision_certified)
        scalar gpu_used = e(gpu_used)
        quietly test z1 z2
        scalar F = r(F)
        scalar df = r(df)
        scalar df_r = r(df_r)
        assert df == dense_df
        assert df_r == dense_df_r
        assert df == iv_first["df","d1"]
        assert df_r == iv_first["df_r","d1"]
        if ("`engine'" == "reghdfe") {
            matrix refb = b
            matrix refV = V
            scalar refF = F
            scalar ref_df_a = e(df_a)
        }
        assert e(df_a) == ref_df_a
        mata: st_numscalar("max_abs_b_ref", max(abs(st_matrix("b") - st_matrix("refb"))))
        mata: st_numscalar("max_abs_V_ref", max(abs(st_matrix("V") - st_matrix("refV"))))
        mata: st_numscalar("max_abs_b_dense", max(abs(st_matrix("b") - st_matrix("dense_b"))))
        mata: st_numscalar("max_abs_V_dense", max(abs(st_matrix("V") - st_matrix("dense_V"))))
        scalar abs_F_ref = abs(F-refF)
        scalar abs_F_ivref = abs(F-iv_F)
        scalar abs_F_dense = abs(F-dense_F)
        scalar abs_F_diagnostic = abs(F-diagnostic_F)
        assert abs_F_ref < 1e-7*max(1,abs(refF))
        assert abs_F_ivref < 1e-7*max(1,abs(iv_F))
        assert abs_F_dense < 1e-7*max(1,abs(dense_F))
        if ("`engine'" != "reghdfe") assert abs_F_diagnostic < 1e-12*max(1,abs(F))
        post `summary' ("`engine'") (`tol') (F) (iv_F) (dense_F) (diagnostic_F) ///
            (abs_F_ref) (abs_F_ivref) (abs_F_dense) (abs_F_diagnostic) ///
            (max_abs_b_ref) (max_abs_V_ref) (max_abs_b_dense) (max_abs_V_dense) ///
            (mreldif(b,refb)) (mreldif(V,refV)) (e(N)) (df) (df_r) ///
            (e(df_a)) (e(N_clust)) (iterations) (method_used) (abs_residual_rel) ///
            (precision_certified) (gpu_used)
        forvalues i=1/4 {
            local vi : word `i' of `vars'
            post `matrices' ("`engine'") (`tol') ("b") ("b") ("`vi'") (b[1,`i'])
            forvalues j=1/4 {
                local vj : word `j' of `vars'
                post `matrices' ("`engine'") (`tol') ("V") ("`vi'") ("`vj'") (V[`i',`j'])
            }
        }
        di "PRECISION `engine' tol=`tol' F=" %24.17g F ///
            " |F-ref|=" %12.5e abs_F_ref " |F-dense|=" %12.5e abs_F_dense ///
            " max|b-ref|=" %12.5e max_abs_b_ref " max|V-ref|=" %12.5e max_abs_V_ref
    }
}
postclose `summary'
postclose `matrices'
use "precision_summary.dta", clear
export delimited using "precision_summary.csv", replace
use "precision_matrices.dta", clear
export delimited using "precision_matrices.csv", replace
di as result "IV_FIRSTSTAGE_PRECISION_PASS"
