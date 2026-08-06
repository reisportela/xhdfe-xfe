noi di as text "xhdfe certification: pweights, iweights, and weighted predictions"

sysuse auto, clear
drop if missing(rep78)
gen double pw = (mod(_n, 5) + 1) / 3
gen double iw = 1 + mod(_n, 3)

local scalars "N rmse tss rss mss r2 r2_a F df_r df_m"

areg price weight length [pw=pw], absorb(rep78) robust
xcert_store_estimates, prefix(ref_pw) scalars("`scalars'")

xhdfe price weight length [pw=pw], absorb(rep78) keepsingletons ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote
xcert_store_estimates, prefix(xhd_pw) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref_pw) testprefix(xhd_pw) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)
if (`"`e(vce)'"' != "robust") {
    di as error "pweight without explicit vce() should use robust inference"
    exit 9
}

local iw_scalars "N rmse tss rss mss r2 r2_a df_r"
regress price weight length i.rep78 [iw=iw]
xcert_store_estimates, prefix(ref_iw) scalars("`iw_scalars'")
scalar ref_iw_tss = ref_iw_rss + ref_iw_mss
xcert_subset_estimates, inprefix(ref_iw) outprefix(ref_iw_common) ///
    cols("weight length") scalars("`iw_scalars'")

xhdfe price weight length [iw=iw], absorb(rep78) keepsingletons ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote
xcert_store_estimates, prefix(xhd_iw) scalars("`iw_scalars'")
xcert_subset_estimates, inprefix(xhd_iw) outprefix(xhd_iw_common) ///
    cols("weight length") scalars("`iw_scalars'")

xcert_assert_matrix_close ref_iw_common_b xhd_iw_common_b, tol(1e-8) name("iweight e(b)")
xcert_assert_matrix_close ref_iw_common_V xhd_iw_common_V, tol(1e-8) name("iweight e(V)")
xcert_assert_scalars_close, leftprefix(ref_iw_common) rightprefix(xhd_iw_common) ///
    scalars("`iw_scalars'") tol(1e-8)

xhdfe price weight length [pw=pw], absorb(rep78) keepsingletons residuals(u) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote
predict double xbd_pw, xbd
predict double score_pw, score
gen double y_minus_u = price - u
xcert_assert_var_close xbd_pw y_minus_u if e(sample), tol(1e-10) name("pweight predict xbd")
xcert_assert_var_close score_pw u if e(sample), tol(1e-10) name("pweight predict score")

exit
