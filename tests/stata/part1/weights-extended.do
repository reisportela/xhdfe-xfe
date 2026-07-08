noi di as text "xhdfe certification: pweights, iweights, and weighted predictions"

sysuse auto, clear
drop if missing(rep78)
gen double pw = (mod(_n, 5) + 1) / 3
gen double iw = (mod(_n, 7) + 1) / 4

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

regress price weight length i.rep78 [iw=iw]
xcert_store_estimates, prefix(ref_iw) scalars("N df_r")
xcert_subset_estimates, inprefix(ref_iw) outprefix(ref_iw_common) cols("weight length") scalars("N df_r")

xhdfe price weight length [iw=iw], absorb(rep78) keepsingletons ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote
xcert_store_estimates, prefix(xhd_iw) scalars("N df_r")
xcert_subset_estimates, inprefix(xhd_iw) outprefix(xhd_iw_common) cols("weight length") scalars("N df_r")

xcert_assert_matrix_close ref_iw_common_b xhd_iw_common_b, tol(1e-8) name("iweight e(b)")
xcert_assert_scalars_close, leftprefix(ref_iw_common) rightprefix(xhd_iw_common) scalars("N df_r") tol(1e-8)
di as text "  matrix iweight e(V): not compared; areg does not allow iweights and full-dummy regress uses a different VCE convention"

xhdfe price weight length [pw=pw], absorb(rep78) keepsingletons residuals(u) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote
predict double xbd_pw, xbd
predict double score_pw, score
gen double y_minus_u = price - u
xcert_assert_var_close xbd_pw y_minus_u if e(sample), tol(1e-10) name("pweight predict xbd")
xcert_assert_var_close score_pw u if e(sample), tol(1e-10) name("pweight predict score")

exit
