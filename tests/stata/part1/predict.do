noi di as text "xhdfe certification: postestimation predict"

sysuse auto, clear
bys turn: gen t = _n
tsset turn t

local lhs price
local rhs weight length gear_ratio displacement
local absvars turn
local scalars "N rmse tss rss mss r2 r2_a F df_r df_m ll ll_0"

areg `lhs' `rhs', absorb(`absvars')
xcert_store_estimates, prefix(ref) scalars("`scalars'")
predict double ref_xb, xb
predict double ref_d, d
predict double ref_xbd, xbd
predict double ref_resid, resid
predict double ref_dr, dr
gen byte ref_sample = e(sample)

xhdfe `lhs' `rhs', absorb(`absvars') keepsingletons residuals(xhd_resid0) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote
xcert_store_estimates, prefix(xhd) scalars("`scalars'")
predict double xhd_xb, xb
predict double xhd_d, d
predict double xhd_xbd, xbd
predict double xhd_resid, residuals
predict double xhd_dr, dresiduals
gen byte xhd_sample = e(sample)

xcert_compare_estimates, refprefix(ref) testprefix(xhd) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

xcert_assert_var_close ref_xb xhd_xb if ref_sample & xhd_sample, tol(1e-10) name("predict xb")
xcert_assert_var_close ref_d xhd_d if ref_sample & xhd_sample, tol(1e-10) name("predict d")
xcert_assert_var_close ref_xbd xhd_xbd if ref_sample & xhd_sample, tol(1e-10) name("predict xbd")
xcert_assert_var_close ref_resid xhd_resid if ref_sample & xhd_sample, tol(1e-10) name("predict residuals")
xcert_assert_var_close ref_dr xhd_dr if ref_sample & xhd_sample, tol(1e-10) name("predict dresiduals")

drop xhd_resid0
capture noisily predict double should_fail, residuals
local rc = _rc
if (`rc' != 111) {
    di as error "expected predict residuals to fail with rc=111 after dropping e(resid); got rc=`rc'"
    exit 9
}

exit
