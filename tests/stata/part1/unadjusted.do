noi di as text "xhdfe certification: one-way absorb, unadjusted VCE"

sysuse auto, clear
bys turn: gen t = _n
tsset turn t
drop if missing(rep78)

local lhs price
local rhs weight length
local absvars turn
local scalars "N rmse tss rss mss r2 r2_a F df_r df_m ll ll_0"

areg `lhs' `rhs', absorb(`absvars')
local ref_df_a = e(df_a)
xcert_store_estimates, prefix(ref) scalars("`scalars'")

xhdfe `lhs' `rhs', absorb(`absvars') keepsingletons tolerancemode(reghdfe-comparable) ///
    tolerance(1e-10) vce(ols) noheader notable nofootnote
scalar xhd_df_a_minus_constant = e(df_a) - 1
xcert_store_estimates, prefix(xhd) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref) testprefix(xhd) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

scalar ref_df_a_minus_constant = `ref_df_a'
xcert_assert_scalars_close, leftprefix(ref) rightprefix(xhd) ///
    scalars("df_a_minus_constant") tol(1e-12)

exit
