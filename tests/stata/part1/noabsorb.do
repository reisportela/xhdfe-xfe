noi di as text "xhdfe certification: no absorb, OLS baseline"

sysuse auto, clear

local scalars "N rmse rss mss r2 r2_a F df_r df_m ll ll_0"

regress price weight length
xcert_store_estimates, prefix(ref) scalars("`scalars'")

xhdfe price weight length, tolerancemode(reghdfe-comparable) noheader notable nofootnote
xcert_store_estimates, prefix(xhd) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref) testprefix(xhd) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

exit
