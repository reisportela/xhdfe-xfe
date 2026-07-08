noi di as text "xhdfe certification: heterogeneous slopes"

xcert_require_reghdfe

clear
set obs 600
gen int firm = mod(_n - 1, 60) + 1
gen int year = mod(floor((_n - 1) / 60), 10) + 1
gen double z = mod(_n * 17, 23) / 10
gen double x1 = sin(_n / 7)
gen double x2 = cos(_n / 11)
gen double y = 1 + .7 * x1 - .3 * x2 + firm / 20 + year / 9 + ///
    .02 * firm * z + sin(_n / 13) / 10

local scalars "N rmse rss tss mss r2 r2_a F df_r df_m"

reghdfe y x1 x2, absorb(firm##c.z year) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(ref) scalars("`scalars'")

xhdfe y x1 x2, absorb(firm##c.z year) tolerancemode(reghdfe-comparable) ///
    tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(xhd) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref) testprefix(xhd) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

exit
