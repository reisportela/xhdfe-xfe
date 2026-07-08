noi di as text "xhdfe certification: noconstant and slope-only absorbs"

xcert_require_reghdfe

clear
set obs 1200
gen int firm = mod(_n - 1, 60) + 1
gen int year = mod(floor((_n - 1) / 60), 10) + 1
gen double z = mod(_n * 17, 23) / 10
gen double x1 = sin(_n / 7)
gen double x2 = cos(_n / 11)
gen double y = 1 + .7 * x1 - .3 * x2 + firm / 20 + year / 9 + .02 * firm * z + sin(_n / 13) / 10

local scalars "N rmse rss tss mss r2 F df_r df_m"

reghdfe y x1 x2, absorb(firm year) noconstant tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(ref_nc) scalars("`scalars'")

xhdfe y x1 x2, absorb(firm year) noconstant tolerancemode(reghdfe-comparable) ///
    tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(xhd_nc) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref_nc) testprefix(xhd_nc) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)
if (e(report_constant) != 0 | colnumb(e(b), "_cons") < .) {
    di as error "noconstant should suppress reported _cons"
    exit 9
}
di as text "  scalar noconstant_report_constant: " e(report_constant)

reghdfe y x1 x2, absorb(firm#c.z) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(ref_slope) scalars("`scalars'")

xhdfe y x1 x2, absorb(firm#c.z) tolerancemode(reghdfe-comparable) ///
    tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(xhd_slope) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref_slope) testprefix(xhd_slope) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)
if (e(report_constant) != 0 | colnumb(e(b), "_cons") < .) {
    di as error "slope-only absorb should suppress reported _cons"
    exit 9
}
di as text "  scalar slope_only_report_constant: " e(report_constant)

exit
