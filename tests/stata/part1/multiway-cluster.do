noi di as text "xhdfe certification: multiway clustered VCE"

xcert_require_reghdfe

set seed 12345
clear
set obs 2000
gen int firm = mod(_n - 1, 100) + 1
gen int year = mod(floor((_n - 1) / 100), 20) + 1
gen int cluster1 = mod(_n * 17, 53) + 1
gen int cluster2 = mod(_n * 31, 47) + 1
gen double x1 = rnormal()
gen double x2 = rnormal()
gen double y = 1 + .7 * x1 - .3 * x2 + firm / 40 + year / 9 + rnormal()

local scalars "N df_r df_m"

reghdfe y x1 x2, absorb(firm year) vce(cluster cluster1 cluster2) ///
    tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(ref) scalars("`scalars'")

xhdfe y x1 x2, absorb(firm year) vce(cluster cluster1 cluster2) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(xhd) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref) testprefix(xhd) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

exit
