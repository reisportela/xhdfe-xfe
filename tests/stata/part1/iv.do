noi di as text "xhdfe certification: IV/2SLS with absorbed fixed effects"

xcert_require_ivreghdfe

set seed 24680
clear
set obs 1000
gen int firm = mod(_n - 1, 80) + 1
gen int year = mod(floor((_n - 1) / 80), 13) + 1
gen double x1 = rnormal()
gen double z = rnormal()
gen double v = rnormal()
gen double endo = .5 * z + .3 * x1 + firm / 200 + v
gen double y = 1 + .7 * x1 + 1.8 * endo + firm / 40 + year / 10 + v + rnormal()

local scalars "N df_r df_m"

ivreghdfe y x1 (endo = z), absorb(firm year) tol(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(ref) scalars("`scalars'")
xcert_subset_estimates, inprefix(ref) outprefix(ref_common) cols("x1 endo") scalars("`scalars'")

xhdfe y x1 endo, absorb(firm year) endogenous(endo) instruments(z) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(xhd) scalars("`scalars'")
xcert_subset_estimates, inprefix(xhd) outprefix(xhd_common) cols("x1 endo") scalars("`scalars'")

xcert_compare_estimates, refprefix(ref_common) testprefix(xhd_common) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

exit
