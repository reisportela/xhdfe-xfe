noi di as text "xhdfe certification: string IDs and interaction IDs"

clear
set obs 1200
gen int firm = mod(_n - 1, 60) + 1
gen int year = mod(floor((_n - 1) / 60), 10) + 1
gen str4 sfirm = "f" + string(firm)
gen str4 syear = "y" + string(year)
gen double x1 = sin(_n / 7)
gen double x2 = cos(_n / 11)
gen double y = 1 + .7 * x1 - .3 * x2 + firm / 20 + year / 9 + sin(_n / 13) / 10
egen long firm_year = group(firm year)

local scalars "N df_r df_m"

xhdfe y x1 x2, absorb(sfirm syear) cluster(sfirm) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(strid) scalars("`scalars'")

xhdfe y x1 x2, absorb(firm year) cluster(firm) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(numid) scalars("`scalars'")

xcert_compare_estimates, refprefix(numid) testprefix(strid) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

xhdfe y x1 x2, absorb(firm#year) cluster(firm#year) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(inter) scalars("`scalars'")
if (`"`e(absvars)'"' != "firm#year" | `"`e(clustvar)'"' != "firm#year") {
    di as error "interaction absorb/cluster metadata mismatch"
    exit 9
}

xhdfe y x1 x2, absorb(firm_year) cluster(firm_year) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(grouped) scalars("`scalars'")

xcert_compare_estimates, refprefix(grouped) testprefix(inter) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

exit
