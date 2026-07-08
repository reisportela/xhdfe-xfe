noi di as text "xhdfe certification: DoF, SSC, and stat-style options"

clear
set obs 600
gen int firm = mod(_n - 1, 60) + 1
gen int year = mod(floor((_n - 1) / 60), 10) + 1
gen int cluster1 = mod(_n * 17, 53) + 1
gen int cluster2 = mod(_n * 31, 47) + 1
gen double x1 = sin(_n / 7)
gen double x2 = cos(_n / 11)
gen double y = 1 + .7 * x1 - .3 * x2 + firm / 20 + year / 9 + sin(_n / 13) / 10

local scalars "N df_r df_m F"

xhdfe y x1 x2, absorb(firm year) vce(cluster cluster1 cluster2) ///
    dofadjustments(none) ssc(K.adj=0 G.adj=0) statstyle(legacy) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(ref) scalars("`scalars'")
if (`"`e(dofmethod)'"' != "none") {
    di as error "dofadjustments(none) should post e(dofmethod)=none"
    exit 9
}
if (missing(e(df_a)) | missing(e(df_r))) {
    di as error "DoF/SSC run posted missing df_a or df_r"
    exit 9
}
di as text "  scalar dof_none_df_a: " e(df_a)
di as text "  scalar dof_none_df_r: " e(df_r)
di as text "  scalar dof_none_F: " %21.9g e(F)

xhdfe y x1 x2, absorb(firm year) vce(cluster cluster1 cluster2) ///
    dof(none) ssc(K.adj=0 G.adj=0) statstyle(current) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(alias) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref) testprefix(alias) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

capture noisily xhdfe y x1 x2, absorb(firm year) statstyle(nope) noheader notable nofootnote
local rc = _rc
if (`rc' != 198) {
    di as error "expected invalid statstyle() to fail with rc=198; got rc=`rc'"
    exit 9
}
di as text "  expected rc invalid_statstyle: `rc'"

exit
