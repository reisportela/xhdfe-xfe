noi di as text "xhdfe certification: one-way absorb, clustered VCE"

sysuse auto, clear
bys turn: gen t = _n
tsset turn t
drop if missing(rep78)

local lhs price
local rhs weight length
local absvars turn
local clustervar rep78
local scalars "N rmse tss rss r2 r2_a F df_r df_m ll ll_0"

areg `lhs' `rhs', absorb(`absvars') cluster(`clustervar')
xcert_store_estimates, prefix(ref) scalars("`scalars'")

xhdfe `lhs' `rhs', absorb(`absvars') keepsingletons tolerancemode(reghdfe-comparable) ///
    tolerance(1e-10) vce(cluster `clustervar') noheader notable nofootnote
xcert_store_estimates, prefix(xhd) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref) testprefix(xhd) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

* reghdfe-compatible shorthand.
xhdfe `lhs' `rhs', absorb(`absvars') keepsingletons tolerancemode(reghdfe-comparable) ///
    tolerance(1e-10) cluster(`clustervar') noheader notable nofootnote
xcert_store_estimates, prefix(short) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref) testprefix(short) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

exit
