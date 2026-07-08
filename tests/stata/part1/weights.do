noi di as text "xhdfe certification: one-way absorb with weights"

sysuse auto, clear
bys turn: gen t = _n
tsset turn t
drop if missing(rep78)
gen double aw = 1 + turn / 100
gen int fw = ceil(weight / 1000)

local lhs price
local rhs weight length
local absvars turn
local scalars "N rmse tss rss mss r2 r2_a F df_r df_m"

areg `lhs' `rhs' [aw=aw], absorb(`absvars')
xcert_store_estimates, prefix(ref) scalars("`scalars'")

xhdfe `lhs' `rhs' [aw=aw], absorb(`absvars') keepsingletons tolerancemode(reghdfe-comparable) ///
    tolerance(1e-10) noheader notable nofootnote
xcert_store_estimates, prefix(xhd) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref) testprefix(xhd) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

areg `lhs' `rhs' [fw=fw], absorb(`absvars')
xcert_store_estimates, prefix(ref_fw) scalars("`scalars'")

xhdfe `lhs' `rhs' [fw=fw], absorb(`absvars') keepsingletons tolerancemode(reghdfe-comparable) ///
    tolerance(1e-10) noheader notable nofootnote
xcert_store_estimates, prefix(xhd_fw) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref_fw) testprefix(xhd_fw) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

exit
