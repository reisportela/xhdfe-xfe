noi di as text "xhdfe certification: factor-variable parsing under absorb()"

sysuse auto, clear
bys turn: gen t = _n
tsset turn t
drop if missing(rep78)

local lhs price
local rhs i.foreign c.weight##c.length
local absvars turn
local scalars "N rmse tss rss mss r2 r2_a F df_r df_m ll ll_0"

areg `lhs' `rhs', absorb(`absvars')
xcert_store_estimates, prefix(ref) scalars("`scalars'")

xhdfe `lhs' `rhs', absorb(`absvars') keepsingletons tolerancemode(reghdfe-comparable) ///
    tolerance(1e-10) noheader notable nofootnote
xcert_store_estimates, prefix(xhd) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref) testprefix(xhd) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

exit
