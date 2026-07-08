noi di as text "xhdfe certification: compatibility options and singleton behavior"

sysuse auto, clear
drop if missing(rep78)

xhdfe price weight gear_ratio, absorb(turn trunk) vce(robust) ///
    level(90) tolerance(1e-8) acceleration(none) transform(cimmino) prune ///
    noheader notable nofootnote
assert e(drop_singletons) == 1
assert e(num_singletons) == 10
assert e(N) == 59

xhdfe price weight gear_ratio, absorb(turn trunk) vce(robust) ///
    level(90) tolerance(1e-8) acceleration(none) transform(cimmino) prune keepsingletons ///
    noheader notable nofootnote
assert e(drop_singletons) == 0
assert e(num_singletons) == 0
assert e(N) == 69

xhdfe price weight gear_ratio, absorb(turn trunk) tolerance(1e-5) ///
    noheader notable nofootnote
local loose_ic = e(ic)
assert e(ic) < .

xhdfe price weight gear_ratio, absorb(turn trunk) tolerance(1e-10) ///
    noheader notable nofootnote
assert e(ic) < .
assert `loose_ic' <= e(ic)

xhdfe price weight gear_ratio, absorb(turn trunk) tolerance(1e-8) ///
    tolerancemode(reghdfe-comparable) absorptionmethod(auto) noheader notable nofootnote
assert "`e(tolerance_mode)'" == "reghdfe-comparable"
assert e(absorption_method_used) < .

exit
