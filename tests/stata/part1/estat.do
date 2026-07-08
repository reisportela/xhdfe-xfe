noi di as text "xhdfe certification: estat commands"

sysuse auto, clear
drop if missing(rep78)

xhdfe price weight length, absorb(rep78) keepsingletons ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote

capture noisily estat summarize
local rc = _rc
if (`rc' != 0) {
    di as error "estat summarize failed with rc=`rc'"
    exit 9
}
di as text "  command estat summarize: rc=0"

capture noisily estat ic
local rc = _rc
if (`rc' != 0) {
    di as error "estat ic failed with rc=`rc'"
    exit 9
}
di as text "  command estat ic: rc=0"

capture noisily estat vce
local rc = _rc
if (`rc' != 0) {
    di as error "estat vce failed with rc=`rc'"
    exit 9
}
di as text "  command estat vce: rc=0"

exit
