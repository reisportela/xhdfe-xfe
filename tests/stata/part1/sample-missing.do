noi di as text "xhdfe certification: missing data, samples, and expected errors"

clear
set obs 20
gen double y = rnormal()
gen double x = rnormal()
gen int id = _n
capture noisily xhdfe y x, absorb(id) noheader notable nofootnote
local rc = _rc
if (`rc' != 198) {
    di as error "expected all-singleton design to fail with rc=198; got rc=`rc'"
    exit 9
}
di as text "  expected rc all-singleton: `rc'"

clear
set obs 100
gen int id = mod(_n - 1, 10) + 1
gen double y = rnormal()
gen double x = rnormal()
gen double z = rnormal()
replace y = . in 1
replace x = . in 2
replace id = . in 3
replace z = . in 4
gen byte keep = _n > 10
replace y = . in 5

xhdfe y x if keep, absorb(id) keepsingletons noheader notable nofootnote
if (e(N) != 90 | e(N_full) != 90) {
    di as error "if-sample missing-data count mismatch: e(N)=" e(N) " e(N_full)=" e(N_full)
    exit 9
}
di as text "  scalar if_sample_N: " e(N)

capture noisily xhdfe y x z, absorb(id) endogenous(x) noheader notable nofootnote
local rc = _rc
if (`rc' != 198) {
    di as error "expected endogenous() without instruments() to fail with rc=198; got rc=`rc'"
    exit 9
}
di as text "  expected rc endogenous_without_instruments: `rc'"

capture noisily xhdfe y x z, absorb(id) instruments(z) noheader notable nofootnote
local rc = _rc
if (`rc' != 198) {
    di as error "expected instruments() without endogenous() to fail with rc=198; got rc=`rc'"
    exit 9
}
di as text "  expected rc instruments_without_endogenous: `rc'"

capture noisily xhdfe y x z, absorb(id) endogenous(x) instruments(z) group(id) noheader notable nofootnote
local rc = _rc
if (`rc' != 198) {
    di as error "expected IV in group mode to fail with rc=198; got rc=`rc'"
    exit 9
}
di as text "  expected rc iv_group_mode: `rc'"

capture noisily xhdfe y x, individual(id) noheader notable nofootnote
local rc = _rc
if (`rc' != 198) {
    di as error "expected individual() without group() to fail with rc=198; got rc=`rc'"
    exit 9
}
di as text "  expected rc individual_without_group: `rc'"

xhdfe y x, absorb(id) endogenous(x) instruments(z) keepsingletons noheader notable nofootnote
if (e(N) != 95) {
    di as error "IV missing-data sample count mismatch: e(N)=" e(N)
    exit 9
}
di as text "  scalar iv_markout_N: " e(N)

capture noisily xhdfe y x z, absorb(id) endogenous(x) instruments(z) ///
    keepsingletons noheader notable nofootnote
local rc = _rc
if (`rc' != 198) {
    di as error "expected duplicated included/excluded instrument to fail with rc=198; got rc=`rc'"
    exit 9
}
di as text "  expected rc duplicated_included_excluded_instrument: `rc'"

exit
