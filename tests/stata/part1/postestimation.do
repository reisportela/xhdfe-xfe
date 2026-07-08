noi di as text "xhdfe certification: postestimation commands and slope predictions"

sysuse auto, clear
drop if missing(rep78)
gen double pw = (mod(_n, 5) + 1) / 3

xhdfe price weight length [pw=pw], absorb(rep78) keepsingletons residuals(u) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote
predict double xb, xb
predict double xbd, xbd
predict double score, score
predict double stdp, stdp
gen double y_minus_u = price - u
xcert_assert_var_close xbd y_minus_u if e(sample), tol(1e-10) name("postestimation xbd identity")
xcert_assert_var_close score u if e(sample), tol(1e-10) name("postestimation score identity")
quietly count if e(sample) & missing(stdp)
if (r(N) != 0) {
    di as error "predict stdp generated missing values in the estimation sample: " r(N)
    exit 9
}
quietly summarize stdp if e(sample), meanonly
di as text "  variable predict stdp: N=" r(N) " min=" %21.9g r(min) " max=" %21.9g r(max)

capture noisily margins, dydx(weight)
local rc = _rc
if (`rc' != 0) {
    di as error "margins failed with rc=`rc'"
    exit 9
}
di as text "  command margins dydx(weight): rc=0"

capture noisily xhdfe
local rc = _rc
if (`rc' != 0) {
    di as error "replay failed with rc=`rc'"
    exit 9
}
di as text "  command replay: rc=0"

capture noisily test weight = length
local rc = _rc
if (`rc' != 0) {
    di as error "test command failed with rc=`rc'"
    exit 9
}
di as text "  command test weight=length: rc=0"

capture noisily estat vce
local rc = _rc
if (`rc' != 0) {
    di as error "estat vce failed with rc=`rc'"
    exit 9
}
di as text "  command estat vce: rc=0"

clear
set obs 600
gen int firm = mod(_n - 1, 60) + 1
gen int year = mod(floor((_n - 1) / 60), 10) + 1
gen double z = mod(_n * 17, 23) / 10
gen double x1 = sin(_n / 7)
gen double x2 = cos(_n / 11)
gen double y = 1 + .7 * x1 - .3 * x2 + firm / 20 + year / 9 + ///
    .02 * firm * z + sin(_n / 13) / 10

xhdfe y x1 x2, absorb(firm##c.z year) residuals(u_slope) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
predict double xb_slope, xb
predict double xbd_slope, xbd
predict double d_slope, d
predict double dresid_slope, dresiduals
gen double y_minus_uslope = y - u_slope
gen double y_minus_xbslope = y - xb_slope
gen double xbd_minus_xb = xbd_slope - xb_slope
xcert_assert_var_close xbd_slope y_minus_uslope if e(sample), tol(1e-10) name("slope predict xbd")
xcert_assert_var_close dresid_slope y_minus_xbslope if e(sample), tol(1e-10) name("slope predict dresiduals")
xcert_assert_var_close d_slope xbd_minus_xb if e(sample), tol(1e-10) name("slope predict d")

exit
