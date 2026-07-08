noi di as text "xhdfe certification: nosample postestimation behavior"

sysuse auto, clear
drop if missing(rep78)

xhdfe price weight length, absorb(rep78) keepsingletons nosample ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote

capture noisily predict double xb_ns, xb
local rc = _rc
if (`rc' != 0) {
    di as error "predict xb should work after nosample; got rc=`rc'"
    exit 9
}
di as text "  command nosample predict xb: rc=0"

capture noisily predict double d_ns, d
local rc = _rc
if (`rc' != 9) {
    di as error "predict d should fail without residuals() after nosample; got rc=`rc'"
    exit 9
}
di as text "  expected rc nosample predict d_without_residuals: `rc'"

xhdfe price weight length, absorb(rep78) keepsingletons nosample residuals(u_ns) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote
predict double xbd_ns, xbd
gen double y_minus_u = price - u_ns
xcert_assert_var_close xbd_ns y_minus_u, tol(1e-10) name("nosample residual-backed xbd")

exit
