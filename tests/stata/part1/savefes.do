noi di as text "xhdfe certification: saved fixed effects"

xcert_require_reghdfe

clear
set obs 600
gen int firm = mod(_n - 1, 60) + 1
gen int year = mod(floor((_n - 1) / 60), 10) + 1
gen double x1 = sin(_n / 7)
gen double x2 = cos(_n / 11)
gen double y = 1 + .7 * x1 - .3 * x2 + firm / 20 + year / 9 + sin(_n / 13) / 10

local scalars "N rmse rss tss mss r2 r2_a F df_r df_m"

reghdfe y x1 x2, absorb(firm year) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(ref) scalars("`scalars'")

xhdfe y x1 x2, absorb(firm year) savefes(xf_) residuals(u) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(xhd) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref) testprefix(xhd) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)

predict double xb, xb
predict double xbd, xbd
gen double saved_xbd = xb + xf_firm + xf_year
gen double y_minus_u = y - u

xcert_assert_var_close saved_xbd xbd if e(sample), tol(1e-10) name("savefes sum vs predict xbd")
xcert_assert_var_close saved_xbd y_minus_u if e(sample), tol(1e-10) name("savefes sum vs y-residual")

if (e(fe_recovery_converged) != 1) {
    di as error "fixed-effect recovery did not converge"
    exit 9
}
if (e(fe_recovery_max_delta) > 1e-8) {
    di as error "fixed-effect recovery max delta too large: " e(fe_recovery_max_delta)
    exit 9
}
di as text "  scalar fe_recovery_converged: " e(fe_recovery_converged)
di as text "  scalar fe_recovery_iterations: " e(fe_recovery_iterations)
di as text "  scalar fe_recovery_max_delta: " %21.9g e(fe_recovery_max_delta)

exit
