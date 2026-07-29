noi di as text "xhdfe certification: forced absorption methods"

clear
set obs 1200
gen int firm = mod(_n - 1, 60) + 1
gen int year = mod(floor((_n - 1) / 60), 10) + 1
gen double x1 = sin(_n / 7)
gen double x2 = cos(_n / 11)
gen double y = 1 + .7 * x1 - .3 * x2 + firm / 20 + year / 9 + sin(_n / 13) / 10

local scalars "N df_r df_m"

xhdfe y x1 x2, absorb(firm year) absorptionmethod(auto) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote
xcert_store_estimates, prefix(ref) scalars("`scalars'")

local methods "gauss-seidel symmetric-gauss-seidel jacobi mlsmr lsmr auto-mlsmr"
local expected "1 2 3 6 5 1"
local i = 0
foreach method of local methods {
    local ++i
    local code : word `i' of `expected'
    xhdfe y x1 x2, absorb(firm year) absorptionmethod(`method') ///
        tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote
    xcert_store_estimates, prefix(test) scalars("`scalars'")
    xcert_compare_estimates, refprefix(ref) testprefix(test) scalars("`scalars'") ///
        btol(1e-8) vtol(1e-6) scaltol(1e-8)
    if (e(absorption_method_used) != `code') {
        di as error "absorptionmethod(`method') expected code `code'; got " e(absorption_method_used)
        exit 9
    }
    capture scalar __abs_residual = e(abs_residual)
    if (_rc | missing(__abs_residual) | __abs_residual < 0) {
        di as error "absorptionmethod(`method') did not expose a valid e(abs_residual)"
        exit 9
    }
    capture scalar __abs_residual_rel = e(abs_residual_rel)
    if (_rc | missing(__abs_residual_rel) | __abs_residual_rel < 0) {
        di as error "absorptionmethod(`method') did not expose a valid e(abs_residual_rel)"
        exit 9
    }
    capture scalar __precision_certified = e(precision_certified)
    if (_rc | !inlist(__precision_certified, 0, 1)) {
        di as error "absorptionmethod(`method') did not expose a Boolean e(precision_certified)"
        exit 9
    }
    di as text "  scalar absorption_method_`method': " e(absorption_method_used)
    di as text "  certificate `method': abs=" %21.14e __abs_residual ///
        " rel=" %21.14e __abs_residual_rel " certified=" __precision_certified
}

exit
