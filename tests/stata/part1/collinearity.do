noi di as text "xhdfe certification: collinearity and omitted regressors"

clear
set obs 600
gen int firm = mod(_n - 1, 60) + 1
gen int year = mod(floor((_n - 1) / 60), 10) + 1
gen double x1 = sin(_n / 7)
gen double y = 1 + .7 * x1 + firm / 20 + year / 9 + sin(_n / 13) / 10

xhdfe y firm x1, absorb(firm year) tolerancemode(reghdfe-comparable) ///
    tolerance(1e-12) noheader notable nofootnote

local firm_col = colnumb(e(b), "firm")
if (`firm_col' == .) {
    di as error "absorbed main effect firm should remain in e(b) as an omitted column"
    exit 9
}
if (abs(_b[firm]) > 1e-12) {
    di as error "absorbed main effect firm should have zero coefficient; got " _b[firm]
    exit 9
}
matrix omit = e(omit_reason)
if (omit[`firm_col', 1] <= 0) {
    di as error "absorbed main effect firm should have positive omit_reason"
    matrix list omit
    exit 9
}
if (e(df_m) != 1) {
    di as error "only x1 should count as an effective regressor; e(df_m)=" e(df_m)
    exit 9
}
di as text "  scalar omitted_firm_coef: " %21.9g _b[firm]
di as text "  scalar omitted_firm_reason: " omit[`firm_col', 1]
di as text "  scalar collinearity_df_m: " e(df_m)

exit
