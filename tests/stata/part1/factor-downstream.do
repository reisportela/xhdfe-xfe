noi di as text "xhdfe certification: factor-variable downstream consumers"

version 16
clear
set more off

xcert_require_reghdfe

set seed 20260805
set obs 6000
gen int cat = 1 + floor(4 * runiform())
gen int grp = 1 + floor(3 * runiform())
gen int div = 1 + floor(2 * runiform())
gen int firm = 1 + floor(120 * runiform())
gen int year = 1 + floor(10 * runiform())
gen double x1 = rnormal()
gen double y = .7*x1 + .15*cat - .08*grp + .04*cat*grp + ///
    .03*div + .02*firm - .03*year + rnormal()

quietly summarize y, meanonly
scalar factor_y_scale = max(1, abs(r(min)), abs(r(max)))

* The single-# model must remain consumable by every standard postestimator.
quietly xhdfe y x1 i.cat#i.grp, absorb(firm year) residuals(x_fit_resid) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
capture noisily xhdfe
assert _rc == 0
local xstripe : colnames e(b)
gen byte x_sample = e(sample)
predict double x_xb if x_sample, xb
predict double x_xbd if x_sample, xbd
predict double x_d if x_sample, d
predict double x_u if x_sample, residuals

gen double x_identity_fit = abs(x_xbd - x_xb - x_d) if x_sample
gen double x_identity_y = abs(y - x_xbd - x_u) if x_sample
quietly summarize x_identity_fit, meanonly
assert r(max) <= 1e-12 * factor_y_scale
quietly summarize x_identity_y, meanonly
assert r(max) <= 1e-12 * factor_y_scale
xcert_assert_var_close x_fit_resid x_u if x_sample, tol(1e-12) ///
    name("xhdfe residuals() vs predict, residuals")

quietly margins cat#grp
matrix x_margins_b = r(b)
matrix x_margins_V = r(V)
quietly lincom 2.cat#2.grp - 2.cat#3.grp
scalar x_lincom_b = r(estimate)
scalar x_lincom_se = r(se)
scalar x_lincom_df = r(df)
quietly testparm i.cat#i.grp
scalar x_test_F = r(F)
scalar x_test_df = r(df)
scalar x_test_dfr = r(df_r)

quietly reghdfe y x1 i.cat#i.grp, absorb(firm year) tolerance(1e-12) ///
    residuals(r_fit_resid)
local rstripe : colnames e(b)
assert `"`xstripe'"' == `"`rstripe'"'
gen byte r_sample = e(sample)
assert x_sample == r_sample
predict double r_xb if r_sample, xb
predict double r_xbd if r_sample, xbd
predict double r_d if r_sample, d
predict double r_u if r_sample, residuals

xcert_assert_var_close x_xb r_xb if x_sample, tol(1e-9) name("factor predict xb")
xcert_assert_var_close x_xbd r_xbd if x_sample, tol(1e-9) name("factor predict xbd")
xcert_assert_var_close x_d r_d if x_sample, tol(1e-9) name("factor predict d")
xcert_assert_var_close x_u r_u if x_sample, tol(1e-9) name("factor predict residuals")

quietly margins cat#grp
matrix r_margins_b = r(b)
matrix r_margins_V = r(V)
xcert_assert_matrix_close x_margins_b r_margins_b, tol(1e-10) ///
    name("factor margins r(b)")
xcert_assert_matrix_close x_margins_V r_margins_V, tol(1e-10) ///
    name("factor margins r(V)")

quietly lincom 2.cat#2.grp - 2.cat#3.grp
assert abs(x_lincom_b - r(estimate)) <= 1e-10 * max(1, abs(r(estimate)))
assert abs(x_lincom_se - r(se)) <= 1e-10 * max(1, abs(r(se)))
assert x_lincom_df == r(df)
quietly testparm i.cat#i.grp
assert abs(x_test_F - r(F)) <= 1e-9 * max(1, abs(r(F)))
assert x_test_df == r(df)
assert x_test_dfr == r(df_r)

* Non-default bases and three-way single-# interactions retain the same fit.
foreach fspec in "ib3.cat#i.grp" "i.cat#i.grp#i.div" {
    tempvar fx_resid fx_sample fx_xbd fr_resid fr_sample fr_xbd
    quietly xhdfe y x1 `fspec', absorb(firm year) residuals(`fx_resid') ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    gen byte `fx_sample' = e(sample)
    predict double `fx_xbd' if `fx_sample', xbd

    quietly reghdfe y x1 `fspec', absorb(firm year) tolerance(1e-12) ///
        residuals(`fr_resid')
    gen byte `fr_sample' = e(sample)
    assert `fx_sample' == `fr_sample'
    predict double `fr_xbd' if `fr_sample', xbd
    xcert_assert_var_close `fx_xbd' `fr_xbd' if `fx_sample', tol(1e-9) ///
        name("`fspec' predict xbd")
    xcert_assert_var_close `fx_resid' `fr_resid' if `fx_sample', tol(1e-9) ///
        name("`fspec' residuals")
}

* margins without absorbed dimensions must agree with exact LSDV regress.
quietly xhdfe y x1 i.cat#i.grp, ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
quietly margins cat#grp
matrix x_noabsorb_b = r(b)
matrix x_noabsorb_V = r(V)
quietly regress y x1 i.cat#i.grp
quietly margins cat#grp
matrix r_noabsorb_b = r(b)
matrix r_noabsorb_V = r(V)
xcert_assert_matrix_close x_noabsorb_b r_noabsorb_b, tol(1e-10) ///
    name("noabsorb factor margins r(b)")
xcert_assert_matrix_close x_noabsorb_V r_noabsorb_V, tol(1e-10) ///
    name("noabsorb factor margins r(V)")

* savefe must remain aligned with the factor-variable prediction stripe.
quietly xhdfe y x1 i.cat#i.grp, absorb(firm year, savefe) ///
    residuals(sf_fit_resid) tolerancemode(reghdfe-comparable) tolerance(1e-12)
gen byte sf_sample = e(sample)
predict double sf_xb if sf_sample, xb
predict double sf_xbd if sf_sample, xbd
predict double sf_d if sf_sample, d
predict double sf_u if sf_sample, residuals
gen double sf_sum = __hdfe1__ + __hdfe2__ if sf_sample
gen double sf_identity = y - sf_xb - sf_sum - sf_u if sf_sample
xcert_assert_var_close sf_sum sf_d if sf_sample, tol(1e-10) ///
    name("factor savefe sum vs predict d")
xcert_assert_var_close sf_fit_resid sf_u if sf_sample, tol(1e-12) ///
    name("factor savefe residuals() vs predict")
assert !missing(sf_xb, sf_xbd, sf_d, sf_u) if sf_sample
quietly summarize sf_identity, meanonly
assert r(max) <= 1e-12 * factor_y_scale
assert r(min) >= -1e-12 * factor_y_scale
local sf_label1 : variable label __hdfe1__
local sf_label2 : variable label __hdfe2__
assert `"`sf_label1'"' == "[FE] 1.firm"
assert `"`sf_label2'"' == "[FE] 1.year"

* The AKM companion accepts factors in controls(); its result must be
* invariant to an explicit full-rank dummy representation.
clear
set seed 20260807
set obs 1200
gen long worker = mod(_n - 1, 200) + 1
gen long firm = mod(floor((_n - 1) / 2) + worker, 55) + 1
gen int cat = 1 + floor(4 * runiform())
gen int grp = 1 + floor(3 * runiform())
egen int cell = group(cat grp)
gen double y = .2*cat - .1*grp + rnormal()

quietly xhdfeakm y, worker(worker) firm(firm) controls(i.cell) ///
    leverages(exact) threads(1)
matrix akm_factor_b = r(b)
assert colsof(akm_factor_b) == 12
assert akm_factor_b[1, 1] == 0
scalar akm_factor_va = r(kss_var_alpha)
scalar akm_factor_vp = r(kss_var_psi)
scalar akm_factor_cov = r(kss_cov)
scalar akm_factor_n = r(n_obs_connected)

tabulate cell, generate(cell_d)
quietly xhdfeakm y, worker(worker) firm(firm) ///
    controls(cell_d2 cell_d3 cell_d4 cell_d5 cell_d6 cell_d7 cell_d8 ///
        cell_d9 cell_d10 cell_d11 cell_d12) leverages(exact) threads(1)
matrix akm_manual_b = r(b)
matrix akm_factor_nonbase = akm_factor_b[1, 2..12]
local akm_manual_cols : colnames akm_manual_b
matrix colnames akm_factor_nonbase = `akm_manual_cols'
xcert_assert_matrix_close akm_factor_nonbase akm_manual_b, tol(1e-12) ///
    name("AKM factor controls vs manual dummies")
assert abs(akm_factor_va - r(kss_var_alpha)) <= 1e-12 * max(1, abs(akm_factor_va))
assert abs(akm_factor_vp - r(kss_var_psi)) <= 1e-12 * max(1, abs(akm_factor_vp))
assert abs(akm_factor_cov - r(kss_cov)) <= 1e-12 * max(1, abs(akm_factor_cov))
assert akm_factor_n == r(n_obs_connected)

noi di as result "PASS: factor-variable downstream consumers"
