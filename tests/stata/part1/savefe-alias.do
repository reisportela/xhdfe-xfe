noi di as text "xhdfe certification: named/aliased saved FEs (mixed unnamed+named absorb)"

* Regression guard for the 2.13.1 alias-mapping bug: with a mix of unnamed and
* named absorb terms -- absorb(year wfe=worker ffe=firm) -- the saved FEs
* landed shifted one dimension left (wfe captured the YEAR effect, constant
* within year; ffe captured the WORKER effect). This test anchors the named
* saved FEs element-wise to reghdfe's named savefe output and asserts the
* within-year variability that the bug destroyed.

xcert_require_reghdfe

clear
set obs 6000
gen int w_id = ceil(_n / 12)
bys w_id: gen int year = 2000 + _n
set seed 20260708
gen int f_id = ceil(runiform() * 60)
gen double x = rnormal()
gen double y = 0.4 * x + 0.5 * rnormal() + w_id / 500 + f_id / 60 + 0.05 * (year - 2000)

* --- xhdfe with an unnamed first term followed by two named terms -----------
xhdfe y x, absorb(year wfe=w_id ffe=f_id) tolerancemode(reghdfe-comparable) ///
    tolerance(1e-10) noheader notable nofootnote
xcert_store_estimates, prefix(xal) scalars("N df_r df_m")

* the named receiving variables must exist; the unnamed year term must NOT
* leave a variable behind
capture confirm variable wfe
if (c(rc)) {
    di as error "named saved FE wfe was not created"
    exit 9
}
capture confirm variable ffe
if (c(rc)) {
    di as error "named saved FE ffe was not created"
    exit 9
}
capture confirm variable __hdfe1__
if (!c(rc)) {
    di as error "unnamed absorb term leaked a __hdfe1__ variable into the dataset"
    exit 9
}

* the 2.13.1 bug signature: wfe held the year FE => zero variance within year
quietly summarize wfe if year == 2005
if (r(sd) < 1e-6) {
    di as error "wfe is (nearly) constant within a year: the named saved FE captured the year effect"
    exit 9
}
di as text "  sd(wfe | year==2005) = " %12.6g r(sd)

* --- oracle: reghdfe's named savefe on the same specification ----------------
reghdfe y x, absorb(year wfe2=w_id ffe2=f_id)
xcert_store_estimates, prefix(ral) scalars("N df_r df_m")

xcert_compare_estimates, refprefix(ral) testprefix(xal) scalars("N df_r df_m") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)
xcert_assert_var_close wfe wfe2, tol(1e-6) name("wfe vs reghdfe wfe")
xcert_assert_var_close ffe ffe2, tol(1e-6) name("ffe vs reghdfe ffe")

* --- re-run safety: the same command must run again after dropping outputs ---
drop wfe ffe wfe2 ffe2
capture noisily xhdfe y x, absorb(year wfe=w_id ffe=f_id) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-10) noheader notable nofootnote
if (c(rc)) {
    di as error "re-running the aliased-savefe command failed with rc=" c(rc)
    exit 9
}
di as text "  re-run after drop: ok"
drop wfe ffe

exit
