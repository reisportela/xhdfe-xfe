args package
adopath ++ "`package'"
findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
findfile xhdfe.plugin
assert "`r(fn)'" == "`package'/xhdfe.plugin"
sysuse auto, clear
regress price ibn.turn#c.gear_ratio, noconstant
predict double oracle_xb, xb
xhdfe price, absorb(turn#c.gear_ratio, save) keepsingletons residuals(fit_resid) fetolerance(1e-8)
assert e(df_m) == 0
capture confirm matrix e(b)
assert _rc == 111
assert reldif(__hdfe1__Slope1*gear_ratio,oracle_xb) < 1e-8
predict double got_xb, xb
predict double got_xbd, xbd
assert got_xb == 0 if e(sample)
assert reldif(got_xbd,oracle_xb) < 1e-8 if e(sample)

* Repeated automatic output is owned; its values are refreshed.
xhdfe price, absorb(turn#c.gear_ratio, save) keepsingletons fetolerance(1e-8)
assert reldif(__hdfe1__Slope1*gear_ratio,oracle_xb) < 1e-8
clonevar protected_input = __hdfe1__Slope1
capture noisily xhdfe price __hdfe1__Slope1, absorb(turn#c.gear_ratio, save) keepsingletons
assert _rc == 198
assert __hdfe1__Slope1 == protected_input
assert "`e(cmd)'" == ""

* A manually created variable in the namespace must survive a refused fit.
sysuse auto, clear
gen double __hdfe1__ = 37
capture noisily xhdfe price weight, absorb(turn, save)
assert _rc == 110
assert __hdfe1__ == 37
assert "`e(cmd)'" == ""
di "SAVEFE_LIFECYCLE_PASS"
