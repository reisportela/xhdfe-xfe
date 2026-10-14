args package
adopath ++ "`package'"
findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
findfile xhdfe.plugin
assert "`r(fn)'" == "`package'/xhdfe.plugin"

* A factor-variable response must reproduce explicit binary OLS and prediction.
sysuse auto, clear
gen double binary_response = (turn == 43)
areg binary_response price weight, absorb(trunk)
matrix B = e(b)
matrix V = e(V)
predict double oracle_xb, xb
predict double oracle_xbd, xbd
predict double oracle_resid, residuals
xhdfe 43.turn price weight, absorb(trunk) keepsingletons residuals(fv_resid)
assert mreldif(B,e(b)) < 1e-10
assert mreldif(V,e(V)) < 1e-10
predict double got_xb, xb
predict double got_xbd, xbd
predict double got_dr, dresiduals
assert abs(oracle_xb-got_xb) < 1e-9 if e(sample)
assert abs(oracle_xbd-got_xbd) < 1e-9 if e(sample)
assert abs(oracle_resid-fv_resid) < 1e-9 if e(sample)
assert abs(got_dr-(binary_response-got_xb)) < 1e-9 if e(sample)

* TS responses and regressors use the same sample as native regress.
sysuse auto, clear
bys turn: gen int period = _n
tsset turn period
regress F.price L.weight
matrix B = e(b)
matrix V = e(V)
gen byte reference_sample = e(sample)
predict double reference_xb, xb
predict double reference_resid, residuals
xhdfe F.price L.weight, noabsorb residuals(ts_resid)
assert reference_sample == e(sample)
assert mreldif(B,e(b)) < 1e-10
assert mreldif(V,e(V)) < 1e-10
predict double fitted, xb
predict double fitted_all, xbd
assert abs(fitted-reference_xb) < 1e-8 if e(sample)
assert abs(ts_resid-reference_resid) < 1e-8 if e(sample)
assert abs(fitted_all-fitted) < 1e-8 if e(sample)
di "NEW_FRONTEND_REGRESSIONS_PASS"
