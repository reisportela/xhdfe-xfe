version 16.0
clear all
set more off
local package : environment XHDFE_STATA_ADOPATH
adopath ++ "`package'"

set obs 2
generate double y = _n
generate long worker = 1
generate long firm = 7
xhdfeakm y, worker(worker) firm(firm) generate(small) threads(2)
assert r(converged) == 0
assert missing(r(var_y), r(sigma2_ho), r(plugin_var_alpha), r(plugin_corr))
assert missing(small_alpha, small_psi)

set obs 3
replace y = 3 in 3
replace worker = 2 in 3
replace firm = 8 in 3
generate byte fw = cond(_n <= 2, 5, 1)
xhdfeakm y [fw=fw], worker(worker) firm(firm) generate(compact) threads(2)
assert r(converged) == 1
assert r(n_obs_input) == 11 & r(n_obs) == 10
assert abs(r(var_y) - 2.5/9) < 1e-14
assert abs(r(sigma2_ho) - 2.5/9) < 1e-14
assert missing(r(plugin_corr), r(agsu_corr), r(kss_corr))
assert compact_alpha == 1.5 & compact_psi == 0 if compact_keep == 1
assert missing(compact_alpha, compact_psi) if compact_keep != 1

display "AKM_SMALL_SAMPLE_CONTRACT PASS"
