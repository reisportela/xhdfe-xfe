* Companion-command release gates: categorical relabelling and plugin binding.

version 16
clear
set more off
set seed 20260710
set obs 600

gen long worker = mod(_n - 1, 120) + 1
gen long firm = mod(floor((_n - 1) / 3) + worker, 35) + 1
gen long fe = mod(_n - 1, 24) + 1
gen long cluster = mod(_n - 1, 18) + 1
gen double x1 = rnormal()
gen double x2 = 0.35 * x1 + rnormal()
gen double y = 1.2 * x1 - 0.7 * x2 + 0.03 * fe + rnormal()

* xhdfeakm verbose must expose progress without changing the estimator.
xhdfeakm y, worker(worker) firm(firm) leverages(jla) draws(8) seed(42)
assert r(converged) == 1
scalar akm_var_alpha_quiet = r(kss_var_alpha)
scalar akm_var_psi_quiet = r(kss_var_psi)
scalar akm_cov_quiet = r(kss_cov)
xhdfeakm y, worker(worker) firm(firm) leverages(jla) draws(8) seed(42) verbose
assert r(converged) == 1
assert r(kss_var_alpha) == akm_var_alpha_quiet
assert r(kss_var_psi) == akm_var_psi_quiet
assert r(kss_cov) == akm_cov_quiet

* The two phases expose their effective teams separately.  The stronger
* restoration assertion (forced four-thread KSS team) lives in
* VALIDATE_AKM_KSS.py, where the process environment can be scoped safely.
xhdfeakm y, worker(worker) firm(firm) controls(x1) leverages(jla) draws(2) ///
    seed(42) threads(4)
assert r(converged) == 1
assert r(fwl_threads_used) >= 1
local akm_team_env : environment XHDFE_AKM_TEAM
if ("`akm_team_env'" == "0") assert r(threads_used) == 4
else assert inrange(r(threads_used), 1, 4)

* Gelbach must be invariant to exact categorical relabelling, including raw
* identifiers outside the plugin's int32 transport range.
xhdfegelbach y, x1(x1) x2groups("observables = x2") fes(fe) ///
    vce(cluster) cluster(cluster)
matrix gel_delta_compact = r(delta)
matrix gel_se_compact = r(se)
matrix gel_total_compact = r(total)
scalar gel_gap_compact = r(identity_gap)

recast double fe cluster
replace fe = 3000000000 + 1009 * fe + 0.25
replace cluster = -3000000000 + 1013 * cluster + 0.25
xhdfegelbach y, x1(x1) x2groups("observables = x2") fes(fe) ///
    vce(cluster) cluster(cluster)
matrix gel_delta_large = r(delta)
matrix gel_se_large = r(se)
matrix gel_total_large = r(total)
scalar gel_gap_large = r(identity_gap)

xcert_assert_matrix_close gel_delta_compact gel_delta_large, tol(1e-12) name("Gelbach delta after id relabelling")
xcert_assert_matrix_close gel_se_compact gel_se_large, tol(1e-12) name("Gelbach SE after id relabelling")
xcert_assert_matrix_close gel_total_compact gel_total_large, tol(1e-12) name("Gelbach total after id relabelling")
assert abs(gel_gap_compact - gel_gap_large) <= 1e-12

* xhdfeconnected has the same public categorical contract.
xhdfeconnected worker firm, generate(keep_compact)
recast double worker firm
replace worker = 4000000000 + 1009 * worker + 0.25
replace firm = -4000000000 + 1013 * firm + 0.25
xhdfeconnected worker firm, generate(keep_large)
assert keep_compact == keep_large

* A loaded dispatcher from another checkout must never be reused silently.
local bound "$XHDFE_PLUGIN_PATH_INTERNAL"
global XHDFE_PLUGIN_PATH_INTERNAL "/xhdfe/release-gate/not-the-active-plugin"
capture noisily xhdfegelbach y, x1(x1) x2groups("observables = x2") fes(fe)
assert _rc == 498
global XHDFE_PLUGIN_PATH_INTERNAL "`bound'"

di as result "companion-command release gates passed"
