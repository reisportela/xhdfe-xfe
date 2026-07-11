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

* Record the process default before nested HDFE/Gelbach fits.  The companion
* calls must restore OpenMP/Eigen state on every return path.
xhdfeconnected worker firm, generate(keep_default_before)
scalar companion_default_threads = r(threads_used)

* xhdfeakm verbose must expose progress without changing the estimator.
xhdfeakm y, worker(worker) firm(firm) leverages(jla) draws(8) seed(42)
local akm_converged_quiet = r(converged)
scalar akm_var_alpha_quiet = r(kss_var_alpha)
scalar akm_var_psi_quiet = r(kss_var_psi)
scalar akm_cov_quiet = r(kss_cov)
assert `akm_converged_quiet' == 1
xhdfeakm y, worker(worker) firm(firm) leverages(jla) draws(8) seed(42) verbose
local akm_converged_verbose = r(converged)
scalar akm_var_alpha_verbose = r(kss_var_alpha)
scalar akm_var_psi_verbose = r(kss_var_psi)
scalar akm_cov_verbose = r(kss_cov)
assert `akm_converged_verbose' == 1
assert akm_var_alpha_verbose == akm_var_alpha_quiet
assert akm_var_psi_verbose == akm_var_psi_quiet
assert akm_cov_verbose == akm_cov_quiet

* The two phases expose their effective teams separately.  The stronger
* restoration assertion (forced four-thread KSS team) lives in
* VALIDATE_AKM_KSS.py, where the process environment can be scoped safely.
xhdfeakm y, worker(worker) firm(firm) controls(x1) leverages(jla) draws(2) ///
    seed(42) threads(4)
local akm_controls_converged = r(converged)
local akm_fwl_threads = r(fwl_threads_used)
local akm_kss_threads = r(threads_used)
assert `akm_controls_converged' == 1
assert `akm_fwl_threads' >= 1
local akm_team_env : environment XHDFE_AKM_TEAM
if ("`akm_team_env'" == "0") assert `akm_kss_threads' == 4
else assert inrange(`akm_kss_threads', 1, 4)

* Canonical leave_out_COMPLETE reports only psi/cov component inference at
* match level, even on this movers-only sample.  The unsupported var(alpha)
* extension must fail loud instead of returning anti-conservative inference.
xhdfeakm y, worker(worker) firm(firm) leverages(exact) ci ///
    sensim(100) eigtracensim(20)
local akm_match_se_alpha = r(se_var_alpha)
local akm_match_theta_alpha = r(theta_var_alpha)
local akm_match_ci_lb_alpha = r(ci_lb_alpha)
local akm_match_ci_ub_alpha = r(ci_ub_alpha)
local akm_match_notes "`r(notes)'"
assert missing(`akm_match_se_alpha')
assert missing(`akm_match_theta_alpha')
assert missing(`akm_match_ci_lb_alpha') & missing(`akm_match_ci_ub_alpha')
assert strpos("`akm_match_notes'", "not identified at match level") > 0

* Collinear controls use the reduced model but must expose the omission.
gen double x1_duplicate = x1
xhdfeakm y, worker(worker) firm(firm) controls(x1 x1_duplicate) ///
    leverages(exact)
matrix akm_b_drop = r(b)
local akm_drop_notes "`r(notes)'"
assert akm_b_drop[1, 2] == 0
assert strpos("`akm_drop_notes'", "control column(s) 2 omitted") > 0

* Gelbach must be invariant to exact categorical relabelling, including raw
* identifiers outside the plugin's int32 transport range.
xhdfegelbach y, x1(x1) x2groups("observables = x2") fes(fe) ///
    vce(cluster) cluster(cluster) threads(4)
local gel_estimand_compact "`r(estimand)'"
local gel_causal_compact "`r(causal_interpretation)'"
local gel_gpu_requested_compact = r(gpu_requested)
local gel_gpu_used_compact = r(gpu_used)
local gel_gpu_code_compact = r(gpu_status_code)
local gel_gpu_status_compact "`r(gpu_status)'"
matrix gel_delta_compact = r(delta)
matrix gel_se_compact = r(se)
matrix gel_total_compact = r(total)
matrix gel_cov_compact = r(cov)
matrix gel_total_cov_compact = r(total_cov)
matrix gel_b_base_compact = r(b_base)
matrix gel_b_full_compact = r(b_full)
matrix gel_fe_total_compact = r(fe_total)
scalar gel_gap_compact = r(identity_gap)
assert rowsof(gel_cov_compact) == 4 & colsof(gel_cov_compact) == 4
assert rowsof(gel_total_cov_compact) == 2 & colsof(gel_total_cov_compact) == 2
assert colsof(gel_b_base_compact) == 1 & colsof(gel_b_full_compact) == 1
assert abs(gel_fe_total_compact[1, 1] - gel_delta_compact[1, 2]) <= 1e-12
assert "`gel_estimand_compact'" == "coefficient_movement"
assert "`gel_causal_compact'" == "no"
assert `gel_gpu_requested_compact' == 0
assert `gel_gpu_used_compact' == 0
assert `gel_gpu_code_compact' == 0
assert "`gel_gpu_status_compact'" == "not_requested"

* A severely near-collinear observed block can retain a valid identity while
* its split SE is tolerance/rounding sensitive; require an audible note.
gen double x2_near = x2 + 1.2e-6 * rnormal()
xhdfegelbach y, x1(x1) x2groups("near = x2 x2_near") fes(fe) ///
    vce(cluster) cluster(cluster) threads(4)
local gel_near_converged = r(converged)
local gel_near_notes "`r(notes)'"
assert `gel_near_converged' == 1
assert strpos("`gel_near_notes'", "x2 group 1 is severely ill-conditioned") > 0

* Verbose is output-only: same configuration must preserve every returned
* number, including the certified FE split and covariance.
xhdfegelbach y, x1(x1) x2groups("observables = x2") fes(fe) ///
    vce(cluster) cluster(cluster) threads(4) verbose
matrix gel_delta_verbose = r(delta)
matrix gel_se_verbose = r(se)
matrix gel_total_verbose = r(total)
xcert_assert_matrix_close gel_delta_compact gel_delta_verbose, tol(0) name("Gelbach quiet vs verbose delta")
xcert_assert_matrix_close gel_se_compact gel_se_verbose, tol(0) name("Gelbach quiet vs verbose SE")
xcert_assert_matrix_close gel_total_compact gel_total_verbose, tol(0) name("Gelbach quiet vs verbose total")

* Ambiguous block partitions and invalid tolerance settings fail closed.
capture noisily xhdfegelbach y, x1(x1) x2groups("A = x2 : B = x2")
assert _rc == 198
capture noisily xhdfegelbach y, x1(x1) x2groups("A = x1")
assert _rc == 198
capture noisily xhdfegelbach y, x1(x1) x2groups("A = x2") tol(0)
assert _rc == 198
capture noisily xhdfegelbach y, x1(x1) x2groups("A = x2") threads(-1)
assert _rc == 198
tempvar one_cluster
gen byte `one_cluster' = 1
capture noisily xhdfegelbach y, x1(x1) x2groups("A = x2") ///
    vce(cluster) cluster(`one_cluster')
assert _rc == 198

* With no absorbed FE there is no GPU absorption to accelerate; report that
* explicitly rather than labelling an ordinary OLS calculation as CUDA.
xhdfegelbach y, x1(x1) x2groups("observables = x2") gpu
local gel_nofe_converged = r(converged)
local gel_nofe_gpu_requested = r(gpu_requested)
local gel_nofe_gpu_used = r(gpu_used)
local gel_nofe_gpu_code = r(gpu_status_code)
local gel_nofe_gpu_status "`r(gpu_status)'"
assert `gel_nofe_converged' == 1
assert `gel_nofe_gpu_requested' == 1
assert `gel_nofe_gpu_used' == 0
assert `gel_nofe_gpu_code' == 6
assert "`gel_nofe_gpu_status'" == "not_applicable"

xhdfeconnected worker firm, generate(keep_default_after)
assert keep_default_before == keep_default_after
assert r(threads_used) == companion_default_threads

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
capture noisily xhdfeconnected worker firm, generate(keep_bad_threads) threads(-1)
assert _rc == 198
xhdfeconnected worker firm, generate(keep_compact) threads(4)
scalar connected_n_compact = r(n_obs)
local connected_threads_compact = r(threads_used)
local connected_gpu_used_compact = r(gpu_used)
local connected_gpu_status_compact "`r(gpu_status)'"
assert `connected_threads_compact' == 4
assert `connected_gpu_used_compact' == 0
assert "`connected_gpu_status_compact'" == "not_requested"
xhdfeconnected worker firm, generate(keep_verbose) threads(4) verbose
local connected_n_verbose = r(n_obs)
assert keep_compact == keep_verbose
assert `connected_n_verbose' == connected_n_compact
* Small samples stay on the faster CPU graph path even when GPU is requested;
* the fallback is explicit in diagnostics rather than silently labelled CUDA.
xhdfeconnected worker firm, generate(keep_gpu_small) threads(4) gpu
local connected_gpu_requested_small = r(gpu_requested)
local connected_gpu_used_small = r(gpu_used)
local connected_gpu_code_small = r(gpu_status_code)
local connected_gpu_status_small "`r(gpu_status)'"
assert keep_compact == keep_gpu_small
assert `connected_gpu_requested_small' == 1
assert `connected_gpu_used_small' == 0
assert `connected_gpu_code_small' == 6
assert "`connected_gpu_status_small'" == "not_beneficial"
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
