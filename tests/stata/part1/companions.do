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
* tests/validation/VALIDATE_AKM_KSS.py, where the process environment can be scoped safely.
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
matrix gel_base_cov_compact = r(base_cov)
matrix gel_cov_db_compact = r(cov_delta_bbase)
matrix gel_cov_tb_compact = r(cov_total_bbase)
matrix gel_gamma_compact = r(gamma)
matrix gel_fe_ratio_compact = r(x1_fe_collinear_ratio)
matrix gel_near_mask_compact = r(x1_near_collinear_mask)
matrix gel_b_base_compact = r(b_base)
matrix gel_b_full_compact = r(b_full)
matrix gel_fe_total_compact = r(fe_total)
scalar gel_gap_compact = r(identity_gap)
assert rowsof(gel_cov_compact) == 4 & colsof(gel_cov_compact) == 4
assert rowsof(gel_total_cov_compact) == 2 & colsof(gel_total_cov_compact) == 2
assert rowsof(gel_base_cov_compact) == 2 & colsof(gel_base_cov_compact) == 2
assert rowsof(gel_cov_db_compact) == 4 & colsof(gel_cov_db_compact) == 2
assert rowsof(gel_cov_tb_compact) == 2 & colsof(gel_cov_tb_compact) == 2
assert rowsof(gel_gamma_compact) == 1 & colsof(gel_gamma_compact) == 1
assert rowsof(gel_fe_ratio_compact) == 1 & colsof(gel_fe_ratio_compact) == 1
assert rowsof(gel_near_mask_compact) == 1 & colsof(gel_near_mask_compact) == 1
assert !missing(gel_gamma_compact[1, 1])
assert gel_near_mask_compact[1, 1] == 0
assert colsof(gel_b_base_compact) == 1 & colsof(gel_b_full_compact) == 1
assert abs(gel_fe_total_compact[1, 1] - gel_delta_compact[1, 2]) <= 1e-12
assert r(df_base) > 0 & r(df_full) > 0
assert r(n_clusters) == 18
assert r(few_cluster_warning_threshold) == 30
assert r(near_fe_warn_upper) == 1e-4
assert strpos("`r(notes)'", "few clusters") > 0
assert strpos("`r(notes)'", "near-FE-collinear focal") == 0
forvalues rr = 1/2 {
    forvalues cc = 1/2 {
        assert abs(gel_cov_tb_compact[`rr', `cc'] - ///
            gel_cov_db_compact[`rr', `cc'] - ///
            gel_cov_db_compact[2 + `rr', `cc']) < 1e-12
    }
}
assert "`gel_estimand_compact'" == "coefficient_movement"
assert "`gel_causal_compact'" == "no"
assert `gel_gpu_requested_compact' == 0
assert `gel_gpu_used_compact' == 0
assert `gel_gpu_code_compact' == 0
assert "`gel_gpu_status_compact'" == "not_requested"
quietly regress y x1 x2 i.fe
assert abs(gel_gamma_compact[1, 1] - _b[x2]) < 1e-10
gen long cluster50 = mod(_n - 1, 50) + 1
quietly xhdfegelbach y, x1(x1) x2groups("observables = x2") fes(fe) ///
    vce(cluster) cluster(cluster50)
assert r(n_clusters) == 50
assert strpos("`r(notes)'", "few clusters") == 0

* FEs common to base and full condition the decomposition; fes() remains the
* added/decomposable surface.  The slope identity is benchmarked against dense
* LSDV while the normalization-dependent intercept row must be missing.
gen long common_fe = mod(floor((_n - 1) / 2), 30) + 1
quietly regress y x1 i.common_fe, vce(robust)
scalar common_bbase_oracle = _b[x1]
quietly regress y x1 x2 i.common_fe i.fe, vce(robust)
scalar common_bfull_oracle = _b[x1]
xhdfegelbach y, x1(x1) x2groups("observables = x2") ///
    commonfes(common_fe) fes(fe) vce(robust)
matrix gel_common_base = r(b_base)
matrix gel_common_full = r(b_full)
matrix gel_common_delta = r(delta)
matrix gel_common_total = r(total)
matrix gel_common_cov = r(cov)
matrix gel_common_basecov = r(base_cov)
assert abs(gel_common_base[1, 1] - common_bbase_oracle) < 1e-10
assert abs(gel_common_full[1, 1] - common_bfull_oracle) < 1e-10
assert abs(gel_common_total[1, 1] - ///
    (common_bbase_oracle - common_bfull_oracle)) < 1e-10
assert abs(gel_common_total[1, 1] - ///
    gel_common_delta[1, 1] - gel_common_delta[1, 2]) < 1e-10
assert missing(gel_common_total[2, 1]) & missing(gel_common_total[2, 2])
assert missing(gel_common_cov[2, 2]) & missing(gel_common_basecov[2, 2])
assert r(identity_gap) < 1e-10
assert r(n_common_fes) == 1
assert r(common_fes_applied) == 1
assert r(intercept_inference_available) == 0
assert "`r(common_fes)'" == "common_fe"
assert "`r(intercept_status)'" == "not_certified_common_fes"
assert "`r(identity_status)'" == "exact_ols_conditional_common_fes"
matrix gel_common_status = r(regular_inference_status_code)
assert gel_common_status[2, 1] == -2
assert strpos("`r(notes)'", "common FEs were conditioned out") > 0
capture noisily xhdfegelbach y, x1(x1) commonfes(common_fe)
assert _rc == 198
capture noisily xhdfegelbach y, x1(x1) x2groups("observables = x2") ///
    commonfes(fe) fes(fe)
assert _rc == 198

* Exactly two FE dimensions are certified only on one retained mobility
* component. Use an explicit worker-firm ring rather than assuming that the
* broader companion fixture is connected (it deliberately is not).
preserve
clear
set obs 64
gen long connected_worker = floor((_n - 1) / 4) + 1
gen byte connected_period = mod(_n - 1, 4)
gen long connected_firm = ///
    mod(connected_worker - 1 + (connected_period >= 2), 8) + 1
gen long connected_occupation = ///
    mod(connected_worker + connected_period, 5) + 1
gen double connected_x1 = rnormal()
gen double connected_x2 = 0.25 * connected_x1 + rnormal()
gen double connected_y = 0.7 * connected_x1 + 0.4 * connected_x2 + ///
    0.1 * connected_worker - 0.07 * connected_firm + rnormal()
xhdfegelbach connected_y, x1(connected_x1) ///
    x2groups("observables = connected_x2") ///
    fes(connected_worker connected_firm)
assert r(n_mobility_components) == 1
assert r(largest_mobility_component_n_obs) == r(n_obs)
assert r(largest_mobility_component_share) == 1
assert r(largest_mobility_weight_share) == 1
assert r(fe_split_identified) == 1
assert "`r(fe_split_status)'" == "identified_two_way"
assert r(connectivity_fe1_index) == 0
assert r(connectivity_fe2_index) == 1
assert r(connectivity_pair_explicit) == 0
assert "`r(connectivity_fes)'" == "connected_worker connected_firm"
assert "`r(connectivity_fe_indices)'" == "0 1"
assert "`r(connectivity_pair_status)'" == "connected"
assert "`r(connected_mode)'" == "diagnose"
assert "`r(mobility_component_scope)'" == "first_two_fe_dimensions"
xhdfegelbach connected_y, x1(connected_x1) ///
    x2groups("observables = connected_x2") ///
    fes(connected_worker connected_firm) connected(require) ///
    connectivityfes(connected_firm connected_worker)
assert r(fe_split_identified) == 1
assert r(connectivity_fe1_index) == 1
assert r(connectivity_fe2_index) == 0
assert r(connectivity_pair_explicit) == 1
assert "`r(connectivity_fes)'" == "connected_firm connected_worker"
assert "`r(connectivity_fe_indices)'" == "1 0"
assert "`r(connectivity_pair_status)'" == "connected"
assert "`r(connected_mode)'" == "require"
assert "`r(mobility_component_scope)'" == "selected_fe_pair"

* Three or more dimensions remain explicitly uncertified even when their
* first pair is connected.
xhdfegelbach connected_y, x1(connected_x1) ///
    x2groups("observables = connected_x2") ///
    fes(connected_worker connected_firm connected_occupation)
assert r(n_mobility_components) == 1
assert r(fe_split_identified) == 0
assert "`r(fe_split_status)'" == "not_certified_multiway"
assert "`r(connectivity_pair_status)'" == "connected"
assert strpos("`r(notes)'", "not connectivity-certified") > 0
capture noisily xhdfegelbach connected_y, x1(connected_x1) ///
    x2groups("observables = connected_x2") ///
    fes(connected_worker connected_firm connected_occupation) ///
    connected(require)
assert _rc != 0
restore

* The disconnected fixture includes a huge-weight raw singleton. Exact
* component shares must be computed after recursive singleton removal.
preserve
clear
set obs 65
gen byte component = cond(_n <= 32, 0, cond(_n <= 64, 1, 2))
gen int within_component = mod(_n - 1, 32)
gen int period = mod(within_component, 4)
gen long worker2 = component * 8 + floor(within_component / 4) + 1
gen long firm2 = component * 4 + ///
    mod(floor(within_component / 4) + (period >= 2), 4) + 1
replace worker2 = 17 in 65
replace firm2 = 9 in 65
gen double target2 = component + 0.2 * rnormal()
gen double observed2 = 0.3 * target2 + rnormal()
gen double outcome2 = 0.8 * target2 + 0.5 * observed2 + ///
    0.1 * worker2 - 0.07 * firm2 + rnormal()
gen double weight2 = cond(component == 0, 1, 3)
replace weight2 = 100 in 65
gen byte bridge2 = mod(period, 2)
xhdfegelbach outcome2 [aweight=weight2], x1(target2) ///
    x2groups("observed = observed2") fes(worker2 firm2)
local disconnected_status "`r(fe_split_status)'"
local disconnected_scope "`r(mobility_component_scope)'"
local disconnected_notes "`r(notes)'"
scalar disconnected_n_input = r(n_obs_input)
scalar disconnected_n = r(n_obs)
scalar disconnected_singletons = r(n_singletons_dropped)
scalar disconnected_components = r(n_mobility_components)
scalar disconnected_largest_n = r(largest_mobility_component_n_obs)
scalar disconnected_largest_share = r(largest_mobility_component_share)
scalar disconnected_largest_wshare = r(largest_mobility_weight_share)
scalar disconnected_identified = r(fe_split_identified)
matrix disconnected_delta = r(delta)
matrix disconnected_fe_total = r(fe_total)
assert disconnected_n_input == 65
assert disconnected_n == 64
assert disconnected_singletons == 1
assert disconnected_components == 2
assert disconnected_largest_n == 32
assert disconnected_largest_share == 0.5
assert disconnected_largest_wshare == 0.75
assert disconnected_identified == 0
assert "`disconnected_status'" == "normalization_dependent"
assert "`disconnected_scope'" == "first_two_fe_dimensions"
assert "`r(connectivity_pair_status)'" == "disconnected"
assert "`r(connected_mode)'" == "diagnose"
assert strpos("`disconnected_notes'", "normalization-dependent") > 0
forvalues rr = 1/2 {
    assert disconnected_fe_total[`rr', 1] == ///
        disconnected_delta[`rr', 2] + disconnected_delta[`rr', 3]
}

capture noisily xhdfegelbach outcome2 [aweight=weight2], x1(target2) ///
    x2groups("observed = observed2") fes(worker2 firm2) connected(require)
assert _rc != 0

* With 3+ FE, selecting another pair changes only the pair diagnostic. The
* global FE split remains explicitly uncertified and the numerics are inert.
tempname C3DEFAULT C3SELECTED C3GAP
xhdfegelbach outcome2 [aweight=weight2], x1(target2) ///
    x2groups("observed = observed2") fes(worker2 firm2 bridge2)
assert r(n_mobility_components) == 2
assert "`r(connectivity_pair_status)'" == "disconnected"
assert "`r(fe_split_status)'" == "not_certified_multiway"
matrix `C3DEFAULT' = r(cov)
xhdfegelbach outcome2 [aweight=weight2], x1(target2) ///
    x2groups("observed = observed2") fes(worker2 firm2 bridge2) ///
    connectivityfes(worker2 bridge2)
assert r(n_mobility_components) == 1
assert r(connectivity_fe1_index) == 0
assert r(connectivity_fe2_index) == 2
assert r(connectivity_pair_explicit) == 1
assert "`r(connectivity_fes)'" == "worker2 bridge2"
assert "`r(connectivity_pair_status)'" == "connected"
assert "`r(fe_split_status)'" == "not_certified_multiway"
assert "`r(mobility_component_scope)'" == "selected_fe_pair"
matrix `C3SELECTED' = r(cov)
mata: st_numscalar("`C3GAP'", max(abs(st_matrix("`C3DEFAULT'") :- st_matrix("`C3SELECTED'"))))
assert scalar(`C3GAP') == 0
restore

* The empirical reporting layer is opt-in and numerically inert. A common
* control remains in x1() while focal() selects only the paper-facing row.
gen double common_control = 0.15 * x1 + rnormal()
quietly xhdfegelbach y, x1(x1 common_control) focal(x1) ///
    x2groups("observables = x2") fes(fe) shares(movement)
assert r(converged) == 1
assert r(focal_selection_explicit) == 1
assert "`r(x1_names)'" == "x1 common_control"
assert "`r(focal_indices)'" == "0"
assert "`r(focal_names)'" == "x1"
assert "`r(share_denominator)'" == "movement"
assert "`r(share_se_type)'" == "joint_covariance_delta_method"
assert "`r(share_units)'" == "fraction"
matrix gel_reporting_delta = r(delta)
matrix gel_movement_share = r(share)
matrix gel_movement_share_se = r(share_se)
assert abs(gel_movement_share[1, 1] + gel_movement_share[1, 2] - 1) < 1e-12
assert !missing(gel_movement_share_se[1, 1])

quietly xhdfegelbach y, x1(x1 common_control) ///
    x2groups("observables = x2") fes(fe)
matrix gel_reporting_default_delta = r(delta)
xcert_assert_matrix_close gel_reporting_delta gel_reporting_default_delta, ///
    tol(0) name("Gelbach focal reporting is numerically inert")

xhdfegelbach y, x1(x1 common_control) focal(x1) ///
    x2groups("observables = x2") fes(fe) shares(base)
matrix gel_base_share = r(share)
matrix gel_base_share_se = r(share_se)
matrix gel_base_share_lo = r(share_ci_low)
matrix gel_base_share_hi = r(share_ci_high)
matrix gel_base_share_delta = r(delta)
matrix gel_base_share_cov = r(cov)
matrix gel_base_share_base_cov = r(base_cov)
matrix gel_base_share_cov_db = r(cov_delta_bbase)
matrix gel_base_share_b = r(b_base)
assert !missing(gel_base_share[1, 1])
assert !missing(gel_base_share_se[1, 1])
assert !missing(gel_base_share_lo[1, 1])
assert !missing(gel_base_share_hi[1, 1])
assert gel_base_share_lo[1, 1] < gel_base_share_hi[1, 1]
assert "`r(share_se_type)'" == "joint_base_covariance_delta_method"
scalar gel_share_manual_var = ///
    gel_base_share_cov[1, 1] / (gel_base_share_b[1, 1]^2) + ///
    (gel_base_share_delta[1, 1]^2 * gel_base_share_base_cov[1, 1]) / ///
        (gel_base_share_b[1, 1]^4) - ///
    (2 * gel_base_share_delta[1, 1] * gel_base_share_cov_db[1, 1]) / ///
        (gel_base_share_b[1, 1]^3)
assert abs(gel_base_share_se[1, 1] - sqrt(max(0, gel_share_manual_var))) < 1e-12

capture noisily xhdfegelbach y, x1(x1 common_control) focal(x1) ///
    x2groups("observables = x2") fes(fe) shares(base) sharetol(1e20)
assert _rc == 0
matrix gel_undefined_share = r(share)
matrix gel_undefined_share_se = r(share_se)
assert missing(gel_undefined_share[1, 1])
assert missing(gel_undefined_share_se[1, 1])
assert strpos("`r(notes)'", "share denominator") > 0

quietly xhdfegelbach y, x1(x1 common_control) focal(x1) ///
    x2groups("observables = x2") fes(fe) shares(base_fixed)
matrix gel_fixed_share_se = r(share_se)
assert !missing(gel_fixed_share_se[1, 1])
assert "`r(share_se_type)'" == "fixed_base_denominator_scaling"

capture noisily xhdfegelbach y, x1(x1 common_control) focal(x2) ///
    x2groups("observables = x2") fes(fe)
assert _rc == 198
capture noisily xhdfegelbach y, x1(x1 common_control) ///
    x2groups("observables = x2") fes(fe) shares(unknown)
assert _rc == 198

* A worker-invariant X1 target is rejected by the standard estimand.  The
* explicit absorbed-target mode constrains only that coefficient to zero and
* labels it as imposed rather than estimated.
gen byte female = mod(worker, 2)
capture noisily xhdfegelbach y, x1(female x1) ///
    x2groups("observables = x2") fes(worker)
assert _rc != 0
xhdfegelbach y, x1(female x1) x2groups("observables = x2") ///
    fes(worker) absorbedtargets(female) vce(cluster) cluster(worker)
local gel_abs_estimand "`r(estimand)'"
local gel_abs_identity "`r(identity_status)'"
local gel_abs_targets "`r(absorbed_targets)'"
local gel_abs_target_names "`r(absorbed_target_names)'"
local gel_abs_bstatus "`r(b_full_status)'"
local gel_abs_fstatus "`r(focal_status)'"
local gel_abs_total_se_type "`r(total_se_type)'"
local gel_abs_inference_status "`r(inference_status)'"
local gel_abs_inference_valid = r(absorbed_target_inference_valid)
local gel_abs_fe_index = r(absorbing_fe_index)
local gel_abs_feclass_tol = r(fe_collinear_ss_ratio_tol)
matrix gel_abs_bbase = r(b_base)
matrix gel_abs_bfull = r(b_full)
matrix gel_abs_total = r(total)
matrix gel_abs_total_cov = r(total_cov)
matrix gel_abs_base_cov = r(base_cov)
matrix gel_abs_cov_tb = r(cov_total_bbase)
matrix gel_abs_mask = r(absorbed_mask)
matrix gel_abs_fe_ratio = r(x1_fe_collinear_ratio)
matrix gel_abs_near_mask = r(x1_near_collinear_mask)
assert r(converged) == 1
assert r(identity_gap) < 1e-10
assert r(n_obs_input) == 600
assert r(n_singletons_dropped) == 0
assert "`gel_abs_estimand'" == "absorbed_target_allocation"
assert "`gel_abs_identity'" == "exact_ols_constrained"
assert "`gel_abs_targets'" == "0"
assert "`gel_abs_target_names'" == "female"
assert "`gel_abs_bstatus'" == "imposed_zero estimated"
assert "`gel_abs_fstatus'" == "absorbed identified"
assert "`gel_abs_total_se_type'" == ///
    "target_exact_base_vce_mixed_components_conditional_only_diagnostic"
assert "`gel_abs_inference_status'" == "clustered_at_absorbing_fe"
assert `gel_abs_inference_valid' == 1
assert `gel_abs_fe_index' == 0
assert `gel_abs_feclass_tol' == 1e-9
assert gel_abs_mask[1, 1] == 1 & gel_abs_mask[1, 2] == 0
assert gel_abs_fe_ratio[1, 1] <= r(fe_collinear_ss_ratio_tol)
assert gel_abs_near_mask[1, 1] == 0
assert gel_abs_bfull[1, 1] == 0
assert abs(gel_abs_total[1, 1] - gel_abs_bbase[1, 1]) < 1e-10
forvalues cc = 1/3 {
    assert abs(gel_abs_cov_tb[1, `cc'] - gel_abs_base_cov[1, `cc']) < 1e-12
}
assert abs(gel_abs_total_cov[1, 1] - gel_abs_base_cov[1, 1]) < 1e-12
quietly regress y female x1, vce(cluster worker)
assert abs(gel_abs_total[1, 2] - _se[female]) < 1e-12

* Robust/crossed inference is deliberately retained for point-accounting but
* must warn loudly because the target is invariant at the worker FE level.
capture noisily xhdfegelbach y, x1(female x1) ///
    x2groups("observables = x2") fes(worker) absorbedtargets(female) vce(robust)
assert _rc == 0
assert r(absorbed_target_inference_valid) == 0
assert "`r(inference_status)'" == "warning_unsupported_vce_or_cluster"
assert strpos("`r(notes)'", "WARNING:") > 0
capture noisily xhdfegelbach y, x1(female x1) ///
    x2groups("observables = x2") fes(worker) absorbedtargets(female x1)
assert _rc != 0
capture noisily xhdfegelbach y, x1(female x1) ///
    x2groups("observables = x2") fes(worker) absorbedtargets(x2)
assert _rc == 198

* A severely near-collinear observed block can retain a valid identity while
* its split SE is tolerance/rounding sensitive; require an audible note.
gen double x2_near = x2 + 1.2e-6 * rnormal()
xhdfegelbach y, x1(x1) x2groups("near = x2 x2_near") fes(fe) ///
    vce(cluster) cluster(cluster) threads(4)
local gel_near_converged = r(converged)
local gel_near_notes "`r(notes)'"
assert `gel_near_converged' == 1
assert strpos("`gel_near_notes'", "x2 group 1 is severely ill-conditioned") > 0

* A focal column just above the FE-omission boundary stays in the standard
* estimand but carries an explicit per-column diagnostic and a loud warning.
gen double x1_near_fe = fe + 1e-3 * rnormal()
xhdfegelbach y, x1(x1_near_fe) x2groups("observables = x2") fes(fe)
matrix gel_x1_fe_ratio = r(x1_fe_collinear_ratio)
matrix gel_x1_near_mask = r(x1_near_collinear_mask)
assert gel_x1_fe_ratio[1, 1] > r(fe_collinear_ss_ratio_tol)
assert gel_x1_fe_ratio[1, 1] <= r(near_fe_warn_upper)
assert gel_x1_near_mask[1, 1] == 1
assert strpos("`r(notes)'", "near-FE-collinear focal") > 0

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

* Gelbach footnote-14 boundary: when both beta2_g and the corresponding
* auxiliary-loading row are zero, normal first-order inference is not regular.
* The command must keep the numerical diagnostic but flag it explicitly.
preserve
clear
set obs 256
gen long reg_i = _n - 1
gen double reg_x1 = cond(mod(floor(reg_i / 1), 2) == 0, 1, -1)
gen double reg_x2 = cond(mod(floor(reg_i / 2), 2) == 0, 1, -1)
gen double reg_e = cond(mod(floor(reg_i / 4), 2) == 0, 1, -1)
gen double reg_y0 = 1.2 * reg_x1 + 0.7 * reg_e

xhdfegelbach reg_y0, x1(reg_x1) x2groups("orthogonal = reg_x2")
matrix reg_b2 = r(beta2)
matrix reg_b2cov = r(beta2_cov)
matrix reg_aux = r(auxiliary_loadings)
matrix reg_ld = r(auxiliary_loading_diagnostics)
matrix reg_lp = r(auxiliary_loading_pvalue)
matrix reg_bw = r(beta2_wald)
matrix reg_grad = r(contribution_gradient_norm)
matrix reg_valid = r(regular_inference_valid)
matrix reg_status = r(regular_inference_status_code)
local reg_status_words "`r(regular_inference_status)'"
local reg_status1 : word 1 of `reg_status_words'
local reg_status2 : word 2 of `reg_status_words'
assert r(converged) == 1
assert r(regular_inference_all_valid) == 0
assert r(regularity_test_alpha) == .05
assert abs(reg_b2[1, 1]) < 2e-14
assert abs(reg_aux[1, 1]) < 2e-14
assert abs(reg_aux[2, 1]) < 2e-14
assert abs(reg_grad[1, 1]) < 2e-14
assert abs(reg_grad[2, 1]) < 2e-14
assert reg_bw[1, 3] > r(regularity_test_alpha)
assert reg_lp[1, 1] > r(regularity_test_alpha)
assert reg_lp[2, 1] > r(regularity_test_alpha)
assert reg_valid[1, 1] == 0 & reg_valid[2, 1] == 0
assert reg_status[1, 1] == 0 & reg_status[2, 1] == 0
assert "`reg_status1'" == "nonregular_not_ruled_out"
assert "`reg_status2'" == "nonregular_not_ruled_out"
assert strpos("`r(notes)'", ///
    "regular first-order delta-method inference is not established") > 0

gen double reg_loaded = .8 * reg_x1 + reg_x2
xhdfegelbach reg_y0, x1(reg_x1) ///
    x2groups("loading_signal = reg_loaded")
matrix reg_loaded_valid = r(regular_inference_valid)
matrix reg_loaded_status = r(regular_inference_status_code)
assert reg_loaded_valid[1, 1] == 1
assert reg_loaded_valid[2, 1] == 0
assert reg_loaded_status[1, 1] == 2
assert reg_loaded_status[2, 1] == 0

gen double reg_ybeta = reg_y0 + .5 * reg_x2
xhdfegelbach reg_ybeta, x1(reg_x1) ///
    x2groups("beta_signal = reg_x2")
matrix reg_beta_valid = r(regular_inference_valid)
matrix reg_beta_status = r(regular_inference_status_code)
assert r(regular_inference_all_valid) == 1
assert reg_beta_valid[1, 1] == 1 & reg_beta_valid[2, 1] == 1
assert reg_beta_status[1, 1] == 1 & reg_beta_status[2, 1] == 1
restore

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

* A saturated full model must fail through Stata's catchable return code,
* never terminate the host process or return a non-finite covariance.
preserve
clear
set obs 5
gen double sat_x1 = _n
gen double sat_z1 = (_n == 1)
gen double sat_z2 = (_n == 2)
gen double sat_z3 = (_n == 3)
gen double sat_y = 2 * sat_x1 + 3 * sat_z1 - sat_z2 + 0.5 * sat_z3
capture noisily xhdfegelbach sat_y, x1(sat_x1) ///
    x2groups("saturated = sat_z1 sat_z2 sat_z3")
assert _rc != 0
restore

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

* Gelbach retained-sample provenance is opt-in.  generate() maps the backend's
* zero-based positions back to the current Stata observations: missing outside
* the marked input, 0 for recursive singletons, and 1 for retained rows.
preserve
clear
set obs 27
gen long prov_i = _n
gen long prov_fe = cond(_n <= 24, ceil(_n / 4), 7)
gen double prov_x = (prov_i - 13) / 7
gen double prov_z = sin(1.3 * prov_i) + .2 * cos(.7 * prov_i)
gen double prov_y = .8 * prov_x + .4 * prov_z + ///
    mod(prov_fe, 3) / 5 + cos(.43 * prov_i)

quietly xhdfegelbach prov_y if prov_i <= 25, x1(prov_x) ///
    x2groups("observed = prov_z") fes(prov_fe)
matrix prov_plain_delta = r(delta)
matrix prov_plain_cov = r(cov)
assert r(sample_info_requested) == 0
assert "`r(sample_hash)'" == ""

quietly xhdfegelbach prov_y if prov_i <= 25, x1(prov_x) ///
    x2groups("observed = prov_z") fes(prov_fe) generate(prov_keep)
matrix prov_audit_delta = r(delta)
matrix prov_audit_cov = r(cov)
assert r(sample_info_requested) == 1
assert r(n_obs_input) == 25
assert r(n_obs) == 24
assert r(n_singletons_dropped) == 1
assert "`r(sample_hash)'" == "2d4dcd55f696e111"
assert "`r(sample_hash_algorithm)'" == "fnv1a64-le-v1"
assert "`r(sample_index_scope)'" == "marked_input_rows_zero_based"
assert "`r(sample_variable)'" == "prov_keep"
assert prov_keep == 1 if inrange(prov_i, 1, 24)
assert prov_keep == 0 if prov_i == 25
assert missing(prov_keep) if prov_i > 25
xcert_assert_matrix_close prov_plain_delta prov_audit_delta, ///
    tol(0) name("Gelbach sample audit delta invariance")
xcert_assert_matrix_close prov_plain_cov prov_audit_cov, ///
    tol(0) name("Gelbach sample audit covariance invariance")

quietly xhdfegelbach prov_y if prov_i <= 25, x1(prov_x) ///
    x2groups("observed = prov_z") fes(prov_fe) sampleaudit
assert r(sample_info_requested) == 1
assert "`r(sample_hash)'" == "2d4dcd55f696e111"
assert "`r(sample_variable)'" == ""
capture noisily xhdfegelbach prov_y if prov_i <= 25, x1(prov_x) ///
    x2groups("observed = prov_z") fes(prov_fe) generate(prov_keep)
assert _rc == 110
restore

* A loaded dispatcher from another checkout must never be reused silently.
local bound "$XHDFE_PLUGIN_PATH_INTERNAL"
global XHDFE_PLUGIN_PATH_INTERNAL "/xhdfe/release-gate/not-the-active-plugin"
capture noisily xhdfegelbach y, x1(x1) x2groups("observables = x2") fes(fe)
assert _rc == 498
global XHDFE_PLUGIN_PATH_INTERNAL "`bound'"

di as result "companion-command release gates passed"
