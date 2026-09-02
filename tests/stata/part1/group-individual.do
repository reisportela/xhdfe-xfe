noi di as text "xhdfe certification: group() and individual() fixed effects"

capture program drop xcert_set_mobility_method
program define xcert_set_mobility_method
    syntax using/, METHOD(string) [LEGACY]
    tempfile rewritten
    tempname profile_in profile_out
    file open `profile_in' using `"`using'"', read text
    file open `profile_out' using `"`rewritten'"', write text replace
    file read `profile_in' line
    while (r(eof) == 0) {
        local write_line 1
        if (`"`legacy'"' != "") {
            if (`"`line'"' == "xhdfe_mobility_profile_v2") {
                local line "xhdfe_mobility_profile_v1"
            }
            if (strpos(`"`line'"', "profile_kind=") == 1) {
                local write_line 0
            }
        }
        if (strpos(`"`line'"', "suggest_method=") == 1) {
            local line "suggest_method=`method'"
        }
        if (`write_line') file write `profile_out' `"`line'"' _n
        file read `profile_in' line
    }
    file close `profile_in'
    file close `profile_out'
    copy `"`rewritten'"' `"`using'"', replace
end

capture program drop xcert_read_mobility_profile
program define xcert_read_mobility_profile, rclass
    syntax using/
    tempname profile_in
    local header ""
    local kind ""
    local signature ""
    local method ""
    file open `profile_in' using `"`using'"', read text
    file read `profile_in' line
    if (r(eof) == 0) local header `"`line'"'
    while (r(eof) == 0) {
        if (strpos(`"`line'"', "profile_kind=") == 1) local kind `"`line'"'
        if (strpos(`"`line'"', "signature=") == 1) local signature `"`line'"'
        if (strpos(`"`line'"', "suggest_method=") == 1) local method `"`line'"'
        file read `profile_in' line
    }
    file close `profile_in'
    return local header `"`header'"'
    return local kind `"`kind'"'
    return local signature `"`signature'"'
    return local method `"`method'"'
end

clear
set obs 500
gen int patent = floor((_n - 1) / 5) + 1
gen int inventor = mod(_n - 1, 80) + 1
gen int year = mod(patent - 1, 10) + 1
gen double funding = mod(patent * 17, 31) / 10
gen double citations = 2 + .4 * funding + patent / 100 + year / 8
bys patent: replace citations = citations[1]
bys patent: replace funding = funding[1]

local scalars "N rmse rss tss mss r2 r2_a F df_r df_m"

preserve
    bys patent: keep if _n == 1
    xhdfe citations funding, absorb(year) keepsingletons ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
    xcert_store_estimates, prefix(ref_group) scalars("`scalars'")
restore

xhdfe citations funding, absorb(year) group(patent) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(xhd_group) scalars("`scalars'")

xcert_compare_estimates, refprefix(ref_group) testprefix(xhd_group) scalars("`scalars'") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)
if (`"`e(group)'"' != "patent") {
    di as error "e(group) should be patent"
    exit 9
}

xhdfe citations funding, absorb(year inventor) group(patent) individual(inventor) ///
    aggregation(mean) tolerancemode(reghdfe-comparable) tolerance(1e-12) ///
    noheader notable nofootnote
xcert_store_estimates, prefix(ref_ind) scalars("N df_r df_m")
if (e(converged) != 1 | e(precision_certified) != 1 | e(abs_residual_rel) > 8e-12) {
    di as error "group/individual mean absorption was not precision-certified"
    exit 9
}
if (`"`e(group)'"' != "patent" | `"`e(individual)'"' != "inventor" | `"`e(aggregation)'"' != "mean") {
    di as error "group/individual metadata mismatch under aggregation(mean)"
    exit 9
}

xhdfe citations funding, absorb(year inventor) group(patent) i(inventor) ///
    aggregation(avg) tolerancemode(reghdfe-comparable) tolerance(1e-12) ///
    noheader notable nofootnote
xcert_store_estimates, prefix(xhd_ind_alias) scalars("N df_r df_m")
if (e(converged) != 1 | e(precision_certified) != 1 | e(abs_residual_rel) > 8e-12) {
    di as error "group/individual avg absorption was not precision-certified"
    exit 9
}

xcert_compare_estimates, refprefix(ref_ind) testprefix(xhd_ind_alias) scalars("N df_r df_m") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)
if (`"`e(group)'"' != "patent" | `"`e(individual)'"' != "inventor" | `"`e(aggregation)'"' != "avg") {
    di as error "group/individual metadata mismatch under aggregation(avg)"
    exit 9
}

xhdfe citations funding, absorb(year inventor) group(patent) individual(inventor) ///
    aggregation(sum) tolerancemode(reghdfe-comparable) tolerance(1e-12) ///
    noheader notable nofootnote
if (e(converged) != 1 | e(precision_certified) != 1 | e(abs_residual_rel) > 8e-12) {
    di as error "group/individual sum absorption was not precision-certified"
    exit 9
}
if (`"`e(aggregation)'"' != "sum" | e(N) != 100) {
    di as error "group/individual aggregation(sum) metadata mismatch"
    exit 9
}
di as text "  scalar group_individual_sum_N: " e(N)

// Raw individual IDs may exceed int32 because categorical equality, not
// magnitude, defines the fixed effect. Keep dense internal codes compact.
gen double inventor_wide = .
replace inventor_wide = 2147483646       if mod(inventor - 1, 9) == 0
replace inventor_wide = 2147483647       if mod(inventor - 1, 9) == 1
replace inventor_wide = 2147483648       if mod(inventor - 1, 9) == 2
replace inventor_wide = 2147483649       if mod(inventor - 1, 9) == 3
replace inventor_wide = 10000000000      if mod(inventor - 1, 9) == 4
replace inventor_wide = 100000000000     if mod(inventor - 1, 9) == 5
replace inventor_wide = 1000000000000    if mod(inventor - 1, 9) == 6
replace inventor_wide = 9007199254740991 if mod(inventor - 1, 9) == 7
replace inventor_wide = 9007199254740992 if mod(inventor - 1, 9) == 8
egen long inventor_dense = group(inventor_wide)
quietly summarize inventor_dense, meanonly
assert r(min) == 1 & r(max) == 9

xhdfe citations funding, absorb(year inventor_wide) group(patent) ///
    individual(inventor_wide) aggregation(mean) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(ref_ind_wide_mean) scalars("N df_r df_m")

xhdfe citations funding, absorb(year inventor_dense) group(patent) ///
    individual(inventor_dense) aggregation(mean) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(xhd_ind_dense_mean) scalars("N df_r df_m")
xcert_compare_estimates, refprefix(ref_ind_wide_mean) testprefix(xhd_ind_dense_mean) ///
    scalars("N df_r df_m") btol(1e-8) vtol(1e-6) scaltol(1e-8)

xhdfe citations funding, absorb(year inventor_wide) group(patent) ///
    individual(inventor_wide) aggregation(sum) residuals(res_wide) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(ref_ind_wide_sum) scalars("N df_r df_m")
local wide_sum_N = e(N)
quietly count if !missing(res_wide)
assert r(N) == `wide_sum_N'
quietly count if e(sample)
assert r(N) == `wide_sum_N'

xhdfe citations funding, absorb(year inventor_dense) group(patent) ///
    individual(inventor_dense) aggregation(sum) residuals(res_dense) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xcert_store_estimates, prefix(xhd_ind_dense_sum) scalars("N df_r df_m")
xcert_compare_estimates, refprefix(ref_ind_wide_sum) testprefix(xhd_ind_dense_sum) ///
    scalars("N df_r df_m") btol(1e-8) vtol(1e-6) scaltol(1e-8)
assert missing(res_wide) == missing(res_dense)
assert abs(res_wide - res_dense) <= 1e-10 if !missing(res_wide)

preserve
    replace inventor_wide = -1 in 1
    capture noisily xhdfe citations funding, absorb(year inventor_wide) group(patent) ///
        individual(inventor_wide) aggregation(sum) noheader notable nofootnote
    if (_rc != 198) {
        di as error "negative individual identifier should fail with r(198)"
        exit 9
    }
restore

preserve
    replace inventor_wide = 1.5 in 1
    capture noisily xhdfe citations funding, absorb(year inventor_wide) group(patent) ///
        individual(inventor_wide) aggregation(sum) noheader notable nofootnote
    if (_rc != 198) {
        di as error "fractional individual identifier should fail with r(198)"
        exit 9
    }
restore

preserve
    replace inventor_wide = 9007199254740994 in 1
    capture noisily xhdfe citations funding, absorb(year inventor_wide) group(patent) ///
        individual(inventor_wide) aggregation(sum) noheader notable nofootnote
    if (_rc != 198) {
        di as error "individual identifier above 2^53 should fail with r(198)"
        exit 9
    }
restore

// A cycle has no singletons but an O(1/N^2) spectral gap. Plain MAP can reach
// maxiter() with a small backward residual while the regression coefficient is
// still wrong; automatic group/individual absorption must use joint LSMR.
clear
set obs 1000
gen long group = _n - 1
gen int ntrab = floor((_n - 1) / 2)
gen byte edge = mod(_n - 1, 2)
gen int p_ntrab = cond(edge == 0, ntrab, mod(ntrab + 1, 500))
gen double x = sin(group / 13) + cos(group / 29) + mod(group, 7) / 11
gen double y = .7 * x + sin(ntrab / 17) + cos(p_ntrab / 23) + .01 * sin(group / 5)

quietly reghdfe y x, absorb(p_ntrab ntrab) group(group) ///
    individual(p_ntrab) aggregation(sum) keepsingletons tolerance(1e-8) ///
    residuals(res_ref)
scalar cycle_ref_b = _b[x]
scalar cycle_ref_cons = _b[_cons]
scalar cycle_ref_rss = e(rss)

gen double x_easy = mod(ntrab, 7)
gen double y_easy = .7 * x_easy + sin(ntrab / 17)
tempfile cycle_profile cycle_v1_profile
quietly xhdfe y_easy x_easy, absorb(p_ntrab ntrab) group(group) ///
    individual(p_ntrab) aggregation(sum) keepsingletons tolerance(1e-8) ///
    maxiter(1000) numthreads(2) gpubackend(cpu) ///
    abscachemode(off) mobfile(`"`cycle_profile'"') noheader notable nofootnote
xcert_set_mobility_method using `"`cycle_profile'"', method(gauss-seidel)
copy `"`cycle_profile'"' `"`cycle_v1_profile'"'
xcert_set_mobility_method using `"`cycle_v1_profile'"', ///
    method(gauss-seidel) legacy
xcert_read_mobility_profile using `"`cycle_profile'"'
assert `"`r(header)'"' == "xhdfe_mobility_profile_v2"
assert `"`r(kind)'"' == "profile_kind=group_individual"
assert `"`r(method)'"' == "suggest_method=gauss-seidel"
xcert_read_mobility_profile using `"`cycle_v1_profile'"'
assert `"`r(header)'"' == "xhdfe_mobility_profile_v1"
assert `"`r(kind)'"' == ""
assert `"`r(method)'"' == "suggest_method=gauss-seidel"

quietly xhdfe y x, absorb(p_ntrab ntrab) group(group) ///
    individual(p_ntrab) aggregation(sum) keepsingletons tolerance(1e-8) ///
    residuals(res_v2_profile) maxiter(1000) numthreads(2) gpubackend(cpu) ///
    abscachemode(off) mobfile(`"`cycle_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
assert e(absorption_method_used) == 5
matrix cycle_v2_b = e(b)
matrix cycle_v2_V = e(V)
scalar cycle_v2_rss = e(rss)
scalar cycle_v2_iter = e(iterations)
xcert_read_mobility_profile using `"`cycle_profile'"'
assert `"`r(method)'"' == "suggest_method=lsmr"

quietly xhdfe y x, absorb(p_ntrab ntrab) group(group) ///
    individual(p_ntrab) aggregation(sum) keepsingletons tolerance(1e-8) ///
    residuals(res_v1_profile) maxiter(1000) numthreads(2) gpubackend(cpu) ///
    abscachemode(off) mobfile(`"`cycle_v1_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
assert e(absorption_method_used) == 5
matrix cycle_v1_b = e(b)
matrix cycle_v1_V = e(V)
scalar cycle_v1_rss = e(rss)
scalar cycle_v1_iter = e(iterations)
xcert_read_mobility_profile using `"`cycle_v1_profile'"'
assert `"`r(header)'"' == "xhdfe_mobility_profile_v2"
assert `"`r(kind)'"' == "profile_kind=group_individual"
assert `"`r(method)'"' == "suggest_method=lsmr"

quietly xhdfe y x, absorb(p_ntrab ntrab) group(group) ///
    individual(p_ntrab) aggregation(sum) keepsingletons tolerance(1e-8) ///
    residuals(res_xhdfe) maxiter(1000) numthreads(2) gpubackend(cpu) ///
    abscachemode(off) noheader notable nofootnote
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(absorption_method_used) == 5
assert e(iterations) < 100000
assert e(threads_requested) == 2
assert e(threads_used) == 2
assert abs(_b[x] - cycle_ref_b) <= 1e-10
assert abs(_b[_cons] - cycle_ref_cons) <= 1e-10
assert abs(e(rss) - cycle_ref_rss) <= 1e-16
matrix cycle_clean_b = e(b)
matrix cycle_clean_V = e(V)
scalar cycle_clean_rss = e(rss)
scalar cycle_clean_iter = e(iterations)
mata: st_numscalar("cycle_v2_bdiff", max(abs(st_matrix("cycle_v2_b") :- st_matrix("cycle_clean_b"))))
mata: st_numscalar("cycle_v2_Vdiff", max(abs(st_matrix("cycle_v2_V") :- st_matrix("cycle_clean_V"))))
mata: st_numscalar("cycle_v1_bdiff", max(abs(st_matrix("cycle_v1_b") :- st_matrix("cycle_clean_b"))))
mata: st_numscalar("cycle_v1_Vdiff", max(abs(st_matrix("cycle_v1_V") :- st_matrix("cycle_clean_V"))))
assert cycle_v2_bdiff == 0 & cycle_v2_Vdiff == 0
assert cycle_v1_bdiff == 0 & cycle_v1_Vdiff == 0
assert cycle_v2_rss == cycle_clean_rss & cycle_v1_rss == cycle_clean_rss
assert cycle_v2_iter == cycle_clean_iter & cycle_v1_iter == cycle_clean_iter
assert missing(res_v2_profile) == missing(res_xhdfe)
assert missing(res_v1_profile) == missing(res_xhdfe)
assert res_v2_profile == res_xhdfe if !missing(res_xhdfe)
assert res_v1_profile == res_xhdfe if !missing(res_xhdfe)
tempvar residual_diff
gen double `residual_diff' = abs(res_xhdfe - res_ref)
quietly summarize `residual_diff', meanonly
assert r(max) <= 1e-10

capture noisily xhdfe y x, absorb(p_ntrab ntrab) group(group) ///
    individual(p_ntrab) aggregation(sum) keepsingletons tolerance(1e-8) ///
    absorptionmethod(gauss-seidel) maxiter(1000) ///
    noheader notable nofootnote
if (_rc != 498) {
    di as error "group/individual Gauss-Seidel must fail closed at maxiter()"
    exit 9
}

// Unequal group sizes make aggregation(mean) and aggregation(sum) distinct.
// Exercise the multi-incidence adjoint, analytic weights, and explicit LSMR
// against an explicit group-level dummy-design oracle for both estimands.
clear
set obs 300
gen long group = _n - 1
gen int standard_fe = mod(group * 7 + floor(group / 9), 23)
gen byte group_size = 2 + mod(group, 3)
gen double x = sin(group / 17) + cos(group / 31) + mod(group, 11) / 13
gen double aw = .5 + mod(group, 9) / 7
gen byte fw = 1 + mod(group, 3)
expand group_size
bysort group: gen byte member = _n - 1
gen int individual = mod(group * 17 + member * 29 + floor(group / 11), 83)
gen double individual_signal = sin(individual / 19)
bysort group: egen double individual_sum = total(individual_signal)
gen double y = .63 * x + sin(standard_fe / 7) + individual_sum + .02 * cos(group / 13)

foreach agg in mean sum {
    preserve
        quietly tabulate individual, generate(__inc_)
        collapse (sum) __inc_* (firstnm) y x standard_fe aw fw group_size, by(group)
        if ("`agg'" == "mean") {
            foreach v of varlist __inc_* {
                quietly replace `v' = `v' / group_size
            }
        }
        quietly regress y x i.standard_fe __inc_* [aw=aw]
        scalar varying_ref_b_`agg' = _b[x]
        scalar varying_ref_rss_`agg' = e(rss)
        predict double oracle_residual, residuals
        keep group oracle_residual
        tempfile oracle
        save "`oracle'", replace
    restore
    merge m:1 group using "`oracle'", assert(match) nogen
    rename oracle_residual res_ref_`agg'

    quietly xhdfe y x [aw=aw], absorb(standard_fe individual) group(group) ///
        individual(individual) aggregation(`agg') keepsingletons ///
        tolerancemode(strict-residual) tolerance(1e-10) ///
        absorptionmethod(lsmr) numthreads(2) ///
        residuals(res_xhdfe_`agg') noheader notable nofootnote
    assert e(converged) == 1
    assert e(precision_certified) == 1
    assert e(absorption_method_used) == 5
    assert abs(_b[x] - varying_ref_b_`agg') <= 1e-9
    assert abs(e(rss) - varying_ref_rss_`agg') <= 1e-9 * max(1, varying_ref_rss_`agg')
    tempvar varying_diff
    gen double `varying_diff' = abs(res_xhdfe_`agg' - res_ref_`agg')
    quietly summarize `varying_diff', meanonly
    assert r(max) <= 1e-8

    scalar varying_xhdfe_b_`agg' = _b[x]
    scalar varying_xhdfe_cons_`agg' = _b[_cons]
    scalar varying_xhdfe_rss_`agg' = e(rss)
    quietly xhdfe y x [aw=aw], absorb(standard_fe individual) group(group) ///
        individual(individual) aggregation(`agg') keepsingletons ///
        tolerancemode(strict-residual) tolerance(1e-10) ///
        absorptionmethod(lsmr) numthreads(1) ///
        residuals(res_xhdfe_1t_`agg') noheader notable nofootnote
    assert e(threads_used) == 1
    assert abs(_b[x] - varying_xhdfe_b_`agg') <= 1e-14
    assert abs(_b[_cons] - varying_xhdfe_cons_`agg') <= 1e-14
    assert abs(e(rss) - varying_xhdfe_rss_`agg') <= 1e-14
    tempvar thread_diff
    gen double `thread_diff' = abs(res_xhdfe_1t_`agg' - res_xhdfe_`agg')
    quietly summarize `thread_diff', meanonly
    assert r(max) <= 1e-14
}
assert abs(varying_ref_b_mean - varying_ref_b_sum) > 1e-6

preserve
    quietly tabulate individual, generate(__inc_)
    collapse (sum) __inc_* (firstnm) y x standard_fe fw, by(group)
    quietly regress y x i.standard_fe __inc_* [fw=fw]
    scalar varying_fw_ref_b = _b[x]
    scalar varying_fw_ref_rss = e(rss)
    scalar varying_fw_ref_N = e(N)
restore
quietly xhdfe y x [fw=fw], absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(sum) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
assert abs(_b[x] - varying_fw_ref_b) <= 1e-9
assert abs(e(rss) - varying_fw_ref_rss) <= 1e-9 * max(1, varying_fw_ref_rss)
assert e(N) == varying_fw_ref_N

// A grouped mobility hint is exact-context: the second run may reuse plain
// LSMR, while graph, aggregation, weight, or fweight changes must miss.
tempfile grouped_profile weighted_profile
quietly xhdfe y x, absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(mean) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`grouped_profile'"') ///
    noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
assert e(absorption_method_used) == 5
tempname profile_check
file open `profile_check' using `"`grouped_profile'"', read text
file read `profile_check' profile_line
assert `"`profile_line'"' == "xhdfe_mobility_profile_v2"
file read `profile_check' profile_line
assert `"`profile_line'"' == "profile_kind=group_individual"
file close `profile_check'

xcert_set_mobility_method using `"`grouped_profile'"', method(lsmr)
quietly xhdfe y x, absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(mean) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`grouped_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
assert e(absorption_method_used) == 5

xcert_set_mobility_method using `"`grouped_profile'"', method(auto)
quietly xhdfe y x, absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(mean) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`grouped_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1

xcert_set_mobility_method using `"`grouped_profile'"', method(gauss-seidel)
xcert_read_mobility_profile using `"`grouped_profile'"'
local base_signature `"`r(signature)'"'
gen int individual_alt = mod(group * 19 + member * 31 + floor(group / 13), 83)
gen double x_extra = cos(group / 23) + mod(group, 5) / 7
quietly xhdfe y x x_extra, absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(mean) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`grouped_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
xcert_read_mobility_profile using `"`grouped_profile'"'
assert `"`r(header)'"' == "xhdfe_mobility_profile_v2"
assert `"`r(kind)'"' == "profile_kind=group_individual"
assert `"`r(signature)'"' != `"`base_signature'"'
quietly xhdfe y x, absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(mean) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(1) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`grouped_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
quietly xhdfe y x, absorb(standard_fe individual_alt) group(group) ///
    individual(individual_alt) aggregation(mean) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`grouped_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
quietly xhdfe y x, absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(sum) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`grouped_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
assert e(absorption_method_used) == 5
quietly xhdfe y x [aw=aw], absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(mean) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`grouped_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1

quietly xhdfe y x [aw=fw], absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(mean) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`weighted_profile'"') ///
    noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
assert e(absorption_method_used) == 5
xcert_set_mobility_method using `"`weighted_profile'"', method(gauss-seidel)
xcert_read_mobility_profile using `"`weighted_profile'"'
local weighted_signature `"`r(signature)'"'
quietly xhdfe y x [fw=fw], absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(mean) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`weighted_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
scalar varying_fw_mean_b = _b[x]
scalar varying_fw_mean_rss = e(rss)
xcert_read_mobility_profile using `"`weighted_profile'"'
assert `"`r(signature)'"' != `"`weighted_signature'"'

// Unscoped v1 profiles are ignored in both standard and grouped paths.
tempfile standard_profile
quietly xhdfe y x, absorb(standard_fe individual) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`standard_profile'"') ///
    noheader notable nofootnote
xcert_set_mobility_method using `"`standard_profile'"', ///
    method(jacobi) legacy
quietly xhdfe y x, absorb(standard_fe individual) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`standard_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
xcert_read_mobility_profile using `"`standard_profile'"'
assert `"`r(header)'"' == "xhdfe_mobility_profile_v2"
assert `"`r(kind)'"' == "profile_kind=standard"

// Version-2 scope mismatches are safe misses in both directions.
xcert_set_mobility_method using `"`standard_profile'"', method(gauss-seidel)
quietly xhdfe y x, absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(mean) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`standard_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
assert e(absorption_method_used) == 5
xcert_read_mobility_profile using `"`standard_profile'"'
assert `"`r(kind)'"' == "profile_kind=group_individual"

xcert_set_mobility_method using `"`grouped_profile'"', method(jacobi)
quietly xhdfe y x, absorb(standard_fe individual) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`grouped_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
assert e(absorption_method_used) != 3
xcert_read_mobility_profile using `"`grouped_profile'"'
assert `"`r(kind)'"' == "profile_kind=standard"

xcert_set_mobility_method using `"`grouped_profile'"', ///
    method(gauss-seidel) legacy
quietly xhdfe y x, absorb(standard_fe individual) group(group) ///
    individual(individual) aggregation(mean) keepsingletons ///
    tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
    gpubackend(cpu) abscachemode(off) mobfile(`"`grouped_profile'"') noheader notable nofootnote
assert e(converged) == 1 & e(precision_certified) == 1
xcert_read_mobility_profile using `"`grouped_profile'"'
assert `"`r(header)'"' == "xhdfe_mobility_profile_v2"
assert `"`r(kind)'"' == "profile_kind=group_individual"

if (`"`: environment XHDFE_TEST_CUDA_PROFILE'"' == "1") {
    tempfile cuda_profile
    quietly xhdfe y x, absorb(standard_fe individual) group(group) ///
        individual(individual) aggregation(mean) keepsingletons ///
        tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
        gpubackend(cuda) abscachemode(off) mobfile(`"`grouped_profile'"') ///
        noheader notable nofootnote
    assert e(converged) == 1 & e(precision_certified) == 1
    assert e(gpu_used) == 1 & e(gpu_status_code) == 1
    assert e(absorption_method_used) == 5
    xcert_read_mobility_profile using `"`grouped_profile'"'
    assert `"`r(method)'"' == "suggest_method=auto"

    quietly xhdfe y x, absorb(standard_fe individual) group(group) ///
        individual(individual) aggregation(mean) keepsingletons ///
        tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
        gpubackend(cuda) abscachemode(off) mobfile(`"`cuda_profile'"') ///
        noheader notable nofootnote
    assert e(converged) == 1 & e(precision_certified) == 1
    assert e(gpu_used) == 1 & e(gpu_status_code) == 1
    assert e(absorption_method_used) == 5
    xcert_read_mobility_profile using `"`cuda_profile'"'
    assert `"`r(method)'"' == "suggest_method=auto"

    foreach agg in mean sum {
        quietly xhdfe y x [aw=aw], absorb(standard_fe individual) group(group) ///
            individual(individual) aggregation(`agg') keepsingletons ///
            tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
            gpubackend(cuda) abscachemode(off) noheader notable nofootnote
        assert e(converged) == 1 & e(precision_certified) == 1
        assert e(gpu_used) == 1 & e(gpu_status_code) == 1
        assert abs(_b[x] - varying_ref_b_`agg') <= 1e-8
        assert abs(e(rss) - varying_ref_rss_`agg') <= ///
            1e-8 * max(1, varying_ref_rss_`agg')
    }

    quietly xhdfe y x [fw=fw], absorb(standard_fe individual) group(group) ///
        individual(individual) aggregation(mean) keepsingletons ///
        tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
        gpubackend(cuda) abscachemode(off) noheader notable nofootnote
    assert e(converged) == 1 & e(precision_certified) == 1
    assert e(gpu_used) == 1 & e(gpu_status_code) == 1
    assert abs(_b[x] - varying_fw_mean_b) <= 1e-8
    assert abs(e(rss) - varying_fw_mean_rss) <= ///
        1e-8 * max(1, varying_fw_mean_rss)

    quietly xhdfe y x [fw=fw], absorb(standard_fe individual) group(group) ///
        individual(individual) aggregation(sum) keepsingletons ///
        tolerancemode(strict-residual) tolerance(1e-10) numthreads(2) ///
        gpubackend(cuda) abscachemode(off) noheader notable nofootnote
    assert e(converged) == 1 & e(precision_certified) == 1
    assert e(gpu_used) == 1 & e(gpu_status_code) == 1
    assert abs(_b[x] - varying_fw_ref_b) <= 1e-8
    assert abs(e(rss) - varying_fw_ref_rss) <= ///
        1e-8 * max(1, varying_fw_ref_rss)

    foreach method in gauss-seidel symmetric-gauss-seidel {
        local expected_method = cond("`method'" == "gauss-seidel", 1, 2)
        quietly xhdfe y x, absorb(standard_fe individual) group(group) ///
            individual(individual) aggregation(mean) keepsingletons ///
            absorptionmethod(`method') tolerancemode(strict-residual) ///
            tolerance(1e-10) maxiter(20000) numthreads(2) ///
            gpubackend(cuda) abscachemode(off) noheader notable nofootnote
        assert e(converged) == 1 & e(precision_certified) == 1
        assert e(gpu_used) == 1 & e(gpu_status_code) == 1
        assert e(absorption_method_used) == `expected_method'
    }

    capture noisily xhdfe y x, absorb(standard_fe individual) group(group) ///
        individual(individual) aggregation(mean) keepsingletons ///
        absorptionmethod(lsmr) gpubackend(cuda) abscachemode(off) ///
        noheader notable nofootnote
    if (_rc != 198) {
        di as error "explicit group/individual LSMR with CUDA should fail with r(198)"
        exit 9
    }
}

foreach method in jacobi schwarz mlsmr auto-mlsmr {
    capture noisily xhdfe y x, absorb(standard_fe individual) group(group) ///
        individual(individual) aggregation(sum) keepsingletons ///
        absorptionmethod(`method') noheader notable nofootnote
    if (_rc != 198) {
        di as error "unsupported group/individual method `method' should fail with r(198)"
        exit 9
    }
}

// Convergence reached and certified on the final allowed sweep is valid.
clear
input double(y x) byte(group individual)
-1 -1 1 1
 1  1 2 1
end
xhdfe y x, absorb(individual) group(group) individual(individual) ///
    aggregation(sum) keepsingletons absorptionmethod(gauss-seidel) ///
    maxiter(1) noheader notable nofootnote
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(iterations) == 1

capture program drop xcert_set_mobility_method
capture program drop xcert_read_mobility_profile

exit
