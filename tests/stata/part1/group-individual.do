noi di as text "xhdfe certification: group() and individual() fixed effects"

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

exit
