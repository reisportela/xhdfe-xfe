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
if (`"`e(group)'"' != "patent" | `"`e(individual)'"' != "inventor" | `"`e(aggregation)'"' != "mean") {
    di as error "group/individual metadata mismatch under aggregation(mean)"
    exit 9
}

xhdfe citations funding, absorb(year inventor) group(patent) i(inventor) ///
    aggregation(avg) tolerancemode(reghdfe-comparable) tolerance(1e-12) ///
    noheader notable nofootnote
xcert_store_estimates, prefix(xhd_ind_alias) scalars("N df_r df_m")

xcert_compare_estimates, refprefix(ref_ind) testprefix(xhd_ind_alias) scalars("N df_r df_m") ///
    btol(1e-8) vtol(1e-6) scaltol(1e-8)
if (`"`e(group)'"' != "patent" | `"`e(individual)'"' != "inventor" | `"`e(aggregation)'"' != "avg") {
    di as error "group/individual metadata mismatch under aggregation(avg)"
    exit 9
}

xhdfe citations funding, absorb(year inventor) group(patent) individual(inventor) ///
    aggregation(sum) tolerancemode(reghdfe-comparable) tolerance(1e-12) ///
    noheader notable nofootnote
if (`"`e(aggregation)'"' != "sum" | e(N) != 100) {
    di as error "group/individual aggregation(sum) metadata mismatch"
    exit 9
}
di as text "  scalar group_individual_sum_N: " e(N)

exit
