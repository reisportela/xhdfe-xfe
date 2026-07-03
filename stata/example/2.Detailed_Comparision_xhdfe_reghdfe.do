// Run tests `xhdfe.ado' vs. `reghdfe.ado'
// Miguel Portela U Minho & BPLIM
// December 26, 2025

clear all
discard
set more off
set rmsg on

//net install xhdfe,from("/PATH/TO/xhdfe/stata") replace
adopath + "/PATH/TO/xhdfe/stata"
which xhdfe
*h xhdfe

cd /PATH/TO/xhdfe/stata/example

capture log close
log using output/Detailed_Comparision_xhdfe_reghdfe.txt, text replace

timer on 1

global SCALAR_TOL 1e-7
global B_TOL 1e-7
global SE_TOL 1e-7

scalar SCALAR_OK = 0
scalar SCALAR_FAIL = 0
scalar COEF_OK = 0
scalar COEF_FAIL = 0

program define _store_escalars, rclass
    syntax, prefix(name)
    local scalars "`e(scalars)'"
    if "`scalars'" == "" {
        local scalars "N N_full num_singletons r2 r2_within r2_a r2_a_within rss tss tss_within mss rmse sigma2 df_r df_m df_a df_a_levels df_a_exact df_a_nested df_a_redundant df_a_initial df_r_unadj ll ll_0 F p"
        di as text "  note: e(scalars) empty; using fallback list"
    }
    foreach s of local scalars {
        capture scalar `prefix'`s' = e(`s')
        if (_rc) {
            scalar `prefix'`s' = .
        }
    }
    return local scalars "`scalars'"
end

program define _drop_escalars
    syntax, prefix(name) scalars(string)
    foreach s of local scalars {
        capture scalar drop `prefix'`s'
    }
end

program define _compare_escalars
    syntax, testname(string) xscalars(string) rscalars(string) tol(real)
    local common : list xscalars & rscalars
    local x_only : list xscalars - rscalars
    local r_only : list rscalars - xscalars
    local ok = 0
    local fail = 0
    local n_common : word count `common'
    local n_xonly : word count `x_only'
    local n_ronly : word count `r_only'

    di as text "Stored scalars comparison: `testname' (tol=`tol')"
    di as text "  common=`n_common' xhdfe-only=`n_xonly' reghdfe-only=`n_ronly'"
    foreach s of local common {
        tempname diff
        if (missing(x_`s') & missing(r_`s')) {
            di as text "  [OK] `s' xhdfe=. reghdfe=. diff=."
            post $POST_SUMMARY ("`testname'") ("`s'") (x_`s') (r_`s') (.) ("OK")
            local ok = `ok' + 1
        }
        else if (missing(x_`s') | missing(r_`s')) {
            di as error "  [NOT OK] `s' xhdfe=" %21.9g x_`s' " reghdfe=" %21.9g r_`s'
            post $POST_SUMMARY ("`testname'") ("`s'") (x_`s') (r_`s') (.) ("NOT OK")
            local fail = `fail' + 1
        }
        else {
            scalar `diff' = abs(x_`s' - r_`s')
            if (`diff' <= `tol') {
                di as text "  [OK] `s' xhdfe=" %21.9g x_`s' " reghdfe=" %21.9g r_`s' " diff=" %21.9g `diff'
                post $POST_SUMMARY ("`testname'") ("`s'") (x_`s') (r_`s') (`diff') ("OK")
                local ok = `ok' + 1
            }
            else {
                di as error "  [NOT OK] `s' xhdfe=" %21.9g x_`s' " reghdfe=" %21.9g r_`s' " diff=" %21.9g `diff'
                post $POST_SUMMARY ("`testname'") ("`s'") (x_`s') (r_`s') (`diff') ("NOT OK")
                local fail = `fail' + 1
            }
        }
    }
    if ("`x_only'" != "") {
        di as error "  [NOT OK] xhdfe-only scalars: `x_only'"
        local fail = `fail' + `n_xonly'
        foreach s of local x_only {
            post $POST_SUMMARY ("`testname'") ("`s'") (x_`s') (.) (.) ("NOT OK")
        }
    }
    if ("`r_only'" != "") {
        di as error "  [NOT OK] reghdfe-only scalars: `r_only'"
        local fail = `fail' + `n_ronly'
        foreach s of local r_only {
            post $POST_SUMMARY ("`testname'") ("`s'") (.) (r_`s') (.) ("NOT OK")
        }
    }
    di as text "  scalar results: ok=`ok' not_ok=`fail'"
    scalar SCALAR_OK = SCALAR_OK + `ok'
    scalar SCALAR_FAIL = SCALAR_FAIL + `fail'
end

program define _compare_coeffs
    syntax, testname(string) xb(name) xv(name) rb(name) rv(name) btol(real) setol(real)
    local xcols : colnames `xb'
    local rcols : colnames `rb'
    local common : list xcols & rcols
    local x_only : list xcols - rcols
    local r_only : list rcols - xcols
    local ok = 0
    local fail = 0
    local n_xonly : word count `x_only'
    local n_ronly : word count `r_only'

    di as text "Coefficient/SE comparison: `testname' (b_tol=`btol' se_tol=`setol')"
    foreach v of local common {
        local xi = colnumb(`xb', "`v'")
        local ri = colnumb(`rb', "`v'")
        tempname xbval rbval xse rse bdiff sediff
        scalar `xbval' = `xb'[1,`xi']
        scalar `rbval' = `rb'[1,`ri']
        scalar `xse' = sqrt(`xv'[`xi',`xi'])
        scalar `rse' = sqrt(`rv'[`ri',`ri'])
        scalar `bdiff' = abs(`xbval' - `rbval')
        scalar `sediff' = abs(`xse' - `rse')
        if (`bdiff' <= `btol' & `sediff' <= `setol') {
            di as text "  [OK] `v' b_diff=" %21.9g `bdiff' " se_diff=" %21.9g `sediff'
            local ok = `ok' + 1
        }
        else {
            di as error "  [NOT OK] `v' x_b=" %21.9g `xbval' " r_b=" %21.9g `rbval' " b_diff=" %21.9g `bdiff' " x_se=" %21.9g `xse' " r_se=" %21.9g `rse' " se_diff=" %21.9g `sediff'
            local fail = `fail' + 1
        }
    }
    if ("`x_only'" != "") {
        di as error "  [NOT OK] xhdfe-only coefficients: `x_only'"
        local fail = `fail' + `n_xonly'
    }
    if ("`r_only'" != "") {
        di as error "  [NOT OK] reghdfe-only coefficients: `r_only'"
        local fail = `fail' + `n_ronly'
    }
    di as text "  coefficient results: ok=`ok' not_ok=`fail'"
    scalar COEF_OK = COEF_OK + `ok'
    scalar COEF_FAIL = COEF_FAIL + `fail'
end

program define run_compare
    syntax, testname(string) xcmd(string) rcmd(string)
    di as text ""
    di as text "== Test: `testname' =="
    quietly `xcmd'
    matrix x_b = e(b)
    matrix x_V = e(V)
    _store_escalars, prefix(x_)
    local x_scalars "`r(scalars)'"

    quietly `rcmd'
    matrix r_b = e(b)
    matrix r_V = e(V)
    _store_escalars, prefix(r_)
    local r_scalars "`r(scalars)'"

    _compare_escalars, testname("`testname'") xscalars("`x_scalars'") rscalars("`r_scalars'") tol($SCALAR_TOL)
    _compare_coeffs, testname("`testname'") xb(x_b) xv(x_V) rb(r_b) rv(r_V) btol($B_TOL) setol($SE_TOL)

    _drop_escalars, prefix(x_) scalars("`x_scalars'")
    _drop_escalars, prefix(r_) scalars("`r_scalars'")
end

tempname post_summary
postfile `post_summary' str20 test str32 stat double xhdfe double reghdfe double diff str6 status using "output/compare_scalar_table.dta", replace
global POST_SUMMARY `post_summary'

// 1. Base data snapshot
webuse nlswork, clear
des
sum
xtdes
xtsum
tempfile base_data
save `base_data', replace

// 2. Comparison tests
use `base_data', clear
run_compare, testname("ols") ///
    xcmd("xhdfe ln_wage c.ttl_exp##c.ttl_exp c_city union, absorb(idcode ind_code occ_code year) numthreads(32) tol(1e-8)") ///
    rcmd("reghdfe ln_wage c.ttl_exp##c.ttl_exp c_city union, absorb(idcode ind_code occ_code year) tol(1e-8)")

use `base_data', clear
run_compare, testname("robust") ///
    xcmd("xhdfe ln_wage c.ttl_exp##c.ttl_exp c_city union, absorb(idcode ind_code occ_code year) vce(robust) numthreads(32) tol(1e-8)") ///
    rcmd("reghdfe ln_wage c.ttl_exp##c.ttl_exp c_city union, absorb(idcode ind_code occ_code year) vce(robust) tol(1e-8)")

use `base_data', clear
run_compare, testname("cluster_ind_code") ///
    xcmd("xhdfe ln_wage c.ttl_exp##c.ttl_exp c_city union, absorb(idcode ind_code occ_code year) vce(cluster ind_code) numthreads(32) tol(1e-8)") ///
    rcmd("reghdfe ln_wage c.ttl_exp##c.ttl_exp c_city union, absorb(idcode ind_code occ_code year) vce(cluster ind_code) tol(1e-8)")

use `base_data', clear
run_compare, testname("cluster_savefe") ///
    xcmd("xhdfe ln_wage c.ttl_exp##c.ttl_exp c_city union, absorb(idcode ind_code occ_code year) savefes(fe_) vce(cluster ind_code) numthreads(32) tol(1e-8)") ///
    rcmd("reghdfe ln_wage c.ttl_exp##c.ttl_exp c_city union, absorb(idcode ind_code occ_code year, savefe) vce(cluster ind_code) tol(1e-8)")

// 3. FE recovery comparison
keep if e(sample)
preserve
    ren fe_idcode workerfe_xhdfe
    ren fe_ind_code indfe_xhdfe
    ren fe_occ_code occupfe_xhdfe
    ren fe_year yearfe_xhdfe
    keep idcode ind_code occ_code year workerfe_xhdfe indfe_xhdfe occupfe_xhdfe yearfe_xhdfe
    order idcode ind_code occ_code year workerfe_xhdfe indfe_xhdfe occupfe_xhdfe yearfe_xhdfe
    compress
    sort idcode ind_code occ_code year
    save data/xhdfe_fes, replace
restore

preserve
    ren __hdfe1__ workerfe_reghdfe
    ren __hdfe2__ indfe_reghdfe
    ren __hdfe3__ occupfe_reghdfe
    ren __hdfe4__ yearfe_reghdfe
    keep idcode ind_code occ_code year workerfe_reghdfe indfe_reghdfe occupfe_reghdfe yearfe_reghdfe
    order idcode ind_code occ_code year workerfe_reghdfe indfe_reghdfe occupfe_reghdfe yearfe_reghdfe
    compress
    sort idcode ind_code occ_code year
    save data/reghdfe_fes, replace
restore

use data/xhdfe_fes, clear
merge 1:1 idcode ind_code occ_code year using data/reghdfe_fes
drop _merge

gen diff_workerfe = workerfe_xhdfe - workerfe_reghdfe
gen diff_indfe = indfe_xhdfe - indfe_reghdfe
gen diff_occupfe = occupfe_xhdfe - occupfe_reghdfe
gen diff_yearfe = yearfe_xhdfe - yearfe_reghdfe

compress

format %21.8f diff_workerfe diff_indfe diff_occupfe diff_yearfe

sum diff_workerfe diff_indfe diff_occupfe diff_yearfe, detail

tempname max_diff
gen abs_workerfe = abs(diff_workerfe)
sum abs_workerfe, meanonly
scalar `max_diff' = r(max)
if (`max_diff' <= $SCALAR_TOL) {
    di as text "FE diff workerfe: [OK] max_abs=" %21.9g `max_diff'
}
else {
    di as error "FE diff workerfe: [NOT OK] max_abs=" %21.9g `max_diff'
}
drop abs_workerfe

gen abs_indfe = abs(diff_indfe)
sum abs_indfe, meanonly
scalar `max_diff' = r(max)
if (`max_diff' <= $SCALAR_TOL) {
    di as text "FE diff indfe: [OK] max_abs=" %21.9g `max_diff'
}
else {
    di as error "FE diff indfe: [NOT OK] max_abs=" %21.9g `max_diff'
}
drop abs_indfe

gen abs_occupfe = abs(diff_occupfe)
sum abs_occupfe, meanonly
scalar `max_diff' = r(max)
if (`max_diff' <= $SCALAR_TOL) {
    di as text "FE diff occupfe: [OK] max_abs=" %21.9g `max_diff'
}
else {
    di as error "FE diff occupfe: [NOT OK] max_abs=" %21.9g `max_diff'
}
drop abs_occupfe

gen abs_yearfe = abs(diff_yearfe)
sum abs_yearfe, meanonly
scalar `max_diff' = r(max)
if (`max_diff' <= $SCALAR_TOL) {
    di as text "FE diff yearfe: [OK] max_abs=" %21.9g `max_diff'
}
else {
    di as error "FE diff yearfe: [NOT OK] max_abs=" %21.9g `max_diff'
}
drop abs_yearfe

reg workerfe_xhdfe workerfe_reghdfe
reg indfe_xhdfe indfe_reghdfe
reg occupfe_xhdfe occupfe_reghdfe
reg yearfe_xhdfe yearfe_reghdfe

di as text ""
di as text "== Overall comparison summary =="
di as text "Scalars: ok=" %12.0g SCALAR_OK " not_ok=" %12.0g SCALAR_FAIL
di as text "Coefficients: ok=" %12.0g COEF_OK " not_ok=" %12.0g COEF_FAIL

postclose $POST_SUMMARY
macro drop POST_SUMMARY

di as text ""
di as text "== Scalar comparison summary table =="
use "output/compare_scalar_table.dta", clear
sort test stat
format xhdfe reghdfe diff %21.9g
list test stat xhdfe reghdfe diff status, sepby(test) noobs

// --- //

webuse nlswork, clear
clonevar tenure_alt = tenure
xhdfe ln_wage grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(idcode ind_code occ_code year)
xhdfe ln_wage grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(idcode ind_code occ_code year) vce(robust)
xhdfe ln_wage grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(idcode ind_code occ_code year) vce(cluster ind_code)
eret li

webuse nlswork, clear
clonevar tenure_alt = tenure
reghdfe ln_wage grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(idcode ind_code occ_code year)
reghdfe ln_wage grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(idcode ind_code occ_code year) vce(robust)
reghdfe ln_wage grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(idcode ind_code occ_code year) vce(cluster ind_code)
eret li


webuse nlswork, clear
clonevar tenure_alt = tenure
set seed 234
sample2 3,cluster(idcode)

xhdfe ln_wage grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(idcode year)
    eret li
    est store xm1
xhdfe ln_wage i.idcode grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(year)
    eret li
    est store xm2
xhdfe ln_wage i.year grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(idcode)
    eret li
    est store xm3

esttab xm1 xm2 xm3, keep(grade ttl_exp c.ttl_exp#c.ttl_exp tenure tenure_alt 1.union)


reghdfe ln_wage grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(idcode year)
    eret li
    est store rm1
reghdfe ln_wage i.idcode grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(year)
    eret li
    est store rm2
reghdfe ln_wage i.year grade c.ttl_exp##c.ttl_exp tenure tenure_alt i.union, absorb(idcode)
    eret li
    est store rm3

esttab xm1 rm1 xm2 rm2 xm3 rm3, keep(grade ttl_exp c.ttl_exp#c.ttl_exp tenure tenure_alt 1.union)

log close
