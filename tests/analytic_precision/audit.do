version 16
clear all
set more off
set linesize 200
set processors 2
set sortseed 651
args root fixture output engine extra savefe
adopath ++ "`root'/stata"
include "`fixture'/options.do"
which xhdfe
findfile xhdfe.ado
if (`"`r(fn)'"' != `"`root'/stata/xhdfe.ado"') {
    di as error "T01 custody mismatch: loaded xhdfe.ado is `r(fn)'"
    error 601
}
findfile xhdfe.plugin
if (`"`r(fn)'"' != `"`root'/stata/xhdfe.plugin"') {
    di as error "T01 custody mismatch: loaded xhdfe.plugin is `r(fn)'"
    error 601
}
which reghdfe

local absorbopt
if ("`absorbs'" != "") local absorbopt "absorb(`absorbs')"
if ("`savefe'" == "yes") local absorbopt "absorb(`absorbs', savefe)"
if ("`engine'" == "ols") {
    use "`fixture'/explicit.dta", clear
    local dummies
    capture unab dummies : d*
    local vceopt "`ols_vceopt'"
    if ("`vceopt'" == "vce(unadjusted)") local vceopt
    capture noisily regress y x1 x2 x3 `dummies' `weights' if expected_sample, noconstant `vceopt'
    local rc = _rc
}
else {
    use "`fixture'/long.dta", clear
    sort group
    capture noisily `engine' y x1 x2 x3 `weights', `absorbopt' `groupopt' `vceopt' `constantopt' residuals(fit_residual) `extra'
    local rc = _rc
}
if (`rc' != 0) {
    clear
    set obs 1
    gen double rc = `rc'
    save "`output'/fit.dta"
    di "ANALYTIC_ATTEMPT_RECORDED"
    exit
}

local iterations = 0
local converged = 1
local method = .
local certified = .
local gpu = 0
local threads = .
local singletons = 0
local intercept = 0
capture local intercept = _b[_cons]
if ("`engine'" == "ols") quietly predict double fit_residual if e(sample), residuals
if ("`engine'" == "reghdfe") {
    local iterations = e(ic)
    local converged = e(converged)
    local singletons = e(num_singletons)
}
if ("`engine'" == "xhdfe") {
    local iterations = e(iterations)
    local converged = e(converged)
    local method = e(absorption_method_used)
    local certified = e(precision_certified)
    local gpu = e(gpu_used)
    local threads = e(threads_used)
    local singletons = e(num_singletons)
}

tempname values V
matrix `V' = e(V)
local covariance
forvalues i = 1/3 {
    forvalues j = 1/3 {
        local covariance "`covariance' (`V'[`i',`j'])"
    }
}
postfile `values' double rc double N double rss double df_r double df_m double df_a ///
    double singletons double iterations double converged double method double certified double gpu double threads ///
    double cons double b1 double b2 double b3 double v11 double v12 double v13 double v21 double v22 ///
    double v23 double v31 double v32 double v33 using "`output'/fit.dta"
post `values' (0) (e(N)) (e(rss)) (e(df_r)) (e(df_m)) (e(df_a)) (`singletons') ///
    (`iterations') (`converged') (`method') (`certified') (`gpu') (`threads') ///
    (`intercept') (_b[x1]) (_b[x2]) (_b[x3]) `covariance'
postclose `values'

gen byte fit_sample = e(sample)
gen double recovered_fe = .
if ("`savefe'" == "yes") {
    unab saved_effects : __hdfe*__
    egen double total_effects = rowtotal(`saved_effects')
    replace recovered_fe = total_effects
    if ("`slope_recovery'" != "") replace recovered_fe = recovered_fe + `slope_recovery'
}
keep group fit_sample fit_residual recovered_fe
save "`output'/residuals.dta"
di "ANALYTIC_ATTEMPT_RECORDED"
