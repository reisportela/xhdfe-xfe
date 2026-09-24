version 16
clear all
set more off
set linesize 200

args package output backend scratch
if (!inlist("`backend'", "cpu", "cuda")) error 198

adopath ++ "`package'"
findfile xhdfe.ado
if (`"`r(fn)'"' != `"`package'/xhdfe.ado"') error 601
findfile xhdfe.plugin
if (`"`r(fn)'"' != `"`package'/xhdfe.plugin"') error 601

foreach key in STATATMP TMPDIR CUDA_CACHE_PATH XDG_CACHE_HOME {
    local value : environment `key'
    if (strpos(`"`value'"', `"`scratch'"') != 1) error 601
}
local cache_mode : environment XHDFE_ABSORPTION_CACHE_MODE
local mobility_mode : environment XHDFE_MOBILITY_MODE
if (lower("`cache_mode'") != "off" | lower("`mobility_mode'") != "off") error 198
local audit : environment XHDFE_CERTIFY
if ("`audit'" == "") local audit 0
if (!inlist(lower("`audit'"), "0", "1", "false", "true", "no", "yes")) error 198

log using "`output'/low_noise_stata.log", text name(low_noise)
set obs 4096
generate long index = _n - 1
generate byte cell = floor(index / 256)
generate byte fe_a = floor(cell / 4)
generate byte fe_b = mod(cell, 4)
generate byte sign_a = 2 * mod(index, 2) - 1
generate byte sign_b = 2 * mod(floor(index / 2), 2) - 1
generate byte leverage = fe_a == 0 & fe_b == 1
generate double x = sign_a * (1 + sign_b) / 2 * leverage
generate double weight = 1 + mod(3 * fe_a + fe_b, 7)
generate double noise = .
generate double y = .

tempname results V
postfile `results' str12 role double exponent offset beta_error variance_relative_error ///
    residual_error reconstruction_error df_r converged certified gpu_used gpu_status ///
    threads_used using "`output'/low_noise_stata_results.dta"

foreach exponent in 0 10 20 30 40 {
    scalar low_noise = 2^(-`exponent')
    scalar expected_variance = 32 * scalar(low_noise)^2 / 4088
    foreach offset in 1 16 64 {
        quietly replace noise = cond(leverage, scalar(low_noise), 1) * sign_b
        quietly replace y = .75 * x + `offset' * (fe_b - fe_a) + noise
        quietly xhdfe y x [aw=weight], absorb(fe_a fe_b) vce(robust) noconstant ///
            residuals(low_noise_residual) numthreads(2) gpubackend(`backend') ///
            tolerance(1e-8) tolerancemode(reghdfe-comparable)

        matrix `V' = e(V)
        scalar beta_error = abs(_b[x] - .75)
        scalar variance_relative_error = abs(`V'[1,1] / scalar(expected_variance) - 1)
        local df_r = e(df_r)
        local converged = e(converged)
        local certified = e(precision_certified)
        local gpu_used = e(gpu_used)
        local gpu_status = e(gpu_status_code)
        local threads_used = e(threads_used)
        if ("`backend'" == "cuda") {
            assert `gpu_used' == 1 & `gpu_status' == 1
            if ("`e(gpu_backend)'" != "cuda" | "`e(gpu_status)'" != "used") error 459
        }
        else assert `gpu_used' == 0

        generate double residual_error = abs(low_noise_residual - noise)
        quietly summarize residual_error, meanonly
        scalar max_residual_error = r(max)
        assert scalar(beta_error) <= 1e-9
        assert scalar(variance_relative_error) <= 1e-8
        assert scalar(max_residual_error) <= 1e-9
        assert `df_r' == 4088 & `converged' == 1 & `certified' == 1
        post `results' ("default") (`exponent') (`offset') (scalar(beta_error)) ///
            (scalar(variance_relative_error)) (scalar(max_residual_error)) (.) ///
            (`df_r') (`converged') (`certified') (`gpu_used') (`gpu_status') (`threads_used')
        drop low_noise_residual residual_error
    }
}

scalar low_noise = 2^(-40)
scalar expected_variance = 32 * scalar(low_noise)^2 / 4088
quietly replace noise = cond(leverage, scalar(low_noise), 1) * sign_b
quietly replace y = .75 * x + 16 * (fe_b - fe_a) + noise
quietly xhdfe y x [aw=weight], absorb(fe_a fe_b, savefe) vce(robust) ///
    residuals(low_noise_residual) numthreads(2) gpubackend(`backend') ///
    tolerance(1e-8) tolerancemode(reghdfe-comparable)
matrix `V' = e(V)
scalar beta_error = abs(_b[x] - .75)
scalar variance_relative_error = abs(`V'[1,1] / scalar(expected_variance) - 1)
local df_r = e(df_r)
local converged = e(converged)
local certified = e(precision_certified)
local gpu_used = e(gpu_used)
local gpu_status = e(gpu_status_code)
local threads_used = e(threads_used)
assert e(fe_recovery_converged) == 1
if ("`backend'" == "cuda") {
    assert `gpu_used' == 1 & `gpu_status' == 1
    if ("`e(gpu_backend)'" != "cuda" | "`e(gpu_status)'" != "used") error 459
}
else assert `gpu_used' == 0
unab saved_effects : __hdfe*__
egen double total_effects = rowtotal(`saved_effects')
generate double residual_error = abs(low_noise_residual - noise)
generate double reconstruction_error = abs(y - _b[x] * x - _b[_cons] - total_effects - noise)
quietly summarize residual_error, meanonly
scalar max_residual_error = r(max)
quietly summarize reconstruction_error, meanonly
scalar max_reconstruction_error = r(max)
assert scalar(beta_error) <= 1e-9
assert scalar(variance_relative_error) <= 1e-8
assert scalar(max_residual_error) <= 1e-9
assert scalar(max_reconstruction_error) <= 1e-6
assert `df_r' == 4088 & `converged' == 1 & `certified' == 1
post `results' ("savefe") (40) (16) (scalar(beta_error)) ///
    (scalar(variance_relative_error)) (scalar(max_residual_error)) ///
    (scalar(max_reconstruction_error)) (`df_r') (`converged') (`certified') ///
    (`gpu_used') (`gpu_status') (`threads_used')

postclose `results'
use "`output'/low_noise_stata_results.dta", clear
generate str8 backend = "`backend'"
generate str8 audit = "`audit'"
save "`output'/low_noise_stata_results.dta", replace
export delimited using "`output'/low_noise_stata_results.csv"
assert _N == 16
assert beta_error <= 1e-9 & variance_relative_error <= 1e-8 & residual_error <= 1e-9
display "LOW_NOISE_STATA_PASS `backend' " _N
log close low_noise
