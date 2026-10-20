* One fresh-session xhdfe fit for the C2 matrix (06oct2026).
* Derived from Verifications/perf_max_20260929/stata_receipt.do. Changes:
* version and plugin path asserted in every session; tol 1e-8, maxiter 100000,
* absorptioncachemode(off); only aggregates are written (estimates.dta with
* b and V, scalars, and _datasignature digests of the sample and residuals),
* never observation-level data.
version 18
clear all
set more off
args package data y xvars absorb options backend mode threads output inputcols expver store_residuals
log using "`output'/receipt.log", text

adopath ++ "`package'/stata"
discard
findfile xhdfe.ado
assert "`r(fn)'" == "`package'/stata/xhdfe.ado"
findfile xhdfe.plugin
assert "`r(fn)'" == "`package'/stata/xhdfe.plugin"
xhdfe, version
assert "`e(version)'" == "`expver'"
display "XHDFE_VERSION_CHECKED `e(version)'"
display "STATA_PROCESSORS `c(processors)' `c(processors_mach)' `c(stata_version)'"

timer clear 2
timer on 2
pq use using "`data'", clear
timer off 2
quietly timer list 2
scalar DATA_READ_SECONDS = r(t2)
quietly _datasignature `inputcols'
local input_signature "`r(datasignature)'"

timer clear 1
timer on 1
xhdfe `y' `xvars', absorb(`absorb') `options' ///
    gpubackend(`backend') tolerancemode(`mode') numthreads(`threads') ///
    tolerance(1e-8) maxiter(100000) absorptioncachemode(off)
timer off 1
quietly timer list 1
scalar FIT_SECONDS = r(t1)
assert e(converged) == 1
assert "$XHDFE_PLUGIN_PATH_INTERNAL" == "`package'/stata/xhdfe.plugin"
if ("`backend'" == "cuda") {
    assert e(gpu_used) == 1
    assert "`e(gpu_backend)'" == "cuda" & "`e(gpu_status)'" == "used"
}

foreach key in N N_full num_singletons df_r df_a df_m rank iterations ///
    converged precision_certified gpu_used threads_requested threads_effective ///
    threads_used parallel_workers_active gpu_attempted gpu_absorption_iterations ///
    absorption_method_used abs_residual_rel rss tss tss_within {
    scalar S_`key' = e(`key')
}
local terms : colfullnames e(b)
local gpu_status "`e(gpu_status)'"
local actual_backend "`e(gpu_backend)'"
local e_version "`e(version)'"
matrix B = e(b)'
matrix V = e(V)
local k = rowsof(B)

generate byte __audit_sample = e(sample)
quietly _datasignature __audit_sample
local sample_signature "`r(datasignature)'"
local residual_signature "not_requested"
if (`store_residuals') {
    quietly _datasignature __audit_residual
    local residual_signature "`r(datasignature)'"
}

clear
svmat double V, names(cov)
svmat double B, names(beta)
generate str80 term = ""
forvalues j = 1/`k' {
    local term : word `j' of `terms'
    replace term = "`term'" in `j'
}
generate double elapsed_seconds = scalar(FIT_SECONDS)
generate double data_read_seconds = scalar(DATA_READ_SECONDS)
foreach key in N N_full num_singletons df_r df_a df_m rank iterations ///
    converged precision_certified gpu_used threads_requested threads_effective ///
    threads_used parallel_workers_active gpu_attempted gpu_absorption_iterations ///
    absorption_method_used abs_residual_rel rss tss tss_within {
    generate double `key' = scalar(S_`key')
}
generate str40 gpu_status = "`gpu_status'"
generate str16 actual_backend = "`actual_backend'"
generate str244 input_signature = "`input_signature'"
generate str244 sample_signature = "`sample_signature'"
generate str244 residual_signature = "`residual_signature'"
generate str40 e_version = "`e_version'"
save "`output'/estimates.dta"
display "XHDFE_STATA_RECEIPT_PASS"
log close
