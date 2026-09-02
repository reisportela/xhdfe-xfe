* Focused current-byte CUDA savefe gate. Run from tests/stata.
version 16.0
clear all
set more off
set linesize 255

local repo "`c(pwd)'/../.."
adopath ++ "`repo'/stata"
discard

which xhdfe
quietly xhdfe, version
assert "`e(version)'" == "2.26.0 02sep2026"

local N 24000
local fit_tol 1e-10
local fe_tol 1e-10

set obs `N'
gen long row_id = _n
gen int firm = mod(_n - 1, 400) + 1
gen byte year = mod(floor((_n - 1) / 37), 50) + 1
gen double x1 = sin(_n / 17) + cos(_n / 71)
gen double x2 = cos(_n / 23) - sin(_n / 53)
gen double y = 1.25 + .8*x1 - .35*x2 + sin(firm/13) + cos(year/7) + .01*sin(_n/31)
isid row_id
assert !missing(y, x1, x2, firm, year)

quietly xhdfe y x1 x2, absorb(firm year, savefe) residuals(cpu_u) ///
    vce(cluster firm) gpubackend(cpu) absorptionmethod(auto) ///
    tolerancemode(reghdfe-comparable) tolerance(`fit_tol') ///
    fetolerance(`fe_tol') numthreads(8)

assert e(N) == `N'
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(gpu_attempted) == 0
assert e(gpu_used) == 0
assert "`e(gpu_backend)'" == "cpu"
assert "`e(gpu_status)'" == "not_requested"
assert e(fe_recovery_converged) == 1
assert e(fe_recovery_iterations) >= 0 & e(fe_recovery_iterations) < .
assert e(fe_recovery_iterations) == floor(e(fe_recovery_iterations))
assert e(fe_recovery_max_delta) >= 0 & e(fe_recovery_max_delta) < .

matrix cpu_b = e(b)
matrix cpu_V = e(V)
mata: assert(!hasmissing(st_matrix("cpu_b")))
mata: assert(!hasmissing(st_matrix("cpu_V")))
gen byte cpu_sample = e(sample)
count if cpu_sample
assert r(N) == `N'
assert !missing(cpu_u, __hdfe1__, __hdfe2__) if cpu_sample
gen double cpu_firm = __hdfe1__ if cpu_sample
gen double cpu_year = __hdfe2__ if cpu_sample
gen double cpu_fe_total = cpu_firm + cpu_year if cpu_sample
gen double cpu_linear = _b[_cons] + _b[x1]*x1 + _b[x2]*x2 if cpu_sample
gen double cpu_reconstructed = y - cpu_linear - cpu_fe_total if cpu_sample
gen double cpu_reconstruction_error = abs(cpu_reconstructed - cpu_u) if cpu_sample
summarize cpu_reconstruction_error, meanonly
scalar cpu_reconstruction_max = r(max)
scalar cpu_fe_recovery_iterations = e(fe_recovery_iterations)
scalar cpu_fe_recovery_max_delta = e(fe_recovery_max_delta)
drop __hdfe1__ __hdfe2__

quietly xhdfe y x1 x2, absorb(firm year, savefe) residuals(gpu_u) ///
    vce(cluster firm) gpubackend(cuda) absorptionmethod(auto) ///
    tolerancemode(reghdfe-comparable) tolerance(`fit_tol') ///
    fetolerance(`fe_tol') numthreads(8)

assert e(N) == `N'
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(gpu_attempted) == 1
assert e(gpu_used) == 1
assert e(gpu_status_code) == 1
assert e(gpu_absorption_converged) == 1
assert e(gpu_absorption_iterations) >= 0 & e(gpu_absorption_iterations) < .
assert "`e(gpu_backend)'" == "cuda"
assert "`e(gpu_status)'" == "used"
assert e(fe_recovery_converged) == 1
assert e(fe_recovery_iterations) >= 0 & e(fe_recovery_iterations) < .
assert e(fe_recovery_iterations) == floor(e(fe_recovery_iterations))
assert e(fe_recovery_max_delta) >= 0 & e(fe_recovery_max_delta) < .

matrix gpu_b = e(b)
matrix gpu_V = e(V)
mata: assert(!hasmissing(st_matrix("gpu_b")))
mata: assert(!hasmissing(st_matrix("gpu_V")))
gen byte gpu_sample = e(sample)
assert gpu_sample == cpu_sample
count if gpu_sample
assert r(N) == `N'
assert !missing(gpu_u, __hdfe1__, __hdfe2__) if gpu_sample
gen double gpu_firm = __hdfe1__ if gpu_sample
gen double gpu_year = __hdfe2__ if gpu_sample
gen double gpu_fe_total = gpu_firm + gpu_year if gpu_sample
gen double gpu_linear = _b[_cons] + _b[x1]*x1 + _b[x2]*x2 if gpu_sample
gen double gpu_reconstructed = y - gpu_linear - gpu_fe_total if gpu_sample
gen double gpu_reconstruction_error = abs(gpu_reconstructed - gpu_u) if gpu_sample
summarize gpu_reconstruction_error, meanonly
scalar gpu_reconstruction_max = r(max)
scalar gpu_fe_recovery_iterations = e(fe_recovery_iterations)
scalar gpu_fe_recovery_max_delta = e(fe_recovery_max_delta)

mata: st_numscalar("b_diff", max(abs(st_matrix("cpu_b") :- st_matrix("gpu_b"))))
mata: st_numscalar("V_diff", max(abs(st_matrix("cpu_V") :- st_matrix("gpu_V"))))
mata: st_numscalar("b_scale", max((1, max(abs(st_matrix("cpu_b"))), max(abs(st_matrix("gpu_b"))))))
mata: st_numscalar("V_scale", max((1, max(abs(st_matrix("cpu_V"))), max(abs(st_matrix("gpu_V"))))))

gen double residual_absdiff = abs(cpu_u - gpu_u) if cpu_sample
summarize residual_absdiff, meanonly
scalar residual_diff = r(max)
summarize cpu_u, meanonly
scalar residual_scale = max(1, abs(r(min)), abs(r(max)))
summarize gpu_u, meanonly
scalar residual_scale = max(residual_scale, abs(r(min)), abs(r(max)))

gen double fe_total_absdiff = abs(cpu_fe_total - gpu_fe_total) if cpu_sample
summarize fe_total_absdiff, meanonly
scalar fe_total_diff = r(max)
summarize cpu_fe_total, meanonly
scalar fe_total_scale = max(1, abs(r(min)), abs(r(max)))
summarize gpu_fe_total, meanonly
scalar fe_total_scale = max(fe_total_scale, abs(r(min)), abs(r(max)))
summarize y, meanonly
scalar y_scale = max(1, abs(r(min)), abs(r(max)))

scalar unit_roundoff = 2^(-53)
scalar gamma_n = `N'*unit_roundoff / (1 - `N'*unit_roundoff)
scalar b_bound = 64*(`fit_tol' + gamma_n)*b_scale
scalar V_bound = 256*(`fit_tol' + gamma_n)*V_scale
scalar residual_bound = 128*(`fit_tol' + gamma_n)*residual_scale
scalar fe_total_bound = 128*(`fit_tol' + `fe_tol' + gamma_n)*max(fe_total_scale, y_scale)
scalar reconstruction_bound = 64*(`fe_tol' + gamma_n)*y_scale

assert b_diff <= b_bound
assert V_diff <= V_bound
assert residual_diff <= residual_bound
assert fe_total_diff <= fe_total_bound
assert cpu_reconstruction_max <= reconstruction_bound
assert gpu_reconstruction_max <= reconstruction_bound

di as text "plugin=2.26.0 02sep2026 N=" `N' " gamma_n=" %21.15g gamma_n
di as text "b max/bound=" %21.15g b_diff " / " %21.15g b_bound
di as text "V max/bound=" %21.15g V_diff " / " %21.15g V_bound
di as text "residual max/bound=" %21.15g residual_diff " / " %21.15g residual_bound
di as text "FE-total max/bound=" %21.15g fe_total_diff " / " %21.15g fe_total_bound
di as text "CPU reconstruction/recovery=" %21.15g cpu_reconstruction_max ///
    " iter=" cpu_fe_recovery_iterations " delta=" %21.15g cpu_fe_recovery_max_delta
di as text "CUDA reconstruction/recovery=" %21.15g gpu_reconstruction_max ///
    " iter=" gpu_fe_recovery_iterations " delta=" %21.15g gpu_fe_recovery_max_delta
di as result "XHDFE_SAVEFE_CUDA_CURRENT_BYTES_PASS"
