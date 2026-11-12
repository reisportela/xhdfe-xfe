version 18
args package
clear all
set more off
set maxvar 32767
adopath ++ "`package'"
which xhdfeakm
which reghdfe

* Deterministic synthetic design matching the dimensions of public issue 18.
set obs 149058
generate long row = _n-1
generate long worker = mod(row,9255)
generate long tenure = floor(row/9255)
generate byte jobs = 2 + (mod(worker,3)==0)
generate byte firm = mod(worker*7+7*floor(tenure*jobs/16),20)
generate int poy = mod(mod(48271*row+214013*worker,2147483647),1000)
generate double x = .5*sin((row+1)*.173)+.3*cos((row+1)*.037)+.1*cos(firm+1)
generate double y = .3*sin((worker+1)*.023)+.5*cos((firm+1)*.317)+.7*x ///
    +.3*cos((poy+1)*.239)+.2*sin((worker+1)*.175)+.25*sin((row+1)*.773)
timer clear
timer on 1
quietly xhdfeakm y, worker(worker) firm(firm) controls(x ib0.poy) ///
    sigmacentering(personyear) se leverages(jla) draws(200) seed(20261110) ///
    threads(16) generate(akm)
timer off 1
assert r(converged)==1 & r(fwl_precision_certified)==1
assert r(n_obs)==149058 & r(n_workers)==9255 & r(n_firms)==20
assert "`r(inference_method)'"=="worker_projection"
assert r(proj_threads_used)==16 & r(proj_workers_active)==16
matrix first_point = (r(kss_var_alpha),r(kss_var_psi),r(kss_cov))
matrix first_se = (r(se_var_psi),r(se_cov))
matrix first_ci = (r(ci_lb_psi),r(ci_ub_psi),r(ci_lb_cov),r(ci_ub_cov))
matrix first_beta = r(b)
scalar native_beta = first_beta[1,1]

* Tight reghdfe reference on the identical retained sample.
quietly reghdfe y x ib0.firm if akm_keep, absorb(worker poy) tolerance(1e-12)
assert e(converged)==1 & e(N)==149058
scalar beta_gap = abs(_b[x]-native_beta)
assert beta_gap <= 1e-8*max(1,abs(_b[x]))
predict double reference_xb if e(sample), xb
generate double reference_psi = reference_xb-_b[x]*x-_b[_cons]
quietly summarize reference_psi if akm_keep, meanonly
replace reference_psi = reference_psi-r(mean) if akm_keep
generate double psi_gap = abs(reference_psi-akm_psi) if akm_keep
quietly summarize psi_gap, meanonly
scalar maximum_psi_gap = r(max)
assert maximum_psi_gap <= 1e-8

* Reference-category and outcome-level changes leave the new results invariant.
replace y = y+1
quietly xhdfeakm y, worker(worker) firm(firm) controls(x ib500.poy) ///
    sigmacentering(personyear) ci leverages(jla) draws(200) seed(20261110) threads(16)
assert r(converged)==1 & r(fwl_precision_certified)==1
matrix second_point = (r(kss_var_alpha),r(kss_var_psi),r(kss_cov))
matrix second_se = (r(se_var_psi),r(se_cov))
matrix second_ci = (r(ci_lb_psi),r(ci_ub_psi),r(ci_lb_cov),r(ci_ub_cov))
assert mreldif(first_point,second_point) < 1e-8
assert mreldif(first_ci,second_ci) < 1e-8
assert abs(second_se[1,1]/first_se[1,1]-1) < 1e-7
assert abs(second_se[1,2]/first_se[1,2]-1) < 1e-7
display "ISSUE18_LARGE_CONTROLS_PASS beta_gap=" %21.15e beta_gap ///
    " psi_gap=" %21.15e maximum_psi_gap
timer list 1
