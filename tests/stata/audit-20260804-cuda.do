version 16.0
clear all
set more off

* The default cell certifies the advertised 1e-8 contract. The tight cell
* preserves the stronger CPU/CUDA comparison at an explicitly tighter solve.
* Since 2.26.0, the bounded CPU retry can improve only the CPU stopping point;
* a 1e-10 comparison between otherwise default 1e-8 solves is not guaranteed.

local repo : environment XHDFE_REPO_ROOT
if ("`repo'" == "") local repo "`c(pwd)'"
adopath ++ "`repo'/stata"
discard

set obs 200000
generate long row = _n - 1
generate long fe1 = mod(row, 20000)
generate long fe2 = mod(37*row, 17003)
generate double x1 = sin(.001*row)
generate double x2 = cos(.0013*row)
generate double y = .6*x1 - .25*x2 + .001*fe1 - .0005*fe2 + sin(.0021*row)

xhdfe y x1 x2, absorb(fe1 fe2) vce(cluster fe1) keepsingletons ///
    numthreads(1) gpubackend(cpu)
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(gpu_used) == 0
assert "`e(tolerance_mode)'" == "reghdfe-comparable"
matrix b_default_cpu = e(b)
matrix V_default_cpu = e(V)

xhdfe y x1 x2, absorb(fe1 fe2) vce(cluster fe1) keepsingletons ///
    numthreads(1) gpubackend(cuda)
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(gpu_used) == 1
assert "`e(gpu_backend)'" == "cuda"
assert "`e(gpu_status)'" == "used"
assert e(gpu_absorption_converged) == 1
assert "`e(tolerance_mode)'" == "reghdfe-comparable"
matrix b_default_cuda = e(b)
matrix V_default_cuda = e(V)

mata: st_numscalar("db_default", max(abs(st_matrix("b_default_cpu") :- st_matrix("b_default_cuda"))))
mata: st_numscalar("dV_default", max(abs(st_matrix("V_default_cpu") :- st_matrix("V_default_cuda"))))
assert scalar(db_default) <= 1e-8
assert scalar(dV_default) <= 1e-8

xhdfe y x1 x2, absorb(fe1 fe2) vce(cluster fe1) keepsingletons ///
    numthreads(1) gpubackend(cpu) tolerancemode(strict-residual) tolerance(1e-12)
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(gpu_used) == 0
matrix b_tight_cpu = e(b)
matrix V_tight_cpu = e(V)

xhdfe y x1 x2, absorb(fe1 fe2) vce(cluster fe1) keepsingletons ///
    numthreads(1) gpubackend(cuda) tolerancemode(strict-residual) tolerance(1e-12)
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(gpu_used) == 1
assert "`e(gpu_backend)'" == "cuda"
assert "`e(gpu_status)'" == "used"
matrix b_tight_cuda = e(b)
matrix V_tight_cuda = e(V)

mata: st_numscalar("db_tight", max(abs(st_matrix("b_tight_cpu") :- st_matrix("b_tight_cuda"))))
mata: st_numscalar("dV_tight", max(abs(st_matrix("V_tight_cpu") :- st_matrix("V_tight_cuda"))))
mata: st_numscalar("db_default_cpu_ref", max(abs(st_matrix("b_default_cpu") :- st_matrix("b_tight_cpu"))))
mata: st_numscalar("db_default_cuda_ref", max(abs(st_matrix("b_default_cuda") :- st_matrix("b_tight_cpu"))))
mata: st_numscalar("dV_default_cpu_ref", max(abs(st_matrix("V_default_cpu") :- st_matrix("V_tight_cpu"))))
mata: st_numscalar("dV_default_cuda_ref", max(abs(st_matrix("V_default_cuda") :- st_matrix("V_tight_cpu"))))
assert scalar(db_tight) <= 1e-10
assert scalar(dV_tight) <= 1e-10
assert scalar(db_default_cpu_ref) <= 1e-8
assert scalar(db_default_cuda_ref) <= 1e-8
assert scalar(dV_default_cpu_ref) <= 1e-8
assert scalar(dV_default_cuda_ref) <= 1e-8

xhdfe y x1 x2, absorb(fe1 fe2) vce(cluster fe1) keepsingletons ///
    numthreads(1) gpubackend(cuda)
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(gpu_used) == 1
matrix b_default_cuda_repeat = e(b)
matrix V_default_cuda_repeat = e(V)
mata: st_numscalar("db_cuda_repeat", max(abs(st_matrix("b_default_cuda") :- st_matrix("b_default_cuda_repeat"))))
mata: st_numscalar("dV_cuda_repeat", max(abs(st_matrix("V_default_cuda") :- st_matrix("V_default_cuda_repeat"))))
assert scalar(db_cuda_repeat) <= 1e-12
assert scalar(dV_cuda_repeat) <= 1e-12

display as text "xhdfe default db=" %12.4e scalar(db_default) ///
    " dV=" %12.4e scalar(dV_default)
display as text "xhdfe tight db=" %12.4e scalar(db_tight) ///
    " dV=" %12.4e scalar(dV_tight)
display as text "xhdfe default CPU/tight-reference db=" ///
    %12.4e scalar(db_default_cpu_ref)
display as text "xhdfe default CUDA/tight-reference db=" ///
    %12.4e scalar(db_default_cuda_ref)
display as text "xhdfe CUDA repeat db=" %12.4e scalar(db_cuda_repeat) ///
    " dV=" %12.4e scalar(dV_cuda_repeat)

xfepout y x1 x2, absorb(fe1 fe2) generate(cpu_) sample(cpu_sample) ///
    keepsingletons numthreads(1) gpubackend(cpu) tolerance(1e-12)
assert e(converged) == 1
assert e(gpu_used) == 0

xfepout y x1 x2, absorb(fe1 fe2) generate(cuda_) sample(cuda_sample) ///
    keepsingletons numthreads(1) gpubackend(cuda) tolerance(1e-12)
assert e(converged) == 1
assert e(gpu_used) == 1
assert "`e(gpu_backend)'" == "cuda"
assert "`e(gpu_status)'" == "used"
assert e(gpu_absorption_converged) == 1
assert cpu_sample == cuda_sample

foreach v in y x1 x2 {
    generate double diff_`v' = abs(cpu_`v' - cuda_`v')
    quietly summarize diff_`v', meanonly
    scalar maxdiff_`v' = r(max)
    display as txt "xfepout max |CPU-CUDA| (`v') = " %12.4e scalar(maxdiff_`v')
    assert scalar(maxdiff_`v') <= 1e-8
}

display as result "PASS: Stata audit 20260804 CUDA contracts; default db=" ///
    %12.4e scalar(db_default) " tight db=" %12.4e scalar(db_tight)
