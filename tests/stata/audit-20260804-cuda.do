version 16.0
clear all
set more off

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
matrix b_cpu = e(b)
matrix V_cpu = e(V)
scalar iter_cpu = e(iterations)
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(gpu_used) == 0

xhdfe y x1 x2, absorb(fe1 fe2) vce(cluster fe1) keepsingletons ///
    numthreads(1) gpubackend(cuda)
assert e(converged) == 1
assert e(precision_certified) == 1
assert e(gpu_used) == 1
assert "`e(gpu_backend)'" == "cuda"
assert "`e(gpu_status)'" == "used"
assert e(gpu_absorption_converged) == 1

matrix b_cuda = e(b)
matrix V_cuda = e(V)
mata: st_numscalar("db", max(abs(st_matrix("b_cpu") :- st_matrix("b_cuda"))))
mata: st_numscalar("dV", max(abs(st_matrix("V_cpu") :- st_matrix("V_cuda"))))
assert scalar(db) <= 1e-10
assert scalar(dV) <= 1e-8

xfe y x1 x2, absorb(fe1 fe2) generate(cpu_) sample(cpu_sample) ///
    keepsingletons numthreads(1) gpubackend(cpu) tolerance(1e-12)
assert e(converged) == 1
assert e(gpu_used) == 0

xfe y x1 x2, absorb(fe1 fe2) generate(cuda_) sample(cuda_sample) ///
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
    display as txt "xfe max |CPU-CUDA| (`v') = " %12.4e scalar(maxdiff_`v')
    assert scalar(maxdiff_`v') <= 1e-8
}

display as result "PASS: Stata audit 20260804 CUDA contracts; db=" ///
    %12.4e scalar(db) " dV=" %12.4e scalar(dV)
