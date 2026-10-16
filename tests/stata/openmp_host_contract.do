* Run in a real Stata host; standalone SPI probes cannot exercise its runtime.
version 16
clear all
set more off
set linesize 160
args threads rows repeats
if ("`threads'" == "") local threads 4
if ("`rows'" == "") local rows 100000
if ("`repeats'" == "") local repeats 3
local adopath : env XHDFE_STATA_ADOPATH
if (`"`adopath'"' != "") adopath ++ `"`adopath'"'
which xhdfe
which xfepout
display "OPENMP_HOST_INFO stata=" c(stata_version) " MP=" c(MP) " machine=" c(machine_type)

program define host_workers
    args requested
    assert e(openmp_enabled) == 1
    assert e(threads_requested) == `requested'
    if (`requested' > 0) assert e(threads_effective) == `requested'
    else assert e(threads_effective) > 1
    assert e(parallel_workers_active) == e(threads_effective)
    assert e(gpu_used) == 0
end

set seed 14102026
set obs `rows'
generate long firm = mod(_n,137)+1
generate long year = mod(_n,53)+1
generate double x = rnormal()
generate double z = rnormal()
generate double u = rnormal()
generate double d = .8*z + .3*x + u
generate double y = 1.5*d + .7*x + firm/17 + year/23 + u + rnormal()
foreach mode in xhdfe-fast reghdfe-comparable {
    local opts "absorb(firm year) vce(cluster firm) gpubackend(cpu) tolerance(1e-10) tolerancemode(`mode')"
    quietly xhdfe y x d, `opts' numthreads(1)
    matrix ols_b = e(b)
    matrix ols_V = e(V)
    local ols_N = e(N)
    local ols_df = e(df_r)
    quietly xhdfe y x d, `opts' endogenous(d) instruments(z) ffirst numthreads(1)
    matrix iv_b = e(b)
    matrix iv_V = e(V)
    matrix iv_first = e(first)
    forvalues trial=1/`repeats' {
        * Exercise the host between plugin calls as well as the plugin workers.
        quietly regress y x d
        quietly xhdfe y x d, `opts' numthreads(`threads')
        host_workers `threads'
        assert mreldif(ols_b,e(b)) < 1e-9 & mreldif(ols_V,e(V)) < 1e-9
        assert e(N) == `ols_N' & e(df_r) == `ols_df'
        display "OPENMP_HOST_CASE_PASS ols `mode' threads=`threads' trial=`trial' workers=" e(parallel_workers_active)
        quietly xhdfe y x d, `opts' endogenous(d) instruments(z) ffirst numthreads(`threads')
        host_workers `threads'
        assert mreldif(iv_b,e(b)) < 1e-9 & mreldif(iv_V,e(V)) < 1e-9
        assert mreldif(iv_first,e(first)) < 1e-9
        quietly estat firststage
        assert mreldif(iv_first,r(first)) < 1e-9
        display "OPENMP_HOST_CASE_PASS iv `mode' threads=`threads' trial=`trial' workers=" e(parallel_workers_active)
    }
}
quietly xfepout y x, absorb(firm year) generate(one_) numthreads(1) ///
    gpubackend(cpu) tolerance(1e-10)
forvalues trial=1/`repeats' {
    quietly xfepout y x, absorb(firm year) generate(many_) numthreads(`threads') ///
        gpubackend(cpu) tolerance(1e-10)
    host_workers `threads'
    foreach v in y x {
        assert abs(one_`v'-many_`v') < 1e-9*max(1,abs(one_`v'))
    }
    drop many_*
    display "OPENMP_HOST_CASE_PASS xfepout threads=`threads' trial=`trial' workers=" e(parallel_workers_active)
}
display "OPENMP_HOST_CONTRACT_PASS threads=`threads' rows=`rows' repeats=`repeats'"
exit
