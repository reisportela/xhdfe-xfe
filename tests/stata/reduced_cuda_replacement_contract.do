* Package 3-lite (17 November 2026): the residual replacement S x on the device (on by default
* within the requested CUDA tier since 17 November 2026; XHDFE_REDUCED_PCG_CUDA_REPLACEMENT=0
* keeps the CPU product), through Stata. Synthetic data only: the fixture of reduced_cuda_request_contract.do (49 152 rows,
* worker-firm-year) and its low-mobility panel for saved effects (whose projection is published by
* the CUDA tier); numthreads(4), gpubackend(cuda) reducedpcgcuda(on).
* tests/stata/reduced_cuda_replacement_contract.sh runs each mode in a fresh Stata process (the
* switch XHDFE_REDUCED_PCG_CUDA_REPLACEMENT is read once per process):
*
*   off          the switch at 0: the fit and the fit with saved effects, saved
*   p2boff       the same through build_p2b's package (the reference of the OFF proof), saved
*   on           no variable (the default): the same fits, saved; the CUDA tier publishes both
*   onexplicit   the switch at 1: the fit, saved
*   offcompare   the qualification mode with the switch off: the fit, saved
*   oncompare    the qualification mode with the switch on: the fit, saved
*   fatal        the default and XHDFE_TEST_REDUCED_PCG_CUDA_FAIL=replacement: the fit fails
*                without estimates (the fatal error at the first device replacement, once per
*                process), then the same process fits again; saved
*   compare      off = p2boff bit for bit (both fits); offcompare = off and oncompare = on bit
*                for bit; on within the limits of the runtime contract of off (both fits);
*                fatal's second fit = on bit for bit; onexplicit = on bit for bit
*
* "The limits of the runtime contract": b within 1e-8*max(1,|b|), V within 1e-8*sqrt(Vjj*Vkk),
* standard errors within 1e-7 relative. Each mode ends with the line REPLACEMENT_PASS <mode>.
version 16
clear all
set more off
set linesize 255
set processors 4
args mode
local adopath : env XHDFE_STATA_ADOPATH
while (strlen(`"`adopath'"') > 1 & substr(`"`adopath'"', -1, 1) == "/") {
    local adopath = substr(`"`adopath'"', 1, strlen(`"`adopath'"') - 1)
}
macro drop XHDFE_REDUCED_PCG XHDFE_REDUCED_PCG_CUDA

capture program drop rp_data
program define rp_data
    clear
    set seed 2026111502
    set obs 49152
    generate long worker = floor((_n-1)/24)+1
    generate int firm = ceil(83*runiform())
    generate byte year = ceil(13*runiform())
    generate double x1 = rnormal()
    generate double x2 = rnormal()
    generate double x3 = rnormal()
    generate double y = .5*x1-.25*x2+.01*x3+sin(worker)+cos(firm)+year/10+rnormal()
end

* 8000 workers over 10 years in 400 firms; each worker leaves its firm with probability 0.005 a year.
capture program drop rp_lowmob
program define rp_lowmob
    clear
    set seed 20261117
    set obs 8000
    generate long worker = _n
    generate long home = floor(400*runiform())
    expand 10
    sort worker, stable
    by worker: generate byte year = _n - 1
    generate double u = runiform()
    by worker: generate long moved = sum(u < .005)
    generate long firm = mod(home + 7*moved, 400)
    generate double x1 = rnormal()
    generate double x2 = rnormal()
    generate double y = .5*x1 - .25*x2 + sin(worker) + cos(firm) + year/10 + rnormal()
    drop home u moved
end

capture program drop rp_check
program define rp_check
    args tag
    display as text "REPLACEMENT `tag' method=" e(absorption_method_used) " status=" e(reduced_pcg_status) ///
        " reason=`e(reduced_pcg_reason)' iterations=" e(iterations) " abs_residual_rel=" e(abs_residual_rel) ///
        " gpu_used=" e(gpu_used) " gpu_backend=`e(gpu_backend)' gpu_status=`e(gpu_status)'"
    assert e(converged) == 1 & e(gpu_used) == 1 & "`e(gpu_backend)'" == "cuda" & "`e(gpu_status)'" == "used"
end

* The fit, then the fit with saved effects on the low-mobility panel; each saved.
capture program drop rp_fits
program define rp_fits
    args tag
    rp_data
    quietly xhdfe y x1 x2 x3, absorb(worker firm year) vce(cluster worker) numthreads(4) ///
        gpubackend(cuda) reducedpcgcuda(on)
    rp_check `tag'
    assert e(reduced_pcg_status) == 1 & e(absorption_method_used) == 8
    estimates save "rp_`tag'", replace
    if ("`2'" == "savefe") {
        rp_lowmob
        quietly xhdfe y x1 x2, absorb(worker firm year, savefe) vce(cluster worker) numthreads(4) ///
            gpubackend(cuda) reducedpcgcuda(on)
        rp_check `tag'_savefe
        quietly ds __hdfe*
        assert wordcount("`r(varlist)'") == 3
        foreach v of varlist __hdfe* {
            quietly count if e(sample) & missing(`v')
            assert r(N) == 0
        }
        estimates save "rp_`tag'_savefe", replace
    }
end

capture mata: mata drop rp_limits_q()
mata:
real scalar rp_limits_q(string scalar bref, string scalar vref)
{
    real rowvector a, b, m, sa, sb, keep
    real matrix A, B
    real scalar db, dv, ds
    a = st_matrix("e(b)")
    A = st_matrix("e(V)")
    b = st_matrix(bref)
    B = st_matrix(vref)
    keep = selectindex(diagonal(B)' :> 0)
    a = a[keep]
    b = b[keep]
    A = A[keep, keep]
    B = B[keep, keep]
    m = abs(b)
    m = m :* (m :> 1) + (m :<= 1)
    db = max(abs(a - b) :/ m)
    sb = sqrt(diagonal(B))'
    sa = sqrt(diagonal(A))'
    dv = max(abs(A - B) :/ (sb' * sb))
    ds = max(abs(sa :/ sb :- 1))
    printf("{txt}REPLACEMENT limits b %9.3e V %9.3e se %9.3e\n", db, dv, ds)
    return(max((db / 1e-8, dv / 1e-8, ds / 1e-7)))
}
end

* The estimates saved as `a' against those saved as `b': bit for bit, or within the limits.
capture program drop rp_same
program define rp_same
    args a b
    estimates use "rp_`b'"
    tempname B V
    matrix `B' = e(b)
    matrix `V' = e(V)
    estimates use "rp_`a'"
    mata: assert(st_matrix("`B'") == st_matrix("e(b)"))
    mata: assert(st_matrix("`V'") == st_matrix("e(V)"))
    display as text "REPLACEMENT same `a' = `b'"
end
capture program drop rp_limits
program define rp_limits
    args a b
    estimates use "rp_`b'"
    tempname B V q
    matrix `B' = e(b)
    matrix `V' = e(V)
    estimates use "rp_`a'"
    mata: st_numscalar("`q'", rp_limits_q("`B'", "`V'"))
    assert scalar(`q') <= 1
    display as text "REPLACEMENT within `a' ~ `b'"
end

if ("`adopath'" != "" & "`mode'" != "compare") {
    adopath ++ `"`adopath'"'
    quietly findfile xhdfe.ado
    assert `"`r(fn)'"' == `"`adopath'/xhdfe.ado"'
    display as text `"xhdfe.ado: `r(fn)'"'
}
local switch : env XHDFE_REDUCED_PCG_CUDA_REPLACEMENT
if inlist("`mode'", "off", "offcompare") assert "`switch'" == "0"
else if ("`mode'" == "onexplicit") assert "`switch'" == "1"
else assert "`switch'" == ""

if inlist("`mode'", "off", "p2boff", "on") {
    rp_fits `mode' savefe
}
else if inlist("`mode'", "offcompare", "oncompare", "onexplicit") {
    rp_fits `mode'
}
else if ("`mode'" == "fatal") {
    rp_data
    capture noisily xhdfe y x1 x2 x3, absorb(worker firm year) vce(cluster worker) numthreads(4) ///
        gpubackend(cuda) reducedpcgcuda(on)
    assert _rc != 0
    assert "`e(cmd)'" == ""
    display as text "REPLACEMENT fatal: rc=" _rc ", no estimates"
    rp_fits fatal
}
else if ("`mode'" == "compare") {
    rp_same off p2boff
    rp_same off_savefe p2boff_savefe
    rp_same offcompare off
    rp_same oncompare on
    rp_same fatal on
    rp_same onexplicit on
    rp_limits on off
    rp_limits on_savefe off_savefe
}
else {
    display as error "unknown mode `mode'"
    exit 198
}
display as text "REPLACEMENT_PASS `mode'"
