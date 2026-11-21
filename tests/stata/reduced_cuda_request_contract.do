* The per-fit request of the experimental CUDA reduced tier (package 1, 16 November 2026, with its
* Amendment 1): reducedpcgcuda(on|off) and the Stata global XHDFE_REDUCED_PCG_CUDA, with the
* process default behind them (the environment variable XHDFE_REDUCED_PCG_CUDA, else off).
* Synthetic data only, numthreads(4): the fixture of reduced_cuda_runtime_contract.do (plus two
* instruments), and for saved effects a low-mobility worker-firm-year panel whose CUDA capturing
* sweep stops above the floor of the mode, so that the projection of the correction E1 runs.
* tests/stata/reduced_cuda_request_contract.sh runs the modes, each in a fresh Stata process, from
* one new scratch directory with stderr kept (the tier's trace):
*
*   session      no XHDFE_REDUCED_PCG_CUDA in the environment: CPU default (tier); CUDA with
*                reducedpcgcuda(on) (CUDA tier); CUDA alone (gpu); CUDA with reducedpcgcuda(off)
*                (gpu); CPU with reducedpcg(off) (off); reducedpcgcuda(ON) (CUDA tier); global 1
*                (CUDA tier); global 1 with reducedpcgcuda(off) (gpu); global 0 with
*                reducedpcgcuda(on) (CUDA tier); global 0 (gpu); global "yes", not a decision
*                (gpu); reducedpcg(off) with reducedpcgcuda(on) (off); the global
*                XHDFE_REDUCED_PCG at 0 with reducedpcgcuda(on) (off); both options on (CUDA
*                tier); xhdfe-fast with reducedpcgcuda(on) (tolerance_mode); reducedpcgcuda(on) on a
*                CPU fit, without and with gpubackend(cpu), gives the e(b) and e(V) of the CPU
*                default fit bit for bit; every CUDA tier fit agrees with the first within the
*                limits of the runtime contract
*   reverse      the first six fits of session in the reverse order
*   envon        XHDFE_REDUCED_PCG_CUDA=1: CUDA alone (CUDA tier); reducedpcgcuda(off) (gpu);
*                reducedpcgcuda(on) (CUDA tier); global 0 (gpu); a CPU fit (tier)
*   envoff       XHDFE_REDUCED_PCG_CUDA=0: CUDA alone (gpu); reducedpcgcuda(on) (CUDA tier);
*                reducedpcgcuda(off) (gpu); global 1 (CUDA tier)
*   profile      a file xhdfe_mobility_profile.txt in the working directory (the implicit
*                profile): reducedpcgcuda(on) gives reason mobility_profile, CUDA alone gives gpu;
*                in a working directory without one, the request enters; no other file appears
*   invalid      reducedpcgcuda(maybe) is r(198) after a fit with saved effects and residuals: those
*                variables, their types and the data are untouched, and the stored results of the
*                earlier fit restore its e(b) and e(V); e() itself is cleared, as by any xhdfe call
*                (the command's documented lifecycle: every attempt starts from an empty e())
*   savefe       the low-mobility panel on CUDA with reducedpcgcuda(on): the twin without saved
*                effects (CUDA tier) and the fit with saved effects, whose b and V agree with the
*                twin within the limits of the runtime contract, whose residuals agree within
*                1e-7, and whose saved effects are present on the estimation sample; its
*                diagnostics are displayed as a user sees them
*   savefefatal  XHDFE_TEST_REDUCED_PCG_CUDA_FATAL=2 (the second attempt of the CUDA tier throws):
*                the fit with saved effects; the same fit fails without estimates; the same fit
*                again gives the e(b) and e(V) of the first bit for bit
*   firstoption  no global: an IV fit on CUDA with reducedpcgcuda(on), then estat firststage: the
*                auxiliary fits follow the global and the environment, not the option, and stay out
*   firstglobal  the same IV fit, then the global at 1 and estat firststage: they enter
*   cpufresh     the CPU default fit alone, saved for compare
*   compare      the CPU fits of session (first and after the CUDA fits) and of reverse equal the fit
*                of cpufresh bit for bit; the CUDA tier fit of reverse agrees with that of session
*                within the limits of the runtime contract
*
* "The limits of the runtime contract": b within 1e-8*max(1,|b|), V within 1e-8*sqrt(Vjj*Vkk),
* standard errors within 1e-7 relative. Each mode ends with the line REQUEST_PASS <mode>.
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

capture program drop rq_data
program define rq_data
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
    generate double z1 = rnormal()
    generate double z2 = rnormal()
    generate double d1 = .6*z1 + .3*z2 + .2*x1 + firm/83 + rnormal()
    generate double yiv = y + .3*d1
end

* 8000 workers over 10 years in 400 firms; each worker leaves its firm with probability 0.005 a year.
capture program drop rq_lowmob
program define rq_lowmob
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

* The fit of most legs: three fixed effects, published by the tier when it enters.
capture program drop rq_fit
program define rq_fit
    quietly xhdfe y x1 x2 x3, absorb(worker firm year) vce(cluster worker) numthreads(4) `0'
end

* expect: tier (the tier published the fit) or the reason of status 0; gpu: 1 or 0.
capture program drop rq_check
program define rq_check
    args tag expect gpu
    local status = e(reduced_pcg_status)
    local why `"`e(reduced_pcg_reason)'"'
    local method = e(absorption_method_used)
    display as text "REQUEST `tag' expect=`expect' gpu=`gpu' method=`method' status=`status' reason=`why'" ///
        " gpu_used=" e(gpu_used) " gpu_status=`e(gpu_status)'"
    assert e(converged) == 1
    if ("`expect'" == "tier") assert `status' == 1 & `"`why'"' == "none" & `method' == 8 & e(precision_certified) == 1
    else assert `status' == 0 & `"`why'"' == "`expect'" & `method' != 8
    assert e(gpu_used) == `gpu'
    if (`gpu') assert "`e(gpu_backend)'" == "cuda" & "`e(gpu_status)'" == "used"
end

* What a user sees in the diagnostics of the current fit.
capture program drop rq_show
program define rq_show
    args tag
    display as text "REQUEST `tag' method=" e(absorption_method_used) " status=" e(reduced_pcg_status) ///
        " reason=`e(reduced_pcg_reason)' iterations=" e(iterations) " abs_residual_rel=" e(abs_residual_rel) ///
        " converged=" e(converged) " certified=" e(precision_certified) " gpu_used=" e(gpu_used) ///
        " gpu_status=`e(gpu_status)' N=" e(N)
end

capture mata: mata drop rq_limits_q()
mata:
real scalar rq_limits_q(string scalar bref, string scalar vref)
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
    printf("{txt}REQUEST limits b %9.3e V %9.3e se %9.3e\n", db, dv, ds)
    return(max((db / 1e-8, dv / 1e-8, ds / 1e-7)))
}
end

* The current e(b) and e(V) against a reference within the limits of the runtime contract.
capture program drop rq_limits
program define rq_limits
    args bref vref
    tempname q
    mata: st_numscalar("`q'", rq_limits_q("`bref'", "`vref'"))
    assert scalar(`q') <= 1
end

* The current e(b) and e(V) equal a reference bit for bit.
capture program drop rq_same
program define rq_same
    args bref vref
    mata: assert(st_matrix("`bref'") == st_matrix("e(b)"))
    mata: assert(st_matrix("`vref'") == st_matrix("e(V)"))
end

if ("`adopath'" != "" & "`mode'" != "compare") {
    adopath ++ `"`adopath'"'
    quietly findfile xhdfe.ado
    assert `"`r(fn)'"' == `"`adopath'/xhdfe.ado"'
    display as text `"xhdfe.ado: `r(fn)'"'
}
local envrequest : env XHDFE_REDUCED_PCG_CUDA
if ("`mode'" == "envon") assert "`envrequest'" == "1"
else if ("`mode'" == "envoff") assert "`envrequest'" == "0"
else assert "`envrequest'" == ""

if ("`mode'" == "session") {
    rq_data
    rq_fit
    rq_check cpu_default tier 0
    matrix RQ_BC = e(b)
    matrix RQ_VC = e(V)
    estimates save "rq_cpu_session", replace
    rq_fit gpubackend(cuda) reducedpcgcuda(on)
    rq_check cuda_request tier 1
    matrix RQ_BT = e(b)
    matrix RQ_VT = e(V)
    estimates save "rq_cuda_session", replace
    rq_fit gpubackend(cuda)
    rq_check cuda_alone gpu 1
    rq_fit gpubackend(cuda) reducedpcgcuda(off)
    rq_check cuda_opt_off gpu 1
    rq_fit reducedpcg(off)
    rq_check cpu_tier_off off 0
    rq_fit gpubackend(cuda) reducedpcgcuda(ON)
    rq_check cuda_request_ON tier 1
    rq_limits RQ_BT RQ_VT
    global XHDFE_REDUCED_PCG_CUDA 1
    rq_fit gpubackend(cuda)
    rq_check global1 tier 1
    rq_limits RQ_BT RQ_VT
    rq_fit gpubackend(cuda) reducedpcgcuda(off)
    rq_check global1_opt_off gpu 1
    global XHDFE_REDUCED_PCG_CUDA 0
    rq_fit gpubackend(cuda) reducedpcgcuda(on)
    rq_check global0_opt_on tier 1
    rq_limits RQ_BT RQ_VT
    rq_fit gpubackend(cuda)
    rq_check global0 gpu 1
    global XHDFE_REDUCED_PCG_CUDA yes
    rq_fit gpubackend(cuda)
    rq_check global_yes gpu 1
    macro drop XHDFE_REDUCED_PCG_CUDA
    rq_fit gpubackend(cuda) reducedpcg(off) reducedpcgcuda(on)
    rq_check veto off 1
    global XHDFE_REDUCED_PCG 0
    rq_fit gpubackend(cuda) reducedpcgcuda(on)
    rq_check general_global0 off 1
    macro drop XHDFE_REDUCED_PCG
    rq_fit gpubackend(cuda) reducedpcg(on) reducedpcgcuda(on)
    rq_check both_on tier 1
    rq_limits RQ_BT RQ_VT
    rq_fit gpubackend(cuda) reducedpcgcuda(on) tolerancemode(xhdfe-fast)
    rq_check fast tolerance_mode 1
    rq_fit reducedpcgcuda(on)
    rq_check cpu_request tier 0
    rq_same RQ_BC RQ_VC
    estimates save "rq_cpu_session_late", replace
    rq_fit gpubackend(cpu) reducedpcgcuda(on)
    rq_check cpu_backend_request tier 0
    rq_same RQ_BC RQ_VC
}
else if ("`mode'" == "reverse") {
    rq_data
    rq_fit gpubackend(cuda) reducedpcgcuda(on)
    rq_check rev_cuda_request tier 1
    estimates save "rq_cuda_reverse", replace
    rq_fit reducedpcg(off)
    rq_check rev_cpu_tier_off off 0
    rq_fit gpubackend(cuda) reducedpcgcuda(off)
    rq_check rev_cuda_opt_off gpu 1
    rq_fit gpubackend(cuda)
    rq_check rev_cuda_alone gpu 1
    rq_fit gpubackend(cuda) reducedpcgcuda(on)
    rq_check rev_cuda_request_again tier 1
    rq_fit
    rq_check rev_cpu_default tier 0
    estimates save "rq_cpu_reverse", replace
}
else if ("`mode'" == "envon") {
    rq_data
    rq_fit gpubackend(cuda)
    rq_check env1 tier 1
    rq_fit gpubackend(cuda) reducedpcgcuda(off)
    rq_check env1_opt_off gpu 1
    rq_fit gpubackend(cuda) reducedpcgcuda(on)
    rq_check env1_opt_on tier 1
    global XHDFE_REDUCED_PCG_CUDA 0
    rq_fit gpubackend(cuda)
    rq_check env1_global0 gpu 1
    macro drop XHDFE_REDUCED_PCG_CUDA
    rq_fit
    rq_check env1_cpu tier 0
}
else if ("`mode'" == "envoff") {
    rq_data
    rq_fit gpubackend(cuda)
    rq_check env0 gpu 1
    rq_fit gpubackend(cuda) reducedpcgcuda(on)
    rq_check env0_opt_on tier 1
    rq_fit gpubackend(cuda) reducedpcgcuda(off)
    rq_check env0_opt_off gpu 1
    global XHDFE_REDUCED_PCG_CUDA 1
    rq_fit gpubackend(cuda)
    rq_check env0_global1 tier 1
    macro drop XHDFE_REDUCED_PCG_CUDA
}
else if ("`mode'" == "profile") {
    rq_data
    local home `"`c(pwd)'"'
    mkdir "rq_profile_dir"
    tempname fh
    file open `fh' using "rq_profile_dir/xhdfe_mobility_profile.txt", write text
    file write `fh' "# synthetic placeholder: an implicit mobility profile (reduced_cuda_request_contract.do)" _n
    file close `fh'
    quietly cd "rq_profile_dir"
    rq_fit gpubackend(cuda) reducedpcgcuda(on)
    rq_check implicit_profile_request mobility_profile 1
    rq_fit gpubackend(cuda)
    rq_check implicit_profile_alone gpu 1
    local written : dir "." files "*"
    display as text `"REQUEST files in the profile directory: `written'"'
    assert `"`written'"' == `""xhdfe_mobility_profile.txt""'
    quietly cd `"`home'"'
    capture confirm file "xhdfe_mobility_profile.txt"
    assert _rc != 0
    rq_fit gpubackend(cuda) reducedpcgcuda(on)
    rq_check no_profile_request tier 1
}
else if ("`mode'" == "invalid") {
    rq_data
    quietly xhdfe y x1 x2 x3, absorb(worker firm year, savefe) vce(cluster worker) numthreads(4) ///
        residuals(rq_res)
    assert e(converged) == 1
    matrix RQ_BP = e(b)
    matrix RQ_VP = e(V)
    estimates store RQ_PREV
    quietly ds __hdfe* rq_res
    local outputs "`r(varlist)'"
    assert wordcount("`outputs'") == 4
    local types
    local k 0
    foreach v of local outputs {
        local ++k
        local types "`types' `: type `v''"
        quietly clonevar rq_keep`k' = `v'
    }
    datasignature set, reset
    capture noisily xhdfe y x1 x2 x3, absorb(worker firm year, savefe) vce(cluster worker) numthreads(4) ///
        residuals(rq_res) gpubackend(cuda) reducedpcgcuda(maybe)
    assert _rc == 198
    assert "`e(cmd)'" == ""
    quietly ds __hdfe* rq_res
    assert "`r(varlist)'" == "`outputs'"
    local k 0
    foreach v of local outputs {
        local ++k
        assert "`: type `v''" == "`: word `k' of `types''"
        assert `v' == rq_keep`k' | (missing(`v') & missing(rq_keep`k'))
    }
    datasignature confirm
    drop rq_keep*
    estimates restore RQ_PREV
    rq_same RQ_BP RQ_VP
    display as text "REQUEST invalid value: outputs `outputs' and the stored results preserved"
}
else if ("`mode'" == "savefe") {
    rq_lowmob
    quietly xhdfe y x1 x2, absorb(worker firm year) vce(cluster worker) numthreads(4) ///
        gpubackend(cuda) reducedpcgcuda(on) residuals(rq_rp)
    rq_check savefe_twin tier 1
    matrix RQ_BW = e(b)
    matrix RQ_VW = e(V)
    quietly xhdfe y x1 x2, absorb(worker firm year, savefe) vce(cluster worker) numthreads(4) ///
        gpubackend(cuda) reducedpcgcuda(on) residuals(rq_rs)
    rq_show savefe_fit
    assert e(converged) == 1 & e(precision_certified) == 1
    assert e(gpu_used) == 1 & "`e(gpu_backend)'" == "cuda" & "`e(gpu_status)'" == "used"
    rq_limits RQ_BW RQ_VW
    generate double rq_dr = abs(rq_rs - rq_rp) if e(sample)
    quietly summarize rq_dr
    display as text "REQUEST savefe residuals: largest difference from the twin " r(max)
    assert r(max) <= 1e-7
    foreach v in __hdfe1__ __hdfe2__ __hdfe3__ {
        confirm numeric variable `v'
        quietly count if e(sample) & missing(`v')
        assert r(N) == 0
    }
    quietly count if e(sample)
    assert r(N) == e(N)
}
else if ("`mode'" == "savefefatal") {
    rq_lowmob
    quietly xhdfe y x1 x2, absorb(worker firm year, savefe) vce(cluster worker) numthreads(4) ///
        gpubackend(cuda) reducedpcgcuda(on)
    rq_show savefe_first
    assert e(converged) == 1 & e(gpu_used) == 1
    matrix RQ_B0 = e(b)
    matrix RQ_V0 = e(V)
    capture noisily xhdfe y x1 x2, absorb(worker firm year, savefe) vce(cluster worker) numthreads(4) ///
        gpubackend(cuda) reducedpcgcuda(on)
    local rc = _rc
    display as text "REQUEST savefe fatal: rc=`rc'"
    assert `rc' != 0
    assert "`e(cmd)'" == ""
    capture matrix RQ_STALE = e(b)
    assert _rc != 0
    quietly xhdfe y x1 x2, absorb(worker firm year, savefe) vce(cluster worker) numthreads(4) ///
        gpubackend(cuda) reducedpcgcuda(on)
    rq_show savefe_again
    assert e(converged) == 1 & e(gpu_used) == 1
    rq_same RQ_B0 RQ_V0
}
else if ("`mode'" == "firstoption" | "`mode'" == "firstglobal") {
    rq_data
    quietly xhdfe yiv x1 x2 d1, absorb(worker firm year) endogenous(d1) instruments(z1 z2) ///
        vce(cluster worker) numthreads(4) gpubackend(cuda) reducedpcgcuda(on)
    display as text "REQUEST `mode' iv status=" e(reduced_pcg_status) " reason=`e(reduced_pcg_reason)'" ///
        " gpu_used=" e(gpu_used)
    assert e(gpu_used) == 1
    assert strpos(`"`e(firststage_options)'"', "reducedpcgcuda") == 0
    if ("`mode'" == "firstglobal") global XHDFE_REDUCED_PCG_CUDA 1
    estat firststage
    matrix RQ_FG = r(first_gpu_used)
    matrix list RQ_FG
    mata: assert(all(st_matrix("RQ_FG") :== 1))
    macro drop XHDFE_REDUCED_PCG_CUDA
}
else if ("`mode'" == "cpufresh") {
    rq_data
    rq_fit
    rq_check cpu_fresh tier 0
    estimates save "rq_cpu_fresh", replace
}
else if ("`mode'" == "compare") {
    estimates use "rq_cpu_fresh"
    matrix RQ_BF = e(b)
    matrix RQ_VF = e(V)
    foreach f in rq_cpu_session rq_cpu_session_late rq_cpu_reverse {
        estimates use "`f'"
        rq_same RQ_BF RQ_VF
    }
    estimates use "rq_cuda_session"
    matrix RQ_BS = e(b)
    matrix RQ_VS = e(V)
    estimates use "rq_cuda_reverse"
    rq_limits RQ_BS RQ_VS
    display as text "compared: the CPU fits of session and reverse against cpufresh, bit for bit;"
    display as text "          the CUDA tier fit of reverse against that of session, within the limits"
}
else {
    display as error "unknown mode `mode'"
    exit 198
}
display as text "REQUEST_PASS `mode'"
