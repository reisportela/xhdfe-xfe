* Package C (18nov2026): the Stata leg of the weighted CPU route (tests/cuda_weighted_cpu_route_contract.py writes the
* synthetic fixtures; tests/stata/cuda_weighted_cpu_route_contract.sh runs one fresh stata-mp session per leg, each with
* its own STATATMP, and grades the logs and the files written here). A weighted fit with two or more absorbed effects
* in reghdfe-comparable or strict-residual whose effective backend is CUDA runs on CPU with weightedcpuroute(on):
* e(gpu_used) = 0, e(gpu_status_code) = 7, e(gpu_status) = weighted_cpu_route, e(gpu_backend) = cpu,
* e(gpu_backend_requested) = cuda, e(gpu_absorption_iterations) missing, and one note line.
* Legs (kind): routed and cpu (the same fits with gpubackend(cuda) and gpubackend(cpu); e(b), e(V) and scalars in
* hexadecimal to LEGDIR/est_<name>.txt, and the residuals and saved effects to LEGDIR/vars_<name>.dta,
* compared bit for bit; e(first) of an instrumented fit; a weighted group() fit without individual();
* xfepout on the requested backend with a numerical comparison of its partialled-out variables and two
* expected precision refusals on each backend), env (the request by
* XHDFE_GPU_BACKEND alone: xhdfe, xfepout, an instrumented fit with first stages), boundaries (absorb(a a) on the device, absorb(a a2) routed, one
* effect, xhdfe-fast, group() with individual() on the device), session (the request by the environment:
* routed and device fits alternating, then a routed fit refused inside the CPU path by maxiter() and a device fit after
* it), switch0 (weightedcpuroute(off): the device route), reghdfe (routed fits against reghdfe at
* tolerance(1e-12): coefficients 1e-8 max(1,|b|), standard errors 1e-7 relative; skipped with a line when reghdfe is
* not found), gelbach (xhdfegelbach with gpu: the full model absorbs commonfes() and fes(), so a weighted call with
* one of each is routed, r(gpu_status) = weighted_cpu_route, and with one effect in all, or unweighted, it is not).
* Each fit prints CWR_OK or CWR_FAIL; CWR_PASS closes a clean leg.
version 18
clear all
set more off
set linesize 255
args package datadir kind legdir reghdfe
adopath ++ "`package'"
discard
findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
findfile xfepout.ado
assert "`r(fn)'" == "`package'/xfepout.ado"
set processors 8
global CWR_FAILS 0
local L "`legdir'"
local D "`datadir'"

capture program drop cwr_fail
program define cwr_fail
    args name what
    display "CWR_FAIL name=`name' what=`what'"
    global CWR_FAILS = $CWR_FAILS + 1
end

capture program drop cwr_dump
program define cwr_dump
    args name legdir
    tempname fh b V
    file open `fh' using "`legdir'/est_`name'.txt", write replace text
    if "`e(cmd)'" == "xhdfe" {
        matrix `b' = e(b)
        matrix `V' = e(V)
        forvalues j = 1/`=colsof(`b')' {
            file write `fh' "b " %21x (`b'[1,`j']) _n
        }
        forvalues i = 1/`=rowsof(`V')' {
            forvalues j = 1/`=colsof(`V')' {
                file write `fh' "V " %21x (`V'[`i',`j']) _n
            }
        }
        foreach s in N rss iterations df_a df_r converged precision_certified absorption_method_used ///
            reduced_pcg_status threads_effective auto_retry_eligible auto_retry_fired {
            file write `fh' "`s' " %21x (e(`s')) _n
        }
        file write `fh' "reduced_pcg_reason `e(reduced_pcg_reason)'" _n
        capture confirm matrix e(first)
        if !_rc {
            tempname F
            matrix `F' = e(first)
            forvalues i = 1/`=rowsof(`F')' {
                forvalues j = 1/`=colsof(`F')' {
                    file write `fh' "first " %21x (`F'[`i',`j']) _n
                }
            }
        }
    }
    else {
        foreach s in N iterations converged method_used {
            file write `fh' "`s' " %21x (e(`s')) _n
        }
    }
    file close `fh'
end

capture program drop cwr_fit
program define cwr_fit
    syntax, name(string) legdir(string) expect(string) cmd(string asis) [savevars(string)]
    local route
    if "`expect'" == "routed" local route "weightedcpuroute(on)"
    capture noisily `cmd' `route'
    local rc = _rc
    if "`expect'" == "refuse" {
        if `rc' == 0 cwr_fail `name' not_refused
        else display "CWR_OK name=`name' refused rc=`rc'"
        exit
    }
    if `rc' {
        cwr_fail `name' rc`rc'
        exit
    }
    local used = e(gpu_used)
    local code = e(gpu_status_code)
    local st "`e(gpu_status)'"
    local be "`e(gpu_backend)'"
    local rq "`e(gpu_backend_requested)'"
    local it = e(gpu_absorption_iterations)
    display "CWR_OK name=`name' cmd=`e(cmd)' gpu=`used' code=`code' st=`st' be=`be' rq=`rq' gpuit=`it'"
    if "`expect'" == "routed" {
        if !(`used' == 0 & `code' == 7 & "`st'" == "weighted_cpu_route" & "`be'" == "cpu" & "`rq'" == "cuda" ///
            & missing(`it')) {
            cwr_fail `name' routed_fields
        }
    }
    else if "`expect'" == "device" {
        if !(`used' == 1 & `code' == 1 & "`st'" == "used" & "`be'" == "cuda") cwr_fail `name' device_fields
    }
    else if "`expect'" == "device_env" {
        * xfepout posts the backend names only for an explicit gpubackend().
        if !(`used' == 1 & `code' == 1 & "`st'" == "used" & `it' > 0 & !missing(`it') ///
            & "`be'" == "" & "`rq'" == "") cwr_fail `name' environment_device_fields
    }
    else if "`expect'" == "cpu" {
        if !(`used' == 0 & `code' == 0) cwr_fail `name' cpu_fields
    }
    else if "`expect'" == "notrouted" {
        if `code' == 7 | `used' != 1 cwr_fail `name' grouped_fit_not_on_device
    }
    cwr_dump `name' `legdir'
    if "`savevars'" != "" {
        preserve
        keep `savevars'
        save "`legdir'/vars_`name'.dta", replace
        restore
    }
end

capture program drop cwr_gel
program define cwr_gel
    syntax, name(string) expect(string) cmd(string asis)
    local route
    if "`expect'" == "routed" local route "weightedcpuroute(on)"
    capture noisily `cmd' `route'
    local rc = _rc
    if `rc' {
        cwr_fail `name' rc`rc'
        exit
    }
    local used = r(gpu_used)
    local code = r(gpu_status_code)
    local st "`r(gpu_status)'"
    display "CWR_OK name=`name' cmd=xhdfegelbach gpu=`used' code=`code' st=`st' be=`r(gpu_backend)'"
    if "`expect'" == "routed" & !(`used' == 0 & `code' == 7 & "`st'" == "weighted_cpu_route") {
        cwr_fail `name' routed_fields
    }
    if "`expect'" == "device" & !(`used' == 1 & `code' == 1 & "`st'" == "used") cwr_fail `name' device_fields
end

local X x1 x2 x3 x4
if inlist("`kind'", "routed", "cpu") {
    local be = cond("`kind'" == "routed", "cuda", "cpu")
    local ex = cond("`kind'" == "routed", "routed", "cpu")
    local xfe_ex = cond("`kind'" == "routed", "device", "cpu")
    use "`D'/cov240k.dta", clear
    cwr_fit, name(aw3) legdir("`L'") expect(`ex') savevars(r_aw3) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) gpubackend(`be') numthreads(8) residuals(r_aw3))
    cwr_fit, name(fw3) legdir("`L'") expect(`ex') cmd(xhdfe y `X' [fw=w_fw], absorb(fe1 fe2 fe3) vce(cluster fe1) ///
        gpubackend(`be') numthreads(8))
    cwr_fit, name(pw3) legdir("`L'") expect(`ex') cmd(xhdfe y `X' [pw=w_aw], absorb(fe1 fe2 fe3) vce(cluster fe1) ///
        gpubackend(`be') numthreads(8))
    cwr_fit, name(iw3) legdir("`L'") expect(`ex') cmd(xhdfe y `X' [iw=w_aw], absorb(fe1 fe2 fe3) vce(cluster fe1) ///
        gpubackend(`be') numthreads(8))
    cwr_fit, name(aw2) legdir("`L'") expect(`ex') cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2) vce(cluster fe1) ///
        gpubackend(`be') numthreads(8))
    cwr_fit, name(aw4) legdir("`L'") expect(`ex') cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3 fe4) ///
        vce(cluster fe1) gpubackend(`be') numthreads(8))
    cwr_fit, name(aw3_strict) legdir("`L'") expect(`ex') savevars(r_aw3s) cmd(xhdfe y `X' [aw=w_aw], ///
        absorb(fe1 fe2 fe3) vce(cluster fe1) gpubackend(`be') numthreads(8) tolerancemode(strict-residual) ///
        residuals(r_aw3s))
    cwr_fit, name(aw3_savefe) legdir("`L'") expect(`ex') savevars(sf_*) cmd(xhdfe y `X' [aw=w_aw], ///
        absorb(fe1 fe2 fe3) vce(cluster fe1) gpubackend(`be') numthreads(8) savefes(sf_))
    cwr_fit, name(aw2_copy) legdir("`L'") expect(`ex') cmd(xhdfe y `X' [aw=w_aw], absorb(fe2 fe2c) ///
        vce(robust) gpubackend(`be') numthreads(8))
    cwr_fit, name(xfe_aw23) legdir("`L'") expect(`xfe_ex') savevars(p_*) cmd(xfepout y `X' [aw=w_aw], ///
        absorb(fe2 fe3) gpubackend(`be') numthreads(8) generate(p_))
    * The explicit sweep keeps its precision refusal on both backends.
    local explicit_xfe absorptionmethod(symmetric-gauss-seidel)
    cwr_fit, name(xfe_aw3_refused) legdir("`L'") expect(refuse) cmd(xfepout y `X' [aw=w_aw], ///
        absorb(fe1 fe2 fe3) gpubackend(`be') numthreads(8) generate(r3_) `explicit_xfe')
    cwr_fit, name(xfe_aw12_refused) legdir("`L'") expect(refuse) cmd(xfepout y `X' [aw=w_aw], ///
        absorb(fe1 fe2) gpubackend(`be') numthreads(8) generate(r12_) `explicit_xfe')
    cwr_fit, name(iv_first) legdir("`L'") expect(`ex') cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) endogenous(x2) instruments(z) first gpubackend(`be') numthreads(8))
    * A weighted group() fit without individual() is the collapsed weighted fit, routed too (c/RESPOSTA_2_C.md).
    use "`D'/groups_dup.dta", clear
    cwr_fit, name(grp_aw2) legdir("`L'") expect(`ex') cmd(xhdfe y x1 [aw=w], absorb(fe_a fe_b) group(gid) ///
        gpubackend(`be') numthreads(8))
}
if "`kind'" == "env" {
    use "`D'/cov240k.dta", clear
    cwr_fit, name(env_aw3) legdir("`L'") expect(routed) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) numthreads(8))
    cwr_fit, name(env_none3) legdir("`L'") expect(device) cmd(xhdfe y `X', absorb(fe1 fe2 fe3) vce(cluster fe1) ///
        numthreads(8))
    cwr_fit, name(env_xfe_aw23) legdir("`L'") expect(device_env) cmd(xfepout y `X' [aw=w_aw], absorb(fe2 fe3) ///
        numthreads(8) generate(q_))
    cwr_fit, name(env_iv) legdir("`L'") expect(routed) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) endogenous(x2) instruments(z) first numthreads(8))
    capture confirm matrix e(first_gpu_used)
    if _rc cwr_fail env_iv no_first_gpu_used
    else {
        tempname fg
        matrix `fg' = e(first_gpu_used)
        display "CWR_OK name=env_iv_first first_gpu_used_max=" el(`fg', 1, 1)
        forvalues i = 1/`=rowsof(`fg')' {
            forvalues j = 1/`=colsof(`fg')' {
                if `fg'[`i',`j'] != 0 cwr_fail env_iv first_stage_on_gpu
            }
        }
    }
}
if "`kind'" == "boundaries" {
    use "`D'/cov240k.dta", clear
    cwr_fit, name(b_same_twice) legdir("`L'") expect(device) cmd(xhdfe y `X' [aw=w_aw], absorb(fe2 fe2) ///
        vce(robust) gpubackend(cuda) numthreads(8) weightedcpuroute(on))
    cwr_fit, name(b_copy) legdir("`L'") expect(routed) cmd(xhdfe y `X' [aw=w_aw], absorb(fe2 fe2c) vce(robust) ///
        gpubackend(cuda) numthreads(8))
    cwr_fit, name(b_one) legdir("`L'") expect(device) cmd(xhdfe y `X' [aw=w_aw], absorb(fe2) vce(robust) ///
        gpubackend(cuda) numthreads(8) weightedcpuroute(on))
    cwr_fit, name(b_fast) legdir("`L'") expect(device) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) gpubackend(cuda) numthreads(8) tolerancemode(xhdfe-fast) weightedcpuroute(on))
    cwr_fit, name(b_unweighted) legdir("`L'") expect(device) cmd(xhdfe y `X', absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) gpubackend(cuda) numthreads(8) weightedcpuroute(on))
    use "`D'/groups_ind.dta", clear
    cwr_fit, name(b_group_ind) legdir("`L'") expect(notrouted) cmd(xhdfe y x1 [aw=w], absorb(fe_b ind) group(gid) ///
        individual(ind) gpubackend(cuda) numthreads(8) weightedcpuroute(on))
}
if "`kind'" == "session" {
    use "`D'/cov240k.dta", clear
    cwr_fit, name(s_default) legdir("`L'") expect(device) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) numthreads(8))
    cwr_fit, name(s_routed_1) legdir("`L'") expect(routed) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) numthreads(8))
    cwr_fit, name(s_device_1) legdir("`L'") expect(device) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) numthreads(8) weightedcpuroute(off))
    cwr_fit, name(s_routed_2) legdir("`L'") expect(routed) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) numthreads(8))
    cwr_fit, name(s_device_2) legdir("`L'") expect(device) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) numthreads(8))
    cwr_fit, name(s_routed_refused) legdir("`L'") expect(refuse) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) numthreads(8) maxiter(3) weightedcpuroute(on))
    cwr_fit, name(s_device_after) legdir("`L'") expect(device) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) numthreads(8))
    capture noisily xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) numthreads(8) weightedcpuroute(auto)
    if _rc != 198 cwr_fail s_invalid expected_198
}
if "`kind'" == "switch0" {
    use "`D'/cov240k.dta", clear
    cwr_fit, name(z_aw3) legdir("`L'") expect(device) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) gpubackend(cuda) numthreads(8) weightedcpuroute(off))
    cwr_fit, name(z_xfe_aw23) legdir("`L'") expect(device) cmd(xfepout y `X' [aw=w_aw], absorb(fe2 fe3) ///
        gpubackend(cuda) numthreads(8) generate(z_))
}
if "`kind'" == "power" {
    use "`D'/cov240k.dta", clear
    cwr_fit, name(power_baseline) legdir("`L'") expect(device) cmd(xhdfe y `X' [aw=w_aw], absorb(fe1 fe2 fe3) ///
        vce(cluster fe1) gpubackend(cuda) numthreads(8))
    if e(converged) != 1 | e(precision_certified) != 1 cwr_fail power_baseline uncertified
}
if "`kind'" == "reghdfe" {
    * Appended, not prepended: the folder may hold an installed xhdfe, and the package's must stay first.
    if "`reghdfe'" != "" adopath + "`reghdfe'"
    findfile xhdfe.ado
    assert "`r(fn)'" == "`package'/xhdfe.ado"
    capture which reghdfe
    if _rc {
        display "CWR_SKIP reghdfe not found: the comparison against reghdfe at tolerance(1e-12) did not run"
    }
    else {
        use "`D'/cov240k.dta", clear
        foreach spec in "aw3 aw w_aw fe1 fe2 fe3" "fw3 fw w_fw fe1 fe2 fe3" "pw3 pw w_aw fe1 fe2 fe3" ///
            "aw4 aw w_aw fe1 fe2 fe3 fe4" "aw2 aw w_aw fe1 fe2" {
            gettoken nm rest : spec
            gettoken wt rest : rest
            gettoken wv absvars : rest
            quietly xhdfe y `X' [`wt'=`wv'], absorb(`absvars') vce(cluster fe1) gpubackend(cuda) ///
                numthreads(8) weightedcpuroute(on)
            if e(gpu_status_code) != 7 cwr_fail `nm' not_routed
            tempname bx Vx br Vr
            matrix `bx' = e(b)
            matrix `Vx' = e(V)
            quietly reghdfe y `X' [`wt'=`wv'], absorb(`absvars') vce(cluster fe1) tolerance(1e-12)
            matrix `br' = e(b)
            matrix `Vr' = e(V)
            local db 0
            local dse 0
            forvalues j = 1/4 {
                local db = max(`db', abs(`bx'[1,`j'] - `br'[1,`j']) / max(1, abs(`br'[1,`j'])))
                local dse = max(`dse', abs(sqrt(`Vx'[`j',`j']) - sqrt(`Vr'[`j',`j'])) / sqrt(`Vr'[`j',`j']))
            }
            display "CWR_R name=`nm' reghdfe_db=`db' reghdfe_dse=`dse'"
            if `db' > 1e-8 | `dse' > 1e-7 cwr_fail `nm' outside_contract_vs_reghdfe
        }
    }
}
if "`kind'" == "gelbach" {
    use "`D'/cov240k.dta", clear
    cwr_gel, name(g_aw_common_fes) expect(routed) cmd(xhdfegelbach y [aw=w_aw], x1(x1) x2groups("A = x2 x3") ///
        commonfes(fe1) fes(fe2) gpu threads(8))
    cwr_gel, name(g_aw_two_common) expect(routed) cmd(xhdfegelbach y [aw=w_aw], x1(x1) x2groups("A = x2 x3") ///
        commonfes(fe1 fe2) gpu threads(8))
    cwr_gel, name(g_aw_one_effect) expect(device) cmd(xhdfegelbach y [aw=w_aw], x1(x1) x2groups("A = x2 x3") ///
        commonfes(fe1) gpu threads(8))
    cwr_gel, name(g_none_common_fes) expect(device) cmd(xhdfegelbach y, x1(x1) x2groups("A = x2 x3") ///
        commonfes(fe1) fes(fe2) gpu threads(8))
    capture noisily xhdfegelbachbootstrap y [aw=w_aw], x1(x1) x2groups("A = x2 x3") ///
        commonfes(fe1) fes(fe2) gpu requiregpu threads(8) method(pairs) reps(2) weightedcpuroute(on)
    local rc = _rc
    if `rc' != 459 cwr_fail g_requiregpu rc`rc'
    else display "CWR_OK name=g_requiregpu refused rc=`rc'"
    use "`D'/groups_dup.dta", clear
    generate double x2 = sin(gid / 7)
    foreach route in on off {
        capture noisily xhdfegelbachbootstrap y [aw=w], x1(x1) x2groups("A = x2") ///
            commonfes(fe_a fe_b) gpu threads(8) method(pairs) reps(2) seed(19) weightedcpuroute(`route')
        local rc = _rc
        if `rc' cwr_fail g_boot_`route' rc`rc'
        else {
            local used = "`route'" == "off"
            if r(reps_valid) != 2 | r(gpu_used_point) != `used' cwr_fail g_boot_`route' point_fields
            matrix boot = r(bootstrap_ledger)
            forvalues draw = 1/2 {
                if boot[`draw',1] != 1 | boot[`draw',7] != `used' cwr_fail g_boot_`route' replicate_fields
            }
            display "CWR_OK name=g_boot_`route' gpu=`used'"
        }
    }
}
if $CWR_FAILS == 0 display "CWR_PASS"
