* The session switch of the reduced-system tier (2.32 candidate): reducedpcg(on|off)
* and the Stata global XHDFE_REDUCED_PCG, with the process default (the environment
* variable) behind them. Synthetic data only (the fixture of reduced_pcg_contract.do),
* CPU, numthreads(1). tests/stata/reduced_pcg_session_switch.sh runs the modes, each
* in a fresh Stata process, from one scratch directory with stderr kept:
*
*   session        ONE session, environment without XHDFE_REDUCED_PCG, trace on:
*                  reducedpcg(off); nothing; global 0; global 0 with reducedpcg(on);
*                  nothing again; global 1 with reducedpcg(off); global "yes"
*                  (not a decision); an invalid value (rc 198); after a savefe
*                  fit, the same request with an invalid value leaves the saved
*                  effects untouched (variables, types, values)
*   envoff         a session started with XHDFE_REDUCED_PCG=0: nothing; reducedpcg(on)
*   first0         global 0, then first and ffirst: the trace has no final line
*                  (neither the IV fit nor its auxiliary fits enter the tier)
*   firstdefault   no global: the auxiliary fits enter the tier (positive control)
*   compare        every off fit bit-identical to the envoff fit; every on fit
*                  bit-identical to the first on fit
*
* Each mode ends with the line SWITCH_PASS <mode>.
version 16
clear all
set more off
set linesize 255
args mode
local adopath : env XHDFE_STATA_ADOPATH
while (strlen(`"`adopath'"') > 1 & substr(`"`adopath'"', -1, 1) == "/") {
    local adopath = substr(`"`adopath'"', 1, strlen(`"`adopath'"') - 1)
}
global XHDFE_STRING_GROUP_STRICT 1
macro drop XHDFE_REDUCED_PCG

capture program drop sw_data
program define sw_data
    clear
    set seed 20261114
    set obs 12000
    generate long worker = 1 + floor((_n - 1) / 6)
    generate int year = 2000 + mod(_n - 1, 6)
    generate double u = runiform()
    generate long home = 1 + floor(150 * runiform())
    quietly replace home = home[_n - 1] if mod(_n - 1, 6) != 0
    generate long firm = cond(u < .2, 1 + floor(150 * runiform()), home)
    generate double x1 = rnormal()
    generate double x2 = rnormal() + mod(firm, 7) / 10
    generate double z1 = rnormal()
    generate double z2 = rnormal()
    generate double d1 = .6 * z1 + .3 * z2 + .2 * x1 + firm / 150 + rnormal()
    generate double y = .5 * x1 - .25 * x2 + .3 * d1 + worker / 2000 + firm / 150 + ///
        (year - 2000) / 6 + rnormal()
    drop u home
end

* The fit of every leg: three fixed effects, published by the tier when it is on.
capture program drop sw_fit
program define sw_fit
    quietly xhdfe y x1 x2, absorb(worker firm year) vce(robust) numthreads(1) `0'
end

* Check the tier's diagnostics and post b, V and the scalars under a tag.
capture program drop sw_record
program define sw_record
    args tag expect
    local status = e(reduced_pcg_status)
    local why `"`e(reduced_pcg_reason)'"'
    local method = e(absorption_method_used)
    display as text "SWITCH `tag' expect=`expect' method=`method' status=`status' reason=`why' iterations=" e(iterations)
    assert e(converged) == 1
    if ("`expect'" == "off") assert `status' == 0 & `"`why'"' == "off" & `method' != 8
    else assert `status' == 1 & `"`why'"' == "none" & `method' == 8
    assert e(reduced_pcg_discarded_iterations) == 0
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local k = colsof(`b')
    forvalues i = 1/`k' {
        post SWPOST ("`tag'") ("`expect'") ("b") (`i') (0) (`b'[1, `i'])
        forvalues j = 1/`k' {
            post SWPOST ("`tag'") ("`expect'") ("V") (`i') (`j') (`V'[`i', `j'])
        }
    }
    foreach s in N df_r df_a rss iterations absorption_method_used threads_used {
        post SWPOST ("`tag'") ("`expect'") ("`s'") (0) (0) (e(`s'))
    }
end

if ("`adopath'" != "" & "`mode'" != "compare") {
    adopath ++ `"`adopath'"'
    quietly findfile xhdfe.ado
    assert `"`r(fn)'"' == `"`adopath'/xhdfe.ado"'
    display as text `"xhdfe.ado: `r(fn)'"'
}
local envswitch : env XHDFE_REDUCED_PCG
if ("`mode'" == "envoff") assert "`envswitch'" == "0"
else assert "`envswitch'" == ""

if ("`mode'" == "session") {
    postfile SWPOST str20 tag str4 expect str24 what int i int j double value ///
        using "sw_session.dta", replace
    sw_data
    sw_fit reducedpcg(off)
    sw_record opt_off off
    sw_fit
    sw_record default_on on
    global XHDFE_REDUCED_PCG 0
    sw_fit
    sw_record global_off off
    sw_fit reducedpcg(on)
    sw_record global0_opt_on on
    sw_fit reducedpcg(ON)
    sw_record global0_opt_ON on
    macro drop XHDFE_REDUCED_PCG
    sw_fit
    sw_record again_on on
    global XHDFE_REDUCED_PCG 1
    sw_fit reducedpcg(off)
    sw_record global1_opt_off off
    sw_fit
    sw_record global1_on on
    global XHDFE_REDUCED_PCG yes
    sw_fit
    sw_record global_yes_on on
    macro drop XHDFE_REDUCED_PCG
    capture noisily xhdfe y x1 x2, absorb(worker firm year) reducedpcg(maybe) numthreads(1)
    assert _rc == 198

    * An invalid request must not touch the outputs of an earlier fit: seed the
    * automatic outputs of savefe, repeat the same request with reducedpcg(maybe),
    * and require rc 198 with the same variables, types and values.
    quietly xhdfe y x1 x2, absorb(worker firm year, savefe) vce(robust) numthreads(1)
    * Since the saved-effects correction of 16nov2026 (E1) such a fit reports "none" when the
    * projection that replaces its capturing sweep is published by the tier, and "savefe" when
    * the sweep is published as it is.
    assert e(converged) == 1 & inlist("`e(reduced_pcg_reason)'", "savefe", "none")
    * Its report, on one marked line: the runner checks the fit's final trace line against it.
    display as text "SWITCH_SAVEFE_REPORT status=" e(reduced_pcg_status) " reason=`e(reduced_pcg_reason)'"
    quietly ds __hdfe*
    local fe_vars "`r(varlist)'"
    assert wordcount("`fe_vars'") == 3
    local fe_types
    local k 0
    foreach v of local fe_vars {
        local ++k
        local fe_types "`fe_types' `: type `v''"
        quietly clonevar sw_keep`k' = `v'
        quietly count if missing(`v')
        local fe_miss`k' = r(N)
    }
    datasignature set, reset
    capture noisily xhdfe y x1 x2, absorb(worker firm year, savefe) vce(robust) reducedpcg(maybe) numthreads(1)
    assert _rc == 198
    quietly ds __hdfe*
    assert "`r(varlist)'" == "`fe_vars'"
    local k 0
    foreach v of local fe_vars {
        local ++k
        assert "`: type `v''" == "`: word `k' of `fe_types''"
        assert `v' == sw_keep`k' | (missing(`v') & missing(sw_keep`k'))
        quietly count if missing(`v')
        assert r(N) == `fe_miss`k''
    }
    datasignature confirm
    drop sw_keep*
    display as text "SWITCH invalid request preserved `fe_vars'"
    postclose SWPOST
}
else if ("`mode'" == "envoff") {
    postfile SWPOST str20 tag str4 expect str24 what int i int j double value ///
        using "sw_envoff.dta", replace
    sw_data
    sw_fit
    sw_record env_off off
    sw_fit reducedpcg(on)
    sw_record env0_opt_on on
    global XHDFE_REDUCED_PCG 1
    sw_fit
    sw_record env0_global_on on
    macro drop XHDFE_REDUCED_PCG
    postclose SWPOST
}
else if ("`mode'" == "first0" | "`mode'" == "firstdefault") {
    sw_data
    if ("`mode'" == "first0") global XHDFE_REDUCED_PCG 0
    quietly xhdfe y x1 x2 d1, absorb(worker firm year) endogenous(d1) instruments(z1 z2) ///
        vce(robust) numthreads(1) first
    matrix list e(first)
    quietly xhdfe y x1 x2 d1, absorb(worker firm year) endogenous(d1) instruments(z1 z2) ///
        vce(robust) numthreads(1) ffirst reducedpcg(on)
    matrix list e(first)
    estat firststage
}
else if ("`mode'" == "compare") {
    use "sw_session.dta", clear
    append using "sw_envoff.dta"
    * Every off fit against the envoff fit, every on fit against default_on.
    generate str20 ref = cond(expect == "off", "env_off", "default_on")
    preserve
    keep if inlist(tag, "env_off", "default_on")
    rename value value_ref
    keep tag what i j value_ref
    rename tag ref
    tempfile refs
    quietly save `refs'
    restore
    merge m:1 ref what i j using `refs', assert(match) nogenerate
    list tag what i j value value_ref if value != value_ref & !(missing(value) & missing(value_ref)), ///
        noobs abbreviate(24)
    assert value == value_ref | (missing(value) & missing(value_ref))
    quietly levelsof tag, local(tags) clean
    display as text "compared: `tags'"
    * Never vacuous: the 9 session fits and the 3 envoff fits must all be there.
    local ntags : word count `tags'
    assert `ntags' == 12
}
else {
    display as error "unknown mode `mode'"
    exit 198
}
display as text "SWITCH_PASS `mode'"
