* The reduced-system tier of the 2.32 candidate through Stata (the module-level
* contract is tests/reduced_pcg_contract.py). Synthetic data only, CPU,
* numthreads(1). Every mode is one fresh Stata process, all launched from one
* scratch directory with stderr kept (the tier's trace goes to stderr);
* tests/stata/reduced_pcg_contract.sh runs them in order:
*
*   battery on        tier on (the default), XHDFE_REDUCED_PCG_TRACE=1
*   battery off       XHDFE_REDUCED_PCG=0
*   battery declined  tier on, trace on, XHDFE_TEST_REDUCED_PCG_DECLINE=iterate
*   compare           on against off within the contract (b at 1e-9 max(1,|b|),
*                     V at 1e-8 sqrt(V_jj V_kk)), declined equal to off bit for
*                     bit, and each trace against the manifest of its run
*   firststage on|off first and ffirst on the fixture of iv_firststage_contract.do,
*                     one case with string identifiers (the five rows of the
*                     table, the IV state after a success and after a failure)
*   firstcompare      e(first) on against off within the same limits; the trace:
*                     every eligible auxiliary fit entered the tier
*   cache write|read [legacy]  an accepted fit with residuals() written to the
*                     absorption cache, then read back in a fresh process; with
*                     legacy, XHDFE_STATA_LEGACY_RESIDUAL_REFINE=1 in both, so
*                     that the residual refinement, which reads the route, runs
*   cachecompare      every output and diagnostic bit-identical; the read marked
*                     as a hit by its trace line
*
* Assertions through e() in every run: a fit the tier publishes reports
* e(absorption_method_used) 8, e(reduced_pcg_status) 1, reason none and 0
* discarded iterations; an ineligible fit status 0 and its exact reason; a
* declined fit status 2, reason forced and the iterations of its final line;
* with the switch off, status 0 and reason off. reduced-pcg as a request is an
* error (rc 198). With XHDFE_RPCG_TEST_CUDA=1 the battery adds a fit with
* gpubackend(cuda), which must use the GPU and be ineligible by gpu.
* Each mode ends with the line RPCG_PASS <mode> <arm>.
version 16
clear all
set more off
set linesize 255
args mode arm extra
local adopath : env XHDFE_STATA_ADOPATH
while (strlen(`"`adopath'"') > 1 & substr(`"`adopath'"', -1, 1) == "/") {
    local adopath = substr(`"`adopath'"', 1, strlen(`"`adopath'"') - 1)
}
global XHDFE_STRING_GROUP_STRICT 1

* The battery and cache data: 2,000 workers over six years, 150 firms, some
* movers; a string copy of the firm.
capture program drop rpcg_data
program define rpcg_data
    clear
    set seed 20261114
    set obs 12000
    generate long worker = 1 + floor((_n - 1) / 6)
    generate int year = 2000 + mod(_n - 1, 6)
    generate double u = runiform()
    generate long home = 1 + floor(150 * runiform())
    quietly replace home = home[_n - 1] if mod(_n - 1, 6) != 0
    generate long firm = cond(u < .2, 1 + floor(150 * runiform()), home)
    generate str6 sfirm = "F" + string(firm, "%05.0f")
    generate double x1 = rnormal()
    generate double x2 = rnormal() + mod(firm, 7) / 10
    generate double s = rnormal()
    generate double z1 = rnormal()
    generate double z2 = rnormal()
    generate double d1 = .6 * z1 + .3 * z2 + .2 * x1 + firm / 150 + rnormal()
    generate double w = .5 + runiform()
    generate double y = .5 * x1 - .25 * x2 + .3 * d1 + worker / 2000 + firm / 150 + ///
        (year - 2000) / 6 + rnormal()
    drop u home
end

* What one fit left in e(): checked against the expectation of this run, posted
* (b, V, scalars, optionally a residual variable) and, when the tier is on,
* written to the manifest that the trace must follow line by line.
capture program drop rpcg_record
program define rpcg_record
    args tag expect reason resid
    local arm "$RPCG_ARM"
    local status = e(reduced_pcg_status)
    local why `"`e(reduced_pcg_reason)'"'
    local spent = e(reduced_pcg_discarded_iterations)
    local method = e(absorption_method_used)
    display as text "RPCG `tag' arm=`arm' method=`method' status=`status' reason=`why' discarded=`spent' iterations=" e(iterations)
    assert e(converged) == 1
    if ("`arm'" == "off") {
        assert `status' == 0 & `"`why'"' == "off" & `spent' == 0 & `method' != 8
    }
    else if ("`expect'" == "ineligible") {
        assert `status' == 0 & `"`why'"' == "`reason'" & `spent' == 0 & `method' != 8
        file write RPCGMAN "`tag' ineligible `reason' ." _n
    }
    else if ("`arm'" == "declined") {
        assert `status' == 2 & `"`why'"' == "forced" & `spent' >= 0 & `method' != 8
        file write RPCGMAN "`tag' declined forced `spent'" _n
    }
    else if ("`arm'" == "read" | "`arm'" == "read_legacy") {
        assert `status' == 1 & `"`why'"' == "none" & `spent' == 0 & `method' == 8
        file write RPCGMAN "`tag' cache reduced-pcg " (e(iterations)) _n
    }
    else {
        assert `status' == 1 & `"`why'"' == "none" & `spent' == 0 & `method' == 8
        file write RPCGMAN "`tag' accepted none " (e(iterations)) _n
    }
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local k = colsof(`b')
    forvalues i = 1/`k' {
        post RPCGPOST ("`tag'") ("b") (`i') (0) (`b'[1, `i']) (max(1, abs(`b'[1, `i'])))
        forvalues j = 1/`k' {
            post RPCGPOST ("`tag'") ("V") (`i') (`j') (`V'[`i', `j']) (sqrt(`V'[`i', `i'] * `V'[`j', `j']))
        }
    }
    quietly count if e(sample)
    post RPCGPOST ("`tag'") ("sample") (0) (0) (r(N)) (.)
    foreach s in N df_r df_a df_m rss iterations absorption_method_used reduced_pcg_status ///
        reduced_pcg_discarded_iterations {
        post RPCGPOST ("`tag'") ("`s'") (0) (0) (e(`s')) (.)
    }
    if ("`resid'" != "") {
        forvalues r = 1/`=_N' {
            post RPCGPOST ("`tag'") ("resid") (`r') (0) (`resid'[`r']) (.)
        }
    }
end

* The lines of a text file that contain a pattern (all lines if empty), in r().
capture program drop rpcg_lines
program define rpcg_lines, rclass
    args file pattern
    tempname fh
    file open `fh' using `"`file'"', read text
    local n 0
    file read `fh' line
    while (r(eof) == 0) {
        if (`"`pattern'"' == "" | strpos(`"`macval(line)'"', `"`pattern'"') > 0) {
            local ++n
            return local line`n' `"`macval(line)'"'
        }
        file read `fh' line
    }
    file close `fh'
    return scalar n = `n'
end

* The final lines of a trace against a manifest, one for one: status, reason (or
* the method of a cache hit) and, where the manifest gives them, the iterations.
* "attempted" accepts accepted or declined: the tier entered the fit.
capture program drop rpcg_trace_check
program define rpcg_trace_check
    args trace manifest
    rpcg_lines `"`trace'"' "reduced_pcg event=final"
    local nt = r(n)
    forvalues i = 1/`nt' {
        local t`i' `"`r(line`i')'"'
    }
    rpcg_lines `"`manifest'"' ""
    local nm = r(n)
    display as text "RPCG trace `trace': `nt' final lines, manifest `nm'"
    assert `nt' == `nm'
    forvalues i = 1/`nm' {
        local m `"`r(line`i')'"'
        tokenize `"`m'"'
        local line `"`t`i''"'
        foreach f in status reason method iter {
            local got_`f'
        }
        if (ustrregexm(`"`line'"', "status=([a-z_]+)")) local got_status = ustrregexs(1)
        if (ustrregexm(`"`line'"', "reason=([a-z_]+)")) local got_reason = ustrregexs(1)
        if (ustrregexm(`"`line'"', "method=([a-z_-]+)")) local got_method = ustrregexs(1)
        if (ustrregexm(`"`line'"', "iterations=([0-9]+)")) local got_iter = ustrregexs(1)
        display as text "RPCG trace `1': expected `2' `3' `4'; got `got_status' `got_reason'`got_method' `got_iter'"
        if ("`2'" == "attempted") assert inlist("`got_status'", "accepted", "declined")
        else assert "`got_status'" == "`2'"
        if ("`2'" == "accepted") assert "`got_reason'" == "none"
        if (inlist("`2'", "ineligible", "declined")) assert "`got_reason'" == "`3'"
        if ("`2'" == "cache") assert "`got_method'" == "`3'"
        if ("`4'" != ".") assert "`got_iter'" == "`4'"
    }
end

* Two posted files merged; value_<left> and value_<right>.
capture program drop rpcg_pair
program define rpcg_pair
    args left right
    use `"rpcg_`right'.dta"', clear
    rename (value scale) (value_r scale_r)
    tempfile r
    quietly save `r'
    use `"rpcg_`left'.dta"', clear
    rename (value scale) (value_l scale_l)
    merge 1:1 tag what i j using `r', assert(match) nogenerate
end

if ("`adopath'" != "" & inlist("`mode'", "battery", "firststage", "cache")) {
    adopath ++ `"`adopath'"'
    quietly findfile xhdfe.ado
    assert `"`r(fn)'"' == `"`adopath'/xhdfe.ado"'
    display as text `"xhdfe.ado: `r(fn)'"'
}
if (inlist("`mode'", "battery", "firststage", "cache")) {
    local switch : env XHDFE_REDUCED_PCG
    local stage : env XHDFE_TEST_REDUCED_PCG_DECLINE
    local trace : env XHDFE_REDUCED_PCG_TRACE
    if ("`arm'" == "off") assert "`switch'" == "0"
    else assert "`switch'" == "" & "`trace'" == "1"
    if ("`arm'" == "declined") assert "`stage'" == "iterate"
    else assert "`stage'" == ""
}

if ("`mode'" == "battery") {
    assert inlist("`arm'", "on", "off", "declined")
    global RPCG_ARM "`arm'"
    postfile RPCGPOST str16 tag str40 what int i int j double(value scale) ///
        using "rpcg_battery_`arm'.dta", replace
    file open RPCGMAN using "rpcg_manifest_battery_`arm'.txt", write text replace
    rpcg_data

    * Published by the tier when it is on.
    quietly xhdfe y x1 x2, absorb(firm year) vce(cluster firm) numthreads(1)
    rpcg_record c2fe accepted
    quietly xhdfe y x1 x2, absorb(worker firm year) vce(robust) numthreads(1)
    rpcg_record c3fe accepted
    quietly xhdfe y x1 x2, absorb(sfirm year) vce(cluster sfirm) numthreads(1)
    rpcg_record cstring accepted
    quietly xhdfe y x1 x2, absorb(worker firm) residuals(r_resid) numthreads(1)
    rpcg_record cresid accepted . r_resid
    * Ineligible, each by its own reason.
    quietly xhdfe y x1 x2 [aw=w], absorb(firm year) vce(robust) numthreads(1)
    rpcg_record cweights ineligible weights
    quietly xhdfe y x1 x2, absorb(firm year) tolerancemode(xhdfe-fast) numthreads(1)
    rpcg_record cfast ineligible tolerance_mode
    quietly xhdfe y x1 x2, absorb(firm year) absorptionmethod(mlsmr) numthreads(1)
    rpcg_record cmethod ineligible method
    quietly xhdfe y x1 x2, absorb(firm year) convergence(reghdfe) numthreads(1)
    rpcg_record ccriterion ineligible criterion
    quietly xhdfe y x1 x2 d1, absorb(firm year) endogenous(d1) instruments(z1 z2) numthreads(1)
    rpcg_record civ ineligible iv
    quietly xhdfe y x1 x2, absorb(firm year, savefe) numthreads(1)
    rpcg_record csavefe ineligible savefe
    drop __hdfe*
    quietly xhdfe y x1, absorb(firm##c.s year) numthreads(1)
    rpcg_record cslopes ineligible slopes
    quietly xhdfe y x1 x2, absorb(firm year) groupvar(mobility_group) numthreads(1)
    rpcg_record cgroupvar ineligible groupvar
    drop mobility_group
    quietly xhdfe y x1 x2, absorb(firm) numthreads(1)
    rpcg_record conefe ineligible fe_count
    * reduced-pcg is reported, never requested.
    foreach request in reduced-pcg reduced_pcg {
        capture xhdfe y x1 x2, absorb(firm year) absorptionmethod(`request') numthreads(1)
        assert _rc == 198
        assert "`e(cmd)'" == ""
    }
    local cuda : env XHDFE_RPCG_TEST_CUDA
    if ("`cuda'" == "1") {
        quietly xhdfe y x1 x2, absorb(firm year) vce(cluster firm) gpubackend(cuda) numthreads(1)
        assert e(gpu_used) == 1 & "`e(gpu_backend)'" == "cuda" & "`e(gpu_status)'" == "used"
        if ("`arm'" == "off") assert e(reduced_pcg_status) == 0 & "`e(reduced_pcg_reason)'" == "off"
        else {
            assert e(reduced_pcg_status) == 0 & "`e(reduced_pcg_reason)'" == "gpu"
            file write RPCGMAN "cgpu ineligible gpu ." _n
        }
        display as text "RPCG cgpu arm=`arm' status=" e(reduced_pcg_status) " reason=`e(reduced_pcg_reason)' gpu_used=" e(gpu_used)
    }
    else display as text "RPCG_NOT_RUN cgpu (XHDFE_RPCG_TEST_CUDA is not 1)"
    postclose RPCGPOST
    file close RPCGMAN
    display as result "RPCG_PASS battery `arm'"
}

if ("`mode'" == "compare") {
    * On against off: the contract.
    rpcg_pair battery_on battery_off
    quietly count
    display as text "RPCG compare on/off: " r(N) " values"
    assert abs(value_l - value_r) <= 1e-9 * max(1, abs(value_r)) if what == "b"
    assert abs(value_l - value_r) <= 1e-8 * scale_r if what == "V"
    assert value_l == value_r if inlist(what, "sample", "N", "df_r", "df_a", "df_m")
    * Declined against off: the current route's result, bit for bit.
    rpcg_pair battery_declined battery_off
    assert value_l == value_r if !inlist(what, "reduced_pcg_status", "reduced_pcg_discarded_iterations")
    assert value_l == 2 if what == "reduced_pcg_status" & inlist(tag, "c2fe", "c3fe", "cstring", "cresid")
    * The traces.
    rpcg_trace_check trace_battery_on.txt rpcg_manifest_battery_on.txt
    rpcg_trace_check trace_battery_declined.txt rpcg_manifest_battery_declined.txt
    rpcg_lines trace_battery_off.txt "reduced_pcg event=final"
    assert r(n) == 0
    display as result "RPCG_PASS compare"
}

if ("`mode'" == "firststage") {
    assert inlist("`arm'", "on", "off")
    postfile RPCGPOST str16 tag int i int j double value using "rpcg_first_`arm'.dta", replace
    file open RPCGMAN using "rpcg_manifest_first_`arm'.txt", write text replace
    * The fixture of iv_firststage_contract.do.
    set seed 71903
    set obs 6000
    generate long firm = 1 + mod(_n,120)
    generate long year = 1 + mod(floor((_n-1)/120),10)
    generate long state = 1 + mod(_n,29)
    generate double x1 = rnormal()
    generate double x2 = rnormal()
    generate double z1 = rnormal()
    generate double z2 = rnormal()
    generate double z3 = rnormal()
    generate double v = rnormal()
    generate double d1 = .6*z1 + .3*z2 + .2*x1 + firm/120 + v
    generate double d2 = .5*z2 + .7*z3 - .2*x2 + year/20 + v + rnormal()
    generate double y = 1.5*d1 - .8*d2 + .5*x1 - .2*x2 + firm/30 + year/10 + v + rnormal()
    generate double aw = .5 + runiform()
    generate str8 sfirm = "firm" + string(firm, "%03.0f")
    local ivwhy = cond("`arm'" == "off", "off", "iv")

    * One endogenous regressor, ffirst: the IV fit, then two auxiliary fits.
    quietly xhdfe y x1 x2 d1, absorb(firm year) endogenous(d1) instruments(z1 z2 z3) ///
        vce(cluster state) numthreads(1) ffirst
    assert e(reduced_pcg_status) == 0 & "`e(reduced_pcg_reason)'" == "`ivwhy'"
    matrix F = e(first)
    forvalues r = 1/5 {
        post RPCGPOST ("e1") (`r') (1) (F[`r', 1])
    }
    file write RPCGMAN "e1 ineligible iv ." _n "e1 attempted . ." _n "e1 attempted . ." _n
    * Two endogenous regressors, first.
    quietly xhdfe y x1 x2 d1 d2, absorb(firm year) endogenous(d1 d2) instruments(z1 z2 z3) ///
        vce(robust) numthreads(1) first
    matrix F = e(first)
    forvalues r = 1/5 {
        forvalues c = 1/2 {
            post RPCGPOST ("e2") (`r') (`c') (F[`r', `c'])
        }
    }
    file write RPCGMAN "e2 ineligible iv ." _n
    forvalues a = 1/4 {
        file write RPCGMAN "e2 attempted . ." _n
    }
    * Weighted: every fit ineligible by weights.
    quietly xhdfe y x1 x2 d1 [aw=aw], absorb(firm year) endogenous(d1) instruments(z1 z2) ///
        vce(robust) numthreads(1) ffirst
    matrix F = e(first)
    forvalues r = 1/5 {
        post RPCGPOST ("aw") (`r') (1) (F[`r', 1])
    }
    file write RPCGMAN "aw ineligible weights ." _n "aw ineligible weights ." _n "aw ineligible weights ." _n

    * String identifiers: ffirst, the five rows, the IV state after a success
    * (estat firststage, first refits the auxiliary regressions) and after a failure.
    quietly xhdfe y x1 x2 d1, absorb(sfirm year) endogenous(d1) instruments(z1 z2) ///
        vce(cluster state) numthreads(1) ffirst
    assert e(reduced_pcg_status) == 0 & "`e(reduced_pcg_reason)'" == "`ivwhy'"
    matrix S = e(first)
    assert rowsof(S) == 5 & colsof(S) == 1
    forvalues r = 1/5 {
        assert !missing(S[`r', 1])
        post RPCGPOST ("string") (`r') (1) (S[`r', 1])
    }
    file write RPCGMAN "string ineligible iv ." _n "string attempted . ." _n "string attempted . ." _n
    matrix b0 = e(b)
    matrix V0 = e(V)
    generate byte sample0 = e(sample)
    local cmdline0 `"`e(cmdline)'"'
    quietly estat firststage, first
    assert mreldif(S, r(first)) < 1e-12
    file write RPCGMAN "string attempted . ." _n "string attempted . ." _n
    assert mreldif(b0, e(b)) == 0 & mreldif(V0, e(V)) == 0
    assert e(sample) == sample0
    assert `"`e(cmdline)'"' == `"`cmdline0'"'
    assert "`e(cmd)'" == "xhdfe" & "`e(endogenous)'" == "d1"
    rename z1 hidden_instrument
    capture noisily estat firststage, first
    assert _rc == 111
    rename hidden_instrument z1
    assert mreldif(b0, e(b)) == 0 & mreldif(V0, e(V)) == 0
    assert e(sample) == sample0
    assert `"`e(cmdline)'"' == `"`cmdline0'"'
    matrix S1 = e(first)
    assert mreldif(S, S1) == 0
    postclose RPCGPOST
    file close RPCGMAN
    display as result "RPCG_PASS firststage `arm'"
}

if ("`mode'" == "firstcompare") {
    use rpcg_first_off, clear
    rename value value_r
    tempfile off
    quietly save `off'
    use rpcg_first_on, clear
    rename value value_l
    merge 1:1 tag i j using `off', assert(match) nogenerate
    list tag i j value_l value_r, noobs sep(5)
    * Rows: pr2 F df df_r pvalue.
    assert value_l == value_r if inlist(i, 3, 4)
    assert abs(value_l - value_r) <= 1e-9 * max(1, abs(value_r)) if !inlist(i, 3, 4)
    rpcg_trace_check trace_firststage_on.txt rpcg_manifest_first_on.txt
    rpcg_lines trace_firststage_off.txt "reduced_pcg event=final"
    assert r(n) == 0
    display as result "RPCG_PASS firstcompare"
}

if ("`mode'" == "cache") {
    assert inlist("`arm'", "write", "read")
    local cache : env XHDFE_ABSORPTION_CACHE
    local cmode : env XHDFE_ABSORPTION_CACHE_MODE
    local legacy : env XHDFE_STATA_LEGACY_RESIDUAL_REFINE
    assert `"`cache'"' != "" & "`cmode'" == "`arm'"
    local leg "`arm'"
    if ("`extra'" == "legacy") {
        assert "`legacy'" == "1"
        local leg "`arm'_legacy"
    }
    else assert "`legacy'" == ""
    if ("`arm'" == "read") confirm file `"`cache'"'
    global RPCG_ARM "`leg'"
    postfile RPCGPOST str16 tag str40 what int i int j double(value scale) ///
        using "rpcg_cache_`leg'.dta", replace
    file open RPCGMAN using "rpcg_manifest_cache_`leg'.txt", write text replace
    rpcg_data
    quietly xhdfe y x1 x2, absorb(worker firm year) residuals(r_cache) numthreads(1)
    rpcg_record cache accepted . r_cache
    postclose RPCGPOST
    file close RPCGMAN
    display as result "RPCG_PASS cache `leg'"
}

if ("`mode'" == "cachecompare") {
    foreach suffix in "" "_legacy" {
        rpcg_pair cache_write`suffix' cache_read`suffix'
        quietly count
        display as text "RPCG cache`suffix': " r(N) " values"
        assert value_l == value_r
        assert value_l == 8 if what == "absorption_method_used"
        rpcg_trace_check trace_cache_write`suffix'.txt rpcg_manifest_cache_write`suffix'.txt
        rpcg_trace_check trace_cache_read`suffix'.txt rpcg_manifest_cache_read`suffix'.txt
    }
    display as result "RPCG_PASS cachecompare"
}
