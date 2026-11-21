* Phase 4 of the Stata-path plan: on ordinary fits e(sample) is the sample
* marker of marksample itself, with no copy (the plugin reads the sample before
* it writes anything and then clears only the rows it dropped), and the option
* values that used to be validated after the output preparation are validated
* before it (item D12). Old package against new on synthetic data only.
* tests/stata/esample_marker_contract.sh runs each mode in a fresh Stata process,
* in its own folder, with the package on XHDFE_STATA_ADOPATH:
*
*   fit                 every case below; e() (each scalar, macro and matrix,
*                       with names), e(sample) (count and a signature), the
*                       dataset's signature, names, types and sort order after
*                       each case go to em.dta, and full copies of the data at
*                       a few points to em_<case>.dta; the output of each case
*                       is delimited in the log for a line by line comparison
*   write               the write through the last two positions of the plugin
*                       list: one absorbed effect, so the singletons can be found
*                       here independently, and e(sample) must be 0 exactly on
*                       them and 1 on the rest of the candidate sample
*                       (contiguous selection, selection with gaps, no row
*                       dropped), plus one group() fit for the positive control
*   d12 <expect>        one test per validation moved before the outputs: rc 198
*                       and the outputs of an earlier savefe/residuals fit kept
*                       (expect=kept) or altered (expect=altered, old package);
*                       the declared changes and a name collision
*   compare <a> <b>     folder a against folder b, bit for bit
*   d12compare <a> <b>  the d12 records of a (new) and b (old)
version 16
clear all
set more off
set linesize 255
args mode leg other
local adopath : env XHDFE_STATA_ADOPATH
while (strlen(`"`adopath'"') > 1 & substr(`"`adopath'"', -1, 1) == "/") {
    local adopath = substr(`"`adopath'"', 1, strlen(`"`adopath'"') - 1)
}
global XHDFE_STRING_GROUP_STRICT 1
macro drop XHDFE_REDUCED_PCG

* --- Synthetic data ----------------------------------------------------------

* Workers with 1 to 6 rows (the one-row workers are singletons), firms, years,
* instruments, missing values, an if flag, weights with zeros and missing
* values, a string firm, categories (one only outside the sample, one only on
* singleton rows) and one isolated row with frequency weight 2.
capture program drop em_panel_data
program define em_panel_data
    clear
    set seed 4401
    set obs 6000
    generate long worker = _n
    generate int len = 1 + floor(6 * runiform())
    generate long home = 1 + floor(400 * runiform())
    expand len
    bysort worker: generate int t = _n
    generate int year = 2000 + mod(t + worker, 8)
    generate long firm = cond(runiform() < .2, 1 + floor(400 * runiform()), home)
    generate str8 sfirm = "f" + string(firm, "%05.0f")
    generate byte cat = 1 + mod(worker + t, 4)
    generate double z1 = rnormal()
    generate double z2 = rnormal()
    generate double z3 = rnormal()
    generate double x1 = .6 * z1 + .3 * z2 + rnormal()
    generate double x2 = .4 * z3 + rnormal() + mod(firm, 5) / 10
    generate double y = .5 * x1 - .25 * x2 + worker / 6000 + firm / 400 + ///
        (year - 2000) / 10 + rnormal()
    replace y = . if mod(_n, 97) == 0
    replace x2 = . if mod(_n, 211) == 0
    generate byte keep = runiform() > .07
    generate double w = .5 + runiform()
    replace w = 0 if mod(_n, 151) == 0
    replace w = . if mod(_n, 173) == 0
    generate long fw = 1 + floor(3 * runiform())
    generate byte cat2 = cond(missing(y), 9, cat)
    generate byte cat3 = cond(len == 1, 8, cat)
    generate long fw2 = 1
    generate byte iso = 0
    quietly count if len == 1 & !missing(y, x1, x2)
    assert r(N) > 0
    generate long iso_order = sum(len == 1 & !missing(y, x1, x2))
    replace iso = 1 if len == 1 & !missing(y, x1, x2) & iso_order == 1
    replace fw2 = 2 if iso
    drop len home t iso_order
    generate long rowid = _n
end

capture program drop em_ts_data
program define em_ts_data
    clear
    set seed 4402
    set obs 3000
    generate long id = _n
    expand 6
    bysort id: generate int t = _n
    generate double x1 = rnormal()
    generate double x2 = rnormal()
    generate double y = .4 * x1 + .2 * x2 + id / 3000 + rnormal()
    xtset id t
    generate long rowid = _n
end

capture program drop em_group_data
program define em_group_data
    clear
    set seed 4404
    set obs 8000
    generate long gid = _n
    generate int size = 2 + floor(3 * runiform())
    generate double gy = rnormal()
    generate double gx = rnormal()
    generate long gfe = 1 + mod(gid, 300)
    expand size
    bysort gid: generate int k = _n
    generate long iid = 1 + mod(gid * 7 + k * 104729, 5000)
    drop size k
    generate long rowid = _n
end

* Every row its own level: all singletons.
capture program drop em_single_data
program define em_single_data
    clear
    set seed 4405
    set obs 500
    generate long uid = _n
    generate double x1 = rnormal()
    generate double y = rnormal()
    generate long rowid = _n
end

* 600 000 rows, one effect with about 250 000 levels (many singletons), large
* enough for the parallel selection of the sample at 16 threads.
capture program drop em_large_data
program define em_large_data
    args gaps
    clear
    set seed 4403
    set obs 600000
    generate long a = ceil(runiform() * 250000)
    generate long b = ceil(runiform() * 3000)
    generate double x1 = rnormal()
    generate double x2 = rnormal()
    generate double y = .5 * x1 - .25 * x2 + a / 250000 + rnormal()
    generate byte keep = 1
    if (`gaps') {
        replace y = . if mod(_n, 397) == 0
        replace keep = runiform() > .1
    }
    generate long rowid = _n
end

* --- Records -----------------------------------------------------------------

* rc, every e() scalar, macro and matrix (with names), e(sample) and the data.
capture program drop em_record
program define em_record
    args tag rc sortbefore
    post EMPOST ("`tag'") ("rc") ("rc") (0) (0) (`rc') ("")
    local sc : e(scalars)
    foreach s of local sc {
        post EMPOST ("`tag'") ("scalar") ("`s'") (0) (0) (e(`s')) ("")
    }
    local mc : e(macros)
    foreach m of local mc {
        * A longer text would be cut silently by post.
        assert strlen(`"`e(`m')'"') < 2045
        post EMPOST ("`tag'") ("macro") ("`m'") (0) (0) (.) (`"`e(`m')'"')
    }
    local mt : e(matrices)
    foreach m of local mt {
        tempname M
        matrix `M' = e(`m')
        local rn : rownames `M'
        local cn : colnames `M'
        assert strlen(`"`rn'"') < 2045 & strlen(`"`cn'"') < 2045
        post EMPOST ("`tag'") ("rownames") ("`m'") (0) (0) (.) (`"`rn'"')
        post EMPOST ("`tag'") ("colnames") ("`m'") (0) (0) (.) (`"`cn'"')
        forvalues i = 1/`=rowsof(`M')' {
            forvalues j = 1/`=colsof(`M')' {
                post EMPOST ("`tag'") ("matrix") ("`m'") (`i') (`j') (`M'[`i', `j']) ("")
            }
        }
    }
    * The data: names and types in order, the sort order (never changed by a
    * case), then the marker and the signature of everything with it.
    quietly describe, varlist
    local vl "`r(varlist)'"
    local types
    foreach v of local vl {
        local types "`types' `: type `v''"
    }
    assert strlen(`"`vl'"') < 2045 & strlen(`"`types'"') < 2045
    post EMPOST ("`tag'") ("data") ("varlist") (0) (0) (c(k)) (`"`vl'"')
    post EMPOST ("`tag'") ("data") ("types") (0) (0) (.) (`"`types'"')
    local sb : sortedby
    post EMPOST ("`tag'") ("data") ("sortedby") (0) (0) (.) (`"`sb'"')
    assert `"`sb'"' == `"`sortbefore'"'
    * A fixed name, so that the signature does not see a temporary name.
    tempvar sig
    quietly generate byte em_esample = e(sample)
    quietly count if em_esample
    post EMPOST ("`tag'") ("esample") ("count") (0) (0) (r(N)) ("")
    quietly generate double `sig' = _n * em_esample
    quietly summarize `sig', meanonly
    post EMPOST ("`tag'") ("esample") ("sum_of_rows") (0) (0) (r(sum)) ("")
    drop `sig'
    quietly datasignature
    post EMPOST ("`tag'") ("data") ("datasignature") (0) (0) (.) (`"`r(datasignature)'"')
    if ("$EM_SNAP" == "1") quietly save "em_`tag'.dta", replace
    drop em_esample
end

* rc and every r() scalar and matrix of an r-class command (estat).
capture program drop em_record_r
program define em_record_r
    args tag rc
    post EMPOST ("`tag'") ("rc") ("rc") (0) (0) (`rc') ("")
    local sc : r(scalars)
    foreach s of local sc {
        post EMPOST ("`tag'") ("rscalar") ("`s'") (0) (0) (r(`s')) ("")
    }
    local mt : r(matrices)
    foreach m of local mt {
        tempname M
        matrix `M' = r(`m')
        forvalues i = 1/`=rowsof(`M')' {
            forvalues j = 1/`=colsof(`M')' {
                post EMPOST ("`tag'") ("rmatrix") ("`m'") (`i') (`j') (`M'[`i', `j']) ("")
            }
        }
    }
end

* One case: its output delimited in the log, then its record. With snap, a full
* copy of the data (and the marker) goes to em_<tag>.dta.
capture program drop em_case
program define em_case
    gettoken tag 0 : 0
    gettoken snap 0 : 0
    global EM_SNAP = ("`snap'" == "snap")
    local sb : sortedby
    display as text "EM_CASE_BEGIN `tag'"
    capture noisily `0'
    local rc = _rc
    display as text "EM_CASE_END `tag' rc=`rc'"
    em_record `tag' `rc' `"`sb'"'
end

capture program drop em_rcase
program define em_rcase
    gettoken tag 0 : 0
    display as text "EM_CASE_BEGIN `tag'"
    capture noisily `0'
    local rc = _rc
    display as text "EM_CASE_END `tag' rc=`rc'"
    em_record_r `tag' `rc'
end

* D12: seed the automatic outputs, repeat the request with one change, and
* record the rc and whether any variable, type or value moved.
capture program drop em_d12
program define em_d12
    args tag opts variant
    quietly xhdfe y x1 x2, absorb(worker firm, savefe) residuals vce(robust) numthreads(16)
    foreach v in __hdfe1__ __hdfe2__ _reghdfe_resid {
        quietly clonevar em_k`v' = `v'
    }
    quietly describe, varlist
    local before "`r(varlist)'"
    local types
    foreach v of local before {
        local types "`types' `: type `v''"
    }
    datasignature set, reset
    display as text "EM_D12_BEGIN `tag'"
    if ("`variant'" == "resid_exist") {
        capture noisily xhdfe y x1 x2, absorb(worker firm, savefe) residuals(r_exist) ///
            vce(robust) numthreads(16) `opts'
    }
    else if ("`variant'" == "collision") {
        capture noisily xhdfe y x1 x2, absorb(worker firm) residuals(cc) groupvar(cc) ///
            vce(robust) numthreads(16) `opts'
    }
    else if ("`variant'" == "nothreads") {
        * numthreads() may be given once: the invalid value replaces 16.
        capture noisily xhdfe y x1 x2, absorb(worker firm, savefe) residuals ///
            vce(robust) `opts'
    }
    else {
        capture noisily xhdfe y x1 x2, absorb(worker firm, savefe) residuals ///
            vce(robust) numthreads(16) `opts'
    }
    local rc = _rc
    display as text "EM_D12_END `tag' rc=`rc'"
    local changed 0
    quietly describe, varlist
    if ("`r(varlist)'" != "`before'") local changed 1
    local types_after
    foreach v of local before {
        capture local types_after "`types_after' `: type `v''"
    }
    if ("`types_after'" != "`types'") local changed 1
    capture datasignature confirm
    if (_rc) local changed 1
    foreach v in __hdfe1__ __hdfe2__ _reghdfe_resid {
        capture assert `v' == em_k`v' | (missing(`v') & missing(em_k`v'))
        if (_rc) local changed 1
    }
    post EMD12 ("`tag'") (`rc') (`changed')
    capture drop cc
    drop em_k*
end

if ("`adopath'" != "" & !inlist("`mode'", "compare", "d12compare")) {
    adopath ++ `"`adopath'"'
    quietly findfile xhdfe.ado
    assert `"`r(fn)'"' == `"`adopath'/xhdfe.ado"'
    display as text `"xhdfe.ado: `r(fn)'"'
}

* --- Modes -------------------------------------------------------------------

if ("`mode'" == "fit") {
    * postfile takes no strL; str2045 holds every text recorded here.
    postfile EMPOST str32 tag str12 kind str48 name int i int j double value str2045 text ///
        using "em.dta", replace
    em_panel_data
    local P "y x1 x2, absorb(worker firm)"
    em_case base nosnap xhdfe `P' vce(cluster worker) numthreads(16)
    em_case base_replay nosnap xhdfe
    em_rcase base_estat_sum estat summarize
    em_rcase base_estat_ic estat ic
    em_case base_pred_xb nosnap predict double xbh, xb
    em_case base_pred_stdp snap predict double sth, stdp
    estimates store em_m1
    em_case ifin nosnap xhdfe y x1 x2 if keep in 200/18000, absorb(worker firm year) ///
        vce(robust) numthreads(16)
    estimates restore em_m1
    em_record est_restore 0 `"`: sortedby'"'
    em_case aw nosnap xhdfe y x1 x2 [aw=w], absorb(worker firm) vce(robust) numthreads(16)
    em_case fw nosnap xhdfe y x1 x2 [fw=fw], absorb(worker firm) numthreads(16)
    em_case fw_isolated nosnap xhdfe y x1 x2 [fw=fw2], absorb(worker firm) numthreads(16)
    quietly count if iso & e(sample)
    assert r(N) == 1
    em_case pw nosnap xhdfe y x1 x2 [pw=w], absorb(worker firm) numthreads(16)
    em_case iw nosnap xhdfe y x1 x2 [iw=w], absorb(worker firm) vce(unadjusted) numthreads(16)
    em_case keepsing nosnap xhdfe `P' keepsingletons numthreads(16)
    em_case savefe nosnap xhdfe y x1 x2, absorb(worker firm, savefe) residuals(r0) numthreads(16)
    em_case savefe_pred_d nosnap predict double dh, d
    em_case savefe_pred_xbd snap predict double xbdh, xbd
    * Sorted by an automatic output: the sortpreserve route.
    sort __hdfe1__ rowid
    em_case sorted_sp snap xhdfe `P' numthreads(16)
    sort rowid
    em_case resid nosnap xhdfe `P' residuals(r1) numthreads(16)
    em_case resid_pred nosnap predict double rh, residuals
    em_case resid2 nosnap xhdfe `P' residuals numthreads(16)
    em_case groupvar snap xhdfe `P' groupvar(gv1) numthreads(16)
    em_case string nosnap xhdfe y x1 x2, absorb(sfirm worker) vce(cluster sfirm) numthreads(16)
    em_case fv nosnap xhdfe y x1 i.cat c.x2#i.cat, absorb(worker) numthreads(16)
    em_case fv_gone_missing nosnap xhdfe y x1 i.cat2, absorb(worker firm) numthreads(16)
    em_case fv_gone_singletons nosnap xhdfe y x1 i.cat3, absorb(worker firm) numthreads(16)
    * Category 8 is in the candidate sample and only on singleton rows: the
    * factor expansion after the call (:2530) must see it gone from the marker.
    quietly count if cat3 == 8 & !missing(y, x1)
    assert r(N) > 0
    quietly count if cat3 == 8 & e(sample)
    assert r(N) == 0
    em_case slope nosnap xhdfe y x1, absorb(worker firm##c.z1) numthreads(16)
    em_case empty_model nosnap xhdfe y, absorb(firm#c.x2) numthreads(16)
    * No parameter is left: the post without e(b) (the empty-model branch).
    capture confirm matrix e(b)
    assert _rc != 0
    assert e(N) > 0 & e(N) < .
    em_case iv_first nosnap xhdfe y x1 x2, absorb(worker firm) endogenous(x1) ///
        instruments(z1 z2) first numthreads(16)
    em_case iv_first2 nosnap xhdfe y x1 x2, absorb(worker firm) endogenous(x1 x2) ///
        instruments(z1 z2 z3) first numthreads(16)
    em_case nosample nosnap xhdfe `P' nosample numthreads(16)
    em_case nosample_first nosnap xhdfe y x1 x2, absorb(worker firm) endogenous(x1) ///
        instruments(z1 z2) first nosample numthreads(16)
    em_case err_if0 nosnap xhdfe y x1 if 0, absorb(worker) numthreads(16)
    em_case err_maxiter nosnap xhdfe y x1 x2, absorb(worker firm year) maxiter(1) numthreads(16)
    * One valid value of each option whose validation moved (D12): accepted,
    * with the same stored results.
    em_case v_convergence nosnap xhdfe `P' convergence(normchange) numthreads(16)
    em_case v_numthreads nosnap xhdfe `P' numthreads(16)
    em_case v_defaultthreads nosnap xhdfe `P' defaultthreads(16) numthreads(16)
    em_case v_maxthreads nosnap xhdfe `P' maxthreads(48) numthreads(16)
    em_case v_minparallelrows nosnap xhdfe `P' minparallelrows(1000) numthreads(16)
    em_case v_targetrows nosnap xhdfe `P' targetrowsperthread(2000) numthreads(16)
    em_case v_statstyle nosnap xhdfe `P' statstyle(legacy) numthreads(16)
    em_case v_tolerancemode nosnap xhdfe `P' tolerancemode(fast) numthreads(16)
    em_case v_gpubackend nosnap xhdfe `P' gpubackend(cpu) numthreads(16)
    em_case v_mobfile nosnap xhdfe `P' mobfile(em_mobility.txt) numthreads(16)
    em_case v_abscache nosnap xhdfe `P' absorptioncache(em_abs.bin) ///
        absorptioncachemode(write) numthreads(16)
    em_case v_fescache nosnap xhdfe `P' fescache(em_fe.bin) fescachemode(write) numthreads(16)
    em_ts_data
    em_case ts nosnap xhdfe y L.x1 D.x1 x2, absorb(id) numthreads(16)
    em_case ts_slope snap xhdfe y x2, absorb(id##c.L.x1) numthreads(16)
    em_group_data
    em_case group_mean snap xhdfe gy gx, absorb(gfe) group(gid) individual(iid) ///
        aggregation(mean) vce(robust) numthreads(16)
    em_single_data
    em_case err_single nosnap xhdfe y x1, absorb(uid) numthreads(16)
    * A new fit in the same session after the errors.
    em_panel_data
    em_case base_again snap xhdfe `P' vce(cluster worker) numthreads(16)
    postclose EMPOST
    * Every case did what it is there for: the fits ran, the errors failed with
    * their own rc (so that no case passes the comparison by failing in both).
    use "em.dta", clear
    quietly keep if kind == "rc"
    assert _N == 51
    assert value == 0 if !inlist(tag, "err_if0", "err_maxiter", "err_single")
    assert value == 2000 if tag == "err_if0"
    assert value == 498 if tag == "err_maxiter"
    assert value == 2001 if tag == "err_single"
}
else if ("`mode'" == "write") {
    postfile EMPOST str32 tag str12 kind str48 name int i int j double value str2045 text ///
        using "em.dta", replace
    foreach spec in "contig 0 a" "gaps 1 a" "nodrop 0 b" {
        tokenize `spec'
        local tag `1'
        em_large_data `2'
        * The candidate sample and, with one absorbed effect, its singletons.
        generate byte cand = keep & !missing(y, x1, x2)
        bysort `3' cand: generate long n_level = _N
        sort rowid
        generate byte expect = cand & n_level >= 2
        quietly count if cand & n_level == 1
        local dropped = r(N)
        if ("`tag'" == "nodrop") assert `dropped' == 0
        else assert `dropped' > 1000
        em_case w_`tag' snap xhdfe y x1 x2 if keep, absorb(`3') vce(robust) numthreads(16)
        assert e(N) < .
        * The plugin wrote through the marker: 0 exactly on the singletons.
        generate byte es = e(sample)
        assert es == expect
        quietly count if expect
        assert e(N) == r(N)
        display as text "EM_WRITE `tag' candidate_dropped=`dropped' esample=" r(N)
    }
    em_group_data
    em_case w_group nosnap xhdfe gy gx, absorb(gfe) group(gid) individual(iid) ///
        aggregation(mean) vce(robust) numthreads(16)
    postclose EMPOST
}
else if ("`mode'" == "d12") {
    local expect "`leg'"
    assert inlist("`expect'", "kept", "altered")
    postfile EMD12 str24 tag int rc byte changed using "em_d12.dta", replace
    em_panel_data
    generate double r_exist = 1
    em_d12 c01 "convergence(bogus)"
    em_d12 c02 "numthreads(-1)" nothreads
    em_d12 c03 "defaultthreads(-1)"
    em_d12 c04 "maxthreads(-1)"
    em_d12 c05 "minparallelrows(0)"
    em_d12 c06 "targetrowsperthread(-1)"
    em_d12 c07 "statstyle(bogus)"
    em_d12 c08 "tolerancemode(bogus)"
    em_d12 c09 "gpubackend(bogus)"
    em_d12 c10 "mobfile(em;mobility.txt)"
    em_d12 c11 "absorptioncachemode(bogus)"
    em_d12 c12 "absorptioncache(em;abs.bin)"
    em_d12 c13 "fescachemode(bogus)"
    em_d12 c14 "festructurecache fescache(em_fe.bin)"
    em_d12 c15 "fescachemode(auto)"
    em_d12 c16 "fescache(em;fe.bin)"
    * Declared changes: an existing residuals() name with an invalid option was
    * rc 110 before the move and is rc 198 after it; the keepsingletons warning
    * was printed before the option error.
    em_d12 dbl "tolerancemode(bogus)" resid_exist
    em_d12 kws "keepsingletons tolerancemode(bogus)"
    * Not moved: a collision between two output names (the same in both).
    em_d12 col "" collision
    postclose EMD12
    use "em_d12.dta", clear
    list, noobs
    forvalues k = 1/16 {
        local t = "c" + string(`k', "%02.0f")
        quietly count if tag == "`t'" & rc == 198
        assert r(N) == 1
        quietly count if tag == "`t'" & changed == ("`expect'" == "altered")
        assert r(N) == 1
    }
    assert rc == cond("`expect'" == "kept", 198, 110) & changed == 0 if tag == "dbl"
    assert rc == 198 & changed == ("`expect'" == "altered") if tag == "kws"
    assert rc == 110 & changed == 1 if tag == "col"
    quietly count
    assert r(N) == 19
}
else if ("`mode'" == "compare") {
    use "../`other'/em.dta", clear
    rename (value text) (value_b text_b)
    tempfile b
    quietly save `b'
    use "../`leg'/em.dta", clear
    merge 1:1 tag kind name i j using `b'
    list tag kind name i j if _merge != 3, noobs abbreviate(24)
    assert _merge == 3
    generate byte same = (value == value_b | (missing(value) & missing(value_b))) & text == text_b
    * A wall-clock duration (e(auto_retry_elapsed_seconds)) differs between any
    * two runs: present in both, value not compared.
    generate byte timing = kind == "scalar" & strpos(name, "elapsed") > 0
    quietly count if timing & !same
    display as text "timing values not compared: " r(N)
    replace same = 1 if timing
    list tag kind name text text_b if !same & text != text_b, noobs abbreviate(24)
    assert same
    quietly levelsof tag, local(tags) clean
    local ntags : word count `tags'
    display as text "compared cases (`ntags'): `tags'"
    * Never vacuous.
    if (substr("`leg'", 1, 3) == "fit") assert `ntags' == 51
    else assert `ntags' == 4
    quietly levelsof tag if kind == "rc" & value == 0, local(ok) clean
    display as text "cases with rc 0: `: word count `ok''"
    local files : dir "../`leg'" files "em_*.dta"
    local nfiles 0
    foreach f of local files {
        if ("`f'" == "em_d12.dta") continue
        quietly describe using "../`other'/`f'", varlist
        local vb "`r(varlist)'"
        use "../`leg'/`f'", clear
        quietly describe, varlist
        assert "`r(varlist)'" == "`vb'"
        cf _all using "../`other'/`f'"
        local ++nfiles
    }
    display as text "full copies compared: `nfiles'"
    if (substr("`leg'", 1, 3) == "fit") assert `nfiles' == 7
    else assert `nfiles' == 3
}
else if ("`mode'" == "d12compare") {
    use "../`other'/em_d12.dta", clear
    rename (rc changed) (rc_b changed_b)
    tempfile b
    quietly save `b'
    use "../`leg'/em_d12.dta", clear
    merge 1:1 tag using `b', assert(match) nogenerate
    list, noobs
    * Same rc everywhere except the declared double error (110 before, 198 now).
    assert rc == rc_b if tag != "dbl"
    assert rc == 198 & rc_b == 110 if tag == "dbl"
    * The new package keeps the outputs where the old one altered them.
    assert changed == 0 & changed_b == 1 if substr(tag, 1, 1) == "c" & tag != "col"
    assert changed == 0 & changed_b == 1 if tag == "kws"
    assert changed == 0 & changed_b == 0 if tag == "dbl"
    assert changed == 1 & changed_b == 1 if tag == "col"
}
else {
    display as error "unknown mode `mode'"
    exit 198
}
display as text "EM_PASS `mode'"
