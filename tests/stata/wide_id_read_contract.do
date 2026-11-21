* Phase 3 of the Stata-path plan: a wide identifier column (a value outside
* exact int32) is read once more on the team after a successful parallel
* fused pass, straight into (value, row) pairs, and ranked on the team, with
* today's codes. Old package against new on synthetic data only.
* tests/stata/wide_id_read_contract.sh runs each mode in a fresh Stata process,
* in its own folder, with the package on XHDFE_STATA_ADOPATH and
* XHDFE_PROFILE_CPU=1 (the positive controls read stderr):
*
*   fit               every case below; e() (each scalar, macro and matrix, with
*                     names), e(sample) (count and a signature), the signature,
*                     names, types and sort order of the data after each case
*                     go to wr.dta, full copies of the data at a few points to
*                     wr_<case>.dta, and each case's output is delimited in the
*                     log (the refusals of individual() are compared by text)
*   compare <a> <b>   folder a against folder b, bit for bit
*
* Data: 500 000 rows (the parallel read at 16 threads), a worker id stored as
* a double beyond int32, the same as float, negative and fractional, one that
* turns wide in its last rows, a wide cluster variable, an int32 firm; a group
* panel of about 660 000 rows with a wide individual identifier and its three
* refusals; and 20 000 rows (fewer than the parallel read asks for).
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

capture program drop wr_data
program define wr_data
    args rows
    clear
    set seed 3301
    set obs `rows'
    generate long worker = ceil(runiform() * 60000)
    generate long firm = ceil(runiform() * 2000)
    generate double wid = 5e10 + 7919 * worker
    * float holds 3e9 + 256 k exactly (spacing 256 in [2^31, 2^32)).
    generate float wfl = 3e9 + 256 * worker
    generate double wneg = -(4e10 + 13 * worker)
    generate double wfrac = worker + .5
    generate double wlate = worker
    replace wlate = 9e9 + worker in -200/l
    generate double cwide = 7e9 + 101 * ceil(firm / 10)
    generate double z = rnormal()
    generate double x1 = rnormal()
    generate double x2 = rnormal() + mod(firm, 7) / 10
    generate double y = .5 * x1 - .25 * x2 + worker / 60000 + firm / 2000 + rnormal()
    replace y = . if mod(_n, 499) == 0
    generate byte keep = runiform() > .1
    generate long rowid = _n
end

capture program drop wr_group_data
program define wr_group_data
    args problem
    clear
    set seed 3302
    * About 660 000 rows: four columns of them reach the parallel read's
    * threshold, so that the individual() column is read on the team too.
    set obs 220000
    generate long gid = _n
    generate int size = 2 + floor(3 * runiform())
    generate double gy = rnormal()
    generate double gx = rnormal()
    generate long gfe = 1 + mod(gid, 500)
    expand size
    bysort gid: generate int k = _n
    generate double iidw = 3e9 + mod(gid * 7 + k * 104729, 40000)
    drop size k
    generate long rowid = _n
    if ("`problem'" == "negative") replace iidw = -5 in 200
    if ("`problem'" == "fraction") replace iidw = 3e9 + .5 in 200
    if ("`problem'" == "huge") replace iidw = 9007199254740994 in 200
    if ("`problem'" == "several") {
        replace iidw = 3e9 + .5 in 300
        replace iidw = -5 in 200
        replace iidw = 9007199254740994 in 100000
    }
end

* rc, every e() scalar, macro and matrix (with names), e(sample) and the data.
capture program drop wr_record
program define wr_record
    args tag rc sortbefore
    post WRPOST ("`tag'") ("rc") ("rc") (0) (0) (`rc') ("")
    local sc : e(scalars)
    foreach s of local sc {
        post WRPOST ("`tag'") ("scalar") ("`s'") (0) (0) (e(`s')) ("")
    }
    local mc : e(macros)
    foreach m of local mc {
        assert strlen(`"`e(`m')'"') < 2045
        post WRPOST ("`tag'") ("macro") ("`m'") (0) (0) (.) (`"`e(`m')'"')
    }
    local mt : e(matrices)
    foreach m of local mt {
        tempname M
        matrix `M' = e(`m')
        local rn : rownames `M'
        local cn : colnames `M'
        assert strlen(`"`rn'"') < 2045 & strlen(`"`cn'"') < 2045
        post WRPOST ("`tag'") ("rownames") ("`m'") (0) (0) (.) (`"`rn'"')
        post WRPOST ("`tag'") ("colnames") ("`m'") (0) (0) (.) (`"`cn'"')
        forvalues i = 1/`=rowsof(`M')' {
            forvalues j = 1/`=colsof(`M')' {
                post WRPOST ("`tag'") ("matrix") ("`m'") (`i') (`j') (`M'[`i', `j']) ("")
            }
        }
    }
    quietly describe, varlist
    local vl "`r(varlist)'"
    local types
    foreach v of local vl {
        local types "`types' `: type `v''"
    }
    assert strlen(`"`vl'"') < 2045 & strlen(`"`types'"') < 2045
    post WRPOST ("`tag'") ("data") ("varlist") (0) (0) (c(k)) (`"`vl'"')
    post WRPOST ("`tag'") ("data") ("types") (0) (0) (.) (`"`types'"')
    local sb : sortedby
    post WRPOST ("`tag'") ("data") ("sortedby") (0) (0) (.) (`"`sb'"')
    assert `"`sb'"' == `"`sortbefore'"'
    tempvar sig
    quietly generate byte wr_esample = e(sample)
    quietly count if wr_esample
    post WRPOST ("`tag'") ("esample") ("count") (0) (0) (r(N)) ("")
    quietly generate double `sig' = _n * wr_esample
    quietly summarize `sig', meanonly
    post WRPOST ("`tag'") ("esample") ("sum_of_rows") (0) (0) (r(sum)) ("")
    drop `sig'
    quietly datasignature
    post WRPOST ("`tag'") ("data") ("datasignature") (0) (0) (.) (`"`r(datasignature)'"')
    if ("$WR_SNAP" == "1") quietly save "wr_`tag'.dta", replace
    drop wr_esample
end

capture program drop wr_case
program define wr_case
    gettoken tag 0 : 0
    gettoken snap 0 : 0
    global WR_SNAP = ("`snap'" == "snap")
    local sb : sortedby
    display as text "WR_CASE_BEGIN `tag'"
    capture noisily `0'
    local rc = _rc
    display as text "WR_CASE_END `tag' rc=`rc'"
    wr_record `tag' `rc' `"`sb'"'
end

if ("`adopath'" != "" & "`mode'" != "compare") {
    adopath ++ `"`adopath'"'
    quietly findfile xhdfe.ado
    assert `"`r(fn)'"' == `"`adopath'/xhdfe.ado"'
    display as text `"xhdfe.ado: `r(fn)'"'
}

if ("`mode'" == "fit") {
    * postfile takes no strL; str2045 holds every text recorded here.
    postfile WRPOST str32 tag str12 kind str48 name int i int j double value str2045 text ///
        using "wr.dta", replace
    wr_data 500000
    * The wide worker id as a fixed effect and as the cluster (the same variable).
    wr_case fe_wide nosnap xhdfe y x1 x2, absorb(wid firm) vce(cluster wid) numthreads(16)
    wr_case cl_wide nosnap xhdfe y x1 x2, absorb(worker firm) vce(cluster cwide) numthreads(16)
    wr_case fe_cl_wide nosnap xhdfe y x1 x2, absorb(wid firm) vce(cluster cwide) numthreads(16)
    * A repeated fixed effect (fe_dup_map): the wide id with its own slope.
    wr_case dup nosnap xhdfe y x1 x2, absorb(wid##c.z firm) numthreads(16)
    wr_case float nosnap xhdfe y x1 x2, absorb(wfl firm) numthreads(16)
    wr_case negative nosnap xhdfe y x1 x2, absorb(wneg firm) numthreads(16)
    wr_case fraction nosnap xhdfe y x1 x2, absorb(wfrac firm) numthreads(16)
    wr_case late nosnap xhdfe y x1 x2, absorb(wlate firm) numthreads(16)
    wr_case ifin nosnap xhdfe y x1 x2 if keep in 1001/490000, absorb(wid firm) numthreads(16)
    wr_case outputs snap xhdfe y x1 x2, absorb(wid firm, savefe) residuals(r1) ///
        vce(cluster cwide) numthreads(16)
    wr_case t1 nosnap xhdfe y x1 x2, absorb(wid firm) vce(cluster wid) numthreads(1)
    wr_case tauto nosnap xhdfe y x1 x2, absorb(wid firm) vce(cluster wid)
    wr_case t24 nosnap xhdfe y x1 x2, absorb(wid firm) vce(cluster wid) numthreads(24)
    wr_data 20000
    wr_case small snap xhdfe y x1 x2, absorb(wid firm) vce(cluster cwide) numthreads(16)
    wr_group_data
    wr_case ind snap xhdfe gy gx, absorb(gfe) group(gid) individual(iidw) aggregation(mean) ///
        vce(robust) numthreads(16)
    foreach problem in negative fraction huge several {
        wr_group_data `problem'
        wr_case ind_`problem' nosnap xhdfe gy gx, absorb(gfe) group(gid) individual(iidw) ///
            aggregation(mean) vce(robust) numthreads(16)
    }
    postclose WRPOST
    use "wr.dta", clear
    quietly keep if kind == "rc"
    assert _N == 19
    assert value == 0 if substr(tag, 1, 4) != "ind_"
    assert value == 198 if substr(tag, 1, 4) == "ind_"
}
else if ("`mode'" == "compare") {
    use "../`other'/wr.dta", clear
    rename (value text) (value_b text_b)
    tempfile b
    quietly save `b'
    use "../`leg'/wr.dta", clear
    merge 1:1 tag kind name i j using `b'
    list tag kind name i j if _merge != 3, noobs abbreviate(24)
    assert _merge == 3
    generate byte same = (value == value_b | (missing(value) & missing(value_b))) & text == text_b
    * A wall-clock duration differs between any two runs: present, not compared.
    generate byte timing = kind == "scalar" & strpos(name, "elapsed") > 0
    quietly count if timing & !same
    display as text "timing values not compared: " r(N)
    replace same = 1 if timing
    list tag kind name i j value value_b if !same, noobs abbreviate(24)
    list tag kind name text text_b if !same & text != text_b, noobs abbreviate(24)
    assert same
    quietly levelsof tag, local(tags) clean
    local ntags : word count `tags'
    display as text "compared cases (`ntags'): `tags'"
    assert `ntags' == 19
    local nfiles 0
    foreach t in outputs small ind {
        quietly describe using "../`other'/wr_`t'.dta", varlist
        local vb "`r(varlist)'"
        use "../`leg'/wr_`t'.dta", clear
        quietly describe, varlist
        assert "`r(varlist)'" == "`vb'"
        cf _all using "../`other'/wr_`t'.dta"
        local ++nfiles
    }
    display as text "full copies compared: `nfiles'"
}
else {
    display as error "unknown mode `mode'"
    exit 198
}
display as text "WR_PASS `mode'"
