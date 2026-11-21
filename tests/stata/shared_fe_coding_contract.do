* Phase 2 of the Stata-path plan: the reduced-system tier's certificate takes the
* tier's codes over for every effect whose indexer numbers the identifiers by
* their ascending compact ranks (design note Verifications/perf_stata_path_20261114/
* fase2/DESENHO_FASE2.md, section 6). Old package against new on synthetic data
* only. tests/stata/shared_fe_coding_contract.sh runs each mode in a fresh Stata
* process, in its own folder, with the package on XHDFE_STATA_ADOPATH and
* XHDFE_PROFILE_CPU=1 (the positive controls read stderr):
*
*   fit               every case below; e() (each scalar, macro and matrix, with
*                     names) and e(sample) (count and a signature) go to sf.dta
*   decline           the tier_t16 case alone (the runner sets the forced stage)
*   compare <a> <b>   folder a against folder b, bit for bit
*
* Data: 400 000 rows with four effects, each on equal shares of the rows in its
* own random order: sparse (40 000 levels, range 3 479 914), b89 (890 values of
* a range of 1000: compacts), c90 (900 of 1000, the holes next to both ends:
* keeps them, not shared) and dfull (500 of 500); and 200 000 rows whose effects
* have ranges of exactly 2n and 2n + 1 and 90 and 91 values of a range of 101.
version 16
clear all
set more off
set linesize 255
args mode leg other
local adopath : env XHDFE_STATA_ADOPATH
while (strlen(`"`adopath'"') > 1 & substr(`"`adopath'"', -1, 1) == "/") {
    local adopath = substr(`"`adopath'"', 1, strlen(`"`adopath'"') - 1)
}
macro drop XHDFE_REDUCED_PCG

* One effect: its levels u = 0..levels-1 on equal shares of the rows, in a fresh
* random order, mapped to values by `expr' (a function of u).
capture program drop sf_assign
program define sf_assign
    args name levels expr
    tempvar key
    generate double `key' = runiform()
    sort `key'
    generate long u = mod(_n - 1, `levels')
    generate long `name' = `expr'
    drop u
end

capture program drop sf_data
program define sf_data
    clear
    set seed 2026
    set obs 400000
    generate long rowid = _n
    sf_assign sparse 40000 "87 * u"
    * The products in parentheses: Stata evaluates u * a / b as u * (a / b), which
    * can lose the exact end of the range.
    sf_assign b89 890 "1000 + floor((u * 999) / 889)"
    sf_assign c90 900 "cond(u == 0, 5000, cond(u == 899, 5999, 5050 + u))"
    sf_assign dfull 500 "3000 + u"
    sort rowid
    generate double x1 = rnormal()
    generate double x2 = rnormal()
    generate double y = .5 * x1 - .25 * x2 + mod(sparse, 7) / 7 + b89 / 2000 - c90 / 6000 ///
        + dfull / 3500 + rnormal()
    generate byte w = 1 + floor(3 * runiform())
end

capture program drop sf_boundary_data
program define sf_boundary_data
    clear
    set seed 2027
    set obs 200000
    generate long rowid = _n
    sf_assign e2n 40000 "floor((u * 399999) / 39999)"
    sf_assign e2n1 40000 "floor((u * 400000) / 39999)"
    sf_assign f90 90 "11 + floor((u * 100) / 89)"
    sf_assign f91 91 "500 + floor((u * 100) / 90)"
    sort rowid
    generate double x1 = rnormal()
    generate double x2 = rnormal()
    generate double y = .5 * x1 - .25 * x2 + mod(e2n, 11) / 11 + f90 / 120 - f91 / 700 + rnormal()
end

* rc, every e() scalar, macro and matrix (with names) and e(sample).
capture program drop sf_record
program define sf_record
    args tag rc
    post SFPOST ("`tag'") ("rc") ("rc") (0) (0) (`rc') ("")
    local sc : e(scalars)
    foreach s of local sc {
        post SFPOST ("`tag'") ("scalar") ("`s'") (0) (0) (e(`s')) ("")
    }
    local mc : e(macros)
    foreach m of local mc {
        assert strlen(`"`e(`m')'"') < 2045
        post SFPOST ("`tag'") ("macro") ("`m'") (0) (0) (.) (`"`e(`m')'"')
    }
    local mt : e(matrices)
    foreach m of local mt {
        tempname M
        matrix `M' = e(`m')
        local rn : rownames `M'
        local cn : colnames `M'
        assert strlen(`"`rn'"') < 2045 & strlen(`"`cn'"') < 2045
        post SFPOST ("`tag'") ("rownames") ("`m'") (0) (0) (.) (`"`rn'"')
        post SFPOST ("`tag'") ("colnames") ("`m'") (0) (0) (.) (`"`cn'"')
        forvalues i = 1/`=rowsof(`M')' {
            forvalues j = 1/`=colsof(`M')' {
                post SFPOST ("`tag'") ("matrix") ("`m'") (`i') (`j') (`M'[`i', `j']) ("")
            }
        }
    }
    tempvar es sig
    quietly generate byte `es' = e(sample)
    quietly count if `es'
    post SFPOST ("`tag'") ("esample") ("count") (0) (0) (r(N)) ("")
    quietly generate double `sig' = rowid * `es'
    quietly summarize `sig', meanonly
    post SFPOST ("`tag'") ("esample") ("sum_of_rowid") (0) (0) (r(sum)) ("")
end

capture program drop sf_case
program define sf_case
    gettoken tag 0 : 0
    display as text "SF_CASE_BEGIN `tag'"
    capture noisily `0'
    local rc = _rc
    display as text "SF_CASE_END `tag' rc=`rc'"
    sf_record `tag' `rc'
end

if ("`adopath'" != "" & "`mode'" != "compare") {
    adopath ++ `"`adopath'"'
    quietly findfile xhdfe.ado
    assert `"`r(fn)'"' == `"`adopath'/xhdfe.ado"'
    display as text `"xhdfe.ado: `r(fn)'"'
}

local A "sparse b89 c90 dfull"
if ("`mode'" == "fit" | "`mode'" == "decline") {
    postfile SFPOST str24 tag str12 kind str48 name int i int j double value str2045 text ///
        using "sf.dta", replace
    sf_data
    if ("`mode'" == "decline") {
        sf_case tier_t16 xhdfe y x1 x2, absorb(`A') vce(cluster sparse) numthreads(16)
    }
    else {
        foreach t in 1 2 8 16 24 48 {
            sf_case tier_t`t' xhdfe y x1 x2, absorb(`A') vce(cluster sparse) numthreads(`t')
        }
        * Rows sorted by the sparse effect: its codes never fall (group_ptr).
        sort sparse rowid
        sf_case tier_sorted xhdfe y x1 x2, absorb(`A') vce(cluster sparse) numthreads(16)
        sort rowid
        sf_case tier_off xhdfe y x1 x2, absorb(`A') vce(cluster sparse) numthreads(16) reducedpcg(off)
        sf_case fast xhdfe y x1 x2, absorb(`A') vce(cluster sparse) numthreads(16) ///
            tolerancemode(xhdfe-fast)
        sf_case fweights xhdfe y x1 x2 [fw=w], absorb(`A') vce(robust) numthreads(16)
        sf_boundary_data
        sf_case boundary xhdfe y x1 x2, absorb(e2n e2n1 f90 f91) vce(robust) numthreads(16)
    }
    postclose SFPOST
    use "sf.dta", clear
    quietly keep if kind == "rc"
    assert value == 0
    assert _N == cond("`mode'" == "decline", 1, 11)
}
else if ("`mode'" == "compare") {
    use "../`other'/sf.dta", clear
    rename (value text) (value_b text_b)
    tempfile b
    quietly save `b'
    use "../`leg'/sf.dta", clear
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
    quietly count if kind == "scalar" & inlist(name, "threads_used", "parallel_workers_active")
    display as text "thread diagnostics compared: " r(N)
    assert r(N) == 2 * `ntags'
}
else {
    display as error "unknown mode `mode'"
    exit 198
}
display as text "SF_PASS `mode'"
