* Phase 1 of the Stata-path plan, sparse identifiers ranked on the team
* (src/sparse_rank.hpp): the Stata contract (test 6 of the design note
* Verifications/perf_stata_path_20261114/fase1/DESENHO_FASE1.md) and the
* reviewer's counterexample. Synthetic data only, made here from fixed seeds.
* tests/stata/sparse_rank_contract.sh runs the modes, each in a fresh Stata
* process, from one scratch directory:
*
*   fit <leg>          every fit below with the package on XHDFE_STATA_ADOPATH;
*                      e(b), e(V) and the scalars posted to sr_<leg>.dta, the row
*                      outputs (e(sample), groupvar labels, saved effects) to
*                      sr_<leg>_<part>.dta for the parts main, grouped, c8, c32
*                      and cbig
*   control <leg>      a few of the same fits, for the positive control of the
*                      runner (it reads the cpu_profile lines of stderr)
*   compare <a> <b>    every value of leg a equal to leg b, bit for bit (the CUDA
*                      fit's b and V within 1e-7 relative: CUDA is not
*                      bit-reproducible); every expected fit present
*
* Fits, numthreads(4) unless stated: comparable with the tier on and off, fast,
* cluster on a sparse grouping of the firm (the nested test), groupvar() with
* dofadjustments(firstpair) and the default, savefe, heterogeneous slopes
* (firm##c.z), group() and individual() with aggregation(mean) and (sum); the
* counterexample with 8 rows (numthreads(1): today's branch), 32 rows
* (numthreads(2)) and 2^17 x 8 rows with spread values; with XHDFE_SR_TEST_CUDA=1
* one fit with gpubackend(cuda), e(gpu_used)==1 asserted. The firm identifier is
* a long with a range of about 8.7 n (above 2n: the sparse branches; within the
* index's guard of 42.7 n). Each mode ends with the line SR_PASS <mode>.
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

capture program drop sr_main_data
program define sr_main_data
    clear
    set seed 20261114
    set obs 30000
    generate long worker = _n
    generate int len = 3 + floor(6 * runiform())
    generate int start = floor(2 * runiform())
    generate long home = 1 + floor(2500 * runiform())
    expand len
    bysort worker: generate int t = _n
    generate int year = 2000 + mod(start + t, 10)
    generate long firm = cond(runiform() < .15, 1 + floor(2500 * runiform()), home)
    * The raw firm identifier: an injective, sparse coding (range about 8.7 n);
    * its ascending order differs from the order of first appearance.
    local K = floor(8.7 * _N / 2500)
    generate long firm_raw = 37 + firm * `K'
    * A sparse grouping of the firm (the firm is nested in it).
    generate long firm_group_raw = 53 + floor(firm / 5) * `K'
    generate double z = rnormal()
    generate double x1 = rnormal()
    generate double x2 = rnormal() + mod(firm, 7) / 10
    generate double y = .5 * x1 - .25 * x2 + worker / 30000 + firm / 2500 + ///
        (year - 2000) / 10 + .2 * z + rnormal()
    drop len start home t
    generate long rowid = _n
end

capture program drop sr_group_data
program define sr_group_data
    clear
    set seed 7
    set obs 40000
    generate long gid = _n
    generate int size = 2 + floor(3 * runiform())
    generate double gy = rnormal()
    generate double gx = rnormal()
    * A sparse group-level effect.
    generate long gfe = 37 + mod(gid, 700) * 4099
    expand size
    bysort gid: generate int k = _n
    * Distinct individuals within a group, sparse raw identifiers.
    generate long iid = 11 + mod(gid * 7 + k * 104729, 20000) * 977
    drop size k
    generate long rowid = _n
end

capture program drop sr_counter_data
program define sr_counter_data
    args copies spread
    clear
    set obs `=8 * `copies''
    generate long rowid = _n
    generate long fe1 = cond(mod(_n - 1, 8) < 4, 1000, 10)
    generate long fe2 = cond(mod(_n - 1, 8) < 4, 500, 5)
    if (`spread') {
        * Values apart: the range exceeds 2n and the index fits its guard.
        replace fe1 = cond(fe1 == 1000, 10 * _N, 10)
        replace fe2 = cond(fe2 == 500, 9 * _N, 5)
    }
    set seed 11
    generate double x = rnormal()
    generate double y = .5 * x + rnormal()
end

* e(b), e(V) and the scalars of the last fit, under a tag.
capture program drop sr_post
program define sr_post
    args tag
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local k = colsof(`b')
    forvalues i = 1/`k' {
        post SRPOST ("`tag'") ("b") (`i') (0) (`b'[1, `i'])
        forvalues j = 1/`k' {
            post SRPOST ("`tag'") ("V") (`i') (`j') (`V'[`i', `j'])
        }
    }
    foreach s in N df_a df_r df_m rss iterations absorption_method_used threads_used ///
        reduced_pcg_status num_singletons df_a_nested N_clust converged {
        post SRPOST ("`tag'") ("`s'") (0) (0) (e(`s'))
    }
end

if ("`adopath'" != "" & "`mode'" != "compare") {
    adopath ++ `"`adopath'"'
    quietly findfile xhdfe.ado
    assert `"`r(fn)'"' == `"`adopath'/xhdfe.ado"'
    display as text `"xhdfe.ado: `r(fn)'"'
}

if ("`mode'" == "fit" | "`mode'" == "control") {
    local full = "`mode'" == "fit"
    postfile SRPOST str24 tag str24 what int i int j double value using "sr_`leg'.dta", replace

    sr_main_data
    local A "worker firm_raw year"
    quietly xhdfe y x1 x2, absorb(`A') vce(cluster worker) numthreads(4)
    sr_post cmp_on
    generate byte s_cmp_on = e(sample)
    quietly xhdfe y x1 x2, absorb(`A') vce(cluster worker) numthreads(4) reducedpcg(off)
    sr_post cmp_off
    quietly xhdfe y x1 x2, absorb(`A') vce(cluster firm_group_raw) numthreads(4)
    sr_post cluster_group
    quietly xhdfe y x1 x2, absorb(firm_raw worker year) vce(robust) numthreads(4) ///
        groupvar(gv_first) dofadjustments(firstpair)
    sr_post groupvar_first
    if (`full') {
        quietly xhdfe y x1 x2, absorb(`A') vce(cluster worker) numthreads(4) ///
            tolerancemode(xhdfe-fast)
        sr_post fast
        quietly xhdfe y x1 x2, absorb(firm_raw worker year) vce(robust) numthreads(4) ///
            groupvar(gv_all)
        sr_post groupvar_all
        quietly xhdfe y x1 x2, absorb(worker firm_raw year, savefe) vce(robust) numthreads(4)
        sr_post savefe
        foreach v of varlist __hdfe* {
            rename `v' sv_`v'
        }
        quietly xhdfe y x1 x2, absorb(firm_raw##c.z worker year) vce(robust) numthreads(4)
        sr_post slopes
        local cuda : env XHDFE_SR_TEST_CUDA
        if ("`cuda'" == "1") {
            quietly xhdfe y x1 x2, absorb(`A') vce(cluster worker) numthreads(4) ///
                gpubackend(cuda)
            assert e(gpu_used) == 1 & "`e(gpu_backend)'" == "cuda" & "`e(gpu_status)'" == "used"
            sr_post cuda_main
        }
        keep rowid s_* gv_* sv_*
        save "sr_`leg'_main.dta", replace

        sr_group_data
        foreach agg in mean sum {
            quietly xhdfe gy gx, absorb(gfe) group(gid) individual(iid) aggregation(`agg') ///
                vce(robust) numthreads(4)
            sr_post group_`agg'
            generate byte s_group_`agg' = e(sample)
        }
        keep rowid s_*
        save "sr_`leg'_grouped.dta", replace
    }

    * The reviewer's counterexample: labels 1 on the 1000/500 rows and 2 on the
    * 10/5 rows, by order of first appearance.
    foreach spec in "c8 1 0 1" "c32 4 0 2" "cbig 131072 1 4" {
        tokenize `spec'
        local name `1'
        local threads `4'
        sr_counter_data `2' `3'
        foreach dof in firstpair all {
            quietly xhdfe y x, absorb(fe1 fe2) vce(robust) numthreads(`threads') ///
                groupvar(gv_`name'_`dof') dofadjustments(`dof')
            sr_post `name'_`dof'
            assert gv_`name'_`dof' == cond(mod(rowid - 1, 8) < 4, 1, 2)
        }
        keep rowid gv_*
        if (`full') save "sr_`leg'_`name'.dta", replace
    }
    postclose SRPOST
}
else if ("`mode'" == "compare") {
    use "sr_`other'.dta", clear
    rename value value_b
    tempfile b
    quietly save `b'
    use "sr_`leg'.dta", clear
    merge 1:1 tag what i j using `b'
    generate byte cuda = substr(tag, 1, 5) == "cuda_"
    * The CUDA fit may be absent from one leg (the verify leg runs on the CPU only).
    assert _merge == 3 if !cuda
    keep if _merge == 3
    drop _merge
    generate byte same = value == value_b | (missing(value) & missing(value_b))
    replace same = 1 if cuda & inlist(what, "b", "V") & ///
        abs(value - value_b) <= 1e-7 * max(1, abs(value_b))
    list tag what i j value value_b if !same, noobs abbreviate(24)
    assert same
    * Never vacuous: every fit of the fit mode is there.
    foreach t in cmp_on cmp_off cluster_group groupvar_first fast groupvar_all savefe slopes ///
        group_mean group_sum c8_firstpair c8_all c32_firstpair c32_all cbig_firstpair cbig_all {
        quietly count if tag == "`t'"
        assert r(N) > 0
    }
    quietly levelsof tag, local(tags) clean
    display as text "compared fits: `tags'"
    foreach part in main grouped c8 c32 cbig {
        use "sr_`other'_`part'.dta", clear
        foreach v of varlist _all {
            if ("`v'" != "rowid") rename `v' B_`v'
        }
        tempfile rows
        quietly save `rows'
        use "sr_`leg'_`part'.dta", clear
        quietly describe, varlist
        local vars `r(varlist)'
        merge 1:1 rowid using `rows', assert(match) nogenerate
        local compared 0
        foreach v of local vars {
            if ("`v'" == "rowid") continue
            assert `v' == B_`v' | (missing(`v') & missing(B_`v'))
            local ++compared
        }
        assert `compared' > 0
        display as text "rows `part': `compared' variables equal over " _N " rows"
    }
}
else {
    display as error "unknown mode `mode'"
    exit 198
}
display as text "SR_PASS `mode'"
