* Phase 5b of the Stata-path plan: multi-way clustering without the repeated passes
* (DESENHO_FASE5B.md, sections 1, 2 and 5; round 23), through the plugin. Synthetic
* data only, made here from fixed seeds. tests/stata/cluster_shared_passes_contract.sh
* runs the legs (the new package; the new package with XHDFE_CLUSTER_SHARED_PASSES=0;
* the frozen package, a previous build with phase 5b, with the switch at 0 as well),
* each in a fresh Stata process, and compares their text files line for line.
*
*   fit <leg>   every fit below; writes csp_<leg>.txt: e(b), the lower triangle of
*               e(V) and every scalar of each fit in hexadecimal (%21x) (e(N_clust),
*               e(N_clust#), e(N_clustervars), e(df_r) among them), for savefe and
*               xfepout the sums of the saved or partialled variables, and for every
*               clustered fit the line "<tag> counts ok" once e(N_clust#) equals the
*               distinct count of each cluster variable on e(sample) and e(N_clust)
*               their minimum
*
* About 140 k rows; worker (25 000, of which 300 one-row workers removed as
* singletons), firm (1 500), region (50), sector (11), year (8); sparse versions of
* worker and firm (range about 8.7 n: the sparse branches of the meats).
* Each mode ends with the line CSP_PASS <mode>.
version 16
clear all
set more off
set linesize 255
args mode leg
local adopath : env XHDFE_STATA_ADOPATH
while (strlen(`"`adopath'"') > 1 & substr(`"`adopath'"', -1, 1) == "/") {
    local adopath = substr(`"`adopath'"', 1, strlen(`"`adopath'"') - 1)
}
adopath ++ `"`adopath'"'
discard
quietly findfile xhdfe.plugin
assert `"`r(fn)'"' == `"`adopath'/xhdfe.plugin"'
quietly findfile xfepout.plugin
assert `"`r(fn)'"' == `"`adopath'/xfepout.plugin"'

capture program drop csp_data
program define csp_data
    clear
    set seed 20261115
    set obs 25000
    generate long worker = _n
    generate int len = 3 + floor(6 * runiform())
    replace len = 1 in 1/300
    generate int start = floor(8 * runiform())
    generate long home = floor(1500 * runiform())
    expand len
    bysort worker: generate int t = _n
    generate long ord = _n
    generate int year = 2000 + mod(start + t, 8)
    generate long firm = cond(runiform() < .15, floor(1500 * runiform()), home)
    local K = floor(8.7 * _N / 1500)
    generate long firm_sparse = 37 + firm * `K'
    local K2 = floor(8.7 * _N / 25000)
    generate long worker_sparse = 11 + worker * `K2'
    generate long region = floor(firm / 30)
    generate long sector = mod(firm * 7, 11)
    generate double x1 = rnormal()
    generate double x2 = rnormal() + mod(firm, 7) / 10
    generate double x3 = x1 + x2
    generate double y = .5 * x1 - .25 * x2 + worker / 25000 + mod(firm, 13) / 100 + (year - 2000) / 10 + rnormal()
    generate double w = .5 + 1.5 * runiform()
    generate double fw = 1 + floor(3 * runiform())
    * Firms 0..9 only in rows of weight zero for the zero-weight fit.
    generate double w0 = cond(firm < 10, 0, w)
    generate double u = runiform()
    drop len start home
end

capture program drop csp_post
program define csp_post
    args fh tag
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    forvalues i = 1/`=colsof(`b')' {
        file write `fh' "`tag' b `i' " %21x (`b'[1, `i']) _n
        forvalues j = 1/`i' {
            file write `fh' "`tag' V `i' `j' " %21x (`V'[`i', `j']) _n
        }
    }
    local scalars : e(scalars)
    foreach s of local scalars {
        if ("`s'" == "auto_retry_elapsed_seconds") continue
        file write `fh' "`tag' `s' " %21x (e(`s')) _n
    }
end

* e(N_clust#) against the distinct count of each cluster variable on e(sample), and
* e(N_clust) against their minimum.
capture program drop csp_counts
program define csp_counts
    args fh tag
    local vars "`e(clustvar)'"
    local k = 0
    local minimum = .
    tempvar s
    generate byte `s' = e(sample)
    foreach v of local vars {
        local ++k
        preserve
        quietly keep if `s'
        quietly bysort `v': generate byte csp_first = _n == 1
        quietly count if csp_first
        local distinct = r(N)
        restore
        assert e(N_clust`k') == `distinct'
        local minimum = min(`minimum', `distinct')
    }
    assert e(N_clustervars) == `k'
    assert e(N_clust) == `minimum'
    drop `s'
    file write `fh' "`tag' counts ok `k'" _n
end

capture program drop csp_sums
program define csp_sums
    args fh tag stub
    foreach v of varlist `stub'* {
        quietly summarize `v', meanonly
        file write `fh' "`tag' sum `v' " %21x (r(sum)) " n " (r(N)) _n
    }
end

if ("`mode'" == "fit") {
    csp_data
    display as text "CSP_DATA N=" _N
    tempname fh
    file open `fh' using "csp_`leg'.txt", write text replace
    local A "gpubackend(cpu) numthreads(4)"
    local F "absorb(worker firm year)"
    quietly xhdfe y x1 x2, `F' vce(cluster worker firm) `A'
    csp_post `fh' tw_sorted
    csp_counts `fh' tw_sorted
    quietly xhdfe y x1 x2, `F' vce(cluster worker firm_sparse) `A'
    csp_post `fh' tw_sparse
    csp_counts `fh' tw_sparse
    quietly xhdfe y x1 x2, `F' vce(cluster worker_sparse firm_sparse) `A'
    csp_post `fh' tw_sparse2
    csp_counts `fh' tw_sparse2
    sort u
    quietly xhdfe y x1 x2, `F' vce(cluster worker firm) `A'
    csp_post `fh' tw_unsorted
    csp_counts `fh' tw_unsorted
    sort ord
    quietly xhdfe y x1 x2 [aw=w], `F' vce(cluster worker firm_sparse) `A'
    csp_post `fh' tw_aw
    csp_counts `fh' tw_aw
    quietly xhdfe y x1 x2 [pw=w], `F' vce(cluster worker firm_sparse) `A'
    csp_post `fh' tw_pw
    csp_counts `fh' tw_pw
    quietly xhdfe y x1 x2 [fw=fw], `F' vce(cluster worker firm_sparse) `A'
    csp_post `fh' tw_fw
    csp_counts `fh' tw_fw
    quietly xhdfe y x1 x2 [aw=w0], `F' vce(cluster worker firm_sparse) `A'
    csp_post `fh' tw_zero_w
    csp_counts `fh' tw_zero_w
    quietly xhdfe y x1 x2, `F' vce(cluster worker firm_sparse) ssc(gdf=min gadj=1) `A'
    csp_post `fh' tw_min
    quietly xhdfe y x1 x2, `F' vce(cluster worker firm_sparse) ssc(gdf=conventional gadj=0) `A'
    csp_post `fh' tw_conv_noadj
    quietly xhdfe y x1 x2, `F' vce(cluster worker firm_sparse region) `A'
    csp_post `fh' three_way
    csp_counts `fh' three_way
    quietly xhdfe y x1 x2, `F' vce(cluster region worker firm_sparse) `A'
    csp_post `fh' three_way_min0
    csp_counts `fh' three_way_min0
    quietly xhdfe y x1 x2, `F' vce(cluster worker firm_sparse sector region) `A'
    csp_post `fh' four_way
    csp_counts `fh' four_way
    quietly xhdfe y x1 x2, `F' vce(cluster worker firm_sparse) noconstant `A'
    csp_post `fh' tw_noconstant
    quietly xhdfe y x1 x2, noabsorb vce(cluster worker firm_sparse) `A'
    csp_post `fh' tw_noabsorb
    csp_counts `fh' tw_noabsorb
    * An omitted regressor (no reconstruction). Clusters firm and region: with worker
    * and a firm id the multiway intercept fallback crashes today in every arm
    * (fase5/repro_omitido_dois_clusters.py; not touched in phase 5b).
    quietly xhdfe y x1 x2 x3, `F' vce(cluster firm region) `A'
    csp_post `fh' tw_omitted
    quietly xhdfe y x1 x2, `F' vce(cluster worker firm_sparse) tolerancemode(xhdfe-fast) `A'
    csp_post `fh' tw_fast
    quietly xhdfe y x1 x2, absorb(worker firm year, savefe) vce(cluster worker firm_sparse) tolerancemode(xhdfe-fast) maxiter(3000) `A'
    csp_post `fh' tw_savefe
    csp_sums `fh' tw_savefe __hdfe
    drop __hdfe*
    quietly xhdfe y x1 x2, `F' vce(cluster worker) `A'
    csp_post `fh' one_way
    * xfepout with two cluster variables: they enter only the nested degrees of freedom.
    quietly xfepout y x1 x2, `F' clustervars(worker firm_sparse) generate(po_) numthreads(4) gpubackend(cpu)
    local scalars : e(scalars)
    foreach s of local scalars {
        file write `fh' "xfepout `s' " %21x (e(`s')) _n
    }
    csp_sums `fh' xfepout po_
    file close `fh'
    display as text "CSP_PASS fit"
}
else {
    display as error "unknown mode `mode'"
    exit 198
}
