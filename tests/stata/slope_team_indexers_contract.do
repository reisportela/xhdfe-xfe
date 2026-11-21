* Phase 5a of the Stata-path plan: the identifier builds of the route with
* heterogeneous slopes on the team, through the plugin. Synthetic data only, made
* here from fixed seeds. tests/stata/slope_team_indexers_contract.sh runs the legs
* (the new package, the new package with XHDFE_SLOPE_TEAM_INDEXERS=0, the frozen
* package), each in a fresh Stata process, and compares their text files line for
* line.
*
*   fit <leg>   every fit below; writes st_<leg>.txt: e(b), the lower triangle of
*               e(V) and every scalar of each fit in hexadecimal (%21x), and for
*               xfepout the scalars and a checksum of each partialled variable
*
* About 1.1 M rows (above 2^20). The firm carries the slopes; its identifier is a
* long, sparse (range about 8.7 n: above 2n, the sparse branches, and within the
* phase-1 index's budget, so that the team builders take the index), or a wide
* double above the int32 range (the plugin's wide-identifier path, phase 3).
* In firms 0, 3, 6, ... the slope variable is almost constant within the firm (the
* determinant boundary of the slope rank; variable zb, in boundary_fast only). Fits:
*   one_raw_fast  absorb(worker firm_raw##c.z year), fast, numthreads(8)
*   two_raw_cmp   absorb(worker firm_raw##c.z firm_raw#c.z2 year), comparable,
*                 numthreads(8) (the secondreg shape)
*   one_wide_fast absorb(worker firm_wide##c.z year), fast, numthreads(8)
*   one_raw_auto  as one_raw_fast with the automatic thread count (below the
*                 4194304-row gate: no team)
*   boundary_fast absorb(worker firm_raw##c.zb year), fast, numthreads(8),
*                 maxiter(3000)
*   xfepout       xfepout y x1 x2, absorb(worker firm_raw##c.z year): the ado
*                 refuses heterogeneous slopes (r(198)); the return code is
*                 compared (partial_out() with slopes, the second caller of the
*                 slope rank, is covered through the C++ entry)
* Each mode ends with the line STI_PASS <mode>.
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

capture program drop sti_data
program define sti_data
    clear
    set seed 20261115
    set obs 200000
    generate long worker = _n
    generate int len = 3 + floor(6 * runiform())
    generate int start = floor(2 * runiform())
    generate long home = floor(12000 * runiform())
    expand len
    bysort worker: generate int t = _n
    generate int year = 2000 + mod(start + t, 10)
    generate long firm = cond(runiform() < .15, floor(12000 * runiform()), home)
    * The raw firm identifier: injective and sparse, a range of about 8.7 n (above
    * 2n: the sparse branches; within the phase-1 index's budget of about 42.7 n).
    local K = floor(8.7 * _N / 12000)
    generate long firm_raw = 37 + firm * `K'
    generate double firm_wide = 4 * firm_raw + 3e9
    * zb: the determinant boundary (firms 0, 3, 6, ...: 1 + eps u), used by the
    * fast fit boundary_fast only (ill-conditioned on purpose).
    generate double eps = cond(mod(firm, 3) == 0, cond(mod(firm, 9) == 0, 1e-7, cond(mod(firm, 9) == 3, 1e-6, 1e-5)), .)
    generate double z = rnormal()
    generate double zb = cond(missing(eps), z, 1 + eps * (runiform() - .5))
    generate double z2 = rnormal()
    generate double x1 = rnormal()
    generate double x2 = rnormal() + mod(firm, 7) / 10
    generate double y = .5 * x1 - .25 * x2 + worker / 200000 + mod(firm, 13) / 100 + ///
        (year - 2000) / 10 + .2 * z + rnormal()
    drop len start home t eps
end

capture program drop sti_post
program define sti_post
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

if ("`mode'" == "fit") {
    sti_data
    display as text "STI_DATA N=" _N
    tempname fh
    file open `fh' using "st_`leg'.txt", write text replace
    local A "vce(cluster worker) gpubackend(cpu)"
    quietly xhdfe y x1 x2, absorb(worker firm_raw##c.z year) `A' tolerancemode(xhdfe-fast) numthreads(8)
    sti_post `fh' one_raw_fast
    quietly xhdfe y x1 x2, absorb(worker firm_raw##c.z firm_raw#c.z2 year) `A' tolerancemode(reghdfe-comparable) numthreads(8)
    sti_post `fh' two_raw_cmp
    quietly xhdfe y x1 x2, absorb(worker firm_wide##c.z year) `A' tolerancemode(xhdfe-fast) numthreads(8)
    sti_post `fh' one_wide_fast
    quietly xhdfe y x1 x2, absorb(worker firm_raw##c.z year) `A' tolerancemode(xhdfe-fast)
    sti_post `fh' one_raw_auto
    quietly xhdfe y x1 x2, absorb(worker firm_raw##c.zb year) `A' tolerancemode(xhdfe-fast) numthreads(8) maxiter(3000)
    sti_post `fh' boundary_fast
    * xfepout refuses heterogeneous slopes (r(198) in the ado, before the plugin):
    * the same refusal in every leg. partial_out() with slopes is exercised through
    * the C++ entry by tests/slope_team_indexers_omp_region.cpp.
    capture quietly xfepout y x1 x2, absorb(worker firm_raw##c.z year) generate(po_) numthreads(8) gpubackend(cpu)
    file write `fh' "xfepout_slopes rc " (_rc) _n
    file close `fh'
    display as text "STI_PASS fit"
}
else {
    display as error "unknown mode `mode'"
    exit 198
}
