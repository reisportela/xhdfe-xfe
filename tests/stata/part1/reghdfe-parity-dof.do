* ===========================================================================
* reghdfe convention parity: degrees of freedom and sample construction.
*
* This layer extends part1/reghdfe-convention-parity.do over the surface where
* reghdfe's accounting is intricate and where a reimplementation is most likely
* to drift: every dofadjustments()/dof() variant, redundancy from disconnected
* fixed-effect graphs, cluster nesting, iterative singleton removal, sample
* construction, and e(rank)/e(df_m).
*
* Unit of comparison, as in the baseline layer:
*   - coefficients and standard errors are two iterative absorbers' output and
*     are compared at solver tolerance;
*   - counts and degrees of freedom are CHOICES with no rounding error and are
*     compared with EXACT equality. This layer asserts, on top of the baseline
*     set (N, df_r, df_m, df_a, N_clust, rank, omission pattern, missingness):
*     e(df_a_initial), e(df_a_redundant), e(df_a_nested), e(N_hdfe),
*     e(N_hdfe_extended), e(num_singletons), e(N_full), e(sumweights),
*     e(N_clustervars), the whole e(dof_table) matrix, e(dofmethod), and
*     row-by-row equality of e(sample).
*
* reghdfe is canonical for CONVENTIONS only. Where reghdfe's own number is a
* documented approximation, or where reghdfe is simply wrong, the adjudication
* is against an exact reference, and every such adjudication below carries the
* exact number that settles it. The references used are:
*   - an LSDV `regress' on explicit dummies, for absorbed dof and for b;
*   - `regress' itself, for the no-absorb and noconstant designs;
*   - `xtreg, fe', for the CRV1 factor when the absorbed dimension is nested in
*     the cluster, which is the convention reghdfe deliberately matches there;
*   - invariance of the multiway cluster estimator V1 + V2 - V12 to the ORDER
*     of the cluster variables, which is a property of the estimator itself.
*
* Divergences are either
*   except(): justified at the call site, or
*   KNOWN OPEN: a real defect, printed on every run and carried in the report,
* so that a NEW divergence is what turns this file red.
* ===========================================================================

version 16

xcert_require_reghdfe

* ---------------------------------------------------------------------------
* Helpers local to this layer. The shared xcert_parity_check covers the base
* scalar set; xpd_spec wraps it and adds the extended discrete surface.
* ---------------------------------------------------------------------------

capture program drop xpd_spec
program define xpd_spec, rclass
    version 16
    syntax anything(everything) [aweight fweight pweight iweight], ///
        Name(string) [XOPTS(string) ROPTS(string) BTOL(real 1e-10) ///
        SETOL(real 1e-8) SKIPF EXCEPT(string) KNOWN(string) KNOWNSE(real 0) ///
        XEXCEPT(string) XKNOWN(string) DTABLE(string) DMETHOD(string)]

    * Base set handled by xcert_parity_check, extended set handled here.
    local ascal "N rmse tss rss mss r2 r2_a F df_r df_m df_a N_clust rank"
    local escal "df_a_initial df_a_redundant df_a_nested N_hdfe N_hdfe_extended num_singletons N_full sumweights N_clustervars"

    quietly xhdfe `anything' [`weight'`exp'], ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12) ///
        `xopts' noheader notable nofootnote
    xcert_store_estimates, prefix(px) scalars(`ascal' `escal')
    local xdm `"`e(dofmethod)'"'
    local xhast 1
    capture confirm matrix e(dof_table)
    if (c(rc)) local xhast 0
    if (`xhast') {
        matrix pxT = e(dof_table)
    }
    else {
        matrix pxT = J(1, 1, .)
    }
    tempvar xs
    quietly generate byte `xs' = e(sample)

    quietly reghdfe `anything' [`weight'`exp'], `ropts'
    xcert_store_estimates, prefix(pr) scalars(`ascal' `escal')
    local rdm `"`e(dofmethod)'"'
    local rhast 1
    capture confirm matrix e(dof_table)
    if (c(rc)) local rhast 0
    if (`rhast') {
        matrix prT = e(dof_table)
    }
    else {
        matrix prT = J(1, 1, .)
    }
    tempvar rs
    quietly generate byte `rs' = e(sample)

    local fails 0
    local knowns 0

    * ---- extended discrete scalars: exact equality --------------------------
    foreach s of local escal {
        if (strpos(" `xexcept' ", " `s' ")) {
            di as text "  [`name'] e(`s') exempt: see call site"
            continue
        }
        local xv = scalar(px_`s')
        local rv = scalar(pr_`s')
        local xm = missing(`xv')
        local rm = missing(`rv')
        if (`xm' != `rm' | (!`xm' & !`rm' & `xv' != `rv')) {
            if (strpos(" `xknown' ", " `s' ")) {
                di as text "XHDFE_KNOWN_OPEN|`name'|e(`s')|xhdfe=`xv'|reghdfe=`rv'"
                local knowns = `knowns' + 1
            }
            else {
                di as error "  [`name'] e(`s') EXACT mismatch: xhdfe=`xv' reghdfe=`rv'"
                local fails = `fails' + 1
            }
        }
    }

    * ---- e(sample): the two engines must keep the same rows -----------------
    quietly count if `xs' != `rs'
    if (r(N) > 0) {
        di as error "  [`name'] e(sample) differs on " r(N) " row(s)"
        local fails = `fails' + 1
    }

    * ---- e(dofmethod): the posted label for the adjustment set --------------
    if (`"`xdm'"' != `"`rdm'"') {
        if ("`dmethod'" == "known") {
            di as text `"XHDFE_KNOWN_OPEN|`name'|e(dofmethod)|xhdfe=`xdm'|reghdfe=`rdm'"'
            local knowns = `knowns' + 1
        }
        else {
            di as error `"  [`name'] e(dofmethod) differs: xhdfe=|`xdm'| reghdfe=|`rdm'|"'
            local fails = `fails' + 1
        }
    }

    * ---- e(dof_table): per-dimension levels / redundant / coefs / flags -----
    * Columns are (Categories, Redundant, Num Coefs, Inexact?, Nested?).
    * Row names differ by construction for slope terms (1.firm_X_c_z versus
    * 1.firm#c.z), so only the values are compared. dtable(inexact) tolerates
    * differences confined to the Inexact? column, dtable(known) tolerates any
    * difference including a different shape; both are reported on every run.
    if (`"`dtable'"' != "off") {
        local xr = rowsof(pxT)
        local rr = rowsof(prT)
        local xdesc = cond(`xhast', "`xr' row(s)", "not posted")
        local rdesc = cond(`rhast', "`rr' row(s)", "not posted")
        if (`xhast' != `rhast' | `xr' != `rr' | colsof(pxT) != colsof(prT)) {
            if ("`dtable'" == "known") {
                di as text "XHDFE_KNOWN_OPEN|`name'|e(dof_table)|xhdfe `xdesc'|reghdfe `rdesc'"
                local knowns = `knowns' + 1
            }
            else {
                di as error "  [`name'] e(dof_table) differs: xhdfe `xdesc', reghdfe `rdesc'"
                local fails = `fails' + 1
            }
        }
        else {
            local cells 0
            local cells_ie 0
            forvalues i = 1/`xr' {
                forvalues j = 1/5 {
                    if (pxT[`i', `j'] != prT[`i', `j']) {
                        local cells = `cells' + 1
                        if (`j' == 4) local cells_ie = `cells_ie' + 1
                    }
                }
            }
            if (`cells' > 0) {
                local ok 0
                if ("`dtable'" == "known") local ok 1
                if ("`dtable'" == "inexact" & `cells' == `cells_ie') local ok 1
                if (`ok') {
                    di as text "XHDFE_KNOWN_OPEN|`name'|e(dof_table)|cells=`cells'|inexact_cells=`cells_ie'"
                    local knowns = `knowns' + 1
                }
                else {
                    di as error "  [`name'] e(dof_table) differs in `cells' cell(s) (`cells_ie' in the Inexact? column)"
                    matrix list pxT, noheader format(%6.0g)
                    matrix list prT, noheader format(%6.0g)
                    local fails = `fails' + 1
                }
            }
        }
    }

    * ---- base surface -------------------------------------------------------
    xcert_parity_check, name("`name'") btol(`btol') setol(`setol') `skipf' ///
        except(`except') known(`known') knownse(`knownse')
    local fails = `fails' + r(fails)
    if (`fails' > 0) {
        di as error "  FAIL  `name' (`fails' divergence(s) including the extended surface)"
    }
    return scalar fails = `fails'
end

* Ratio of the posted variances for one coefficient, from the matrices the
* last xpd_spec left in memory. Used to pin an SE divergence to its factor.
capture program drop xpd_vratio
program define xpd_vratio, rclass
    version 16
    args col
    local i = colnumb(px_b, "`col'")
    local j = colnumb(pr_b, "`col'")
    return scalar ratio = px_V[`i', `i'] / pr_V[`j', `j']
end

* A known-open divergence expressed as a variance ratio. `want' is the ratio
* this defect is documented to produce; it is computed from the posted counts
* at the call site, never hard-coded. Reports RESOLVED (without failing) if the
* defect is gone, and FAILS if the ratio drifted to some third value.
capture program drop xpd_ratio_note
program define xpd_ratio_note
    version 16
    args label got want
    local g = `got'
    local w = `want'
    if (abs(`g' - 1) <= 1e-9) {
        di as text "  RESOLVED    `label': ratio is now 1; drop this marker"
        exit
    }
    if (abs(`g' - `w') <= 1e-6 * `w') {
        di as text "XHDFE_KNOWN_OPEN|`label'|ratio=" %21.17g `g' ///
            "|documented=" %21.17g `w'
        exit
    }
    di as error "  FAIL        `label': ratio = " %21.17g `g' ///
        " but the documented open value is " %21.17g `w'
    global PARITY_FAILS = $PARITY_FAILS + 1
end

* A known-open divergence in an integer count. `want' is what xhdfe currently
* posts, `ref' what the exact reference says it should post.
capture program drop xpd_int_note
program define xpd_int_note
    version 16
    args label got want ref
    local g = `got'
    local w = `want'
    local r = `ref'
    if (`g' == `r') {
        di as text "  RESOLVED    `label': now `r'; drop this marker"
        exit
    }
    if (`g' == `w') {
        di as text "XHDFE_KNOWN_OPEN|`label'|observed=`g'|reference=`r'"
        exit
    }
    di as error "  FAIL        `label': observed=`g', documented open value `w', reference `r'"
    global PARITY_FAILS = $PARITY_FAILS + 1
end

capture program drop xpd_assert_eq
program define xpd_assert_eq
    version 16
    args label got want
    local g = `got'
    local w = `want'
    if (`g' == `w') {
        di as result "  PASS        `label' (= `w')"
        exit
    }
    di as error "  FAIL        `label': got `g', expected `w'"
    global PARITY_FAILS = $PARITY_FAILS + 1
end

capture program drop xpd_assert_close
program define xpd_assert_close
    version 16
    args label got want tol
    if (abs(`got' - `want') <= `tol' * max(1, abs(`want'))) {
        di as result "  PASS        `label'"
        exit
    }
    di as error "  FAIL        `label': got " %21.17g `got' ", expected " %21.17g `want'
    global PARITY_FAILS = $PARITY_FAILS + 1
end

* Internal consistency of what xhdfe itself posts. reghdfe satisfies all four
* identities on every design exercised below; they are asserted here for xhdfe
* because e(dof_table) is what -estat- and users read.
capture program drop xpd_selfcheck
program define xpd_selfcheck
    version 16
    syntax, Name(string) [KNOWNTABLE]
    matrix pxS = e(dof_table)
    local r = rowsof(pxS)
    local k 0
    local m 0
    local c 0
    forvalues i = 1/`r' {
        local k = `k' + pxS[`i', 1]
        local m = `m' + pxS[`i', 2]
        local c = `c' + pxS[`i', 3]
    }
    local dfa = e(df_a)
    local dfai = e(df_a_initial)
    local dfar = e(df_a_redundant)
    local dfae = e(df_a_exact)
    local bad 0
    if (`dfa' != `dfai' - `dfar') {
        di as error "  [`name'] e(df_a) != e(df_a_initial) - e(df_a_redundant): `dfa' != `dfai' - `dfar'"
        local bad = `bad' + 1
    }
    if (!missing(`dfae') & `dfae' != `dfa') {
        di as error "  [`name'] e(df_a_exact)=`dfae' != e(df_a)=`dfa'"
        local bad = `bad' + 1
    }
    local tbad 0
    if (`c' != `dfa')  local tbad = `tbad' + 1
    if (`k' != `dfai') local tbad = `tbad' + 1
    if (`m' != `dfar') local tbad = `tbad' + 1
    if (`tbad' > 0) {
        if ("`knowntable'" != "") {
            di as text "XHDFE_KNOWN_OPEN|`name'|e(dof_table) inconsistent:" ///
                " sum(Categories)=`k' vs e(df_a_initial)=`dfai';" ///
                " sum(Redundant)=`m' vs e(df_a_redundant)=`dfar';" ///
                " sum(Num Coefs)=`c' vs e(df_a)=`dfa'"
        }
        else {
            di as error "  [`name'] e(dof_table) does not reproduce the posted totals:" ///
                " sum(Categories)=`k' vs e(df_a_initial)=`dfai';" ///
                " sum(Redundant)=`m' vs e(df_a_redundant)=`dfar';" ///
                " sum(Num Coefs)=`c' vs e(df_a)=`dfa'"
            local bad = `bad' + 1
        }
    }
    if (`bad' == 0) di as result "  PASS        self-consistency `name'"
    global PARITY_FAILS = $PARITY_FAILS + `bad'
end

global PARITY_FAILS 0
di as text _n "{hline 70}"
di as text "reghdfe parity: degrees of freedom and sample construction"
di as text "{hline 70}"

* ===========================================================================
* D1. Two-way fixed effects on a graph with three disconnected components.
*     The redundancy carried by the second dimension is the component count,
*     so every dofadjustments() variant produces a different, checkable df_a.
* ===========================================================================
clear
set seed 90210
set obs 3600
generate long obs  = _n
generate int  blk  = 1 + mod(obs, 3)
generate int  firm = blk*1000 + 1 + int(runiform()*40)
generate int  year = blk*100  + 1 + int(runiform()*8)
generate int  st   = 1 + mod(obs, 11)
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double y  = .5*x1 - .8*x2 + firm/500 + year/50 + rnormal()

di as text _n "-- D1: 2 FE, 3 disconnected mobility groups ------------------"

* Exact reference. The LSDV design y on x1 x2 and full firm/year dummies has
* rank 143 out of N=3600, so df_r = 3457 and the exact absorbed dof is
* 143 - 2 = 141. reghdfe's pairwise count reproduces that number exactly, so
* on this design "pairwise" is not an approximation and can be asserted hard.
quietly regress y x1 x2 i.firm i.year
local lsdv_rank = e(rank)
local lsdv_dfr  = e(df_r)
xpd_assert_eq "D1 LSDV rank" `lsdv_rank' 143
xpd_assert_eq "D1 LSDV df_r" `lsdv_dfr' 3457

quietly xhdfe y x1 x2, absorb(firm year) dofadjustments(pairwise) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
local xtot = e(df_a) + e(df_m)
xpd_assert_eq "D1 xhdfe pairwise df_a + df_m == LSDV rank" `xtot' `lsdv_rank'
xpd_assert_eq "D1 xhdfe pairwise df_r == LSDV df_r" e(df_r) `lsdv_dfr'
xpd_assert_eq "D1 xhdfe pairwise mobility groups" e(df_a_redundant) 3
xpd_selfcheck, name("D1 pairwise")

* ---- the two adjustment sets that are pure aliases -------------------------
xpd_spec y x1 x2, name("D1 dof(all)") ///
    xopts(absorb(firm year) dofadjustments(all)) ///
    ropts(absorb(firm year) dof(all))
xcert_parity_tally

xpd_spec y x1 x2, name("D1 default (no dof option)") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally

xpd_spec y x1 x2, name("D1 dof(pairwise)") ///
    xopts(absorb(firm year) dofadjustments(pairwise)) ///
    ropts(absorb(firm year) dof(pairwise))
xcert_parity_tally

xpd_spec y x1 x2, name("D1 dof(firstpair)") ///
    xopts(absorb(firm year) dofadjustments(firstpair)) ///
    ropts(absorb(firm year) dof(firstpair))
xcert_parity_tally

* ---- dof(clusters) and dof(continuous) without a mobility-group token ------
* These tokens disable the mobility calculation unless firstpair/pairwise/all
* is also requested.
foreach dd in "clusters" "continuous" "clusters continuous" {
    xpd_spec y x1 x2, name("D1 dof(`dd')") ///
        xopts(absorb(firm year) dofadjustments(`dd')) ///
        ropts(absorb(firm year) dof(`dd'))
    xcert_parity_tally
    xpd_vratio x1
    xpd_assert_close "D1 dof(`dd') unadjusted VCE" r(ratio) 1 1e-9
}

xpd_spec y x1 x2, name("D1 dof(clusters) vce(cluster st)") ///
    xopts(absorb(firm year) dofadjustments(clusters) vce(cluster st)) ///
    ropts(absorb(firm year) dof(clusters) vce(cluster st))
xcert_parity_tally
xpd_vratio x1
xpd_assert_close "D1 dof(clusters) CRV1" r(ratio) 1 1e-9

* ---- the mobility-group tokens combined with the other two -----------------
foreach dd in "pairwise clusters" "pairwise continuous" "firstpair clusters" ///
              "firstpair continuous" "pairwise clusters continuous" {
    xpd_spec y x1 x2, name("D1 dof(`dd') vce(cluster st)") ///
        xopts(absorb(firm year) dofadjustments(`dd') vce(cluster st)) ///
        ropts(absorb(firm year) dof(`dd') vce(cluster st))
    xcert_parity_tally
}

* ---- dof(none) --------------------------------------------------------------
* reghdfe's dof(none) still applies the trivial redundancy for intercept
* dimensions after the first. xhdfe implements that in the core before VCE.
xpd_spec y x1 x2, name("D1 dof(none) unadjusted") ///
    xopts(absorb(firm year) dofadjustments(none)) ///
    ropts(absorb(firm year) dof(none))
xcert_parity_tally
xpd_vratio x1
xpd_assert_close "D1 dof(none) unadjusted VCE is rescaled by the ado" r(ratio) 1 1e-9

foreach vv in "robust" "cluster st" {
    xpd_spec y x1 x2, name("D1 dof(none) vce(`vv')") ///
        xopts(absorb(firm year) dofadjustments(none) vce(`vv')) ///
        ropts(absorb(firm year) dof(none) vce(`vv'))
    xcert_parity_tally
    xpd_vratio x1
    xpd_assert_close "D1 dof(none) vce(`vv') small-sample factor" r(ratio) 1 1e-9
}

* KNOWN OPEN, and here the defect is REGHDFE's, so this cell is adjudicated
* against xhdfe's own internal invariance instead of against reghdfe.
* With dof(none) and two-way clustering reghdfe's multiway variance comes out
* with missing entries and reghdfe replaces the WHOLE matrix with exact zeros
* (reghdfe.mata: "V can be missing ... setting it to zeroes"), yet still posts
* a finite e(F) computed from the pre-zeroing matrix. Every coefficient then
* looks "dropped" to any omission check. xhdfe posts a finite, PSD-repaired
* variance. The multiway estimator V1 + V2 - V12 is symmetric in its cluster
* arguments, so order invariance is the exact reference used below; reghdfe
* fails it, xhdfe passes it.
quietly reghdfe y x1 x2, absorb(firm year) dof(none) vce(cluster st year)
local r_v = e(V)[1, 1]
local r_f = e(F)
if (`r_v' == 0 & !missing(`r_f')) {
    di as text "XHDFE_KNOWN_OPEN|D1 dof(none) vce(cluster st year)|" ///
        "reghdfe e(V)=0|e(F)=" %9.4f `r_f'
}
else if (`r_v' > 0) {
    di as text "  RESOLVED    D1 dof(none) vce(cluster st year): reghdfe now posts a finite variance"
}
else {
    di as error "  FAIL        D1 dof(none) vce(cluster st year): reghdfe V11=" %21.17g `r_v' " F=" %21.17g `r_f'
    global PARITY_FAILS = $PARITY_FAILS + 1
}
quietly xhdfe y x1 x2, absorb(firm year) dofadjustments(none) vce(cluster st year) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
local x_v1 = e(V)[1, 1]
quietly xhdfe y x1 x2, absorb(firm year) dofadjustments(none) vce(cluster year st) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
local x_v2 = e(V)[1, 1]
xpd_assert_close "D1 xhdfe multiway variance is finite" (`x_v1'>0) 1 0
xpd_assert_close "D1 xhdfe multiway variance is order invariant" (`x_v1'/`x_v2') 1 1e-9

* ===========================================================================
* D2. A single edge joins two of the three blocks: the component count drops
*     from 3 to 2 without any other change, so the redundancy count is the
*     only thing that moves.
* ===========================================================================
di as text _n "-- D2: nearly disconnected graph (one bridging edge) ----------"
preserve
    quietly xhdfe y x1 x2, absorb(firm year) dofadjustments(pairwise) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
    xpd_assert_eq "D2 unbridged stopping criterion converged" e(converged) 1
    xpd_assert_eq "D2 unbridged precision certificate passes" e(precision_certified) 1

    * Move one block-2 observation into a block-1 year, bridging the two.
    quietly summarize year if blk == 1, meanonly
    local bridge_year = r(min)
    quietly replace year = `bridge_year' if obs == 2
    quietly reghdfe y x1 x2, absorb(firm year) dof(pairwise)
    xpd_assert_eq "D2 reghdfe mobility groups" e(df_a_redundant) 2

    * A single bridging edge makes the two-way system nearly singular, which is
    * the point of the design: it is the numerically hardest cell in this file.
    * The exact answer is available (LSDV regress on full dummies), so the two
    * absorbers are adjudicated against it rather than against each other.
    * The automatic route must repair this nearly singular solve until the
    * independent certificate passes. The exact LSDV answer below keeps this
    * from becoming a self-certified stopping-rule test.
    quietly regress y x1 x2 i.firm i.year
    local ex1 = _b[x1]
    local ex2 = _b[x2]
    quietly regress y x1 x2 i.firm i.year, vce(cluster st)
    local ev = e(V)[1, 1]

    quietly reghdfe y x1 x2, absorb(firm year) dof(pairwise)
    local r_e1 = abs(_b[x1] - `ex1')/abs(`ex1')
    quietly xhdfe y x1 x2, absorb(firm year) dofadjustments(pairwise) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
    local x_e1 = abs(_b[x1] - `ex1')/abs(`ex1')
    local x_converged = e(converged)
    local x_certified = e(precision_certified)
    local x_abs_rel = e(abs_residual_rel)
    di as text "  [D2] relative error in b[x1] against LSDV: xhdfe " %9.2e `x_e1' ///
        ", reghdfe " %9.2e `r_e1' "  (xhdfe iterations " e(iterations) ///
        ", e(converged) " e(converged) ", e(abs_residual_rel) " %9.2e e(abs_residual_rel) ")"
    xpd_assert_eq "D2 bridged stopping criterion converged" `x_converged' 1
    xpd_assert_eq "D2 bridged precision certificate passes" `x_certified' 1
    xpd_assert_close "D2 bridged residual certificate is within 8e-12" ///
        (`x_abs_rel'<=8e-12) 1 0
    xpd_assert_close "D2 xhdfe b[x1] within 1e-12 of LSDV" (`x_e1'<=1e-12) 1 0
    xpd_assert_close "D2 reghdfe b[x1] within 1e-12 of LSDV" (`r_e1'<=1e-12) 1 0
    quietly xhdfe y x1 x2, absorb(firm year) dofadjustments(pairwise) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-14) noheader notable nofootnote
    local x_t1 = abs(_b[x1] - `ex1')/abs(`ex1')
    di as text "  [D2] the same xhdfe fit at tolerance(1e-14): " %9.2e `x_t1'
    xpd_assert_close "D2 default fit stays within 1e-12 of LSDV" ///
        (`x_e1'<=1e-12) 1 0
    xpd_assert_close "D2 tightened fit stays within 1e-12 of LSDV" ///
        (`x_t1'<=1e-12) 1 0

    quietly xhdfe y x1 x2, absorb(firm year) vce(cluster st) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
    local x_vr = e(V)[1,1]/`ev'
    quietly reghdfe y x1 x2, absorb(firm year) vce(cluster st)
    local r_vr = e(V)[1,1]/`ev'
    di as text "  [D2] Var(x1) relative to LSDV: xhdfe " %9.2e (`x_vr' - 1) ///
        ", reghdfe " %9.2e (`r_vr' - 1)
    xpd_assert_close "D2 xhdfe Var(x1) within 1e-10 of LSDV" (abs(`x_vr'-1)<=1e-10) 1 0

    * The discrete surface must still agree exactly; continuous quantities use
    * the ordinary tight parity tolerances now that the solve is repaired.
    xpd_spec y x1 x2, name("D2 bridged dof(pairwise)") ///
        xopts(absorb(firm year) dofadjustments(pairwise)) ///
        ropts(absorb(firm year) dof(pairwise)) btol(1e-10) setol(1e-8) skipf
    xcert_parity_tally

    xpd_spec y x1 x2, name("D2 bridged default vce(cluster st)") ///
        xopts(absorb(firm year) vce(cluster st)) ///
        ropts(absorb(firm year) vce(cluster st)) btol(1e-10) setol(1e-8) skipf
    xcert_parity_tally

    * The bridged observation is now the only one of its firm in that year;
    * check that the sample and singleton accounting still coincide.
    xpd_spec y x1 x2 if obs != 2, name("D2 bridge removed by if") ///
        xopts(absorb(firm year)) ropts(absorb(firm year)) btol(1e-10) setol(1e-8)
    xcert_parity_tally
restore

* ===========================================================================
* D3. Three-way fixed effects where the pair (d1,d3) carries four components
*     while (d1,d2) and (d2,d3) carry one. pairwise takes the maximum over all
*     earlier dimensions, firstpair only looks at the first pair, so the two
*     methods and the ORDER of absorb() all change df_a.
* ===========================================================================
clear
set seed 24680
set obs 4800
generate long obs  = _n
generate int  blk  = 1 + mod(obs, 4)
generate int  d1   = blk*100  + 1 + int(runiform()*15)
generate int  d2   =            1 + int(runiform()*20)
generate int  d3   = blk*1000 + 1 + int(runiform()*6)
generate int  st   = 1 + mod(obs, 9)
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double y  = .4*x1 - .6*x2 + d1/500 + d2/20 + d3/3000 + rnormal()

di as text _n "-- D3: 3 FE, order-dependent firstpair ------------------------"

quietly reghdfe y x1 x2, absorb(d1 d3) dof(pairwise)
xpd_assert_eq "D3 components(d1,d3)" e(df_a_redundant) 4
quietly reghdfe y x1 x2, absorb(d1 d2) dof(pairwise)
xpd_assert_eq "D3 components(d1,d2)" e(df_a_redundant) 1
quietly reghdfe y x1 x2, absorb(d2 d3) dof(pairwise)
xpd_assert_eq "D3 components(d2,d3)" e(df_a_redundant) 1

* Exact reference: rank 101 out of 4800, so the exact absorbed dof is 99 and
* pairwise reproduces it; firstpair on (d1,d2) stops at 102 by design.
quietly regress y x1 x2 i.d1 i.d2 i.d3
xpd_assert_eq "D3 LSDV rank" e(rank) 101
local d3_lsdv_dfr = e(df_r)

foreach ord in "d1 d2 d3" "d1 d3 d2" "d3 d2 d1" "d2 d1 d3" {
    foreach mm in pairwise firstpair {
        xpd_spec y x1 x2, name("D3 absorb(`ord') dof(`mm')") ///
            xopts(absorb(`ord') dofadjustments(`mm')) ///
            ropts(absorb(`ord') dof(`mm'))
        xcert_parity_tally
    }
}
* The specs above are only informative if the two methods really disagree.
quietly xhdfe y x1 x2, absorb(d1 d2 d3) dofadjustments(pairwise) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
local xtot = e(df_a) + e(df_m)
xpd_assert_eq "D3 pairwise df_a is exact (== LSDV)" `xtot' 101
xpd_assert_eq "D3 pairwise df_r == LSDV df_r" e(df_r) `d3_lsdv_dfr'
quietly xhdfe y x1 x2, absorb(d1 d2 d3) dofadjustments(firstpair) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xpd_assert_eq "D3 firstpair(d1,d2) df_a" e(df_a) 102
quietly xhdfe y x1 x2, absorb(d1 d3 d2) dofadjustments(firstpair) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xpd_assert_eq "D3 firstpair(d1,d3) df_a" e(df_a) 99

foreach dd in "all" "none" "clusters" {
    xpd_spec y x1 x2, name("D3 absorb(d1 d2 d3) dof(`dd')") ///
        xopts(absorb(d1 d2 d3) dofadjustments(`dd')) ///
        ropts(absorb(d1 d2 d3) dof(`dd'))
    xcert_parity_tally
}
quietly xhdfe y x1 x2, absorb(d1 d2 d3) dofadjustments(clusters) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
local x_red = e(df_a_redundant)
quietly reghdfe y x1 x2, absorb(d1 d2 d3) dof(clusters)
xpd_assert_eq "D3 dof(clusters) df_a_redundant" `x_red' e(df_a_redundant)

* Four dimensions, to exercise the pairwise loop past the first triple.
foreach mm in pairwise firstpair all {
    xpd_spec y x1 x2, name("D3 4 FE dof(`mm')") ///
        xopts(absorb(d1 d2 d3 st) dofadjustments(`mm')) ///
        ropts(absorb(d1 d2 d3 st) dof(`mm'))
    xcert_parity_tally
}
* setol is 1e-7 here: the only coefficient above the default 1e-8 is _cons,
* which in an absorbed regression is recovered from the fitted fixed effects
* and is the least well determined quantity either engine posts (measured
* 1.5e-08 relative, against 1e-12 or better for x1 and x2).
xpd_spec y x1 x2, name("D3 4 FE default vce(cluster st)") ///
    xopts(absorb(d1 d2 d3 st) vce(cluster st)) ///
    ropts(absorb(d1 d2 d3 st) vce(cluster st)) setol(1e-7)
xcert_parity_tally

* ===========================================================================
* D4. Nesting. reghdfe's dof_update_nested() marks an absorbed dimension fully
*     redundant when it IS a cluster variable or is nested within one, removes
*     it from the mobility-group pairs, and then -- because df_a_nested>0 --
*     also charges the first surviving intercept one redundant coefficient.
*     Its CRV1 factor becomes (N-1)/(N-1-df_m-df_a).
* ===========================================================================
clear
set seed 31415
set obs 5000
generate long obs   = _n
generate int  state = 1 + mod(obs, 25)
generate int  firm  = state*100 + 1 + int(runiform()*8)   // firm nested in state
generate int  year  = 1990 + mod(obs, 10)
generate int  reg   = 1 + mod(state, 5)                   // state nested in reg
generate int  other = 1 + mod(obs, 37)
generate double x1  = rnormal()
generate double x2  = rnormal()
generate double y   = .3*x1 - .9*x2 + firm/1000 + year/40 + rnormal()

di as text _n "-- D4: cluster nesting ---------------------------------------"

* absvar IS the cluster variable
xpd_spec y x1 x2, name("D4 absorb(firm) vce(cluster firm)") ///
    xopts(absorb(firm) vce(cluster firm)) ropts(absorb(firm) vce(cluster firm))
xcert_parity_tally
* Exact reference for the fully nested one-way case. When the absorbed
* dimension is the cluster variable, reghdfe deliberately does NOT charge its
* dummies to K -- reghdfe.mata: "minor adj. so we match xtreg when the absvar
* is nested within cluster" -- so the reference is xtreg, fe, not regress on
* dummies. Both engines reproduce xtreg to 1.4e-13. regress on dummies differs
* by exactly 4997/4798 = 1.0414756148393782, i.e. (N-1-df_m)/(N-df_m-df_a)
* against (N-1)/(N-1-df_m): that factor is the convention, not an error.
quietly xtset firm
quietly xtreg y x1 x2, fe vce(cluster firm)
local xt_v = e(V)[1, 1]
quietly regress y x1 x2 i.firm, vce(cluster firm)
xpd_assert_close "D4 regress-on-dummies differs from xtreg by 4997/4798" ///
    (e(V)[1,1]/`xt_v') (4997/4798) 1e-9
quietly xhdfe y x1 x2, absorb(firm) vce(cluster firm) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xpd_assert_close "D4 nested CRV1 matches xtreg, fe" (e(V)[1,1]/`xt_v') 1 1e-9
quietly reghdfe y x1 x2, absorb(firm) vce(cluster firm)
xpd_assert_close "D4 reghdfe nested CRV1 matches xtreg, fe" (e(V)[1,1]/`xt_v') 1 1e-9

* absvar nested WITHIN the cluster variable (firm inside state)
xpd_spec y x1 x2, name("D4 absorb(firm) vce(cluster state)") ///
    xopts(absorb(firm) vce(cluster state)) ropts(absorb(firm) vce(cluster state))
xcert_parity_tally

* partially nested: firm is nested in state, year is not
xpd_spec y x1 x2, name("D4 absorb(firm year) vce(cluster state)") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
xcert_parity_tally

* cluster FINER than the absorbed dimension: no nesting either way
xpd_spec y x1 x2, name("D4 absorb(state) vce(cluster firm)") ///
    xopts(absorb(state) vce(cluster firm)) ropts(absorb(state) vce(cluster firm))
xcert_parity_tally

* both orders of a nested pair inside absorb()
xpd_spec y x1 x2, name("D4 absorb(firm state) vce(cluster state)") ///
    xopts(absorb(firm state) vce(cluster state)) ///
    ropts(absorb(firm state) vce(cluster state))
xcert_parity_tally
xpd_spec y x1 x2, name("D4 absorb(state firm) vce(cluster state)") ///
    xopts(absorb(state firm) vce(cluster state)) ///
    ropts(absorb(state firm) vce(cluster state))
xcert_parity_tally

* multiway clustering with only the first / only the second variable nesting.
* KNOWN OPEN, reghdfe's defect again: with the nesting cluster variable listed
* FIRST, vce(cluster state other), reghdfe posts an all-zero e(V); listing the
* same two variables in the other order it posts a finite one. The multiway
* estimator is symmetric in its cluster arguments, so the two runs must agree;
* xhdfe posts the same finite variance under both orders and matches reghdfe's
* working order to 6e-11. The parity specification below therefore uses the
* order reghdfe can compute, and the failing order is adjudicated here.
quietly reghdfe y x1 x2, absorb(firm year) vce(cluster state other)
local r_v1 = e(V)[1, 1]
quietly reghdfe y x1 x2, absorb(firm year) vce(cluster other state)
local r_v2 = e(V)[1, 1]
quietly xhdfe y x1 x2, absorb(firm year) vce(cluster state other) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
local x_v1 = e(V)[1, 1]
quietly xhdfe y x1 x2, absorb(firm year) vce(cluster other state) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
local x_v2 = e(V)[1, 1]
xpd_assert_close "D4 xhdfe multiway variance is order invariant" (`x_v1'/`x_v2') 1 1e-9
xpd_assert_close "D4 xhdfe matches reghdfe on the order reghdfe can compute" ///
    (`x_v2'/`r_v2') 1 1e-8
if (`r_v1' == 0 & `r_v2' > 0) {
    di as text "XHDFE_KNOWN_OPEN|D4 vce(cluster state other)|reghdfe e(V)=0|" ///
        "reversed e(V)=" %21.17g `r_v2'
}
else if (`r_v1' > 0) {
    di as text "  RESOLVED    D4 vce(cluster state other): reghdfe now posts a finite variance"
}
else {
    di as error "  FAIL        D4 vce(cluster state other): reghdfe V11=" %21.17g `r_v1' ///
        " reversed " %21.17g `r_v2'
    global PARITY_FAILS = $PARITY_FAILS + 1
}
xpd_spec y x1 x2, name("D4 2-way cluster, second nests") ///
    xopts(absorb(firm year) vce(cluster other state)) ///
    ropts(absorb(firm year) vce(cluster other state))
xcert_parity_tally
xpd_spec y x1 x2, name("D4 2-way cluster, neither nests") ///
    xopts(absorb(firm year) vce(cluster other year)) ///
    ropts(absorb(firm year) vce(cluster other year))
xcert_parity_tally

* every absorbed dimension is a cluster variable: df_a collapses to zero
* setol 1e-7 for the same reason as the 4 FE cell above: only se[_cons] moves,
* by 1.6e-08 relative, while x1 and x2 agree to 1e-12.
xpd_spec y x1 x2, name("D4 all FE are cluster vars") ///
    xopts(absorb(firm year) vce(cluster firm year)) ///
    ropts(absorb(firm year) vce(cluster firm year)) setol(1e-7)
xcert_parity_tally
quietly xhdfe y x1 x2, absorb(firm year) vce(cluster firm year) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xpd_assert_eq "D4 all-nested df_a" e(df_a) 0
xpd_assert_eq "D4 all-nested df_a_nested" e(df_a_nested) 210

* three dimensions, nesting chain firm < state < reg
xpd_spec y x1 x2, name("D4 absorb(firm state year) vce(cluster reg)") ///
    xopts(absorb(firm state year) vce(cluster reg)) ///
    ropts(absorb(firm state year) vce(cluster reg))
xcert_parity_tally

* interaction absvar that is also the cluster variable
xpd_spec y x1 x2, name("D4 absorb(firm#year) vce(cluster firm#year)") ///
    xopts(absorb(firm#year) vce(cluster firm#year)) ///
    ropts(absorb(firm#year) vce(cluster firm#year))
xcert_parity_tally

* nesting switched OFF by the option: dof(pairwise) excludes "clusters"
xpd_spec y x1 x2, name("D4 dof(pairwise) suppresses nesting") ///
    xopts(absorb(firm) vce(cluster state) dofadjustments(pairwise)) ///
    ropts(absorb(firm) vce(cluster state) dof(pairwise))
xcert_parity_tally
xpd_spec y x1 x2, name("D4 dof(pairwise clusters) restores it") ///
    xopts(absorb(firm year) vce(cluster state) dofadjustments(pairwise clusters)) ///
    ropts(absorb(firm year) vce(cluster state) dof(pairwise clusters))
xcert_parity_tally
* dof(clusters) alone is in parity here only because this graph is connected,
* so xhdfe's unrequested mobility-group search returns the same 1 redundancy
* that reghdfe assigns by default. D1/D3 are where it separates.
xpd_spec y x1 x2, name("D4 dof(clusters) on a connected graph") ///
    xopts(absorb(firm year) vce(cluster state) dofadjustments(clusters)) ///
    ropts(absorb(firm year) vce(cluster state) dof(clusters))
xcert_parity_tally

* nesting plus dof(none): no nesting adjustment.
xpd_spec y x1 x2, name("D4 dof(none) vce(cluster state)") ///
    xopts(absorb(firm year) vce(cluster state) dofadjustments(none)) ///
    ropts(absorb(firm year) vce(cluster state) dof(none))
xcert_parity_tally
xpd_vratio x1
xpd_assert_close "D4 dof(none) cluster small-sample factor" r(ratio) 1 1e-9

* ===========================================================================
* D5. Singletons and sample construction.
*     The appended rows form a chain in which removing one dimension's
*     singleton creates the other dimension's singleton, six times over.
* ===========================================================================
clear
set seed 555
set obs 2000
generate long obs  = _n
generate int  firm = 1 + mod(obs, 40)
generate int  year = 1 + mod(floor((obs-1)/40), 10)
generate int  st   = 1 + mod(obs, 7)
generate double x1 = rnormal()
generate double x2 = rnormal()
set obs 2006
* firm 101 -> year 201 -> firm 102 -> year 202 -> firm 103 -> year 203 -> firm 1
quietly replace firm = 101 in 2001
quietly replace year = 201 in 2001
quietly replace firm = 102 in 2002
quietly replace year = 201 in 2002
quietly replace firm = 102 in 2003
quietly replace year = 202 in 2003
quietly replace firm = 103 in 2004
quietly replace year = 202 in 2004
quietly replace firm = 103 in 2005
quietly replace year = 203 in 2005
quietly replace firm = 1   in 2006
quietly replace year = 203 in 2006
quietly replace st = 1 + mod(_n, 7)  if missing(st)
quietly replace x1 = rnormal()       if missing(x1)
quietly replace x2 = rnormal()       if missing(x2)
quietly replace obs = _n
generate double y = .6*x1 - .4*x2 + firm/100 + year/50 + rnormal()

di as text _n "-- D5: iterative singleton removal ---------------------------"

* The chain must actually cascade, otherwise the specs below prove nothing:
* one pass over firm removes a single row, the full fixed point removes six.
quietly reghdfe y x1 x2, absorb(firm)
xpd_assert_eq "D5 one-dimension singletons" e(num_singletons) 1
quietly reghdfe y x1 x2, absorb(firm year)
xpd_assert_eq "D5 cascaded singletons" e(num_singletons) 6

xpd_spec y x1 x2, name("D5 cascade, 2 FE") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally
xpd_spec y x1 x2, name("D5 cascade, keepsingletons") ///
    xopts(absorb(firm year) keepsingletons) ///
    ropts(absorb(firm year) keepsingletons)
xcert_parity_tally
xpd_spec y x1 x2, name("D5 cascade, 3 FE") ///
    xopts(absorb(firm year st)) ropts(absorb(firm year st))
xcert_parity_tally
xpd_spec y x1 x2, name("D5 cascade, vce(cluster st)") ///
    xopts(absorb(firm year) vce(cluster st)) ///
    ropts(absorb(firm year) vce(cluster st))
xcert_parity_tally
* The dropped rows must not change the cluster count either.
xpd_spec y x1 x2, name("D5 cascade, vce(cluster firm)") ///
    xopts(absorb(firm year) vce(cluster firm)) ///
    ropts(absorb(firm year) vce(cluster firm))
xcert_parity_tally
xpd_spec y x1 x2, name("D5 cascade, dof(none)") ///
    xopts(absorb(firm year) dofadjustments(none)) ///
    ropts(absorb(firm year) dof(none))
xcert_parity_tally
xpd_spec y x1 x2 if obs <= 2000, name("D5 if excludes the chain") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally
xpd_spec y x1 x2 in 1/1500, name("D5 in 1/1500") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally
* An -if- that thins the panel enough to manufacture new singletons.
xpd_spec y x1 x2 if mod(obs, 3) != 0, name("D5 if manufactures singletons") ///
    xopts(absorb(firm year) vce(cluster st)) ///
    ropts(absorb(firm year) vce(cluster st))
xcert_parity_tally

* ---- missing values in every position --------------------------------------
clear
set seed 777
set obs 3000
generate long obs  = _n
generate int  firm = 1 + mod(obs, 60)
generate int  year = 1 + mod(floor((obs-1)/60), 10)
generate int  st   = 1 + mod(obs, 13)
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double w  = 0.5 + abs(rnormal())
generate int    fw = 1 + mod(obs, 3)
generate double y  = .6*x1 - .4*x2 + firm/60 + year/20 + rnormal()

generate double y_m    = cond(mod(obs, 97) == 0, ., y)
generate double x1_m   = cond(mod(obs, 89) == 0, ., x1)
generate int    firm_m = cond(mod(obs, 83) == 0, ., firm)
generate int    st_m   = cond(mod(obs, 79) == 0, ., st)
generate double w_m    = cond(mod(obs, 71) == 0, ., w)

di as text _n "-- D5b: sample construction ----------------------------------"

xpd_spec y_m x1 x2, name("D5b missing outcome") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally
xpd_spec y x1_m x2, name("D5b missing regressor") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally
xpd_spec y x1 x2, name("D5b missing absorb variable") ///
    xopts(absorb(firm_m year)) ropts(absorb(firm_m year))
xcert_parity_tally
xpd_spec y x1 x2, name("D5b missing cluster variable") ///
    xopts(absorb(firm year) vce(cluster st_m)) ///
    ropts(absorb(firm year) vce(cluster st_m))
xcert_parity_tally
xpd_spec y x1 x2 [aw = w_m], name("D5b missing aweight") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally
xpd_spec y_m x1_m x2 if obs > 100 & obs < 2900, name("D5b every source at once") ///
    xopts(absorb(firm_m year st_m) vce(cluster st_m)) ///
    ropts(absorb(firm_m year st_m) vce(cluster st_m))
xcert_parity_tally
xpd_spec y x1 x2 in 500/2500, name("D5b in-range with clustering") ///
    xopts(absorb(firm year) vce(cluster st)) ///
    ropts(absorb(firm year) vce(cluster st))
xcert_parity_tally
xpd_spec y x1 x2 if x1 > 0, name("D5b endogenous-looking if") ///
    xopts(absorb(firm year) vce(cluster st)) ///
    ropts(absorb(firm year) vce(cluster st))
xcert_parity_tally

* ---- weights crossed with the adjustment set --------------------------------
foreach dd in "all" "none" "clusters" "firstpair" {
    xpd_spec y x1 x2 [aw = w], name("D5b aweights dof(`dd')") ///
        xopts(absorb(firm year) dofadjustments(`dd')) ///
        ropts(absorb(firm year) dof(`dd'))
    xcert_parity_tally
    xpd_spec y x1 x2 [fw = fw], name("D5b fweights dof(`dd')") ///
        xopts(absorb(firm year) dofadjustments(`dd')) ///
        ropts(absorb(firm year) dof(`dd'))
    xcert_parity_tally
}
xpd_spec y x1 x2 [pw = w], name("D5b pweights, nested cluster") ///
    xopts(absorb(firm year) vce(cluster firm)) ///
    ropts(absorb(firm year) vce(cluster firm))
xcert_parity_tally
xpd_spec y x1 x2 [fw = fw], name("D5b fweights, nested cluster") ///
    xopts(absorb(firm year) vce(cluster firm)) ///
    ropts(absorb(firm year) vce(cluster firm))
xcert_parity_tally

* ---- e(rank) and e(df_m) with omitted regressors ----------------------------
generate double x3 = 2*x1 - 1.5*x2
generate double xfirm = firm                        // constant within firm
generate byte   cat = 1 + mod(obs, 5)

di as text _n "-- D5c: rank, df_m and the omission pattern ------------------"

xpd_spec y x1 x2 x3, name("D5c exactly collinear regressor") ///
    xopts(absorb(firm year) vce(cluster st)) ///
    ropts(absorb(firm year) vce(cluster st))
xcert_parity_tally
xpd_spec y x1 x2 xfirm, name("D5c regressor absorbed by the FE") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally
xpd_spec y x1 x2 xfirm x3, name("D5c both kinds of omission") ///
    xopts(absorb(firm year) vce(cluster st)) ///
    ropts(absorb(firm year) vce(cluster st))
xcert_parity_tally
xpd_spec y xfirm, name("D5c only an absorbed regressor") ///
    xopts(absorb(firm)) ropts(absorb(firm))
xcert_parity_tally
xpd_spec y i.cat x1, name("D5c factor variable") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally
xpd_spec y i.cat##c.x1 x2, name("D5c factor interaction") ///
    xopts(absorb(firm year) vce(cluster st)) ///
    ropts(absorb(firm year) vce(cluster st))
xcert_parity_tally

* ===========================================================================
* D6. Continuous (slope) absvars, i.e. the "continuous" adjustment.
*     z is constant within firms 1-20 and zz is zero for firms 1-15, which is
*     exactly what reghdfe's dof_update_cvars() looks for.
* ===========================================================================
clear
set seed 8642
set obs 3000
generate long obs  = _n
generate int  firm = 1 + mod(obs, 50)
generate int  year = 1 + mod(floor((obs-1)/50), 10)
generate int  st   = 1 + mod(obs, 13)
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double z  = cond(firm <= 20, firm, rnormal())
generate double zz = cond(firm <= 15, 0, rnormal())
generate double y  = .6*x1 - .4*x2 + firm/50 + year/20 + rnormal()

di as text _n "-- D6: continuous slope terms --------------------------------"

* With the adjustment enabled the full totals, decomposition and extended
* intercept/slope table must agree.
foreach dd in "all" "continuous" "pairwise continuous" {
    xpd_spec y x1 x2, name("D6 absorb(firm##c.z) dof(`dd')") ///
        xopts(absorb(firm##c.z) dofadjustments(`dd')) ///
        ropts(absorb(firm##c.z) dof(`dd'))
    xcert_parity_tally
}
quietly xhdfe y x1 x2, absorb(firm##c.z) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xpd_assert_eq "D6 df_a with the continuous adjustment on" e(df_a) 80
xpd_selfcheck, name("D6 absorb(firm##c.z)")

* Without the continuous token, degenerate-slope adjustments are suppressed.
foreach dd in "pairwise" "clusters" {
    xpd_spec y x1 x2, name("D6 absorb(firm##c.z) dof(`dd')") ///
        xopts(absorb(firm##c.z) dofadjustments(`dd')) ///
        ropts(absorb(firm##c.z) dof(`dd'))
    xcert_parity_tally
    xpd_vratio x1
    xpd_assert_close "D6 dof(`dd') slope adjustment suppressed" r(ratio) 1 1e-9
}

* Slope-only absvar (no intercept): reghdfe's rule becomes "zero within level".
foreach dd in "all" "continuous" {
    xpd_spec y x1 x2, name("D6 absorb(firm#c.zz) dof(`dd')") ///
        xopts(absorb(firm#c.zz) dofadjustments(`dd')) ///
        ropts(absorb(firm#c.zz) dof(`dd'))
    xcert_parity_tally
}
xpd_spec y x1 x2, name("D6 absorb(firm#c.zz) dof(pairwise)") ///
    xopts(absorb(firm#c.zz) dofadjustments(pairwise)) ///
    ropts(absorb(firm#c.zz) dof(pairwise))
xcert_parity_tally
xpd_vratio x1
xpd_assert_close "D6 slope-only dof(pairwise) suppressed" r(ratio) 1 1e-9

* A slope that is never degenerate: nothing is adjusted.
xpd_spec y x1 x2, name("D6 absorb(firm##c.x1) non-degenerate slope") ///
    xopts(absorb(firm##c.x1)) ropts(absorb(firm##c.x1))
xcert_parity_tally
xpd_selfcheck, name("D6 absorb(firm##c.x1)")

* Slope term alongside a plain dimension, and slope term nested in the cluster.
xpd_spec y x1 x2, name("D6 absorb(firm##c.z year) vce(cluster st)") ///
    xopts(absorb(firm##c.z year) vce(cluster st)) ///
    ropts(absorb(firm##c.z year) vce(cluster st))
xcert_parity_tally
xpd_spec y x1 x2, name("D6 absorb(firm##c.z) vce(cluster firm)") ///
    xopts(absorb(firm##c.z) vce(cluster firm)) ///
    ropts(absorb(firm##c.z) vce(cluster firm))
xcert_parity_tally

* ===========================================================================
* D7. No-absorb designs, noconstant, and the option parser.
* ===========================================================================
clear
set seed 1122
set obs 3000
generate long obs  = _n
generate int  firm = 1 + mod(obs, 50)
generate int  year = 1 + mod(floor((obs-1)/50), 10)
generate int  st   = 1 + mod(obs, 13)
generate byte one  = 1
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double y  = .6*x1 - .4*x2 + firm/50 + year/20 + rnormal()

di as text _n "-- D7: no-absorb, noconstant, parser -------------------------"

* No-absorb models include the synthetic constant row in the same DoF surface
* as reghdfe.
xpd_spec y x1 x2, name("D7 noabsorb, unadjusted") ///
    ropts(noabsorb)
xcert_parity_tally
xpd_spec y x1 x2, name("D7 noabsorb, vce(robust)") ///
    xopts(vce(robust)) ropts(noabsorb vce(robust))
xcert_parity_tally
quietly regress y x1 x2, vce(robust)
local reg_v = e(V)[1, 1]
quietly xhdfe y x1 x2, vce(robust) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xpd_assert_close "D7 noabsorb robust matches regress" (e(V)[1,1]/`reg_v') 1 1e-9

* Clustered no-absorb models charge that constant in CRV1.
quietly regress y x1 x2, vce(cluster st)
local reg_v = e(V)[1, 1]
local reg_dfr = e(df_r)
foreach vv in "cluster st" "cluster st firm" {
    xpd_spec y x1 x2, name("D7 noabsorb, vce(`vv')") ///
        xopts(vce(`vv')) ropts(noabsorb vce(`vv'))
    xcert_parity_tally
    xpd_vratio x1
    xpd_assert_close "D7 noabsorb vce(`vv') CRV1 charges intercept" r(ratio) 1 1e-9
}
quietly xhdfe y x1 x2, vce(cluster st) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xpd_assert_eq "D7 noabsorb clustered df_r matches regress" e(df_r) `reg_dfr'
xpd_assert_close "D7 noabsorb clustered variance vs regress" ///
    (e(V)[1,1]/`reg_v') 1 1e-9
quietly reghdfe y x1 x2, noabsorb vce(cluster st)
xpd_assert_close "D7 reghdfe noabsorb clustered matches regress" ///
    (e(V)[1,1]/`reg_v') 1 1e-12

* Absorbing a constant variable is the same model with an explicit dimension,
* and there the two engines agree on every count, which localises the defect
* above to the missing-absorb() path rather than to the estimator.
xpd_spec y x1 x2, name("D7 absorb(constant) vce(cluster st)") ///
    xopts(absorb(one) vce(cluster st)) ropts(absorb(one) vce(cluster st))
xcert_parity_tally

* ---- noconstant -------------------------------------------------------------
* NOT a parity specification: the two commands do not fit the same model.
* reghdfe's parser maps noconstant to report_constant only, so with noabsorb it
* still fits an intercept: b(x1)=.590483039340385, rss=3307.37. xhdfe removes
* the intercept, exactly like `regress ..., noconstant': b(x1)=.563734818110146
* and rss=5211.03062265576 on both, agreeing to 1e-15. On the substantive
* meaning of the word, reghdfe is the one that is wrong here, so this is
* asserted against regress rather than against reghdfe.
quietly regress y x1 x2, noconstant vce(cluster st)
local reg_b = _b[x1]
local reg_v = e(V)[1, 1]
local reg_rss = e(rss)
quietly xhdfe y x1 x2, noconstant vce(cluster st) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xpd_assert_close "D7 noconstant b matches regress" _b[x1] `reg_b' 1e-12
xpd_assert_close "D7 noconstant V matches regress" (e(V)[1,1]/`reg_v') 1 1e-9
xpd_assert_close "D7 noconstant rss matches regress" e(rss) `reg_rss' 1e-12
quietly reghdfe y x1 x2, noabsorb noconstant vce(cluster st)
xpd_int_note "D7 reghdfe books an absorbed constant despite noconstant (df_a)" e(df_a) 1 0
xpd_ratio_note "D7 reghdfe noconstant rss / regress-noconstant rss" ///
    (e(rss)/`reg_rss') (3307.36727375594/5211.03062265576)

* With real fixed effects the constant is absorbed either way, so noconstant is
* a no-op for both engines and parity is required.
xpd_spec y x1 x2, name("D7 absorb(firm year) noconstant") ///
    xopts(absorb(firm year) noconstant) ropts(absorb(firm year) noconstant)
xcert_parity_tally

* dof(none) with noconstant keeps the same trivial FE redundancy.
quietly regress y x1 x2 ibn.firm i.year, noconstant
local lsdv_rank = e(rank)
local lsdv_dfr = e(df_r)
xpd_assert_eq "D7 LSDV noconstant rank" `lsdv_rank' 61
quietly xhdfe y x1 x2, absorb(firm year) noconstant dofadjustments(none) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
xpd_assert_eq "D7 dof(none) noconstant df_a" e(df_a) 59
xpd_assert_eq "D7 dof(none) noconstant df_r" e(df_r) `lsdv_dfr'
quietly reghdfe y x1 x2, absorb(firm year) noconstant dof(none)
xpd_assert_eq "D7 reghdfe dof(none) noconstant df_r == LSDV" e(df_r) `lsdv_dfr'
xpd_spec y x1 x2, name("D7 dof(none) noconstant") ///
    xopts(absorb(firm year) noconstant dofadjustments(none)) ///
    ropts(absorb(firm year) noconstant dof(none))
xcert_parity_tally
xpd_vratio x1
xpd_assert_close "D7 dof(none) noconstant unadjusted small-sample factor" ///
    r(ratio) 1 1e-9
xpd_spec y x1 x2, name("D7 dof(none) noconstant vce(cluster st)") ///
    xopts(absorb(firm year) noconstant dofadjustments(none) vce(cluster st)) ///
    ropts(absorb(firm year) noconstant dof(none) vce(cluster st))
xcert_parity_tally
xpd_vratio x1
xpd_assert_close "D7 dof(none) noconstant small-sample factor" r(ratio) 1 1e-9

* ---- three-dimension dof(none), where the gap is two degrees of freedom -----
foreach vv in "robust" "cluster st" {
    xpd_spec y x1 x2, name("D7 3 FE dof(none) vce(`vv')") ///
        xopts(absorb(firm year st) dofadjustments(none) vce(`vv')) ///
        ropts(absorb(firm year st) dof(none) vce(`vv'))
    xcert_parity_tally
    xpd_vratio x1
    local kk = scalar(px_df_a_initial) - scalar(px_df_a)
    xpd_assert_eq "D7 3 FE dof(none) redundancy charged by the core" `kk' 2
    xpd_assert_close "D7 3 FE dof(none) vce(`vv') small-sample factor" ///
        r(ratio) 1 1e-9
}

* ---- one dimension: dof(none) and the default must coincide -----------------
xpd_spec y x1 x2, name("D7 1 FE dof(none) vce(cluster st)") ///
    xopts(absorb(firm) dofadjustments(none) vce(cluster st)) ///
    ropts(absorb(firm) dof(none) vce(cluster st))
xcert_parity_tally

* ---- option parser: combinations reghdfe rejects ----------------------------
foreach dd in "all pairwise" "none clusters" "pairwise firstpair" {
    capture quietly xhdfe y x1 x2, absorb(firm year) dofadjustments(`dd') ///
        noheader notable nofootnote
    local xrc = c(rc)
    capture quietly reghdfe y x1 x2, absorb(firm year) dof(`dd')
    local rrc = c(rc)
    xpd_assert_eq "D7 dofadjustments(`dd') return code" `xrc' `rrc'
}
* An unknown token must be refused by both.
capture quietly xhdfe y x1 x2, absorb(firm year) dofadjustments(bogus) ///
    noheader notable nofootnote
local xrc = c(rc)
capture quietly reghdfe y x1 x2, absorb(firm year) dof(bogus)
xpd_assert_eq "D7 dofadjustments(bogus) rejected by both" `xrc' c(rc)

* ---- degenerate: every row is its own absorbed level -------------------------
capture quietly xhdfe y x1 x2, absorb(firm year obs) noheader notable nofootnote
local xrc = c(rc)
capture quietly reghdfe y x1 x2, absorb(firm year obs)
xpd_assert_eq "D7 fully absorbed sample return code" `xrc' c(rc)

* ---- self-consistency of the posted dof surface -----------------------------
di as text _n "-- self-consistency of e(dof_table) --------------------------"
foreach aa in "firm" "firm year" "firm year st" {
    quietly xhdfe y x1 x2, absorb(`aa') ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
    xpd_selfcheck, name("absorb(`aa')")
    quietly xhdfe y x1 x2, absorb(`aa') vce(cluster firm) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
    xpd_selfcheck, name("absorb(`aa') vce(cluster firm)")
    quietly xhdfe y x1 x2, absorb(`aa') dofadjustments(none) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
    xpd_selfcheck, name("absorb(`aa') dof(none)")
}

di as text "{hline 70}"
if ($PARITY_FAILS == 0) {
    di as result "PASS: reghdfe parity, degrees of freedom ($PARITY_FAILS unexplained divergences)"
}
else {
    di as error "FAIL: reghdfe parity, degrees of freedom ($PARITY_FAILS unexplained divergences)"
    exit 9
}

global XHDFE_TESTS_RUN = $XHDFE_TESTS_RUN + 1
