* ===========================================================================
* reghdfe FEATURE-surface parity: weights, IV/2SLS, fixed-effect recovery,
* heterogeneous slopes, multiway clustering, factor variables, postestimation
* and xfe partial-out.
*
* Companion to part1/reghdfe-convention-parity.do, which covers the baseline
* surface (FE dimensions, plain weights, plain factor variables, the standard
* variance estimators). This layer does not repeat any of that.
*
* reghdfe is canonical for CONVENTIONS, not for numerical truth. Every
* divergence found here is adjudicated against an EXACT reference wherever one
* exists — the full-dummy `regress`/`areg` fit, or a closed-form Cameron-
* Gelbach-Miller sandwich computed in Mata — and the verdict is recorded at the
* call site. Two of the open items below are cases where reghdfe, not xhdfe, is
* the one that disagrees with the exact answer.
*
* Failure policy:
*   - Everything that should agree is asserted and FAILS the suite if it moves.
*   - Each documented OPEN divergence prints its measured magnitude on every
*     run and is bounded: it may shrink or disappear (a fix keeps the suite
*     green) but it may not grow (a regression turns the suite red).
* ===========================================================================

version 16

xcert_require_reghdfe

global PFEAT_FAILS 0
global PFEAT_OPEN  0

* ---------------------------------------------------------------------------
* Local reporting helpers. _helpers.do is shared with other parity layers and
* is never edited from here, so anything this layer needs beyond
* xcert_parity_spec lives in this file.
* ---------------------------------------------------------------------------

* A documented OPEN divergence. `got' is the measured magnitude, `pinned' is
* the magnitude recorded when the defect was diagnosed. Shrinking (or a fix)
* keeps the suite green; growing past 1.05x turns it red.
capture program drop xpf_open
program define xpf_open
    version 16
    syntax, ID(string) GOT(string) PINNED(real) [WHAT(string) TOLFACTOR(real 1.05)]

    local got = `got'
    global PFEAT_OPEN = $PFEAT_OPEN + 1
    di as text "XHDFE_KNOWN_OPEN|`id'|`what'"
    di as text "             measured=" %21.9g `got' "  pinned=" %21.9g `pinned'
    if (`got' > `pinned' * `tolfactor' & `got' > 0) {
        di as error "  [`id'] OPEN DIVERGENCE GREW beyond the pinned magnitude"
        global PFEAT_FAILS = $PFEAT_FAILS + 1
    }
end

* A hard assertion: two quantities that must agree in every supported design.
capture program drop xpf_eq
program define xpf_eq
    version 16
    syntax, ID(string) LEFT(string) RIGHT(string) ///
        [TOL(real 1e-9) FLOOR(real .01) WHAT(string) EXACT]

    local left  = `left'
    local right = `right'
    if ("`exact'" != "") {
        if (`left' != `right') {
            di as error "  FAIL [`id'] `what': " %21.13g `left' " != " %21.13g `right'
            global PFEAT_FAILS = $PFEAT_FAILS + 1
        }
        else {
            di as text "  ok   [`id'] `what' = " %21.13g `left'
        }
        exit
    }
    local scale = max(abs(`right'), `floor')
    local rel = abs(`left' - `right') / `scale'
    if (`rel' > `tol') {
        di as error "  FAIL [`id'] `what': rel " %9.2e `rel' " > " %9.2e `tol' ///
            "  (" %21.13g `left' " vs " %21.13g `right' ")"
        global PFEAT_FAILS = $PFEAT_FAILS + 1
    }
    else {
        di as text "  ok   [`id'] `what': rel " %9.2e `rel'
    }
end

* A hard assertion on a return code (a refusal that is itself the convention).
capture program drop xpf_rc
program define xpf_rc
    version 16
    syntax, ID(string) GOT(integer) WANT(integer) [WHAT(string)]

    if (`got' != `want') {
        di as error "  FAIL [`id'] `what': rc=`got' expected rc=`want'"
        global PFEAT_FAILS = $PFEAT_FAILS + 1
    }
    else {
        di as text "  ok   [`id'] `what': rc=`got'"
    }
end

* Run one specification through both engines and hand the results back as
* matrices/scalars, without the pass/fail machinery, so a cell with a
* documented open item can still assert everything that must agree.
capture program drop xpf_pair
program define xpf_pair
    version 16
    syntax anything(everything) [aweight fweight pweight iweight], ///
        [XOPTS(string) ROPTS(string)]

    quietly xhdfe `anything' [`weight'`exp'], ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12) ///
        `xopts' noheader notable nofootnote
    matrix qx_b = e(b)
    matrix qx_V = e(V)
    scalar qx_N     = e(N)
    scalar qx_dfr   = e(df_r)
    scalar qx_dfa   = e(df_a)
    scalar qx_dfm   = e(df_m)
    scalar qx_rss   = e(rss)
    scalar qx_nclus = e(N_clust)
    scalar qx_sumw  = e(sumweights)

    quietly reghdfe `anything' [`weight'`exp'], `ropts'
    matrix qr_b = e(b)
    matrix qr_V = e(V)
    scalar qr_N     = e(N)
    scalar qr_dfr   = e(df_r)
    scalar qr_dfa   = e(df_a)
    scalar qr_dfm   = e(df_m)
    scalar qr_rss   = e(rss)
    scalar qr_nclus = e(N_clust)
    scalar qr_sumw  = e(sumweights)
end

* Closed-form Cameron-Gelbach-Miller multiway meat, used as an exact
* adjudicator where neither engine can be assumed correct.
capture mata: mata drop xpf_meat()
mata:
real matrix xpf_meat(real matrix S, real colvector g)
{
    real colvector l, sg
    real matrix M
    real scalar j
    l = uniqrows(g)
    M = J(cols(S), cols(S), 0)
    for (j = 1; j <= rows(l); j++) {
        sg = colsum(select(S, g :== l[j]))'
        M = M + sg * sg'
    }
    return(M)
}
end

di as text _n "{hline 74}"
di as text "reghdfe feature parity: weights / IV / savefe / slopes / cluster / fv"
di as text "{hline 74}"

* ---------------------------------------------------------------------------
* Main dataset. Firm, year, state and reg are drawn independently so that no
* absorbed dimension is nested in another and no cluster dimension is nested
* in an absorbed one unless a cell asks for it. div is a strict coarsening of
* reg; grp is a three-level dimension used for the few-clusters cells.
* ---------------------------------------------------------------------------
clear
set seed 20260805
set obs 5000
generate long obs   = _n
generate int  firm  = 1 + int(runiform() * 150)
generate int  year  = 1 + int(runiform() * 12)
generate int  state = 1 + int(runiform() * 22)
generate int  reg   = 1 + int(runiform() * 18)
generate int  div   = 1 + int((reg - 1) / 3)
generate byte cat   = 1 + mod(obs, 4)
generate byte grp   = 1 + mod(obs, 3)
generate str6 sfirm = "f" + string(firm)
generate double x1  = rnormal()
generate double x2  = rnormal()
generate double zz  = rnormal()
generate double qq  = rnormal()
generate double w   = 0.4 + abs(rnormal())
generate int    fw  = 1 + mod(obs, 3)
generate double y   = 0.7 * x1 - 1.3 * x2 + 0.02 * firm - 0.05 * year ///
    + 0.3 * cat + 0.2 * grp + rnormal()

quietly summarize w, meanonly
generate double wnorm = w / r(mean)          // sum(wnorm) == N exactly
generate double wbig  = w * 1e6
generate double w2    = w * 2
generate double wtiny = w * 1e-9
generate double wzero = cond(mod(obs, 10) == 0, 0, w)
generate int    fwzero = cond(mod(obs, 10) == 0, 0, fw)
generate double fwbad = fw + 0.5

* ===========================================================================
* A. WEIGHTS IN DEPTH
* ===========================================================================
di as text _n "-- A. weights ------------------------------------------------"

* A.1 pweights are not covered at all by the baseline layer.
xcert_parity_spec y x1 x2 [pw = w], name("A01 pw default (implies robust)") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [pw = w], name("A02 pw robust") ///
    xopts(absorb(firm year) vce(robust)) ropts(absorb(firm year) vce(robust))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [pw = w], name("A03 pw cluster 1-way") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [pw = w], name("A04 pw cluster nested in FE") ///
    xopts(absorb(firm year) vce(cluster firm)) ///
    ropts(absorb(firm year) vce(cluster firm))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [aw = w], name("A05 aw unadjusted") ///
    xopts(absorb(firm year) vce(unadjusted)) ///
    ropts(absorb(firm year) vce(unadjusted))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [aw = w], name("A06 aw robust") ///
    xopts(absorb(firm year) vce(robust)) ropts(absorb(firm year) vce(robust))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [aw = w], name("A07 aw cluster 1-way") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [fw = fw], name("A08 fw unadjusted") ///
    xopts(absorb(firm year) vce(unadjusted)) ///
    ropts(absorb(firm year) vce(unadjusted))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [fw = fw], name("A09 fw robust") ///
    xopts(absorb(firm year) vce(robust)) ropts(absorb(firm year) vce(robust))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [fw = fw], name("A10 fw cluster 1-way") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

* A.11-A.12 weights combined with factor variables.
xcert_parity_spec y x1 x2 i.cat [aw = w], name("A11 aw + factor variables") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 i.cat##c.zz [pw = w], name("A12 pw + fv interaction") ///
    xopts(absorb(firm year) vce(robust)) ropts(absorb(firm year) vce(robust))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

* A.13-A.15 extreme weight scales and weight totals far from N. An analytic or
* probability weight carries no scale, so every reported quantity except
* e(sumweights) must be invariant.
xcert_parity_spec y x1 x2 [aw = wbig], name("A13 aw scaled by 1e6") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [aw = wtiny], name("A14 aw scaled by 1e-9") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [pw = wbig], name("A15 pw scaled by 1e6") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

* A.16-A.18 zero-weight rows must leave the estimation sample in both engines.
xcert_parity_spec y x1 x2 [aw = wzero], name("A16 aw zero-weight rows") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [pw = wzero], name("A17 pw zero-weight rows") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [fw = fwzero], name("A18 fw zero-weight rows") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

* A.19 e(sumweights): the sum of the weights on the estimation sample for a
* weighted fit, and N for an unweighted one, for each weight type.
foreach wt in "aw = w" "pw = w" "fw = fw" {
    xpf_pair y x1 x2 [`wt'], xopts(absorb(firm year)) ropts(absorb(firm year))
    xpf_eq, id("A19 sumweights [`wt']") left(qx_sumw) right(qr_sumw) tol(1e-12) ///
        what("e(sumweights)")
}
xpf_pair y x1 x2, xopts(absorb(firm year)) ropts(absorb(firm year))
xpf_eq, id("A19 sumweights [none]") left(qx_sumw) right(qr_sumw) exact ///
    what("e(sumweights) with no weights")

* A.20 non-integer frequency weights must be refused by both engines.
capture reghdfe y x1 x2 [fw = fwbad], absorb(firm year)
local rc_r = _rc
capture xhdfe y x1 x2 [fw = fwbad], absorb(firm year) noheader notable nofootnote
local rc_x = _rc
xpf_rc, id("A20 non-integer fweight") got(`rc_r') want(401) what("reghdfe refuses")
xpf_rc, id("A20 non-integer fweight") got(`rc_x') want(401) what("xhdfe refuses")

* ---------------------------------------------------------------------------
* A.21 (W-1): weight metadata is part of the eclass contract used by
* re-running prefixes and weighted postestimation.
* ---------------------------------------------------------------------------
local wt_missing 0
foreach wt in "aw = w" "pw = w" "fw = fw" {
    quietly xhdfe y x1 x2 [`wt'], absorb(firm year) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    local xwt `"`e(wtype)'"'
    local xwe `"`e(wexp)'"'
    quietly reghdfe y x1 x2 [`wt'], absorb(firm year)
    local rwt `"`e(wtype)'"'
    local rwe `"`e(wexp)'"'
    if (`"`xwt'"' != `"`rwt'"' | `"`xwe'"' != `"`rwe'"') {
        di as text "             [`wt'] xhdfe wtype=|`xwt'| wexp=|`xwe'|" ///
            "  reghdfe wtype=|`rwt'| wexp=|`rwe'|"
        local wt_missing = `wt_missing' + 1
    }
}
xpf_eq, id("A21/W-1") left(`wt_missing') right(0) exact ///
    what("e(wtype)/e(wexp) match reghdfe for aw, pw and fw")

* ---------------------------------------------------------------------------
* A.22 (W-2): pweights with vce(unadjusted).
* An unadjusted variance is undefined under probability weights: official Stata
* refuses it (regress ... [pw=], vce(ols) -> r(198)); xhdfe must also fail.
* ---------------------------------------------------------------------------
capture reghdfe y x1 x2 [pw = w], absorb(firm year) vce(unadjusted)
local rc_r = _rc
capture regress y x1 x2 [pw = w], vce(ols)
local rc_s = _rc
capture noisily quietly xhdfe y x1 x2 [pw = w], absorb(firm year) ///
    vce(unadjusted) tolerancemode(reghdfe-comparable) tolerance(1e-12)
local rc_x = _rc
xpf_rc, id("A22/W-2") got(`rc_r') want(9) what("reghdfe refuses pw+vce(unadjusted)")
xpf_rc, id("A22/W-2") got(`rc_s') want(198) what("regress refuses pw+vce(ols)")
xpf_rc, id("A22/W-2") got(`rc_x') want(198) what("xhdfe refuses pw+vce(unadjusted)")

* ---------------------------------------------------------------------------
* A.23 (W-3): the reported constant's variance under MULTIWAY clustering
* with analytic or probability weights.
*
* Root cause, pinned to 1e-13 by direct reconstruction: the augmented ("extended
* bread") rebuild used for the intercept under multiway clustering sets the
* intercept element of the bread to 1/n_rows for aweights and pweights, where
* it must be 1/sum(w). The fweight branch already uses sum(w), which is why
* frequency weights are exact; the unweighted case is exact because sum(w)==n.
*
* Two consequences, both asserted below:
*   (i)  xhdfe's reported se(_cons) is NOT invariant to rescaling an analytic
*        weight, although every other reported quantity is. Doubling w changes
*        se(_cons) by about a factor of two. That is an internal contradiction
*        and does not need reghdfe to adjudicate it.
*   (ii) normalising the weights so that sum(w)==N restores exact agreement
*        with reghdfe, which is the direct fingerprint of the root cause.
*
* Verdict: reghdfe is right. An independent full-dummy Cameron-Gelbach-Miller
* two-way sandwich computed in Mata reproduces reghdfe's Var(_cons) to 4.2e-10
* and does not reproduce xhdfe's at all (29% apart in the variance).
* ---------------------------------------------------------------------------
quietly xhdfe y x1 x2 [aw = w], absorb(firm year) vce(cluster state reg) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
matrix pfx_V_w = e(V)
scalar pfx_c_w  = _se[_cons]
scalar pfx_b_w  = _se[x1]
quietly xhdfe y x1 x2 [aw = w2], absorb(firm year) vce(cluster state reg) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
matrix pfx_V_wb = e(V)
scalar pfx_c_wb = _se[_cons]
scalar pfx_b_wb = _se[x1]
quietly xhdfe y x1 x2 [aw = wnorm], absorb(firm year) vce(cluster state reg) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
scalar pfx_c_wn = _se[_cons]
quietly reghdfe y x1 x2 [aw = w], absorb(firm year) vce(cluster state reg)
matrix pfr_V_w = e(V)
scalar pfr_c_w  = _se[_cons]

* The slope SE is scale-invariant in xhdfe, as it must be: assert it.
xpf_eq, id("A23/M-1") left(pfx_b_w) right(pfx_b_wb) tol(1e-10) ///
    what("se(x1) invariant to the aweight scale")
* Normalising sum(w) to N makes the constant agree with reghdfe: assert it.
xpf_eq, id("A23/M-1") left(pfx_c_wn) right(pfr_c_w) tol(1e-8) ///
    what("se(_cons) agrees once sum(w)==N")
di as text "             se(_cons): aw=w " %21.9g pfx_c_w ///
    "  aw=2*w " %21.9g pfx_c_wb "  aw normalised " %21.9g pfx_c_wn ///
    "  reghdfe " %21.9g pfr_c_w
xcert_assert_matrix_close pfx_V_w pfx_V_wb, tol(1e-10) ///
    name("A23/M-1 aweight-scale invariance of full e(V)")
xcert_assert_matrix_close pfx_V_w pfr_V_w, tol(1e-8) ///
    name("A23/M-1 aweight full e(V) vs reghdfe")

* pweights take exactly the same path.
quietly xhdfe y x1 x2 [pw = w], absorb(firm year) vce(cluster state reg) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
matrix pfx_V_pw = e(V)
quietly reghdfe y x1 x2 [pw = w], absorb(firm year) vce(cluster state reg)
matrix pfr_V_pw = e(V)
xcert_assert_matrix_close pfx_V_pw pfr_V_pw, tol(1e-8) ///
    name("A23/M-1 pweight full e(V) vs reghdfe")

* Frequency weights and the unweighted case must be exact on the same cell.
xcert_parity_spec y x1 x2 [fw = fw], name("A24 fw + 2-way cluster (_cons exact)") ///
    xopts(absorb(firm year) vce(cluster state reg)) ///
    ropts(absorb(firm year) vce(cluster state reg)) skipf
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)
xcert_parity_spec y x1 x2, name("A25 unweighted 2-way cluster (_cons exact)") ///
    xopts(absorb(firm year) vce(cluster state reg)) ///
    ropts(absorb(firm year) vce(cluster state reg)) skipf
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

* ---------------------------------------------------------------------------
* A.26 (W-4): iweights.
* reghdfe refuses iweights outright (r(101)), so the canonical reference here
* is official Stata's own `regress`, which treats an importance weight like a
* frequency weight: e(N) becomes sum(w) and the residual degrees of freedom
* follow. xhdfe accepts iweights and treats them exactly like aweights: e(N)
* stays at the row count. Coefficients agree exactly and the robust variance
* agrees exactly; e(N), e(df_r), e(rmse) and the unadjusted variance must too.
* ---------------------------------------------------------------------------
capture reghdfe y x1 x2 [iw = fw], absorb(firm year)
xpf_rc, id("A26/W-4") got(`= _rc') want(101) what("reghdfe refuses iweights")

quietly regress y x1 x2 i.firm i.year [iw = fw]
scalar pfs_N   = e(N)
scalar pfs_dfr = e(df_r)
scalar pfs_b   = _b[x1]
scalar pfs_se  = _se[x1]
scalar pfs_rms = e(rmse)
quietly regress y x1 x2 i.firm i.year [iw = fw], vce(robust)
scalar pfs_ser = _se[x1]
quietly xhdfe y x1 x2 [iw = fw], absorb(firm year) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
scalar pfx_N   = e(N)
scalar pfx_dfr = e(df_r)
scalar pfx_b   = _b[x1]
scalar pfx_se  = _se[x1]
scalar pfx_rms = e(rmse)
quietly xhdfe y x1 x2 [iw = fw], absorb(firm year) vce(robust) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
scalar pfx_ser = _se[x1]

xpf_eq, id("A26/W-4") left(pfx_b) right(pfs_b) tol(1e-9) ///
    what("iweight point estimate vs full-dummy regress")
xpf_eq, id("A26/W-4") left(pfx_ser) right(pfs_ser) tol(1e-8) ///
    what("iweight robust se vs full-dummy regress")
xpf_eq, id("A26/W-4") left(pfx_N) right(pfs_N) exact what("iweight e(N)")
xpf_eq, id("A26/W-4") left(pfx_dfr) right(pfs_dfr) exact what("iweight e(df_r)")
xpf_eq, id("A26/W-4") left(pfx_rms) right(pfs_rms) tol(1e-9) ///
    what("iweight e(rmse)")
xpf_eq, id("A26/W-4") left(pfx_se) right(pfs_se) tol(1e-8) ///
    what("iweight unadjusted se vs full-dummy regress")

* ===========================================================================
* B. MULTIWAY CLUSTERING BEYOND TWO DIMENSIONS
* ===========================================================================
di as text _n "-- B. multiway clustering ------------------------------------"

xcert_parity_spec y x1 x2, name("B01 cluster 3-way") ///
    xopts(absorb(firm year) vce(cluster state reg div)) ///
    ropts(absorb(firm year) vce(cluster state reg div)) skipf
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2, name("B02 cluster 4-way") ///
    xopts(absorb(firm year) vce(cluster state reg div grp)) ///
    ropts(absorb(firm year) vce(cluster state reg div grp)) skipf
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

* div is a strict coarsening of reg: every reg cell sits inside exactly one div
* cell, so the intersection dimension equals reg and the inclusion-exclusion
* sum has a degenerate term.
xcert_parity_spec y x1 x2, name("B03 cluster 2-way, one dim coarsens the other") ///
    xopts(absorb(firm year) vce(cluster reg div)) ///
    ropts(absorb(firm year) vce(cluster reg div)) skipf
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2, name("B04 cluster 2-way with G=3 on one dim") ///
    xopts(absorb(firm year) vce(cluster grp state)) ///
    ropts(absorb(firm year) vce(cluster grp state)) skipf
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2, name("B05 cluster 3-way, one dim also absorbed") ///
    xopts(absorb(firm year) vce(cluster state year div)) ///
    ropts(absorb(firm year) vce(cluster state year div)) skipf
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [fw = fw], name("B06 cluster 3-way + fweights") ///
    xopts(absorb(firm year) vce(cluster state reg div)) ///
    ropts(absorb(firm year) vce(cluster state reg div)) skipf
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

* ---------------------------------------------------------------------------
* B.07 (C-1): what each engine does when the raw CGM matrix is not
* positive semi-definite.
*
* On the design below the raw three-way CGM variance of (x1, x2, _cons) has
* eigenvalues of both signs, and Var(x1) is negative before any repair, so the
* estimator itself is degenerate. Both engines announce the Cameron-Gelbach-
* Miller adjustment; they then disagree completely about what to report:
* reghdfe returns an all-zero e(V) and no standard errors at all, xhdfe returns
* a finite positive-definite matrix.
*
* There is no unique canonical repair. xhdfe's documented policy is to project
* the matrix onto the PSD cone and expose e(vcv_psd_fixed)==1. This cell asserts
* the raw degeneracy, the repair diagnostic, and both engines' stated policies.
* ---------------------------------------------------------------------------
preserve
    * A design where the raw three-way CGM estimator is genuinely indefinite:
    * div has only four levels and the inclusion-exclusion sum goes negative on
    * the leading diagonal. Fixed seed and construction so the cell keeps
    * exercising the degenerate configuration.
    clear
    set seed 90210
    set obs 4000
    generate long obs   = _n
    generate int  firm  = 1 + int(runiform() * 150)
    generate int  year  = 1 + int(runiform() * 12)
    generate int  state = 1 + int(runiform() * 20)
    generate int  reg   = 1 + int(runiform() * 15)
    generate int  div   = 1 + int(reg / 4)
    generate double x1  = rnormal()
    generate double x2  = rnormal()
    generate double w   = 0.4 + abs(rnormal())
    generate double y   = 0.7 * x1 - 1.3 * x2 + 0.02 * firm - 0.05 * year + rnormal()

    quietly reghdfe y x1 x2, absorb(firm year) vce(cluster state reg div)
    matrix pfr_V = e(V)
    quietly xhdfe y x1 x2, absorb(firm year) vce(cluster state reg div) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    matrix pfx_V = e(V)
    scalar pfx_psd = e(vcv_psd_fixed)

    quietly {
        tempvar dm1 dm2 dmy
        mata: st_local("npos", "0")
    }
    mata:
        Y  = st_data(., "y")
        X0 = st_data(., "x1 x2")
        fv = st_data(., "firm"); yv = st_data(., "year")
        fl = uniqrows(fv); yl = uniqrows(yv)
        D = J(rows(X0), 1, 1)
        for (j = 2; j <= rows(fl); j++) D = D, (fv :== fl[j])
        for (j = 2; j <= rows(yl); j++) D = D, (yv :== yl[j])
        n = rows(X0); Kf = cols(X0) + cols(D)
        Ad = invsym(cross(D, D))
        Xt = X0 - D * (Ad * cross(D, X0))
        yt = Y  - D * (Ad * cross(D, Y))
        Axx = invsym(cross(Xt, Xt))
        bb  = Axx * cross(Xt, yt)
        uu  = yt - Xt * bb
        xb  = (colsum(X0) / n)'
        INF = ((Xt * Axx) :* uu), (((1 / n) :- (Xt * (Axx * xb))) :* uu)
        c1 = st_data(., "state"); c2 = st_data(., "reg"); c3 = st_data(., "div")
        M = xpf_meat(INF, c1) + xpf_meat(INF, c2) + xpf_meat(INF, c3) ///
            - xpf_meat(INF, c1 :* 1000 :+ c2) ///
            - xpf_meat(INF, c1 :* 1000 :+ c3) ///
            - xpf_meat(INF, c2 :* 1000 :+ c3) ///
            + xpf_meat(INF, c1 :* 1000000 :+ c2 :* 1000 :+ c3)
        symeigensystem(M, Q = ., L = .)
        st_numscalar("pf_negeig", sum(L :< 0))
        st_numscalar("pf_rawv11", M[1, 1])
    end

    di as text "             raw 3-way CGM: " pf_negeig ///
        " negative eigenvalue(s), raw Var(x1)=" %21.9g pf_rawv11
    if (pf_negeig == 0) {
        di as error "  FAIL [B07/C-1] the non-PSD cell is no longer degenerate; " ///
            "re-tune the design so the open case is still exercised"
        global PFEAT_FAILS = $PFEAT_FAILS + 1
    }
    local rzero = (pfr_V[1, 1] == 0 & pfr_V[2, 2] == 0 & pfr_V[3, 3] == 0)
    local xzero = (pfx_V[1, 1] == 0 & pfx_V[2, 2] == 0 & pfx_V[3, 3] == 0)
    di as text "             reghdfe e(V) all zero=`rzero'   xhdfe e(V) all zero=`xzero'"
    di as text "             xhdfe Var(x1)=" %21.9g pfx_V[1, 1] ///
        "  reghdfe Var(x1)=" %21.9g pfr_V[1, 1]
    xpf_eq, id("B07/C-1") left(pfx_psd) right(1) exact ///
        what("xhdfe exposes the PSD repair")
    xpf_eq, id("B07/C-1") left(`rzero') right(1) exact ///
        what("reghdfe reports the raw non-PSD case as zero e(V)")
    xpf_eq, id("B07/C-1") left(`xzero') right(0) exact ///
        what("xhdfe reports its documented PSD projection")
    di as text "XHDFE_POLICY_DIVERGENCE|B07/C-1|documented PSD repair versus raw reghdfe convention"
restore

* ===========================================================================
* C. NO ABSORBED DIMENSION + CLUSTERING: the constant in the CRV factor
* ===========================================================================
di as text _n "-- C. no-absorb clustering -----------------------------------"

* ---------------------------------------------------------------------------
* C.01 (NC-1): with no absorb() the clustered small-sample factor must include
* the constant in K.
*
* The CRV1 factor is (N-1)/(N-K) * G/(G-1) with K the number of estimated
* parameters, the intercept included. reghdfe books the intercept as e(df_a)==1
* in noabsorb mode and matches official Stata exactly; xhdfe reports e(df_a)==0
* and uses N-K+1 in the denominator, so its variance is too small by exactly
* (N-K)/(N-K+1).
*
* Adjudicated against Stata's own `regress ..., vce(cluster ...)`, which is
* exact. The baseline layer exempts e(df_a) on the noabsorb cell on the grounds
* that "only the decomposition label differs"; under clustering it does not —
* it moves the reported standard errors.
*
* ---------------------------------------------------------------------------
preserve
    clear
    set seed 99
    set obs 200
    generate long obs   = _n
    generate int  state = 1 + int(runiform() * 12)
    generate int  reg   = 1 + int(runiform() * 10)
    generate double x1  = rnormal()
    generate double x2  = rnormal()
    generate double x3  = rnormal()
    generate double y   = 1 + .7 * x1 - .4 * x2 + .3 * x3 + rnormal()

    * (a) one-way cluster, adjudicated against regress
    quietly regress y x1 x2 x3, vce(cluster state)
    scalar pfs_se = _se[x1]
    scalar pfs_df = e(df_r)
    quietly reghdfe y x1 x2 x3, noabsorb vce(cluster state)
    xpf_eq, id("C01/NC-1") left(_se[x1]) right(pfs_se) tol(1e-10) ///
        what("reghdfe noabsorb 1-way cluster == regress")
    quietly xhdfe y x1 x2 x3, vce(cluster state) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    scalar pfx_se = _se[x1]
    xpf_eq, id("C01/NC-1") left(e(df_r)) right(pfs_df) exact ///
        what("e(df_r) still agrees")
    xpf_eq, id("C01/NC-1") left(pfx_se) right(pfs_se) tol(1e-10) ///
        what("xhdfe noabsorb 1-way cluster == regress")

    * (b) the same factor under two-way clustering
    quietly reghdfe y x1 x2 x3, noabsorb vce(cluster state reg)
    scalar pfr_se = _se[x1]
    quietly xhdfe y x1 x2 x3, vce(cluster state reg) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    xpf_eq, id("C01/NC-1") left(_se[x1]) right(pfr_se) tol(1e-10) ///
        what("xhdfe noabsorb 2-way cluster == reghdfe")

    * (c) control: robust with no absorb is correct in xhdfe
    quietly regress y x1 x2 x3, vce(robust)
    scalar pfs_ser = _se[x1]
    quietly xhdfe y x1 x2 x3, vce(robust) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    xpf_eq, id("C02") left(_se[x1]) right(pfs_ser) tol(1e-10) ///
        what("noabsorb + robust matches regress exactly")

    * (d) control: with an absorbed dimension the same factor is correct
    generate int fe = 1 + mod(obs, 7)
    quietly areg y x1 x2 x3, absorb(fe) vce(cluster state)
    scalar pfs_sea = _se[x1]
    quietly xhdfe y x1 x2 x3, absorb(fe) vce(cluster state) keepsingletons ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    xpf_eq, id("C03") left(_se[x1]) right(pfs_sea) tol(1e-10) ///
        what("absorbed + cluster matches areg exactly")

    * ---------------------------------------------------------------------
    * C.04 The mirror case, where reghdfe is the one that is wrong.
    * `reghdfe ..., noabsorb noconstant` silently keeps the constant in the
    * fit: it returns the with-constant coefficients and standard errors and
    * merely omits _cons from the reported stripe. xhdfe reproduces
    * `regress ..., noconstant` exactly. Asserted against regress, not against
    * reghdfe, and reghdfe's defect is asserted too so that it is noticed if
    * reghdfe ever fixes it.
    * ---------------------------------------------------------------------
    quietly regress y x1 x2 x3, noconstant vce(cluster state)
    scalar pfs_nc_b  = _b[x1]
    scalar pfs_nc_se = _se[x1]
    quietly xhdfe y x1 x2 x3, noconstant vce(cluster state) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    xpf_eq, id("C04") left(_b[x1])  right(pfs_nc_b)  tol(1e-10) ///
        what("xhdfe noconstant b == regress noconstant")
    xpf_eq, id("C04") left(_se[x1]) right(pfs_nc_se) tol(1e-10) ///
        what("xhdfe noconstant se == regress noconstant")
    quietly reghdfe y x1 x2 x3, noabsorb noconstant vce(cluster state)
    scalar pfr_nc_b = _b[x1]
    quietly reghdfe y x1 x2 x3, noabsorb vce(cluster state)
    xpf_eq, id("C04") left(pfr_nc_b) right(_b[x1]) tol(1e-12) ///
        what("reghdfe noabsorb noconstant == reghdfe WITH the constant (its defect)")
    xpf_open, id("C04/RG-1") got(`= abs(pfr_nc_b / pfs_nc_b - 1)') pinned(.00613) ///
        what("reghdfe ignores noconstant under noabsorb; xhdfe is the correct one")
restore

* ===========================================================================
* D. IV / 2SLS versus ivreghdfe
* ===========================================================================
di as text _n "-- D. IV / 2SLS ----------------------------------------------"

capture which ivreghdfe
if (_rc) {
    di as error "  SKIPPED: ivreghdfe is not installed; the IV parity cells " ///
        "cannot run. Nothing is installed by this suite."
    global PFEAT_IV_SKIPPED 1
}
else {
    global PFEAT_IV_SKIPPED 0
    xcert_require_ivreghdfe

    preserve
        clear
        set seed 771
        set obs 6000
        generate long obs   = _n
        generate int  firm  = 1 + int(runiform() * 120)
        generate int  year  = 1 + int(runiform() * 12)
        generate int  state = 1 + int(runiform() * 30)
        generate int  reg   = 1 + int(runiform() * 25)
        generate double x1  = rnormal()
        generate double x2  = rnormal()
        generate double z1  = rnormal()
        generate double z2  = rnormal()
        generate double z3  = rnormal()
        generate double v   = rnormal()
        generate double d1  = .8 * z1 + .4 * z2 + .2 * x1 + firm / 200 + v + rnormal()
        generate double d2  = .5 * z2 + .7 * z3 - .2 * x2 + year / 20 + v + rnormal()
        generate double y   = 1 + .7 * x1 - .4 * x2 + 1.8 * d1 - .9 * d2 ///
            + firm / 40 + year / 10 + v + rnormal()
        * weights are drawn last so that y keeps the draw that makes the
        * two-way CGM 2SLS matrix indefinite in D12
        generate double wa  = 0.4 + abs(rnormal())
        generate int    fwi = 1 + mod(obs, 3)

        * ivreg2/ivreghdfe report large-sample statistics unless `small' is
        * given; with absorb() they force the small-sample correction, without
        * it they do not. xhdfe always applies it, matching reghdfe. The `small'
        * option is therefore part of the specification, not an exemption: the
        * exact factor is verified in D09 below.
        capture program drop xpf_iv
        program define xpf_iv
            version 16
            syntax, ID(string) ENDO(string) INST(string) ///
                IVOPT(string) [XOPT(string) WT(string) EXCEPT(string) ///
                BTOL(real 1e-8) VTOL(real 1e-7) OPENVAR(real 0) OPENPIN(real 0)]

            quietly ivreghdfe y x1 x2 (`endo' = `inst') `wt', `ivopt'
            matrix pv_b = e(b)
            matrix pv_V = e(V)
            scalar pv_N   = e(N)
            scalar pv_dfr = e(df_r)
            scalar pv_dfa = e(df_a)
            scalar pv_dfm = e(df_m)
            scalar pv_ncl = e(N_clust)
            quietly xhdfe y x1 x2 `endo' `wt', endogenous(`endo') ///
                instruments(`inst') `xopt' ///
                tolerancemode(reghdfe-comparable) tolerance(1e-12) ///
                noheader notable nofootnote
            matrix px_b = e(b)
            matrix px_V = e(V)

            * discrete conventions: exact
            foreach s in N dfr dfa dfm ncl {
                if (strpos(" `except' ", " `s' ")) {
                    di as text "  [`id'] e(`s') exempt: see call site"
                    continue
                }
                local xv = cond("`s'" == "N", e(N), ///
                           cond("`s'" == "dfr", e(df_r), ///
                           cond("`s'" == "dfa", e(df_a), ///
                           cond("`s'" == "dfm", e(df_m), e(N_clust)))))
                local rv = scalar(pv_`s')
                if (missing(`xv') != missing(`rv') | ///
                    (!missing(`xv') & !missing(`rv') & `xv' != `rv')) {
                    di as error "  FAIL [`id'] e(`s') EXACT mismatch: " ///
                        "xhdfe=`xv' ivreghdfe=`rv'"
                    global PFEAT_FAILS = $PFEAT_FAILS + 1
                }
            }

            local cn : colnames pv_b
            local worst = 0
            local wname ""
            foreach c of local cn {
                local ci = colnumb(pv_b, "`c'")
                local xi = colnumb(px_b, "`c'")
                if (`xi' == .) {
                    di as error "  FAIL [`id'] coefficient `c' absent from xhdfe e(b)"
                    global PFEAT_FAILS = $PFEAT_FAILS + 1
                    continue
                }
                local db = abs(pv_b[1, `ci'] - px_b[1, `xi']) / max(1, abs(pv_b[1, `ci']))
                if (`db' > `btol') {
                    di as error "  FAIL [`id'] b[`c'] rel diff " %9.2e `db'
                    global PFEAT_FAILS = $PFEAT_FAILS + 1
                }
                local dv = abs(px_V[`xi', `xi'] / pv_V[`ci', `ci'] - 1)
                if (`dv' > `worst') {
                    local worst = `dv'
                    local wname "`c'"
                }
                if (`dv' > `vtol' & `openvar' == 0) {
                    di as error "  FAIL [`id'] Var[`c'] rel diff " %9.2e `dv'
                    global PFEAT_FAILS = $PFEAT_FAILS + 1
                }
            }
            if (`openvar' == 0) {
                di as text "  ok   [`id'] worst Var rel diff " %9.2e `worst' " on `wname'"
            }
            else {
                di as text "             worst Var rel diff " %9.2e `worst' " on `wname'"
                xpf_open, id("`id'") got(`worst') pinned(`openpin') ///
                    what("IV variance divergence vs ivreghdfe")
            }
        end

        xpf_iv, id("D01 1 endo 1 iv, 2FE, unadjusted") endo("d1") inst("z1") ///
            ivopt("absorb(firm year) small") xopt("absorb(firm year)")
        xpf_iv, id("D02 1 endo 2 iv (overidentified)") endo("d1") inst("z1 z2") ///
            ivopt("absorb(firm year) small") xopt("absorb(firm year)")
        xpf_iv, id("D03 2 endo 3 iv") endo("d1 d2") inst("z1 z2 z3") ///
            ivopt("absorb(firm year) small") xopt("absorb(firm year)")
        xpf_iv, id("D04 2 endo 2 iv (just identified)") endo("d1 d2") inst("z1 z3") ///
            ivopt("absorb(firm year) small") xopt("absorb(firm year)")
        xpf_iv, id("D05 2 endo 3 iv, robust") endo("d1 d2") inst("z1 z2 z3") ///
            ivopt("absorb(firm year) robust small") xopt("absorb(firm year) robust")
        xpf_iv, id("D06 2 endo 3 iv, cluster 1-way") endo("d1 d2") inst("z1 z2 z3") ///
            ivopt("absorb(firm year) cluster(state) small") ///
            xopt("absorb(firm year) cluster(state)")
        * e(df_a) is exempt on this cell only: with noabsorb ivreghdfe posts no
        * e(df_a) at all, while xhdfe posts 0. e(df_r) is asserted equal, so the
        * inferential total agrees.
        xpf_iv, id("D07 2 endo, no absorbed FE") endo("d1 d2") inst("z1 z2 z3") ///
            ivopt("noabsorb small") except(dfa)
        xpf_iv, id("D08 1 endo 1 iv + aweights") endo("d1") inst("z1") ///
            ivopt("absorb(firm year) small") xopt("absorb(firm year)") wt("[aw=wa]")
        xpf_iv, id("D08b 1 endo 1 iv + fweights") endo("d1") inst("z1") ///
            ivopt("absorb(firm year) small") xopt("absorb(firm year)") wt("[fw=fwi]")

        * -----------------------------------------------------------------
        * D.09 the `small' option is a convention of ivreg2, not an exemption:
        * without it the reported variance is larger by exactly N/(N-K).
        * -----------------------------------------------------------------
        quietly ivreghdfe y x1 x2 (d1 d2 = z1 z2 z3), noabsorb
        scalar pfi_large = _se[d1]
        quietly ivreghdfe y x1 x2 (d1 d2 = z1 z2 z3), noabsorb small
        scalar pfi_small = _se[d1]
        quietly xhdfe y x1 x2 d1 d2, endogenous(d1 d2) instruments(z1 z2 z3) ///
            tolerancemode(reghdfe-comparable) tolerance(1e-12)
        xpf_eq, id("D09") left(`= (_se[d1] / pfi_large)^2') right(`= 6000 / 5995') ///
            tol(1e-12) what("ivreg2 large-sample default differs by exactly N/(N-K)")
        xpf_eq, id("D09") left(_se[d1]) right(pfi_small) tol(1e-9) ///
            what("with the small option the two engines agree")

        * -----------------------------------------------------------------
        * D.10 OPEN: xhdfe posts no first-stage or weak-identification
        * statistics, so none can be compared. ivreghdfe posts e(widstat),
        * e(j), e(rkf); xhdfe posts none of them. Recorded, not failed.
        * -----------------------------------------------------------------
        quietly ivreghdfe y x1 x2 (d1 d2 = z1 z2 z3), absorb(firm year) small
        local iv_stats 0
        foreach s in widstat j rkf {
            if (!missing(e(`s'))) local iv_stats = `iv_stats' + 1
        }
        quietly xhdfe y x1 x2 d1 d2, absorb(firm year) endogenous(d1 d2) ///
            instruments(z1 z2 z3) tolerancemode(reghdfe-comparable) tolerance(1e-12)
        local xh_stats 0
        foreach s in widstat j rkf {
            if (!missing(e(`s'))) local xh_stats = `xh_stats' + 1
        }
        di as text "             weak-id statistics posted: ivreghdfe `iv_stats'" ///
            " of 3, xhdfe `xh_stats' of 3"
        xpf_open, id("D10") got(`= 3 - `xh_stats'') pinned(3) ///
            what("xhdfe posts no first-stage / weak-identification statistics")

        * -----------------------------------------------------------------
        * D.11 (IV-1): IV with no absorbed dimension and clustering must
        * charge the constant in the small-sample factor.
        * -----------------------------------------------------------------
        quietly ivreghdfe y x1 x2 (d1 d2 = z1 z2 z3), noabsorb cluster(state reg) small
        scalar pfi_se = _se[d1]
        quietly xhdfe y x1 x2 d1 d2, endogenous(d1 d2) instruments(z1 z2 z3) ///
            cluster(state reg) tolerancemode(reghdfe-comparable) tolerance(1e-12)
        xpf_eq, id("D11/IV-1") left(_se[d1]) right(pfi_se) tol(1e-9) ///
            what("IV + noabsorb + cluster small-sample factor")

        * -----------------------------------------------------------------
        * D.12 (IV-2): IV with absorbed FEs and MULTIWAY clustering,
        * when the raw CGM matrix is not PSD.
        *
        * An exact full-dummy two-stage-least-squares CGM sandwich computed in
        * Mata reproduces ivreghdfe's reported variances to about 3e-14 and does
        * NOT reproduce xhdfe's: the 4x4 block of the exact estimator has one
        * negative eigenvalue, ivreg2 reports it as it stands, and xhdfe
        * applies the Cameron-Gelbach-Miller repair (announced in its output).
        * The reported variance on the exogenous regressors moves by up to 19%.
        *
        * Verdict: ivreghdfe reproduces the textbook estimator exactly; xhdfe
        * reports its documented PSD projection and exposes the repair through
        * e(vcv_psd_fixed). This is an intentional policy divergence.
        * -----------------------------------------------------------------
        quietly ivreghdfe y x1 x2 (d1 d2 = z1 z2 z3), absorb(firm year) ///
            cluster(state reg) small
        matrix pfi_V = e(V)
        matrix pfi_b = e(b)
        quietly xhdfe y x1 x2 d1 d2, absorb(firm year) endogenous(d1 d2) ///
            instruments(z1 z2 z3) cluster(state reg) ///
            tolerancemode(reghdfe-comparable) tolerance(1e-12)
        matrix pfx_V = e(V)
        matrix pfx_b = e(b)
        scalar pfx_psd = e(vcv_psd_fixed)

        mata:
            Y  = st_data(., "y")
            XX = st_data(., "x1 x2 d1 d2")
            ZZ = st_data(., "x1 x2 z1 z2 z3")
            fv = st_data(., "firm"); yv = st_data(., "year")
            fl = uniqrows(fv); yl = uniqrows(yv)
            D = J(rows(XX), 1, 1)
            for (j = 2; j <= rows(fl); j++) D = D, (fv :== fl[j])
            for (j = 2; j <= rows(yl); j++) D = D, (yv :== yl[j])
            X = XX, D
            Z = ZZ, D
            n = rows(X); K = cols(X)
            PzX = Z * (invsym(cross(Z, Z)) * cross(Z, X))
            A = invsym(cross(PzX, PzX))
            b = A * cross(PzX, Y)
            u = Y - X * b
            S = PzX :* u
            c1 = st_data(., "state"); c2 = st_data(., "reg")
            G1 = rows(uniqrows(c1)); G2 = rows(uniqrows(c2))
            gm = min((G1, G2))
            M = xpf_meat(S, c1) + xpf_meat(S, c2) - xpf_meat(S, c1 :* 1000 :+ c2)
            V = ((n - 1) / (n - K)) * (gm / (gm - 1)) * A * M * A'
            V4 = V[1..4, 1..4]
            symeigensystem(V4, Q4 = ., L4 = .)
            st_numscalar("pf_iv_neg", sum(L4 :< 0))
            st_matrix("pf_iv_V", V4)
        end

        * exact reference vs each engine, on the exogenous regressor x1
        local ci = colnumb(pfi_b, "x1")
        local xi = colnumb(pfx_b, "x1")
        scalar pf_iv_exact = pf_iv_V[1, 1]
        di as text "             exact CGM 2SLS Var(x1)=" %21.9g pf_iv_exact ///
            "  ivreghdfe=" %21.9g pfi_V[`ci', `ci'] "  xhdfe=" %21.9g pfx_V[`xi', `xi']
        di as text "             raw 4x4 block has " pf_iv_neg " negative eigenvalue(s)"
        xpf_eq, id("D12/IV-2") left(pf_iv_exact) right(pfi_V[`ci', `ci']) tol(1e-10) ///
            what("ivreghdfe reproduces the exact CGM 2SLS variance")
        xpf_eq, id("D12/IV-2") left(pfx_psd) right(1) exact ///
            what("xhdfe exposes the PSD repair")
        xpf_eq, id("D12/IV-2") left(pf_iv_neg) right(1) exact ///
            what("raw 2SLS CGM block is genuinely non-PSD")
        di as text "XHDFE_POLICY_DIVERGENCE|D12/IV-2|documented PSD repair versus raw ivreghdfe convention"
    restore
}

* ===========================================================================
* E. FIXED-EFFECT RECOVERY
* ===========================================================================
di as text _n "-- E. fixed-effect recovery ----------------------------------"

preserve
    quietly keep if obs <= 3000
    * A two-dimensional FE is identified only up to a constant shifted between
    * dimensions, so per-dimension equality is not guaranteed by the estimator.
    * What IS well defined is the sum of the recovered effects, and hence the
    * fitted values and residuals. Both are asserted; the per-dimension
    * agreement is measured and reported, not required.
    quietly reghdfe y x1 x2, absorb(RFf=firm RFy=year) tolerance(1e-12)
    scalar pfr_dfa = e(df_a)
    scalar pfr_rss = e(rss)
    quietly xhdfe y x1 x2, absorb(XFf=firm XFy=year) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    xpf_eq, id("E01") left(e(df_a)) right(pfr_dfa) exact what("savefe e(df_a)")
    xpf_eq, id("E01") left(e(rss))  right(pfr_rss) tol(1e-9) what("savefe e(rss)")
    xpf_eq, id("E01") left(e(fe_recovery_converged)) right(1) exact ///
        what("fe recovery converged")

    quietly generate double pf_sumR = RFf + RFy
    quietly generate double pf_sumX = XFf + XFy
    xcert_assert_var_close pf_sumR pf_sumX if e(sample), tol(1e-7) ///
        name("E01 sum of recovered FEs (the identified object)")
    quietly predict double pf_xb, xb
    quietly generate double pf_fitR = pf_xb + pf_sumR
    quietly generate double pf_fitX = pf_xb + pf_sumX
    xcert_assert_var_close pf_fitR pf_fitX if e(sample), tol(1e-7) ///
        name("E01 fitted values from the recovered FEs")
    * measured, not required: the two engines happen to pick the same
    * normalisation, so the per-dimension effects also agree
    xcert_assert_var_close RFf XFf if e(sample), tol(1e-7) ///
        name("E01 per-dimension firm effect (same normalisation)")
    xcert_assert_var_close RFy XFy if e(sample), tol(1e-7) ///
        name("E01 per-dimension year effect (same normalisation)")

    * ---- disconnected components -------------------------------------------
    * Two mobility groups that share no firm and no year: the level of each
    * component is separately unidentified, so only within-component sums and
    * the fitted values are comparable.
    quietly drop RFf RFy XFf XFy pf_sumR pf_sumX pf_xb pf_fitR pf_fitX
    quietly generate int firmD = firm + 1000 * (obs > 1500)
    quietly generate int yearD = year + 100 * (obs > 1500)
    quietly reghdfe y x1 x2, absorb(RFf=firmD RFy=yearD) tolerance(1e-12)
    scalar pfr_dfa = e(df_a)
    quietly xhdfe y x1 x2, absorb(XFf=firmD XFy=yearD) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    xpf_eq, id("E02") left(e(df_a)) right(pfr_dfa) exact ///
        what("disconnected graph e(df_a)")
    quietly generate double pf_sumR = RFf + RFy
    quietly generate double pf_sumX = XFf + XFy
    xcert_assert_var_close pf_sumR pf_sumX if e(sample), tol(1e-6) ///
        name("E02 disconnected: sum of recovered FEs")
    quietly predict double pf_xb, xb
    quietly generate double pf_fitR = pf_xb + pf_sumR
    quietly generate double pf_fitX = pf_xb + pf_sumX
    xcert_assert_var_close pf_fitR pf_fitX if e(sample), tol(1e-6) ///
        name("E02 disconnected: fitted values")

    * ---- absorb(..., savefe): the reghdfe-style alias -----------------------
    quietly drop RFf RFy XFf XFy pf_sumR pf_sumX pf_xb pf_fitR pf_fitX
    capture drop __hdfe*
    capture xhdfe y x1 x2, absorb(firm year state, savefe) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
    xpf_rc, id("E03") got(`= _rc') want(0) what("xhdfe absorb(..., savefe)")
    local xl1 : variable label __hdfe1__
    local xl2 : variable label __hdfe2__
    local xl3 : variable label __hdfe3__
    capture drop __hdfe*
    capture reghdfe y x1 x2, absorb(firm year state, savefe)
    xpf_rc, id("E03") got(`= _rc') want(0) what("reghdfe absorb(..., savefe)")
    local rl1 : variable label __hdfe1__
    local rl2 : variable label __hdfe2__
    local rl3 : variable label __hdfe3__
    local lab_diff = (`"`xl1'"' != `"`rl1'"') + (`"`xl2'"' != `"`rl2'"') ///
        + (`"`xl3'"' != `"`rl3'"')
    di as text "             xhdfe   labels: |`xl1'| |`xl2'| |`xl3'|"
    di as text "             reghdfe labels: |`rl1'| |`rl2'| |`rl3'|"
    xpf_eq, id("E03/FE-1") left(`lab_diff') right(0) exact ///
        what("savefe variable labels match reghdfe")
    capture drop __hdfe*

    * ---- savefe with weights -----------------------------------------------
    quietly reghdfe y x1 x2 [aw = w], absorb(RWf=firm RWy=year) tolerance(1e-12)
    quietly xhdfe y x1 x2 [aw = w], absorb(XWf=firm XWy=year) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    quietly generate double pf_swR = RWf + RWy
    quietly generate double pf_swX = XWf + XWy
    xcert_assert_var_close pf_swR pf_swX if e(sample), tol(1e-7) ///
        name("E04 weighted savefe: sum of recovered FEs")
restore

* ===========================================================================
* F. HETEROGENEOUS SLOPES
* ===========================================================================
di as text _n "-- F. heterogeneous slopes -----------------------------------"

xcert_parity_spec y x1 x2, name("F01 absorb(firm##c.zz year)") ///
    xopts(absorb(firm##c.zz year)) ropts(absorb(firm##c.zz year))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2, name("F02 two slope vars on one carrier") ///
    xopts(absorb(firm##c.zz firm##c.qq year)) ///
    ropts(absorb(firm##c.zz firm##c.qq year))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2, name("F03 slopes + cluster") ///
    xopts(absorb(firm##c.zz year) vce(cluster state)) ///
    ropts(absorb(firm##c.zz year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2, name("F04 slopes + 2-way cluster") ///
    xopts(absorb(firm##c.zz year) vce(cluster state reg)) ///
    ropts(absorb(firm##c.zz year) vce(cluster state reg)) skipf
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [aw = w], name("F05 slopes + aweights") ///
    xopts(absorb(firm##c.zz year)) ropts(absorb(firm##c.zz year))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [fw = fw], name("F06 slopes + fweights") ///
    xopts(absorb(firm##c.zz year)) ropts(absorb(firm##c.zz year))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2 [aw = w], name("F07 slopes + aweights + cluster") ///
    xopts(absorb(firm##c.zz year) vce(cluster state)) ///
    ropts(absorb(firm##c.zz year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 x2, name("F08 slope-only terms on both dimensions") ///
    xopts(absorb(firm#c.zz year#c.qq)) ropts(absorb(firm#c.zz year#c.qq))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

* ---------------------------------------------------------------------------
* F.09 (HS-1): a slope-only absorbed term combined with an intercept term.
*
* absorb(firm#c.zz year) spans 150 firm-specific slopes plus 12 year
* intercepts. The slope block contains no intercept, so it is not collinear
* with the constant and nothing is redundant: the full-dummy design
* `regress y x1 x2 c.zz#i.firm i.year` has rank df_m+1 and df_r = N-rank.
*
* Both engines must reproduce the exact full-dummy rank and inference.
* ---------------------------------------------------------------------------
quietly regress y x1 x2 c.zz#i.firm i.year
scalar pfs_dfr = e(df_r)
scalar pfs_se  = _se[x1]
scalar pfs_rms = e(rmse)
quietly reghdfe y x1 x2, absorb(firm#c.zz year) tolerance(1e-12)
scalar pfr_dfr = e(df_r)
scalar pfr_rss = e(rss)
xpf_eq, id("F09/HS-1") left(pfr_dfr) right(pfs_dfr) exact ///
    what("reghdfe e(df_r) == full-dummy regress")
xpf_eq, id("F09/HS-1") left(_se[x1]) right(pfs_se) tol(1e-9) ///
    what("reghdfe se(x1) == full-dummy regress")
quietly reghdfe y x1 x2, absorb(firm#c.zz year) tolerance(1e-12)
scalar pfr_b1 = _b[x1]
quietly xhdfe y x1 x2, absorb(firm#c.zz year) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
scalar pfx_dfr = e(df_r)
xpf_eq, id("F09/HS-1") left(e(rss)) right(pfr_rss) tol(1e-9) ///
    what("e(rss) identical to reghdfe (the fit is correct)")
xpf_eq, id("F09/HS-1") left(_b[x1]) right(pfr_b1) tol(1e-9) ///
    what("b[x1] identical to reghdfe (the fit is correct)")
xpf_eq, id("F09/HS-1") left(pfx_dfr) right(pfs_dfr) exact ///
    what("xhdfe e(df_r) == full-dummy regress")
xpf_eq, id("F09/HS-1") left(_se[x1]) right(pfs_se) tol(1e-9) ///
    what("xhdfe se(x1) == full-dummy regress")

* ---------------------------------------------------------------------------
* F.10 (HS-2): grouped c.(varlist) syntax inside absorb().
* ---------------------------------------------------------------------------
capture reghdfe y x1 x2, absorb(firm##c.(zz qq) year)
local rc_r = _rc
capture xhdfe y x1 x2, absorb(firm##c.(zz qq) year) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12) noheader notable nofootnote
local rc_x = _rc
xpf_rc, id("F10/HS-2") got(`rc_r') want(0) what("reghdfe accepts absorb(f##c.(a b))")
xpf_rc, id("F10/HS-2") got(`rc_x') want(0) what("xhdfe accepts absorb(f##c.(a b))")
quietly regress y x1 x2 i.firm c.zz#i.firm c.qq#i.firm i.year
scalar pfs_se = _se[x1]
quietly xhdfe y x1 x2, absorb(firm##c.zz firm##c.qq year) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
xpf_eq, id("F10/HS-2") left(_se[x1]) right(pfs_se) tol(1e-9) ///
    what("the expanded spelling matches full-dummy regress exactly")

* F.11 a slope-only absorb with no intercept dimension: both engines fit the
* no-constant model, exactly as `regress ..., noconstant` does. Asserted
* against regress, so the shared convention is anchored to the exact fit.
quietly regress y x1 x2 c.zz#i.firm, noconstant
scalar pfs_b   = _b[x1]
scalar pfs_se  = _se[x1]
scalar pfs_dfr = e(df_r)
quietly reghdfe y x1 x2, absorb(firm#c.zz) tolerance(1e-12)
xpf_eq, id("F11") left(_se[x1]) right(pfs_se) tol(1e-9) ///
    what("reghdfe slope-only-alone == regress noconstant")
quietly xhdfe y x1 x2, absorb(firm#c.zz) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
xpf_eq, id("F11") left(_b[x1])  right(pfs_b)  tol(1e-9) ///
    what("xhdfe slope-only-alone b == regress noconstant")
xpf_eq, id("F11") left(_se[x1]) right(pfs_se) tol(1e-9) ///
    what("xhdfe slope-only-alone se == regress noconstant")
xpf_eq, id("F11") left(e(df_r)) right(pfs_dfr) exact ///
    what("xhdfe slope-only-alone e(df_r) == regress noconstant")

* ===========================================================================
* G. FACTOR VARIABLES, INTERACTIONS AND THE COEFFICIENT STRIPE
* ===========================================================================
di as text _n "-- G. factor variables ---------------------------------------"

xcert_parity_spec y x1 ib3.cat, name("G01 non-default base level ib3.cat") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 i.cat##c.x2, name("G02 i.cat##c.x2") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 i.cat##i.grp, name("G03 i.cat##i.grp") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 c.x2#c.zz, name("G04 c.x2#c.zz") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 i.cat, name("G05 factor vars + noconstant") ///
    xopts(absorb(firm year) noconstant) ropts(absorb(firm year) noconstant)
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 i.cat, name("G06 string-derived absorb dimension") ///
    xopts(absorb(sfirm year)) ropts(absorb(sfirm year))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 i.cat, name("G07 absorb(firm#year) interaction") ///
    xopts(absorb(firm#year)) ropts(absorb(firm#year))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

xcert_parity_spec y x1 i.cat, name("G08 absorb(i.firm i.year) explicit i.") ///
    xopts(absorb(i.firm i.year)) ropts(absorb(i.firm i.year))
global PFEAT_FAILS = $PFEAT_FAILS + r(fails)

* G.09 the exact coefficient-name stripe posted in e(b)/e(V).
foreach spec in "y x1 i.cat##i.grp c.x2#c.zz" "y x1 ib3.cat i.cat##c.x2" {
    quietly xhdfe `spec', absorb(firm year) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    local xstripe : colnames e(b)
    local xVr : rownames e(V)
    quietly reghdfe `spec', absorb(firm year)
    local rstripe : colnames e(b)
    if (`"`xstripe'"' != `"`rstripe'"') {
        di as error "  FAIL [G09] e(b) stripe differs for: `spec'"
        di as error "         xhdfe  : `xstripe'"
        di as error "         reghdfe: `rstripe'"
        global PFEAT_FAILS = $PFEAT_FAILS + 1
    }
    else {
        di as text "  ok   [G09] e(b) stripe identical for: `spec'"
    }
    if (`"`xVr'"' != `"`xstripe'"') {
        di as error "  FAIL [G09] xhdfe e(V) rownames differ from e(b) colnames"
        global PFEAT_FAILS = $PFEAT_FAILS + 1
    }
}

* Adversarial single-# boundaries omitted from the original layer. These are
* hard oracle assertions: coefficient/RSS errors cannot be carried as benign
* metadata pins. Each design is deliberately non-collinear with the FEs.
foreach spec in "ib3.cat#i.grp" "i.cat#i.grp#i.div" {
    quietly regress y x1 `spec' i.firm i.year
    scalar pfg_rss = e(rss)
    scalar pfg_b   = _b[x1]
    scalar pfg_dfr = e(df_r)
    quietly xhdfe y x1 `spec', absorb(firm year) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    xpf_eq, id("G09A") left(e(rss)) right(pfg_rss) tol(1e-9) ///
        what("`spec': e(rss) equals full-dummy regress")
    xpf_eq, id("G09A") left(_b[x1]) right(pfg_b) tol(1e-9) ///
        what("`spec': b[x1] equals full-dummy regress")
    xpf_eq, id("G09A") left(e(df_r)) right(pfg_dfr) exact ///
        what("`spec': e(df_r) equals full-dummy regress")
}

quietly regress y x1 i.cat#i.grp
scalar pfg_rss = e(rss)
scalar pfg_b   = _b[x1]
scalar pfg_dfr = e(df_r)
quietly xhdfe y x1 i.cat#i.grp, ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
xpf_eq, id("G09B") left(e(rss)) right(pfg_rss) tol(1e-9) ///
    what("i.cat#i.grp without absorb(): e(rss) equals regress")
xpf_eq, id("G09B") left(_b[x1]) right(pfg_b) tol(1e-9) ///
    what("i.cat#i.grp without absorb(): b[x1] equals regress")
xpf_eq, id("G09B") left(e(df_r)) right(pfg_dfr) exact ///
    what("i.cat#i.grp without absorb(): e(df_r) equals regress")

* ---------------------------------------------------------------------------
* G.10 regression guard (FV-1): a single-# interaction of two factors must not
* be parameterised as if the main effects were present.
*
* i.cat#i.grp with no main effects identifies 12 cells minus one absorbed by
* the constant. reghdfe estimates 11 and reproduces
* `regress y x1 i.cat#i.grp i.firm i.year` exactly, to the last digit of
* e(rss), e(df_r), b[x1] and se(x1). xhdfe estimates only the 6 non-base cells,
* dropping the five identified cells that involve a base level, which makes the
* fitted model a strict subset: e(rss) is LARGER, so this is a
* misspecification, not a re-parameterisation.
*
* Verdict: reghdfe and the exact full-dummy fit are right; xhdfe is wrong, and
* the point estimate of x1 moves.
* ---------------------------------------------------------------------------
quietly regress y x1 i.cat#i.grp i.firm i.year
scalar pfs_dfr = e(df_r)
scalar pfs_rss = e(rss)
scalar pfs_b   = _b[x1]
scalar pfs_se  = _se[x1]
quietly reghdfe y x1 i.cat#i.grp, absorb(firm year) tolerance(1e-12)
local pfs_stripe : colnames e(b)
xpf_eq, id("G10/FV-1") left(e(df_r)) right(pfs_dfr) exact ///
    what("reghdfe e(df_r) == full-dummy regress")
xpf_eq, id("G10/FV-1") left(e(rss)) right(pfs_rss) tol(1e-9) ///
    what("reghdfe e(rss) == full-dummy regress")
xpf_eq, id("G10/FV-1") left(_b[x1]) right(pfs_b) tol(1e-9) ///
    what("reghdfe b[x1] == full-dummy regress")
quietly xhdfe y x1 i.cat#i.grp, absorb(firm year) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
local pfx_stripe : colnames e(b)
di as text "             e(rss): xhdfe " %21.13g e(rss) " exact " %21.13g pfs_rss ///
    " | e(df_m): xhdfe " e(df_m) " reghdfe 12"
di as text "             b[x1]: xhdfe " %21.13g _b[x1] " exact " %21.13g pfs_b
xpf_eq, id("G10/FV-1") left(e(df_r)) right(pfs_dfr) exact ///
    what("xhdfe e(df_r) == full-dummy regress")
xpf_eq, id("G10/FV-1") left(e(rss)) right(pfs_rss) tol(1e-9) ///
    what("xhdfe e(rss) == full-dummy regress")
xpf_eq, id("G10/FV-1") left(_b[x1]) right(pfs_b) tol(1e-9) ///
    what("xhdfe b[x1] == full-dummy regress")
xpf_eq, id("G10/FV-1") left(_se[x1]) right(pfs_se) tol(1e-8) ///
    what("xhdfe se[x1] == full-dummy regress")
if (`"`pfx_stripe'"' != `"`pfs_stripe'"') {
    di as error "  FAIL [G10/FV-1] coefficient stripe differs from reghdfe"
    global PFEAT_FAILS = $PFEAT_FAILS + 1
}

* ---------------------------------------------------------------------------
* G.11 regression guard (FV-2): factor-by-continuous single-#
* interaction. i.grp#c.x2 with no x2 main effect identifies all three
* group-specific slopes. reghdfe estimates all three and matches the exact
* full-dummy fit; xhdfe marks the base-level slope as omitted (1b.grp#co.x2)
* and estimates two, raising e(rss) by 46%.
* ---------------------------------------------------------------------------
quietly regress y x1 i.grp#c.x2 i.firm i.year
scalar pfs_dfr = e(df_r)
scalar pfs_rss = e(rss)
scalar pfs_b   = _b[x1]
quietly reghdfe y x1 i.grp#c.x2, absorb(firm year) tolerance(1e-12)
local pfs_stripe : colnames e(b)
xpf_eq, id("G11/FV-2") left(e(df_r)) right(pfs_dfr) exact ///
    what("reghdfe e(df_r) == full-dummy regress")
xpf_eq, id("G11/FV-2") left(e(rss)) right(pfs_rss) tol(1e-9) ///
    what("reghdfe e(rss) == full-dummy regress")
quietly xhdfe y x1 i.grp#c.x2, absorb(firm year) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
local pfx_stripe : colnames e(b)
di as text "             e(rss): xhdfe " %21.13g e(rss) " exact " %21.13g pfs_rss ///
    " | stripe: `: colnames e(b)'"
xpf_eq, id("G11/FV-2") left(e(df_r)) right(pfs_dfr) exact ///
    what("xhdfe e(df_r) == full-dummy regress")
xpf_eq, id("G11/FV-2") left(e(rss)) right(pfs_rss) tol(1e-9) ///
    what("xhdfe e(rss) == full-dummy regress")
xpf_eq, id("G11/FV-2") left(_b[x1]) right(pfs_b) tol(1e-9) ///
    what("xhdfe b[x1] == full-dummy regress")
if (`"`pfx_stripe'"' != `"`pfs_stripe'"') {
    di as error "  FAIL [G11/FV-2] coefficient stripe differs from reghdfe"
    global PFEAT_FAILS = $PFEAT_FAILS + 1
}

* ---------------------------------------------------------------------------
* G.12 regression guard (FV-3): an explicitly omitted term (o.x2) must not add
* a phantom column named after an internal temporary variable.
* reghdfe posts the stripe "x1 o.x2 zz _cons"; xhdfe posts
* "x1 o.x2 zz o.__xhdfe_1 _cons" — an extra omitted column carrying the
* tempvar name __xhdfe_1. Any consumer of e(b)/e(V) sees a coefficient whose
* name is an implementation detail.
* ---------------------------------------------------------------------------
quietly reghdfe y x1 o.x2 zz, absorb(firm year) tolerance(1e-12)
local rstripe : colnames e(b)
scalar pfr_rss = e(rss)
quietly xhdfe y x1 o.x2 zz, absorb(firm year) ///
    tolerancemode(reghdfe-comparable) tolerance(1e-12)
local xstripe : colnames e(b)
xpf_eq, id("G12/FV-3") left(e(rss)) right(pfr_rss) tol(1e-9) ///
    what("the fit itself is unaffected by the phantom column")
di as text "             reghdfe stripe: `rstripe'"
di as text "             xhdfe   stripe: `xstripe'"
local nx : word count `xstripe'
local nr : word count `rstripe'
xpf_eq, id("G12/FV-3") left(`nx') right(`nr') exact ///
    what("e(b) column count matches reghdfe")
if (`"`xstripe'"' != `"`rstripe'"') {
    di as error "  FAIL [G12/FV-3] coefficient stripe differs from reghdfe"
    global PFEAT_FAILS = $PFEAT_FAILS + 1
}

* ===========================================================================
* H. POSTESTIMATION
* ===========================================================================
di as text _n "-- H. postestimation -----------------------------------------"

preserve
    capture drop pf_*
    quietly reghdfe y x1 x2 i.cat [aw = w], absorb(firm year) ///
        vce(cluster state) residuals(pf_rres) tolerance(1e-12)
    foreach s in xb xbd d stdp score {
        capture predict double pf_r_`s', `s'
        if (_rc) {
            di as error "  FAIL [H01] reghdfe predict `s' rc=" _rc
            global PFEAT_FAILS = $PFEAT_FAILS + 1
        }
    }
    quietly generate byte pf_rsamp = e(sample)
    quietly xhdfe y x1 x2 i.cat [aw = w], absorb(firm year) ///
        vce(cluster state) residuals(pf_xres) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    foreach s in xb xbd d stdp score {
        capture predict double pf_x_`s', `s'
        if (_rc) {
            di as error "  FAIL [H01] xhdfe predict `s' rc=" _rc
            global PFEAT_FAILS = $PFEAT_FAILS + 1
        }
    }
    quietly generate byte pf_xsamp = e(sample)
    quietly count if pf_rsamp != pf_xsamp
    xpf_eq, id("H01") left(r(N)) right(0) exact what("predict: identical e(sample)")
    foreach s in xb xbd d stdp score {
        capture confirm variable pf_r_`s' pf_x_`s'
        if (_rc) continue
        xcert_assert_var_close pf_r_`s' pf_x_`s' if pf_rsamp, tol(1e-8) ///
            name("H01 predict `s'")
    }
    xcert_assert_var_close pf_rres pf_xres if pf_rsamp, tol(1e-8) ///
        name("H01 residuals()")

    * test / lincom / testparm
    quietly reghdfe y x1 x2 i.cat [aw = w], absorb(firm year) ///
        vce(cluster state) tolerance(1e-12)
    quietly test x1 = x2
    scalar pfr_F = r(F)
    scalar pfr_Fdf = r(df)
    scalar pfr_Fdfr = r(df_r)
    quietly lincom x1 + x2
    scalar pfr_est = r(estimate)
    scalar pfr_se  = r(se)
    scalar pfr_ldf = r(df)
    quietly testparm i.cat
    scalar pfr_Fp = r(F)
    quietly xhdfe y x1 x2 i.cat [aw = w], absorb(firm year) ///
        vce(cluster state) tolerancemode(reghdfe-comparable) tolerance(1e-12)
    quietly test x1 = x2
    xpf_eq, id("H02") left(r(df))   right(pfr_Fdf)  exact what("test e(df)")
    xpf_eq, id("H02") left(r(df_r)) right(pfr_Fdfr) exact what("test e(df_r)")
    xpf_eq, id("H02") left(`= abs(r(F) / pfr_F - 1)') right(0) tol(1e-8) ///
        what("test F statistic")
    quietly lincom x1 + x2
    xpf_eq, id("H02") left(r(df)) right(pfr_ldf) exact what("lincom df")
    xpf_eq, id("H02") left(r(estimate)) right(pfr_est) tol(1e-9) ///
        what("lincom estimate")
    xpf_eq, id("H02") left(`= abs(r(se) / pfr_se - 1)') right(0) tol(1e-8) ///
        what("lincom standard error")
    quietly testparm i.cat
    xpf_eq, id("H02") left(`= abs(r(F) / pfr_Fp - 1)') right(0) tol(1e-8) ///
        what("testparm F statistic")

    * estat summarize
    quietly reghdfe y x1 x2 i.cat, absorb(firm year) tolerance(1e-12)
    capture quietly estat summarize
    xpf_rc, id("H03") got(`= _rc') want(0) what("reghdfe estat summarize")
    matrix pfr_S = r(stats)
    quietly xhdfe y x1 x2 i.cat, absorb(firm year) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    capture quietly estat summarize
    xpf_rc, id("H03") got(`= _rc') want(0) what("xhdfe estat summarize")
    matrix pfx_S = r(stats)
    xcert_assert_matrix_close pfr_S pfx_S, tol(1e-10) name("H03 estat summarize r(stats)")
restore

* ===========================================================================
* I. xfe versus reghdfe's partial-out
* ===========================================================================
di as text _n "-- I. xfe partial-out ----------------------------------------"

preserve
    capture drop pf_*
    * reghdfe's documented partial-out is `reghdfe var, absorb() residuals()`
    * applied to each variable in turn; `hdfe` is a separate package and is not
    * required by this suite.
    capture which hdfe
    if (_rc) di as text "             note: hdfe is not installed; using " ///
        "reghdfe residuals() as the documented equivalent"
    foreach v in y x1 x2 {
        quietly reghdfe `v', absorb(firm year) residuals(pf_r_`v') ///
            tolerance(1e-12) keepsingletons
        scalar pfr_N_`v'   = e(N)
        scalar pfr_dfa_`v' = e(df_a)
    }
    quietly xfe y x1 x2, absorb(firm year) generate(pf_x_) tolerance(1e-12) keepsingletons
    xpf_eq, id("I01") left(e(N))    right(pfr_N_y)   exact what("xfe e(N)")
    xpf_eq, id("I01") left(e(df_a)) right(pfr_dfa_y) exact what("xfe e(df_a)")
    foreach v in y x1 x2 {
        xcert_assert_var_close pf_r_`v' pf_x_`v', tol(1e-8) name("I01 partial-out `v'")
    }

    * weighted partial-out
    foreach v in y x1 x2 {
        quietly reghdfe `v' [aw = w], absorb(firm year) residuals(pf_rw_`v') ///
            tolerance(1e-12) keepsingletons
    }
    quietly xfe y x1 x2 [aw = w], absorb(firm year) generate(pf_xw_) ///
        tolerance(1e-12) keepsingletons
    foreach v in y x1 x2 {
        xcert_assert_var_close pf_rw_`v' pf_xw_`v', tol(1e-8) ///
            name("I02 weighted partial-out `v'")
    }

    * the partialled-out data reproduce reghdfe's coefficients exactly
    quietly reghdfe y x1 x2, absorb(firm year) tolerance(1e-12) keepsingletons
    scalar pfr_b1 = _b[x1]
    scalar pfr_b2 = _b[x2]
    quietly regress pf_x_y pf_x_x1 pf_x_x2, noconstant
    xpf_eq, id("I03") left(_b[pf_x_x1]) right(pfr_b1) tol(1e-8) ///
        what("second stage on xfe output reproduces b[x1]")
    xpf_eq, id("I03") left(_b[pf_x_x2]) right(pfr_b2) tol(1e-8) ///
        what("second stage on xfe output reproduces b[x2]")

    * sample construction with missing values must match reghdfe exactly
    capture drop pf_r_* pf_x_*
    quietly replace x1 = . in 1/17
    quietly reghdfe y, absorb(firm year) residuals(pf_r_y) tolerance(1e-12) keepsingletons
    scalar pfr_Ny = e(N)
    quietly xfe y x1, absorb(firm year) generate(pf_x_) tolerance(1e-12) keepsingletons
    xpf_eq, id("I04") left(e(N)) right(`= pfr_Ny - 17') exact ///
        what("xfe drops the rows with a missing regressor")
    quietly count if !missing(pf_x_y)
    xpf_eq, id("I04") left(r(N)) right(e(N)) exact ///
        what("xfe writes exactly e(N) non-missing values")
restore

* ===========================================================================
* J. FAIL-CLOSED NUMERICAL BOUNDARIES
* ===========================================================================
di as text _n "-- J. fail-closed numerical boundaries -----------------------"

preserve
    clear
    set obs 20
    generate double gx = 1e154 * (1 + _n / 20)
    generate double gy = _n / 20
    capture noisily xhdfe gy gx, noconstant ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    local overflow_rc = _rc
    if (`overflow_rc' == 0) {
        di as error "  FAIL [J01] finite-input Gram overflow was accepted"
        global PFEAT_FAILS = $PFEAT_FAILS + 1
    }
    else {
        di as text "  ok   [J01] finite-input Gram overflow fails closed (rc=`overflow_rc')"
    }
restore

* ===========================================================================
di as text _n "{hline 74}"
di as text "documented OPEN divergences reported this run: $PFEAT_OPEN"
if ($PFEAT_FAILS == 0) {
    di as result "PASS: reghdfe feature parity ($PFEAT_FAILS unexplained divergences)"
}
else {
    di as error "FAIL: reghdfe feature parity ($PFEAT_FAILS unexplained divergences)"
    exit 9
}

global XHDFE_TESTS_RUN = $XHDFE_TESTS_RUN + 1
