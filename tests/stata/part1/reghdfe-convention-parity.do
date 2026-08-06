* ===========================================================================
* reghdfe is the canonical package for CONVENTIONS, not for numerical truth.
*
* Coefficients and standard errors are the output of two different iterative
* absorbers and can only ever agree to solver tolerance. Degrees of freedom,
* observation and cluster counts, which terms are dropped, and which
* statistics are reported as missing are CHOICES, with no rounding error, and
* those must agree EXACTLY. This layer asserts both, with the right unit for
* each.
*
* Divergence here is a finding, not necessarily a defect: reghdfe itself is an
* approximation (a documented under-converged case exists), so a continuous
* mismatch is adjudicated against the exact answer, never against reghdfe
* alone. The discrete assertions carry no such caveat.
* ===========================================================================

version 16

xcert_require_reghdfe

* xcert_parity_check / xcert_parity_spec live in _helpers.do so that every
* reghdfe parity layer can share them without editing another layer's file.

* ---------------------------------------------------------------------------
* Data: unbalanced panel with singletons, a collinear column, factor levels.
* ---------------------------------------------------------------------------
clear
set seed 20260804
set obs 6000
generate long obs = _n
generate int firm = 1 + mod(obs, 220)
generate int year = 1990 + mod(obs, 12)
generate int state = 1 + mod(obs, 17)
replace firm = 900 + obs if obs <= 25          // singleton firms
generate byte cat = 1 + mod(obs, 4)
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double x3 = 2 * x1 - 1.5 * x2         // exactly collinear with x1, x2
generate double w = 0.5 + abs(rnormal())
generate int fw = 1 + mod(obs, 3)
generate double y = 0.7 * x1 - 1.3 * x2 + 0.02 * firm - 0.05 * year ///
    + 0.3 * cat + rnormal()

global PARITY_FAILS 0
di as text _n "{hline 70}"
di as text "reghdfe convention parity"
di as text "{hline 70}"

* ---- absorption dimensions -------------------------------------------------
* xhdfe takes the absence of absorb() as the no-FE case; reghdfe requires the
* option to be named.
* e(df_a) is exempt here only: with no absorbed dimension reghdfe still counts
* the constant as one (df_a=1, its internal constant-as-absorbed bookkeeping)
* while xhdfe counts zero and carries the constant in df_m. e(df_r) — the
* number that drives inference — is asserted equal, so the totals agree and
* only the decomposition label differs.
xcert_parity_spec y x1 x2, name("noabsorb") ropts(noabsorb) except(df_a)
xcert_parity_tally

* With clustering the constant is part of CRV1's K even without absorb().
* This cell intentionally has no exception/pin: a variance or df_a mismatch
* is an inferential failure, not a harmless bookkeeping label.
xcert_parity_spec y x1 x2, name("noabsorb cluster") ///
    xopts(vce(cluster state)) ropts(noabsorb vce(cluster state)) skipf
xcert_parity_tally

xcert_parity_spec y x1 x2, name("1 FE") ///
    xopts(absorb(firm)) ropts(absorb(firm))
xcert_parity_tally

xcert_parity_spec y x1 x2, name("2 FE") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally

xcert_parity_spec y x1 x2, name("3 FE") ///
    xopts(absorb(firm year state)) ropts(absorb(firm year state))
xcert_parity_tally

* ---- variance estimators ---------------------------------------------------
xcert_parity_spec y x1 x2, name("robust") ///
    xopts(absorb(firm year) vce(robust)) ropts(absorb(firm year) vce(robust))
xcert_parity_tally

xcert_parity_spec y x1 x2, name("cluster 1-way") ///
    xopts(absorb(firm year) vce(cluster state)) ///
    ropts(absorb(firm year) vce(cluster state))
xcert_parity_tally

xcert_parity_spec y x1 x2, name("cluster 2-way") ///
    xopts(absorb(firm year) vce(cluster state year)) ///
    ropts(absorb(firm year) vce(cluster state year))
xcert_parity_tally

xcert_parity_spec y x1 x2, name("cluster nested in FE") ///
    xopts(absorb(firm year) vce(cluster firm)) ///
    ropts(absorb(firm year) vce(cluster firm))
xcert_parity_tally

* ---- weights ---------------------------------------------------------------
xcert_parity_spec y x1 x2 [aw = w], name("aweights") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally

xcert_parity_spec y x1 x2 [fw = fw], name("fweights") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally

* ---- structural conventions ------------------------------------------------
xcert_parity_spec y x1 x2 i.cat, name("factor variables") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally

* The omission pattern is the object under test: x3 = 2*x1 - 1.5*x2 exactly.
xcert_parity_spec y x1 x2 x3, name("collinear regressor") ///
    xopts(absorb(firm year)) ropts(absorb(firm year))
xcert_parity_tally

xcert_parity_spec y x1 x2, name("singletons kept") ///
    xopts(absorb(firm year) keepsingletons) ///
    ropts(absorb(firm year) keepsingletons)
xcert_parity_tally

xcert_parity_spec y x1 x2, name("noconstant") ///
    xopts(absorb(firm year) noconstant) ropts(absorb(firm year) noconstant)
xcert_parity_tally

* The 2.23.0 DoF rework removed the former one-residual-df variance gap in this
* cell. Keep both SE and F under the ordinary comparator contract.
xcert_parity_spec y x1 x2, name("dof(none)") ///
    xopts(absorb(firm year) dofadjustments(none) vce(cluster state)) ///
    ropts(absorb(firm year) dof(none) vce(cluster state))
xcert_parity_tally

* ---- the degenerate conventions this release changed -----------------------
* Both were wrong before 2.23.0 (SEs near zero with one cluster; R-squared of
* one on a constant outcome). They must now agree with reghdfe exactly.
generate byte onecluster = 0
xcert_parity_spec y x1 x2, name("single cluster (G=1)") ///
    xopts(absorb(firm year) vce(cluster onecluster)) ///
    ropts(absorb(firm year) vce(cluster onecluster)) skipf
xcert_parity_tally

preserve
    replace y = 3
    * e(df_m) records the model dimension and follows reghdfe. e(rank) records
    * the estimable rank of the coefficient covariance and follows regress;
    * on TSS=0 these are deliberately different objects.
    xcert_parity_spec y x1 x2, name("constant outcome (TSS=0)") ///
        xopts(absorb(firm year)) ropts(absorb(firm year)) skipf ///
        except(rank)
    xcert_parity_tally

    quietly regress y x1 x2 i.firm i.year
    scalar pc_rank = e(rank)
    quietly reghdfe y x1 x2, absorb(firm year)
    scalar pc_dfm = e(df_m)
    quietly xhdfe y x1 x2, absorb(firm year) ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12)
    if (e(df_m) != pc_dfm) {
        di as error "  FAIL [constant outcome] e(df_m) != reghdfe"
        global PARITY_FAILS = $PARITY_FAILS + 1
    }
    if (e(rank) != pc_rank) {
        di as error "  FAIL [constant outcome] e(rank) != regress"
        global PARITY_FAILS = $PARITY_FAILS + 1
    }
    di as text "XHDFE_POLICY_DIVERGENCE|TSS0|df_m follows model dimension; rank follows regress"
restore

di as text "{hline 70}"
if ($PARITY_FAILS == 0) {
    di as result "PASS: reghdfe convention parity ($PARITY_FAILS divergences)"
}
else {
    di as error "FAIL: reghdfe convention parity ($PARITY_FAILS divergences)"
    exit 9
}

global XHDFE_TESTS_RUN = $XHDFE_TESTS_RUN + 1
