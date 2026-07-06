*! Gelbach (2016) conditional decomposition worked example for xhdfe (Stata).
*!
*! Decomposes the movement of a base coefficient (the return to education)
*! between a short regression (y on educ) and a long regression (y on educ
*! plus mediating channels and a firm fixed effect) into an additive,
*! order-invariant contribution per channel. Semantics match Gelbach's b1x2.
*!
*! Run with:  stata-mp -b do examples/gelbach_example.do
*! Requires xhdfe on the adopath.

clear all
set more off
set seed 20260706

*------------------------------------------------------------------------------
* Synthetic data: educ is correlated with ability and with firm sorting, so the
* short-regression return to education is inflated by both channels.
*------------------------------------------------------------------------------
local n 5000
local n_firms 60
set obs `n'
gen double ability = rnormal()
gen double educ    = 0.5 * ability + rnormal()
gen long   firm_id = 1 + int((`n_firms' - 1) * normal(0.4 * educ + rnormal()))
gen double tenure  = rnormal() + 0.2 * educ
gen double exper   = rnormal()
* latent firm pay premium
tempname fe
matrix `fe' = J(`n_firms', 1, .)
forvalues f = 1/`n_firms' {
    matrix `fe'[`f', 1] = rnormal(0, 0.7)
}
gen double firmpay = .
forvalues f = 1/`n_firms' {
    qui replace firmpay = `fe'[`f', 1] if firm_id == `f'
}
gen double y = 0.5 * educ + 0.8 * ability + 0.1 * tenure + firmpay + rnormal()

*------------------------------------------------------------------------------
* Gelbach decomposition of the education coefficient into:
*   - a "skill" channel  (ability)
*   - a "job"   channel  (tenure, exper)
*   - a firm fixed-effect channel (firm_id, absorbed with the xhdfe backend)
*------------------------------------------------------------------------------
xhdfegelbach y, x1(educ) ///
    x2groups("skill = ability : job = tenure exper") ///
    fes(firm_id)

di as txt _n "Total movement in the educ coefficient (short - long):"
matrix list r(total), noheader format(%9.5f)
di as txt "Contribution of each channel (delta):"
matrix list r(delta), noheader format(%9.5f)
di as txt "Standard errors:"
matrix list r(se), noheader format(%9.5f)
di as txt "summation identity residual = " as res %9.2e r(identity_gap)

*------------------------------------------------------------------------------
* Cluster-robust inference by firm.
*------------------------------------------------------------------------------
xhdfegelbach y, x1(educ) ///
    x2groups("skill = ability : job = tenure exper") ///
    fes(firm_id) vce(cluster) cluster(firm_id)
di as txt _n "cluster-robust standard errors:"
matrix list r(se), noheader format(%9.5f)

di as txt _n "done."
