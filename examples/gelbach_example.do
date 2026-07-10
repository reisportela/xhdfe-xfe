*! Gelbach (2016) conditional decomposition worked example for xhdfe (Stata).
*!
*! Decomposes the movement of a base coefficient (the education coefficient)
*! between a short and a long regression into additive, order-invariant
*! contributions from declared covariate and fixed-effect blocks. This is
*! specification accounting, not causal mediation. Semantics match b1x2.
*!
*! Run with:  stata-mp -b do examples/gelbach_example.do
*! Requires xhdfe on the adopath.

clear all
set more off
set seed 20260706

*------------------------------------------------------------------------------
* Education, job covariates and firm assignment share latent determinants.
* The example does not impose or validate a causal ordering among them.
*------------------------------------------------------------------------------
local n 5000
local n_firms 60
set obs `n'
gen double ability = rnormal()
gen double educ    = 0.5 * ability + rnormal()
gen long   firm_id = 1 + int((`n_firms' - 1) * normal(0.4 * ability + rnormal()))
gen double tenure  = rnormal() + 0.2 * ability
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
*   - an ability block
*   - a job-covariate block (tenure, exper)
*   - a firm fixed-effect block (firm_id, absorbed with the xhdfe backend)
*------------------------------------------------------------------------------
xhdfegelbach y, x1(educ) ///
    x2groups("ability = ability : job_covariates = tenure exper") ///
    fes(firm_id)

di as txt _n "Total movement in the educ coefficient (short - long):"
matrix list r(total), noheader format(%9.5f)
di as txt "Contribution of each declared block (delta):"
matrix list r(delta), noheader format(%9.5f)
di as txt "Standard errors:"
matrix list r(se), noheader format(%9.5f)
di as txt "summation identity residual = " as res %9.2e r(identity_gap)
di as txt "absorbed-FE aggregate (conditional/gamma0 SE):"
matrix list r(fe_total), noheader format(%9.5f)
di as txt "interpretation: coefficient-movement accounting; not causal mediation"

*------------------------------------------------------------------------------
* Cluster-robust inference by firm.
*------------------------------------------------------------------------------
xhdfegelbach y, x1(educ) ///
    x2groups("ability = ability : job_covariates = tenure exper") ///
    fes(firm_id) vce(cluster) cluster(firm_id)
di as txt _n "cluster-robust standard errors:"
matrix list r(se), noheader format(%9.5f)

di as txt _n "done."
