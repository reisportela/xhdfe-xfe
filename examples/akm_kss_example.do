*! AKM + leave-out (KSS) worked example for xhdfe (Stata front-end)
*! Two-way worker-firm model, leave-out variance decomposition, component
*! standard errors and Andrews-Mikusheva confidence intervals.
*!
*! Numerical semantics follow Kline, Saggio & Solvsten (2020) as implemented
*! in Saggio's LeaveOutTwoWay (the canonical KSS reference). Run with:
*!   stata-mp -b do examples/akm_kss_example.do
*!
*! Requires xhdfe on the adopath, e.g.:
*!   adopath ++ "`c(pwd)'/stata"

clear all
set more off
set seed 20260706

*------------------------------------------------------------------------------
* 1. A reproducible synthetic worker-firm panel with mobility.
*    n_workers workers, each observed `reps' periods; workers change firm every
*    period so the panel is rich in movers (a well-connected AKM design).
*------------------------------------------------------------------------------
local n_workers 400
local n_firms   40
local reps      5

set obs `=`n_workers' * `reps''
gen long worker = ceil(_n / `reps')
bysort worker: gen byte t = _n
* draw a firm per (worker, period), forcing a move each period
gen long firm = .
gen double _alpha = .
gen double _psi   = .
* worker and firm latent effects
tempname weff feff
matrix `weff' = J(`n_workers', 1, .)
matrix `feff' = J(`n_firms', 1, .)
forvalues w = 1/`n_workers' {
    matrix `weff'[`w', 1] = rnormal(0, 0.6)
}
forvalues f = 1/`n_firms' {
    matrix `feff'[`f', 1] = rnormal(0, 0.4)
}
* assign firms with a move every period (deterministic given the seed)
sort worker t
by worker: gen long _f = .
gen double _u = runiform()
forvalues w = 1/`n_workers' {
    local cur = ceil(runiform() * `n_firms')
    forvalues k = 1/`reps' {
        local nxt = ceil(runiform() * (`n_firms' - 1))
        if (`nxt' >= `cur') local nxt = `nxt' + 1
        qui replace firm = `cur' if worker == `w' & t == `k'
        local cur = `nxt'
    }
}
forvalues w = 1/`n_workers' {
    qui replace _alpha = `weff'[`w', 1] if worker == `w'
}
forvalues f = 1/`n_firms' {
    qui replace _psi = `feff'[`f', 1] if firm == `f'
}
gen double x1 = rnormal()
gen double y  = _alpha + _psi + 0.3 * x1 + rnormal() * 0.5

*------------------------------------------------------------------------------
* 2. Sample preparation: the leave-one-out connected set (KSS-identified).
*------------------------------------------------------------------------------
xhdfeconnected worker firm, generate(in_leaveout)
di as txt "leave-out sample: " as res r(n_obs) as txt " obs, " ///
    as res r(n_workers) as txt " workers (" as res r(n_movers) ///
    as txt " movers), " as res r(n_firms) as txt " firms"

*------------------------------------------------------------------------------
* 3. AKM two-way estimation + variance decomposition (plug-in / AGSU / KSS).
*    Default: leave-a-match-out, exact leverages on this sample size.
*------------------------------------------------------------------------------
xhdfeakm y, worker(worker) firm(firm)

di as txt _n "KSS-corrected decomposition:"
di as txt "  var(psi)        = " as res %9.5f r(kss_var_psi)
di as txt "  var(alpha)      = " as res %9.5f r(kss_var_alpha)
di as txt "  cov(alpha,psi)  = " as res %9.5f r(kss_cov)
di as txt "  corr(alpha,psi) = " as res %9.5f r(kss_corr)
di as txt "  share of var(y): alpha " as res %5.3f r(kss_share_alpha) ///
    as txt ", psi " as res %5.3f r(kss_share_psi) ///
    as txt ", 2*cov " as res %6.3f r(kss_share_2cov)

*------------------------------------------------------------------------------
* 4. Component standard errors + Andrews-Mikusheva weak-id confidence intervals.
*------------------------------------------------------------------------------
xhdfeakm y, worker(worker) firm(firm) ci

di as txt _n "Inference on var(psi):"
di as txt "  KSS point est.  = " as res %9.5f r(theta_var_psi)
di as txt "  standard error  = " as res %9.5f r(se_var_psi)
di as txt "  AM 95% CI       = [" as res %7.5f r(ci_lb_psi) ///
    as txt ", " as res %7.5f r(ci_ub_psi) as txt "]"
di as txt "  F statistic     = " as res %7.3f r(fstat_psi) ///
    as txt "  (curvature " as res %5.3f r(curvature_psi) as txt ")"

*------------------------------------------------------------------------------
* 5. Controls partialled out (FWL), effects saved, JLA leverages (scales to
*    large data). Add gpu for the CUDA backend when the plugin has CUDA.
*------------------------------------------------------------------------------
xhdfeakm y, worker(worker) firm(firm) controls(x1) leverages(jla) draws(200) ///
    generate(akm) replace
di as txt _n "control coefficient on x1:"
matrix list r(b)

di as txt _n "done."
