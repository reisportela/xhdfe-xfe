program define _up_ll_convention, eclass
    version 16
    args weightvar
    // Stata 19.5 adds the Gaussian log-weight determinant. reghdfe's
    // established likelihood omits this data-only constant. Preserve all
    // native estimates; translate only that explicitly documented convention.
    quietly summarize `weightvar' if e(sample), meanonly
    local weight_mean = r(mean)
    tempvar log_weight
    quietly gen double `log_weight' = ln(`weightvar'/`weight_mean') if e(sample)
    quietly summarize `log_weight' if e(sample), meanonly
    local determinant = 0.5*r(sum)
    noisily di "UPSTREAM_LL_DETERMINANT=" %21.15e `determinant'
    ereturn scalar ll = e(ll) - `determinant'
    ereturn scalar ll_0 = e(ll_0) - `determinant'
end
