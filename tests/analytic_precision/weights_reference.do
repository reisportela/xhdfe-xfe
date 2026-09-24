version 16
clear all
set more off
set linesize 200
args root fixture output
adopath ++ "`root'/stata"
use "`fixture'/long.dta", clear
gen long audit_cluster = mod(group*13 + floor(group/5), 23)
gen double audit_weight = weight

tempname results
postfile `results' str20 weight_case str12 vce str8 engine double N double df_r double rss double rmse double r2_a ///
    double b1 double b2 double b3 double se1 double se2 double se3 ///
    double p1 double p2 double p3 double lo1 double lo2 double lo3 double hi1 double hi2 double hi3 ///
    using "`output'/weights_results.dta"

foreach weight_case in iw_integer iw_fractional aw fw pw {
    quietly replace audit_weight = weight
    local wt iw
    if ("`weight_case'" == "iw_fractional") quietly replace audit_weight = 0.73 + mod(g1+2*g2,7)/13
    if ("`weight_case'" != "iw_integer" & "`weight_case'" != "iw_fractional") local wt `weight_case'
    foreach vce in unadjusted robust cluster {
        if ("`wt'" == "pw" & "`vce'" == "unadjusted") continue
        local vceopt "vce(`vce')"
        if ("`vce'" == "cluster") local vceopt "vce(cluster audit_cluster)"
        local refvce "`vceopt'"
        if ("`vce'" == "unadjusted") local refvce
        foreach engine in regress cpu cuda {
            if ("`engine'" == "regress") quietly regress y x1 x2 x3 i.g1 i.g2 [`wt'=audit_weight], `refvce'
            else quietly xhdfe y x1 x2 x3 [`wt'=audit_weight], absorb(g1 g2) `vceopt' numthreads(2) gpubackend(`engine') tolerance(1e-12) tolerancemode(strict-residual)
            local values
            forvalues j = 1/3 {
                local values "`values' (_b[x`j'])"
            }
            forvalues j = 1/3 {
                local values "`values' (_se[x`j'])"
            }
            forvalues j = 1/3 {
                local values "`values' (2*ttail(e(df_r),abs(_b[x`j']/_se[x`j'])))"
            }
            forvalues j = 1/3 {
                local values "`values' (_b[x`j']-invttail(e(df_r),0.025)*_se[x`j'])"
            }
            forvalues j = 1/3 {
                local values "`values' (_b[x`j']+invttail(e(df_r),0.025)*_se[x`j'])"
            }
            post `results' ("`weight_case'") ("`vce'") ("`engine'") (e(N)) (e(df_r)) (e(rss)) (e(rmse)) (e(r2_a)) `values'
            if ("`engine'" == "cuda") assert e(gpu_used) == 1 & "`e(gpu_backend)'" == "cuda"
        }
    }
}
postclose `results'
use "`output'/weights_results.dta", clear
export delimited using "`output'/weights_results.csv"
di "WEIGHTS_REFERENCE_RECORDED"
