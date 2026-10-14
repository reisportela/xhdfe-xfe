* CPU/CUDA plugin guard against premature covariance rounding and disabled PSD.
* Generate the independent inputs with iv_covariance_psd_stata_fixtures.py.
version 16
clear all
set more off
set linesize 255
args backend fixtures
assert inlist("`backend'", "cpu", "cuda")
local adopath : env XHDFE_STATA_ADOPATH
if (`"`adopath'"' != "") adopath ++ `"`adopath'"'
which xhdfe
tempname handle refV norms maxV maxb expectedfix
postfile `handle' int caseid str24 mode str4 backend double maxV maxb psd_fixed gpu_used ///
    using iv_covariance_psd_results.dta, replace
foreach mode in xhdfe-fast reghdfe-comparable {
    forvalues case=1/44 {
        local id : display %03.0f `case'
        quietly import delimited using "`fixtures'/case_`id'.csv", varnames(1) asdouble clear
        local wt = weight_type[1]
        local gdf = g_df[1]
        local gadj = g_adj[1]
        local kadj = k_adj[1]
        scalar `expectedfix' = psd_fixed[1]
        matrix `refV' = (v11[1],v12[1],v13[1] \ v21[1],v22[1],v23[1] \ v31[1],v32[1],v33[1])
        matrix `norms' = (norm1[1],norm2[1],norm3[1])
        local weight
        if ("`wt'" == "analytic") local weight "[aw=weight]"
        if ("`wt'" == "frequency") local weight "[fw=weight]"
        quietly xhdfe y x1 x2 `weight', absorb(fe) endogenous(x1 x2) instruments(z1 z2) ///
            vce(cluster cl1 cl2) gpubackend(`backend') numthreads(2) tolerancemode(`mode') ///
            ssc(k.adj=`kadj' g.adj=`gadj' g.df=`gdf')
        assert e(vcv_psd_fixed) == `expectedfix'
        scalar `maxb' = max(abs(_b[x1]-.5),abs(_b[x2]+.25),abs(_b[_cons]-3))
        assert `maxb' <= 1e-9
        scalar `maxV' = 0
        forvalues j=1/3 {
            forvalues k=1/3 {
                scalar `maxV' = max(`maxV',abs(e(V)[`j',`k']-`refV'[`j',`k']) / ///
                    (`norms'[1,`j']*`norms'[1,`k']))
            }
        }
        assert `maxV' <= 1e-8
        if ("`backend'" == "cuda") {
            assert e(gpu_used) == 1
            assert "`e(gpu_backend)'" == "cuda"
            assert "`e(gpu_status)'" == "used"
        }
        else assert e(gpu_used) == 0
        post `handle' (`case') ("`mode'") ("`backend'") (`maxV') (`maxb') ///
            (e(vcv_psd_fixed)) (e(gpu_used))
        di as result "PASS covariance_psd `case' `mode' `backend' " %21.15e `maxV'
    }
}
postclose `handle'
use iv_covariance_psd_results.dta, clear
export delimited using iv_covariance_psd_results.csv, replace
assert _N == 88
di as result "IV_COVARIANCE_PSD_CONTRACT_PASS `backend'"
