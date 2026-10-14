* Standalone first-stage diagnostics contract. Run with cpu (default) or cuda.
* Requires ivreghdfe and reghdfe; XHDFE_STATA_ADOPATH selects the candidate.
version 16
clear all
set more off
set linesize 255
args backend
if ("`backend'" == "") local backend cpu
assert inlist("`backend'", "cpu", "cuda")
local adopath : env XHDFE_STATA_ADOPATH
if (`"`adopath'"' != "") adopath ++ `"`adopath'"'
di "Stata version: " c(stata_version) "; OS: " c(os) "; MP: " c(MP)
which xhdfe
which ivreghdfe
which reghdfe
which ivreg2

program define fs_case
    syntax, ID(name) BACKEND(string) HANDLE(name) ENDOGenous(varlist) ///
        INSTRuments(varlist) [CONTROLS(varlist) ABSorb(string) ///
        VCE(string) WEIGHT(string asis) NOConstant NOIVREF]
    local xa
    local ra noabsorb
    local manualcmd regress
    local manuala
    if ("`absorb'" != "") {
        local xa "absorb(`absorb')"
        local ra "`xa'"
        local manualcmd reghdfe
        local manuala "`xa'"
    }
    local xv
    local rv
    if ("`vce'" != "") {
        local xv "vce(`vce')"
        if ("`vce'" == "robust") local rv robust
        else {
            gettoken kind cl : vce
            local rv "cluster(`cl')"
        }
    }
    tempname ref baseb baseV result used held rssu manual manualp rdf rpr2 factor
    tempvar sample
    if ("`noivref'" == "") {
        quietly ivreghdfe y `controls' (`endogenous' = `instruments') `weight', ///
            `ra' `rv' `noconstant' small ffirst
        matrix `ref' = e(first)
    }
    quietly xhdfe y `controls' `endogenous' `weight', `xa' `xv' `noconstant' ///
        endogenous(`endogenous') instruments(`instruments') gpubackend(`backend')
    matrix `baseb' = e(b)
    matrix `baseV' = e(V)
    generate byte `sample' = e(sample)
    capture confirm matrix e(first)
    assert _rc != 0
    local iv_cmdline `"`e(cmdline)'"'
    quietly estat firststage
    matrix `result' = r(first)
    matrix `used' = r(first_gpu_used)
    assert mreldif(`baseb', e(b)) == 0
    assert mreldif(`baseV', e(V)) == 0
    assert e(sample) == `sample'
    assert `"`e(cmdline)'"' == `"`iv_cmdline'"'
    quietly xhdfe y `controls' `endogenous' `weight', `xa' `xv' `noconstant' ///
        endogenous(`endogenous') instruments(`instruments') gpubackend(`backend') ffirst
    * Independent GPU fits may differ by roundoff. The postestimation checks
    * above still require exact preservation of the original IV estimates.
    di "REPEAT_DIFF `id' b=" %21.15e mreldif(`baseb', e(b)) ///
        " V=" %21.15e mreldif(`baseV', e(V)) ///
        " first=" %21.15e mreldif(`result', e(first))
    assert mreldif(`baseb', e(b)) < 1e-12
    assert mreldif(`baseV', e(V)) < 1e-12
    assert e(sample) == `sample'
    assert mreldif(`result', e(first)) < 1e-12
    matrix `held' = e(first)
    if ("`absorb'" != "") {
        assert e(gpu_used) == ("`backend'" == "cuda")
        if ("`backend'" == "cuda") {
            assert "`e(gpu_backend)'" == "cuda"
            assert "`e(gpu_status)'" == "used"
        }
        forvalues j=1/`=colsof(`used')' {
            assert `used'[1,`j'] == ("`backend'" == "cuda")
            assert `used'[2,`j'] == ("`backend'" == "cuda")
        }
    }
    quietly estat firststage
    assert mreldif(`held', r(first)) == 0
    local j 0
    local excluded : list instruments - controls
    foreach d of local endogenous {
        local ++j
        quietly `manualcmd' `d' `controls' `instruments' if `sample' `weight', ///
            `manuala' `xv' `noconstant'
        assert e(sample) == `sample'
        scalar `rssu' = e(rss)
        scalar `factor' = 1
        * reghdfe charges one extra intercept df when an FE is nested in a
        * cluster. ivreghdfe uses (N-K); reghdfe uses (N-K-1).
        if ("`vce'" != "" & !missing(e(df_a_nested)) & e(df_a_nested) > 0) {
            scalar `factor' = (e(N)-e(df_m)-e(df_a)-1) / (e(N)-e(df_m)-e(df_a))
        }
        quietly testparm `excluded'
        scalar `manual' = r(F)
        scalar `manualp' = r(p)
        scalar `rdf' = r(df_r)
        assert abs(`result'["F",`j']-`manual') < 1e-7*max(1,abs(`manual'))
        assert `result'["df",`j'] == r(df)
        assert `result'["df_r",`j'] == `rdf'
        assert abs(`result'["pvalue",`j']-`manualp') < 1e-9
        quietly `manualcmd' `d' `controls' if `sample' `weight', `manuala' `xv' `noconstant'
        scalar `rpr2' = 1-`rssu'/e(rss)
        assert abs(`result'["pr2",`j']-`rpr2') < 1e-9
        local ivf .
        if ("`noivref'" == "") {
            local ivf = `ref'["F","`d'"]
            assert abs(`result'["F",`j']-`ivf'*`factor') < 1e-7*max(1,abs(`ivf'))
            assert `result'["df",`j'] == `ref'["df","`d'"]
            assert `result'["df_r",`j'] == `ref'["df_r","`d'"]
            assert abs(`result'["pr2",`j']-`ref'["pr2","`d'"]) < 1e-9
        }
        post `handle' ("`id'") ("`backend'") ("`d'") ("`manualcmd'") ///
            (`result'["F",`j']) (`manual') (`ivf') (`factor') ///
            (`result'["pr2",`j']) (`result'["df",`j']) (`rdf')
    }
    di as result "PASS `id' `backend'"
end

set seed 71903
set obs 6000
generate long firm = 1 + mod(_n,120)
generate long year = 1 + mod(floor((_n-1)/120),10)
generate long state = 1 + mod(_n,29)
generate long county = 1 + mod(floor((_n-1)/29),37)
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double z1 = rnormal()
generate double z2 = rnormal()
generate double z3 = rnormal()
generate double v = rnormal()
generate double d1 = .6*z1 + .3*z2 + .2*x1 + firm/120 + v
generate double d2 = .5*z2 + .7*z3 - .2*x2 + year/20 + v + rnormal()
generate double weak = .01*z1 + .02*z2 + .2*x1 + firm/120 + v
generate double y = 1.5*d1 - .8*d2 + .5*x1 - .2*x2 + firm/30 + year/10 + v + rnormal()
generate double aw = .5 + runiform()
generate byte fw = 1 + mod(_n,3)
generate double zdup = 2*z1
generate double zzero = 0
tempname handle
postfile `handle' str24 case str4 backend str12 endogenous str8 reference_cmd ///
    double(F_xhdfe F_reference F_ivreghdfe nested_df_factor pr2 df df_r) ///
    using "iv_firststage_`backend'.dta", replace
foreach endog in d1 "d1 d2" {
    local count : word count `endog'
    foreach vce in unadjusted robust state firm {
        local vopt
        if ("`vce'" == "robust") local vopt "vce(robust)"
        else if ("`vce'" != "unadjusted") local vopt "vce(cluster `vce')"
        fs_case, id(e`count'_`vce') backend(`backend') handle(`handle') ///
            endogenous(`endog') instruments(z1 z2 z3) controls(x1 x2) ///
            absorb(firm year) `vopt'
    }
}
fs_case, id(aweight) backend(`backend') handle(`handle') endogenous(d1) ///
    instruments(z1 z2) controls(x1 x2) absorb(firm year) vce(robust) weight([aw=aw])
fs_case, id(fweight) backend(`backend') handle(`handle') endogenous(d1) ///
    instruments(z1 z2) controls(x1 x2) absorb(firm year) weight([fw=fw])
fs_case, id(pweight) backend(`backend') handle(`handle') endogenous(d1) ///
    instruments(z1 z2) controls(x1 x2) absorb(firm year) vce(cluster state) weight([pw=aw])
fs_case, id(multiway_cluster) backend(`backend') handle(`handle') endogenous(d1 d2) ///
    instruments(z1 z2 z3) controls(x1 x2) absorb(firm year) vce(cluster state county)
* With only ten year clusters, ivreghdfe cannot invert the full system's
* moment covariance. The per-equation tests are still checked against reghdfe.
quietly ivreghdfe y x1 x2 (d1 d2=z1 z2 z3), absorb(firm year) cluster(state year) ffirst
matrix unavailable = e(first)
assert missing(unavailable["F","d1"])
fs_case, id(multiway_fewclusters) backend(`backend') handle(`handle') endogenous(d1 d2) ///
    instruments(z1 z2 z3) controls(x1 x2) absorb(firm year) vce(cluster state year) noivref
fs_case, id(interaction_fe) backend(`backend') handle(`handle') endogenous(d1) ///
    instruments(z1 z2) controls(x1 x2) absorb(firm#year) vce(cluster state)
* CUDA absorption requires an FE dimension; ordinary IV is CPU-only.
if ("`backend'" == "cpu") {
    fs_case, id(noabsorb) backend(`backend') handle(`handle') endogenous(d1) ///
        instruments(z1 z2) controls(x1 x2) vce(robust)
    fs_case, id(noconstant) backend(`backend') handle(`handle') endogenous(d1) ///
        instruments(z1 z2) controls(x1 x2) noconstant
}
fs_case, id(nocontrols) backend(`backend') handle(`handle') endogenous(d1) ///
    instruments(z1) absorb(firm year) vce(cluster state)
fs_case, id(redundant_zero) backend(`backend') handle(`handle') endogenous(d1) ///
    instruments(z1 z2 zzero) controls(x1 x2) absorb(firm year) noivref
fs_case, id(repeated_control) backend(`backend') handle(`handle') endogenous(d1) ///
    instruments(z1 z2 x1) controls(x1 x2) absorb(firm year)
fs_case, id(zero_repeated_control) backend(`backend') handle(`handle') endogenous(d1) ///
    instruments(x2 z1 z2 zzero x1) controls(x1 x2) absorb(firm year) noivref
fs_case, id(redundant_duplicate) backend(`backend') handle(`handle') endogenous(d1) ///
    instruments(z1 z2 zdup zzero x1) controls(x1 x2) absorb(firm year) noivref
fs_case, id(weak) backend(`backend') handle(`handle') endogenous(weak) ///
    instruments(z1 z2) controls(x1 x2) absorb(firm year) vce(robust)
preserve
replace y = . in 1/13
replace z1 = . in 14/23
replace firm = 1000 in 24
fs_case, id(sample_singletons) backend(`backend') handle(`handle') endogenous(d1) ///
    instruments(z1 z2) controls(x1 x2) absorb(firm year) vce(cluster state)
restore
postclose `handle'

* Non-default inference settings are forwarded to the auxiliary regressions.
foreach settings in "statstyle(legacy)" "ssc(k.adj=0 g.adj=0 t.df=conventional)" {
    quietly xhdfe y x1 x2 d1, absorb(firm year) endogenous(d1) instruments(z1 z2) ///
        vce(cluster firm) gpubackend(`backend') ffirst `settings'
    matrix configured = e(first)
    quietly xhdfe d1 x1 x2 z1 z2, absorb(firm year) ///
        vce(cluster firm) gpubackend(`backend') `settings'
    quietly test z1 z2
    assert abs(configured["F",1]-r(F)) < 1e-9*max(1,abs(r(F)))
    assert configured["df_r",1] == r(df_r)
}

* nosample still supports diagnostics requested during estimation and replay.
quietly xhdfe y x1 x2 d1, absorb(firm year) endogenous(d1) instruments(z1 z2) ///
    gpubackend(`backend') nosample ffirst
matrix cached = e(first)
quietly estat firststage
assert mreldif(cached,r(first)) == 0
quietly count if e(sample)
assert r(N) == 0
quietly xhdfe y x1 x2 d1, absorb(firm year) endogenous(d1) instruments(z1 z2) ///
    gpubackend(`backend') nosample
capture noisily estat firststage
assert _rc == 498
assert "`e(cmd)'" == "xhdfe"

* Full display and replay, plus restoration after a failed on-demand request.
xhdfe y x1 x2 d1, absorb(firm year) endogenous(d1) instruments(z1 z2) ///
    gpubackend(`backend') first
matrix b_before = e(b)
matrix V_before = e(V)
xhdfe, ffirst
estat firststage, first
assert mreldif(b_before,e(b)) == 0
assert mreldif(V_before,e(V)) == 0
quietly xhdfe y x1 x2 d1, absorb(firm year) endogenous(d1) instruments(z1 z2) ///
    gpubackend(`backend') residuals(saved_iv_residual)
matrix b_before = e(b)
matrix V_before = e(V)
clonevar original_residual = saved_iv_residual
quietly estat firststage
assert saved_iv_residual == original_residual
assert "`e(resid)'" == "saved_iv_residual"
rename z1 hidden_instrument
capture noisily estat firststage
assert _rc == 111
assert mreldif(b_before,e(b)) == 0
assert mreldif(V_before,e(V)) == 0
rename hidden_instrument z1
quietly xhdfe y x1, absorb(firm year) gpubackend(`backend')
capture noisily estat firststage
assert _rc == 321
capture noisily xhdfe y x1, first
assert _rc == 198
assert "`e(cmd)'" == ""

use "iv_firststage_`backend'.dta", clear
export delimited using "iv_firststage_`backend'.csv", replace
di as result "IV_FIRSTSTAGE_CONTRACT_PASS `backend'"
