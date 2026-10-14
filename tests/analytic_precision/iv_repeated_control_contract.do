* Exact overlap between included controls and instruments must not change IV.
* Standalone CPU/CUDA guard; run in a scratch directory, never on user data.
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

program define overlap_case
    syntax, ID(name) BACKEND(string) HANDLE(name) ENDOGenous(varlist) ///
        INSTRuments(varlist) [ABSorb(string) VCE(string) ///
        WEIGHT(string asis) NOConstant SELFInstrument]
    local xa
    local ra noabsorb
    local dummies
    if ("`absorb'" != "") {
        local xa "absorb(`absorb')"
        local ra "`xa'"
        foreach fe of local absorb {
            local dummies "`dummies' i.`fe'"
        }
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
    tempname baseb baseV refb dummyb nb dfr dfa dmodel rss maxb maxV
    tempvar sample
    quietly xhdfe y x1 x2 `endogenous' `weight', `xa' `xv' `noconstant' ///
        endogenous(`endogenous') instruments(`instruments') gpubackend(`backend')
    matrix `baseb' = e(b)
    matrix `baseV' = e(V)
    generate byte `sample' = e(sample)
    scalar `nb' = e(N)
    scalar `dfr' = e(df_r)
    scalar `dfa' = e(df_a)
    scalar `dmodel' = e(df_m)
    scalar `rss' = e(rss)
    if ("`backend'" == "cuda") {
        assert e(gpu_used) == 1
        assert "`e(gpu_backend)'" == "cuda"
        assert "`e(gpu_status)'" == "used"
    }

    if ("`selfinstrument'" == "") {
        quietly ivreghdfe y x1 x2 (`endogenous' = `instruments') `weight', ///
            `ra' `rv' `noconstant' small
    }
    else {
        * ivreghdfe rejects endogenous variables in the excluded list. In this
        * special case all structural regressors instrument themselves: OLS.
        quietly reghdfe y x1 x2 `endogenous' `weight', `ra' `xv' `noconstant'
    }
    assert e(sample) == `sample'
    matrix `refb' = e(b)
    * Explicit FE dummies provide an independent, non-HDFE coefficient check.
    if ("`selfinstrument'" == "") {
        quietly ivregress 2sls y x1 x2 `dummies' (`endogenous' = `instruments') ///
            if `sample' `weight', `xv' `noconstant'
    }
    else {
        quietly regress y x1 x2 `endogenous' `dummies' if `sample' `weight', ///
            `xv' `noconstant'
    }
    assert e(sample) == `sample'
    matrix `dummyb' = e(b)
    foreach v in x1 x2 `endogenous' {
        assert abs(`baseb'[1,"`v'"]-`refb'[1,"`v'"]) < 1e-9*max(1,abs(`refb'[1,"`v'"]))
        assert abs(`baseb'[1,"`v'"]-`dummyb'[1,"`v'"]) < 1e-9*max(1,abs(`dummyb'[1,"`v'"]))
    }

    * Repeated controls in different positions, with multiple controls and a
    * wildcard. Remaining excluded-instrument order stays the same.
    local variant 0
    foreach inst in "`instruments' x1" "x2 `instruments' x1" "x* `instruments'" {
        local ++variant
        forvalues rep=1/5 {
            quietly xhdfe y x1 x2 `endogenous' `weight', `xa' `xv' `noconstant' ///
                endogenous(`endogenous') instruments(`inst') gpubackend(`backend')
            di "OVERLAP_DIFF `id' `variant' `rep' b=" %21.15e mreldif(`baseb',e(b)) ///
                " V=" %21.15e mreldif(`baseV',e(V))
            assert e(sample) == `sample'
            assert e(N) == `nb'
            assert e(df_r) == `dfr'
            assert e(df_a) == `dfa'
            assert e(df_m) == `dmodel'
            assert abs(e(rss)-`rss') < 1e-10*max(1,abs(`rss'))
            assert "`e(exogenous)'" == "x1 x2"
            assert "`e(instruments)'" == "`instruments'"
            assert strpos(`"`e(cmdline)'"', "instruments(`inst')") > 0
            assert mreldif(`baseb',e(b)) < 1e-10
            assert mreldif(`baseV',e(V)) < 1e-10
            scalar `maxb' = 0
            scalar `maxV' = 0
            forvalues j=1/`=colsof(`baseb')' {
                scalar `maxb' = max(`maxb',abs(`baseb'[1,`j']-e(b)[1,`j']))
                forvalues k=1/`=colsof(`baseb')' {
                    scalar `maxV' = max(`maxV',abs(`baseV'[`j',`k']-e(V)[`j',`k']))
                }
            }
            if ("`backend'" == "cuda") {
                assert e(gpu_used) == 1
                assert "`e(gpu_backend)'" == "cuda"
                assert "`e(gpu_status)'" == "used"
            }
            post `handle' ("`id'") ("`backend'") (`variant') (`rep') ///
                (`maxb') (`maxV') (_b[`=word("`endogenous'",1)']) (e(N)) (e(gpu_used))
        }
    }
    di as result "PASS repeated_control `id' `backend'"
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
generate double zzero = 0
tempname handle
postfile `handle' str24 case str4 backend int(variant repetition) ///
    double(max_abs_b_diff max_abs_V_diff b_endogenous N gpu_used) ///
    using "iv_repeated_control_`backend'.dta", replace
foreach vce in unadjusted robust state firm {
    local vopt
    if ("`vce'" == "robust") local vopt "vce(robust)"
    else if ("`vce'" != "unadjusted") local vopt "vce(cluster `vce')"
    overlap_case, id(`vce') backend(`backend') handle(`handle') ///
        endogenous(d1) instruments(z1 z2) absorb(firm year) `vopt'
}
overlap_case, id(aweight) backend(`backend') handle(`handle') ///
    endogenous(d1) instruments(z1 z2) absorb(firm year) vce(robust) weight([aw=aw])
overlap_case, id(fweight) backend(`backend') handle(`handle') ///
    endogenous(d1) instruments(z1 z2) absorb(firm year) weight([fw=fw])
overlap_case, id(pweight) backend(`backend') handle(`handle') ///
    endogenous(d1) instruments(z1 z2) absorb(firm year) vce(cluster state) weight([pw=aw])
overlap_case, id(two_endogenous) backend(`backend') handle(`handle') ///
    endogenous(d1 d2) instruments(z1 z2 z3) absorb(firm year) vce(robust)
overlap_case, id(zero_column) backend(`backend') handle(`handle') ///
    endogenous(d1) instruments(z1 z2 zzero) absorb(firm year)
overlap_case, id(weak) backend(`backend') handle(`handle') ///
    endogenous(weak) instruments(z1 z2) absorb(firm year) vce(robust)
* Only exogenous overlaps are removed. A declared endogenous variable in the
* instrument list retains its existing self-instrumentation semantics.
overlap_case, id(self_instrument) backend(`backend') handle(`handle') ///
    endogenous(d1) instruments(d1 z1 z2) absorb(firm year) selfinstrument
preserve
replace x1 = . in 1/13
replace z1 = . in 14/23
replace firm = 1000 in 24
overlap_case, id(sample_singletons) backend(`backend') handle(`handle') ///
    endogenous(d1) instruments(z1 z2) absorb(firm year) vce(cluster state)
restore
if ("`backend'" == "cpu") {
    overlap_case, id(noabsorb) backend(`backend') handle(`handle') ///
        endogenous(d1) instruments(z1 z2) vce(robust)
    overlap_case, id(noconstant) backend(`backend') handle(`handle') ///
        endogenous(d1) instruments(z1 z2) noconstant
}
postclose `handle'

* Filtering all excluded instruments must never silently change IV into OLS.
foreach inst in "x1" "x1 x2" {
    capture noisily xhdfe y x1 x2 d1, absorb(firm year) endogenous(d1) ///
        instruments(`inst') gpubackend(`backend')
    assert _rc == 198
    assert "`e(cmd)'" == ""
    capture confirm matrix e(b)
    assert _rc != 0
}
* Too few distinct excluded instruments still fail after overlap removal.
capture noisily xhdfe y x1 x2 d1 d2, absorb(firm year) endogenous(d1 d2) ///
    instruments(z1 x1 x2) gpubackend(`backend')
assert _rc == 198
assert "`e(cmd)'" == ""
capture confirm matrix e(b)
assert _rc != 0

use "iv_repeated_control_`backend'.dta", clear
format max_abs_b_diff max_abs_V_diff b_endogenous %24.17e
export delimited using "iv_repeated_control_`backend'.csv", replace
di "IV_REPEATED_CONTROL_CONTRACT_PASS `backend'"
