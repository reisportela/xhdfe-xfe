* Multi-component categorical absorbs: grouped-ID and reghdfe references.
version 16
clear all
set more off
set linesize 160
args backend
if ("`backend'" == "") local backend cpu
local adopath : env XHDFE_STATA_ADOPATH
if (`"`adopath'"' != "") adopath ++ `"`adopath'"'
which xhdfe
which reghdfe

program define interaction_case
    syntax, TERM(string) GROUPED(varname) BACKEND(string) MODE(string) ///
        [WEIGHT(string asis) VCE(string) EXTRA(string) REFERENCE(string)]
    if ("`reference'" == "") local reference "`term'"
    local vopt
    if ("`vce'" != "") local vopt "vce(`vce')"
    local opts "`vopt' numthreads(2) gpubackend(`backend') tolerance(1e-12) tolerancemode(`mode')"
    tempname b V n dfa dfr dfm
    tempvar sample
    quietly xhdfe y x1 x2 `weight', absorb(`grouped' `extra') `opts'
    matrix `b' = e(b)
    matrix `V' = e(V)
    scalar `n' = e(N)
    scalar `dfa' = e(df_a)
    scalar `dfr' = e(df_r)
    scalar `dfm' = e(df_m)
    generate byte `sample' = e(sample)
    quietly xhdfe y x1 x2 `weight', absorb(`term' `extra') `opts'
    display "GROUPED_PARITY `term' b=" mreldif(`b',e(b)) " V=" mreldif(`V',e(V))
    assert mreldif(`b',e(b)) < 1e-10
    assert mreldif(`V',e(V)) < 1e-10
    assert e(N) == `n' & e(df_a) == `dfa' & e(df_r) == `dfr' & e(df_m) == `dfm'
    assert e(sample) == `sample'
    assert e(gpu_used) == ("`backend'" == "cuda")
    local refopts
    local vce_words : word count `vce'
    if (`vce_words' > 2) {
        * reghdfe PSD-corrects the full covariance, including the recovered
        * constant; xhdfe's documented D-C policy corrects the slope block.
        * Compare the shared no-constant contract for multiway inference.
        * The complete default b/V has already matched the grouped-ID fit.
        local refopts noconstant
        quietly xhdfe y x1 x2 `weight', absorb(`term' `extra') `opts' `refopts'
        matrix `b' = e(b)
        matrix `V' = e(V)
        assert e(N) == `n' & e(df_a) == `dfa' & e(df_r) == `dfr' & e(df_m) == `dfm'
    }
    quietly reghdfe y x1 x2 `weight', absorb(`reference' `extra') `vopt' `refopts' tolerance(1e-12)
    display "REFERENCE_DF `term' N=" e(N) "/" `n' " df_a=" e(df_a) "/" `dfa' ///
        " df_r=" e(df_r) "/" `dfr' " df_m=" e(df_m) "/" `dfm'
    assert e(sample) == `sample'
    assert e(N) == `n' & e(df_a) == `dfa' & e(df_r) == `dfr' & e(df_m) == `dfm'
    * The recovered-constant multiway covariance has its own documented policy.
    foreach v in x1 x2 {
        display "REFERENCE_PRECISION `term' `v' b=" abs(_b[`v']-`b'[1,"`v'"]) ///
            " se=" abs(_se[`v']-sqrt(`V'["`v'","`v'"]))
        assert abs(_b[`v']-`b'[1,"`v'"]) < 1e-9*max(1,abs(_b[`v']))
        assert abs(_se[`v']-sqrt(`V'["`v'","`v'"])) < 1e-7*max(1,abs(_se[`v']))
    }
    display "ABSORB_INTERACTION_CASE_PASS `backend' `mode' `term' `vce'"
end

set seed 16102026
set obs 6000
generate int a = 1 + mod(_n-1,7)
generate int b = 1 + mod(floor((_n-1)/7),5)
generate int c = 1 + mod(floor((_n-1)/35),4)
generate int d = 1 + mod(floor((_n-1)/140),3)
generate int time = 1 + mod(_n,11)
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double z1 = rnormal()
generate double z2 = rnormal()
generate double endog = .8*z1 + .3*z2 + .2*x1 + rnormal()
generate double y = .7*x1 - .3*x2 + .5*endog + a*b*c/90 + time/30 + rnormal()
generate double w = .5 + runiform()
replace a = 9999 in 1
replace b = . in 11/12
replace w = 0 in 24
generate str6 sa = string(a)
generate str6 sb = cond(missing(b), "", string(b))
generate str6 sc = string(c)
egen long abc = group(a b c)
egen long abcd = group(a b c d)
foreach mode in xhdfe-fast reghdfe-comparable {
    foreach vce in unadjusted robust "cluster a" "cluster a time" {
        local vo
        if ("`vce'" != "unadjusted") local vo "vce(`vce')"
        interaction_case, term(a#b#c) grouped(abc) backend(`backend') mode(`mode') `vo'
        interaction_case, term(a#b#c#d) grouped(abcd) backend(`backend') mode(`mode') ///
            weight([aw=w]) extra(time) `vo'
    }
    foreach term in "i.a#i.b#i.c" "a##b##c" "c#b#a" "a#a#b#c" "sa#sb#sc" {
        local refopt
        * reghdfe requires numeric factor components; these IDs encode the
        * same categories as sa/sb/sc, including their missing observations.
        if ("`term'" == "sa#sb#sc") local refopt "reference(a#b#c)"
        interaction_case, term(`term') grouped(abc) backend(`backend') mode(`mode') ///
            vce(cluster a) extra(time) `refopt'
    }
    * The IV auxiliaries must inherit the complete interaction, without aliases.
    quietly xhdfe y x1 x2 endog [aw=w], absorb(abc time) endogenous(endog) ///
        instruments(z1 z2) vce(cluster a) ffirst gpubackend(`backend') ///
        numthreads(2) tolerancemode(`mode') tolerance(1e-12)
    matrix iv_b = e(b)
    matrix iv_V = e(V)
    matrix iv_first = e(first)
    generate byte iv_sample = e(sample)
    quietly xhdfe y x1 x2 endog [aw=w], absorb(saved=a#b#c time) endogenous(endog) ///
        instruments(z1 z2) vce(cluster a) ffirst gpubackend(`backend') ///
        numthreads(2) tolerancemode(`mode') tolerance(1e-12)
    assert mreldif(iv_b,e(b)) < 1e-10 & mreldif(iv_V,e(V)) < 1e-10
    assert mreldif(iv_first,e(first)) < 1e-10
    assert e(sample) == iv_sample
    clonevar held_saved = saved
    quietly estat firststage, first
    assert mreldif(iv_first,r(first)) < 1e-10
    assert saved == held_saved
    assert mreldif(iv_b,e(b)) < 1e-10 & mreldif(iv_V,e(V)) < 1e-10
    * Independent auxiliary first-stage F reference, with matched FE/SSC.
    quietly reghdfe endog x1 x2 z1 z2 if iv_sample [aw=w], ///
        absorb(a#b#c time) vce(cluster a) tolerance(1e-12)
    quietly test z1 z2
    assert abs(iv_first["F",1]-r(F)) < 1e-7*max(1,abs(r(F)))
    assert iv_first["df",1] == r(df) & iv_first["df_r",1] == r(df_r)
    drop saved held_saved iv_sample
    display "ABSORB_INTERACTION_IV_PASS `backend' `mode'"
}
* This change does not reinterpret unsupported continuous-factor products.
capture noisily xhdfe y x1 x2, absorb(a#b#c.x1) numthreads(1)
assert _rc == 198
assert "`e(cmd)'" == ""
display "ABSORB_INTERACTION_CONTRACT_PASS `backend'"
exit
