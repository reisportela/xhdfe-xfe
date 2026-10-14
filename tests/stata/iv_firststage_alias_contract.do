* Named structural FEs must not become auxiliary-regression output targets.
version 16
clear all
set more off
set seed 14102026
args backend
if ("`backend'" == "") local backend cpu
local adopath : env XHDFE_STATA_ADOPATH
if (`"`adopath'"' != "") adopath ++ `"`adopath'"'
which xhdfe

foreach mode in xhdfe-fast reghdfe-comparable {
    foreach spec in level interaction slope lag {
        clear
        set obs 768
        generate int firm = floor((_n-1)/24)+1
        generate int time = mod(_n-1,24)+1
        generate byte region = mod(firm,4)
        generate byte period = ceil(time/6)
        generate double s = rnormal()
        generate double x = rnormal()
        generate double z = rnormal()
        generate double d = .8*z + .3*x + rnormal()
        generate double y = .7*d + .4*x + firm/30 + rnormal()
        generate double aw = 1 + mod(firm+time,3)/4
        xtset firm time
        local terms firm
        local named effect=firm
        if ("`spec'" == "interaction") {
            local terms "region#period"
            local named "effect=region#period"
        }
        if ("`spec'" == "slope") {
            local terms "firm#c.s"
            local named "effect=firm#c.s"
        }
        if ("`spec'" == "lag") {
            local terms "firm#c.L.s"
            local named "effect=firm#c.L.s"
        }
        local options "endogenous(d) instruments(z) numthreads(2) gpubackend(`backend') tolerancemode(`mode') vce(dkraay 3)"
        quietly xhdfe y d x [aw=aw], absorb(`terms') `options' ffirst
        matrix control_b = e(b)
        matrix control_V = e(V)
        matrix control_first = e(first)
        generate byte control_sample = e(sample)
        quietly xhdfe y d x [aw=aw], absorb(`named') `options'
        assert mreldif(control_b,e(b)) < 1e-8
        assert mreldif(control_V,e(V)) < 1e-8
        assert control_sample == e(sample)
        matrix saved_b = e(b)
        matrix saved_V = e(V)
        * Slope aliases use the documented Slope1 suffix.
        local output effect
        if inlist("`spec'", "slope", "lag") local output effectSlope1
        clonevar original_effect = `output'
        quietly estat firststage
        assert mreldif(control_first,r(first)) < 1e-8
        assert mreldif(saved_b,e(b)) == 0
        assert mreldif(saved_V,e(V)) == 0
        assert `output' == original_effect
        quietly xhdfe, first
        assert `output' == original_effect
        assert mreldif(saved_b,e(b)) == 0
        drop `output' original_effect
        quietly xhdfe y d x [aw=aw], absorb(`named') `options' ffirst nosample
        assert mreldif(control_first,e(first)) < 1e-8
        clonevar original_effect = `output'
        quietly estat firststage
        assert mreldif(control_first,r(first)) < 1e-8
        assert `output' == original_effect
        assert e(gpu_used) == ("`backend'" == "cuda")
        if ("`backend'" == "cuda") {
            assert "`e(gpu_backend)'" == "cuda"
            assert "`e(gpu_status)'" == "used"
        }
        display "FIRSTSTAGE_ALIAS_CASE_PASS `mode' `spec' `backend'"
    }
}
display "IV_FIRSTSTAGE_ALIAS_CONTRACT_PASS `backend'"
exit
