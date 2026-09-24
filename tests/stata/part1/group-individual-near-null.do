// Exact beta=0.7; ill-conditioned incidence must not yield false convergence.
clear
set obs 96
generate double wave = 1
replace wave = -1 in 2
forvalues j = 4/96 {
    replace wave = -wave[`j'-1]-wave[`j'-3] in `j'
}
generate double magnitude = abs(wave)
quietly summarize magnitude, meanonly
replace wave = wave/r(max)
generate int node = _n-1
expand 2
bysort node: generate byte component = _n-1
generate int pattern = node+96*component
replace wave = wave*(1-2*component)
expand 16
bysort pattern: generate byte rep = _n-1
generate long group_id = 16*pattern+rep
generate double within = .2*(2*mod(rep,2)-1)
generate double noise = .113*(2*mod(floor(rep/2),2)-1)
generate double x = wave+within
generate double y = .7*x+3*wave+noise
generate double good_y = 1.5+.7*within+noise
generate byte c = 1

quietly regress y x i.pattern
assert abs(_b[x]-.7)<1e-10
assert abs(e(rss)-3072*.113^2)<1e-9

generate byte members = 1+(node+1<96)+(node+3<96)
expand members
bysort group_id: generate byte position = _n-1
generate int individual = pattern+cond(position==0,0,cond(position==1,1,3))
isid group_id individual
local backend : env XHDFE_GPU_BACKEND

foreach mode in reghdfe-comparable strict-residual {
    quietly xhdfe good_y within, absorb(c individual) group(group_id) ///
        individual(individual) aggregation(sum) keepsingletons numthreads(2) ///
        tolerancemode(`mode') tolerance(1e-12)
    assert e(converged)==1 & abs(_b[within]-.7)<1e-8
    if "`backend'" == "cuda" assert e(gpu_used)==1 & "`e(gpu_status)'"=="used"

    capture noisily xhdfe y x, absorb(c individual) group(group_id) ///
        individual(individual) aggregation(sum) keepsingletons numthreads(2) ///
        tolerancemode(`mode') tolerance(1e-12)
    local rc = _rc
    assert `rc'==498 & "`e(cmd)'"==""
    capture matrix failed_b = e(b)
    assert _rc!=0
    capture matrix failed_V = e(V)
    assert _rc!=0
    capture noisily xhdfe
    assert _rc==301
}
display "GROUP_INDIVIDUAL_NEAR_NULL_FAIL_CLOSED_PASS"
