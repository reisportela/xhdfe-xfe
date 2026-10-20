version 16
clear all
set more off
args backend
if ("`backend'" == "") local backend cpu
local package : env XHDFE_STATA_ADOPATH
adopath ++ "`package'"
discard

sysuse auto, clear
generate long one_per_row = _n
quietly areg price weight length, absorb(one_per_row)
assert e(df_r) == 0 & e(r2_a) == . & e(F) == .
quietly xhdfe price weight length, absorb(one_per_row) keepsingletons gpubackend(`backend')
assert e(df_r) == 0 & e(r2_a) == . & e(r2_a_within) == . & e(F) == .
assert e(rss) == 0
matrix default_b = e(b)
matrix default_V = e(V)
if ("`backend'" == "cuda") assert e(gpu_used) == 1

quietly xhdfe price weight length, absorb(one_per_row) keepsingletons ///
    statstyle(legacy) gpubackend(`backend')
assert e(df_r) == 0 & e(r2_a) == 1
assert mreldif(default_b, e(b)) == 0 & mreldif(default_V, e(V)) == 0

quietly areg price weight length, absorb(turn)
scalar reference_adjusted = e(r2_a)
quietly xhdfe price weight length, absorb(turn) keepsingletons ///
    statstyle(reghdfe) gpubackend(`backend')
assert reldif(e(r2_a), reference_adjusted) < 1e-10
if ("`backend'" == "cuda") assert e(gpu_used) == 1

* Preserve the existing group/individual display with conservative negative DoF.
clear
input double(y x) byte(group individual)
-1 -1 1 1
-1 -1 1 2
 1  1 2 2
 1  1 2 3
end
generate byte constant_fe = 1
quietly xhdfe y x, absorb(constant_fe individual) group(group) individual(individual) ///
    aggregation(sum) keepsingletons absorptionmethod(gauss-seidel) ///
    statstyle(reghdfe) gpubackend(`backend')
display "SATURATED_GROUP backend=`backend' df_r=" e(df_r) " r2_a=" %24.17e e(r2_a) " r2_a_within=" %24.17e e(r2_a_within) " rss=" %24.17e e(rss)
assert e(df_r) == -1 & e(r2_a) == 1
if (e(tss_within) > 0 & !missing(e(tss_within))) assert e(r2_a_within) == 1
else assert e(r2_a_within) == .
if ("`backend'" == "cuda") assert e(gpu_used) == 1
display "SATURATED_STATISTICS_CONTRACT_PASS `backend'"
