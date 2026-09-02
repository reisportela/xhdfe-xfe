version 16

local repo : environment XHDFE_REPO_ROOT
if (`"`repo'"' != "") adopath ++ `"`repo'/stata"'

* A connected two-way FE chain with duplicated end edges has exactly two
* within dimensions. With two slopes and id1 nested in the cluster, reghdfe's
* derived-statistics denominator is exactly zero although inferential df_r is
* positive. This is the small deterministic analogue of synthetic-zigzag.
clear
set seed 30830
set obs 102
generate int id1 = .
replace id1 = 1 in 1/3
replace id1 = 2 + floor((_n - 4) / 2) in 4/99
replace id1 = 50 in 100/102
generate int id2 = .
replace id2 = 1 in 1/2
replace id2 = 2 in 3
replace id2 = id1 + mod(_n - 4, 2) in 4/99
replace id2 = 50 in 100
replace id2 = 51 in 101/102
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double y = rnormal() + id1 / 7 - id2 / 11

quietly xhdfe y x1 x2, absorb(id1 id2) vce(cluster id1) keepsingletons ///
    statstyle(reghdfe) tolerancemode(reghdfe-comparable) tolerance(1e-12) ///
    noheader notable nofootnote
scalar x_used_df_r = e(df_r_unadj) - e(df_a_nested)
assert x_used_df_r == 0
assert e(df_r) > 0
assert abs(e(rmse)^2 - e(rss)) <= 1e-12 * max(1, e(rss))
assert missing(e(F))
assert missing(e(p))
assert missing(e(r2_a))
assert missing(e(r2_a_within))

quietly xhdfe y x1 x2, absorb(id1 id2) vce(cluster id1) keepsingletons ///
    statstyle(legacy) tolerancemode(reghdfe-comparable) tolerance(1e-12) ///
    noheader notable nofootnote
assert abs(e(rmse)^2 - e(rss) / e(df_r_unadj)) <= 1e-12 * max(1, e(rss))
assert !missing(e(r2_a))
assert !missing(e(r2_a_within))

quietly reghdfe y x1 x2, absorb(id1 id2) vce(cluster id1) keepsingletons
scalar r_used_df_r = e(N) - e(df_a) - e(df_m) - e(df_a_nested)
assert r_used_df_r == 0
assert e(df_r) > 0
assert abs(e(rmse)^2 - e(rss)) <= 1e-12 * max(1, e(rss))
assert missing(e(F))
assert missing(e(r2_a))
assert missing(e(r2_a_within))

* The existing group/individual saturated toy has negative inferential df_r.
* Its machine-precision fit retains the established adjusted-R2 display of 1.
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
    statstyle(reghdfe) noheader notable nofootnote
assert e(df_r) == -2
assert e(df_r_unadj) - e(df_a_nested) < 0
assert e(r2_a) == 1
assert e(r2_a_within) == 1
