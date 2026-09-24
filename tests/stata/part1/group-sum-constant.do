noi di as text "xhdfe certification: aggregation(sum) without an ordinary FE reports the constant only when spanned"

* Individuals with true effects; groups of 1-4 distinct members (uneven sizes:
* the ones vector is not in the span of the sum-aggregated membership columns).
clear
set seed 20260922
set obs 60
gen long ind = _n
gen double alpha = rnormal()
tempfile inds
save `inds'

clear
set obs 400
gen long g = _n
gen int size = 1 + floor(4 * runiform())
gen double x1 = rnormal()
gen double x2 = rnormal()
gen double eps = 0.1 * rnormal()
expand size
gen long ind = 1 + floor(60 * runiform())
bysort g ind: keep if _n == 1
merge m:1 ind using `inds', keep(match) nogen
bysort g: egen double fe = total(alpha)
gen double y = 1.5 + 0.7 * x1 - 0.4 * x2 + fe + eps
tempfile long
save `long'

* Reference: group-level OLS on the incidence columns without a constant.
tab ind, gen(d_)
collapse (sum) d_* (first) x1 x2 y, by(g)
regress y x1 x2 d_*, noconstant
scalar ref_b1 = _b[x1]
scalar ref_b2 = _b[x2]
scalar ref_rss = e(rss)
scalar ref_tss = e(mss) + e(rss)
scalar ref_r2 = e(r2)

use `long', clear
xhdfe y x1 x2, absorb(ind) group(g) individual(ind) aggregation(sum) ///
    tolerancemode(reghdfe-comparable) residuals(res) noheader notable nofootnote
assert e(report_constant) == 0
local names : colnames e(b)
assert strpos(" `names' ", " _cons ") == 0
assert reldif(_b[x1], scalar(ref_b1)) < 1e-8
assert reldif(_b[x2], scalar(ref_b2)) < 1e-8
assert reldif(e(rss), scalar(ref_rss)) < 1e-8
assert reldif(e(tss), scalar(ref_tss)) < 1e-8
assert reldif(e(r2), scalar(ref_r2)) < 1e-8
assert e(N) == 400
* r2_a follows the noconstant convention (no constant counted in the TSS df).
assert reldif(e(r2_a), 1 - (e(rss) / e(df_r)) / (e(tss) / e(N))) < 1e-10
* predict d is the summed individual effect: y - xb - e with xb free of _cons.
predict double dd, d
gen double dd_ref = y - _b[x1] * x1 - _b[x2] * x2 - res
assert reldif(dd, dd_ref) < 1e-10

* Uniform team size: the constant is spanned and _cons is recovered.
clear
set seed 20260923
set obs 300
gen long g = _n
gen double x1 = rnormal()
gen double x2 = rnormal()
gen double eps = 0.1 * rnormal()
expand 3
bysort g: gen int pos = _n
gen long ind = 1 + mod(floor(60 * runiform()) + pos * 20, 60)
bysort g ind: keep if _n == 1
merge m:1 ind using `inds', keep(match) nogen
bysort g: egen double fe = total(alpha)
bysort g: egen int n_members = count(ind)
keep if n_members == 3
gen double y = 1.5 + 0.7 * x1 - 0.4 * x2 + fe + eps
xhdfe y x1 x2, absorb(ind) group(g) individual(ind) aggregation(sum) ///
    tolerancemode(reghdfe-comparable) residuals(res) noheader notable nofootnote
assert e(report_constant) == 1
local names : colnames e(b)
assert strpos(" `names' ", " _cons ") > 0
collapse (first) res, by(g)
summarize res, meanonly
assert abs(r(mean)) < 1e-8

exit
