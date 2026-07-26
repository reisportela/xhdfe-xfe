clear all
set more off
set seed 20260725
adopath ++ "/home/mangelo/Documents/GitHub/xhdfe/stata"

* P0-1: weak base denominator is retained numerically but flagged.
set obs 800
generate double x = rnormal()
generate double z = rnormal()
generate double y0 = .7*z + rnormal()
quietly regress y0 x
predict double yres, residuals
generate double yweak = yres + 1e-11*x
quietly xhdfegelbach yweak, x1(x) x2groups(A = z) shares(base)
assert r(share_defined)[1, 1] == 1
assert r(share_denominator_t)[1, 1] < r(share_t_min)
local share_status "`r(share_interval_status)'"
local first_status : word 1 of `share_status'
assert "`first_status'" == "weak_denominator_delta_method_unreliable"
assert strpos("`r(share_se_type)'", ///
    "_weak_denominator_diagnostic_only") > 0

* P0-1: a strong denominator remains first-order valid.
replace yweak = .6*x + .8*z + rnormal()
quietly xhdfegelbach yweak, x1(x) x2groups(A = z) ///
    vce(robust) shares(base)
assert r(share_denominator_t)[1, 1] >= r(share_t_min)
local share_status "`r(share_interval_status)'"
local first_status : word 1 of `share_status'
assert "`first_status'" == "valid_first_order"

* P0-2: between-FE-dominant X1 variation triggers the conditional gate.
clear
set obs 6000
set seed 20260726
generate long firm = ceil(runiform()*80)
bysort firm: generate double alpha = rnormal() if _n == 1
bysort firm: replace alpha = alpha[1]
generate double x = sqrt(3)*alpha + rnormal()
generate double y = x + alpha + rnormal()
quietly xhdfegelbach y, x1(x) fes(firm)
assert r(x1_fe_collinear_ratio)[1, 1] <= r(fe_variance_ratio_min)
assert "`r(fe_variance_status)'" == ///
    "conditional_only_between_fe_dominant"
assert strpos("`r(fe_se_type)'", ///
    "_conditional_only_diagnostic") > 0
assert strpos("`r(total_se_type)'", ///
    "_conditional_only_diagnostic") > 0

* P0-2: a within-dominant design remains valid.
replace x = .5*alpha + rnormal()
replace y = x + alpha + rnormal()
quietly xhdfegelbach y, x1(x) fes(firm)
assert r(x1_fe_collinear_ratio)[1, 1] > r(fe_variance_ratio_min)
assert "`r(fe_variance_status)'" == "valid_first_order"
assert strpos("`r(fe_se_type)'", ///
    "_conditional_only_diagnostic") == 0
assert strpos("`r(total_se_type)'", ///
    "_conditional_only_diagnostic") == 0

* P0-3: filtered etables include a joint-covariance Other row by default.
clear
set obs 1200
set seed 20260727
generate double x = rnormal()
generate double a = .2*x + rnormal()
generate double b = .3*x + rnormal()
generate double c = .4*x + rnormal()
generate double d = .5*x + rnormal()
generate double y = x + .15*a + .30*b + .45*c + .60*d + rnormal()
quietly xhdfegelbach y, x1(x) ///
    x2groups(a = a : b = b : c = c : d = d) vce(robust)
matrix __p03_cov = r(cov)
matrix __p03_delta = r(delta)
matrix __p03_total = r(total)
matrix __p03_bbase = r(b_base)
scalar __p03_unfiltered_sum = __p03_delta[1, 1] + ///
    __p03_delta[1, 2] + __p03_delta[1, 3] + __p03_delta[1, 4]
assert abs(__p03_unfiltered_sum - __p03_total[1, 1]) <= 1e-12
assert abs(__p03_unfiltered_sum / __p03_bbase[1, 1] - ///
    __p03_total[1, 1] / __p03_bbase[1, 1]) <= 1e-12
assert abs(__p03_unfiltered_sum / __p03_total[1, 1] - 1) <= 1e-12
scalar __p03_other_var = 0
foreach i in 3 5 7 {
    foreach j in 3 5 7 {
        scalar __p03_other_var = __p03_other_var + __p03_cov[`i', `j']
    }
}

* The unfiltered table is unchanged and needs no residual row.
tempfile p03_all
xhdfegelbachetable, panels(all) format(csv) digits(12) ///
    saving("`p03_all'") replace
preserve
quietly import delimited using "`p03_all'", clear varnames(1) asdouble
count if component == "Other (filtered)"
assert r(N) == 0
foreach panel in levels share_base share_movement {
    quietly summarize estimate if panel == "`panel'" & ///
        inlist(component, "a", "b", "c", "d"), meanonly
    scalar __p03_sum = r(sum)
    quietly summarize estimate if panel == "`panel'" & ///
        component == "total_movement", meanonly
    * The pre-existing unfiltered rows remain byte-compatible; their
    * independently rounded CSV cells may accumulate a few last-place units.
    assert abs(__p03_sum - r(mean)) <= 5e-12
}
assert _N == 19
restore

quietly xhdfegelbach y, x1(x) ///
    x2groups(a = a : b = b : c = c : d = d) vce(robust)
tempfile p03
xhdfegelbachetable, panels(all) keep(a) exact format(csv) digits(12) ///
    saving("`p03'") replace
preserve
quietly import delimited using "`p03'", clear varnames(1) asdouble
foreach panel in levels share_base share_movement {
    quietly summarize estimate if panel == "`panel'" & ///
        inlist(component, "a", "Other (filtered)"), meanonly
    scalar __p03_sum = r(sum)
    quietly summarize estimate if panel == "`panel'" & ///
        component == "total_movement", meanonly
    assert abs(__p03_sum - r(mean)) <= 1e-12
}
quietly summarize std_error if panel == "levels" & ///
    component == "Other (filtered)", meanonly
assert abs(r(mean) - sqrt(__p03_other_var)) <= 5e-4
assert _N == 13
restore

* The same identity must hold with more than one displayed component.
quietly xhdfegelbach y, x1(x) ///
    x2groups(a = a : b = b : c = c : d = d) vce(robust)
tempfile p03_ab
xhdfegelbachetable, panels(all) keep(a b) exact format(csv) digits(12) ///
    saving("`p03_ab'") replace
preserve
quietly import delimited using "`p03_ab'", clear varnames(1) asdouble
foreach panel in levels share_base share_movement {
    quietly summarize estimate if panel == "`panel'" & ///
        inlist(component, "a", "b", "Other (filtered)"), meanonly
    scalar __p03_sum = r(sum)
    quietly summarize estimate if panel == "`panel'" & ///
        component == "total_movement", meanonly
    assert abs(__p03_sum - r(mean)) <= 1e-12
}
assert _N == 16
restore

* Explicit noother reproduces the former filtered shape.
quietly xhdfegelbach y, x1(x) ///
    x2groups(a = a : b = b : c = c : d = d) vce(robust)
tempfile p03_noother
xhdfegelbachetable, panels(share_movement) keep(a) exact noother ///
    format(csv) digits(12) saving("`p03_noother'") replace
preserve
quietly import delimited using "`p03_noother'", clear varnames(1) asdouble
count if component == "Other (filtered)"
assert r(N) == 0
restore

* P0-4a: two labels remain two independent labels in every panel.
quietly xhdfegelbach y, x1(x) ///
    x2groups(a = a : b = b : c = c : d = d) vce(robust)
tempfile p04_labels
xhdfegelbachetable, panels(all) keep(a b) exact noother ///
    labels("a = Human capital : b = Job controls") ///
    format(csv) digits(12) saving("`p04_labels'") replace
preserve
quietly import delimited using "`p04_labels'", clear varnames(1) asdouble
count if component == "Human capital"
assert r(N) == 3
count if component == "Job controls"
assert r(N) == 3
count if component == "Human capital : b = Job controls"
assert r(N) == 0
restore

* The same two-entry label grammar must be accepted by the waterfall.
quietly xhdfegelbach y, x1(x) ///
    x2groups(a = a : b = b : c = c : d = d) vce(robust)
xhdfegelbachcoefplot, focal(x) keep(a b) exact noother ///
    labels("a = Human capital : b = Job controls") ///
    name(xgel_remediation_labels) nodraw
graph describe xgel_remediation_labels

* P0-4b: component-level rows consume the stored bootstrap intervals.
xhdfegelbachbootstrap y, x1(x) x2groups(a = a : b = b) ///
    method(pairs) reps(5) minvalid(4) seed(20260728)
matrix __p04_bdci = r(bootstrap_delta_ci)
local __p04_bootci "`r(ci_method)'"
tempfile p04_boot
xhdfegelbachetable, panels(levels) format(csv) digits(12) ///
    saving("`p04_boot'") replace
preserve
quietly import delimited using "`p04_boot'", clear varnames(1) asdouble
count if inlist(component, "a", "b") & ///
    confidence_method == "bootstrap_`__p04_bootci'"
assert r(N) == 2
quietly summarize std_error if component == "a", meanonly
assert abs(r(mean) - __p04_bdci[3, 1]) <= 1e-12
quietly summarize conf_low if component == "a", meanonly
assert abs(r(mean) - __p04_bdci[1, 1]) <= 1e-12
quietly summarize conf_high if component == "a", meanonly
assert abs(r(mean) - __p04_bdci[2, 1]) <= 1e-12
restore

* P0-4c: store volatile r() provenance before another rclass command.
quietly xhdfegelbach y, x1(x) x2groups(a = a) ///
    sampleaudit generate(xgel_remediation_sample)
local __p04_hash "`r(sample_hash)'"
local __p04_hash_algorithm "`r(sample_hash_algorithm)'"
quietly tab xgel_remediation_sample, missing
assert strlen("`__p04_hash'") == 16
assert "`__p04_hash_algorithm'" == "fnv1a64-le-v1"

display as result "XHDFEGELBACH_REMEDIATION_P0_OK"
