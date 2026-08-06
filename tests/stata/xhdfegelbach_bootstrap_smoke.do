version 14.0
clear all
set more off
set seed 20260725

local STATA_DIR : environment XHDFE_STATA_ADOPATH
if ("`STATA_DIR'" == "") local STATA_DIR "/home/mangelo/Documents/GitHub/xhdfe/stata"
adopath ++ "`STATA_DIR'"

set obs 240
generate long cluster = ceil(_n / 10)
generate long year = mod(_n - 1, 10) + 1
generate double x = rnormal()
generate double z1 = .35 * x + rnormal()
generate double z2 = -.20 * x + rnormal()
generate double aw = .4 + runiform() * 1.8
generate int fw = mod(_n, 3) + 1
sort cluster
by cluster: generate double cluster_effect = rnormal() if _n == 1
by cluster: replace cluster_effect = cluster_effect[1]
sort year
by year: generate double year_effect = rnormal() if _n == 1
by year: replace year_effect = year_effect[1]
generate double y = (.8 * x + .5 * z1 - .3 * z2 + cluster_effect + ///
    year_effect + rnormal() * .3)

set seed 99173
local rng_before "`c(rngstate)'"
xhdfegelbachbootstrap y, x1(x) ///
    x2groups("human = z1 : job = z2") commonfes(year) ///
    method(pairs) reps(19) minvalid(15) seed(123)
assert `"`c(rngstate)'"' == `"`rng_before'"'
assert r(reps_valid) == 19
assert r(reps_failed) == 0
assert "`r(method)'" == "pairs"
assert "`r(ci_method)'" == "percentile"
* Runtime version captured here; compared to the ado header after the
* remaining r() assertions, because findfile/file overwrite r().
local runver "`r(version)'"
assert r(gpu_requested) == 0
assert r(gpu_required) == 0
assert r(gpu_used_point) == 0
assert r(gpu_used_all_valid) == 0
matrix PA = r(bootstrap_delta_draws)
matrix PCA = r(bootstrap_delta_ci)
assert rowsof(PA) == 19
assert rowsof(PCA) == 4

* The literal version string went stale at every release (25jul, then 30jul);
* the invariant actually worth asserting is that the version the command
* reports at runtime equals its own ado header — a header that has moved ahead
* of the internal string is what broke the 2.22.0 staging.
quietly findfile xhdfegelbachbootstrap.ado
tempname fh
file open `fh' using "`r(fn)'", read text
file read `fh' adoline
file close `fh'
* Compare the number and date as tokens: header spacing is cosmetic and is not
* uniform across the shipped ados.
local hdrnum : word 3 of `adoline'
local hdrdate : word 4 of `adoline'
local runnum : word 1 of `runver'
local rundate : word 2 of `runver'
assert "`runnum'" != "" & "`runnum'" == "`hdrnum'" & "`rundate'" == "`hdrdate'"

* Independent first-replication oracle: same Stata RNG draw, full public refit.
preserve
set seed 123
quietly bsample
quietly xhdfegelbach y, x1(x) ///
    x2groups("human = z1 : job = z2") commonfes(year) vce(unadjusted)
matrix MANUALD = r(delta)
restore
quietly set rngstate `rng_before'
mata: st_numscalar("__xgel_boot_iid_oracle", max(abs( ///
    st_matrix("PA")[1, ] :- ///
    (st_matrix("MANUALD")[, 1]' , st_matrix("MANUALD")[, 2]'))))
assert scalar(__xgel_boot_iid_oracle) == 0
scalar drop __xgel_boot_iid_oracle

xhdfegelbachbootstrap y, x1(x) ///
    x2groups("human = z1 : job = z2") commonfes(year) ///
    method(pairs) reps(19) minvalid(15) seed(123)
matrix PB = r(bootstrap_delta_draws)
mata: st_numscalar("__xgel_boot_repro", ///
    max(abs(st_matrix("PA") :- st_matrix("PB"))))
assert scalar(__xgel_boot_repro) == 0
scalar drop __xgel_boot_repro

xhdfegelbachbootstrap y, x1(x) ///
    x2groups("human = z1 : job = z2") commonfes(year) ///
    method(cluster_pairs) bootcluster(cluster) ///
    reps(19) minvalid(15) seed(456) bootci(basic)
assert r(reps_valid) == 19
assert "`r(method)'" == "cluster_pairs"
assert "`r(resampling_unit)'" == "declared_cluster"
assert "`r(ci_method)'" == "basic"
matrix CB = r(bootstrap_delta_draws)
matrix CCI = r(bootstrap_delta_ci)
matrix CL = r(bootstrap_ledger)
matrix CT = r(total)
matrix CTD = r(bootstrap_total_draws)
matrix CTI = r(bootstrap_total_ci)
assert rowsof(CB) == 19
assert rowsof(CL) == 19
assert colsof(CL) == 7
assert CCI[4, 1] == 19
mata: __xgel_v = sort(st_matrix("CTD")[, 1], 1)
mata: __xgel_h = 1 + (rows(__xgel_v) - 1) * .975
mata: __xgel_q = __xgel_v[floor(__xgel_h)] + ///
    (__xgel_h - floor(__xgel_h)) * ///
    (__xgel_v[ceil(__xgel_h)] - __xgel_v[floor(__xgel_h)])
mata: st_numscalar("__xgel_boot_basic_low", ///
    2 * st_matrix("CT")[1, 1] - __xgel_q)
mata: __xgel_h = 1 + (rows(__xgel_v) - 1) * .025
mata: __xgel_q = __xgel_v[floor(__xgel_h)] + ///
    (__xgel_h - floor(__xgel_h)) * ///
    (__xgel_v[ceil(__xgel_h)] - __xgel_v[floor(__xgel_h)])
mata: st_numscalar("__xgel_boot_basic_high", ///
    2 * st_matrix("CT")[1, 1] - __xgel_q)
assert abs(CTI[1, 1] - scalar(__xgel_boot_basic_low)) < 1e-15
assert abs(CTI[2, 1] - scalar(__xgel_boot_basic_high)) < 1e-15
scalar drop __xgel_boot_basic_low __xgel_boot_basic_high

xhdfegelbachbootstrap y [aweight=aw], x1(x) ///
    x2groups("human = z1 : job = z2") commonfes(year) fes(cluster) ///
    method(pairs) reps(5) minvalid(4) seed(884)
assert r(reps_valid) >= 4
assert strpos("`r(groups)'", "cluster") > 0

tempfile gel_md gel_tex gel_html gel_csv
xhdfegelbachetable, format(markdown) panels(all) ///
    keep("human|job") labels("human = Human | capital : job = Job controls") ///
    caption("A | B") ///
    saving("`gel_md'") replace
confirm file "`gel_md'"
tempname md_handle
file open `md_handle' using "`gel_md'", read text
local markdown_text
file read `md_handle' md_line
while (r(eof) == 0) {
    local markdown_text `"`markdown_text'`md_line'"'
    file read `md_handle' md_line
}
file close `md_handle'
assert strpos(`"`markdown_text'"', "A \| B") > 0
assert strpos(`"`markdown_text'"', "Human \| capital") > 0

xhdfegelbachbootstrap y, x1(x) ///
    x2groups("human = z1 : job = z2") commonfes(year) ///
    method(cluster_pairs) bootcluster(cluster) ///
    reps(5) minvalid(4) seed(456)
xhdfegelbachetable, format(latex) panels(levels) ///
    labels("human = Human & capital") caption("A_table & B") ///
    saving("`gel_tex'") replace
confirm file "`gel_tex'"
tempname tex_handle
file open `tex_handle' using "`gel_tex'", read text
local latex_text
file read `tex_handle' tex_line
while (r(eof) == 0) {
    local latex_text `"`latex_text'`tex_line'"'
    file read `tex_handle' tex_line
}
file close `tex_handle'
assert strpos(`"`latex_text'"', "A\_table \& B") > 0
assert strpos(`"`latex_text'"', "Human \& capital") > 0
assert strpos(`"`latex_text'"', "\\\\") == 0

tempfile gel_normal
xhdfegelbachbootstrap y, x1(x) ///
    x2groups("human = z1 : job = z2") commonfes(year) ///
    method(pairs) reps(3) minvalid(3) seed(458)
xhdfegelbachetable, format(markdown) panels(share_base) ///
    interval(normal) sharetol(1e-12) sharetmin(3) ///
    saving("`gel_normal'") replace
tempname normal_handle
file open `normal_handle' using "`gel_normal'", read text
local normal_text
file read `normal_handle' normal_line
while (r(eof) == 0) {
    local normal_text `"`normal_text'`normal_line'"'
    file read `normal_handle' normal_line
}
file close `normal_handle'
assert strpos(`"`normal_text'"', "normal_delta") > 0
assert strpos(`"`normal_text'"', "bootstrap_") == 0

xhdfegelbachbootstrap y, x1(x) ///
    x2groups("human = z1 : job = z2") commonfes(year) ///
    method(pairs) reps(3) minvalid(3) seed(4567)
xhdfegelbachetable, format(html) panels(levels) ///
    labels("human = Human & capital") caption("<A>") ///
    saving("`gel_html'") replace
confirm file "`gel_html'"
tempname html_handle
file open `html_handle' using "`gel_html'", read text
local html_text
file read `html_handle' html_line
while (r(eof) == 0) {
    local html_text `"`html_text'`html_line'"'
    file read `html_handle' html_line
}
file close `html_handle'
assert strpos(`"`html_text'"', "&lt;A&gt;") > 0
assert strpos(`"`html_text'"', "Human &amp; capital") > 0

xhdfegelbachbootstrap y, x1(x) ///
    x2groups("human = z1 : job = z2") commonfes(year) ///
    method(pairs) reps(5) minvalid(4) seed(789)
xhdfegelbachcoefplot, focal(x) keep("human") ///
    labels("human = Human capital") name(xgel_boot_test) nodraw
graph describe xgel_boot_test

* Reporting commands must also consume an immediate point decomposition
* without inventing bootstrap intervals.
tempfile gel_point_md gel_point_csv
quietly xhdfegelbach y, x1(x) ///
    x2groups("human = z1 : job = z2") commonfes(year)
xhdfegelbachetable, format(markdown) panels(levels) ///
    saving("`gel_point_md'") replace
tempname point_handle
file open `point_handle' using "`gel_point_md'", read text
local point_text
file read `point_handle' point_line
while (r(eof) == 0) {
    local point_text `"`point_text'`point_line'"'
    file read `point_handle' point_line
}
file close `point_handle'
assert strpos(`"`point_text'"', "normal_delta") > 0
assert strpos(`"`point_text'"', "bootstrap_") == 0

quietly xhdfegelbach y, x1(x) ///
    x2groups("human = z1 : job = z2") commonfes(year)
xhdfegelbachetable, format(csv) panels(all) ///
    saving("`gel_point_csv'") replace
preserve
quietly import delimited using "`gel_point_csv'", clear varnames(1)
assert !missing(std_error) if ///
    inlist(panel, "share_base", "share_movement") & ///
    inlist(component, "human", "job", "total_movement")
assert strpos(confidence_method, "normal_delta") == 1 if ///
    inlist(panel, "share_base", "share_movement") & ///
    inlist(component, "human", "job")
restore

capture noisily xhdfegelbachbootstrap y, x1(x) ///
    x2groups("human = z1") method(cluster_pairs) reps(3) seed(1)
assert _rc == 198

capture noisily xhdfegelbachbootstrap y [fweight=fw], x1(x) ///
    x2groups("human = z1") method(pairs) reps(3) seed(1)
assert _rc == 198
capture noisily xhdfegelbachbootstrap y, x1(x) ///
    x2groups("human = z1") method(pairs) reps(3) seed(1) requiregpu
assert _rc == 198

quietly count
capture noisily xhdfegelbachetable
assert _rc == 301
quietly count
capture noisily xhdfegelbachcoefplot
assert _rc == 301

display as result "XHDFEGELBACH_BOOTSTRAP_SMOKE_PASS"
