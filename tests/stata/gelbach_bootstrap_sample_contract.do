version 16.0
clear all
set more off
set seed 86
local package : environment XHDFE_STATA_ADOPATH
adopath ++ "`package'"

set obs 480
generate long fe = cond(_n <= 400, ceil(_n/20), _n - 380)
generate double x = rnormal()
generate double z = rnormal()
generate double y = .7*x + .3*z + sin(fe) + rnormal()
replace y = y + 20*x + 100 if _n > 400

foreach method in pairs cluster_pairs {
    local cluster_option
    if "`method'" == "cluster_pairs" local cluster_option "bootcluster(fe)"
    quietly xhdfegelbachbootstrap y, x1(x) x2groups("z = z") fes(fe) ///
        method(`method') `cluster_option' reps(12) minvalid(10) seed(381) threads(2)
    matrix FULL = r(bootstrap_delta_draws)
    matrix BASE = r(b_base)
    preserve
    keep in 1/400
    quietly xhdfegelbachbootstrap y, x1(x) x2groups("z = z") fes(fe) ///
        method(`method') `cluster_option' reps(12) minvalid(10) seed(381) threads(2)
    matrix TRIM = r(bootstrap_delta_draws)
    mata: assert(mreldif(st_matrix("FULL"), st_matrix("TRIM")) < 1e-12)
    quietly regress y x
    assert abs(_b[x] - BASE[1,1]) < 1e-10
    restore
}
display "GELBACH_BOOTSTRAP_SAMPLE_CONTRACT PASS"
