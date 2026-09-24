version 16.0
clear all
set more off
set seed 824
local package : environment XHDFE_STATA_ADOPATH
adopath ++ "`package'"
set obs 2000
generate long id = mod(_n,20)+1
generate long common = mod(floor(_n/20),7)+1
generate long cluster = mod(_n,13)+1
generate double near = 1+id*1e-8
generate double near_common = 2+common*1e-8
generate double near_cluster = 3+cluster*1e-8
generate double x = rnormal()+id/3+common/7
generate double z = rnormal()+id/7
generate double y = .7*x+.4*z+sin(id)+id/10+common+rnormal()

quietly xhdfegelbach y, x1(x) x2groups("z = z") fes(near) ///
    commonfes(near_common) vce(cluster) cluster(near_cluster) threads(2)
matrix FULL = r(b_full)
matrix BASE = r(b_base)
matrix DELTA = r(delta)
matrix COV = r(cov)
quietly xhdfegelbach y, x1(x) x2groups("z = z") fes(id) ///
    commonfes(common) vce(cluster) cluster(cluster) threads(2)
matrix FULL_ID = r(b_full)
matrix BASE_ID = r(b_base)
matrix DELTA_ID = r(delta)
matrix COV_ID = r(cov)
mata: assert(mreldif(st_matrix("FULL"), st_matrix("FULL_ID")) < 1e-12)
mata: assert(mreldif(st_matrix("BASE"), st_matrix("BASE_ID")) < 1e-12)
mata: assert(mreldif(st_matrix("DELTA"), st_matrix("DELTA_ID")) < 1e-12)
mata: assert(mreldif(st_matrix("COV"), st_matrix("COV_ID")) < 1e-12)
quietly regress y x z i.id i.common
assert abs(_b[x]-FULL[1,1]) < 1e-9
quietly regress y x i.common
assert abs(_b[x]-BASE[1,1]) < 1e-9
display "GELBACH_EXACT_LABELS_CONTRACT PASS"
