version 14.0
clear all
set more off
set seed 20260725

adopath ++ "/home/mangelo/Documents/GitHub/xhdfe/stata"

set obs 1200
generate long row = _n - 1
generate long common_fe = mod(row, 40) + 1
generate long added_fe = mod(floor(row / 3) + 7 * common_fe, 31) + 1
generate long boot_cluster = mod(row, 60) + 1
generate double x = rnormal()
generate double z = .35 * x + rnormal()
sort common_fe
by common_fe: generate double common_effect = rnormal() if _n == 1
by common_fe: replace common_effect = common_effect[1]
sort added_fe
by added_fe: generate double added_effect = rnormal() if _n == 1
by added_fe: replace added_effect = added_effect[1]
generate double y = 1.1 * x + .7 * z + common_effect + ///
    added_effect + rnormal() * .3

xhdfegelbachbootstrap y, x1(x) x2groups("observed = z") ///
    commonfes(common_fe) fes(added_fe) method(pairs) ///
    reps(3) minvalid(3) seed(919) gpu requiregpu
assert r(gpu_requested) == 1
assert r(gpu_required) == 1
assert r(gpu_used_point) == 1
assert r(gpu_used_all_valid) == 1
assert "`r(point_gpu_backend)'" == "cuda"
assert "`r(point_gpu_status)'" == "used"
matrix IID_LEDGER = r(bootstrap_ledger)
assert rowsof(IID_LEDGER) == 3
forvalues b = 1/3 {
    assert IID_LEDGER[`b', 1] == 1
    assert IID_LEDGER[`b', 7] == 1
}

xhdfegelbachbootstrap y, x1(x) x2groups("observed = z") ///
    commonfes(common_fe) fes(added_fe) ///
    method(cluster_pairs) bootcluster(boot_cluster) ///
    reps(3) minvalid(3) seed(920) gpu requiregpu
assert r(gpu_requested) == 1
assert r(gpu_required) == 1
assert r(gpu_used_point) == 1
assert r(gpu_used_all_valid) == 1
assert "`r(point_gpu_backend)'" == "cuda"
assert "`r(point_gpu_status)'" == "used"
matrix CLUSTER_LEDGER = r(bootstrap_ledger)
assert rowsof(CLUSTER_LEDGER) == 3
forvalues b = 1/3 {
    assert CLUSTER_LEDGER[`b', 1] == 1
    assert CLUSTER_LEDGER[`b', 7] == 1
}

display as result "XHDFEGELBACH_BOOTSTRAP_CUDA_SMOKE_PASS"
