* CPU/CUDA parity smoke for Gelbach's opt-in absorbed-target estimand.
* Fails unless the CUDA row proves real device use.

version 16
clear all
set more off
set seed 20260719
set obs 500000

gen long worker = mod(_n - 1, 5000) + 1
gen long firm = mod(floor((_n - 1) / 5) + 7 * worker, 1000) + 1
gen long cluster = worker
gen byte focal = mod(worker, 2)
gen double experience = mod(_n - 1, 10) + rnormal(0, 0.2)
gen double job = 0.3 * focal + 0.2 * experience + rnormal()
gen double y = 0.25 * focal + 0.08 * experience + 0.5 * job + ///
    0.03 * worker - 0.02 * firm + rnormal()

adopath ++ "stata"

xhdfegelbach y, x1(focal experience) x2groups("job = job") ///
    fes(worker firm) absorbedtargets(focal) ///
    vce(cluster) cluster(cluster) threads(32)
assert r(converged) == 1
assert r(gpu_used) == 0
matrix bbase_cpu = r(b_base)
matrix bfull_cpu = r(b_full)
matrix delta_cpu = r(delta)
matrix cov_cpu = r(cov)
matrix total_cpu = r(total)
matrix total_cov_cpu = r(total_cov)
scalar gap_cpu = r(identity_gap)

discard
adopath ++ "stata"
xhdfegelbach y, x1(focal experience) x2groups("job = job") ///
    fes(worker firm) absorbedtargets(focal) ///
    vce(cluster) cluster(cluster) threads(32) gpu
assert r(converged) == 1
assert r(gpu_used) == 1
assert r(gpu_status_code) == 1
assert r(gpu_absorption_converged) == 1
assert "`r(gpu_backend)'" == "cuda"
assert "`r(gpu_status)'" == "used"
matrix bbase_gpu = r(b_base)
matrix bfull_gpu = r(b_full)
matrix delta_gpu = r(delta)
matrix cov_gpu = r(cov)
matrix total_gpu = r(total)
matrix total_cov_gpu = r(total_cov)
scalar gap_gpu = r(identity_gap)

mata:
void assert_close(string scalar lhs, string scalar rhs, real scalar tol)
{
    real matrix a, b
    a = st_matrix(lhs)
    b = st_matrix(rhs)
    if (rows(a) != rows(b) | cols(a) != cols(b) |
        max(abs(a :- b)) > tol) {
        _error(9)
    }
}

assert_close("bbase_cpu", "bbase_gpu", 1e-10)
assert_close("bfull_cpu", "bfull_gpu", 1e-10)
assert_close("delta_cpu", "delta_gpu", 1e-9)
assert_close("cov_cpu", "cov_gpu", 1e-9)
assert_close("total_cpu", "total_gpu", 1e-9)
assert_close("total_cov_cpu", "total_cov_gpu", 1e-9)
end

assert gap_cpu < 1e-8
assert gap_gpu < 1e-8
di as result "XHDFEGELBACH ABSORBED-TARGET CPU/CUDA PARITY PASSED"
