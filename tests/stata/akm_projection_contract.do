version 18
args package backend
clear all
set more off
adopath ++ "`package'"
which xhdfeakm
local gpuopt ""
if ("`backend'" == "cuda") local gpuopt "gpu directmaxfirms(0)"

* Synthetic balanced two-firm panel; the SE/CI oracle is scalar algebra.
set obs 256
generate long worker = floor((_n-1)/4)
generate byte firm = floor(mod(_n-1,4)/2)
generate double fp = cond(firm==0,-.5,.5)
generate double s = 1 + mod(worker,3)
generate double v = s * cond(mod(_n-1,2)==0,1,-1)
generate double x = 5*fp + v
generate double y = .3*sin(.37*(worker+1)) + .7*fp + .35*x + .1*sin(.73*_n)
generate double vv = v^2
quietly summarize vv, meanonly
scalar total_vv = r(sum)
generate double fy = fp*y
quietly summarize fy, meanonly
scalar raw_difference = r(sum)/64
generate double vy = v*y
quietly summarize vy, meanonly
scalar gamma = r(sum)/total_vv
bysort worker: egen double local_fy = total(fy)
bysort worker: egen double local_vy = total(vy)
generate double local_vv = 4*s^2
generate double lv = local_vv/total_vv
scalar lf = 1/64
scalar hf = 1+lf+lf^2+lf^3+lf^4
generate double hv = 1+lv+lv^2+lv^3+lv^4
generate double score = -.5/64*(hf*(local_fy-raw_difference)+lf^5*local_fy) ///
    +2.5*local_vv/total_vv*(hv*(local_vy/local_vv-gamma)+lv^5*local_vy/local_vv)
generate double score2 = score^2
quietly summarize score2, meanonly
scalar variance = r(sum)/4
scalar centre = abs(raw_difference-5*gamma)/2
scalar radius = sqrt(-2*ln(.05)*variance)
scalar expected_se = 2*256/255*centre*sqrt(variance)
scalar expected_lb = 256/255*max(0,centre-radius)^2
scalar expected_ub = 256/255*(centre+radius)^2
quietly regress y x ibn.worker ib1.firm, noconstant
scalar reference_beta = _b[x]

foreach method in exact jla {
    quietly xhdfeakm y, worker(worker) firm(firm) controls(x) threads(2) ///
        sigmacentering(personyear) leverages(`method') `gpuopt'
    matrix point = (r(kss_var_alpha), r(kss_var_psi), r(kss_cov))
    quietly xhdfeakm y, worker(worker) firm(firm) controls(x) threads(2) ///
        sigmacentering(personyear) leverages(`method') se `gpuopt'
    assert r(converged) == 1
    assert "`r(inference_method)'" == "worker_projection"
    assert "`r(inference_cluster)'" == "worker"
    assert "`r(inference_backend)'" == "cpu"
    assert r(proj_dimension) == 2 & r(proj_clusters) == 64
    assert r(proj_threads_used) == 2 & r(proj_workers_active) == 2
    assert abs(r(se_var_psi)-expected_se) < 1e-9
    assert abs(r(ci_lb_psi)-expected_lb) < 1e-9
    assert abs(r(ci_ub_psi)-expected_ub) < 1e-9
    assert missing(r(se_var_alpha)) & missing(r(lambda1_psi))
    matrix with_se = (r(kss_var_alpha), r(kss_var_psi), r(kss_cov))
    assert mreldif(point, with_se) == 0
    matrix controls_beta = r(b)
    assert abs(controls_beta[1,1]-reference_beta) < 1e-10
    if ("`backend'" == "cuda") {
        assert r(gpu_used) == 1
        assert "`r(gpu_backend)'" == "cuda" & "`r(gpu_status)'" == "used"
        assert r(solver_iterations) > 0
    }
    replace y = y + 1e6
    quietly xhdfeakm y, worker(worker) firm(firm) controls(x) threads(2) ///
        sigmacentering(personyear) leverages(`method') ci `gpuopt'
    assert abs(r(se_var_psi)-expected_se) < 1e-8
    assert abs(r(ci_lb_psi)-expected_lb) < 1e-8
    assert abs(r(ci_ub_psi)-expected_ub) < 1e-8
    replace y = y - 1e6
}
display "AKM_PROJECTION_STATA_CONTRACT_PASS backend=`backend'"
