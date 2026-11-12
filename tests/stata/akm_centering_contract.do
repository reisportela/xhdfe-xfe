version 18
args package backend
clear all
set more off
adopath ++ "`package'"
which xhdfeakm
local gpuopt ""
if ("`backend'" == "cuda") local gpuopt "gpu directmaxfirms(0)"

* Synthetic connected worker-firm graph, unequal matches and two stayers.
set obs 82
generate long worker = ceil(_n/4)
replace worker = 21 if _n == 81
replace worker = 22 if _n == 82
generate long firm = mod(_n-1,4)+1
generate long copies = 1 + mod(3*worker+2*firm,5)
replace copies = 5 if worker > 20
expand copies
set seed 18112026
generate double x = rnormal()
generate byte g = floor(4*runiform())+1
generate double y = .21*worker - .37*firm + .5*x + .17*g + .3*rnormal()
generate double shifted = y + 10
generate byte fw = 1 + mod(_n,3)

foreach level in match obs {
    foreach method in exact jla {
        quietly xhdfeakm y, worker(worker) firm(firm) leaveoutlevel(`level') ///
            leverages(`method') seed(18112026) threads(1) `gpuopt'
        assert r(converged) == 1
        assert "`r(sigma_centering)'" == "leaveouttwoway"
        matrix old = (r(kss_var_alpha), r(kss_var_psi), r(kss_cov))
        quietly xhdfeakm y, worker(worker) firm(firm) leaveoutlevel(`level') ///
            leverages(`method') seed(18112026) threads(1) `gpuopt' ///
            sigmacentering(leaveouttwoway)
        matrix explicit = (r(kss_var_alpha), r(kss_var_psi), r(kss_cov))
        assert mreldif(old, explicit) == 0
        quietly xhdfeakm y, worker(worker) firm(firm) leaveoutlevel(`level') ///
            leverages(`method') seed(18112026) threads(1) `gpuopt' ///
            sigmacentering(personyear)
        assert r(converged) == 1
        assert "`r(sigma_centering)'" == "person_year"
        if ("`backend'" == "cuda") assert r(gpu_used) == 1
        matrix inv = (r(kss_var_alpha), r(kss_var_psi), r(kss_cov))
        if ("`level'" == "obs") assert mreldif(old, inv) < 1e-10
        quietly xhdfeakm shifted, worker(worker) firm(firm) leaveoutlevel(`level') ///
            leverages(`method') seed(18112026) threads(1) `gpuopt' ///
            sigmacentering(personyear)
        matrix moved = (r(kss_var_alpha), r(kss_var_psi), r(kss_cov))
        assert mreldif(inv, moved) < 1e-10
    }
}

* The reference category of a nuisance factor must not change the alternative.
quietly xhdfeakm y, worker(worker) firm(firm) controls(x ib1.g) ///
    leverages(exact) threads(1) sigmacentering(personyear) `gpuopt'
matrix b1 = (r(kss_var_alpha), r(kss_var_psi), r(kss_cov))
assert r(fwl_precision_certified) == 1
quietly xhdfeakm y, worker(worker) firm(firm) controls(x ib3.g) ///
    leverages(exact) threads(1) sigmacentering(personyear) `gpuopt'
matrix b3 = (r(kss_var_alpha), r(kss_var_psi), r(kss_cov))
assert mreldif(b1, b3) < 1e-9

* Frequency weights retain literal-replication semantics.
quietly xhdfeakm y [fw=fw], worker(worker) firm(firm) leverages(exact) ///
    threads(1) sigmacentering(personyear) `gpuopt'
matrix weighted = (r(kss_var_alpha), r(kss_var_psi), r(kss_cov))
preserve
expand fw
quietly xhdfeakm y, worker(worker) firm(firm) leverages(exact) ///
    threads(1) sigmacentering(personyear) `gpuopt'
matrix expanded = (r(kss_var_alpha), r(kss_var_psi), r(kss_cov))
assert mreldif(weighted, expanded) < 1e-10
restore

* Unsupported lowess must fail, never return legacy inference.
foreach option in sigmalowess {
    capture xhdfeakm y, worker(worker) firm(firm) sigmacentering(personyear) `option'
    assert _rc == 198
    assert missing(r(kss_var_alpha))
}
capture xhdfeakm y, worker(worker) firm(firm) sigmacentering(invalid)
assert _rc == 198

* Explicit alternate calls must not change the next default call.
quietly xhdfeakm y, worker(worker) firm(firm) leverages(exact) threads(1)
assert "`r(sigma_centering)'" == "leaveouttwoway"
display "AKM_CENTERING_CONTRACT_PASS backend=`backend'"
