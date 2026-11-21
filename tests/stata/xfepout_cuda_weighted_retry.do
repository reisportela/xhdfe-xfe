* Uses the synthetic cov240k.dta from tests/cuda_weighted_cpu_route_contract.py.
version 16
clear all
set more off
args package fixture output
adopath ++ "`package'"
quietly findfile xfepout.ado
assert "`r(fn)'" == "`package'/xfepout.ado"
quietly findfile xfepout.plugin
assert "`r(fn)'" == "`package'/xfepout.plugin"

foreach n in 2 3 {
    use "`fixture'", clear
    local effects fe1 fe2
    if `n' == 3 local effects fe1 fe2 fe3
    local columns y x1 x2 x3 x4
    local common absorb(`effects') numthreads(8)
    xfepout `columns' [aw=w_aw], `common' gpubackend(cuda) generate(d_)
    assert e(gpu_used) == 1 & e(gpu_status_code) == 1 & e(N) == _N
    assert "`e(gpu_backend)'" == "cuda" & "`e(gpu_status)'" == "used"
    assert e(iterations) <= 100000
    display "XFEP_OUT_CUDA_RETRY `n' it=" e(iterations) " method=" e(method_used)
    xfepout `columns' [aw=w_aw], `common' gpubackend(cpu) ///
        generate(m_) absorptionmethod(mlsmr)
    foreach col of local columns {
        tempvar sq0 sq1
        quietly generate double `sq0' = m_`col'^2
        quietly generate double `sq1' = (d_`col'-m_`col')^2
        quietly summarize `sq0' [aw=w_aw], meanonly
        local total = r(mean)
        quietly summarize `sq1' [aw=w_aw], meanonly
        assert sqrt(r(mean)/`total') <= 1e-8
    }

    capture noisily xfepout `columns' [aw=w_aw], `common' gpubackend(cuda) ///
        generate(short_) maxiter(1)
    assert _rc != 0
    capture confirm variable short_y
    assert _rc == 111
    keep d_*
    save "`output'/projected`n'.dta"
}
display "XFEP_OUT_WEIGHTED_CUDA_PASS"
