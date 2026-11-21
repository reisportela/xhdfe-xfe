* Uses the public cov240k.dta from tests/cuda_weighted_cpu_route_contract.py.
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
    local opts absorb(`effects') gpubackend(cpu) numthreads(8)
    local cache absorptioncache("`output'/aw`n'.bin")

    xfepout `columns' [aw=w_aw], `opts' generate(d_) `cache' abscachemode(write)
    assert e(method_used) == 6 & "`e(absorption_method)'" == "mlsmr" & e(gpu_used) == 0 & e(N) == _N
    local iterations = e(iterations)
    display "XFEP_OUT_RETRY `n' it=" e(iterations)

    xfepout `columns' [aw=w_aw], `opts' generate(c_) `cache' abscachemode(read)
    assert e(method_used) == 6 & e(iterations) == `iterations'

    capture noisily xfepout `columns' [aw=w_aw], `opts' generate(f_) ///
        `cache' abscachemode(read) absorptionmethod(symmetric-gauss-seidel)
    local status = _rc
    assert `status' == 198
    capture confirm variable f_y
    assert _rc == 111

    capture noisily xfepout `columns' [aw=w_aw], `opts' generate(b_) maxiter(1) ///
        groupvar(b_group) sample(b_sample)
    local status = _rc
    assert `status' != 0
    capture confirm variable b_y
    assert _rc == 111
    capture confirm variable b_group
    assert _rc == 111
    capture confirm variable b_sample
    assert _rc == 111

    generate byte blocked_x4 = 17
    capture noisily xfepout `columns' [aw=w_aw], `opts' generate(blocked_)
    assert _rc == 110
    capture confirm variable blocked_y
    assert _rc == 111
    assert blocked_x4 == 17

    xfepout `columns' [aw=w_aw], `opts' generate(m_) absorptionmethod(mlsmr) ///
        groupvar(m_group) sample(m_sample)
    assert e(method_used) == 6 & e(gpu_used) == 0 & e(N) == _N
    assert m_sample == 1 & !missing(m_group)
    foreach var of local columns {
        assert c_`var' == d_`var'
        assert m_`var' == d_`var'
    }
    keep d_*
    save "`output'/projected`n'.dta"
}
display "XFEP_OUT_WEIGHTED_CPU_PASS"
