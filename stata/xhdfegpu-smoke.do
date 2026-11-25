* Validation in a fresh session before the installer changes either alias.
version 16.0
set more off
local stage : env XHDFEGPU_STAGE
local target : env XHDFEGPU_CPU
local arch : env XHDFEGPU_SM
local source_version : env XHDFEGPU_VERSION
local threads : env XHDFEGPU_THREADS
local with_xfepout : env XHDFEGPU_XFE
adopath ++ `"`stage'"'

program define _xhdfegpu_smoke_info
    args role target arch source_version
    local cmd "__`role'_plugin_dispatch"
    plugin call `cmd', "task=buildinfo"
    assert "`xhdfe_buildinfo_protocol'" == "1" & "`xhdfe_buildinfo_role'" == "`role'"
    assert "`xhdfe_buildinfo_cpu'" == "`target'" & "`xhdfe_buildinfo_sms'" == "`arch'"
    assert "`xhdfe_buildinfo_cuda'" == "1" & "`xhdfe_buildinfo_openmp'" == "1"
    assert "`xhdfe_buildinfo_version'" == "`source_version'"
end

* Within every eight rows, the regressor contrasts and error are orthogonal.
set obs 240000
generate long blockid = floor((_n - 1)/8)
generate long w = mod(blockid, 1500) + 1
generate int f = mod(17*w + floor(blockid/1500), 317) + 1
generate byte t = mod(3*w + floor(blockid/600), 12) + 1
generate byte phase = mod(_n - 1, 8)
generate double x1 = 2*mod(phase, 2) - 1 + mod(w, 11)/16 + mod(f, 3)/8
generate double x2 = 2*mod(floor(phase/2), 2) - 1 + t/32 - mod(f, 5)/16
generate double eps = (2*mod(floor(phase/4), 2) - 1)/8
generate double y = 1.5*x1 - .375*x2 + mod(w, 101)/32 + mod(f, 31)/16 - t/4 + eps

tempname b_cpu v_cpu v_discrepancy
foreach backend in cpu cuda {
    local gpu = ("`backend'" == "cuda")
    local reduced_cuda = cond(`gpu', "on", "off")
    quietly xhdfe y x1 x2, absorb(w f t) vce(robust) numthreads(`threads') ///
        gpubackend(`backend') tolerance(1e-8) tolerancemode(reghdfe-comparable) ///
        absorptioncachemode(off) reducedpcg(on) reducedpcgcuda(`reduced_cuda')
    assert "$XHDFE_PLUGIN_PATH_INTERNAL" == "`stage'/xhdfe.plugin"
    assert e(N) == 240000 & e(converged) == 1 & e(precision_certified) == 1
    assert e(threads_used) == `threads'
    assert e(gpu_used) == `gpu' & "`e(gpu_backend)'" == "`backend'"
    if (`gpu') assert "`e(gpu_status)'" == "used"
    assert abs(_b[x1] - 1.5) < 1e-8 & abs(_b[x2] + .375) < 1e-8
    if (!`gpu') {
        matrix `b_cpu' = e(b)
        matrix `v_cpu' = e(V)
        _xhdfegpu_smoke_info xhdfe `target' `arch' `source_version'
    }
    else {
        assert mreldif(e(b), `b_cpu') < 1e-9
        mata: vb = st_matrix("`v_cpu'"); st_numscalar("`v_discrepancy'", max(abs(st_matrix("e(V)")-vb):/sqrt(diagonal(vb)*diagonal(vb)')))
        assert scalar(`v_discrepancy') < 1e-7
    }
    display "xhdfe `backend': Stata fit, `threads' OpenMP worker(s)"

    if (`with_xfepout') {
        quietly xfepout y x1 x2, absorb(w f t) generate(po_) ///
            numthreads(`threads') gpubackend(`backend') tolerance(1e-8)
        assert "$XFEPOUT_PLUGIN_PATH_INTERNAL" == "`stage'/xfepout.plugin"
        assert e(N) == 240000 & e(converged) == 1 & e(threads_used) == `threads'
        assert e(gpu_used) == `gpu' & "`e(gpu_backend)'" == "`backend'"
        if (`gpu') assert "`e(gpu_status)'" == "used"
        if (!`gpu') _xhdfegpu_smoke_info xfepout `target' `arch' `source_version'
        assert abs(po_x1 - (2*mod(phase, 2) - 1)) < 1e-7
        assert abs(po_x2 - (2*mod(floor(phase/2), 2) - 1)) < 1e-7
        assert abs(po_y - (1.5*po_x1 - .375*po_x2 + eps)) < 1e-7
        quietly regress po_y po_x1 po_x2, noconstant
        assert abs(_b[po_x1] - 1.5) < 1e-8 & abs(_b[po_x2] + .375) < 1e-8
        drop po_y po_x1 po_x2
        display "xfepout `backend': Stata partial-out, `threads' OpenMP worker(s)"
    }
}
display "XHDFEGPU_VALIDATED"
