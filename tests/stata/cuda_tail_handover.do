* F2-A: e(cuda_tail) on the synthetic three-FE fixture (240 000 rows, MP80 oracle).
* mode default: CPU and two-FE CUDA fits report none; the three-FE CUDA fit never
*   hands over (n < 750 000: the CPU selector does not apply) and meets the oracle.
* mode force (the driver sets XHDFE_CUDA_TAIL_HANDOVER=force): the three-FE CUDA fit
*   reports host-mlsmr with absorption method 6, real GPU use, and meets the oracle.
* Both modes: residuals() of each three-FE CUDA fit against those of the CPU fit
*   (relative L2), asserted <= 1e-8 when the completion ran on the host. The driver
*   sets XHDFE_MLSMR_TRACE=1, so the stderr log shows which host Krylov solver
*   recomputed the stored residuals (R6 observation, recorded by the driver).
* F2-B, mode restart (the driver passes the public pf_difficult_10m_3fe parquet as the
*   fixture and "-" as the oracle): CPU and CUDA Comparable fits at tolerance 1e-10.
*   On CUDA the sweeps hand over to the conjugate-gradient stage, which either converges
*   or stops; after a stop the fit restarts cold from the original data and the restart
*   is handed to the CPU MLSMR. The route is not deterministic, so (round 2) up to five
*   CUDA fits run, stopping at the first restart. Every fit: converged and certified, real
*   GPU use (gpu_used 1, gpu_backend cuda, gpu_status used), the same sample, b and V
*   within the CUDA floors of the CPU fit, residuals() against those of the CPU fit
*   (relative L2 over the estimation sample) computed and printed with its route. A
*   restart: also e(cuda_tail) host-mlsmr (2), method 6, and the residuals asserted
*   <= 1e-8 (a bound of this harness: the host route reproduces the CPU route). A
*   conjugate-gradient stage that converged (a device route): e(cuda_tail) none, method
*   1, and the residuals recorded without PASS/FAIL, labelled
*   DEVICE_ROUTE_UNGRADED_RESIDUALS (coordinator, 08nov2026; missing coverage, brief
*   P2/R4); the same label on the device tails of mode default. Five fits without a
*   restart end with the label NOT_EXERCISED_ROUTE (missing coverage, not a pass). The
*   driver sets XHDFE_GI_LSMR_TRACE=1 and records the cuda_tail_restart lines.
version 18
clear all
set more off
args package fixture oracle mode
adopath ++ "`package'"
discard
findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
assert inlist("`mode'", "default", "force", "restart")
if "`mode'" == "restart" {
    quietly pq use using "`fixture'", clear
    local xs x1 x2 x3 x4 x5 x6 x7 x8 x9 x10
    quietly xhdfe y `xs', absorb(indiv_id firm_id year) vce(cluster indiv_id) ///
        tolerance(1e-10) tolerancemode(reghdfe-comparable) absorptioncachemode(off) ///
        numthreads(16) gpubackend(cpu) residuals(r_cpu)
    assert e(gpu_used) == 0
    assert e(cuda_tail_code) == 0
    assert "`e(cuda_tail)'" == "none"
    assert e(converged) == 1
    assert e(precision_certified) == 1
    matrix cpu_b = e(b)
    matrix cpu_V = e(V)
    local cpu_N = e(N)
    generate byte cpu_sample = e(sample)
    display "CUDA_TAIL_RESTART_CPU method=" e(absorption_method_used) " iterations=" e(iterations)
    local k = colsof(cpu_b)
    local restarted = 0
    forvalues a = 1/5 {
        quietly xhdfe y `xs', absorb(indiv_id firm_id year) vce(cluster indiv_id) ///
            tolerance(1e-10) tolerancemode(reghdfe-comparable) absorptioncachemode(off) ///
            numthreads(16) gpubackend(cuda) residuals(r_cuda)
        assert e(N) == `cpu_N'
        assert e(converged) == 1
        assert e(precision_certified) == 1
        assert e(gpu_used) == 1
        assert "`e(gpu_backend)'" == "cuda"
        assert "`e(gpu_status)'" == "used"
        assert e(iterations) > 0 & e(iterations) <= 100000
        quietly count if e(sample) != cpu_sample
        assert r(N) == 0
        mata: st_numscalar("db", max(abs(st_matrix("e(b)")-st_matrix("cpu_b")) :/ rowmax((J(`k',1,1),abs(st_matrix("cpu_b")')))'))
        mata: st_numscalar("dv", max(abs(st_matrix("e(V)")-st_matrix("cpu_V")) :/ sqrt(diagonal(st_matrix("cpu_V"))*diagonal(st_matrix("cpu_V"))')))
        quietly count if cpu_sample & (missing(r_cuda) | missing(r_cpu))
        assert r(N) == 0
        mata: st_numscalar("dr", norm(st_data(., "r_cuda", "cpu_sample") - st_data(., "r_cpu", "cpu_sample")) / norm(st_data(., "r_cpu", "cpu_sample")))
        if e(cuda_tail_code) == 2 {
            local route restart
        }
        else if e(cuda_tail_code) == 0 & e(absorption_method_used) == 1 {
            local route cg_converged
        }
        else {
            local route other
        }
        display "CUDA_TAIL_RESTART attempt=`a' route=`route' tail=`e(cuda_tail)' method=" ///
            e(absorption_method_used) " iterations=" e(iterations) " gpu_status=`e(gpu_status)'" ///
            " b=" %21.14e db " V=" %21.14e dv " residuals_rel=" %21.14e dr
        assert db <= 1e-9
        assert dv <= 1e-8
        assert inlist("`route'", "restart", "cg_converged")
        drop r_cuda
        if "`route'" == "restart" {
            assert "`e(cuda_tail)'" == "host-mlsmr"
            assert e(absorption_method_used) == 6
            assert dr <= 1e-8
            local restarted = 1
            continue, break
        }
        assert "`e(cuda_tail)'" == "none"
        display "DEVICE_ROUTE_UNGRADED_RESIDUALS attempt=`a' route=`route' rel=" %21.14e dr
    }
    if `restarted' {
        display "CUDA_TAIL_HANDOVER_STATA_PASS restart"
    }
    else {
        display "CUDA_TAIL_RESTART_NOT_EXERCISED_ROUTE attempts=5"
        display "CUDA_TAIL_HANDOVER_STATA_NOT_EXERCISED restart"
    }
    exit
}
use "`fixture'", clear
do "`oracle'"
quietly xhdfe y x1 x2 x3 x4, absorb(worker firm year) vce(cluster worker) ///
    tolerance(1e-8) tolerancemode(reghdfe-comparable) absorptioncachemode(off) ///
    numthreads(16) gpubackend(cpu) residuals(r_cpu)
assert e(gpu_used) == 0
assert e(cuda_tail_code) == 0
assert "`e(cuda_tail)'" == "none"
display "CUDA_TAIL_CPU method=" e(absorption_method_used)
if "`mode'" == "default" {
    quietly xhdfe y x1 x2 x3 x4, absorb(worker firm) vce(cluster worker) ///
        tolerance(1e-8) tolerancemode(reghdfe-comparable) absorptioncachemode(off) ///
        numthreads(16) gpubackend(cuda)
    assert e(gpu_used) == 1
    assert e(cuda_tail_code) == 0
    assert "`e(cuda_tail)'" == "none"
}
foreach threads in 1 16 {
    quietly xhdfe y x1 x2 x3 x4, absorb(worker firm year) vce(cluster worker) ///
        tolerance(1e-8) tolerancemode(reghdfe-comparable) ///
        absorptionmethod(auto) absorptioncachemode(off) ///
        numthreads(`threads') gpubackend(cuda) residuals(r_cuda`threads')
    assert e(N) == 240000
    assert e(converged) == 1
    assert e(precision_certified) == 1
    assert e(gpu_used) == 1
    assert "`e(gpu_backend)'" == "cuda"
    assert "`e(gpu_status)'" == "used"
    if "`mode'" == "force" {
        assert e(cuda_tail_code) == 2
        assert "`e(cuda_tail)'" == "host-mlsmr"
        assert e(absorption_method_used) == 6
    }
    else {
        assert inlist(e(cuda_tail_code), 0, 1)
        assert inlist("`e(cuda_tail)'", "none", "device-lsmr")
    }
    mata: st_numscalar("db", max(abs(st_matrix("e(b)")-st_matrix("oracle_b")) :/ rowmax((J(5,1,1),abs(st_matrix("oracle_b")')))'))
    mata: st_numscalar("dv", max(abs(st_matrix("e(V)")-st_matrix("oracle_V")) :/ sqrt(diagonal(st_matrix("oracle_V"))*diagonal(st_matrix("oracle_V"))')))
    assert db <= 1e-9
    assert dv <= 1e-8
    quietly count if missing(r_cuda`threads') | missing(r_cpu)
    assert r(N) == 0
    mata: st_numscalar("dr", norm(st_data(., "r_cuda`threads'") - st_data(., "r_cpu")) / norm(st_data(., "r_cpu")))
    if "`mode'" == "force" {
        assert dr <= 1e-8
    }
    display "CUDA_TAIL_HANDOVER `mode' `threads' tail=`e(cuda_tail)' b=" %21.14e db " V=" %21.14e dv
    display "CUDA_TAIL_RESIDUALS `mode' `threads' tail=`e(cuda_tail)' method=" ///
        e(absorption_method_used) " rel=" %21.14e dr
    if "`e(cuda_tail)'" != "host-mlsmr" {
        display "DEVICE_ROUTE_UNGRADED_RESIDUALS `mode' `threads' tail=`e(cuda_tail)' rel=" %21.14e dr
    }
}
display "CUDA_TAIL_HANDOVER_STATA_PASS `mode'"
