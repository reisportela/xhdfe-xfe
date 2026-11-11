* P3-fix (3'): the strict recovery sequence cases B3 and B4 through the Stata plugin (round 22: B1-B4
* also in Stata). Driven by Verifications/perf_fereg_gap_20261106/p3fix/p3fix_strict_sequence_stata.py,
* which writes the two fixtures (an easy dense three-FE design and the path-like two-FE design of the
* permanent strict contract), runs one fresh Stata session per case with the test hooks
* XHDFE_TEST_CUDA_CG_BREAKDOWN=25:all and XHDFE_TEST_CUDA_STRICT_LSMR_REJECT=1 and the trace
* variables, and grades the markers and the session trace.
* B3: the path-like fit is refused (the repair's CG breaks again after the LSMR was used); no e().
* B4: the easy fit succeeds, the path-like fit is refused leaving no e() estimates, the easy fit
*     succeeds again with the same coefficients.
* Arguments: package case(B3|B4) easy_dta pathlike_dta convergence(auto|reghdfe).
version 18
clear all
set more off
set linesize 255
args package case easy pathlike convergence
adopath ++ "`package'"
discard
findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
local common tolerance(1e-8) tolerancemode(strict-residual) absorptioncachemode(off) ///
    numthreads(16) gpubackend(cuda) convergence(`convergence')

if "`case'" == "B4" {
    use "`easy'", clear
    quietly xhdfe y x1 x2, absorb(f1 f2 f3) `common'
    assert e(converged) == 1
    assert e(precision_certified) == 1
    assert e(gpu_used) == 1
    assert "`e(gpu_backend)'" == "cuda"
    assert "`e(gpu_status)'" == "used"
    matrix b_first = e(b)
    display "P3S_SEQ_B4_A_OK iterations=" e(iterations)
}
use "`pathlike'", clear
capture noisily xhdfe y x, absorb(fa fb) `common'
local rc = _rc
local empty = ("`e(cmd)'" == "")
capture matrix stale_b = e(b)
local nob = (_rc != 0)
display "P3S_SEQ_`case'_REFUSED rc=`rc' ecmd_empty=`empty' no_eb=`nob'"
assert `rc' != 0
assert `empty' == 1
assert `nob' == 1
if "`case'" == "B4" {
    use "`easy'", clear
    quietly xhdfe y x1 x2, absorb(f1 f2 f3) `common'
    assert e(converged) == 1
    assert e(precision_certified) == 1
    assert e(gpu_used) == 1
    assert mreldif(e(b), b_first) == 0
    display "P3S_SEQ_B4_C_OK iterations=" e(iterations)
}
display "P3S_SEQ_`case'_PASS"
