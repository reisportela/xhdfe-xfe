version 16
clear all
set more off
set linesize 200
args root fixture
adopath ++ "`root'/stata"
use "`fixture'/long.dta", clear

xhdfe y x1 x2 x3, absorb(g1 g2 g3) numthreads(2)
assert e(converged) == 1
capture noisily xhdfe y x1 x2 x3, absorb(g1 g2 g3) maxiter(1) tolerance(1e-12) tolerancemode(strict-residual) numthreads(2)
local rc = _rc
assert `rc' != 0
assert "`e(cmd)'" == ""
capture matrix failed_b = e(b)
assert _rc != 0
capture matrix failed_V = e(V)
assert _rc != 0
capture noisily xhdfe
assert _rc == 301
di "NONCONVERGENCE_NO_ESTIMATES_PASS"

di "QUIET_BEGIN"
quietly xhdfe y x1 x2 x3, absorb(g1 g2 g3) numthreads(2)
di "QUIET_END"
assert e(converged) == 1
capture noisily xhdfe y x1 x2 x3, absorb(g1 g2 g3) unexpected_option
local rc = _rc
assert `rc' != 0
assert "`e(cmd)'" == ""
capture matrix failed_b = e(b)
assert _rc != 0
di "SYNTAX_FAILURE_NO_ESTIMATES_PASS"

quietly xhdfe y x1 x2 x3, absorb(g1 g2 g3) numthreads(2)
capture noisily xhdfe y x1 x2 x3, absorb(g1 g2 g3) cformat(not_a_format) numthreads(2)
local rc = _rc
assert `rc' != 0
assert "`e(cmd)'" == ""
capture matrix failed_b = e(b)
assert _rc != 0
di "DISPLAY_FAILURE_NO_ESTIMATES_PASS"
di "STATA_FAIL_CLOSED_PASS"
