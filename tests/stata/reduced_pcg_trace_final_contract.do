* Point 13 (17nov2026): the final line of the reduced tier's trace in an xhdfe fit with saved
* effects (correction E1), through the plugin. Synthetic data only, made here from fixed seeds.
* tests/stata/reduced_pcg_trace_final_contract.sh runs every mode in its own fresh Stata process
* with XHDFE_REDUCED_PCG_TRACE=1 and XHDFE_AUTO_ROUTING_RETRY_DIAG=1 (stderr kept), for the new
* and the frozen package, and compares.
*
*   do reduced_pcg_trace_final_contract.do <mode> <leg>
*   modes: lowmob_savefe lowmob_plain declined_savefe declined_plain weighted_savefe
*          below_savefe below_plain e1off_savefe krylov_savefe krylov_plain sequence
*   (declined_*: the runner sets XHDFE_TEST_REDUCED_PCG_DECLINE=iterate, so that the tier declines
*   E1's projection and another route publishes it; the module-level contract uses the public
*   path-like fixture for the same path)
*
* Writes rpcgf_<leg>.txt: per fit, in order, one line "FIT <tag> rc status reason method iterations"
* (e(reduced_pcg_status), e(reduced_pcg_reason), e(absorption_method_used), e(iterations)) and the
* coefficients and the lower triangle of e(V) in hexadecimal (%21x). Ends with RPCGF_PASS <mode>.
version 16
clear all
set more off
set linesize 255
set processors 4
args mode leg
local adopath : env XHDFE_STATA_ADOPATH
while (strlen(`"`adopath'"') > 1 & substr(`"`adopath'"', -1, 1) == "/") {
    local adopath = substr(`"`adopath'"', 1, strlen(`"`adopath'"') - 1)
}
adopath ++ `"`adopath'"'
discard
quietly findfile xhdfe.plugin
assert `"`r(fn)'"' == `"`adopath'/xhdfe.plugin"'

* A worker-firm-year panel with few movers: the capturing sweep stops above the floor of the
* mode and E1 publishes the projection (the tier accepts it).
capture program drop lowmob_data
program define lowmob_data
    args weighted
    clear
    set seed 20261117
    set obs 20000
    generate long worker = _n
    generate long home = floor(2000 * runiform())
    generate double aw = rnormal()
    expand 8
    bysort worker: generate int year = _n
    generate long firm = cond(runiform() < .03, floor(2000 * runiform()), home)
    generate double x1 = rnormal() + .4 * aw + .4 * mod(firm, 13) / 13
    generate double x2 = rnormal()
    generate double x3 = rnormal()
    generate double y = .5 * x1 - .25 * x2 + aw + mod(firm, 29) / 29 + year / 8 + rnormal()
    generate double w = .5 + 1.5 * runiform()
end

* Ten rows per worker, firms at random: the capturing sweep stops at or below the floor, and E1
* publishes the sweep as before.
capture program drop below_data
program define below_data
    clear
    set seed 20261116
    set obs 6000
    generate long worker = _n
    generate double aw = rnormal()
    expand 10
    bysort worker: generate int year = _n
    generate long firm = floor(300 * runiform())
    generate double x1 = rnormal() + .3 * aw + .3 * mod(firm, 17) / 17
    generate double x2 = rnormal() + .3 * aw
    generate double x3 = rnormal()
    generate double y = rnormal() + aw + mod(firm, 23) / 23 + year / 10 + .5 * x1 - .25 * x2
end

capture program drop emit
program define emit
    args tag rc
    if `rc' != 0 {
        file write receipt "FIT `tag' `rc' . . . ." _n
        exit
    }
    file write receipt "FIT `tag' 0 `=e(reduced_pcg_status)' `e(reduced_pcg_reason)' " ///
        "`=e(absorption_method_used)' `=e(iterations)'" _n
    tempname B V
    matrix `B' = e(b)
    matrix `V' = e(V)
    forvalues j = 1/`=colsof(`B')' {
        file write receipt "b`j' " %21x (`B'[1,`j']) _n
        forvalues i = `j'/`=colsof(`B')' {
            file write receipt "V`i'_`j' " %21x (`V'[`i',`j']) _n
        }
    }
end

* One fit: data, saved effects or not, weights or not.
capture program drop one_fit
program define one_fit
    args tag data savefe weighted
    `data'_data
    local rhs "x1 x2 x3"
    local fes "worker firm year"
    local wt ""
    if "`weighted'" == "1" local wt "[aw=w]"
    local save ""
    if "`savefe'" == "1" local save ", savefe"
    capture noisily xhdfe y `rhs' `wt', absorb(`fes'`save') tolerancemode(reghdfe-comparable) ///
        tolerance(1e-8) numthreads(4) gpubackend(cpu) absorptioncachemode(off)
    emit `tag' `=_rc'
    capture drop __hdfe*
end

file open receipt using "rpcgf_`leg'.txt", write text replace
if "`mode'" == "lowmob_savefe" one_fit lowmob_savefe lowmob 1 0
if "`mode'" == "lowmob_plain" one_fit lowmob_plain lowmob 0 0
if "`mode'" == "declined_savefe" one_fit declined_savefe lowmob 1 0
if "`mode'" == "declined_plain" one_fit declined_plain lowmob 0 0
if "`mode'" == "weighted_savefe" one_fit weighted_savefe lowmob 1 1
if "`mode'" == "below_savefe" one_fit below_savefe below 1 0
if "`mode'" == "below_plain" one_fit below_plain below 0 0
if "`mode'" == "e1off_savefe" one_fit e1off_savefe lowmob 1 0
if "`mode'" == "krylov_savefe" one_fit krylov_savefe lowmob 1 0
if "`mode'" == "krylov_plain" one_fit krylov_plain lowmob 0 0
if "`mode'" == "sequence" {
    one_fit seq1_lowmob_savefe lowmob 1 0
    one_fit seq2_lowmob_plain lowmob 0 0
    one_fit seq3_below_savefe below 1 0
    one_fit seq4_weighted_savefe lowmob 1 1
    one_fit seq5_lowmob_savefe lowmob 1 0
    one_fit seq6_below_plain below 0 0
}
file close receipt
display "RPCGF_PASS `mode'"
