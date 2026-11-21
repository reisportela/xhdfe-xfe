* Point 14 (17nov2026): the method xhdfe and xfepout declare when the Krylov-PCG route is forced,
* through the plugins. Synthetic data only, made here from fixed seeds.
* tests/stata/absorption_method_declared_contract.sh runs every mode in its own fresh Stata process,
* for the new and the frozen package, and compares.
*
*   do absorption_method_declared_contract.do <mode> <leg>
*   modes: forced_fast forced_comparable plain_fast plain_comparable forced_savefe plain_savefe
*          request xfepout_env xfepout_plain profile_make xfepout_profile xhdfe_profile
*   (forced_*: the runner sets XHDFE_USE_KRYLOV=1; xfepout_env as well, which xfepout does not read)
*
* Writes amd_<leg>.txt: per fit one line "FIT <tag> rc method iterations status reason", then the
* coefficients and the lower triangle of e(V) (xhdfe) or the signature and the sums of the
* partialled variables (xfepout) in hexadecimal (%21x). Ends with AMD_PASS <mode>.
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
quietly findfile xfepout.plugin
assert `"`r(fn)'"' == `"`adopath'/xfepout.plugin"'

* Ten rows per worker, firms at random (well connected).
capture program drop panel_data
program define panel_data
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

* Two rows per worker: the levels are more than 35% of the rows, so the automatic sparse solver of
* the partialling-out entry does not serve the call before the Krylov-PCG route.
capture program drop short_data
program define short_data
    clear
    set seed 20261118
    set obs 25000
    generate long worker = _n
    generate double aw = rnormal()
    expand 2
    bysort worker: generate int year = _n
    generate long firm = floor(500 * runiform())
    generate double x1 = rnormal() + .3 * aw + .3 * mod(firm, 17) / 17
    generate double x2 = rnormal() + .3 * aw
    generate double x3 = rnormal()
    generate double y = rnormal() + aw + mod(firm, 23) / 23 + year / 10 + .5 * x1 - .25 * x2
end

* Few movers: the capturing sweep of a fit with saved effects stops above the floor (E1).
capture program drop lowmob_data
program define lowmob_data
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
end

capture program drop emit_xhdfe
program define emit_xhdfe
    args tag rc
    if `rc' != 0 {
        file write receipt "FIT `tag' `rc' . . . ." _n
        exit
    }
    file write receipt "FIT `tag' 0 `=e(absorption_method_used)' `=e(iterations)' `=e(reduced_pcg_status)' " ///
        "`e(reduced_pcg_reason)'" _n
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

capture program drop emit_xfepout
program define emit_xfepout
    args tag rc
    if `rc' != 0 {
        file write receipt "FIT `tag' `rc' . . . ." _n
        exit
    }
    file write receipt "FIT `tag' 0 `=e(method_used)' `=e(iterations)' . ." _n
    quietly _datasignature p_y p_x1 p_x2 p_x3, nonames
    file write receipt "signature `r(datasignature)'" _n
    foreach v in p_y p_x1 p_x2 p_x3 {
        quietly summarize `v', meanonly
        file write receipt "sum_`v' " %21x (r(sum)) _n
    }
end

local fast "tolerancemode(xhdfe-fast) tolerance(1e-8) numthreads(4) gpubackend(cpu) absorptioncachemode(off)"
local comparable "tolerancemode(reghdfe-comparable) tolerance(1e-8) numthreads(4) gpubackend(cpu) absorptioncachemode(off)"
* xfepout's own default method is symmetric-gauss-seidel; a profile's Krylov suggestion applies only
* to an Auto request (hdfe_regressor_v11.cpp, partial_out), so every xfepout call here asks for auto.
local xfe "absorptionmethod(auto) tolerance(1e-8) numthreads(4) gpubackend(cpu) absorptioncachemode(off)"
file open receipt using "amd_`leg'.txt", write text replace
if inlist("`mode'", "forced_fast", "plain_fast") {
    panel_data
    capture noisily xhdfe y x1 x2 x3, absorb(worker firm year) `fast'
    emit_xhdfe `mode' `=_rc'
}
if inlist("`mode'", "forced_comparable", "plain_comparable") {
    panel_data
    capture noisily xhdfe y x1 x2 x3, absorb(worker firm year) `comparable'
    emit_xhdfe `mode' `=_rc'
}
if inlist("`mode'", "forced_savefe", "plain_savefe") {
    lowmob_data
    capture noisily xhdfe y x1 x2 x3, absorb(worker firm year, savefe) `comparable'
    emit_xhdfe `mode' `=_rc'
    capture drop __hdfe*
}
if "`mode'" == "request" {
    panel_data
    foreach m in reduced-pcg krylov-pcg krylov_pcg {
        capture noisily xhdfe y x1 x2 x3, absorb(worker firm year) absorptionmethod(`m') `comparable'
        file write receipt "REQUEST `m' rc=`=_rc'" _n
    }
}
if inlist("`mode'", "xfepout_env", "xfepout_plain") {
    short_data
    capture noisily xfepout y x1 x2 x3, absorb(worker firm year) generate(p_) `xfe'
    emit_xfepout `mode' `=_rc'
}
if "`mode'" == "profile_make" {
    * A mobility profile written by xfepout, then the same profile suggesting the Krylov-PCG route.
    short_data
    capture noisily xfepout y x1 x2 x3, absorb(worker firm year) generate(p_) mobilityprofile ///
        mobfile("amd_profile_`leg'.txt") `xfe'
    emit_xfepout `mode' `=_rc'
    tempname in out
    file open `in' using "amd_profile_`leg'.txt", read text
    file open `out' using "amd_profile_krylov_`leg'.txt", write text replace
    local changed 0
    file read `in' line
    while r(eof) == 0 {
        if `"`line'"' == "suggest_krylov=0" {
            local line "suggest_krylov=1"
            local ++changed
        }
        file write `out' `"`line'"' _n
        file read `in' line
    }
    file close `in'
    file close `out'
    assert `changed' == 1
}
if "`mode'" == "xfepout_profile" {
    local profile : env AMD_PROFILE
    confirm file "`profile'"
    short_data
    capture noisily xfepout y x1 x2 x3, absorb(worker firm year) generate(p_) mobfile("`profile'") `xfe'
    emit_xfepout `mode' `=_rc'
}
if "`mode'" == "xhdfe_profile" {
    local profile : env AMD_PROFILE
    confirm file "`profile'"
    short_data
    capture noisily xhdfe y x1 x2 x3, absorb(worker firm year) mobfile("`profile'") `comparable'
    emit_xhdfe `mode' `=_rc'
}
file close receipt
display "AMD_PASS `mode'"
