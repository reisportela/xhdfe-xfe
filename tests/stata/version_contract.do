* The displayed version, ado header, OLS and IV results must agree.
version 16
clear all
set more off
args backend
if ("`backend'" == "") local backend cpu
local adopath : env XHDFE_STATA_ADOPATH
if (`"`adopath'"' != "") adopath ++ `"`adopath'"'
findfile xhdfe.ado
local found `"`r(fn)'"'
* When a package directory is requested, the ado must have been loaded from it
* (a wrapped log line cannot be grepped reliably, so emit one short marker).
program define version_contract_source
    args command found adopath
    if (`"`adopath'"' != "") {
        local want `"`adopath'/`command'.ado"'
        local want : subinstr local want "//" "/", all
        local have : subinstr local found "//" "/", all
        if (`"`have'"' != `"`want'"') {
            display as error "`command'.ado was loaded from `have', not from `want'"
            exit 9
        }
    }
    display "VERSION_CONTRACT_SOURCE_OK `command'"
end
version_contract_source xhdfe `"`found'"' `"`adopath'"'
tempname source
file open `source' using `"`found'"', read text
file read `source' header
file close `source'
assert regexm(`"`header'"', "^[*]! version (.+)$")
local expected = regexs(1)
quietly xhdfe, version
assert `"`e(version)'"' == `"`expected'"'
set seed 16102026
set obs 1200
generate int firm = mod(_n,40)+1
generate double x = rnormal()
generate double z = rnormal()
generate double d = .8*z + .3*x + rnormal()
generate double y = .7*d + .4*x + firm/30 + rnormal()
quietly xhdfe y x d, absorb(firm) numthreads(1) gpubackend(`backend')
assert `"`e(version)'"' == `"`expected'"'
quietly xhdfe y x d, absorb(firm) endogenous(d) instruments(z) ///
    ffirst numthreads(1) gpubackend(`backend')
assert `"`e(version)'"' == `"`expected'"'
quietly estat firststage
assert `"`e(version)'"' == `"`expected'"'
quietly xhdfe, ffirst
assert `"`e(version)'"' == `"`expected'"'
* xfepout: header, query, partial-out fit and replay share one identity too.
findfile xfepout.ado
local found `"`r(fn)'"'
version_contract_source xfepout `"`found'"' `"`adopath'"'
file open `source' using `"`found'"', read text
file read `source' header
file close `source'
assert regexm(`"`header'"', "^[*]! version (.+)$")
local expected_xfepout = regexs(1)
quietly xfepout, version
assert `"`e(version)'"' == `"`expected_xfepout'"'
quietly xfepout y x, absorb(firm) generate(po_) numthreads(1) gpubackend(`backend')
assert `"`e(version)'"' == `"`expected_xfepout'"'
quietly xfepout
assert `"`e(version)'"' == `"`expected_xfepout'"'
display "VERSION_CONTRACT_PASS `backend'"
exit
