* The displayed version, ado header, OLS and IV results must agree.
version 16
clear all
set more off
args backend
if ("`backend'" == "") local backend cpu
local adopath : env XHDFE_STATA_ADOPATH
if (`"`adopath'"' != "") adopath ++ `"`adopath'"'
findfile xhdfe.ado
tempname source
file open `source' using `"`r(fn)'"', read text
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
display "VERSION_CONTRACT_PASS `backend'"
exit
