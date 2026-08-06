version 16.0
clear all
set more off

local repo : environment XHDFE_REPO_ROOT
if ("`repo'" == "") local repo "`c(pwd)'"
adopath ++ "`repo'/stata"
discard

set seed 20260804
set obs 240
generate long row = _n - 1
generate int fe1 = mod(row, 17)
generate int fe2 = mod(row, 13)
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double y = .4*x1 - .7*x2 + .1*fe1 - .05*fe2 + rnormal()
generate byte one_cluster = 0

xhdfe y x1 x2, absorb(fe1 fe2) vce(cluster one_cluster) keepsingletons numthreads(1)
assert e(N_clust) == 1
assert missing(_se[x1])
assert missing(_se[x2])
assert missing(_se[_cons])

replace y = 1
xhdfe y x1 x2, absorb(fe1 fe2) keepsingletons numthreads(1)
assert e(tss) == 0
assert e(tss_within) == 0
assert missing(e(r2))
assert missing(e(r2_within))

clear
set obs 3001
generate long row = _n - 1
generate int chain1 = cond(_n <= 3000, floor(row/3), 999)
generate int chain2 = cond(_n <= 3000, chain1 + (mod(row, 3) == 2), 1000)
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double y = .7*x1 - .2*x2 + rnormal()

xhdfe y x1 x2, absorb(chain1 chain2) keepsingletons numthreads(1) ///
    tolerance(1e-8) tolerancemode(reghdfe-comparable)
assert e(converged) == 1
assert e(precision_certified) == 1

xhdfe y x1 x2, absorb(chain1 chain2) keepsingletons numthreads(1) ///
    tolerance(1e-8) tolerancemode(reghdfe-comparable) ///
    absorptionmethod(gauss-seidel)
assert e(converged) == 1
assert e(precision_certified) == 0
assert e(absorption_method_used) == 1

display as result "PASS: Stata audit 20260804 remediation contracts"
