version 18.0
args stata_path scratch
clear all
set more off
adopath ++ "`stata_path'"
set seed 901
set obs 3000
generate long f = mod(_n,20)
generate double z = rnormal()
generate double x = .8*z + rnormal()
generate double y = 1.3*x + f/10 + rnormal()
generate double w = .5 + runiform()
quietly ivregress 2sls y i.f (x=z) [pw=w], small
scalar iv_b = _b[x]
scalar iv_v = _se[x]^2
quietly xhdfe y x [pw=w], absorb(f) endogenous(x) instruments(z) numthreads(1)
assert abs(_b[x]-iv_b) <= 1e-9*max(1,abs(iv_b))
assert abs(_se[x]^2-iv_v) <= 1e-8*iv_v
quietly summarize y [aw=w], meanonly
scalar ybar = r(mean)
quietly summarize x [aw=w], meanonly
assert abs(_b[_cons]-(ybar-r(mean)*_b[x])) < 1e-9
quietly regress y x i.f [pw=w]
scalar ols_b = _b[x]
scalar ols_v = _se[x]^2
quietly xhdfe y x f [pw=w], absorb(f) numthreads(1)
assert _b[f] == 0
assert abs(_b[x]-ols_b) <= 1e-9*max(1,abs(ols_b))
assert abs(_se[x]^2-ols_v) <= 1e-8*ols_v
quietly xhdfe y x, absorb(f) symmetricsweep numthreads(1)
quietly xhdfe y x, absorb(f) sym numthreads(1)
quietly xhdfe y x, absorb(f) mobilityprofile mobfile("`scratch'/mobility.txt") numthreads(1)
confirm file "`scratch'/mobility.txt"
quietly xhdfe y x, absorb(f) festructurecache numthreads(1)
confirm file "`stata_path'/xhdfe_fe_structure_cache.bin"
quietly xhdfe y x, absorb(f) fescache("`scratch'/fe.bin") numthreads(1)
confirm file "`scratch'/fe.bin"
matrix b = e(b)
matrix V = e(V)
quietly xhdfe y x, absorb(f) fescache("`scratch'/fe.bin") fecachemode(read) numthreads(1)
assert mreldif(b,e(b)) == 0
assert mreldif(V,e(V)) == 0
display "STATA_PWEIGHT_OPTIONS_CONTRACT PASS"
