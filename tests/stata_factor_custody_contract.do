version 18.0
args stata_path
clear all
set more off
adopath ++ "`stata_path'"
set seed 7
set obs 1000
generate long id = ceil(_n/10)
generate byte g = mod(_n,4)
generate double x = rnormal()
generate double __xhdfe_user = rnormal()
generate double __xhdfe_1 = _n
generate double y = .5*x + __xhdfe_user + id/10 + rnormal()
clonevar expected = __xhdfe_user
quietly regress y x __xhdfe_user i.g i.id
scalar oracle_x = _b[x]
scalar oracle_user = _b[__xhdfe_user]
quietly xhdfe y x __xhdfe_user i.g, absorb(id) numthreads(1)
assert abs(_b[x]-oracle_x) < 1e-9
assert abs(_b[__xhdfe_user]-oracle_user) < 1e-9
assert __xhdfe_user == expected
assert __xhdfe_1 == _n
quietly predict double xb, xb
assert !missing(xb)
generate byte selected = abs(x)<1
quietly xhdfe y x if selected, absorb(id) numthreads(1)
matrix b = e(b)
matrix V = e(V)
quietly xhdfe y x if max(0, abs(x))<1, abs(id) numthreads(1)
assert mreldif(b,e(b)) == 0
assert mreldif(V,e(V)) == 0
quietly xfepout y x if max(0, abs(x))<1, abs(id) generate(p_)
assert __xhdfe_user == expected
assert __xhdfe_1 == _n
display "STATA_FACTOR_CUSTODY_CONTRACT PASS"
