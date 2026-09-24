clear all
set more off
args package_root
adopath ++ "`package_root'/stata"
findfile xhdfe.ado
if (`"`r(fn)'"' != `"`package_root'/stata/xhdfe.ado"') error 601
findfile xhdfe.plugin
if (`"`r(fn)'"' != `"`package_root'/stata/xhdfe.plugin"') error 601
di "T01_PURE_SLOPE_START"
set obs 192
which xhdfe
generate byte worker = floor((_n-1)/48)
generate double x = 2*mod(_n-1,2)-1 + .5*worker
generate double z = 2*mod(floor((_n-1)/2),2)-1
generate double y = 3 + .75*x + (1+worker)*z + .125*(2*mod(floor((_n-1)/4),2)-1)

regress y x ibn.worker#c.z, noconstant
predict double u_ols, residuals
scalar t01_b = _b[x]
scalar t01_rss = e(rss)
display "NO_CONSTANT_OLS_B " %21.15g _b[x]
display "NO_CONSTANT_OLS_RSS " %21.15g e(rss)

xhdfe y x, absorb(worker#c.z) residuals(u_xhdfe) numthreads(2) gpubackend(cpu)
generate double error_xhdfe = abs(u_xhdfe-u_ols)
summarize error_xhdfe, meanonly
assert r(max)<1e-8
assert abs(_b[x]-t01_b)<1e-9
display "XHDFE_NO_CONSTANT_RESIDUAL_ERROR " %21.15g r(max)
display "XHDFE_NO_CONSTANT_B " %21.15g _b[x]
display "XHDFE_REPORTED_RSS " %21.15g e(rss)
generate double u2 = u_xhdfe^2
summarize u2, meanonly
display "XHDFE_RECONSTRUCTED_RSS " %21.15g r(sum)
assert abs(r(sum)-t01_rss)<=1e-8+1e-10*abs(t01_rss)
assert abs(e(rss)-t01_rss)<=1e-8+1e-10*abs(t01_rss)
di "T01_PURE_SLOPE_PASS"
