* Requires Stata 16 or later and the installed xhdfe package.
sysuse nlsw88, clear

* Absorb industry and occupation effects and cluster by industry.
xhdfe wage tenure ttl_exp, absorb(industry occupation) vce(cluster industry)
