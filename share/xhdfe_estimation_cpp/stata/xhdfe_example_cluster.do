*! xhdfe example 2.35.0 25nov2026
* Requires Stata 16 or later and xhdfe installed from SSC or the release.
* nlsw88 ships with Stata. The do-file runs offline and replaces data in memory.

sysuse nlsw88, clear

* Explain hourly wages using job tenure and total work experience.
* Absorb industry and occupation effects and cluster by industry.
xhdfe wage tenure ttl_exp, absorb(industry occupation) vce(cluster industry) ///
    tolerancemode(reghdfe-comparable) reducedpcg(on) reducedpcgfine(auto)
