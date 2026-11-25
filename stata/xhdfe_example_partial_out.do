*! xhdfe example 2.35.1 25nov2026
* Requires Stata 16 or later, xhdfe and xfepout. SSC xhdfe includes both.
* auto ships with Stata. The do-file runs offline and replaces data in memory.

sysuse auto, clear

* Partial out the effects of origin and repair rating from every column.
xfepout price mpg weight, absorb(foreign rep78) generate(within_)

* Regress the transformed outcome on the transformed regressors.
* The slopes match xhdfe with absorb(foreign rep78) on the same sample.
* Use xhdfe for standard errors that account for the absorbed effects.
regress within_price within_mpg within_weight, noconstant

* Fit the same slopes with fixed-effect-adjusted inference.
xhdfe price mpg weight, absorb(foreign rep78)
