* Requires Stata 16 or later and xhdfe, which includes xfepout.
sysuse auto, clear

* Partial out the effects of origin and repair rating from every column.
xfepout price mpg weight, absorb(foreign rep78) generate(within_)

* Regress the transformed outcome on the transformed regressors.
* The slopes match xhdfe. Use xhdfe for FE-adjusted standard errors.
regress within_price within_mpg within_weight, noconstant
