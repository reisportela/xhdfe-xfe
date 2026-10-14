clear all
prog drop _all
discard
set rmsg on
set more off

cd "/home/mangelo/Documents/BigData/Paulo/logs"

capture program drop xhdfe
discard

* Ensure latest xhdfe is loaded
adopath + "/home/mangelo/Documents/GitHub/xhdfe/stata"
capture program drop xhdfe
discard

capture log close
log using log_xhdfe_felsdvsimul_CPU_recover_FEs_V02.txt,text replace

pq use using /home/mangelo/Documents/BigData/Paulo/felsdvsimul.parquet, clear
gunique i


di _new(1) "1) xhdfe"
xhdfe y x1 x2, a(i j, savefe) ////
    vce(cluster i) keepsing tolerance(1e-8) numthreads(32)
rename __hdfe1__ xFEw0
rename __hdfe2__ xFEf0

xhdfe y x1 x2, a(xFEw1=i xFEf1=j) ////
    vce(cluster i) keepsing tolerance(1e-8) numthreads(32)

di _new(1) "2) reghdfe baseline"
reghdfe y x1 x2, a(i j, savefe) ///
    vce(cluster i) keepsing tolerance(1e-8)
rename __hdfe1__ rFEw0
rename __hdfe2__ rFEf0

reghdfe y x1 x2, a(rFEw1=i rFEf1=j) ///
    vce(cluster i) keepsing tolerance(1e-8)

compare rFEw0 xFEw0
compare rFEw1 xFEw1

compare rFEf0 xFEf0
compare rFEf1 xFEf1

gen dFEw0=rFEw0-xFEw0
gen dFEw1=rFEw1-xFEw1

gen dFEf0=rFEf0-xFEf0
gen dFEf1=rFEf1-xFEf1

gen pFEw0=dFEw0/rFEw0*100
gen pFEw1=dFEw1/rFEw1*100

gen pFEf0=dFEf0/rFEf0*100
gen pFEf1=dFEf1/rFEf1*100

format %20.6f pFEw0 pFEw1 pFEf0 pFEf1

order rFEw0 xFEw0 dFEw0 pFEw0 rFEf0 xFEf0 dFEf0 pFEf0 rFEw1 xFEw1 dFEw1 pFEw1 rFEf1 xFEf1 dFEf1 pFEf1, after(y)

sum d*, detail
sum p*, detail

preserve
	contract i rFEw0 xFEw0 dFEw0 pFEw0
	gunique i
	sum rFEw0 xFEw0 dFEw0 pFEw0, detail
restore

preserve
	contract i rFEw1 xFEw1 dFEw1 pFEw1
	gunique i
	sum rFEw1 xFEw1 dFEw1 pFEw1, detail
restore

preserve
	contract j rFEf0 xFEf0 dFEf0 pFEf0
	gunique j
	sum rFEf0 xFEf0 dFEf0 pFEf0, detail
restore

preserve
	contract j rFEf1 xFEf1 dFEf1 pFEf1
	gunique j
	sum rFEf1 xFEf1 dFEf1 pFEf1, detail
restore

sort i j

sort i j
save felsdvsimul_with_FEs, replace
export excel using "/home/mangelo/Documents/BigData/Paulo/logs/felsdvsimul_with_FEs.xlsx", firstrow(variables) replace
log close
