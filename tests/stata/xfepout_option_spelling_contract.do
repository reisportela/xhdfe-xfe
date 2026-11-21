version 16
clear
set more off
args package
adopath ++ "`package'"
findfile xfepout.ado
assert "`r(fn)'" == "`package'/xfepout.ado"
set obs 400
set seed 7119
generate int a = 1 + mod(_n,10)
generate int b = 1 + mod(floor((_n-1)/10),10)
generate double x = rnormal()
generate double y = 2*x + a + b + rnormal()
xfepout y x, absorb(a b) generate(z_) minparallelrows(200) ///
    targetrowsperthread(100) symmetricsweep festructurecache fecachemode(off)
assert e(converged) == 1
display "XFEPOUT_OPTION_SPELLING_PASS"
