version 18.0
args stata_path
clear all
set more off
adopath ++ "`stata_path'"
set seed 73
set obs 1000
generate long f = mod(_n,20)
generate double x = rnormal()
generate double y = .7*x + f/10 + rnormal()
generate double w = 2
quietly xhdfe y x [iw=w], absorb(f) numthreads(1)
replace w = -1 in 1/50
capture noisily xhdfe y x [iw=w], absorb(f) numthreads(1)
assert _rc == 198
tempname stale
capture matrix `stale' = e(b)
assert _rc != 0
capture noisily xfepout y x [iw=w], absorb(f) generate(rejected_)
assert _rc == 198
capture confirm variable rejected_y
assert _rc != 0
quietly regress y x i.f [iw=w] if w > 0
scalar positive_b = _b[x]
scalar positive_v = _se[x]^2
quietly xhdfe y x [iw=w] if w > 0, absorb(f) numthreads(1)
assert e(N) == 1900
assert abs(_b[x]-positive_b) <= 1e-9
assert abs(_se[x]^2-positive_v) <= 1e-8*positive_v
replace w = 0 in 1/50
quietly xhdfe y x [iw=w], absorb(f) numthreads(1)
assert e(N) == 1900
assert abs(_b[x]-positive_b) <= 1e-9
assert abs(_se[x]^2-positive_v) <= 1e-8*positive_v
quietly xfepout y x [iw=w], absorb(f) generate(partial_)
assert missing(partial_y) if w == 0
assert !missing(partial_y) if w > 0
display "STATA_NEGATIVE_IWEIGHT_CONTRACT PASS"
