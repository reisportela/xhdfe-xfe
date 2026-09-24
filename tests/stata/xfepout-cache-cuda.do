* CUDA falsifier: a matching CPU profile must not force Krylov/sparse hints.
version 16.0
clear all
set more off

local repo "`c(pwd)'/../.."
adopath ++ "`repo'/stata"
discard
which xfepout
quietly xfepout, version
assert "`e(version)'" == "1.13.1 20sep2026"

set seed 20260901
set obs 6000
gen long fe1 = ceil(_n / 30)
gen long fe2 = mod(_n - 1, 37) + 1
gen double x1 = rnormal()
gen double x2 = rnormal()
gen double y = .7*x1 - .3*x2 + fe1/100 + fe2/70 + rnormal()
replace fe1 = 999 if _n == _N
replace fe2 = 999 if _n == _N
sort fe1 fe2

tempfile profile cpu_hint cpu_cache
quietly xfepout y x1 x2, absorb(fe1 fe2) generate(cp_) ///
    tolerance(1e-12) numthreads(8) gpubackend(cpu) absorptionmethod(auto) ///
    mobfile(`"`profile'"') mobilityprofile ///
    absorptioncache(`"`cpu_cache'"') abscachemode(write)
assert e(converged) == 1
confirm file `"`profile'"'
confirm file `"`cpu_cache'"'

file open profile_in using `"`profile'"', read text
file open profile_out using `"`cpu_hint'"', write text replace
local replaced 0
file read profile_in line
while (r(eof) == 0) {
    local output `"`line'"'
    if (strtrim(`"`line'"') == "suggest_krylov=0") {
        local output "suggest_krylov=1"
        local replaced 1
    }
    file write profile_out `"`output'"' _n
    file read profile_in line
}
file close profile_in
file close profile_out
assert `replaced' == 1

quietly xfepout y x1 x2, absorb(fe1 fe2) generate(gp_) ///
    tolerance(1e-12) numthreads(8) gpubackend(cuda) absorptionmethod(auto) ///
    mobfile(`"`cpu_hint'"') ///
    absorptioncache(`"`cpu_cache'"') abscachemode(read)
assert e(converged) == 1
assert e(gpu_attempted) == 1
assert e(gpu_used) == 1
assert "`e(gpu_backend)'" == "cuda"
assert "`e(gpu_status)'" == "used"

di as result "XFEPOUT_CUDA_CPU_HINT_GUARD_PASS"
