version 16
clear all
set more off
set processors 1
args package threads team
adopath ++ "`package'"
quietly findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
quietly findfile xhdfe.plugin
assert "`r(fn)'" == "`package'/xhdfe.plugin"

* Cross both the write-back threshold and a boundary in the wide-ID reader.
set obs 1100032
set seed 20261122
generate double x = rnormal()
generate double z = rnormal()
generate long fe = 1 + mod(_n, 17)
generate double y = .4*x - .2*z + .01*fe + rnormal()
generate double ascending = 2^40 + ceil(_n/16)
generate double inverted = ascending
replace inverted = ascending[16385] in 16384
replace inverted = ascending[16384] in 16385

foreach cluster of varlist ascending inverted {
    quietly regress y x z, vce(cluster `cluster')
    matrix reference_b = e(b)
    matrix reference_V = e(V)
    scalar reference_tss = e(mss) + e(rss)
    quietly xhdfe y x z, noabsorb vce(cluster `cluster') ///
        gpubackend(cpu) numthreads(`threads') abscachemode(off)
    assert e(N) == _N & e(converged) == 1 & e(sample)
    assert mreldif(e(b), reference_b) < 1e-9
    assert mreldif(e(V), reference_V) < 1e-8
    assert reldif(e(tss), reference_tss) < 1e-9
    assert e(threads_used) == `team' & e(parallel_workers_active) == `team'
}

* The two singleton levels lie at different write-back blocks.
replace fe = 18 in 16384
replace fe = 19 in l
quietly regress y x z i.fe if fe <= 17
scalar reference_x = _b[x]
scalar reference_z = _b[z]
scalar reference_rss = e(rss)
quietly xhdfe y x z, absorb(fe) gpubackend(cpu) ///
    numthreads(`threads') abscachemode(off)
assert e(N) == _N - 2 & e(num_singletons) == 2 & e(converged) == 1
assert e(sample) == (fe <= 17)
assert abs(_b[x] - reference_x) < 1e-9
assert abs(_b[z] - reference_z) < 1e-9
assert reldif(e(rss), reference_rss) < 1e-9
assert e(threads_used) == `team' & e(parallel_workers_active) == `team'
display "POSTFIT_PARALLEL_CONTRACT_PASS"
