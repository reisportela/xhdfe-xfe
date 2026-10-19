* Issue #16: synthetic fixture supplied by Cristian Sanchez, October 2026.
* CPU/CUDA regression contract. Explicit LSMR is checked on CPU only.
* No external data. Check both coefficients and clustered SEs, not just the
* absorber's orthogonality certificate, which cannot detect null-space drift.
version 16
clear all
set more off
set linesize 180
args backend threads
if "`backend'" == "" local backend cpu
assert inlist("`backend'", "cpu", "cuda")
if "`threads'" == "" local threads 1
local modes "default fast strict lsmr cg"
if "`backend'" == "cuda" local modes "default fast strict cg"
local package : env XHDFE_STATA_ADOPATH
if `"`package'"' != "" adopath ++ `"`package'"'
which xhdfe
which reghdfe
which ivreghdfe

program define cg_fixture
    args seed y0
    clear
    set seed `seed'
    set obs 1536
    gen long sid = _n
    gen int fy = cond(sid <= 1515, 0, cond(sid <= 1522, 2015, cond(sid <= 1526, 2016, cond(sid <= 1532, 2017, 2018))))
    gen long did = ceil(92 * runiform()^1.6)
    gen double share = cond(runiform() < 0.55, exp(rnormal(-0.3, 1.1)), 0)
    gen double a = rnormal()
    expand 5
    bys sid: gen int year = 2014 + _n
    drop if year < `y0' | year < fy
    gen byte entry_age = cond(fy == 0, 0, year - fy + 1)
    gen double x = rnormal()
    gen byte rare = inlist(sid, 1, 2) | (sid == 3 & year != 2017) | (sid == 4 & year == 2017)
    gen double lvl = 10 + 0.03 * (year - 2017) + 0.02 * rnormal()
    bys did (sid year): gen double dfe = rnormal(0, 0.9) if _n == 1
    bys did (dfe): replace dfe = dfe[1]
    bys did year (sid): gen double dsh = rnormal(0, 0.02) if _n == 1
    bys did year (dsh): replace dsh = dsh[1]
    gen double dagg = 10 + dfe + 0.03 * (year - 2017) + dsh
    gen double z = 20 * share * cond(year == 2015, 1.4, cond(year == 2016, 2.1, cond(year == 2017, 4.2, cond(year == 2018, 19, 52))))
    gen double expo = max(0, 0.004 * z + 0.3 * a + 0.5 * rnormal())
    sort sid year
end

tempname out bref seref nref exact reduced first
postfile `out' str12 cas str12 mode str10 ordering double(seed b se bref seref iter seconds gpu) using "cg_near_redundant.dta", replace
foreach seed in 1 27 93 {
    foreach spec in "rare 2015 x x" "dagg 2017 x x" "dagg 2017 expo z" {
        tokenize `spec'
        local outcome `1'
        local start `2'
        local regressor `3'
        local instrument `4'
        local cas "`outcome'_`regressor'"
        cg_fixture `seed' `start'
        local ivopts
        if "`regressor'" == "x" {
            quietly areg `outcome' x i.year i.entry_age, absorb(sid)
            scalar `exact' = _b[x]
            quietly reghdfe `outcome' x, absorb(sid year entry_age) cluster(did) tolerance(1e-12)
        }
        else {
            quietly areg `outcome' z i.year i.entry_age, absorb(sid)
            scalar `reduced' = _b[z]
            quietly areg expo z i.year i.entry_age, absorb(sid)
            scalar `first' = _b[z]
            scalar `exact' = `reduced' / `first'
            quietly ivreghdfe `outcome' (expo = z), absorb(sid year entry_age) cluster(did) tolerance(1e-12)
            local ivopts "endogenous(expo) instruments(z) statstyle(legacy)"
        }
        scalar `bref' = _b[`regressor']
        scalar `seref' = _se[`regressor']
        scalar `nref' = e(N)
        assert abs(`bref' - `exact') < 1e-11
        gen double shuffle = runiform()
        foreach ordering in original reverse shuffled {
            foreach mode of local modes {
                if "`ordering'" == "original" sort sid year
                if "`ordering'" == "reverse" gsort -sid -year
                if "`ordering'" == "shuffled" sort shuffle
                local opt
                if "`mode'" == "fast" local opt "tolerancemode(xhdfe-fast)"
                if "`mode'" == "strict" local opt "tolerancemode(strict-residual)"
                if "`mode'" == "lsmr" local opt "absorptionmethod(lsmr)"
                if "`mode'" == "cg" local opt "convergence(reghdfe)"
                timer clear 1
                timer on 1
                quietly xhdfe `outcome' `regressor', absorb(sid year entry_age) cluster(did) ///
                    numthreads(`threads') gpubackend(`backend') `ivopts' `opt'
                timer off 1
                quietly timer list 1
                post `out' ("`cas'") ("`mode'") ("`ordering'") (`seed') ///
                    (_b[`regressor']) (_se[`regressor']) (`bref') (`seref') (e(iterations)) (r(t1)) (e(gpu_used))
                di "CG_CHECK `cas' seed=`seed' `ordering' `mode' b=" %24.17g _b[`regressor'] " se=" %24.17g _se[`regressor'] " iter=" e(iterations)
                assert e(N) == `nref'
                assert e(converged) == 1 & e(precision_certified) == 1
                assert e(gpu_used) == ("`backend'" == "cuda")
                assert abs(_b[`regressor'] - `bref') < 1e-12 + 1e-7 * abs(`bref')
                assert abs(_se[`regressor'] - `seref') < 1e-12 + 1e-6 * `seref'
                assert e(iterations) < 1000
            }
        }
    }
}
postclose `out'
use "cg_near_redundant.dta", clear
format b-gpu %24.17g
export delimited using "cg_near_redundant.csv", replace datafmt
di "CG_NEAR_REDUNDANT_CONTRACT_PASS `backend'"
