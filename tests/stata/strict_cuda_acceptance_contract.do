* Package B (18nov2026): the Stata leg of tests/analytic_precision/strict_cuda_acceptance_contract.py, which writes
* the synthetic fixtures and their oracles (oracle_<name>.do: oracle_b, oracle_V, oracle_rss) into DATADIR, runs one
* fresh stata-mp session per leg with XHDFE_STRICT_ACCEPT_TRACE=1 and XHDFE_P7_REFINE_TRACE=1 (plus the leg's
* switches), and grades the trace (route, refined=0, the continuation) from the session's stderr. In this file:
* the limits of a fit in scope (coefficients 1e-8 max(1,|b|), standard errors 1e-7 relative, the covariance
* 1e-8 sqrt(V_jj V_kk), the RSS 1e-8 relative, D <= tolerance() on the stored residuals, their sum of squares
* equal to e(rss) to 1e-10, e(tolerance_mode), the GPU diagnostics, iterations within maxiter()). Each fit prints
* one SCA_OK or SCA_REFUSED line (names in order); a failed expectation prints SCA_FAIL; SCA_PASS closes a clean leg.
* Estimates are written in hexadecimal to LEGDIR/est_<name>.txt and, where a leg compares them, the stored residuals
* to LEGDIR/resid_<name>.dta (synthetic data only).
* Legs: scope (F0-F4, T3, W0, cov240k; the continuation on cov240k x 2^17; the short budget three times, then an
* easy fit; the large-scale row, then W0; cov240k with a fourth effect at tolerance(1e-8), against reghdfe at 1e-12
* when its adopath is given, its standard errors on a SCA_K line), fourfe_tight (the same at tolerance(1e-10), its
* standard errors asserted; the driver runs three sessions), legacy_accept (F4), legacy_resid and both (F0),
* reference (F4 and F0 with the reference package), excluded (group() without and with individual(), CPU strict
* and an explicit criterion on CUDA, each with residuals()).
version 18
clear all
set more off
set linesize 255
args package datadir kind legdir large_exp large_budget short_budget reghdfe
adopath ++ "`package'"
discard
findfile xhdfe.ado
assert "`r(fn)'" == "`package'/xhdfe.ado"
set processors 16
global SCA_FAILS 0

capture program drop sca_fail
program define sca_fail
    args name what
    display "SCA_FAIL name=`name' what=`what'"
    global SCA_FAILS = $SCA_FAILS + 1
end

capture program drop sca_dump
program define sca_dump
    args name legdir
    mata: sca_fh = fopen("`legdir'/est_`name'.txt", "w")
    mata: sca_b = st_matrix("e(b)"); sca_V = st_matrix("e(V)")
    mata: for (sca_i = 1; sca_i <= cols(sca_b); sca_i++) fput(sca_fh, "b " + strofreal(sca_b[sca_i], "%21x"))
    mata: for (sca_i = 1; sca_i <= rows(sca_V); sca_i++) for (sca_j = 1; sca_j <= cols(sca_V); sca_j++) fput(sca_fh, "V " + strofreal(sca_V[sca_i, sca_j], "%21x"))
    mata: fput(sca_fh, "rss " + strofreal(st_numscalar("e(rss)"), "%21x")); fclose(sca_fh)
end

* The slopes of the last xhdfe fit against reghdfe at tolerance(1e-12) (adopath given): coefficients asserted within
* 1e-8 max(1,|b|); standard errors asserted within 1e-7 relative, or, with known(), printed on a SCA_K line that the
* driver grades against the known defect's declared band. Without an adopath it says so.
capture program drop sca_vs_reghdfe
program define sca_vs_reghdfe
    syntax , name(string) legdir(string) [adopath(string) known(string)]
    if "`adopath'" == "" {
        display "SCA_R name=`name' reghdfe=skipped"
        exit
    }
    matrix sca_xb = e(b)
    matrix sca_xV = e(V)
    adopath ++ "`adopath'"
    capture noisily reghdfe y x1 x2 x3 x4, absorb(fe1 fe2 fe3 fe4) vce(cluster fe1) tolerance(1e-12) iterate(100000)
    if _rc != 0 {
        display "SCA_R name=`name' reghdfe_rc=" _rc
        sca_fail `name' reghdfe_failed
        exit
    }
    sca_dump reghdfe_`name' "`legdir'"
    mata: sca_rb = st_matrix("e(b)"); sca_rV = st_matrix("e(V)"); sca_xb = st_matrix("sca_xb"); sca_xV = st_matrix("sca_xV")
    mata: st_local("rdb", strofreal(max(abs(sca_xb[1..4]-sca_rb[1..4]) :/ rowmax((J(4,1,1),abs(sca_rb[1..4]')))'), "%21.14e"))
    mata: st_local("rdse", strofreal(max(abs(sqrt(diagonal(sca_xV[1..4,1..4])) :/ sqrt(diagonal(sca_rV[1..4,1..4])) :- 1)), "%21.14e"))
    mata: st_local("rdv", strofreal(max(abs(sca_xV[1..4,1..4]-sca_rV[1..4,1..4]) :/ sqrt(diagonal(sca_rV[1..4,1..4])*diagonal(sca_rV[1..4,1..4])')), "%21.14e"))
    if `rdb' > 1e-8 sca_fail `name' reghdfe_b
    if "`known'" != "" {
        display "SCA_R name=`name' reghdfe_db=`rdb'"
        display "SCA_K name=`name' known=`known' reghdfe_dse=`rdse' reghdfe_dv=`rdv'"
    }
    else {
        display "SCA_R name=`name' reghdfe_db=`rdb' reghdfe_dse=`rdse' reghdfe_dv=`rdv'"
        if `rdse' > 1e-7 sca_fail `name' reghdfe_se
    }
end

* One fit: expect(accept|refuse|record), the oracle (scaled by yscale) when given, D and SS of residuals().
capture program drop sca_fit
program define sca_fit
    syntax , name(string) data(string) absorb(string) legdir(string) [vce(string) yscale(real 1) ///
        maxiter(integer 100000) tol(real 1e-8) mode(string) backend(string) conv(string) oracle(string) ///
        expect(string) group(string) individual(string) bonly saveresid]
    if "`mode'" == "" local mode "strict-residual"
    if "`backend'" == "" local backend "cuda"
    if "`conv'" == "" local conv "auto"
    use "`data'", clear
    if `yscale' != 1 quietly replace y = y*`yscale'
    capture ds x*
    local xvars "`r(varlist)'"
    local opts
    if "`vce'" != "" local opts "vce(`vce')"
    if "`group'" != "" local opts "`opts' group(`group')"
    if "`individual'" != "" local opts "`opts' individual(`individual')"
    capture noisily xhdfe y `xvars', absorb(`absorb') `opts' tolerance(`tol') tolerancemode(`mode') ///
        maxiter(`maxiter') gpubackend(`backend') numthreads(16) absorptioncachemode(off) ///
        convergence(`conv') residuals(r_sca)
    local rc = _rc
    if `rc' != 0 {
        local empty = ("`e(cmd)'" == "")
        capture matrix sca_stale = e(b)
        local nob = (_rc != 0)
        display "SCA_REFUSED name=`name' rc=`rc' ecmd_empty=`empty' no_eb=`nob'"
        if "`expect'" == "accept" sca_fail `name' refused
        if "`expect'" == "refuse" & !(`empty' & `nob') sca_fail `name' estimates_left
        exit
    }
    if "`expect'" == "refuse" sca_fail `name' accepted
    sca_dump `name' "`legdir'"
    tempvar s r2
    generate byte `s' = e(sample)
    quietly generate double `r2' = r_sca^2 if `s'
    quietly summarize `r2', meanonly
    scalar sca_ss = abs(r(sum)/e(rss) - 1)
    scalar sca_D = 0
    foreach fe of local absorb {
        tempvar m
        quietly bysort `fe': egen double `m' = mean(r_sca) if `s'
        quietly summarize `m', meanonly
        scalar sca_D = max(sca_D, abs(r(min)), abs(r(max)))
        drop `m'
    }
    local db = -1
    local dv = -1
    local dse = -1
    local drss = -1
    if "`oracle'" != "" {
        do "`oracle'"
        local k = colsof(oracle_b)
        matrix sca_ob = oracle_b*`yscale'
        mata: st_local("db", strofreal(max(abs(st_matrix("e(b)")-st_matrix("sca_ob")) :/ rowmax((J(`k',1,1),abs(st_matrix("sca_ob")')))'), "%21.14e"))
        if "`bonly'" == "" {
            matrix sca_oV = oracle_V*(`yscale'^2)
            mata: st_local("dv", strofreal(max(abs(st_matrix("e(V)")-st_matrix("sca_oV")) :/ sqrt(diagonal(st_matrix("sca_oV"))*diagonal(st_matrix("sca_oV"))')), "%21.14e"))
            mata: st_local("dse", strofreal(max(abs(sqrt(diagonal(st_matrix("e(V)"))) :/ sqrt(diagonal(st_matrix("sca_oV"))) :- 1)), "%21.14e"))
            if oracle_rss > 0 local drss = abs(e(rss)/(`yscale'^2)/oracle_rss - 1)
        }
    }
    mata: st_local("bhex", invtokens(strofreal(st_matrix("e(b)"), "%21x"), ","))
    display "SCA_OK name=`name' it=" e(iterations) " conv=" e(converged) " cert=" e(precision_certified) ///
        " gpu=" e(gpu_used) " be=`e(gpu_backend)' st=`e(gpu_status)' mode=`e(tolerance_mode)' tail=`e(cuda_tail)'" ///
        " tier=`e(reduced_pcg_reason)'"
    display "SCA_M name=`name' D=" %21.14e sca_D " ss=" %21.14e sca_ss " db=`db' dv=`dv' dse=`dse' drss=`drss'"
    display "SCA_X name=`name' bhex=`bhex'"
    if "`saveresid'" != "" {
        preserve
        keep r_sca
        rename r_sca r
        quietly save "`legdir'/resid_`name'.dta", replace
        restore
    }
    if "`expect'" == "accept" & "`mode'" == "strict-residual" & "`backend'" == "cuda" {
        if !(e(converged) == 1 & e(precision_certified) == 1 & e(gpu_used) == 1 & "`e(gpu_backend)'" == "cuda" ///
            & "`e(gpu_status)'" == "used" & "`e(tolerance_mode)'" == "strict-residual" & e(iterations) <= `maxiter' ///
            & "`e(reduced_pcg_reason)'" == "tolerance_mode") ///
            sca_fail `name' diagnostics
        if !(sca_D <= `tol') sca_fail `name' D
        if !(sca_ss <= 1e-10) sca_fail `name' stored_not_own
        if `db' > 1e-8 sca_fail `name' b
        if `dv' > 1e-8 sca_fail `name' V
        if `dse' > 1e-7 sca_fail `name' se
        if `drss' > 1e-8 sca_fail `name' rss
        if "`oracle'" != "" & `db' < 0 sca_fail `name' not_graded
    }
end

local D "`datadir'"
local L "`legdir'"
if "`kind'" == "scope" {
    foreach f in F0 F1 F2 F3 F4 W0 {
        sca_fit, name(`f') data("`D'/`f'.dta") absorb(fe1 fe2) legdir("`L'") oracle("`D'/oracle_`f'.do") expect(accept)
    }
    sca_fit, name(T3) data("`D'/T3.dta") absorb(fe1 fe2 fe3) legdir("`L'") oracle("`D'/oracle_T3.do") expect(accept)
    sca_fit, name(cov240k) data("`D'/cov240k.dta") absorb(fe1 fe2 fe3) vce(cluster fe1) legdir("`L'") ///
        oracle("`D'/oracle_cov240k.do") expect(accept)
    * The continuation (outcome times 2^17, exact), then the short budget, then an easy fit in the same session.
    sca_fit, name(cont_2p17) data("`D'/cov240k.dta") absorb(fe1 fe2 fe3) vce(cluster fe1) yscale(131072) ///
        legdir("`L'") oracle("`D'/oracle_cov240k.do") expect(accept)
    * Three times: the budget is the calibrated primary + 24 (package C), so a slower primary may still refuse before
    * the continuation; 0 means no calibrated budget (the driver records missing coverage) and skips these fits.
    if `short_budget' > 0 {
        forvalues k = 1/3 {
            sca_fit, name(cont_short_`k') data("`D'/cov240k.dta") absorb(fe1 fe2 fe3) vce(cluster fe1) ///
                yscale(131072) maxiter(`short_budget') legdir("`L'") expect(refuse)
        }
        sca_fit, name(easy_after_short) data("`D'/easy.dta") absorb(fe1 fe2 fe3) maxiter(`short_budget') ///
            legdir("`L'") expect(accept)
    }
    else display "SCA_SKIP cont_short: no calibrated short budget"
    * The large-scale row (W0 x 2^large_exp), then W0 in the same session.
    sca_fit, name(large_default) data("`D'/W0.dta") absorb(fe1 fe2) yscale(`=2^`large_exp'') ///
        maxiter(`large_budget') legdir("`L'") expect(refuse)
    sca_fit, name(W0_after_large) data("`D'/W0.dta") absorb(fe1 fe2) maxiter(`large_budget') legdir("`L'") ///
        oracle("`D'/oracle_W0.do") expect(accept)
    * Four effects at tolerance(1e-8): D, the diagnostics and the coefficients (oracle and reghdfe at 1e-12) are
    * asserted; the standard errors and covariance against reghdfe go on a SCA_K line, which the driver grades as
    * the known defect KNOWN_DEFECT_FOUR_EFFECT_COVARIANCE_CUDA (band (1e-7, 2e-7], a failure above it).
    sca_fit, name(cov240k_4fe) data("`D'/cov240k_4fe.dta") absorb(fe1 fe2 fe3 fe4) vce(cluster fe1) legdir("`L'") ///
        oracle("`D'/oracle_cov240k_4fe.do") expect(accept) bonly
    sca_vs_reghdfe, name(cov240k_4fe) legdir("`L'") adopath("`reghdfe'") known(KNOWN_DEFECT_FOUR_EFFECT_COVARIANCE_CUDA)
}
if "`kind'" == "fourfe_tight" {
    * The same four-effect fit at tolerance(1e-10) (stage 2c, pre-registered): accepted, D <= 1e-10 on the stored
    * residuals, coefficients within 1e-8 of the oracle and of reghdfe at 1e-12, standard errors within 1e-7 of it.
    sca_fit, name(cov240k_4fe_1e10) data("`D'/cov240k_4fe.dta") absorb(fe1 fe2 fe3 fe4) vce(cluster fe1) ///
        tol(1e-10) legdir("`L'") oracle("`D'/oracle_cov240k_4fe.do") expect(accept) bonly
    sca_vs_reghdfe, name(cov240k_4fe_1e10) legdir("`L'") adopath("`reghdfe'")
}
if "`kind'" == "legacy_accept" | "`kind'" == "reference" {
    sca_fit, name(F4) data("`D'/F4.dta") absorb(fe1 fe2) legdir("`L'") oracle("`D'/oracle_F4.do") expect(record)
}
if inlist("`kind'", "legacy_resid", "both", "reference") {
    sca_fit, name(F0) data("`D'/F0.dta") absorb(fe1 fe2) legdir("`L'") oracle("`D'/oracle_F0.do") expect(record) ///
        saveresid
}
if "`kind'" == "excluded" {
    sca_fit, name(group) data("`D'/groups.dta") absorb(year) group(patent) legdir("`L'") expect(record) saveresid
    sca_fit, name(group_individual) data("`D'/groups.dta") absorb(year inventor) group(patent) ///
        individual(inventor) legdir("`L'") expect(record) saveresid
    sca_fit, name(cpu_strict) data("`D'/cov240k.dta") absorb(fe1 fe2 fe3) vce(cluster fe1) backend(cpu) ///
        legdir("`L'") expect(record) saveresid
    sca_fit, name(criterion_reghdfe) data("`D'/cov240k.dta") absorb(fe1 fe2 fe3) vce(cluster fe1) conv(reghdfe) ///
        legdir("`L'") expect(record) saveresid
}
if $SCA_FAILS == 0 display "SCA_PASS"
