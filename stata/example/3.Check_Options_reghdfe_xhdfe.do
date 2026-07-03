// Catalog of xhdfe uses and options (based on xhdfe 6.12.5 help, 27Dec2023)
// Sources:
//   /PATH/TO/ado/plus/r/xhdfe.sthlp
//   /PATH/TO/ado/plus/r/xhdfe_programming.sthlp
//
// Set RUN_EXAMPLES or RUN_undocumented to 1 to execute. Some blocks expect data in data/ or webuse.

// MAJOR issues
// # 1. MAJOR ISSUE in 'Group and individual fixed effects'     == CORRECTED
// # 2. Problem with 'test' after xhdfe -->> EMPTY VALUES       == CORRECTED
// # 3. Introduce the i AND # operators inside the absorb       == CORRECTED
// # 4. No 'estat summarize' -->> last estimates not found      == CORRECTED

// MINOR issues
// # 4. make clear that for the option residuals we have to define a name for the variable -->> I agree, so no change needed
//      'xhdfe price weight length, absorb(rep78) residuals(tmp_resid)
// # 5. missing options:
//      technique, iterate, fastregress, nosample, poolsize, compact, noheader, notable, nofootnote, noomitted, verbose, timeit, version, nowarn, nopartialout, noregress, keepmata, noabsorb2

clear all
discard

set more off

adopath + "/PATH/TO/xhdfe/stata"
which xhdfe
*h xhdfe

cd /PATH/TO/xhdfe/stata/example

capture log close
log using output/3.Check_Options_reghdfe_xhdfe.txt, text replace

local RUN_EXAMPLES = 1
local RUN_UNDOCUMENTED = 0
global DATA_DIR "/PATH/TO/xhdfe/test/tests_xhdfe/data"

// Optional install (uncomment if needed)
// cap which xhdfe
// if _rc {
//     ssc install ftools, replace
//     ssc install xhdfe, replace
// }

// ------------------------------------------------------------
// 1) Basic OLS usage (no fixed effects)
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {
    sysuse auto, clear
    xhdfe price weight length
    xhdfe price c.weight##c.length i.foreign
    xhdfe price weight length if foreign == 1
}

// ------------------------------------------------------------
// 2) Standard fixed effects with absorb()
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {

    sysuse auto, clear
    xhdfe price weight length, absorb(i.rep78)
    // One-way and multi-way fixed effects
    xhdfe price weight length, absorb(rep78)
    xhdfe price weight length, absorb(rep78 turn)
    xhdfe price weight length, absorb(i.rep78)

    // Absorb interactions and heterogeneous slopes
xhdfe price weight length, absorb(i.rep78#i.foreign)
    // Slope-only (use carefully; slower convergence)
xhdfe price weight length, absorb(rep78#c.weight)
xhdfe price weight length, absorb(rep78##c.weight)

//xhdfe price weight length, absorb(rep78##c.(weight length))                   -->> NOT YET IMPLEMENTED
//xhdfe price weight length, absorb(rep78#foreign#turn##c.(weight length))      -->> NOT YET IMPLEMENTED

    // Save fixed effects
clear all
discard
adopath + "/PATH/TO/xhdfe/stata"
    sysuse auto, clear
    xhdfe price weight length, absorb(rep78 turn) savefes(fe_)
    xhdfe price weight length, absorb(rep78 turn, savefe)
    xhdfe price weight length, absorb(FE_rep=rep78 FE_turn=turn)
}

// ------------------------------------------------------------
// 3) VCE options (unadjusted, robust, cluster, multi-way cluster)
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {
    sysuse auto, clear
    xhdfe price weight length, absorb(rep78) vce(unadjusted)
    xhdfe price weight length, absorb(rep78) vce(robust)
    xhdfe price weight length, absorb(rep78) vce(cluster rep78)
    xhdfe price weight length, absorb(rep78 turn) vce(cluster rep78 turn)
    xhdfe price weight length, absorb(rep78 turn) vce(cluster rep78#turn)

    // cluster() is accepted as an alias of vce(cluster ...)
    xhdfe price weight length, absorb(rep78) cluster(rep78)
}

// ------------------------------------------------------------
// 4) Weights (fw, aw, pw)
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {
    sysuse auto, clear
    gen double aw = runiform() + 1
    gen double pw = runiform() + 1
    gen int fw = ceil(runiform() * 5)
    xhdfe price weight length [aw=aw], absorb(rep78)
    xhdfe price weight length [pw=pw], absorb(rep78)
    xhdfe price weight length [fw=fw], absorb(rep78)
}

// ------------------------------------------------------------
// 5) Residuals and predict
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {
    sysuse auto, clear
// -->> tHE FOLLOWING LINE DOES NOT WORK; WE HAVE TO define a name for the variable with the residuals -->> I AGREE IT SHOULD BE LIKE THIS
//**xx    xhdfe price weight length, absorb(rep78) residuals
    xhdfe price weight length, absorb(rep78) residuals(tmp_resid)
    predict double xb_hat, xb
    predict double stdp_hat, stdp
    predict double resid_hat, residuals
    predict double score_hat, score
    predict double d_hat, d
    predict double xbd_hat, xbd

    xhdfe price weight length, absorb(rep78) residuals(resid_custom)
}

// ------------------------------------------------------------
// 6) Group and individual fixed effects
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {
    use "https://github.com/sergiocorreia/reghdfe/raw/master/test/part2/toy-patents-long.dta", clear
    // Data should uniquely identify group+individual
    // isid patent_id inventor_id

    // Individual and group fixed effects
    xhdfe citations funding, absorb(inventor_id) group(patent_id) individual(inventor_id)

    // Individual + group + standard fixed effects
    xhdfe citations funding, absorb(year inventor_id) group(patent_id) individual(inventor_id)

    // Aggregation method for individual effects within groups
    xhdfe citations funding, absorb(inventor_id) group(patent_id) individual(inventor_id) aggregation(sum)
    xhdfe citations funding, absorb(inventor_id) group(patent_id) individual(inventor_id) aggregation(mean)

    // Use one observation per group (group only)
    xhdfe citations funding, group(patent_id)
    xhdfe citations funding, absorb(year) group(patent_id)
}

// ------------------------------------------------------------
// 7) Degrees-of-freedom adjustments + groupvar()
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {
    webuse nlswork, clear
    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year)
    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) dofadjustments(all)
    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) dofadjustments(none)
    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) dofadjustments(firstpair)
    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) dofadjustments(pairwise clusters continuous)
    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) groupvar(mobility_group)
}

// ------------------------------------------------------------
// 8) Optimization options -->> NOT YET IMPLEMENTED (keepsingletons is implemented; see below)
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {
    webuse nlswork, clear

    // Technique selection

//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) technique(map)
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) technique(lsmr) preconditioner(block_diagonal)
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) technique(lsqr) preconditioner(diagonal)
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) technique(lsmr) preconditioner(none)
    
    // technique(gt) and prune are currently disabled

    // MAP acceleration and transforms
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) technique(map) acceleration(none) transform(kaczmarz)
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) technique(map) acceleration(cg) transform(symmetric_kaczmarz)
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) technique(map) acceleration(sd) transform(cimmino)
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) technique(map) acceleration(aitken)

    // Convergence controls
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) tolerance(1e-6) iterate(1000)		// option iterate() not allowed
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) iterate(.)				// option iterate() not allowed

    // Misc optimization flags
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) fastregress		// option fastregress not allowed

    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) keepsingletons

//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) nosample		// option nosample not allowed
}

// ------------------------------------------------------------
// 10) Memory usage
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {
    webuse nlswork, clear
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) poolsize(20)		// option poolsize() not allowed
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) compact		// option compact not allowed
}

// ------------------------------------------------------------
// 11) Reporting and display options
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {
    sysuse auto, clear
    xhdfe price weight length, absorb(rep78) level(90)
//**xx    xhdfe price weight length, absorb(rep78) noheader		// option noheader not allowed
//**xx    xhdfe price weight length, absorb(rep78) notable		// option notable not allowed
//**xx    xhdfe price weight length, absorb(rep78) nofootnote		// option nofootnote not allowed
    xhdfe price weight length, absorb(rep78) noconstant

    // display_options are passed through (see: help display_options)
//**xx    xhdfe price weight length, absorb(rep78) noomitted noempty vsquish cformat(%9.3f)		// option noomitted not allowed
}

// ------------------------------------------------------------
// 12) Diagnostics and versioning  -->> NOT YET IMPLEMENTED
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {
    webuse nlswork, clear
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) verbose(1)		// option verbose() not allowed
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) timeit		// option timeit not allowed
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) version(3)		// option version() not allowed
//**xx    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year) version(5) nowarn	// option version() not allowed + option nowarn not allowed
}

// ------------------------------------------------------------
// 13) Postestimation (estat, predict, test)
// ------------------------------------------------------------
if (`RUN_EXAMPLES') {

    webuse nlswork, clear
    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year)
    test ttl_exp
    test ttl_exp = 0.09
    testparm c.ttl_exp##c.ttl_exp

    predict double xb_hat, xb
    
    webuse nlswork, clear
    xhdfe ln_wage c.ttl_exp##c.ttl_exp, absorb(idcode year)
    estat summarize

    // suest is not supported; see help for manual workarounds.
}

// ------------------------------------------------------------
// 14) Undocumented/programmer options (see xhdfe_programming)  -->> NOT YET IMPLEMENTED
// ------------------------------------------------------------
if (`RUN_UNDOCUMENTED') {
    webuse nlswork, clear
//**xx    xhdfe ln_wage c.ttl_exp, absorb(idcode year) nopartialout	// option nopartialout not allowed
//**xx    xhdfe ln_wage c.ttl_exp, absorb(idcode year) noregress	// option noregress not allowed
//**xx    xhdfe ln_wage c.ttl_exp, absorb(idcode year) keepmata		// option keepmata not allowed
    // Legacy/ignored option (kept for backward compatibility)
//**xx    xhdfe ln_wage c.ttl_exp, absorb(idcode year) noabsorb2	// option noabsorb2 not allowed
}

// End of catalog

log close
