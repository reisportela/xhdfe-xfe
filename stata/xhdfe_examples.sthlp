{smcl}
{* *! version 2.35.0 25nov2026}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{vieweralsosee "xfepout" "help xfepout"}{...}
{title:xhdfe examples with public data}

{pstd}The two standalone examples use datasets supplied with Stata and
run offline after installation. From SSC, use {cmd:ssc install xhdfe, all}
in the directory where you want the example files. From the GitHub release,
install the package first and use {cmd:net get} to download the examples.{p_end}
{asis}
    net get xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata")
    do xhdfe_example_cluster.do
    do xhdfe_example_partial_out.do
{smcl}

{pstd}From an extracted release or SSC ZIP, give its package directory
to {cmd:net install xhdfe, from()} and {cmd:net get xhdfe, from()}.
Use the directory that contains {cmd:xhdfe.pkg}.{p_end}

{pstd}The clustered example requests the Comparable reduced-system
route and automatic finer blocks when the fit is eligible. The
partial-out example shows the regression of residualised columns
and the corresponding {cmd:xhdfe} fit on the original variables.{p_end}

{pstd}Copy a block into the Do-file Editor and run it on its own.
Each {cmd:use} loads the dataset from its public URL and replaces
the dataset in memory.{p_end}

{pstd}{help xhdfe_examples##ols:OLS and a single fixed effect}{p_end}
{pstd}{help xhdfe_examples##panel:Panel effects and multiway clustering}{p_end}
{pstd}{help xhdfe_examples##firm_year:Firm and year effects with cluster VCE}{p_end}
{pstd}{help xhdfe_examples##postestimation:Save effects and use postestimation}{p_end}
{pstd}{help xhdfe_examples##slopes:Heterogeneous slopes}{p_end}
{pstd}{help xhdfe_examples##dk:Time operators and Driscoll-Kraay inference}{p_end}
{pstd}{help xhdfe_examples##probability:Probability weights}{p_end}
{pstd}{help xhdfe_examples##weights:Analytic, frequency and importance weights}{p_end}
{pstd}{help xhdfe_examples##iv:IV and the first stage}{p_end}
{pstd}{help xhdfe_examples##groups:Group outcomes and individual effects}{p_end}
{pstd}{help xhdfe_examples##partial:Partial out effects and retain the sample marker}{p_end}
{pstd}{help xhdfe_examples##caches:Reuse absorption and FE structure}{p_end}
{pstd}{help xhdfe_examples##strict:Strict convergence and a mobility profile}{p_end}
{pstd}{help xhdfe_examples##akm:Connected sample and AKM/KSS components}{p_end}
{pstd}{help xhdfe_examples##akm_inference:AKM standard errors and Andrews-Mikusheva intervals}{p_end}
{pstd}{help xhdfe_examples##gelbach:Gelbach decomposition, table and graph}{p_end}
{pstd}{help xhdfe_examples##gelbach_absorbed:Gelbach decomposition for an absorbed coefficient}{p_end}
{pstd}{help xhdfe_examples##bootstrap:Gelbach pairs bootstrap}{p_end}
{pstd}{help xhdfe_examples##cluster_bootstrap:Gelbach cluster-pairs bootstrap}{p_end}
{pstd}{help xhdfe_examples##cuda:CUDA and explicit weighted routing}{p_end}

{marker ols}{...}
{title:OLS and a single fixed effect}

{pstd}Automobile prices with conventional inference, then with repair-rating effects and cluster VCE.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/auto.dta", clear
    xhdfe price mpg weight if foreign == 0, noabsorb
    xhdfe price mpg weight, absorb(rep78) vce(cluster rep78)
{smcl}

{marker panel}{...}
{title:Panel effects and multiway clustering}

{pstd}The two cluster dimensions are person and year. Add an occupation effect in the second fit.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/nlswork.dta", clear
    xhdfe ln_wage tenure ttl_exp i.union, absorb(idcode year) ///
        vce(cluster idcode year)
    xhdfe ln_wage tenure ttl_exp i.union, absorb(idcode year occ_code) ///
        vce(cluster idcode year)
{smcl}

{marker firm_year}{...}
{title:Firm and year effects with cluster VCE}

{pstd}The Grunfeld investment panel illustrates firm-clustered inference with firm and year effects.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/grunfeld.dta", clear
    xhdfe invest mvalue kstock, absorb(company year) vce(cluster company)
{smcl}

{marker postestimation}{...}
{title:Save effects and use postestimation}

{pstd}The labelled effects and residuals are variables. The predictions distinguish xb, effects and the complete fitted value.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/nlswork.dta", clear
    xhdfe ln_wage tenure i.union, absorb(person_fe=idcode year_fe=year) ///
        vce(cluster idcode) residuals(wage_u)
    predict double slope_fit, xb
    predict double effects_fit, d
    predict double complete_fit, xbd
    predict double slope_se, stdp
    predict double saved_residual, residuals
    predict double fe_plus_residual, dresiduals
    estat summarize
    estat vce
    estat ic
{smcl}

{marker slopes}{...}
{title:Heterogeneous slopes}

{pstd}Give each person an intercept and an experience slope, alongside year effects.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/nlswork.dta", clear
    xhdfe ln_wage tenure union, absorb(idcode##c.ttl_exp year) ///
        vce(cluster idcode)
{smcl}

{marker dk}{...}
{title:Time operators and Driscoll-Kraay inference}

{pstd}The time declaration enables the lag operator. Bandwidth three includes two score lags.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/grunfeld.dta", clear
    xtset company year
    xhdfe invest L.mvalue kstock, absorb(company year) vce(dkraay 3)
{smcl}

{marker probability}{...}
{title:Probability weights}

{pstd}Use the public NHANES weight to demonstrate the pweight syntax and its default robust VCE.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/nhanes2.dta", clear
    xhdfe bpsystol age weight i.female [pw=finalwgt], absorb(region)
{smcl}

{marker weights}{...}
{title:Analytic, frequency and importance weights}

{pstd}These weights illustrate syntax. Frequency weights treat the integer repair rating as a replication count.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/auto.dta", clear
    xhdfe price mpg weight [aw=headroom], absorb(foreign) vce(robust)
    xhdfe price mpg weight [fw=rep78], absorb(foreign)
    xhdfe price mpg weight [iw=headroom], absorb(foreign)
{smcl}

{marker iv}{...}
{title:IV and the first stage}

{pstd}The housing data illustrate a 2SLS fit with region effects. Family income supplies the excluded instrument.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/hsng2.dta", clear
    xhdfe rent pcturban hsngval, absorb(region) endogenous(hsngval) ///
        instruments(faminc) vce(robust) ffirst
    estat firststage
    estat firststage, first level(90)
    matrix list r(first)
{smcl}

{marker groups}{...}
{title:Group outcomes and individual effects}

{pstd}The public teaching fixture has one row per inventor and patent. Citations and funding are patent-level variables.{p_end}
{asis}
    local source "https://raw.githubusercontent.com/reisportela"
    use "`source'/HDFE/main/toy-patents-long.dta", clear
    isid patent_id inventor_id
    xhdfe citations funding, absorb(year inventor_id) group(patent_id) ///
        individual(inventor_id)
    xhdfe citations funding, absorb(year inventor_id) group(patent_id) ///
        individual(inventor_id) aggregation(sum) dofadjustments(exact)
    xhdfe citations funding, absorb(year) group(patent_id)
{smcl}

{marker partial}{...}
{title:Partial out effects and retain the sample marker}

{pstd}The regression on the transformed variables illustrates the slope calculation. Use xhdfe to report absorbed DoF and inference.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/grunfeld.dta", clear
    xfepout invest mvalue kstock, absorb(company year) ///
        generate(within_) sample(in_fit)
    regress within_invest within_mvalue within_kstock if in_fit, noconstant
{smcl}

{marker caches}{...}
{title:Reuse absorption and FE structure}

{pstd}Reuse the transformed columns for an identical CPU model. Reuse the FE structure for a different regressor list.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/grunfeld.dta", clear
    tempfile absorption fe_structure
    xhdfe invest mvalue kstock, absorb(company year) gpubackend(cpu) ///
        absorptioncache("`absorption'") abscachemode(write)
    xhdfe invest mvalue kstock, absorb(company year) gpubackend(cpu) ///
        absorptioncache("`absorption'") abscachemode(read)
    xhdfe invest mvalue kstock, absorb(company year) gpubackend(cpu) ///
        fescache("`fe_structure'") fecachemode(write)
    xhdfe invest mvalue, absorb(company year) gpubackend(cpu) ///
        fescache("`fe_structure'") fecachemode(read)
{smcl}

{pstd}Use a separate file for each cache to reuse both in one fit.{p_end}
{asis}
    tempfile combined_abs combined_fe
    xhdfe invest mvalue kstock, absorb(company year) gpubackend(cpu) ///
        absorptioncache("`combined_abs'") abscachemode(write) ///
        fescache("`combined_fe'") fecachemode(write)
    xhdfe invest mvalue kstock, absorb(company year) gpubackend(cpu) ///
        absorptioncache("`combined_abs'") abscachemode(read) ///
        fescache("`combined_fe'") fecachemode(read)
{smcl}

{marker strict}{...}
{title:Strict convergence and a mobility profile}

{pstd}Strict applies its absolute check. The profile file can guide a later fit of the same structure.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/grunfeld.dta", clear
    tempfile profile
    xhdfe invest mvalue kstock, absorb(company year) ///
        tolerancemode(strict-residual) mobilityprofile mobfile("`profile'")
    display e(converged)
{smcl}

{marker akm}{...}
{title:Connected sample and AKM/KSS components}

{pstd}The project simulation provides outcome y, worker i and firm j.{p_end}
{asis}
    local source "https://raw.githubusercontent.com/reisportela"
    use "`source'/xhdfe-xfe/main/data/felsdvsimul.dta", clear
    xhdfeconnected i j, generate(in_connected)
    xhdfeakm y if in_connected == 1, worker(i) firm(j) noprune ///
        leverages(exact) sigmacentering(personyear) generate(akm)
    display r(kss_var_psi), r(kss_corr)
{smcl}

{marker akm_inference}{...}
{title:AKM standard errors and Andrews-Mikusheva intervals}

{pstd}Industry codes supply the second effect. The interval and SE accompany the reported inference point.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/nlswork.dta", clear
    xhdfeakm ln_wage if inrange(year,78,83), worker(idcode) firm(ind_code) ///
        controls(tenure) leverages(exact) ci threads(4)
    display r(theta_var_psi), r(se_var_psi), r(ci_lb_psi), r(ci_ub_psi)
{smcl}

{marker gelbach}{...}
{title:Gelbach decomposition, table and graph}

{pstd}Decompose the change in the tenure coefficient after adding observed blocks and occupation effects.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/nlswork.dta", clear
    xhdfegelbach ln_wage, x1(tenure) ///
        x2groups("experience = ttl_exp : job = union south not_smsa") ///
        commonfes(year) fes(occ_code) vce(cluster) cluster(idcode) ///
        shares(movement) sampleaudit generate(in_gelbach)
    matrix list r(delta)
    xhdfegelbachetable, focal(tenure) format(markdown)
    xhdfegelbachcoefplot, focal(tenure) nodraw
{smcl}

{marker gelbach_absorbed}{...}
{title:Gelbach decomposition for an absorbed coefficient}

{pstd}Person effects absorb the time-invariant race indicator. The full model imposes a zero
coefficient for this target. Read the point decomposition here. Inference for the absorbed target
is unavailable with this VCE.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/nlswork.dta", clear
    generate byte black = race == 2 if !missing(race)
    xhdfegelbach ln_wage, x1(black tenure) ///
        x2groups("experience = ttl_exp : job = union south not_smsa") ///
        fes(idcode year) absorbedtargets(black) focal(black) ///
        connected(diagnose) connectivityfes(idcode year)
    matrix list r(delta)
    matrix list r(b_full)
{smcl}

{marker bootstrap}{...}
{title:Gelbach pairs bootstrap}

{pstd}Use a small replication count for the example. Each valid draw refits the full decomposition.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/auto.dta", clear
    xhdfegelbachbootstrap price, x1(weight) ///
        x2groups("size = length : efficiency = mpg") ///
        method(pairs) reps(39) seed(42)
    xhdfegelbachetable, panels(all) format(markdown) interval(bootstrap)
{smcl}

{marker cluster_bootstrap}{...}
{title:Gelbach cluster-pairs bootstrap}

{pstd}Resample entire firms. Ten firms and 39 replications illustrate the syntax on a small dataset.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/grunfeld.dta", clear
    xhdfegelbachbootstrap invest, x1(mvalue) x2groups("capital = kstock") ///
        commonfes(year) fes(company) method(cluster_pairs) ///
        bootcluster(company) reps(39) seed(42) bootci(basic)
    xhdfegelbachetable, panels(levels) interval(bootstrap)
{smcl}

{marker cuda}{...}
{title:CUDA and explicit weighted routing}

{pstd}These commands require a CUDA-enabled plugin and an available device. The first fit requests the reduced CUDA tier.{p_end}
{asis}
    use "https://www.stata-press.com/data/r19/grunfeld.dta", clear
    xhdfe invest mvalue kstock, absorb(company year) gpubackend(cuda) ///
        reducedpcgcuda(on) numthreads(4)
    display e(gpu_used), "`e(gpu_status)'", "`e(reduced_pcg_reason)'"
    generate double aw = 1 + kstock/100
    xhdfe invest mvalue kstock [aw=aw], absorb(company year) ///
        gpubackend(cuda) weightedcpuroute(on) numthreads(4)
    display "`e(gpu_status)'"
    xfepout invest mvalue kstock, absorb(company year) generate(cuda_) ///
        gpubackend(cuda)
    display e(gpu_used), "`e(gpu_status)'"
{smcl}
