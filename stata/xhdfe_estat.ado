*! version 2.29.0 14oct2026
program xhdfe_estat, rclass
    version `=cond(c(version)<14, c(version), 13)'
    if ("`e(cmd)'" != "xhdfe") {
        error 301
    }

    gettoken key 0 : 0, parse(", ")
    local lkey = length(`"`key'"')

    if `"`key'"' == substr("summarize", 1, max(2, `lkey')) {
        syntax [anything] , [*] [noheader]
        estat_summ `anything', `options' noheader
    }
    else if `"`key'"' == "vce" {
        vce `0'
    }
    else if `"`key'"' == "ic" {
        syntax, [*]
        estat_default ic, df(`=e(df_m)+1') `options'
    }
    else if `"`key'"' == substr("firststage", 1, max(5, `lkey')) {
        _xhdfe_firststage `0'
    }
    else {
        di as error `"invalid subcommand `key'"'
        exit 321
    }
    return add
end

program define _xhdfe_firststage, rclass
    version 16.0
    // sample() is used internally by xhdfe before a nosample fit returns.
    syntax [, FIRST Level(cilevel) SAMPLE(varname numeric)]
    if ("`e(endogenous)'" == "") {
        di as err "estat firststage requires an xhdfe IV regression"
        exit 321
    }
    tempname results gpuused held
    capture confirm matrix e(first)
    local cached = !_rc
    if (`cached' & "`first'" == "") {
        matrix `results' = e(first)
        matrix `gpuused' = e(first_gpu_used)
    }
    else {
        tempvar touse
        if ("`sample'" == "") quietly generate byte `touse' = e(sample)
        else quietly generate byte `touse' = `sample'
        quietly count if `touse'
        if (r(N) == 0) {
            di as err "the IV estimation sample is unavailable; specify first or ffirst at estimation time when using nosample"
            exit 498
        }
        local endog "`e(endogenous)'"
        local exog "`e(exogenous)'"
        local inst "`e(instruments)'"
        local fsopts `"`e(firststage_options)'"'
        local wopt
        if ("`e(wtype)'" != "") local wopt "[`e(wtype)'`e(wexp)']"
        local exogopt
        if ("`exog'" != "") local exogopt "exogenous(`exog')"
        local output = cond(c(noisily), "noisily", "quietly")
        _estimates hold `held', copy restore
        capture `output' _xhdfe_firststage_fit, sample(`touse') ///
            endogenous(`endog') instruments(`inst') `exogopt' ///
            fitoptions(`fsopts') weightoptions(`wopt') `first' level(`level')
        local rc = _rc
        if (!`rc') {
            matrix `results' = r(first)
            matrix `gpuused' = r(first_gpu_used)
        }
        _estimates unhold `held'
        if (`rc') exit `rc'
    }
    _xhdfe_firststage_display, results(`results')
    return matrix first = `results'
    return matrix first_gpu_used = `gpuused'
end

program define _xhdfe_firststage_fit, rclass
    version 16.0
    syntax, SAMPLE(varname numeric) ENDOGenous(varlist numeric) ///
        INSTRuments(varlist numeric) [EXOGenous(varlist numeric) ///
        FITOptions(string asis) WEIGHTOptions(string asis) FIRST Level(cilevel)]
    tempname results gpuused rssu
    local nendog : word count `endogenous'
    matrix `results' = J(5, `nendog', .)
    matrix rownames `results' = pr2 F df df_r pvalue
    matrix colnames `results' = `endogenous'
    matrix `gpuused' = J(2, `nendog', .)
    matrix rownames `gpuused' = unrestricted restricted
    matrix colnames `gpuused' = `endogenous'
    local j 0
    foreach d of local endogenous {
        local ++j
        quietly xhdfe `d' `exogenous' `instruments' if `sample' `weightoptions', `fitoptions'
        quietly assert e(sample) == `sample'
        scalar `rssu' = e(rss)
        matrix `gpuused'[1, `j'] = e(gpu_used)
        if ("`first'" != "") {
            di as txt _n "First-stage regression of `d'"
            xhdfe, level(`level')
        }
        // Stata omits redundant restrictions and reports the effective rank.
        // Undefined tests (e.g. zero residual df) remain missing.
        capture quietly testparm `instruments'
        if (!_rc) {
            matrix `results'[2, `j'] = r(F)
            matrix `results'[3, `j'] = r(df)
            matrix `results'[4, `j'] = r(df_r)
            matrix `results'[5, `j'] = r(p)
        }
        quietly xhdfe `d' `exogenous' if `sample' `weightoptions', `fitoptions'
        quietly assert e(sample) == `sample'
        matrix `gpuused'[2, `j'] = e(gpu_used)
        if (!missing(e(rss), `rssu') & e(rss) > 0) {
            matrix `results'[1, `j'] = max(0, min(1, 1 - `rssu' / e(rss)))
        }
    }
    return matrix first = `results'
    return matrix first_gpu_used = `gpuused'
end

program define _xhdfe_firststage_display
    version 16.0
    syntax, RESULTS(name)
    local endog : colnames `results'
    di as txt _n "First-stage tests of excluded instruments"
    di as txt "{hline 79}"
    di as txt %16s "Variable" _col(19) %10s "Partial R2" ///
        _col(32) %10s "F" _col(45) %7s "df1" _col(55) %9s "df2" _col(68) %10s "Prob > F"
    di as txt "{hline 79}"
    local j 0
    foreach d of local endog {
        local ++j
        di as txt %16s abbrev("`d'", 16) as res ///
            _col(19) %10.6f `results'[1,`j'] _col(32) %10.3f `results'[2,`j'] ///
            _col(45) %7.0f `results'[3,`j'] _col(55) %9.0f `results'[4,`j'] ///
            _col(68) %10.4f `results'[5,`j']
    }
    di as txt "{hline 79}"
    di as txt "Excluded instruments: `e(instruments)'"
    local clusters
    if ("`e(clustvar)'" != "") local clusters " (`e(clustvar)')"
    di as txt "VCE: `e(vce)'`clusters'"
    if (colsof(`results') > 1) {
        di as txt "Each F-test assesses the excluded instruments in one first-stage equation."
    }
end
