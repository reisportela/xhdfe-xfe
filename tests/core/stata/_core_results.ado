* Result snapshots for the xhdfe compatibility tests; no estimation is performed.
program define _core_results, sclass
    version 16
    gettoken action 0 : 0
    gettoken label 0 : 0, parse(" ,")
    confirm name `label'
    local slot "${XCORE_SLOT_`label'}"

    if ("`action'" == "drop") {
        if ("`slot'" == "") exit 111
        foreach kind in s m M {
            local fields "${XCORE_`slot'_`kind'}"
            local j 0
            foreach field of local fields {
                local ++j
                if ("`kind'" == "s") capture scalar drop XCORE_`slot'_s`j'
                if ("`kind'" == "M") capture matrix drop XCORE_`slot'_M`j'
                if ("`kind'" == "m") macro drop XCORE_`slot'_m`j'
            }
            macro drop XCORE_`slot'_`kind'
        }
        macro drop XCORE_SLOT_`label'
        exit
    }

    gettoken scope 0 : 0, parse(" ,")
    if (!inlist("`scope'", "e()", "r()")) exit 198
    local ns = substr("`scope'", 1, 1)
    if ("`action'" == "save") {
        if ("`slot'" != "") _core_results drop `label'
        if ("${XCORE_SERIAL}" == "") global XCORE_SERIAL 0
        global XCORE_SERIAL = ${XCORE_SERIAL} + 1
        local slot ${XCORE_SERIAL}
        global XCORE_SLOT_`label' `slot'
        local fields_s : `ns'(scalars)
        local fields_m : `ns'(macros)
        local fields_M : `ns'(matrices)
        foreach kind in s m M {
            global XCORE_`slot'_`kind' "`fields_`kind''"
            local j 0
            foreach field of local fields_`kind' {
                local ++j
                if ("`kind'" == "s") scalar XCORE_`slot'_s`j' = `ns'(`field')
                if ("`kind'" == "M") matrix XCORE_`slot'_M`j' = `ns'(`field')
                if ("`kind'" == "m") global XCORE_`slot'_m`j' `"``ns'(`field')'"'
            }
        }
        exit
    }

    if ("`action'" != "compare" | "`slot'" == "") exit 198
    syntax [, INclude(string) EXclude(string) REVerse TOLerance(real 0) Verbose]
    if (`tolerance' < 0) exit 198
    if ("`include'" != "" & ("`exclude'" != "" | "`reverse'" != "")) exit 198
    local current_s : `ns'(scalars)
    local current_m : `ns'(macros)
    local current_M : `ns'(matrices)
    foreach kind in s m M {
        local saved_`kind' "${XCORE_`slot'_`kind'}"
        local selected_`kind' "`saved_`kind''"
        if ("`reverse'" != "") local selected_`kind' "`current_`kind''"
    }
    foreach option in include exclude {
        local kind
        local words "``option''"
        foreach word of local words {
            if (inlist("`word'", "scalar:", "scalars:")) local kind s
            else if (inlist("`word'", "macro:", "macros:")) local kind m
            else if (inlist("`word'", "matrix:", "matrices:")) local kind M
            else {
                if ("`kind'" == "") exit 198
                local `option'_`kind' "``option'_`kind'' `word'"
            }
        }
    }
    local equivalent : env XHDFE_UPSTREAM_EQUIVALENT
    local task_root : env XHDFE_UPSTREAM_ROOT
    if ("`equivalent'" == "1" & "`include'" == "" & "`ns'" == "e") {
        include "`task_root'/upstream_fields.do"
    }
    local failed 0
    local repeat_contract : env XHDFE_CORE_REPEAT_CONTRACT
    foreach kind in s m M {
        if ("`include'" != "") local selected_`kind' "`include_`kind''"
        local fields : list selected_`kind' - exclude_`kind'
        if ("`equivalent'" == "1" & "`include'" == "" & "`ns'" == "e") {
            local fields : list fields & upstream_fields
        }
        foreach field of local fields {
            local j : list posof "`field'" in saved_`kind'
            local present : list field in current_`kind'
            local equal 0
            // Stata exposes an absent result macro as the empty string.
            if ("`kind'" == "m" & `j' == 0 & !`present') local equal 1
            if (`j' > 0 & `present') {
                if ("`kind'" == "s") {
                    local equal = reldif(`ns'(`field'), XCORE_`slot'_s`j') <= `tolerance'
                    // Same fixed scalar contract as tests/stata/_helpers.do.
                    if ("`repeat_contract'" == "1" & ///
                        strpos(" F rss rmse mss tss tss_within r2 r2_a r2_within r2_a_within ll ll_0 ", " `field' ") & ///
                        !missing(`ns'(`field')) & !missing(XCORE_`slot'_s`j')) {
                        local metric = abs(`ns'(`field')-XCORE_`slot'_s`j') / max(1,abs(XCORE_`slot'_s`j'))
                        if (!`equal' & `metric' <= 1e-8) ///
                            display "CORE_STRICT_UPSTREAM_DIFFERENCE `field' scaled=" %24.17e `metric' " scalar_limit=1e-8"
                        local equal = `metric' <= 1e-8
                    }
                }
                if ("`kind'" == "M") {
                    local equal = mreldif(`ns'(`field'), XCORE_`slot'_M`j') <= `tolerance'
                    local dk_contract : env XHDFE_CORE_DK_CONTRACT
                    if (( !`equal' & "`dk_contract'" == "1" & inlist("`field'", "V", "trim_V")) | ///
                        ("`repeat_contract'" == "1" & "`field'" == "V")) {
                        local strict_equal = `equal'
                        tempname reference observed
                        matrix `reference' = XCORE_`slot'_M`j'
                        matrix `observed' = `ns'(`field')
                        local valid = rowsof(`reference') == rowsof(`observed') & ///
                            colsof(`reference') == colsof(`observed')
                        local maximum 0
                        if (`valid') {
                            forvalues row=1/`=rowsof(`reference')' {
                                forvalues column=1/`=colsof(`reference')' {
                                    local scale = sqrt(`reference'[`row',`row']) * sqrt(`reference'[`column',`column'])
                                    local difference = abs(`reference'[`row',`column'] - `observed'[`row',`column'])
                                    if (missing(`scale') | missing(`difference')) local valid 0
                                    else if (`scale' == 0 & `difference' != 0) local valid 0
                                    else if (`scale' > 0) local maximum = max(`maximum', `difference'/`scale')
                                }
                            }
                        }
                        local equal = `valid' & `maximum' <= 1e-8
                        if (`equal' & !`strict_equal') {
                            display "CORE_STRICT_UPSTREAM_DIFFERENCE `field' diagonal_scaled=" %24.17e `maximum' " published_limit=1e-8"
                        }
                    }
                    if ("`repeat_contract'" == "1" & "`field'" == "b") {
                        local strict_equal = `equal'
                        tempname reference observed
                        matrix `reference' = XCORE_`slot'_M`j'
                        matrix `observed' = `ns'(`field')
                        local equal = rowsof(`reference') == rowsof(`observed') & colsof(`reference') == colsof(`observed')
                        local maximum 0
                        if (`equal') {
                            forvalues column=1/`=colsof(`reference')' {
                                local metric = abs(`reference'[1,`column']-`observed'[1,`column']) / max(1,abs(`reference'[1,`column']))
                                if (missing(`metric')) local equal 0
                                else local maximum = max(`maximum', `metric')
                            }
                        }
                        local equal = `equal' & `maximum' <= 1e-9
                        if (`equal' & !`strict_equal') ///
                            display "CORE_STRICT_UPSTREAM_DIFFERENCE b scaled=" %24.17e `maximum' " published_limit=1e-9"
                    }
                }
                if ("`kind'" == "m") {
                    local a `"${XCORE_`slot'_m`j'}"'
                    local b `"``ns'(`field')'"'
                    local equal = (`"`a'"' == `"`b'"')
                    if (!`equal') {
                        capture confirm number `a'
                        local numeric_a = !_rc
                        capture confirm number `b'
                        if (`numeric_a' & !_rc) local equal = reldif(`a', `b') <= `tolerance'
                    }
                }
            }
            if (!`equal') {
                display as error "CORE_RESULT_DIFFERENCE `kind' `ns'(`field')"
                if (`j' > 0 & `present' & "`kind'" == "s") ///
                    display "CORE_DIAGNOSTIC reldif=" %24.17e reldif(`ns'(`field'), XCORE_`slot'_s`j')
                if (`j' > 0 & `present' & "`kind'" == "M") ///
                    display "CORE_DIAGNOSTIC mreldif=" %24.17e mreldif(`ns'(`field'), XCORE_`slot'_M`j')
                local failed 1
            }
        }
    }
    if (`failed') exit 9
end
