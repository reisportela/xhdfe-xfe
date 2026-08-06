*! version 1.0.0  06aug2026
*! Multi-format reporting for the most recent xhdfegelbach result.

program define xhdfegelbachetable
    version 14.0

    * Capture r() before parsing or emitting anything.
    tempname D SE TOT BBASE BFULL BASECOV COV TOTCOV COVDB COVTB
    tempname BDCI BTCI BBCI BFCI BSB BSM BFS BTS
    capture confirm matrix r(delta)
    if (_rc) {
        di as err "xhdfegelbachetable must immediately follow xhdfegelbach or xhdfegelbachbootstrap"
        exit 301
    }
    matrix `D' = r(delta)
    matrix `SE' = r(se)
    matrix `TOT' = r(total)
    matrix `BBASE' = r(b_base)
    matrix `BFULL' = r(b_full)
    matrix `COV' = r(cov)
    matrix `TOTCOV' = r(total_cov)
    matrix `COVDB' = r(cov_delta_bbase)
    matrix `COVTB' = r(cov_total_bbase)
    capture confirm matrix r(base_cov)
    local has_basecov = (_rc == 0)
    if (`has_basecov') matrix `BASECOV' = r(base_cov)
    capture confirm matrix r(bootstrap_delta_ci)
    local has_bootstrap = (_rc == 0)
    if (`has_bootstrap') {
        matrix `BDCI' = r(bootstrap_delta_ci)
        matrix `BTCI' = r(bootstrap_total_ci)
        matrix `BBCI' = r(bootstrap_base_ci)
        matrix `BFCI' = r(bootstrap_full_ci)
        matrix `BSB' = r(bootstrap_share_base_ci)
        matrix `BSM' = r(bootstrap_share_movement_ci)
        matrix `BFS' = r(bootstrap_full_share_base_ci)
        matrix `BTS' = r(bootstrap_total_share_base_ci)
    }
    local groups "`r(groups)'"
    local x1_names "`r(x1_names)'"
    local focal_names "`r(focal_names)'"
    local point_vce "`r(vce)'"
    if ("`point_vce'" == "") local point_vce "`r(point_vce)'"
    local point_notes `"`r(notes)'"'
    local identity_status "`r(identity_status)'"
    local bootstrap_method "`r(method)'"
    local bootstrap_ci "`r(ci_method)'"
    local bootstrap_valid = r(reps_valid)
    local bootstrap_requested = r(reps_requested)
    local stored_level = r(conf_level)
    local stored_sharetol = r(share_tol)
    local stored_sharetmin = r(share_t_min)
    if (missing(`stored_sharetmin')) local stored_sharetmin 3

    syntax [, PANELS(string) FORMAT(string) FOCAL(string) ///
        KEEP(string asis) DROP(string asis) EXACT ///
        LABELS(string) NOOTHER DIGITS(integer 3) ///
        CAPTION(string asis) NOTES(string asis) ///
        Level(cilevel) INTERVAL(string) ///
        SHARETOL(real -1) SHARETMIN(real -1) ///
        SAVING(string asis) REPLACE ]

    local saving : subinstr local saving `"""' "", all
    local keep : subinstr local keep `"""' "", all
    local drop : subinstr local drop `"""' "", all
    local caption : subinstr local caption `"""' "", all
    local notes : subinstr local notes `"""' "", all
    if (`digits' < 0 | `digits' > 12) {
        di as err "digits() must lie between 0 and 12"
        exit 198
    }
    local format = lower(trim("`format'"))
    if ("`format'" == "") local format "display"
    if (!inlist("`format'", "display", "markdown", "md", "latex", ///
        "tex", "html", "csv")) {
        di as err "format() must be display, markdown/md, latex/tex, html, or csv"
        exit 198
    }
    if ("`format'" == "md") local format "markdown"
    if ("`format'" == "tex") local format "latex"
    local interval = lower(trim("`interval'"))
    if ("`interval'" == "") local interval "auto"
    if (!inlist("`interval'", "auto", "normal", "bootstrap")) {
        di as err "interval() must be auto, normal or bootstrap"
        exit 198
    }
    if (`sharetol' == -1) {
        if (missing(`stored_sharetol')) local stored_sharetol 1e-12
    }
    else if (missing(`sharetol') | `sharetol' < 0) {
        di as err "sharetol() must be finite and nonnegative"
        exit 198
    }
    else local stored_sharetol = `sharetol'
    if (`sharetmin' == -1) {
        if (missing(`stored_sharetmin')) local stored_sharetmin 3
    }
    else if (missing(`sharetmin') | `sharetmin' < 0) {
        di as err "sharetmin() must be finite and nonnegative"
        exit 198
    }
    else local stored_sharetmin = `sharetmin'
    local use_bootstrap = `has_bootstrap'
    if ("`interval'" == "normal") local use_bootstrap 0
    if ("`interval'" == "bootstrap" & !`has_bootstrap') {
        di as err "interval(bootstrap) requires xhdfegelbachbootstrap results"
        exit 198
    }
    local panels = lower(trim("`panels'"))
    if ("`panels'" == "") local panels "all"
    if (!inlist("`panels'", "all", "levels", "share_base", ///
        "share_full", "share_movement", "share_explained")) {
        di as err "panels() must be all, levels, share_base/share_full, or share_movement/share_explained"
        exit 198
    }
    if ("`panels'" == "share_full") local panels "share_base"
    if ("`panels'" == "share_explained") local panels "share_movement"
    local panel_list "`panels'"
    if ("`panels'" == "all") ///
        local panel_list "levels share_base share_movement"

    if ("`focal'" == "") local focal "`focal_names'"
    if ("`focal'" == "") local focal "`x1_names'"
    local invalid_focal : list focal - x1_names
    if ("`invalid_focal'" != "") {
        di as err "focal() contains unknown x1 name(s): `invalid_focal'"
        exit 198
    }
    local duplicate_focal : list dups focal
    if ("`duplicate_focal'" != "") {
        di as err "focal() entries must be unique"
        exit 198
    }

    * Parse labels("group = Display label : group2 = Other label").
    foreach group of local groups {
        local xgel_label_`group' "`group'"
    }
    if (`"`labels'"' != "") {
        tokenize `"`labels'"', parse(":")
        while (`"`1'"' != "") {
            if (`"`1'"' != ":") {
                gettoken __label_group __label_rest : 1, parse("=")
                gettoken __label_eq __label_value : __label_rest, parse("=")
                local __label_group = trim(`"`__label_group'"')
                local __label_value = trim(`"`__label_value'"')
                local __known : list posof "`__label_group'" in groups
                if (`__known' == 0 | `"`__label_value'"' == "") {
                    di as err "labels() entries must be existing_group = display label"
                    exit 198
                }
                local xgel_label_`__label_group' `"`__label_value'"'
            }
            macro shift
        }
    }
    local selected_groups
    local omitted_groups
    foreach group of local groups {
        local selected 1
        if (`"`keep'"' != "") {
            if ("`exact'" != "") {
                local selected : list posof "`group'" in keep
            }
            else local selected = regexm("`group'", `"`keep'"')
        }
        if (`selected' & `"`drop'"' != "") {
            if ("`exact'" != "") {
                local dropped : list posof "`group'" in drop
                if (`dropped') local selected 0
            }
            else if (regexm("`group'", `"`drop'"')) local selected 0
        }
        if (`selected') local selected_groups "`selected_groups' `group'"
        else local omitted_groups "`omitted_groups' `group'"
    }
    local selected_groups = trim("`selected_groups'")
    local omitted_groups = trim("`omitted_groups'")
    local filtered_identity_warning 0
    if ("`omitted_groups'" != "" & "`noother'" != "") {
        local filtered_identity_warning 1
        di as err "warning: noother with keep()/drop() omits components; displayed rows do not preserve the Gelbach accounting identity"
    }

    tempname output
    local write_file = (`"`saving'"' != "")
    local caption_output `"`caption'"'
    if ("`format'" == "markdown") {
        local caption_output : subinstr local caption_output "|" "\|", all
    }
    else if ("`format'" == "latex") {
        local slash = char(92)
        foreach symbol in "&" "%" "$" "#" "_" "{" "}" {
            local caption_output : subinstr local caption_output ///
                "`symbol'" "`slash'`symbol'", all
        }
    }
    else if ("`format'" == "html") {
        local caption_output : subinstr local caption_output "&" "&amp;", all
        local caption_output : subinstr local caption_output "<" "&lt;", all
        local caption_output : subinstr local caption_output ">" "&gt;", all
    }
    if (`write_file') {
        local replace_option
        if ("`replace'" != "") local replace_option "replace"
        file open `output' using `"`saving'"', write text `replace_option'
    }

    if (`write_file') {
        if ("`format'" == "markdown") {
            if (`"`caption_output'"' != "") ///
                file write `output' "**`caption_output'**" _n _n
            file write `output' ///
                "| Panel | Coefficient | Component | Estimate | Std. error | CI low | CI high | CI method |" _n
            file write `output' "|---|---|---|---:|---:|---:|---:|---|" _n
        }
        else if ("`format'" == "latex") {
            file write `output' "\begin{table}[htbp]" _n "\centering" _n
            if (`"`caption_output'"' != "") ///
                file write `output' "\caption{`caption_output'}" _n
            file write `output' "\begin{tabular}{lllrrrrl}" _n "\hline" _n
            file write `output' ///
                "Panel & Coefficient & Component & Estimate & SE & CI low & CI high & Method \\" _n "\hline" _n
        }
        else if ("`format'" == "html") {
            file write `output' `"<table class="xhdfe-gelbach">"' _n
            if (`"`caption_output'"' != "") ///
                file write `output' "<caption>`caption_output'</caption>" _n
            file write `output' ///
                "<thead><tr><th>Panel</th><th>Coefficient</th><th>Component</th><th>Estimate</th><th>Std. error</th><th>CI low</th><th>CI high</th><th>CI method</th></tr></thead><tbody>" _n
        }
        else if ("`format'" == "csv") {
            file write `output' ///
                "panel,coefficient,component,estimate,std_error,conf_low,conf_high,confidence_method" _n
        }
    }
    else if ("`format'" != "display") {
        di as txt "note: saving() was not supplied; markup is printed to the Results window"
    }

    local zcrit = invnormal(0.5 + `level' / 200)
    local k1 = rowsof(`D')
    local G = colsof(`D')
    local p = colsof(`BBASE')

    foreach panel of local panel_list {
        foreach coef of local focal {
            local r : list posof "`coef'" in x1_names
            if (`r' == 0) continue

            if ("`panel'" == "levels" | "`panel'" == "share_base") {
                if ("`panel'" == "levels") {
                    local estimate = `BBASE'[1, `r']
                    local se .
                    local low .
                    local high .
                    local method_label "not_available"
                    if (`use_bootstrap') {
                        local se = `BBCI'[3, `r']
                        local low = `BBCI'[1, `r']
                        local high = `BBCI'[2, `r']
                        local method_label "bootstrap_`bootstrap_ci'"
                    }
                    else if (`has_basecov') {
                        local se = sqrt(max(0, `BASECOV'[`r', `r']))
                        local low = `estimate' - `zcrit' * `se'
                        local high = `estimate' + `zcrit' * `se'
                        local method_label "normal_delta"
                    }
                }
                else {
                    local estimate 1
                    local se 0
                    local low 1
                    local high 1
                    local method_label "identity"
                }
                __xgel_etable_emit_20260728 `"`format'"' `"`output'"' ///
                    `"`panel'"' `"`coef'"' `"base_model"' ///
                    `estimate' `se' `low' `high' `"`method_label'"' ///
                    `digits' `write_file'
            }

            forvalues g = 1/`G' {
                local group : word `g' of `groups'
                local selected : list posof "`group'" in selected_groups
                if (!`selected') continue
                local flat = (`g' - 1) * `k1' + `r'
                local estimate = `D'[`r', `g']
                local se = `SE'[`r', `g']
                local low = `estimate' - `zcrit' * `se'
                local high = `estimate' + `zcrit' * `se'
                local method_label "normal_delta"
                if ("`panel'" == "levels" & `use_bootstrap') {
                    local se = `BDCI'[3, `flat']
                    local low = `BDCI'[1, `flat']
                    local high = `BDCI'[2, `flat']
                    local method_label "bootstrap_`bootstrap_ci'"
                }
                else if ("`panel'" == "share_base") {
                    local raw_estimate = `estimate'
                    local denominator = `BBASE'[1, `r']
                    if (missing(`denominator') | abs(`denominator') <= ///
                        cond(missing(`stored_sharetol'), 1e-12, `stored_sharetol')) {
                        local estimate .
                    }
                    else local estimate = `raw_estimate' / `denominator'
                    local se .
                    local low .
                    local high .
                    local method_label "not_available"
                    if (`use_bootstrap') {
                        local se = `BSB'[3, `flat']
                        local low = `BSB'[1, `flat']
                        local high = `BSB'[2, `flat']
                        local method_label "bootstrap_`bootstrap_ci'"
                    }
                    else if (!missing(`estimate') & `has_basecov') {
                        tempname component_share_var
                        scalar `component_share_var' = ///
                            `COV'[`flat', `flat'] / (`denominator'^2) + ///
                            (`raw_estimate'^2 * `BASECOV'[`r', `r']) / ///
                                (`denominator'^4) - ///
                            (2 * `raw_estimate' * `COVDB'[`flat', `r']) / ///
                                (`denominator'^3)
                        local se = sqrt(max(0, scalar(`component_share_var')))
                        local low = `estimate' - `zcrit' * `se'
                        local high = `estimate' + `zcrit' * `se'
                        local method_label "normal_delta"
                        local denominator_se = ///
                            sqrt(max(0, `BASECOV'[`r', `r']))
                        local denominator_t = cond(`denominator_se' > 0, ///
                            abs(`denominator') / `denominator_se', c(maxdouble))
                        if (`denominator_t' < `stored_sharetmin') ///
                            local method_label ///
                                "normal_delta_diagnostic_only_weak_denominator"
                    }
                }
                else if ("`panel'" == "share_movement") {
                    local raw_estimate = `estimate'
                    local denominator = `TOT'[`r', 1]
                    if (missing(`denominator') | abs(`denominator') <= ///
                        cond(missing(`stored_sharetol'), 1e-12, `stored_sharetol')) {
                        local estimate .
                    }
                    else local estimate = `raw_estimate' / `denominator'
                    local se .
                    local low .
                    local high .
                    local method_label "not_available"
                    if (`use_bootstrap') {
                        local se = `BSM'[3, `flat']
                        local low = `BSM'[1, `flat']
                        local high = `BSM'[2, `flat']
                        local method_label "bootstrap_`bootstrap_ci'"
                    }
                    else if (!missing(`estimate')) {
                        tempname component_share_var
                        scalar `component_share_var' = 0
                        forvalues h = 1/`G' {
                            local gh = -`raw_estimate' / (`denominator'^2)
                            if (`h' == `g') ///
                                local gh = `gh' + 1 / `denominator'
                            local hi = (`h' - 1) * `k1' + `r'
                            forvalues l = 1/`G' {
                                local gl = ///
                                    -`raw_estimate' / (`denominator'^2)
                                if (`l' == `g') ///
                                    local gl = `gl' + 1 / `denominator'
                                local li = (`l' - 1) * `k1' + `r'
                                scalar `component_share_var' = ///
                                    scalar(`component_share_var') + ///
                                    `gh' * `COV'[`hi', `li'] * `gl'
                            }
                        }
                        local se = sqrt(max(0, scalar(`component_share_var')))
                        local low = `estimate' - `zcrit' * `se'
                        local high = `estimate' + `zcrit' * `se'
                        local method_label "normal_delta"
                        local denominator_se = ///
                            sqrt(max(0, `TOTCOV'[`r', `r']))
                        local denominator_t = cond(`denominator_se' > 0, ///
                            abs(`denominator') / `denominator_se', c(maxdouble))
                        if (`denominator_t' < `stored_sharetmin') ///
                            local method_label ///
                                "normal_delta_diagnostic_only_weak_denominator"
                    }
                }
                local display_label `"`xgel_label_`group''"'
                __xgel_etable_emit_20260728 `"`format'"' `"`output'"' ///
                    `"`panel'"' `"`coef'"' `"`display_label'"' ///
                    `estimate' `se' `low' `high' `"`method_label'"' ///
                    `digits' `write_file'
            }

            if ("`omitted_groups'" != "" & "`noother'" == "") {
                tempname other_var
                local other_estimate 0
                scalar `other_var' = 0
                foreach omitted_group of local omitted_groups {
                    local og : list posof "`omitted_group'" in groups
                    local other_estimate = ///
                        `other_estimate' + `D'[`r', `og']
                    local oi = (`og' - 1) * `k1' + `r'
                    foreach omitted_group2 of local omitted_groups {
                        local oh : list posof "`omitted_group2'" in groups
                        local oj = (`oh' - 1) * `k1' + `r'
                        scalar `other_var' = scalar(`other_var') + ///
                            `COV'[`oi', `oj']
                    }
                }
                local estimate = `other_estimate'
                local se = sqrt(max(0, scalar(`other_var')))
                local low = `estimate' - `zcrit' * `se'
                local high = `estimate' + `zcrit' * `se'
                local method_label "normal_delta"
                if ("`panel'" == "share_base") {
                    local denominator = `BBASE'[1, `r']
                    if (missing(`denominator') | abs(`denominator') <= ///
                        cond(missing(`stored_sharetol'), 1e-12, `stored_sharetol')) {
                        local estimate .
                        local se .
                        local low .
                        local high .
                    }
                    else {
                        tempname other_cross other_share_var
                        scalar `other_cross' = 0
                        foreach omitted_group of local omitted_groups {
                            local og : list posof "`omitted_group'" in groups
                            local oi = (`og' - 1) * `k1' + `r'
                            scalar `other_cross' = scalar(`other_cross') + ///
                                `COVDB'[`oi', `r']
                        }
                        scalar `other_share_var' = ///
                            scalar(`other_var') / (`denominator'^2) + ///
                            (`other_estimate'^2 * `BASECOV'[`r', `r']) / ///
                                (`denominator'^4) - ///
                            (2 * `other_estimate' * scalar(`other_cross')) / ///
                                (`denominator'^3)
                        local estimate = `other_estimate' / `denominator'
                        local se = sqrt(max(0, scalar(`other_share_var')))
                        local low = `estimate' - `zcrit' * `se'
                        local high = `estimate' + `zcrit' * `se'
                        local denominator_se = ///
                            sqrt(max(0, `BASECOV'[`r', `r']))
                        local denominator_t = cond(`denominator_se' > 0, ///
                            abs(`denominator') / `denominator_se', c(maxdouble))
                        if (`denominator_t' < `stored_sharetmin') ///
                            local method_label ///
                                "normal_delta_diagnostic_only_weak_denominator"
                    }
                }
                else if ("`panel'" == "share_movement") {
                    local denominator = `TOT'[`r', 1]
                    if (missing(`denominator') | abs(`denominator') <= ///
                        cond(missing(`stored_sharetol'), 1e-12, `stored_sharetol')) {
                        local estimate .
                        local se .
                        local low .
                        local high .
                    }
                    else {
                        tempname other_share_var
                        scalar `other_share_var' = 0
                        forvalues g = 1/`G' {
                            local gname : word `g' of `groups'
                            local wg : list posof "`gname'" in omitted_groups
                            local gradg = (`wg' > 0) / `denominator' - ///
                                `other_estimate' / (`denominator'^2)
                            local gi = (`g' - 1) * `k1' + `r'
                            forvalues h = 1/`G' {
                                local hname : word `h' of `groups'
                                local wh : list posof "`hname'" in omitted_groups
                                local gradh = (`wh' > 0) / `denominator' - ///
                                    `other_estimate' / (`denominator'^2)
                                local hi = (`h' - 1) * `k1' + `r'
                                scalar `other_share_var' = ///
                                    scalar(`other_share_var') + ///
                                    `gradg' * `COV'[`gi', `hi'] * `gradh'
                            }
                        }
                        local estimate = `other_estimate' / `denominator'
                        local se = sqrt(max(0, scalar(`other_share_var')))
                        local low = `estimate' - `zcrit' * `se'
                        local high = `estimate' + `zcrit' * `se'
                        local denominator_se = ///
                            sqrt(max(0, `TOTCOV'[`r', `r']))
                        local denominator_t = cond(`denominator_se' > 0, ///
                            abs(`denominator') / `denominator_se', c(maxdouble))
                        if (`denominator_t' < `stored_sharetmin') ///
                            local method_label ///
                                "normal_delta_diagnostic_only_weak_denominator"
                    }
                }

                * Reconcile the new residual row at the requested display
                * precision so the printed component rows close exactly to
                * the printed total.  Variance calculations above continue
                * to use the unrounded omitted-component aggregate.
                local display_unit = 10^(-`digits')
                local displayed_selected 0
                local reconcile_other 1
                foreach selected_group of local selected_groups {
                    local sg : list posof "`selected_group'" in groups
                    local selected_estimate = `D'[`r', `sg']
                    if ("`panel'" == "share_base") {
                        local selected_denominator = `BBASE'[1, `r']
                        if (missing(`selected_denominator') | ///
                            abs(`selected_denominator') <= ///
                            cond(missing(`stored_sharetol'), 1e-12, ///
                            `stored_sharetol')) local reconcile_other 0
                        else local selected_estimate = ///
                            `selected_estimate' / `selected_denominator'
                    }
                    else if ("`panel'" == "share_movement") {
                        local selected_denominator = `TOT'[`r', 1]
                        if (missing(`selected_denominator') | ///
                            abs(`selected_denominator') <= ///
                            cond(missing(`stored_sharetol'), 1e-12, ///
                            `stored_sharetol')) local reconcile_other 0
                        else local selected_estimate = ///
                            `selected_estimate' / `selected_denominator'
                    }
                    if (missing(`selected_estimate')) local reconcile_other 0
                    else local displayed_selected = `displayed_selected' + ///
                        round(`selected_estimate', `display_unit')
                }
                local displayed_total = `TOT'[`r', 1]
                if ("`panel'" == "share_base") {
                    local displayed_denominator = `BBASE'[1, `r']
                    if (missing(`displayed_denominator') | ///
                        abs(`displayed_denominator') <= ///
                        cond(missing(`stored_sharetol'), 1e-12, ///
                        `stored_sharetol')) local reconcile_other 0
                    else local displayed_total = ///
                        `displayed_total' / `displayed_denominator'
                }
                else if ("`panel'" == "share_movement") ///
                    local displayed_total 1
                if (`reconcile_other' & !missing(`displayed_total')) {
                    local estimate = round(`displayed_total', `display_unit') - ///
                        `displayed_selected'
                    local low = `estimate' - `zcrit' * `se'
                    local high = `estimate' + `zcrit' * `se'
                }
                __xgel_etable_emit_20260728 `"`format'"' `"`output'"' ///
                    `"`panel'"' `"`coef'"' `"Other (filtered)"' ///
                    `estimate' `se' `low' `high' `"`method_label'"' ///
                    `digits' `write_file'
            }

            local estimate = `TOT'[`r', 1]
            local se = `TOT'[`r', 2]
            local low = `estimate' - `zcrit' * `se'
            local high = `estimate' + `zcrit' * `se'
            local method_label "normal_delta"
            if ("`panel'" == "share_base") {
                local raw_estimate = `estimate'
                local denominator = `BBASE'[1, `r']
                if (missing(`denominator') | abs(`denominator') <= ///
                    cond(missing(`stored_sharetol'), 1e-12, `stored_sharetol')) {
                    local estimate .
                }
                else local estimate = `raw_estimate' / `denominator'
                local se .
                local low .
                local high .
                local method_label "not_available"
                if (`use_bootstrap') {
                    local se = `BTS'[3, `r']
                    local low = `BTS'[1, `r']
                    local high = `BTS'[2, `r']
                    local method_label "bootstrap_`bootstrap_ci'"
                }
                else if (!missing(`estimate') & `has_basecov') {
                    tempname total_share_var
                    scalar `total_share_var' = ///
                        `TOTCOV'[`r', `r'] / (`denominator'^2) + ///
                        (`raw_estimate'^2 * `BASECOV'[`r', `r']) / ///
                            (`denominator'^4) - ///
                        (2 * `raw_estimate' * `COVTB'[`r', `r']) / ///
                            (`denominator'^3)
                    local se = sqrt(max(0, scalar(`total_share_var')))
                    local low = `estimate' - `zcrit' * `se'
                    local high = `estimate' + `zcrit' * `se'
                    local method_label "normal_delta"
                    local denominator_se = ///
                        sqrt(max(0, `BASECOV'[`r', `r']))
                    local denominator_t = cond(`denominator_se' > 0, ///
                        abs(`denominator') / `denominator_se', c(maxdouble))
                    if (`denominator_t' < `stored_sharetmin') ///
                        local method_label ///
                            "normal_delta_diagnostic_only_weak_denominator"
                }
            }
            else if ("`panel'" == "share_movement") {
                local estimate 1
                local se 0
                local low 1
                local high 1
                local method_label "identity"
            }
            else if (`use_bootstrap') {
                local se = `BTCI'[3, `r']
                local low = `BTCI'[1, `r']
                local high = `BTCI'[2, `r']
                local method_label "bootstrap_`bootstrap_ci'"
            }
            __xgel_etable_emit_20260728 `"`format'"' `"`output'"' ///
                `"`panel'"' `"`coef'"' `"total_movement"' ///
                `estimate' `se' `low' `high' `"`method_label'"' ///
                `digits' `write_file'

            if ("`panel'" != "share_movement") {
                local estimate = `BFULL'[1, `r']
                local se .
                local low .
                local high .
                local method_label "not_available"
                if ("`panel'" == "share_base") {
                    local denominator = `BBASE'[1, `r']
                    if (missing(`denominator') | abs(`denominator') <= ///
                        cond(missing(`stored_sharetol'), 1e-12, `stored_sharetol')) {
                        local estimate .
                    }
                    else local estimate = `estimate' / `denominator'
                    if (`use_bootstrap') {
                        local se = `BFS'[3, `r']
                        local low = `BFS'[1, `r']
                        local high = `BFS'[2, `r']
                        local method_label "bootstrap_`bootstrap_ci'"
                    }
                }
                else if (`use_bootstrap') {
                    local se = `BFCI'[3, `r']
                    local low = `BFCI'[1, `r']
                    local high = `BFCI'[2, `r']
                    local method_label "bootstrap_`bootstrap_ci'"
                }
                __xgel_etable_emit_20260728 `"`format'"' `"`output'"' ///
                    `"`panel'"' `"`coef'"' `"full_model"' ///
                    `estimate' `se' `low' `high' `"`method_label'"' ///
                    `digits' `write_file'
            }
        }
    }

    local default_notes "Specification accounting, not causal mediation; VCE=`point_vce'; identity=`identity_status'."
    if (`"`point_notes'"' != "") {
        local default_notes `"`default_notes' `point_notes'"'
    }
    if (`use_bootstrap') {
        local default_notes "`default_notes' Bootstrap=`bootstrap_method', valid reps=`bootstrap_valid'/`bootstrap_requested'; resampling intervals do not by themselves cure nonregularity."
    }
    else if (`has_bootstrap') {
        local default_notes "`default_notes' A bootstrap result was supplied, but interval(normal) requested analytic reporting."
    }
    if (`filtered_identity_warning') {
        local default_notes "`default_notes' Filtered components were omitted because noother was requested; displayed component rows do not preserve the accounting identity."
    }
    if (`"`notes'"' != "") local default_notes `"`default_notes' `notes'"'
    local notes_output `"`default_notes'"'
    if ("`format'" == "markdown") {
        local notes_output : subinstr local notes_output "|" "\|", all
    }
    else if ("`format'" == "latex") {
        local slash = char(92)
        foreach symbol in "&" "%" "$" "#" "_" "{" "}" {
            local notes_output : subinstr local notes_output ///
                "`symbol'" "`slash'`symbol'", all
        }
    }
    else if ("`format'" == "html") {
        local notes_output : subinstr local notes_output "&" "&amp;", all
        local notes_output : subinstr local notes_output "<" "&lt;", all
        local notes_output : subinstr local notes_output ">" "&gt;", all
    }
    if (`write_file') {
        if ("`format'" == "latex") {
            file write `output' "\hline" _n "\end{tabular}" _n ///
                "\footnotesize `notes_output'" _n "\end{table}" _n
        }
        else if ("`format'" == "html") {
            file write `output' "</tbody><tfoot><tr><td colspan=" ///
                `""8">`notes_output'</td></tr></tfoot></table>"' _n
        }
        else if ("`format'" == "markdown") {
            file write `output' _n "Notes: `notes_output'" _n
        }
        file close `output'
        di as txt "Gelbach table written to " as res `"`saving'"'
    }
    else {
        di as txt _n "Notes: `default_notes'"
    }
end

capture program drop __xgel_etable_emit_20260728
program define __xgel_etable_emit_20260728
    version 14.0
    args format handle panel coefficient component estimate se low high method digits write_file
    local display_width = max(12, `digits' + 4)
    local estimate_text : display %`display_width'.`digits'f `estimate'
    local se_text : display %`display_width'.`digits'f `se'
    local low_text : display %`display_width'.`digits'f `low'
    local high_text : display %`display_width'.`digits'f `high'
    foreach value in estimate se low high {
        local `value'_text = trim("``value'_text'")
        if (missing(``value'')) local `value'_text ""
    }
    if (`write_file') {
        if ("`format'" == "markdown") {
            foreach value in panel coefficient component method {
                local `value' : subinstr local `value' "|" "\|", all
            }
            file write `handle' ///
                "| `panel' | `coefficient' | `component' | `estimate_text' | `se_text' | `low_text' | `high_text' | `method' |" _n
        }
        else if ("`format'" == "latex") {
            local slash = char(92)
            foreach value in panel coefficient component method {
                foreach symbol in "&" "%" "$" "#" "_" "{" "}" {
                    local `value' : subinstr local `value' ///
                        "`symbol'" "`slash'`symbol'", all
                }
            }
            file write `handle' ///
                "`panel' & `coefficient' & `component' & `estimate_text' & `se_text' & `low_text' & `high_text' & `method' \\" _n
        }
        else if ("`format'" == "html") {
            foreach value in panel coefficient component method {
                local `value' : subinstr local `value' "&" "&amp;", all
                local `value' : subinstr local `value' "<" "&lt;", all
                local `value' : subinstr local `value' ">" "&gt;", all
            }
            file write `handle' ///
                "<tr><td>`panel'</td><td>`coefficient'</td><td>`component'</td><td>`estimate_text'</td><td>`se_text'</td><td>`low_text'</td><td>`high_text'</td><td>`method'</td></tr>" _n
        }
        else if ("`format'" == "csv") {
            file write `handle' ///
                `""`panel'","`coefficient'","`component'",`estimate_text',`se_text',`low_text',`high_text',"`method'""' _n
        }
        else {
            file write `handle' ///
                "`panel' `coefficient' `component' `estimate_text' `se_text' `low_text' `high_text' `method'" _n
        }
    }
    else {
        if ("`format'" == "display") {
            di as txt %-16s "`panel'" %-18s "`coefficient'" ///
                %-24s "`component'" as res %12s "`estimate_text'" ///
                %12s "`se_text'" %12s "`low_text'" %12s "`high_text'" ///
                as txt "  `method'"
        }
        else if ("`format'" == "markdown") {
            di as txt "| `panel' | `coefficient' | `component' | `estimate_text' | `se_text' | `low_text' | `high_text' | `method' |"
        }
        else if ("`format'" == "latex") {
            di as txt "`panel' & `coefficient' & `component' & `estimate_text' & `se_text' & `low_text' & `high_text' & `method' \\"
        }
        else if ("`format'" == "html") {
            di as txt "<tr><td>`panel'</td><td>`coefficient'</td><td>`component'</td><td>`estimate_text'</td><td>`se_text'</td><td>`low_text'</td><td>`high_text'</td><td>`method'</td></tr>"
        }
        else di as txt `""`panel'","`coefficient'","`component'",`estimate_text',`se_text',`low_text',`high_text',"`method'""'
    }
end
