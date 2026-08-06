*! version 1.0.0  06aug2026
*! Identity-preserving waterfall plot for the most recent Gelbach result.

program define xhdfegelbachcoefplot
    version 14.0

    tempname D TOT BBASE BFULL
    capture confirm matrix r(delta)
    if (_rc) {
        di as err "xhdfegelbachcoefplot must immediately follow xhdfegelbach or xhdfegelbachbootstrap"
        exit 301
    }
    matrix `D' = r(delta)
    matrix `TOT' = r(total)
    matrix `BBASE' = r(b_base)
    matrix `BFULL' = r(b_full)
    local groups "`r(groups)'"
    local x1_names "`r(x1_names)'"
    local focal_names "`r(focal_names)'"

    syntax [, FOCAL(string) KEEP(string asis) DROP(string asis) EXACT ///
        LABELS(string) NOANNOTATE NOOTHER ///
        TITLE(string asis) NOTES(string asis) NAME(name) ///
        SAVING(string asis) REPLACE NODRAW ]

    local saving : subinstr local saving `"""' "", all
    local keep : subinstr local keep `"""' "", all
    local drop : subinstr local drop `"""' "", all
    local title : subinstr local title `"""' "", all
    local notes : subinstr local notes `"""' "", all
    if ("`focal'" == "") {
        local focal : word 1 of `focal_names'
    }
    if ("`focal'" == "") local focal : word 1 of `x1_names'
    local focal_count : word count `focal'
    local focal_pos : list posof "`focal'" in x1_names
    if (`focal_count' != 1 | `focal_pos' == 0) {
        di as err "focal() must select exactly one x1 coefficient"
        exit 198
    }

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

    local selected
    local omitted
    foreach group of local groups {
        local keep_group 1
        if (`"`keep'"' != "") {
            if ("`exact'" != "") {
                local keep_group : list posof "`group'" in keep
            }
            else local keep_group = regexm("`group'", `"`keep'"')
        }
        if (`keep_group' & `"`drop'"' != "") {
            if ("`exact'" != "") {
                local dropped : list posof "`group'" in drop
                if (`dropped') local keep_group 0
            }
            else if (regexm("`group'", `"`drop'"')) local keep_group 0
        }
        if (`keep_group') local selected "`selected' `group'"
        else local omitted "`omitted' `group'"
    }
    local selected = trim("`selected'")
    local omitted = trim("`omitted'")

    local nsteps : word count `selected'
    if ("`noother'" == "" & "`omitted'" != "") local ++nsteps
    local nrows = `nsteps' + 2
    local base = `BBASE'[1, `focal_pos']
    local full = `BFULL'[1, `focal_pos']
    local movement = `TOT'[`focal_pos', 1]
    local current = `base'

    preserve
    quietly clear
    quietly set obs `nrows'
    quietly generate double position = _n
    quietly generate double endpoint_low = .
    quietly generate double endpoint_high = .
    quietly generate double positive_low = .
    quietly generate double positive_high = .
    quietly generate double negative_low = .
    quietly generate double negative_high = .
    quietly generate double filtered_low = .
    quietly generate double filtered_high = .
    quietly generate double connector = .
    quietly generate double annotation_y = .
    quietly generate str24 annotation = ""
    quietly generate str80 step_label = ""

    quietly replace endpoint_low = min(0, `base') in 1
    quietly replace endpoint_high = max(0, `base') in 1
    quietly replace connector = `base' in 1
    quietly replace step_label = "Base model" in 1
    local xlabel `"1 "Base model""'
    local position 1
    foreach group of local selected {
        local ++position
        local g : list posof "`group'" in groups
        local contribution = `D'[`focal_pos', `g']
        local ending = `current' - `contribution'
        local low = min(`current', `ending')
        local high = max(`current', `ending')
        if (`contribution' >= 0) {
            quietly replace positive_low = `low' in `position'
            quietly replace positive_high = `high' in `position'
        }
        else {
            quietly replace negative_low = `low' in `position'
            quietly replace negative_high = `high' in `position'
        }
        local display_label `"`xgel_label_`group''"'
        quietly replace step_label = `"`display_label'"' in `position'
        quietly replace connector = `ending' in `position'
        quietly replace annotation_y = (`current' + `ending') / 2 ///
            in `position'
        if ("`noannotate'" == "" & !missing(`movement') & ///
            abs(`movement') > 1e-12) {
            local share_text : display %5.1f (100 * `contribution' / `movement')
            quietly replace annotation = trim("`share_text'") + "%" ///
                in `position'
        }
        local xlabel `"`xlabel' `position' "`display_label'""'
        local current = `ending'
    }
    if ("`noother'" == "" & "`omitted'" != "") {
        local ++position
        local contribution 0
        foreach group of local omitted {
            local g : list posof "`group'" in groups
            local contribution = `contribution' + `D'[`focal_pos', `g']
        }
        local ending = `current' - `contribution'
        quietly replace filtered_low = min(`current', `ending') in `position'
        quietly replace filtered_high = max(`current', `ending') in `position'
        quietly replace connector = `ending' in `position'
        quietly replace annotation_y = (`current' + `ending') / 2 ///
            in `position'
        quietly replace step_label = "Other (filtered)" in `position'
        if ("`noannotate'" == "" & !missing(`movement') & ///
            abs(`movement') > 1e-12) {
            local share_text : display %5.1f (100 * `contribution' / `movement')
            quietly replace annotation = trim("`share_text'") + "%" ///
                in `position'
        }
        local xlabel `"`xlabel' `position' "Other (filtered)""'
        local current = `ending'
    }
    local ++position
    quietly replace endpoint_low = min(0, `full') in `position'
    quietly replace endpoint_high = max(0, `full') in `position'
    quietly replace connector = `full' in `position'
    quietly replace step_label = "Full model" in `position'
    local xlabel `"`xlabel' `position' "Full model""'

    if (`"`title'"' == "") ///
        local title `"Gelbach decomposition of `focal'"'
    if (`"`notes'"' == "") {
        local notes "Movement = base - full. Filtered components are aggregated as Other so the waterfall remains an accounting identity."
    }
    local graph_options ///
        `"xlabel(`xlabel', angle(35) labsize(small)) xtitle("") ytitle("Coefficient: `focal'") title(`"`title'"') note(`"`notes'"', size(vsmall)) legend(order(1 "Base/full" 2 "Positive contribution" 3 "Negative contribution" 4 "Other (filtered)"))"'
    if ("`name'" != "") ///
        local graph_options `"`graph_options' name(`name', replace)"'
    if ("`nodraw'" != "") local graph_options "`graph_options' nodraw"

    twoway ///
        (rbar endpoint_low endpoint_high position, ///
            barwidth(.72) color("55 65 81")) ///
        (rbar positive_low positive_high position, ///
            barwidth(.72) color("37 99 235")) ///
        (rbar negative_low negative_high position, ///
            barwidth(.72) color("220 38 38")) ///
        (rbar filtered_low filtered_high position, ///
            barwidth(.72) color("107 114 128")) ///
        (line connector position, lcolor("156 163 175") lpattern(dash)) ///
        (scatter annotation_y position, msymbol(none) ///
            mlabel(annotation) mlabcolor(white) mlabsize(small)), ///
        `graph_options'

    if (`"`saving'"' != "") {
        local replace_option
        if ("`replace'" != "") local replace_option "replace"
        graph save `"`saving'"', `replace_option'
    }
    restore

    di as txt "Gelbach waterfall: base=" as res %10.6g `base' ///
        as txt ", full=" as res %10.6g `full' ///
        as txt ", filtered residual before endpoint=" ///
        as res %10.3e (`current' - `full')
end
