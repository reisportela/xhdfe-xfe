*! version 1.1.2  10jul2026
*! Leave-one-out connected set (KSS / LeaveOutTwoWay semantics) as a
*! standalone sample-preparation utility on the xhdfe backend: largest
*! connected component (firm count), iterative removal of articulation
*! workers from the mover-firm bipartite graph, drop of single-observation
*! workers. Same computation as the sample step of xhdfeakm.

program define xhdfeconnected, rclass sortpreserve
    version 14.0
    syntax varlist(min=2 max=2 numeric) [if] [in] [fweight], ///
        GENerate(name) [REPLACE]

    tokenize `varlist'
    local worker `1'
    local firm `2'
    marksample touse
    markout `touse' `worker' `firm'
    quietly count if `touse'
    if (r(N) == 0) {
        error 2000
    }

    // Recode worker/firm ids that fall outside int32 range (e.g. NISS/NIF
    // person codes) or are non-integer to a compact 1..N integer for the
    // plugin. The connected set depends only on the worker-firm graph, so the
    // recoding leaves the result unchanged.
    local worker_use "`worker'"
    local firm_use "`firm'"
    local id_recoded 0
    foreach part in worker firm {
        local src "`worker'"
        if ("`part'" == "firm") local src "`firm'"
        quietly summarize `src' if `touse', meanonly
        local need = (r(N) > 0 & (r(min) < -2147483648 | r(max) > 2147483647))
        if (!`need') {
            quietly count if `touse' & abs(`src' - floor(`src' + 0.5)) > 1e-6
            local need = (r(N) > 0)
        }
        if (`need') {
            tempvar idc_`part'
            quietly egen long `idc_`part'' = group(`src') if `touse'
            local `part'_use "`idc_`part''"
            local id_recoded 1
        }
    }
    if (`id_recoded') {
        di as txt "note: worker/firm ids outside int32 range recoded to compact integers (graph and results unchanged)"
    }

    local has_fweight 0
    tempvar fwvar
    if ("`weight'" != "") {
        local has_fweight 1
        quietly gen double `fwvar' `exp' if `touse'
        markout `touse' `fwvar'
    }

    capture confirm new variable `generate'
    if (_rc & "`replace'" == "") {
        di as err "variable `generate' already exists (use replace)"
        exit 110
    }
    capture drop `generate'

    tempvar keepvar
    quietly gen double `keepvar' = . if `touse'

    local cfg "cfg=task=akm_leave_out_set;has_fweight=`has_fweight';"

    // Bind the plugin next to the active xhdfe.ado (same guards as xhdfeakm).
    quietly findfile xhdfe.ado
    local plugin_path "`r(fn)'"
    local plugin_path : subinstr local plugin_path "xhdfe.ado" "xhdfe.plugin", all
    // `program ... , plugin using()' does not expand a leading ~ the way
    // `confirm file' does, so a tilde'd PLUS dir (the Unix default ~/ado/plus)
    // makes the load fail with r(601). Expand it here so the confirm, the
    // stale-path guard, and the load all see one absolute path.
    if (substr(`"`plugin_path'"', 1, 1) == "~") {
        local __plugin_home : env HOME
        if (`"`__plugin_home'"' != "") {
            local plugin_path : subinstr local plugin_path "~" `"`__plugin_home'"'
        }
    }
    capture confirm file "`plugin_path'"
    if (_rc) {
        di as err "xhdfe.plugin not found next to xhdfe.ado; rebuild the plugin in `plugin_path'"
        exit _rc
    }
    local plugin_prog "__xhdfe_plugin_dispatch"
    if ("$XHDFE_PLUGIN_PATH_INTERNAL" != "" & "$XHDFE_PLUGIN_PATH_INTERNAL" != "`plugin_path'") {
        di as err "xhdfeconnected: the active session is still bound to an older xhdfe.plugin path"
        di as err "xhdfeconnected: run discard (with no arguments) and rerun the command"
        exit 498
    }
    capture program `plugin_prog', plugin using("`plugin_path'")
    if (_rc & _rc != 110) {
        di as err "xhdfe.plugin could not be loaded from `plugin_path'"
        exit _rc
    }
    global XHDFE_PLUGIN_PATH_INTERNAL "`plugin_path'"

    local fw_var ""
    if (`has_fweight') {
        local fw_var "`fwvar'"
    }
    capture noisily plugin call `plugin_prog' `worker_use' `firm_use' `fw_var' `keepvar' ///
        if `touse', "`cfg'"
    local rc = _rc
    if (`rc') {
        exit `rc'
    }

    foreach name in n_obs n_obs_input n_obs_connected n_workers n_firms ///
        n_matches n_movers n_stayers prune_iterations {
        return scalar `name' = scalar(__xakm_`name')
        capture scalar drop __xakm_`name'
    }
    return local cmd "xhdfeconnected"

    quietly gen byte `generate' = (`keepvar' == 1) if `touse'
    label variable `generate' "leave-one-out connected sample flag"

    di as txt ""
    di as txt "Leave-one-out connected set" as res "  (KSS / LeaveOutTwoWay semantics)"
    di as txt "  input obs:      " as res %12.0fc return(n_obs_input)
    di as txt "  connected obs:  " as res %12.0fc return(n_obs_connected)
    di as txt "  leave-out obs:  " as res %12.0fc return(n_obs)
    di as txt "  workers: " as res %10.0fc return(n_workers) ///
        as txt "  (movers " as res %10.0fc return(n_movers) ///
        as txt ", stayers " as res %10.0fc return(n_stayers) as txt ")"
    di as txt "  firms:   " as res %10.0fc return(n_firms) ///
        as txt "   matches " as res %10.0fc return(n_matches)
    di as txt "  flag saved in " as res "`generate'"
end
