*! version 1.0.0  06jul2026
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
    capture noisily plugin call `plugin_prog' `worker' `firm' `fw_var' `keepvar' ///
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
