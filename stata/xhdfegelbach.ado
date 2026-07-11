*! version 1.2.0  11jul2026
*! Gelbach (2016) conditional decomposition, HDFE-aware (xhdfe backend).
*! Same compiled implementation as Python xhdfe.gelbach and R xhdfe_gelbach;
*! inference matches Gelbach's b1x2 (unadjusted/robust/cluster, gamma0/cov0).

program define xhdfegelbach, rclass sortpreserve
    version 14.0
    syntax varname(numeric) [if] [in] [aweight fweight /], x1(varlist numeric) ///
        [ x2groups(string) fes(varlist numeric) vce(string) cluster(varname numeric) ///
          gamma0 cov0 tol(real 1e-8) threads(integer 0) GPU VERBose ]

    local y `varlist'
    if ("`x2groups'" == "" & "`fes'" == "") {
        di as err "specify x2groups() and/or fes()"
        exit 198
    }
    if ("`vce'" == "") local vce "unadjusted"
    if (!inlist("`vce'", "unadjusted", "robust", "cluster")) {
        di as err "vce() must be unadjusted, robust or cluster"
        exit 198
    }
    if ("`vce'" == "cluster" & "`cluster'" == "") {
        di as err "vce(cluster) requires cluster()"
        exit 198
    }
    if ("`vce'" != "cluster" & "`cluster'" != "") {
        di as err "cluster() requires vce(cluster)"
        exit 198
    }
    if (!(`tol' > 0) | missing(`tol')) {
        di as err "tol() must be finite and strictly positive"
        exit 198
    }
    if (`threads' < 0) {
        di as err "threads() must be nonnegative"
        exit 198
    }

    * parse x2groups("name = varlist : name2 = varlist2") like b1x2's x2delta
    local gnames
    local x2vars
    local gsizes
    if ("`x2groups'" != "") {
        tokenize "`x2groups'", parse(":")
        while ("`1'" != "") {
            if ("`1'" != ":") {
                gettoken gname rest : 1, parse("=")
                gettoken eq gvars : rest, parse("=")
                local gname = trim("`gname'")
                local gvars = trim("`gvars'")
                if ("`gname'" == "" | "`gvars'" == "") {
                    di as err "each x2groups() block must be name = varlist"
                    exit 198
                }
                capture confirm name `gname'
                if (_rc) {
                    di as err "x2groups() block name `gname' is not a valid Stata name"
                    exit 198
                }
                unab gvars : `gvars'
                local gnames "`gnames' `gname'"
                local x2vars "`x2vars' `gvars'"
                local gsizes "`gsizes' `: word count `gvars''"
            }
            macro shift
        }
    }
    foreach f of local fes {
        local gnames "`gnames' `f'"
    }
    local duplicate_vars : list dups x2vars
    if ("`duplicate_vars'" != "") {
        di as err "x2 variables may not appear in more than one block: `duplicate_vars'"
        exit 198
    }
    local overlap : list x1 & x2vars
    if ("`overlap'" != "") {
        di as err "variables may not appear in both x1() and x2groups(): `overlap'"
        exit 198
    }
    local duplicate_names : list dups gnames
    if ("`duplicate_names'" != "") {
        di as err "x2 and FE block names must be unique: `duplicate_names'"
        exit 198
    }
    local gsizes_csv : subinstr local gsizes " " ",", all
    local gsizes_csv = trim("`gsizes_csv'")
    local gsizes_csv : subinstr local gsizes_csv " " "", all

    marksample touse
    markout `touse' `x1' `x2vars' `fes' `cluster'
    local has_weight 0
    tempvar wvar
    if ("`weight'" != "") {
        local has_weight 1
        quietly gen double `wvar' = `exp' if `touse'
        markout `touse' `wvar'
    }
    quietly count if `touse'
    if (r(N) == 0) error 2000

    // FE and cluster ids enter the shared plugin as int32. Compact raw codes
    // outside that range (for example NIF/NISS) or non-integer categorical
    // codes to 1..N labels. Gelbach's decomposition depends only on category
    // membership, so this is an exact relabelling with unchanged estimates.
    local fes_use
    local cluster_use "`cluster'"
    local id_recoded 0
    foreach src of local fes {
        quietly summarize `src' if `touse', meanonly
        local need = (r(N) > 0 & (r(min) < -2147483648 | r(max) > 2147483647))
        if (!`need') {
            quietly count if `touse' & abs(`src' - floor(`src' + 0.5)) > 1e-6
            local need = (r(N) > 0)
        }
        if (`need') {
            tempvar idc
            quietly egen long `idc' = group(`src') if `touse'
            local fes_use "`fes_use' `idc'"
            local id_recoded 1
        }
        else local fes_use "`fes_use' `src'"
    }
    if ("`cluster'" != "") {
        quietly summarize `cluster' if `touse', meanonly
        local need = (r(N) > 0 & (r(min) < -2147483648 | r(max) > 2147483647))
        if (!`need') {
            quietly count if `touse' & abs(`cluster' - floor(`cluster' + 0.5)) > 1e-6
            local need = (r(N) > 0)
        }
        if (`need') {
            tempvar clc
            quietly egen long `clc' = group(`cluster') if `touse'
            local cluster_use "`clc'"
            local id_recoded 1
        }
    }
    if (`id_recoded') {
        di as txt "note: FE/cluster ids outside int32 range or non-integer recoded to compact integers (groups and results unchanged)"
    }

    local p : word count `x1'
    local q : word count `x2vars'
    local nfe : word count `fes'
    local k1 = `p' + 1
    local G : word count `gnames'

    tempname D SE TOT BBASE BFULL COV TOTCOV FEAGG
    matrix `D' = J(`k1', `G', .)
    matrix `SE' = J(`k1', `G', .)
    matrix `TOT' = J(`k1', 2, .)
    matrix `BBASE' = J(1, `p', .)
    matrix `BFULL' = J(1, `p', .)
    matrix `COV' = J(`G' * `k1', `G' * `k1', .)
    matrix `TOTCOV' = J(`k1', `k1', .)

    local cfg "cfg=task=gelbach;p=`p';q=`q';nfe=`nfe';has_cluster=`=("`cluster'"!="")';"
    local cfg "`cfg'has_weight=`has_weight';weight_type=`weight';"
    local cfg "`cfg'group_sizes=`gsizes_csv';vce=`vce';"
    local cfg "`cfg'gamma0=`=("`gamma0'"!="")';cov0=`=("`cov0'"!="")';tol=`tol';num_threads=`threads';"
    local cfg "`cfg'use_gpu=`=("`gpu'"!="")';verbose=`=("`verbose'"!="")';"
    local cfg "`cfg'dmat=`D';semat=`SE';totmat=`TOT';bbasemat=`BBASE';bfullmat=`BFULL';"
    local cfg "`cfg'covmat=`COV';totcovmat=`TOTCOV';"

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
        di as err "xhdfe.plugin not found next to xhdfe.ado"
        exit _rc
    }
    local plugin_prog "__xhdfe_plugin_dispatch"
    if ("$XHDFE_PLUGIN_PATH_INTERNAL" != "" & "$XHDFE_PLUGIN_PATH_INTERNAL" != "`plugin_path'") {
        di as err "xhdfegelbach: the active session is still bound to an older xhdfe.plugin path"
        di as err "xhdfegelbach: run discard (with no arguments) and rerun the command"
        exit 498
    }
    capture program `plugin_prog', plugin using("`plugin_path'")
    if (_rc & _rc != 110) {
        di as err "xhdfe.plugin could not be loaded from `plugin_path'"
        exit _rc
    }
    global XHDFE_PLUGIN_PATH_INTERNAL "`plugin_path'"

    local wpass ""
    if (`has_weight') local wpass "`wvar'"
    capture noisily plugin call `plugin_prog' `y' `x1' `x2vars' `fes_use' `cluster_use' `wpass' ///
        if `touse', "`cfg'"
    if (_rc) exit _rc

    matrix rownames `D' = `x1' _cons
    matrix colnames `D' = `gnames'
    matrix rownames `SE' = `x1' _cons
    matrix colnames `SE' = `gnames'
    matrix rownames `TOT' = `x1' _cons
    matrix colnames `TOT' = delta se
    matrix colnames `BBASE' = `x1'
    matrix colnames `BFULL' = `x1'
    local covnames
    foreach g of local gnames {
        foreach x in `x1' _cons {
            local covnames "`covnames' `g':`x'"
        }
    }
    matrix rownames `COV' = `covnames'
    matrix colnames `COV' = `covnames'
    matrix rownames `TOTCOV' = `x1' _cons
    matrix colnames `TOTCOV' = `x1' _cons

    if (`nfe' > 0) {
        matrix `FEAGG' = J(`k1', 2, .)
        local first_fe = `G' - `nfe' + 1
        tempname fec fev
        forvalues r = 1/`k1' {
            scalar `fec' = 0
            scalar `fev' = 0
            forvalues g = `first_fe'/`G' {
                scalar `fec' = scalar(`fec') + `D'[`r', `g']
                local gr = (`g' - 1) * `k1' + `r'
                forvalues h = `first_fe'/`G' {
                    local hr = (`h' - 1) * `k1' + `r'
                    scalar `fev' = scalar(`fev') + `COV'[`gr', `hr']
                }
            }
            matrix `FEAGG'[`r', 1] = scalar(`fec')
            matrix `FEAGG'[`r', 2] = sqrt(max(0, scalar(`fev')))
        }
        matrix rownames `FEAGG' = `x1' _cons
        matrix colnames `FEAGG' = delta se
    }

    di as txt ""
    di as txt "Gelbach coefficient-movement decomposition" as res "  (xhdfe backend, vce: `vce'`gamma0')"
    di as txt "Specification accounting; not causal mediation."
    di as txt "Contributions (delta):"
    matrix list `D', noheader format(%12.6f)
    di as txt "Standard errors:"
    matrix list `SE', noheader format(%12.6f)
    di as txt "Total (= b_base - b_full):"
    matrix list `TOT', noheader format(%12.6f)
    if (`nfe' > 0) {
        di as txt "Absorbed-FE aggregate (conditional/gamma0 SE):"
        matrix list `FEAGG', noheader format(%12.6f)
        di as txt "note: SEs for absorbed-FE blocks condition on the recovered FE estimates"
    }
    if (scalar(__xgel_converged) != 1) {
        di as err "warning: computation flagged non-converged; see r(notes)"
    }
    else if (strpos(lower("`xgel_notes'"), "warning:") > 0) {
        di as err "warning: Gelbach inferential diagnostic; see r(notes)"
    }

    return scalar identity_gap = scalar(__xgel_identity_gap)
    return scalar n_obs = scalar(__xgel_n_obs)
    return scalar df_full = scalar(__xgel_df_full)
    return scalar converged = scalar(__xgel_converged)
    return scalar tol = `tol'
    foreach name in threads_used gpu_used gpu_status_code gpu_attempted ///
        gpu_absorption_converged gpu_absorption_iterations {
        return scalar `name' = scalar(__xgel_`name')
        capture scalar drop __xgel_`name'
    }
    return scalar gpu_requested = ("`gpu'" != "")
    capture scalar drop __xgel_identity_gap __xgel_n_obs __xgel_df_full __xgel_converged
    return local vce "`vce'"
    return local groups "`gnames'"
    return local estimand "coefficient_movement"
    return local causal_interpretation "no"
    return local fe_se_type "conditional_gamma0"
    local gpu_status "not_requested"
    if (return(gpu_status_code) == 1) local gpu_status "used"
    else if (return(gpu_status_code) == 2) local gpu_status "unavailable"
    else if (return(gpu_status_code) == 3) local gpu_status "not_converged"
    else if (return(gpu_status_code) == 4) local gpu_status "failed"
    else if (return(gpu_status_code) == 5) local gpu_status "cpu_cache"
    else if (return(gpu_status_code) == 6) local gpu_status "not_applicable"
    local gpu_backend = cond(return(gpu_used) == 1, "cuda", "cpu")
    return local gpu_backend "`gpu_backend'"
    return local gpu_status "`gpu_status'"
    if ("`xgel_notes'" != "") return local notes "`xgel_notes'"
    return matrix total = `TOT'
    return matrix total_cov = `TOTCOV'
    return matrix cov = `COV'
    return matrix b_full = `BFULL'
    return matrix b_base = `BBASE'
    if (`nfe' > 0) return matrix fe_total = `FEAGG'
    return matrix se = `SE'
    return matrix delta = `D'
    if ("`gpu'" != "") {
        di as txt "GPU backend: " as res "`gpu_backend'" ///
            as txt " (" as res "`gpu_status'" as txt ")"
    }
end
