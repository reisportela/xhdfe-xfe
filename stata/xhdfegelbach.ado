*! version 1.0.1  09jul2026
*! Gelbach (2016) conditional decomposition, HDFE-aware (xhdfe backend).
*! Same compiled implementation as Python xhdfe.gelbach and R xhdfe_gelbach;
*! inference matches Gelbach's b1x2 (unadjusted/robust/cluster, gamma0/cov0).

program define xhdfegelbach, rclass sortpreserve
    version 14.0
    syntax varname(numeric) [if] [in] [aweight fweight /], x1(varlist numeric) ///
        [ x2groups(string) fes(varlist numeric) vce(string) cluster(varname numeric) ///
          gamma0 cov0 threads(integer 0) ]

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

    local p : word count `x1'
    local q : word count `x2vars'
    local nfe : word count `fes'
    local k1 = `p' + 1
    local G : word count `gnames'

    tempname D SE TOT
    matrix `D' = J(`k1', `G', .)
    matrix `SE' = J(`k1', `G', .)
    matrix `TOT' = J(`k1', 2, .)

    local cfg "cfg=task=gelbach;p=`p';q=`q';nfe=`nfe';has_cluster=`=("`cluster'"!="")';"
    local cfg "`cfg'has_weight=`has_weight';weight_type=`weight';"
    local cfg "`cfg'group_sizes=`gsizes_csv';vce=`vce';"
    local cfg "`cfg'gamma0=`=("`gamma0'"!="")';cov0=`=("`cov0'"!="")';num_threads=`threads';"
    local cfg "`cfg'dmat=`D';semat=`SE';totmat=`TOT';"

    quietly findfile xhdfe.ado
    local plugin_path "`r(fn)'"
    local plugin_path : subinstr local plugin_path "xhdfe.ado" "xhdfe.plugin", all
    capture confirm file "`plugin_path'"
    if (_rc) {
        di as err "xhdfe.plugin not found next to xhdfe.ado"
        exit _rc
    }
    local plugin_prog "__xhdfe_plugin_dispatch"
    capture program `plugin_prog', plugin using("`plugin_path'")
    if (_rc & _rc != 110) {
        di as err "xhdfe.plugin could not be loaded from `plugin_path'"
        exit _rc
    }
    global XHDFE_PLUGIN_PATH_INTERNAL "`plugin_path'"

    local wpass ""
    if (`has_weight') local wpass "`wvar'"
    capture noisily plugin call `plugin_prog' `y' `x1' `x2vars' `fes' `cluster' `wpass' ///
        if `touse', "`cfg'"
    if (_rc) exit _rc

    matrix rownames `D' = `x1' _cons
    matrix colnames `D' = `gnames'
    matrix rownames `SE' = `x1' _cons
    matrix colnames `SE' = `gnames'
    matrix rownames `TOT' = `x1' _cons
    matrix colnames `TOT' = delta se

    di as txt ""
    di as txt "Gelbach conditional decomposition" as res "  (xhdfe backend, vce: `vce'`gamma0')"
    di as txt "Contributions (delta):"
    matrix list `D', noheader format(%12.6f)
    di as txt "Standard errors:"
    matrix list `SE', noheader format(%12.6f)
    di as txt "Total (= b_base - b_full):"
    matrix list `TOT', noheader format(%12.6f)
    if (scalar(__xgel_converged) != 1) {
        di as err "warning: computation flagged non-converged; see r(notes)"
    }

    return scalar identity_gap = scalar(__xgel_identity_gap)
    return scalar n_obs = scalar(__xgel_n_obs)
    return scalar df_full = scalar(__xgel_df_full)
    return scalar converged = scalar(__xgel_converged)
    capture scalar drop __xgel_identity_gap __xgel_n_obs __xgel_df_full __xgel_converged
    return local vce "`vce'"
    return local groups "`gnames'"
    if ("`xgel_notes'" != "") return local notes "`xgel_notes'"
    return matrix total = `TOT'
    return matrix se = `SE'
    return matrix delta = `D'
end
