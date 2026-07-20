*! version 1.3.0  20jul2026
*! Gelbach (2016) conditional decomposition, HDFE-aware (xhdfe backend).
*! Same compiled implementation as Python xhdfe.gelbach and R xhdfe_gelbach;
*! inference matches Gelbach's b1x2 (unadjusted/robust/cluster, gamma0/cov0).

program define xhdfegelbach, rclass sortpreserve
    version 14.0
    syntax varname(numeric) [if] [in] [aweight fweight /], x1(varlist numeric) ///
        [ x2groups(string) fes(varlist numeric) vce(string) cluster(varname numeric) ///
          gamma0 cov0 ABSorbedtargets(varlist numeric) tol(real 1e-8) ///
          threads(integer 0) GPU VERBose FOCAL(varlist numeric) ///
          SHARES(string) SHARETOL(real 1e-12) Level(cilevel) ]

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
    local shares = lower(trim("`shares'"))
    if ("`shares'" != "" & !inlist("`shares'", "base", "base_fixed", "movement")) {
        di as err "shares() must be base, base_fixed, or movement"
        exit 198
    }
    if (`sharetol' < 0 | missing(`sharetol')) {
        di as err "sharetol() must be finite and nonnegative"
        exit 198
    }
    local focal_requested = ("`focal'" != "")
    if (`focal_requested') {
        unab focal : `focal'
        local invalid_focal : list focal - x1
        if ("`invalid_focal'" != "") {
            di as err "focal() must be a subset of x1(): `invalid_focal'"
            exit 198
        }
    }
    else local focal "`x1'"

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
    local invalid_absorbed : list absorbedtargets - x1
    if ("`invalid_absorbed'" != "") {
        di as err "absorbedtargets() must be a subset of x1(): `invalid_absorbed'"
        exit 198
    }
    if ("`absorbedtargets'" != "" & "`fes'" == "") {
        di as err "absorbedtargets() requires fes()"
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
    local focal_indices
    foreach target of local focal {
        local pos : list posof "`target'" in x1
        local idx0 = `pos' - 1
        local focal_indices "`focal_indices' `idx0'"
    }
    local focal_indices = trim("`focal_indices'")
    local nabs : word count `absorbedtargets'
    local absorbed_indices
    foreach target of local absorbedtargets {
        local pos : list posof "`target'" in x1
        local idx = `pos' - 1
        local absorbed_indices "`absorbed_indices' `idx'"
    }
    local absorbed_indices_csv : subinstr local absorbed_indices " " ",", all
    local absorbed_indices_csv = trim("`absorbed_indices_csv'")
    local absorbed_indices_csv : subinstr local absorbed_indices_csv " " "", all

    tempname D SE TOT BBASE BFULL ABS COV TOTCOV FEAGG
    matrix `D' = J(`k1', `G', .)
    matrix `SE' = J(`k1', `G', .)
    matrix `TOT' = J(`k1', 2, .)
    matrix `BBASE' = J(1, `p', .)
    matrix `BFULL' = J(1, `p', .)
    matrix `ABS' = J(1, `p', .)
    matrix `COV' = J(`G' * `k1', `G' * `k1', .)
    matrix `TOTCOV' = J(`k1', `k1', .)

    local cfg "cfg=task=gelbach;p=`p';q=`q';nfe=`nfe';has_cluster=`=("`cluster'"!="")';"
    local cfg "`cfg'has_weight=`has_weight';weight_type=`weight';"
    local cfg "`cfg'group_sizes=`gsizes_csv';vce=`vce';"
    local cfg "`cfg'gamma0=`=("`gamma0'"!="")';cov0=`=("`cov0'"!="")';"
    local cfg "`cfg'absorbed_x1=`absorbed_indices_csv';tol=`tol';num_threads=`threads';"
    local cfg "`cfg'use_gpu=`=("`gpu'"!="")';verbose=`=("`verbose'"!="")';"
    local cfg "`cfg'dmat=`D';semat=`SE';totmat=`TOT';bbasemat=`BBASE';bfullmat=`BFULL';"
    local cfg "`cfg'absmat=`ABS';covmat=`COV';totcovmat=`TOTCOV';"

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
    matrix colnames `ABS' = `x1'

    // The backend classification is authoritative.  Never infer an imposed
    // zero merely from the user's request: a shared-core classification bug
    // must be visible and must not be relabelled by the ado layer.
    local nabs 0
    local absorbed_backend_indices
    local absorbed_backend_names
    forvalues c = 1/`p' {
        if (`ABS'[1, `c'] == 1) {
            local ++nabs
            local idx0 = `c' - 1
            local absorbed_backend_indices "`absorbed_backend_indices' `idx0'"
            local xname : word `c' of `x1'
            local absorbed_backend_names "`absorbed_backend_names' `xname'"
        }
    }
    local absorbed_backend_indices = trim("`absorbed_backend_indices'")
    local absorbed_backend_names = trim("`absorbed_backend_names'")
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

    // Optional empirical-reporting layer. It is pure post-processing of the
    // certified delta/covariance objects and never changes a fit or a matrix
    // returned by the core. Shares remain signed and are not renormalized.
    tempname SHARE SHARESE SHARELO SHAREHI SHAREDEF SHARERESID
    local share_se_type ""
    local share_undefined 0
    if ("`shares'" != "") {
        matrix `SHARE' = J(`k1', `G', .)
        matrix `SHARESE' = J(`k1', `G', .)
        matrix `SHARELO' = J(`k1', `G', .)
        matrix `SHAREHI' = J(`k1', `G', .)
        matrix `SHAREDEF' = J(`k1', 1, 0)
        matrix `SHARERESID' = J(1, `p', .)
        local alpha = (100 - `level') / 200
        local zcrit = invnormal(1 - `alpha')
        if ("`shares'" == "base") ///
            local share_se_type "not_available_joint_base_covariance"
        else if ("`shares'" == "base_fixed") ///
            local share_se_type "fixed_base_denominator_scaling"
        else local share_se_type "joint_covariance_delta_method"

        forvalues r = 1/`k1' {
            local denom .
            if (inlist("`shares'", "base", "base_fixed") & `r' <= `p') ///
                local denom = `BBASE'[1, `r']
            else if ("`shares'" == "movement") local denom = `TOT'[`r', 1]
            if (!missing(`denom') & abs(`denom') > `sharetol') {
                matrix `SHAREDEF'[`r', 1] = 1
                if (inlist("`shares'", "base", "base_fixed") & `r' <= `p') {
                    matrix `SHARERESID'[1, `r'] = `BFULL'[1, `r'] / `denom'
                }
                forvalues g = 1/`G' {
                    matrix `SHARE'[`r', `g'] = `D'[`r', `g'] / `denom'
                    if ("`shares'" == "base_fixed") {
                        matrix `SHARESE'[`r', `g'] = `SE'[`r', `g'] / abs(`denom')
                    }
                    else if ("`shares'" == "movement") {
                        tempname sharevar
                        scalar `sharevar' = 0
                        forvalues h = 1/`G' {
                            local gh = -`D'[`r', `g'] / (`denom' * `denom')
                            if (`h' == `g') local gh = `gh' + 1 / `denom'
                            local hi = (`h' - 1) * `k1' + `r'
                            forvalues l = 1/`G' {
                                local gl = -`D'[`r', `g'] / (`denom' * `denom')
                                if (`l' == `g') local gl = `gl' + 1 / `denom'
                                local li = (`l' - 1) * `k1' + `r'
                                scalar `sharevar' = scalar(`sharevar') + ///
                                    `gh' * `COV'[`hi', `li'] * `gl'
                            }
                        }
                        matrix `SHARESE'[`r', `g'] = sqrt(max(0, scalar(`sharevar')))
                    }
                    if (!missing(`SHARESE'[`r', `g'])) {
                        matrix `SHARELO'[`r', `g'] = `SHARE'[`r', `g'] - ///
                            `zcrit' * `SHARESE'[`r', `g']
                        matrix `SHAREHI'[`r', `g'] = `SHARE'[`r', `g'] + ///
                            `zcrit' * `SHARESE'[`r', `g']
                    }
                }
            }
            else if (`r' <= `p') local share_undefined 1
        }
        foreach M in SHARE SHARESE SHARELO SHAREHI {
            matrix rownames ``M'' = `x1' _cons
            matrix colnames ``M'' = `gnames'
        }
        matrix rownames `SHAREDEF' = `x1' _cons
        matrix colnames `SHAREDEF' = defined
        matrix colnames `SHARERESID' = `x1'
    }

    // Human-facing display.  Keep the numerical API entirely in r(); this
    // panel only reorganizes the same matrices into one readable block per
    // coefficient and therefore has no estimator or precision side effects.
    local nx2g = `G' - `nfe'
    local coefnames "`x1' _cons"
    local display_coefs "`coefnames'"
    if (`focal_requested') local display_coefs "`focal'"
    local cov_word = cond(`nx2g' == 1, "block", "blocks")
    local fe_word = cond(`nfe' == 1, "fixed effect", "fixed effects")

    local vce_display "`vce'"
    if ("`vce'" == "cluster") local vce_display "clustered by `cluster'"
    if ("`gamma0'" != "") local vce_display "`vce_display', gamma0"
    if ("`cov0'" != "") local vce_display "`vce_display', cov0"
    if (`has_weight') local vce_display "`vce_display'; `weight'"

    local gpu_status "not_requested"
    if (scalar(__xgel_gpu_status_code) == 1) local gpu_status "used"
    else if (scalar(__xgel_gpu_status_code) == 2) local gpu_status "unavailable"
    else if (scalar(__xgel_gpu_status_code) == 3) local gpu_status "not_converged"
    else if (scalar(__xgel_gpu_status_code) == 4) local gpu_status "failed"
    else if (scalar(__xgel_gpu_status_code) == 5) local gpu_status "cpu_cache"
    else if (scalar(__xgel_gpu_status_code) == 6) local gpu_status "not_applicable"
    local gpu_backend = cond(scalar(__xgel_gpu_used) == 1, "cuda", "cpu")
    local backend_display = upper("`gpu_backend'")
    if ("`gpu'" != "" & "`gpu_status'" != "used") {
        if ("`gpu_status'" == "unavailable") local backend_display "CPU (CUDA unavailable)"
        else if ("`gpu_status'" == "not_converged") local backend_display "CPU (CUDA did not converge)"
        else if ("`gpu_status'" == "failed") local backend_display "CPU (CUDA failed)"
        else if ("`gpu_status'" == "cpu_cache") local backend_display "CPU (cached result)"
        else if ("`gpu_status'" == "not_applicable") local backend_display "CPU (CUDA not applicable)"
    }

    di as txt _n "{hline 78}"
    di as res "Gelbach decomposition of coefficient movement"
    di as txt "{hline 78}"
    di as txt "Outcome: " as res "`y'" as txt "    Observations: " ///
        as res %12.0fc scalar(__xgel_n_obs_eff)
    di as txt "Base model: " as res "`y' on `x1' (with intercept)"
    di as txt "Full model adds: " as res `nx2g' as txt " covariate `cov_word', " ///
        as res `nfe' as txt " absorbed `fe_word'"
    if (`nabs' > 0) {
        di as txt "Estimand: " as res "absorbed-target allocation"
        di as txt "Full-model coefficient(s) imposed at zero, not estimated: " ///
            as res "`absorbed_backend_names'"
    }
    di as txt "Inference: " as res "`vce_display'"
    di as txt "Computation: " as res "`backend_display'" as txt ", " ///
        as res %5.0f scalar(__xgel_threads_used) as txt " thread(s)"
    di as txt "Movement = base coefficient - full coefficient; block contributions sum to it."
    di as txt "Interpretation: specification accounting, not causal mediation."
    if (`focal_requested') {
        di as txt "Reported focal coefficient(s): " as res "`focal'" ///
            as txt "  (all x1() columns remain in both models)"
    }
    if ("`shares'" != "") {
        local share_label = cond("`shares'" == "movement", ///
            "total movement", "base coefficient")
        di as txt "Shares: signed fraction of " as res "`share_label'" ///
            as txt "; displayed as percent; never truncated or renormalized."
        if ("`shares'" == "base") {
            di as txt "Share inference: unavailable until Cov(delta,b_base) is exposed; point shares only."
        }
        else if ("`shares'" == "base_fixed") {
            di as txt "Share inference: component SE scaled holding the reported base coefficient fixed."
        }
        else di as txt "Share inference: delta method using the joint component covariance."
    }
    if (scalar(__xgel_converged) != 1) {
        di as err "Status: NOT CONVERGED; values below are shown for diagnosis only."
    }

    foreach coef of local display_coefs {
        local r : list posof "`coef'" in coefnames
        di as txt _n "Coefficient: " as res "`coef'"
        if (`r' <= `p') {
            di as txt %18s "" %14s "Base" %14s "Full" ///
                %14s "Movement" %14s "Std. err."
            if (`ABS'[1, `r'] == 1) {
                di as res %18s "" %14.7g `BBASE'[1, `r'] ///
                    %14s "0 (imposed)" %14.7g `TOT'[`r', 1] ///
                    %14.7g `TOT'[`r', 2]
            }
            else {
                di as res %18s "" %14.7g `BBASE'[1, `r'] ///
                    %14.7g `BFULL'[1, `r'] %14.7g `TOT'[`r', 1] ///
                    %14.7g `TOT'[`r', 2]
            }
        }
        else {
            di as txt "Intercept movement: " as res %14.7g `TOT'[`r', 1] ///
                as txt "    Std. err.: " as res %14.7g `TOT'[`r', 2]
        }

        if ("`shares'" == "") {
            di as txt "{hline 64}"
            di as txt %-36s "Added block" %14s "Contribution" %14s "Std. err."
            di as txt "{hline 64}"
        }
        else {
            di as txt "{hline 78}"
            di as txt %-34s "Added block" %14s "Contribution" ///
                %14s "Std. err." %14s "Share (%)"
            di as txt "{hline 78}"
        }
        if (`nx2g' > 0) {
            di as txt "Covariate blocks"
            forvalues g = 1/`nx2g' {
                local gname : word `g' of `gnames'
                local rowlabel "  `gname'"
                if ("`shares'" == "") {
                    di as txt %-36s "`rowlabel'" as res ///
                        %14.7g `D'[`r', `g'] %14.7g `SE'[`r', `g']
                }
                else {
                    di as txt %-34s "`rowlabel'" as res ///
                        %14.7g `D'[`r', `g'] %14.7g `SE'[`r', `g'] ///
                        %14.3f (100 * `SHARE'[`r', `g'])
                }
            }
        }
        if (`nfe' > 0) {
            local first_fe = `G' - `nfe' + 1
            di as txt "Absorbed fixed effects"
            forvalues g = `first_fe'/`G' {
                local gname : word `g' of `gnames'
                local rowlabel "  `gname'"
                if ("`shares'" == "") {
                    di as txt %-36s "`rowlabel'" as res ///
                        %14.7g `D'[`r', `g'] %14.7g `SE'[`r', `g']
                }
                else {
                    di as txt %-34s "`rowlabel'" as res ///
                        %14.7g `D'[`r', `g'] %14.7g `SE'[`r', `g'] ///
                        %14.3f (100 * `SHARE'[`r', `g'])
                }
            }
            if ("`shares'" == "") {
                di as txt %-36s "  All fixed effects (subtotal)" as res ///
                    %14.7g `FEAGG'[`r', 1] %14.7g `FEAGG'[`r', 2]
            }
            else {
                local fe_share 0
                forvalues g = `first_fe'/`G' {
                    local fe_share = `fe_share' + `SHARE'[`r', `g']
                }
                di as txt %-34s "  All fixed effects (subtotal)" as res ///
                    %14.7g `FEAGG'[`r', 1] %14.7g `FEAGG'[`r', 2] ///
                    %14.3f (100 * `fe_share')
            }
        }
        if ("`shares'" == "") {
            di as txt "{hline 64}"
            di as txt %-36s "Total movement" as res ///
                %14.7g `TOT'[`r', 1] %14.7g `TOT'[`r', 2]
            di as txt "{hline 64}"
        }
        else {
            local total_share 0
            forvalues g = 1/`G' {
                local total_share = `total_share' + `SHARE'[`r', `g']
            }
            di as txt "{hline 78}"
            di as txt %-34s "Total movement" as res ///
                %14.7g `TOT'[`r', 1] %14.7g `TOT'[`r', 2] ///
                %14.3f (100 * `total_share')
            di as txt "{hline 78}"
        }
    }
    if (`share_undefined') {
        di as err "warning: at least one requested share denominator is within sharetol(); its shares are missing"
    }

    di as txt _n "Result status"
    if (scalar(__xgel_converged) == 1) {
        di as txt "  Converged: " as res "yes"
    }
    else {
        di as txt "  Converged: " as err "NO"
    }
    di as txt "  Summation check (max absolute residual): " ///
        as res %10.3e scalar(__xgel_identity_gap)
    if (`nfe' > 0) {
        di as txt "  FE standard errors are conditional on the recovered FE estimates."
    }
    if (scalar(__xgel_converged) != 1) {
        di as err "warning: computation did not pass all convergence checks; do not interpret these results"
        if ("`xgel_notes'" != "") di as err "details: `xgel_notes'"
    }
    else if (strpos(lower("`xgel_notes'"), "warning:") > 0) {
        di as err "`xgel_notes'"
    }
    di as txt "{hline 78}"

    local absorbed_inf_valid = scalar(__xgel_abs_inf_valid)
    return scalar identity_gap = scalar(__xgel_identity_gap)
    return scalar n_obs_input = scalar(__xgel_n_obs_input)
    return scalar n_obs = scalar(__xgel_n_obs)
    return scalar n_obs_effective = scalar(__xgel_n_obs_eff)
    return scalar n_singletons_dropped = scalar(__xgel_n_singletons_dropped)
    return scalar df_full = scalar(__xgel_df_full)
    return scalar fe_collinear_ss_ratio_tol = scalar(__xgel_feclass_tol)
    return scalar absorbed_target_inference_valid = scalar(__xgel_abs_inf_valid)
    return scalar absorbing_fe_index = scalar(__xgel_abs_fe_index)
    return scalar converged = scalar(__xgel_converged)
    return scalar tol = `tol'
    return scalar focal_selection_explicit = `focal_requested'
    return scalar conf_level = `level' / 100
    return scalar share_tol = `sharetol'
    foreach name in threads_used gpu_used gpu_status_code gpu_attempted ///
        gpu_absorption_converged gpu_absorption_iterations {
        return scalar `name' = scalar(__xgel_`name')
        capture scalar drop __xgel_`name'
    }
    return scalar gpu_requested = ("`gpu'" != "")
    capture scalar drop __xgel_identity_gap __xgel_n_obs_input __xgel_n_obs ///
        __xgel_n_obs_eff ///
        __xgel_n_singletons_dropped __xgel_df_full __xgel_feclass_tol ///
        __xgel_abs_inf_valid __xgel_abs_fe_index __xgel_converged
    return local vce "`vce'"
    return local groups "`gnames'"
    return local x1_names "`x1'"
    return local focal_indices "`focal_indices'"
    return local focal_names "`focal'"
    local estimand "coefficient_movement"
    local identity_status "exact_ols"
    if (`nabs' > 0) {
        local estimand "absorbed_target_allocation"
        local identity_status "exact_ols_constrained"
    }
    local b_full_status
    local focal_status
    local x_pos 0
    foreach x of local x1 {
        local ++x_pos
        local is_abs = (`ABS'[1, `x_pos'] == 1)
        if (`is_abs') {
            local b_full_status "`b_full_status' imposed_zero"
            local focal_status "`focal_status' absorbed"
        }
        else {
            local b_full_status "`b_full_status' estimated"
            local focal_status "`focal_status' identified"
        }
    }
    local b_full_status = trim("`b_full_status'")
    local focal_status = trim("`focal_status'")
    local observed_se_type "full"
    if ("`gamma0'" != "") local observed_se_type "gamma0"
    else if ("`cov0'" != "" & "`vce'" != "unadjusted") local observed_se_type "cov0"
    local total_se_type "`observed_se_type'"
    if (`nfe' > 0) {
        if ("`gamma0'" != "" | `nx2g' == 0) local total_se_type "conditional_gamma0"
        else if ("`cov0'" != "" & "`vce'" != "unadjusted") ///
            local total_se_type "mixed_cov0_observed_conditional_fe"
        else local total_se_type "mixed_full_observed_conditional_fe"
    }
    if (`nabs' > 0) local total_se_type "target_exact_base_vce_mixed_components"
    local inference_status "not_applicable"
    if (`nabs' > 0) {
        if (`absorbed_inf_valid' == 1) ///
            local inference_status "clustered_at_absorbing_fe"
        else local inference_status "warning_unsupported_vce_or_cluster"
    }
    return local estimand "`estimand'"
    return local identity_status "`identity_status'"
    return local absorbed_targets "`absorbed_backend_indices'"
    return local absorbed_target_names "`absorbed_backend_names'"
    return local b_full_status "`b_full_status'"
    return local focal_status "`focal_status'"
    return local observed_se_type "`observed_se_type'"
    return local total_se_type "`total_se_type'"
    return local inference_status "`inference_status'"
    return local causal_interpretation "no"
    return local fe_se_type "conditional_gamma0"
    return local gpu_backend "`gpu_backend'"
    return local gpu_status "`gpu_status'"
    if ("`shares'" != "") {
        return local share_denominator "`shares'"
        return local share_se_type "`share_se_type'"
        return local share_units "fraction"
        return matrix residual_share = `SHARERESID'
        return matrix share_defined = `SHAREDEF'
        return matrix share_ci_high = `SHAREHI'
        return matrix share_ci_low = `SHARELO'
        return matrix share_se = `SHARESE'
        return matrix share = `SHARE'
    }
    if ("`xgel_notes'" != "") return local notes "`xgel_notes'"
    return matrix total = `TOT'
    return matrix total_cov = `TOTCOV'
    return matrix cov = `COV'
    return matrix b_full = `BFULL'
    return matrix b_base = `BBASE'
    return matrix absorbed_mask = `ABS'
    if (`nfe' > 0) return matrix fe_total = `FEAGG'
    return matrix se = `SE'
    return matrix delta = `D'
end
