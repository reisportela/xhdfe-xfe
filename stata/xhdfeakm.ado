*! version 1.7.2  20jul2026
*! AKM estimation + leave-out (KSS) variance decomposition on the xhdfe backend.
*! Numerical semantics follow Saggio's LeaveOutTwoWay (Kline-Saggio-Soelvsten 2020);
*! identical compiled core as the Python py_hdfe_v11.akm_kss and R xhdfe_akm_kss.

program define xhdfeakm, rclass sortpreserve
    version 14.0
    syntax varname(numeric) [if] [in] [fweight], WORKer(varname numeric) FIRM(varname numeric) ///
        [ CONTROLs(varlist fv ts numeric) LEAVEOUTlevel(string) LEVerages(string)    ///
          DRAWS(integer 200) SEED(real 20260705) noPRUNE GENerate(name) REPLACE      ///
          THREADS(integer 0) EXACTMAXrows(integer 10000)                             ///
          DIRECTMAXfirms(integer 50000) CGTOL(real 1e-10) FWLTOL(real 1e-10) ///
          SE SENSIM(integer 1000) CI EIGTRACENSIM(integer 100) SIGMALOWess GPU  ///
          VERBose ]

    local y `varlist'
    marksample touse
    markout `touse' `worker' `firm'
    // controls() may contain factor / time-series terms (e.g. i.year). Expand
    // them to numeric columns for the plugin (base/omitted levels dropped, as
    // manual dummies would) and keep the expanded names for the e(b) labels.
    local control_disp "`controls'"
    local control_vars "`controls'"
    if ("`controls'" != "") {
        quietly fvexpand `controls' if `touse'
        local control_disp "`r(varlist)'"
        quietly fvrevar `controls' if `touse', substitute
        local control_vars "`r(varlist)'"
        markout `touse' `control_vars'
    }
    // Worker/firm ids are passed to the plugin as int32 (it relabels them to
    // dense indices internally, so only the int32 range matters, not the exact
    // codes). Raw ids outside int32 range (e.g. NISS/NIF person codes) or
    // non-integer ids would be rejected, so recode such an id to a compact
    // 1..N integer here. This preserves the worker-firm graph exactly, so the
    // leave-out decomposition is unchanged.
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
        if ("`se'`ci'" != "" | inlist("`leaveoutlevel'", "obs", "observation")) {
            di as err "fweights are supported for the match-level point decomposition only (expand the data for se/ci/obs; identical results)"
            exit 198
        }
    }
    quietly count if `touse'
    if (r(N) == 0) {
        error 2000
    }

    if ("`leaveoutlevel'" == "") local leaveoutlevel "match"
    if (!inlist("`leaveoutlevel'", "match", "matches", "obs", "observation")) {
        di as err "leaveoutlevel() must be match or obs"
        exit 198
    }
    if ("`leverages'" == "") local leverages "auto"
    if (!inlist("`leverages'", "auto", "exact", "jla")) {
        di as err "leverages() must be auto, exact or jla"
        exit 198
    }
    local do_prune = cond("`prune'" == "noprune", 0, 1)
    local ncontrols : word count `control_vars'

    local store_effects 0
    if ("`generate'" != "") {
        local store_effects 1
        foreach suff in alpha psi keep {
            capture confirm new variable `generate'_`suff'
            if (_rc & "`replace'" == "") {
                di as err "variable `generate'_`suff' already exists (use replace)"
                exit 110
            }
            capture drop `generate'_`suff'
        }
    }

    tempvar keepvar effalpha effpsi
    quietly gen double `keepvar' = . if `touse'
    local effect_vars ""
    if (`store_effects') {
        quietly gen double `effalpha' = . if `touse'
        quietly gen double `effpsi' = . if `touse'
        local effect_vars "`effalpha' `effpsi'"
    }
    tempname bmat
    if (`ncontrols' > 0) {
        matrix `bmat' = J(1, `ncontrols', .)
    }

    // Assemble the packed plugin config (task=akm_kss selects the AKM branch).
    local cfg "cfg=task=akm_kss;ncontrols=`ncontrols';store_effects=`store_effects';has_fweight=`has_fweight';"
    local cfg "`cfg'leave_out_level=`leaveoutlevel';leverages=`leverages';"
    local cfg "`cfg'jla_draws=`draws';seed=`seed';prune=`do_prune';"
    local cfg "`cfg'exact_max_rows=`exactmaxrows';direct_max_firms=`directmaxfirms';"
    if ("`ci'" != "") {
        local se "se"
    }
    if ("`se'" != "") {
        local cfg "`cfg'compute_se=1;se_nsim=`sensim';"
    }
    if ("`ci'" != "") {
        local cfg "`cfg'eigen_diagnostics=1;eig_trace_nsim=`eigtracensim';"
    }
    if ("`sigmalowess'" != "") {
        local cfg "`cfg'se_sigma_lowess=1;"
    }
    if ("`gpu'" != "") {
        local cfg "`cfg'use_gpu=1;"
    }
    if ("`verbose'" != "") {
        local cfg "`cfg'verbose=1;"
    }
    local cfg "`cfg'cg_tol=`cgtol';fwl_tol=`fwltol';num_threads=`threads';"
    if (`ncontrols' > 0) {
        local cfg "`cfg'b=`bmat';"
    }

    // Bind the plugin to the same directory as the active xhdfe.ado (shared
    // binary with the xhdfe command; same stale-path guards).
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
        di as err "xhdfeakm: the active session is still bound to an older xhdfe.plugin path"
        di as err "xhdfeakm: run discard (with no arguments) and rerun the command"
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
    if ("`verbose'" != "") {
        di as txt "[xhdfeakm] live progress enabled; entering the C++ backend"
    }
    capture noisily plugin call `plugin_prog' `y' `worker_use' `firm_use' `fw_var' `control_vars' ///
        `effect_vars' `keepvar' if `touse', "`cfg'"
    local rc = _rc
    if (`rc') {
        exit `rc'
    }

    // Collect the plugin scalars into r() and clean the hidden names up.
    local smap "gpu_used plugin_var_alpha plugin_var_psi plugin_cov agsu_var_alpha agsu_var_psi agsu_cov kss_var_alpha kss_var_psi kss_cov var_y sigma2_ho n_obs n_obs_input n_obs_connected n_workers n_firms n_matches n_movers n_stayers n_rows max_pii mean_pii leverages_exact solver_direct fwl_threads_used threads_used jla_draws seed solver_iterations converged"
    foreach name of local smap {
        return scalar `name' = scalar(__xakm_`name')
        capture scalar drop __xakm_`name'
    }
    if ("`se'" != "") {
        foreach nm in se_var_psi se_cov se_var_alpha theta_var_psi theta_cov theta_var_alpha {
            return scalar `nm' = scalar(__xakm_`nm')
            capture scalar drop __xakm_`nm'
        }
    }
    if ("`ci'" != "") {
        foreach comp in psi cov alpha {
            foreach nm in lambda1 eigshare1 lindeberg gammasq fstat theta1 ci_lb ci_ub curvature {
                return scalar `nm'_`comp' = scalar(__xakm_`nm'_`comp')
                capture scalar drop __xakm_`nm'_`comp'
            }
        }
    }
    // Derived summary per estimator table (pytwoway-style): correlation of
    // effects, var(alpha+psi) and the shares of var(y) explained.
    foreach est in plugin agsu kss {
        local va = return(`est'_var_alpha)
        local vp = return(`est'_var_psi)
        local cv = return(`est'_cov)
        return scalar `est'_corr = `cv' / sqrt(`va' * `vp')
        return scalar `est'_var_sum = `va' + `vp' + 2 * `cv'
        if (return(var_y) > 0) {
            return scalar `est'_share_alpha = `va' / return(var_y)
            return scalar `est'_share_psi = `vp' / return(var_y)
            return scalar `est'_share_2cov = 2 * `cv' / return(var_y)
        }
    }
    return local leave_out_level "`leaveoutlevel'"
    return local cmd "xhdfeakm"
    if ("`xakm_notes'" != "") {
        return local notes "`xakm_notes'"
    }
    if (`ncontrols' > 0) {
        if (`: word count `control_disp'' == `ncontrols') {
            matrix colnames `bmat' = `control_disp'
        }
        else {
            matrix colnames `bmat' = `control_vars'
        }
        return matrix b = `bmat'
    }

    if (`store_effects') {
        quietly gen double `generate'_alpha = `effalpha'
        quietly gen double `generate'_psi = `effpsi'
        label variable `generate'_alpha "AKM worker effect (leave-out sample)"
        label variable `generate'_psi "AKM firm effect (psi, person-year mean 0)"
    }
    if ("`generate'" != "") {
        quietly gen byte `generate'_keep = (`keepvar' == 1) if `touse'
        label variable `generate'_keep "leave-out connected sample flag"
    }

    // Display.
    di as txt ""
    di as txt "AKM + leave-out (KSS) variance decomposition" as res "  (xhdfe backend)"
    di as txt "Leave-out sample: " as res return(n_obs) as txt " obs, " ///
        as res return(n_workers) as txt " workers (" as res return(n_movers) ///
        as txt " movers), " as res return(n_firms) as txt " firms, " ///
        as res return(n_matches) as txt " matches"
    di as txt "{hline 64}"
    di as txt %-14s "estimator" %16s "var(alpha)" %16s "var(psi)" %16s "cov(a,psi)"
    di as txt "{hline 64}"
    di as txt %-14s "plug-in" as res %16.6f return(plugin_var_alpha) %16.6f return(plugin_var_psi) %16.6f return(plugin_cov)
    di as txt %-14s "AGSU (ho)" as res %16.6f return(agsu_var_alpha) %16.6f return(agsu_var_psi) %16.6f return(agsu_cov)
    di as txt %-14s "KSS (he)" as res %16.6f return(kss_var_alpha) %16.6f return(kss_var_psi) %16.6f return(kss_cov)
    if ("`se'" != "") {
        di as txt %-14s "  (SE)" as res %16.6f return(se_var_alpha) %16.6f return(se_var_psi) %16.6f return(se_cov)
    }
    if ("`ci'" != "") {
        local ci_a = "[" + strofreal(return(ci_lb_alpha), "%8.4f") + "," + strofreal(return(ci_ub_alpha), "%8.4f") + "]"
        local ci_p = "[" + strofreal(return(ci_lb_psi), "%8.4f") + "," + strofreal(return(ci_ub_psi), "%8.4f") + "]"
        local ci_c = "[" + strofreal(return(ci_lb_cov), "%8.4f") + "," + strofreal(return(ci_ub_cov), "%8.4f") + "]"
        di as txt %-14s "  AM 95% CI" as res %16s "`ci_a'" %16s "`ci_p'" %16s "`ci_c'"
    }
    di as txt "{hline 64}"
    di as txt "KSS: corr(a,psi) = " as res %7.4f return(kss_corr) ///
        as txt "  var(a+psi) = " as res %9.6f return(kss_var_sum) ///
        as txt "  shares of var(y): a " as res %5.3f return(kss_share_alpha) ///
        as txt " psi " as res %5.3f return(kss_share_psi) ///
        as txt " 2cov " as res %6.3f return(kss_share_2cov)
    local levtxt = cond(return(leverages_exact) == 1, "exact", "JLA (" + string(return(jla_draws)) + " draws, seed " + string(return(seed), "%12.0f") + ")")
    di as txt "var(y) = " as res %-12.6f return(var_y) as txt "  leverages: " as res "`levtxt'"
    if (return(converged) != 1) {
        di as err "warning: the AKM/KSS computation did not fully converge; results may be unreliable"
        if ("`xakm_notes'" != "") {
            di as err "  details: `xakm_notes'"
        }
    }
    else if (strpos(lower("`xakm_notes'"), "warning:") > 0) {
        di as err "`xakm_notes'"
    }
end
