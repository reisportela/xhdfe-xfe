*! version 1.10.1 23jul2026
program define xfe, eclass sortpreserve
    version 16.0

    // xfe: hdfe-style partialling-out using the xhdfe C++ backend (CPU/GPU).
    // Notes:
    // - This is a building block (like hdfe), so it intentionally prints no table by default.
    // - We require explicit numeric variables in varlist (no factor/time-series operators in the varlist).

    capture syntax, version
    if (!_rc) {
        local version "1.10.1 23jul2026"
        ereturn clear
        di as txt "`version'"
        ereturn local version "`version'"
        exit
    }

    if replay() {
        if ("`e(cmd)'" != "xfe") error 301
        exit
    }

    local __xfe_profile_env : environment XHDFE_PROFILE_CPU
    local __xfe_profile 0
    if ("`__xfe_profile_env'" != "" & "`__xfe_profile_env'" != "0") {
        local __xfe_profile 1
    }
    if (`__xfe_profile') {
        timer clear 99
        timer on 99
    }

    local cmdline : copy local 0

    // Normalize hdfe/reghdfe-style aliases to keep xfe drop-in friendly.
    local optline : copy local 0
    local optline : subinstr local optline ",a(" ", absorb(", all
    local optline : subinstr local optline ", a(" ", absorb(", all
    local optline : subinstr local optline " a(" " absorb(", all
    local optline : subinstr local optline ",abs(" ", absorb(", all
    local optline : subinstr local optline ", abs(" ", absorb(", all
    local optline : subinstr local optline " abs(" " absorb(", all
    local optline : subinstr local optline ",cl(" ", clustervars(", all
    local optline : subinstr local optline ", cl(" ", clustervars(", all
    local optline : subinstr local optline " cl(" " clustervars(", all
    local optline : subinstr local optline ",cluster(" ", clustervars(", all
    local optline : subinstr local optline ", cluster(" ", clustervars(", all
    local optline : subinstr local optline " cluster(" " clustervars(", all
    local optline : subinstr local optline ",dof(" ", dofadjustments(", all
    local optline : subinstr local optline ", dof(" ", dofadjustments(", all
    local optline : subinstr local optline " dof(" " dofadjustments(", all
    local optline : subinstr local optline ",groupv(" ", groupvar(", all
    local optline : subinstr local optline ", groupv(" ", groupvar(", all
    local optline : subinstr local optline " groupv(" " groupvar(", all
    local optline : subinstr local optline ",iter(" ", maxiter(", all
    local optline : subinstr local optline ", iter(" ", maxiter(", all
    local optline : subinstr local optline " iter(" " maxiter(", all
    local optline : subinstr local optline ",iterate(" ", maxiter(", all
    local optline : subinstr local optline ", iterate(" ", maxiter(", all
    local optline : subinstr local optline " iterate(" " maxiter(", all
    local optline : subinstr local optline ",iterations(" ", maxiter(", all
    local optline : subinstr local optline ", iterations(" ", maxiter(", all
    local optline : subinstr local optline " iterations(" " maxiter(", all
    local optline : subinstr local optline ",transf(" ", transform(", all
    local optline : subinstr local optline ", transf(" ", transform(", all
    local optline : subinstr local optline " transf(" " transform(", all
    local optline : subinstr local optline " maxiterations(" " maxiter(", all
    local optline : subinstr local optline ",maxiterations(" ", maxiter(", all
    local optline : subinstr local optline ", maxiterations(" ", maxiter(", all
    local 0 `"`optline'"'

    syntax varlist(min=1 numeric) [if] [in] [aw fw pw iw], ///
        ABSorb(string asis) ///
        [ ///
        CLEAR ///
        GENerate(name) ///
        SAMPLE(name) ///
        CLUSTERVars(string asis) ///
        KEEPIDs ///
        KEEPVars(varlist) ///
        VERBOSE(integer 0) ///
        TIMEIT ///
        TOLerance(real 1e-8) ///
        MAXITer(integer 100000) ///
        POOLsize(integer 10) ///
        ACCELeration(string) ///
        TRANSForm(string) ///
        PRUNE ///
        NOWARN ///
        NUMThreads(integer 0) ///
        DEFAULTThreads(integer 0) ///
        MAXThreads(integer 0) ///
        MINParallelRows(integer 20000) ///
        TARGETRowsPerThread(integer 500000) ///
        GPUBACKEND(string) ///
        ABSORPTIONMethod(string) ///
        SYMmetricSweep ///
        JACOBIRelaxation(real 0) ///
        KEEPsingletons ///
        DOFAdjustments(string asis) ///
        GROUPVAR(name) ///
        MOBilityProfile ///
        MOBfile(string) ///
        ABSORPTIONCache(string) ///
        ABSCACHEMode(string) ///
        FESTructureCache ///
        FESCache(string) ///
        FECACHEMODE(string) ///
        ]

    // Mode selection: clear vs generate().
    if (("`clear'" == "") + ("`generate'" == "") != 1) {
        di as err "xfe error: specify one and only one of: clear or generate(...)"
        exit 198
    }
    if ("`clear'" != "" & "`sample'" != "") {
        di as err "option sample() not compatible with clear"
        exit 198
    }
    if ("`generate'" != "" & "`keepids'" != "") {
        di as err "option keepids not compatible with generate()"
        exit 198
    }
    if ("`generate'" != "" & "`keepvars'" != "") {
        di as err "option keepvars() not compatible with generate()"
        exit 198
    }

    if ("`absorb'" == "") {
        di as err "option absorb() is required"
        exit 198
    }
    local unsupported_het_slopes_msg ///
        "heterogeneous slopes (i.var##c.x, var#c.x) not yet supported in xfe"

    // Expand keepvars (in clear mode) to a concrete varlist early.
    if ("`keepvars'" != "") {
        quietly ds `keepvars'
        local keepvars `r(varlist)'
    }

    // Weights.
    local has_weight 0
    local wvar
    if ("`weight'" != "") {
        local has_weight 1
        local wvar : subinstr local exp "=" "", all
        if (!inlist("`weight'", "aweight", "fweight", "pweight", "iweight")) {
            di as err "unsupported weight type: `weight'"
            exit 198
        }
    }

    // Defaults and aliases.
    local absorptionmethod = lower(strtrim("`absorptionmethod'"))
    if ("`absorptionmethod'" == "") local absorptionmethod "symmetric-gauss-seidel"
    if ("`minparallelrows'" == "") local minparallelrows 20000
    if ("`targetrowsperthread'" == "") local targetrowsperthread 500000

    if (`tolerance' <= 0 | `tolerance' >= 1) {
        di as err "tolerance() must be in (0, 1)"
        exit 198
    }
    if (`maxiter' <= 0) {
        di as err "maxiter() must be positive"
        exit 198
    }
    if (`numthreads' < 0) {
        di as err "numthreads() must be >= 0"
        exit 198
    }
    if (`defaultthreads' < 0) {
        di as err "defaultthreads() must be >= 0"
        exit 198
    }
    if (`maxthreads' < 0) {
        di as err "maxthreads() must be >= 0"
        exit 198
    }
    if (`minparallelrows' <= 0) {
        di as err "minparallelrows() must be positive"
        exit 198
    }
    if (`targetrowsperthread' <= 0) {
        di as err "targetrowsperthread() must be positive"
        exit 198
    }
    if (`jacobirelaxation' < 0) {
        di as err "jacobirelaxation() must be >= 0"
        exit 198
    }
    local method_ok = inlist("`absorptionmethod'", "auto", "gauss-seidel", "gauss_seidel", "gs")
    if (!`method_ok') {
        local method_ok = inlist("`absorptionmethod'", ///
            "symmetric-gauss-seidel", "symmetric_gauss_seidel", "sym", "symmetric", "symgs", "jacobi")
    }
    if (!`method_ok') {
        local method_ok = inlist("`absorptionmethod'", ///
            "mlsmr", "modified-lsmr", "modified_lsmr", "within", "lsmr", "auto-mlsmr", "auto_mlsmr")
    }
    if (!`method_ok') {
        di as err "absorptionmethod() must be auto, gauss-seidel, symmetric-gauss-seidel, jacobi, mlsmr, lsmr, or auto-mlsmr"
        exit 198
    }

    local gpu_backend ""
    if ("`gpubackend'" != "") {
        local gpu_backend = lower(strtrim("`gpubackend'"))
        if (!inlist("`gpu_backend'", "cpu", "cuda", "metal")) {
            di as err "gpubackend() must be one of cpu, cuda, or metal"
            exit 198
        }
    }

    local drop_singletons 1
    if ("`keepsingletons'" != "") {
        local drop_singletons 0
        if ("`nowarn'" == "") {
            di as err "WARNING: Singleton observations not dropped; statistical significance is biased (http://scorreia.com/reghdfe/nested_within_cluster.pdf)"
        }
    }

    local symmetric_sweep 0
    if ("`symmetricsweep'" != "") local symmetric_sweep 1

    // hdfe-style transform() mapping (optional; affects speed only).
    if ("`transform'" != "") {
        local tr = lower(strtrim("`transform'"))
        if (inlist("`tr'", "sym", "symmetric")) {
            local symmetric_sweep 1
        }
        else if (inlist("`tr'", "kac", "kaczmarz")) {
            // Closest analogue: Gauss-Seidel sweep.
            if ("`absorptionmethod'" == "auto") local absorptionmethod "gauss-seidel"
        }
        else if (inlist("`tr'", "cim", "cimmino")) {
            // Closest analogue: Jacobi (simultaneous projection-like).
            if ("`absorptionmethod'" == "auto") local absorptionmethod "jacobi"
        }
        else {
            di as err "transform() must be kac, cim, or sym"
            exit 198
        }
    }
    // acceleration() and poolsize() are accepted for hdfe compatibility; xfe uses the xhdfe absorber.

    // Mobility profile cache (same semantics as xhdfe).
    local mobility_profile ""
    local mobility_profile_mode ""
    if ("`mobfile'" != "") {
        local mobility_profile = strtrim("`mobfile'")
        local mobility_profile_mode "auto"
    }
    if ("`mobilityprofile'" != "") {
        if ("`mobility_profile'" == "") {
            local mobility_profile "xfe_mobility_profile.txt"
        }
        local mobility_profile_mode "write"
    }
    if ("`mobility_profile'" != "") {
        // Strip surrounding quotes to avoid passing literal quotes to the plugin.
        if (substr("`mobility_profile'", 1, 1) == `"""' & ///
            substr("`mobility_profile'", strlen("`mobility_profile'"), 1) == `"""') {
            local mobility_profile = substr("`mobility_profile'", 2, ///
                strlen("`mobility_profile'") - 2)
        }
        if (strpos("`mobility_profile'", ";") > 0) {
            di as err "mobilityprofile path cannot contain ';'"
            exit 198
        }
    }

    // Absorption cache (same semantics as xhdfe).
    local absorption_cache ""
    local absorption_cache_mode ""
    if ("`abscachemode'" != "") {
        local absorption_cache_mode = lower(strtrim("`abscachemode'"))
        if (!inlist("`absorption_cache_mode'", "off", "auto", "read", "write")) {
            di as err "abscachemode() must be off, auto, read, or write"
            exit 198
        }
    }
    if ("`absorptioncache'" != "") {
        local absorption_cache = strtrim("`absorptioncache'")
        if ("`absorption_cache_mode'" == "") {
            local absorption_cache_mode "auto"
        }
    }
    if ("`absorption_cache'" != "") {
        if (substr("`absorption_cache'", 1, 1) == `"""' & ///
            substr("`absorption_cache'", strlen("`absorption_cache'"), 1) == `"""') {
            local absorption_cache = substr("`absorption_cache'", 2, ///
                strlen("`absorption_cache'") - 2)
        }
        if (strpos("`absorption_cache'", ";") > 0) {
            di as err "absorptioncache path cannot contain ';'"
            exit 198
        }
    }

    // Fixed-effect structure cache (same semantics as xhdfe).
    local fe_structure_cache ""
    local fe_structure_cache_mode ""
    if ("`fecachemode'" != "") {
        local fe_structure_cache_mode = lower(strtrim("`fecachemode'"))
        if (!inlist("`fe_structure_cache_mode'", "off", "auto", "read", "write")) {
            di as err "fescachemode() must be off, auto, read, or write"
            exit 198
        }
    }
    if ("`fescache'" != "") {
        local fe_structure_cache = strtrim("`fescache'")
        if ("`fe_structure_cache_mode'" == "") {
            local fe_structure_cache_mode "auto"
        }
    }
    if ("`festructurecache'" != "") {
        if ("`fe_structure_cache'" != "") {
            di as err "festructurecache cannot be combined with fescache()"
            exit 198
        }
        local ado_path ""
        capture findfile xfe.ado
        if (!_rc) local ado_path "`r(fn)'"
        local cache_dir ""
        if ("`ado_path'" != "") {
            local cache_dir = substr("`ado_path'", 1, strlen("`ado_path'") - strlen("xfe.ado"))
        }
        if ("`cache_dir'" == "") {
            local cache_dir "`c(pwd)'/"
        }
        local fe_structure_cache "`cache_dir'xfe_fe_structure_cache.bin"
        if ("`fe_structure_cache_mode'" == "") {
            local fe_structure_cache_mode "auto"
        }
    }
    if ("`fe_structure_cache_mode'" != "" & "`fe_structure_cache'" == "") {
        di as err "fescachemode() requires fescache() or festructurecache"
        exit 198
    }
    if ("`fe_structure_cache'" != "") {
        // Strip surrounding quotes to avoid passing literal quotes to the plugin.
        if (substr("`fe_structure_cache'", 1, 1) == `"""' & ///
            substr("`fe_structure_cache'", strlen("`fe_structure_cache'"), 1) == `"""') {
            local fe_structure_cache = substr("`fe_structure_cache'", 2, ///
                strlen("`fe_structure_cache'") - 2)
        }
        if (strpos("`fe_structure_cache'", ";") > 0) {
            di as err "festructurecache path cannot contain ';'"
            exit 198
        }
    }

    // Parse absorb() tokens (supports categorical factor prefixes and #/## interactions).
    local absorb_markvars
    local absorb_ordered
    local absorb_display
    local absorb_raw = strtrim("`absorb'")
    local absorb_vars "`absorb_raw'"
    local absorb_opts ""
    if (strpos("`absorb_raw'", ",") > 0) {
        gettoken absorb_vars absorb_opts : absorb_raw, parse(",")
        local absorb_vars = strtrim("`absorb_vars'")
        local absorb_opts : subinstr local absorb_opts "," "", all
        local absorb_opts = strtrim("`absorb_opts'")
    }
    if ("`absorb_opts'" != "") {
        di as err "absorb() suboptions are not supported in xfe"
        exit 198
    }
    if ("`absorb_vars'" != "") {
        foreach tok of local absorb_vars {
            local tok = strtrim("`tok'")
            if ("`tok'" == "") continue

            // Match reghdfe/xhdfe: categorical ## in absorb() is treated as
            // the joint FE. Continuous-slope ## is unsupported and rejected.
            local include_main 0
            local tok_clean : subinstr local tok "##" "#", all

            if (strpos("`tok_clean'", "#") > 0) {
                local parts
                local rest "`tok_clean'"
                while ("`rest'" != "") {
                    gettoken part rest : rest, parse("#")
                    if ("`part'" == "#") continue
                    local clean "`part'"
                    if (substr("`clean'", 1, 2) == "c.") {
                        di as err "`unsupported_het_slopes_msg'"
                        exit 198
                    }
                    if (substr("`clean'", 1, 2) == "i.") local clean = substr("`clean'", 3, .)
                    else if (substr("`clean'", 1, 4) == "ibn.") local clean = substr("`clean'", 5, .)
                    else if (substr("`clean'", 1, 3) == "bn.") local clean = substr("`clean'", 4, .)
                    else if (regexm("`clean'", "^ib[0-9]+\\.")) local clean = regexr("`clean'", "^ib[0-9]+\\.", "")
                    else if (regexm("`clean'", "^b[0-9]+\\.")) local clean = regexr("`clean'", "^b[0-9]+\\.", "")
                    unab clean_list : `clean'
                    local clean_count : word count `clean_list'
                    if (`clean_count' != 1) {
                        di as err "absorb() interaction component `clean' is ambiguous"
                        exit 198
                    }
                    local clean : word 1 of `clean_list'
                    local parts "`parts' `clean'"
                }
                local nparts : word count `parts'
                if (`nparts' != 2) {
                    di as err "absorb() interactions must have exactly two components"
                    exit 198
                }
                local v1 : word 1 of `parts'
                local v2 : word 2 of `parts'

                local pos : list posof "`v1'" in absorb_markvars
                if (`pos' == 0) local absorb_markvars "`absorb_markvars' `v1'"
                local pos : list posof "`v2'" in absorb_markvars
                if (`pos' == 0) local absorb_markvars "`absorb_markvars' `v2'"

                if (`include_main') {
                    local pos : list posof "`v1'" in absorb_ordered
                    if (`pos' == 0) local absorb_ordered "`absorb_ordered' `v1'"
                    local pos : list posof "`v2'" in absorb_ordered
                    if (`pos' == 0) local absorb_ordered "`absorb_ordered' `v2'"
                }

                local inter_token "`v1'#`v2'"
                local pos : list posof "`inter_token'" in absorb_ordered
                if (`pos' == 0) local absorb_ordered "`absorb_ordered' `inter_token'"
            }
            else {
                local clean "`tok_clean'"
                if (substr("`clean'", 1, 2) == "c.") {
                    di as err "`unsupported_het_slopes_msg'"
                    exit 198
                }
                if (substr("`clean'", 1, 2) == "i.") local clean = substr("`clean'", 3, .)
                else if (substr("`clean'", 1, 4) == "ibn.") local clean = substr("`clean'", 5, .)
                else if (substr("`clean'", 1, 3) == "bn.") local clean = substr("`clean'", 4, .)
                else if (regexm("`clean'", "^ib[0-9]+\\.")) local clean = regexr("`clean'", "^ib[0-9]+\\.", "")
                else if (regexm("`clean'", "^b[0-9]+\\.")) local clean = regexr("`clean'", "^b[0-9]+\\.", "")
                unab clean_list : `clean'
                local clean_count : word count `clean_list'
                if (`clean_count' != 1) {
                    di as err "absorb() variable `clean' is ambiguous"
                    exit 198
                }
                local clean : word 1 of `clean_list'
                local pos : list posof "`clean'" in absorb_markvars
                if (`pos' == 0) local absorb_markvars "`absorb_markvars' `clean'"
                local pos : list posof "`clean'" in absorb_ordered
                if (`pos' == 0) local absorb_ordered "`absorb_ordered' `clean'"
            }
        }
    }
    local absorb_display "`absorb_ordered'"

    // Parse clustervars() tokens (same grammar as absorb()).
    local cluster_markvars
    local cluster_ordered
    local cluster_display
    if ("`clustervars'" != "") {
        local cluster_raw = strtrim("`clustervars'")
        foreach tok of local cluster_raw {
            local tok = strtrim("`tok'")
            if ("`tok'" == "") continue

            local include_main 0
            if (strpos("`tok'", "##") > 0) local include_main 1
            local tok_clean : subinstr local tok "##" "#", all

            if (strpos("`tok_clean'", "#") > 0) {
                local parts
                local rest "`tok_clean'"
                while ("`rest'" != "") {
                    gettoken part rest : rest, parse("#")
                    if ("`part'" == "#") continue
                    local clean "`part'"
                    if (substr("`clean'", 1, 2) == "c.") {
                        di as err "`unsupported_het_slopes_msg'"
                        exit 198
                    }
                    if (substr("`clean'", 1, 2) == "i.") local clean = substr("`clean'", 3, .)
                    else if (substr("`clean'", 1, 4) == "ibn.") local clean = substr("`clean'", 5, .)
                    else if (substr("`clean'", 1, 3) == "bn.") local clean = substr("`clean'", 4, .)
                    else if (regexm("`clean'", "^ib[0-9]+\\.")) local clean = regexr("`clean'", "^ib[0-9]+\\.", "")
                    else if (regexm("`clean'", "^b[0-9]+\\.")) local clean = regexr("`clean'", "^b[0-9]+\\.", "")
                    unab clean_list : `clean'
                    local clean_count : word count `clean_list'
                    if (`clean_count' != 1) {
                        di as err "clustervars() interaction component `clean' is ambiguous"
                        exit 198
                    }
                    local clean : word 1 of `clean_list'
                    local parts "`parts' `clean'"
                }
                local nparts : word count `parts'
                if (`nparts' != 2) {
                    di as err "clustervars() interactions must have exactly two components"
                    exit 198
                }
                local v1 : word 1 of `parts'
                local v2 : word 2 of `parts'

                local pos : list posof "`v1'" in cluster_markvars
                if (`pos' == 0) local cluster_markvars "`cluster_markvars' `v1'"
                local pos : list posof "`v2'" in cluster_markvars
                if (`pos' == 0) local cluster_markvars "`cluster_markvars' `v2'"

                if (`include_main') {
                    local pos : list posof "`v1'" in cluster_ordered
                    if (`pos' == 0) local cluster_ordered "`cluster_ordered' `v1'"
                    local pos : list posof "`v2'" in cluster_ordered
                    if (`pos' == 0) local cluster_ordered "`cluster_ordered' `v2'"
                }

                local inter_token "`v1'#`v2'"
                local pos : list posof "`inter_token'" in cluster_ordered
                if (`pos' == 0) local cluster_ordered "`cluster_ordered' `inter_token'"
            }
            else {
                local clean "`tok_clean'"
                if (substr("`clean'", 1, 2) == "c.") {
                    di as err "`unsupported_het_slopes_msg'"
                    exit 198
                }
                if (substr("`clean'", 1, 2) == "i.") local clean = substr("`clean'", 3, .)
                else if (substr("`clean'", 1, 4) == "ibn.") local clean = substr("`clean'", 5, .)
                else if (substr("`clean'", 1, 3) == "bn.") local clean = substr("`clean'", 4, .)
                else if (regexm("`clean'", "^ib[0-9]+\\.")) local clean = regexr("`clean'", "^ib[0-9]+\\.", "")
                else if (regexm("`clean'", "^b[0-9]+\\.")) local clean = regexr("`clean'", "^b[0-9]+\\.", "")
                unab clean_list : `clean'
                local clean_count : word count `clean_list'
                if (`clean_count' != 1) {
                    di as err "clustervars() variable `clean' is ambiguous"
                    exit 198
                }
                local clean : word 1 of `clean_list'
                local pos : list posof "`clean'" in cluster_markvars
                if (`pos' == 0) local cluster_markvars "`cluster_markvars' `clean'"
                local pos : list posof "`clean'" in cluster_ordered
                if (`pos' == 0) local cluster_ordered "`cluster_ordered' `clean'"
            }
        }
    }
    local cluster_display "`cluster_ordered'"

    // Sample restriction and missing values.
    marksample touse, strok
    local markvars_raw "`varlist' `absorb_markvars' `cluster_markvars' `wvar'"
    local markvars
    foreach v of local markvars_raw {
        local pos : list posof "`v'" in markvars
        if (`pos' == 0) local markvars "`markvars' `v'"
    }
    // Keep string absorb()/cluster()/group()/individual() variables in sample marking.
    markout `touse' `markvars', strok
    if (`has_weight') {
        quietly replace `touse' = 0 if `wvar' <= 0 | missing(`wvar')
    }
    if (`__xfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=xfe_sample_marking ms=" %12.3f (1000 * r(t99))
        timer clear 99
        timer on 99
    }

    // Build interaction fixed effects after the estimation sample is defined.
    local absorb_list ""
    if ("`absorb_ordered'" != "") {
        foreach entry of local absorb_ordered {
            if (strpos("`entry'", "#") > 0) {
                local pair "`entry'"
                gettoken v1 rest : pair, parse("#")
                gettoken hash v2 : rest, parse("#")
                if ("`v1'" == "" | "`v2'" == "") {
                    di as err "absorb() interactions must have exactly two components"
                    exit 198
                }
                tempvar feint
                quietly egen long `feint' = group(`v1' `v2') if `touse'
                local absorb_list "`absorb_list' `feint'"
            }
            else {
                local absorb_list "`absorb_list' `entry'"
            }
        }
    }

    // Build interaction cluster variables after the estimation sample is defined.
    local cluster_list ""
    if ("`cluster_ordered'" != "") {
        foreach entry of local cluster_ordered {
            if (strpos("`entry'", "#") > 0) {
                local pair "`entry'"
                gettoken v1 rest : pair, parse("#")
                gettoken hash v2 : rest, parse("#")
                if ("`v1'" == "" | "`v2'" == "") {
                    di as err "clustervars() interactions must have exactly two components"
                    exit 198
                }
                tempvar clint
                quietly egen long `clint' = group(`v1' `v2') if `touse'
                local cluster_list "`cluster_list' `clint'"
            }
            else {
                local cluster_list "`cluster_list' `entry'"
            }
        }
    }

    // Encode absorb() and clustervars() IDs only when needed.
    // Numeric raw IDs are passed through to the plugin, which mirrors xhdfe's
    // exact categorical compression for non-integer float/double values.
    // keepids forces materialized IDs so they can be preserved in clear mode.
    local materialize_ids 0
    if ("`keepids'" != "") local materialize_ids 1

    local nfe 0
    local fe_ids
    local fe_ids_generated
    local fe_raw
    if ("`absorb_list'" != "") {
        local idx = 0
        foreach v of varlist `absorb_list' {
            local ++idx
            local ++nfe
            local fe_raw "`fe_raw' `v'"

            local t : type `v'
            local feid "`v'"
            if (`materialize_ids' | substr("`t'", 1, 3) == "str") {
                local idname "__xfe_ID`idx'__"
                capture confirm new variable `idname'
                if (_rc) {
                    // If a previous run left it around, reuse.
                    capture drop `idname'
                    confirm new variable `idname'
                }
                if (substr("`t'", 1, 3) == "str" | inlist("`t'", "float", "double")) {
                    quietly egen long `idname' = group(`v') if `touse'
                }
                else {
                    quietly gen long `idname' = `v' if `touse'
                }
                local feid "`idname'"
                local fe_ids_generated "`fe_ids_generated' `idname'"
            }
            local fe_ids "`fe_ids' `feid'"
        }
    }

    local nclust 0
    local clust_ids
    local clust_ids_generated
    local cluster_fe_map
    if ("`cluster_list'" != "") {
        local cidx = 0
        foreach v of varlist `cluster_list' {
            local ++cidx
            local ++nclust
            local cluster_fe_pos -1

            // Reuse absorb() encodings when the same variable appears in absorb() and clustervars().
            local pos : list posof "`v'" in fe_raw
            if (`pos' > 0) {
                local clid : word `pos' of `fe_ids'
                local cluster_fe_pos = `pos' - 1
            }
            else {
                local t : type `v'
                local clid "`v'"
                if (`materialize_ids' | substr("`t'", 1, 3) == "str") {
                    local idname "__xfe_CL`cidx'__"
                    capture confirm new variable `idname'
                    if (_rc) {
                        capture drop `idname'
                        confirm new variable `idname'
                    }
                    if (substr("`t'", 1, 3) == "str" | inlist("`t'", "float", "double")) {
                        quietly egen long `idname' = group(`v') if `touse'
                    }
                    else {
                        quietly gen long `idname' = `v' if `touse'
                    }
                    local clid "`idname'"
                    local clust_ids_generated "`clust_ids_generated' `idname'"
                }
            }
            local clust_ids "`clust_ids' `clid'"
            local cluster_fe_map "`cluster_fe_map' `cluster_fe_pos'"
        }
    }
    if (`__xfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=xfe_fe_cluster_ids ms=" %12.3f (1000 * r(t99))
        timer clear 99
        timer on 99
    }

    // Output variables (residualized vars).
    local k : word count `varlist'
    local out_vars
    if ("`generate'" != "") {
        foreach v of local varlist {
            local out "`generate'`v'"
            confirm new variable `out'
            quietly gen double `out' = .
            local out_vars "`out_vars' `out'"
        }
    }
    else {
        foreach v of local varlist {
            tempvar out
            quietly gen double `out' = .
            local out_vars "`out_vars' `out'"
        }
    }

    local store_groupvar 0
    if ("`groupvar'" != "") {
        confirm new variable `groupvar'
        quietly gen long `groupvar' = .
        local store_groupvar 1
    }

    // e(sample)-style marker (prefilled with candidate sample).
    local esample "`sample'"
    if ("`esample'" != "") {
        confirm new variable `esample'
    }
    else {
        tempvar esample
    }
    quietly gen byte `esample' = `touse'

    // Prepare scalars to receive plugin outputs.
    tempname sN sNfull sNs sDfa sDfaL sDfaE sDfaN sIt sConv sThr sGpuUsed sGpuStatus sGpuAttempted sGpuAbsConv sGpuAbsIter sMeth
    scalar `sN' = .
    scalar `sNfull' = .
    scalar `sNs' = .
    scalar `sDfa' = .
    scalar `sDfaL' = .
    scalar `sDfaE' = .
    scalar `sDfaN' = .
    scalar `sIt' = .
    scalar `sConv' = .
    scalar `sThr' = .
    scalar `sGpuUsed' = 0
    scalar `sGpuStatus' = 0
    scalar `sGpuAttempted' = 0
    scalar `sGpuAbsConv' = .
    scalar `sGpuAbsIter' = .
    scalar `sMeth' = .

    // Assemble plugin varlist in the exact order expected by the plugin's xfe mode.
    local plugin_varlist "`varlist' `fe_ids' `clust_ids'"
    if (`has_weight') local plugin_varlist "`plugin_varlist' `wvar'"
    local plugin_varlist "`plugin_varlist' `out_vars'"
    if (`store_groupvar') local plugin_varlist "`plugin_varlist' `groupvar'"
    local plugin_varlist "`plugin_varlist' `esample'"

    // Build packed plugin args.
    // Align tolerance handling with xhdfe: pass tolerance directly to backend.
    local tol_backend = `tolerance'
    local cfg "cfg="
    local cfg "`cfg'k=`k';"
    local cfg "`cfg'nfe=`nfe';"
    local cfg "`cfg'nclust=`nclust';"
    local cfg "`cfg'has_weight=`has_weight';"
    if (`has_weight') local cfg "`cfg'weight_type=`weight';"
    local cfg "`cfg'store_groupvar=`store_groupvar';"
    local cfg "`cfg'esample_prefilled=1;"
    local cfg "`cfg'tol=`tol_backend';"
    local cfg "`cfg'max_iter=`maxiter';"
    local cfg "`cfg'num_threads=`numthreads';"
    local cfg "`cfg'drop_singletons=`drop_singletons';"
    local cfg "`cfg'symmetric_sweep=`symmetric_sweep';"
    local cfg "`cfg'absorption_method=`absorptionmethod';"
    local cfg "`cfg'jacobi_relaxation=`jacobirelaxation';"
    local cfg "`cfg'default_threads=`defaultthreads';"
    local cfg "`cfg'max_threads=`maxthreads';"
    local cfg "`cfg'min_parallel_rows=`minparallelrows';"
    local cfg "`cfg'target_rows_per_thread=`targetrowsperthread';"
    if ("`gpu_backend'" != "") local cfg "`cfg'gpu_backend=`gpu_backend';"
    if ("`dofadjustments'" != "") local cfg "`cfg'dofadjustments=`dofadjustments';"
    if ("`cluster_fe_map'" != "") {
        local cluster_fe_map : list retokenize cluster_fe_map
        local cluster_fe_map_cfg : subinstr local cluster_fe_map " " "," , all
        local cfg "`cfg'cluster_fe_map=`cluster_fe_map_cfg';"
    }

    // Optional cache/profile plumbing (same keys used by xhdfe).
    if ("`mobility_profile'" != "") local cfg "`cfg'mobility_profile=`mobility_profile';"
    if ("`mobility_profile_mode'" != "") local cfg "`cfg'mobility_profile_mode=`mobility_profile_mode';"
    if ("`absorption_cache'" != "") local cfg "`cfg'absorption_cache=`absorption_cache';"
    if ("`absorption_cache_mode'" != "") local cfg "`cfg'absorption_cache_mode=`absorption_cache_mode';"
    if ("`fe_structure_cache'" != "") local cfg "`cfg'fe_structure_cache=`fe_structure_cache';"
    if ("`fe_structure_cache_mode'" != "") local cfg "`cfg'fe_structure_cache_mode=`fe_structure_cache_mode';"

    // Output scalars.
    local cfg "`cfg's_N=`sN';"
    local cfg "`cfg's_N_full=`sNfull';"
    local cfg "`cfg's_num_singletons=`sNs';"
    local cfg "`cfg's_df_a=`sDfa';"
    local cfg "`cfg's_df_a_levels=`sDfaL';"
    local cfg "`cfg's_df_a_exact=`sDfaE';"
    local cfg "`cfg's_df_a_nested=`sDfaN';"
    local cfg "`cfg's_iterations=`sIt';"
    local cfg "`cfg's_converged=`sConv';"
    local cfg "`cfg's_threads_used=`sThr';"
    local cfg "`cfg's_gpu_used=`sGpuUsed';"
    local cfg "`cfg's_gpu_status_code=`sGpuStatus';s_gpu_attempted=`sGpuAttempted';"
    local cfg "`cfg's_gpu_absorption_converged=`sGpuAbsConv';s_gpu_absorption_iterations=`sGpuAbsIter';"
    local cfg "`cfg's_method_used=`sMeth';"

    // Bind the plugin to the same directory as the active xfe.ado so we do not
    // silently pick up a stale CPU-only plugin from another adopath entry.
    quietly findfile xfe.ado
    local plugin_path "`r(fn)'"
    local plugin_path : subinstr local plugin_path "xfe.ado" "xfe.plugin", all
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
        di as err "xfe.plugin not found next to xfe.ado; rebuild the plugin in `plugin_path'"
        exit 198
    }
    local plugin_prog "__xfe_plugin_dispatch"
    if ("$XFE_PLUGIN_PATH_INTERNAL" != "" & "$XFE_PLUGIN_PATH_INTERNAL" != "`plugin_path'") {
        di as err "xfe: the active session is still bound to an older xfe.plugin path"
        di as err "xfe: run discard (with no arguments) and rerun the command so Stata reloads the current plugin"
        exit 498
    }
    capture program `plugin_prog', plugin using("`plugin_path'")
    if (_rc & _rc != 110) {
        di as err "xfe.plugin could not be loaded from `plugin_path'"
        exit _rc
    }
    global XFE_PLUGIN_PATH_INTERNAL "`plugin_path'"

    // Clear previous results.
    ereturn clear

    if (`__xfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=xfe_pre_plugin ms=" %12.3f (1000 * r(t99))
        timer clear 99
        timer on 99
    }
    if ("`timeit'" != "") {
        timer clear 91
        timer on 91
    }
    capture noisily plugin call `plugin_prog' `plugin_varlist' if `touse', "`cfg'"
    local rc = _rc
    if ("`timeit'" != "") {
        timer off 91
        quietly timer list 91
        di as txt "xfe: plugin time = " %9.3f r(t91) " seconds"
        timer clear 91
    }
    if (`__xfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=xfe_plugin_call ms=" %12.3f (1000 * r(t99))
        timer clear 99
        timer on 99
    }
    if (`rc') {
        if (`rc' == 2000) {
            quietly count if `touse'
            if (r(N) == 0) {
                di as err "xfe: no observations after applying if/in and removing missing values"
            }
            else {
                di as err "xfe: no observations left for estimation (all observations may have been dropped as singletons)"
            }
        }
        exit `rc'
    }

    local gpu_status_code = scalar(`sGpuStatus')
    local gpu_status "not_requested"
    if (`gpu_status_code' == 1) local gpu_status "used"
    else if (`gpu_status_code' == 2) local gpu_status "backend_unavailable"
    else if (`gpu_status_code' == 3) local gpu_status "gpu_absorption_not_converged"
    else if (`gpu_status_code' == 4) local gpu_status "gpu_backend_failed"
    else if (`gpu_status_code' == 5) local gpu_status "cpu_cache_or_profile_result"

    // When the user explicitly requests a GPU backend, do not silently accept
    // CPU results. This catches stale plugin loads, unavailable GPU runtimes,
    // non-converged GPU absorption, and CPU cache/profile reuse.
    if ("`gpu_backend'" != "" & "`gpu_backend'" != "cpu" & scalar(`sGpuUsed') < 0.5) {
        local gpu_backend_disp = upper("`gpu_backend'")
        if ("`gpu_status'" == "backend_unavailable") {
            di as err "xfe: requested gpubackend(`gpu_backend'), but `gpu_backend_disp' was not available at runtime"
            di as err "xfe: check the GPU device/runtime and that xfe.plugin was built with `gpu_backend' support"
            di as err "xfe: if you recently rebuilt or switched plugins, run discard (with no arguments) and rerun the command"
        }
        else if ("`gpu_status'" == "gpu_absorption_not_converged") {
            di as err "xfe: requested gpubackend(`gpu_backend'), but `gpu_backend_disp' absorption did not converge"
            if (scalar(`sGpuAbsIter') < .) {
                di as err "xfe: `gpu_backend_disp' absorption reached " %9.0g scalar(`sGpuAbsIter') " iterations without convergence"
            }
            if (scalar(`sConv') == 0 & scalar(`sIt') < .) {
                di as err "xfe: CPU fallback also failed to converge in " %9.0g scalar(`sIt') " iterations"
            }
            else {
                di as err "xfe: CPU fallback produced a result, but it was rejected because a GPU backend was explicitly requested"
            }
        }
        else if ("`gpu_status'" == "gpu_backend_failed") {
            di as err "xfe: requested gpubackend(`gpu_backend'), but `gpu_backend_disp' absorption failed before producing a converged result"
            di as err "xfe: CPU fallback was rejected because a GPU backend was explicitly requested"
        }
        else if ("`gpu_status'" == "cpu_cache_or_profile_result") {
            di as err "xfe: requested gpubackend(`gpu_backend'), but the effective absorption result came from CPU cache/profile state"
            di as err "xfe: rerun with matching GPU cache/profile state, disable absorption cache, or clear stale mobility/profile files"
        }
        else {
            di as err "xfe: requested gpubackend(`gpu_backend') but the effective backend was CPU"
            di as err "xfe: rebuild the plugin with `gpu_backend' support and ensure the requested backend is available at runtime"
            di as err "xfe: if you recently rebuilt or switched plugins, run discard (with no arguments) and rerun the command"
        }
        exit 498
    }

    // Convergence gate.
    if (scalar(`sConv') != 1) {
        di as err "xfe: absorber did not converge"
        exit 430
    }

    // Copy residualized values back to original vars in clear mode.
    if ("`clear'" != "") {
        local i = 0
        foreach v of local varlist {
            local ++i
            local out : word `i' of `out_vars'
            quietly recast double `v'
            quietly replace `v' = `out' if `esample'
        }
        quietly keep if `esample'

        // Keep transformed variables + ancillary vars.
        local keep_list "`varlist'"
        if ("`keepvars'" != "") local keep_list "`keep_list' `keepvars'"
        if ("`absorb_markvars'" != "") local keep_list "`keep_list' `absorb_markvars'"
        if ("`cluster_markvars'" != "") local keep_list "`keep_list' `cluster_markvars'"
        if (`has_weight') local keep_list "`keep_list' `wvar'"
        if (`store_groupvar') local keep_list "`keep_list' `groupvar'"
        if ("`keepids'" != "") {
            if ("`fe_ids'" != "") local keep_list "`keep_list' `fe_ids'"
            if ("`clust_ids'" != "") local keep_list "`keep_list' `clust_ids'"
        }
        quietly keep `keep_list'
    }

    // Label residualized variables in generate() mode.
    if ("`generate'" != "") {
        local i = 0
        foreach v of local varlist {
            local ++i
            local out : word `i' of `out_vars'
            local label : variable label `v'
            if ("`label'" == "") local label "`v'"
            label var `out' "Residuals: `label'"
        }
        if ("`sample'" != "") {
            label var `sample' "[XFE Sample]"
        }

        // Drop plugin-only id variables (keepids is not allowed with generate()).
        if ("`fe_ids_generated'" != "") capture drop `fe_ids_generated'
        if ("`clust_ids_generated'" != "") capture drop `clust_ids_generated'
    }

    // Stored results (hdfe-compatible subset).
    ereturn scalar df_a = scalar(`sDfa')
    ereturn scalar N_hdfe = `nfe'
    ereturn scalar N = scalar(`sN')
    ereturn scalar N_full = scalar(`sNfull')
    ereturn scalar num_singletons = scalar(`sNs')
    ereturn scalar iterations = scalar(`sIt')
    ereturn scalar converged = scalar(`sConv')
    ereturn scalar threads_used = scalar(`sThr')
    ereturn scalar gpu_used = scalar(`sGpuUsed')
    ereturn scalar gpu_status_code = scalar(`sGpuStatus')
    ereturn scalar gpu_attempted = scalar(`sGpuAttempted')
    ereturn scalar gpu_absorption_converged = scalar(`sGpuAbsConv')
    ereturn scalar gpu_absorption_iterations = scalar(`sGpuAbsIter')
    ereturn scalar method_used = scalar(`sMeth')
    ereturn scalar df_a_levels = scalar(`sDfaL')
    ereturn scalar df_a_exact = scalar(`sDfaE')
    ereturn scalar df_a_nested = scalar(`sDfaN')
    local method_name "auto"
    if (scalar(`sMeth') == 1) local method_name "gauss-seidel"
    else if (scalar(`sMeth') == 2) local method_name "symmetric-gauss-seidel"
    else if (scalar(`sMeth') == 3) local method_name "jacobi"
    ereturn local absvars "`absorb_display'"
    ereturn local extended_absvars "`absorb_display'"
    ereturn local absorption_method "`method_name'"
    ereturn local gpu_status "`gpu_status'"
    if ("`gpu_backend'" != "") {
        ereturn local gpu_backend_requested "`gpu_backend'"
        if (e(gpu_used) > 0.5) ereturn local gpu_backend "`gpu_backend'"
        else ereturn local gpu_backend "cpu"
    }
    ereturn local version "1.10.1 23jul2026"
    ereturn local cmd "xfe"
    ereturn local cmdline `"`cmdline'"'
    if (`__xfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=xfe_post_plugin ms=" %12.3f (1000 * r(t99))
        timer clear 99
    }
end
