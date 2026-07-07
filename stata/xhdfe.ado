*! version 2.13.0 07jul2026
program define xhdfe, eclass sortpreserve
    version 16.0

    capture syntax, version
    if (!_rc) {
        local version "2.13.0 07jul2026"
        ereturn clear
        di as txt "`version'"
        ereturn local version "`version'"
        exit
    }

    if replay() {
        if ("`e(cmd)'" != "xhdfe") {
            error 301
        }
        xhdfe_display `0'
        exit
    }

    local __xhdfe_profile_env : environment XHDFE_PROFILE_CPU
    local __xhdfe_profile 0
    if ("`__xhdfe_profile_env'" != "" & "`__xhdfe_profile_env'" != "0") {
        local __xhdfe_profile 1
    }
    if (`__xhdfe_profile') {
        timer clear 99
        timer on 99
    }

    local cmdline : copy local 0
    // Normalize short option aliases for reghdfe-style usage.
    local optline : copy local 0
    local optline : subinstr local optline ",g(" ", group(", all
    local optline : subinstr local optline ", g(" ", group(", all
    local optline : subinstr local optline " g(" " group(", all
    local optline : subinstr local optline ",group_id(" ", group(", all
    local optline : subinstr local optline ", group_id(" ", group(", all
    local optline : subinstr local optline " group_id(" " group(", all
    local optline : subinstr local optline ",i(" ", individual(", all
    local optline : subinstr local optline ", i(" ", individual(", all
    local optline : subinstr local optline " i(" " individual(", all
    local optline : subinstr local optline ",ind(" ", individual(", all
    local optline : subinstr local optline ", ind(" ", individual(", all
    local optline : subinstr local optline " ind(" " individual(", all
    local optline : subinstr local optline ",individual_id(" ", individual(", all
    local optline : subinstr local optline ", individual_id(" ", individual(", all
    local optline : subinstr local optline " individual_id(" " individual(", all
    local optline : subinstr local optline ",ag(" ", aggregation(", all
    local optline : subinstr local optline ", ag(" ", aggregation(", all
    local optline : subinstr local optline " ag(" " aggregation(", all
    local optline : subinstr local optline ",a(" ", absorb(", all
    local optline : subinstr local optline ", a(" ", absorb(", all
    local optline : subinstr local optline " a(" " absorb(", all
    local optline : subinstr local optline ",abs(" ", absorb(", all
    local optline : subinstr local optline ", abs(" ", absorb(", all
    local optline : subinstr local optline " abs(" " absorb(", all
    local optline : subinstr local optline ",cl(" ", cluster(", all
    local optline : subinstr local optline ", cl(" ", cluster(", all
    local optline : subinstr local optline " cl(" " cluster(", all
    local optline : subinstr local optline ",dof(" ", dofadjustments(", all
    local optline : subinstr local optline ", dof(" ", dofadjustments(", all
    local optline : subinstr local optline " dof(" " dofadjustments(", all
    local optline : subinstr local optline ",iter(" ", maxiter(", all
    local optline : subinstr local optline ", iter(" ", maxiter(", all
    local optline : subinstr local optline " iter(" " maxiter(", all
    local optline : subinstr local optline ",iterate(" ", maxiter(", all
    local optline : subinstr local optline ", iterate(" ", maxiter(", all
    local optline : subinstr local optline " iterate(" " maxiter(", all
    local optline : subinstr local optline ",iterations(" ", maxiter(", all
    local optline : subinstr local optline ", iterations(" ", maxiter(", all
    local optline : subinstr local optline " iterations(" " maxiter(", all
    local optline : subinstr local optline ",tech(" ", technique(", all
    local optline : subinstr local optline ", tech(" ", technique(", all
    local optline : subinstr local optline " tech(" " technique(", all
    local optline : subinstr local optline ",transf(" ", transform(", all
    local optline : subinstr local optline ", transf(" ", transform(", all
    local optline : subinstr local optline " transf(" " transform(", all
    local optline : subinstr local optline ",prec(" ", preconditioner(", all
    local optline : subinstr local optline ", prec(" ", preconditioner(", all
    local optline : subinstr local optline " prec(" " preconditioner(", all
    local optline : subinstr local optline ",par(" ", parallel(", all
    local optline : subinstr local optline ", par(" ", parallel(", all
    local optline : subinstr local optline " par(" " parallel(", all
    local optline : subinstr local optline ",v(" ", verbose(", all
    local optline : subinstr local optline ", v(" ", verbose(", all
    local optline : subinstr local optline " v(" " verbose(", all
    local optline : subinstr local optline ",l(" ", level(", all
    local optline : subinstr local optline ", l(" ", level(", all
    local optline : subinstr local optline " l(" " level(", all
    local optline : subinstr local optline ",res(" ", residuals(", all
    local optline : subinstr local optline ", res(" ", residuals(", all
    local optline : subinstr local optline " res(" " residuals(", all
    // Alias mobility*file() to mobfile() to avoid option-name ambiguity in Stata parsing.
    local optline : subinstr local optline ",mobilityprofilefile(" ", mobfile(", all
    local optline : subinstr local optline ", mobilityprofilefile(" ", mobfile(", all
    local optline : subinstr local optline " mobilityprofilefile(" " mobfile(", all
    local optline : subinstr local optline ",mobilityfile(" ", mobfile(", all
    local optline : subinstr local optline ", mobilityfile(" ", mobfile(", all
    local optline : subinstr local optline " mobilityfile(" " mobfile(", all
    // Alias absorptioncachemode() to abscachemode() to avoid option-name ambiguity in Stata parsing.
    local optline : subinstr local optline ",absorptioncachemode(" ", abscachemode(", all
    local optline : subinstr local optline ", absorptioncachemode(" ", abscachemode(", all
    local optline : subinstr local optline " absorptioncachemode(" " abscachemode(", all
    // Alias fescachemode() to fecachemode() to avoid option-name ambiguity in Stata parsing.
    local optline : subinstr local optline ",fescachemode(" ", fecachemode(", all
    local optline : subinstr local optline ", fescachemode(" ", fecachemode(", all
    local optline : subinstr local optline " fescachemode(" " fecachemode(", all
    // Alias festructurecache*file() to fescache() to avoid option-name ambiguity in Stata parsing.
    local optline : subinstr local optline ",festructurecachefile(" ", fescache(", all
    local optline : subinstr local optline ", festructurecachefile(" ", fescache(", all
    local optline : subinstr local optline " festructurecachefile(" " fescache(", all
    local optline : subinstr local optline ",festructurecache(" ", fescache(", all
    local optline : subinstr local optline ", festructurecache(" ", fescache(", all
    local optline : subinstr local optline " festructurecache(" " fescache(", all
    local optline : subinstr local optline ",festructurecachemode(" ", fecachemode(", all
    local optline : subinstr local optline ", festructurecachemode(" ", fecachemode(", all
    local optline : subinstr local optline " festructurecachemode(" " fecachemode(", all
    // fvwrapon() shares the fvwrap prefix; route it to a parser-safe internal name.
    local optline : subinstr local optline ",fvwrapon(" ", fvwon(", all
    local optline : subinstr local optline ", fvwrapon(" ", fvwon(", all
    local optline : subinstr local optline " fvwrapon(" " fvwon(", all
    // Alias long threading controls to short internal names; Stata's
    // syntax abbreviation parser can reject the documented full names here.
    local optline : subinstr local optline ",minparallelrows(" ", minparrows(", all
    local optline : subinstr local optline ", minparallelrows(" ", minparrows(", all
    local optline : subinstr local optline " minparallelrows(" " minparrows(", all
    local optline : subinstr local optline ",targetrowsperthread(" ", targetrpt(", all
    local optline : subinstr local optline ", targetrowsperthread(" ", targetrpt(", all
    local optline : subinstr local optline " targetrowsperthread(" " targetrpt(", all
    local 0 `"`optline'"'

    syntax varlist(min=1 fv numeric) [if] [in] [aw fw pw iw], ///
        [ ///
        ABSorb(string asis) ///
        VCE(string asis) ///
        CLuster(string asis) ///
        ROBust ///
        NOHeader ///
        NOTABle ///
        NOFOOTnote ///
        NOOMITTED ///
        NOEMPty ///
        NOCI ///
        NOPVALUES ///
        BASElevels ///
        ALLBASElevels ///
        CFORMAT(string asis) ///
        PFORMAT(string asis) ///
        SFORMAT(string asis) ///
        FVWrap(integer 0) ///
        FVWON(string asis) ///
        VSQUISH ///
        NOLSTRetch ///
        NOFVLABEL ///
        NOConstant ///
        TOLerance(real 1e-8) ///
        CONVERGENCE(string) ///
        FETOLerance(real 1e-6) ///
        TOLERANCEMode(string) ///
        FERECOVERYMethod(string) ///
        MAXITer(integer 100000) ///
        NUMThreads(integer 0) ///
        DEFAULTThreads(integer 0) ///
        MAXThreads(integer 0) ///
        MINPARRows(integer 20000) ///
        TARGETRPT(integer 500000) ///
        NOSAMPle ///
        TIMEit ///
        VERBose(integer -1) ///
        TECHnique(string asis) ///
        ACCELeration(string asis) ///
        TRANSform(string asis) ///
        PREConditioner(string asis) ///
        PRUNE ///
        FASTRegress ///
        POOLsize(integer 0) ///
        COMPact ///
        VERSion(integer 0) ///
        NOWARN ///
        PARAllel(string asis) ///
        GPUBACKEND(string) ///
        MOBilityProfile ///
        MOBfile(string) ///
        ABSORPTIONCache(string) ///
        ABSCACHEMode(string) ///
        FESTructureCache ///
        FESCache(string) ///
        FECACHEMODE(string) ///
        SYMmetricSweep ///
        ABSORPTIONMethod(string) ///
        JACOBIRelaxation(real 0) ///
        LEVEL(real 95) ///
        KEEPsingletons ///
        DOFAdjustments(string asis) ///
        SSC(string asis) ///
        STATStyle(string) ///
        GROUP(varname) ///
        INDIVIDUAL(varname) ///
        I(varname) ///
        AGGregation(string) ///
        ENDOGenous(varlist numeric) ///
        INSTRuments(varlist numeric) ///
        RESiduals(name) ///
        RESiduals2 ///
        GROUPVAR(name) ///
        SAVEFEs(string) ///
        ]

    local unsupported_het_slopes_msg ///
        "continuous interactions are supported as absorbed heterogeneous slopes only; use absorb(fe#c.x) or absorb(fe##c.x)"

    local compat_ignored
    if ("`timeit'" != "") local compat_ignored "`compat_ignored' timeit"
    if (`verbose' >= 0) local compat_ignored "`compat_ignored' verbose(`verbose')"
    if ("`technique'" != "") local compat_ignored `"`compat_ignored' technique(`technique')"' 
    if ("`acceleration'" != "") local compat_ignored `"`compat_ignored' acceleration(`acceleration')"' 
    if ("`transform'" != "") local compat_ignored `"`compat_ignored' transform(`transform')"' 
    if ("`preconditioner'" != "") local compat_ignored `"`compat_ignored' preconditioner(`preconditioner')"' 
    if ("`prune'" != "") local compat_ignored "`compat_ignored' prune"
    if ("`fastregress'" != "") local compat_ignored "`compat_ignored' fastregress"
    if (`poolsize' > 0) local compat_ignored "`compat_ignored' poolsize(`poolsize')"
    if ("`compact'" != "") local compat_ignored "`compat_ignored' compact"
    if (`version' > 0) local compat_ignored "`compat_ignored' version(`version')"
    if ("`parallel'" != "") local compat_ignored `"`compat_ignored' parallel(`parallel')"' 

    // i() is an alias of individual() (reghdfe-style shorthand).
    if ("`i'" != "") {
        if ("`individual'" != "") {
            di as err "specify either i() or individual(), not both"
            exit 198
        }
        local individual "`i'"
    }

    // Parse y and X.
    gettoken depvar indepvars_raw : varlist

    // Parse absorb() allowing reghdfe-style suboptions (e.g., absorb(..., savefe)).
    local absorb_list
    local absorb_ordered
    local absorb_labels
    local absorb_aliases
    local absorb_kinds
    local absorb_slopevars
    local absorb_slopeintercepts
    local absorb_markvars
    local absorb_subopts
    local savefe_absorb 0
    local cluster_raw
    local cluster_ordered
    local cluster_markvars
    local cluster_display
    local cluster_list
    if ("`absorb'" != "") {
        local absorb_raw = strtrim("`absorb'")
        local absorb_vars "`absorb_raw'"
        local absorb_opts ""
        if (strpos("`absorb_raw'", ",") > 0) {
            gettoken absorb_vars absorb_opts : absorb_raw, parse(",")
            local absorb_vars = strtrim("`absorb_vars'")
            local absorb_opts : subinstr local absorb_opts "," "", all
            local absorb_opts = strtrim("`absorb_opts'")
        }
        if ("`absorb_vars'" != "") {
            foreach tok of local absorb_vars {
                local tok = strtrim("`tok'")
                if ("`tok'" == "") continue

                local alias ""
                local body "`tok'"
                if (strpos("`tok'", "=") > 0) {
                    gettoken alias body : tok, parse("=")
                    local alias = strtrim("`alias'")
                    local body : subinstr local body "=" "", all
                    local body = strtrim("`body'")
                    if ("`alias'" == "" | "`body'" == "") {
                        di as err "invalid absorb() alias specification: `tok'"
                        exit 198
                    }
                    capture confirm name `alias'
                    if (_rc) {
                        di as err "absorb() alias `alias' is not a valid name"
                        exit 198
                    }
                }

                // In absorb(), g#c.z is a group-specific slope and g##c.z is
                // a joint group intercept plus group-specific slope.
                local include_slope_intercept 0
                if (strpos("`body'", "##") > 0) local include_slope_intercept 1
                local include_main 0
                local tok_clean : subinstr local body "##" "#", all
                if (strpos("`tok_clean'", "#") > 0) {
                    if (`include_main' & "`alias'" != "") {
                        di as err "absorb() aliases cannot be used with ##; alias each component separately"
                        exit 198
                    }
                    local parts
                    local part_is_cont
                    local rest "`tok_clean'"
                    while ("`rest'" != "") {
                        gettoken part rest : rest, parse("#")
                        if ("`part'" == "#") continue
                        local clean "`part'"
                        local is_cont 0
                        if (substr("`clean'", 1, 2) == "c.") {
                            local is_cont 1
                            local clean = substr("`clean'", 3, .)
                        }
                        else if (substr("`clean'", 1, 2) == "i.") {
                            local clean = substr("`clean'", 3, .)
                        }
                        else if (substr("`clean'", 1, 4) == "ibn.") {
                            local clean = substr("`clean'", 5, .)
                        }
                        else if (substr("`clean'", 1, 3) == "bn.") {
                            local clean = substr("`clean'", 4, .)
                        }
                        else if (regexm("`clean'", "^ib[0-9]+\\.")) {
                            local clean = regexr("`clean'", "^ib[0-9]+\\.", "")
                        }
                        else if (regexm("`clean'", "^b[0-9]+\\.")) {
                            local clean = regexr("`clean'", "^b[0-9]+\\.", "")
                        }
                        unab clean_list : `clean'
                        local clean_count : word count `clean_list'
                        if (`clean_count' != 1) {
                            di as err "absorb() interaction component `clean' is ambiguous"
                            exit 198
                        }
                        local clean : word 1 of `clean_list'
                        local parts "`parts' `clean'"
                        local part_is_cont "`part_is_cont' `is_cont'"
                    }
                    local nparts : word count `parts'
                    if (`nparts' != 2) {
                        di as err "absorb() interactions must have exactly two components"
                        exit 198
                    }
                    local v1 : word 1 of `parts'
                    local v2 : word 2 of `parts'
                    local c1 : word 1 of `part_is_cont'
                    local c2 : word 2 of `part_is_cont'
                    local ncont = `c1' + `c2'

                    if (`ncont' == 1) {
                        if (`c1') {
                            local slope_var "`v1'"
                            local slope_group "`v2'"
                        }
                        else {
                            local slope_group "`v1'"
                            local slope_var "`v2'"
                        }

                        local pos : list posof "`slope_group'" in absorb_markvars
                        if (`pos' == 0) local absorb_markvars "`absorb_markvars' `slope_group'"
                        local pos : list posof "`slope_var'" in absorb_markvars
                        if (`pos' == 0) local absorb_markvars "`absorb_markvars' `slope_var'"

                        local slope_token "`slope_group'#c.`slope_var'"
                        if (`include_slope_intercept') local slope_token "`slope_group'##c.`slope_var'"
                        local slope_label "`slope_token'"
                        if ("`alias'" != "") local slope_label "`alias'"
                        local pos : list posof "`slope_token'" in absorb_ordered
                        if (`pos' == 0) {
                            local absorb_ordered "`absorb_ordered' `slope_token'"
                            local absorb_labels "`absorb_labels' `slope_label'"
                            local alias_flag "."
                            if ("`alias'" != "") local alias_flag "`alias'"
                            local absorb_aliases "`absorb_aliases' `alias_flag'"
                            local absorb_kinds "`absorb_kinds' slope"
                            local absorb_slopevars "`absorb_slopevars' `slope_var'"
                            local absorb_slopeintercepts "`absorb_slopeintercepts' `include_slope_intercept'"
                        }
                        else if ("`alias'" != "") {
                            local existing_label : word `pos' of `absorb_labels'
                            if ("`existing_label'" != "`alias'") {
                                di as err "absorb() alias `alias' conflicts with existing alias `existing_label'"
                                exit 198
                            }
                        }
                        continue
                    }
                    else if (`ncont' > 1) {
                        di as err "absorb() heterogeneous slope terms must have one categorical and one continuous component"
                        exit 198
                    }

                    local pos : list posof "`v1'" in absorb_markvars
                    if (`pos' == 0) local absorb_markvars "`absorb_markvars' `v1'"
                    local pos : list posof "`v2'" in absorb_markvars
                    if (`pos' == 0) local absorb_markvars "`absorb_markvars' `v2'"

                    if (`include_main') {
                        local pos : list posof "`v1'" in absorb_ordered
                        if (`pos' == 0) {
                            local absorb_ordered "`absorb_ordered' `v1'"
                            local absorb_labels "`absorb_labels' `v1'"
                            local absorb_aliases "`absorb_aliases' ."
                            local absorb_kinds "`absorb_kinds' cat"
                            local absorb_slopevars "`absorb_slopevars' ."
                            local absorb_slopeintercepts "`absorb_slopeintercepts' ."
                        }
                        local pos : list posof "`v2'" in absorb_ordered
                        if (`pos' == 0) {
                            local absorb_ordered "`absorb_ordered' `v2'"
                            local absorb_labels "`absorb_labels' `v2'"
                            local absorb_aliases "`absorb_aliases' ."
                            local absorb_kinds "`absorb_kinds' cat"
                            local absorb_slopevars "`absorb_slopevars' ."
                            local absorb_slopeintercepts "`absorb_slopeintercepts' ."
                        }
                    }
                    local inter_token "`v1'#`v2'"
                    local inter_label "`inter_token'"
                    if ("`alias'" != "") local inter_label "`alias'"
                    local pos : list posof "`inter_token'" in absorb_ordered
                    if (`pos' == 0) {
                        local absorb_ordered "`absorb_ordered' `inter_token'"
                        local absorb_labels "`absorb_labels' `inter_label'"
                        local alias_flag "."
                        if ("`alias'" != "") local alias_flag "`alias'"
                        local absorb_aliases "`absorb_aliases' `alias_flag'"
                        local absorb_kinds "`absorb_kinds' cat"
                        local absorb_slopevars "`absorb_slopevars' ."
                        local absorb_slopeintercepts "`absorb_slopeintercepts' ."
                    }
                    else if ("`alias'" != "") {
                        local existing_label : word `pos' of `absorb_labels'
                        if ("`existing_label'" != "`alias'") {
                            di as err "absorb() alias `alias' conflicts with existing alias `existing_label'"
                            exit 198
                        }
                    }
                }
                else {
                    local clean "`body'"
                    if (substr("`clean'", 1, 2) == "c.") {
                        di as err "`unsupported_het_slopes_msg'"
                        exit 198
                    }
                    if (substr("`clean'", 1, 2) == "i.") {
                        local clean = substr("`clean'", 3, .)
                    }
                    else if (substr("`clean'", 1, 4) == "ibn.") {
                        local clean = substr("`clean'", 5, .)
                    }
                    else if (substr("`clean'", 1, 3) == "bn.") {
                        local clean = substr("`clean'", 4, .)
                    }
                    else if (regexm("`clean'", "^ib[0-9]+\\.")) {
                        local clean = regexr("`clean'", "^ib[0-9]+\\.", "")
                    }
                    else if (regexm("`clean'", "^b[0-9]+\\.")) {
                        local clean = regexr("`clean'", "^b[0-9]+\\.", "")
                    }
                    unab clean_list : `clean'
                    local clean_count : word count `clean_list'
                    if (`clean_count' != 1) {
                        di as err "absorb() variable `clean' is ambiguous"
                        exit 198
                    }
                    local clean : word 1 of `clean_list'
                    local label "`clean'"
                    if ("`alias'" != "") local label "`alias'"

                    local pos : list posof "`clean'" in absorb_ordered
                    if (`pos' == 0) {
                        local absorb_ordered "`absorb_ordered' `clean'"
                        local absorb_labels "`absorb_labels' `label'"
                        local alias_flag "."
                        if ("`alias'" != "") local alias_flag "`alias'"
                        local absorb_aliases "`absorb_aliases' `alias_flag'"
                        local absorb_kinds "`absorb_kinds' cat"
                        local absorb_slopevars "`absorb_slopevars' ."
                        local absorb_slopeintercepts "`absorb_slopeintercepts' ."
                    }
                    else if ("`alias'" != "") {
                        local existing_label : word `pos' of `absorb_labels'
                        if ("`existing_label'" != "`alias'") {
                            di as err "absorb() alias `alias' conflicts with existing alias `existing_label'"
                            exit 198
                        }
                    }
                    local pos : list posof "`clean'" in absorb_markvars
                    if (`pos' == 0) local absorb_markvars "`absorb_markvars' `clean'"
                }
            }
        }
        local absorb_subopts "`absorb_opts'"
    }
    if ("`absorb_subopts'" != "") {
        local absorb_subopts_clean : subinstr local absorb_subopts "," " " , all
        local absorb_subopts_clean = strtrim("`absorb_subopts_clean'")
        foreach tok of local absorb_subopts_clean {
            local tok_l = lower("`tok'")
            if ("`tok_l'" == "savefe") {
                local savefe_absorb 1
            }
            else {
                di as err "unsupported absorb() suboption: `tok'"
                exit 198
            }
        }
    }
    if (`savefe_absorb' & "`absorb_ordered'" == "") {
        di as err "absorb(..., savefe) requires at least one fixed effect variable"
        exit 198
    }

    if ("`absorb_ordered'" == "" & "`absorb_markvars'" != "") {
        local absorb_ordered "`absorb_markvars'"
        local absorb_aliases ""
        local absorb_kinds ""
        local absorb_slopevars ""
        local absorb_slopeintercepts ""
        foreach v of local absorb_ordered {
            local absorb_aliases "`absorb_aliases' ."
            local absorb_kinds "`absorb_kinds' cat"
            local absorb_slopevars "`absorb_slopevars' ."
            local absorb_slopeintercepts "`absorb_slopeintercepts' ."
        }
    }
    if ("`absorb_labels'" == "" & "`absorb_ordered'" != "") {
        local absorb_labels "`absorb_ordered'"
        if ("`absorb_aliases'" == "") {
            foreach v of local absorb_ordered {
                local absorb_aliases "`absorb_aliases' ."
            }
        }
        if ("`absorb_kinds'" == "") {
            foreach v of local absorb_ordered {
                local absorb_kinds "`absorb_kinds' cat"
                local absorb_slopevars "`absorb_slopevars' ."
                local absorb_slopeintercepts "`absorb_slopeintercepts' ."
            }
        }
    }

    // Fixed effects (absorb()) list; ensure individual() is last when present.
    if ("`individual'" != "") {
        if ("`group'" == "") {
            di as err "option individual() requires option group()"
            exit 198
        }
        local new_ordered ""
        local new_labels ""
        local new_aliases ""
        local new_kinds ""
        local new_slopevars ""
        local new_slopeintercepts ""
        local ind_label ""
        local ind_alias "."
        local ind_kind "cat"
        local ind_slopevar "."
        local ind_slopeintercept "."
        local idx = 0
        foreach entry of local absorb_ordered {
            local ++idx
            local label : word `idx' of `absorb_labels'
            local alias_name : word `idx' of `absorb_aliases'
            local kind : word `idx' of `absorb_kinds'
            local slopevar : word `idx' of `absorb_slopevars'
            local slopeintercept : word `idx' of `absorb_slopeintercepts'
            if ("`entry'" == "`individual'") {
                local ind_label "`label'"
                local ind_alias "`alias_name'"
                local ind_kind "`kind'"
                local ind_slopevar "`slopevar'"
                local ind_slopeintercept "`slopeintercept'"
                continue
            }
            local new_ordered "`new_ordered' `entry'"
            local new_labels "`new_labels' `label'"
            local new_aliases "`new_aliases' `alias_name'"
            local new_kinds "`new_kinds' `kind'"
            local new_slopevars "`new_slopevars' `slopevar'"
            local new_slopeintercepts "`new_slopeintercepts' `slopeintercept'"
        }
        local absorb_ordered "`new_ordered' `individual'"
        if ("`ind_label'" == "") local ind_label "`individual'"
        local absorb_labels "`new_labels' `ind_label'"
        local absorb_aliases "`new_aliases' `ind_alias'"
        local absorb_kinds "`new_kinds' `ind_kind'"
        local absorb_slopevars "`new_slopevars' `ind_slopevar'"
        local absorb_slopeintercepts "`new_slopeintercepts' `ind_slopeintercept'"
        local pos : list posof "`individual'" in absorb_markvars
        if (`pos' == 0) local absorb_markvars "`absorb_markvars' `individual'"
    }

    // Default aggregation for group() mode.
    if ("`aggregation'" == "") {
        local aggregation "mean"
    }

    // IV / instruments.
    local ninst : word count `instruments'
    local nendog : word count `endogenous'
    if (`ninst' > 0 | `nendog' > 0) {
        local has_fv = (strpos("`indepvars_raw'", ".") > 0) | (strpos("`indepvars_raw'", "#") > 0)
        if (`has_fv') {
            di as err "factor-variable operators (i., c., #) are not supported with IV/instruments yet"
            exit 198
        }
        if (`ninst' == 0 | `nendog' == 0) {
            di as err "options endogenous() and instruments() must be specified together"
            exit 198
        }
        if ("`group'" != "") {
            di as err "group()/individual() mode does not support IV/instruments"
            exit 198
        }
        local endogenous_idx
        foreach v of varlist `endogenous' {
            local pos : list posof "`v'" in indepvars_raw
            if (`pos' == 0) {
                di as err "endogenous variable `v' must be among regressors"
                exit 198
            }
            local idx0 = `pos' - 1
            if ("`endogenous_idx'" == "") local endogenous_idx "`idx0'"
            else local endogenous_idx "`endogenous_idx',`idx0'"
        }
    }

    // Determine SE type.
    local vce_given 0
    local se_type "unadjusted"
    if ("`vce'" != "") {
        local vce_given 1
        if ("`robust'" != "" | "`cluster'" != "") {
            di as err "specify either vce() or robust/cluster(), not both"
            exit 198
        }
        local vce_clean = strtrim("`vce'")
        gettoken vce_key vce_rest : vce_clean
        local vce_key = lower("`vce_key'")
        if (inlist("`vce_key'", "robust", "hc1")) {
            local se_type "robust"
        }
        else if (inlist("`vce_key'", "ols", "unadjusted", "homoskedastic", "classical")) {
            local se_type "unadjusted"
        }
        else if (inlist("`vce_key'", "cluster", "clustered")) {
            local se_type "cluster"
            local vce_rest = strtrim("`vce_rest'")
            if ("`vce_rest'" == "") {
                di as err "vce(cluster ...) requires one or more cluster variables"
                exit 198
            }
            local cluster_raw "`vce_rest'"
        }
        else {
            di as err "unsupported vce(): `vce'"
            exit 198
        }
    }
    else {
        // Backwards-compatible robust/cluster options; default is unadjusted.
        if ("`robust'" != "") {
            local se_type "robust"
        }
        if ("`cluster'" != "") {
            local se_type "cluster"
            local cluster_raw "`cluster'"
        }
    }

    // Parse cluster variables, allowing interactions like var1#var2.
    if ("`cluster_raw'" != "") {
        local cluster_vars = strtrim("`cluster_raw'")
        foreach tok of local cluster_vars {
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
                    if (substr("`clean'", 1, 2) == "i.") {
                        local clean = substr("`clean'", 3, .)
                    }
                    else if (substr("`clean'", 1, 4) == "ibn.") {
                        local clean = substr("`clean'", 5, .)
                    }
                    else if (substr("`clean'", 1, 3) == "bn.") {
                        local clean = substr("`clean'", 4, .)
                    }
                    else if (regexm("`clean'", "^ib[0-9]+\\.")) {
                        local clean = regexr("`clean'", "^ib[0-9]+\\.", "")
                    }
                    else if (regexm("`clean'", "^b[0-9]+\\.")) {
                        local clean = regexr("`clean'", "^b[0-9]+\\.", "")
                    }
                    unab clean_list : `clean'
                    local clean_count : word count `clean_list'
                    if (`clean_count' != 1) {
                        di as err "cluster() interaction component `clean' is ambiguous"
                        exit 198
                    }
                    local clean : word 1 of `clean_list'
                    local parts "`parts' `clean'"
                }
                local nparts : word count `parts'
                if (`nparts' != 2) {
                    di as err "cluster() interactions must have exactly two components"
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
                    if (`pos' == 0) {
                        local cluster_ordered "`cluster_ordered' `v1'"
                        local cluster_display "`cluster_display' `v1'"
                    }
                    local pos : list posof "`v2'" in cluster_ordered
                    if (`pos' == 0) {
                        local cluster_ordered "`cluster_ordered' `v2'"
                        local cluster_display "`cluster_display' `v2'"
                    }
                }

                local inter_token "`v1'#`v2'"
                local pos : list posof "`inter_token'" in cluster_ordered
                if (`pos' == 0) {
                    local cluster_ordered "`cluster_ordered' `inter_token'"
                    local cluster_display "`cluster_display' `inter_token'"
                }
            }
            else {
                local clean "`tok'"
                if (substr("`clean'", 1, 2) == "c.") {
                    di as err "`unsupported_het_slopes_msg'"
                    exit 198
                }
                if (substr("`clean'", 1, 2) == "i.") {
                    local clean = substr("`clean'", 3, .)
                }
                else if (substr("`clean'", 1, 4) == "ibn.") {
                    local clean = substr("`clean'", 5, .)
                }
                else if (substr("`clean'", 1, 3) == "bn.") {
                    local clean = substr("`clean'", 4, .)
                }
                else if (regexm("`clean'", "^ib[0-9]+\\.")) {
                    local clean = regexr("`clean'", "^ib[0-9]+\\.", "")
                }
                else if (regexm("`clean'", "^b[0-9]+\\.")) {
                    local clean = regexr("`clean'", "^b[0-9]+\\.", "")
                }
                unab clean_list : `clean'
                local clean_count : word count `clean_list'
                if (`clean_count' != 1) {
                    di as err "cluster() variable `clean' is ambiguous"
                    exit 198
                }
                local clean : word 1 of `clean_list'
                local pos : list posof "`clean'" in cluster_ordered
                if (`pos' == 0) {
                    local cluster_ordered "`cluster_ordered' `clean'"
                    local cluster_display "`cluster_display' `clean'"
                }
                local pos : list posof "`clean'" in cluster_markvars
                if (`pos' == 0) local cluster_markvars "`cluster_markvars' `clean'"
            }
        }
    }
    if ("`cluster_ordered'" == "" & "`cluster_markvars'" != "") {
        local cluster_ordered "`cluster_markvars'"
    }
    if ("`cluster_display'" == "" & "`cluster_ordered'" != "") {
        local cluster_display "`cluster_ordered'"
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
        // For pweights, Stata typically uses robust inference.
        if ("`weight'" == "pweight" & "`cluster_raw'" == "" & `vce_given' == 0) {
            local se_type "robust"
        }
    }

    // Sample restriction and missing values.
    marksample touse, strok
    local markvars_raw "`absorb_markvars' `cluster_markvars' `wvar' `group' `individual' `endogenous' `instruments'"
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
    if (`__xhdfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=sample_marking ms=" %12.3f (1000 * r(t99))
        timer clear 99
        timer on 99
    }

    // Build interaction fixed effects after the estimation sample is defined.
    local absorb_display ""
    local absorb_list ""
    if ("`absorb_ordered'" != "") {
        local idx = 0
        foreach entry of local absorb_ordered {
            local ++idx
            local label : word `idx' of `absorb_labels'
            if ("`label'" == "") local label "`entry'"
            local alias_name : word `idx' of `absorb_aliases'
            if ("`alias_name'" == ".") local alias_name ""
            local kind : word `idx' of `absorb_kinds'
            local slopevar : word `idx' of `absorb_slopevars'
            if ("`kind'" == "slope") {
                local pair "`entry'"
                local pair_clean : subinstr local pair "##" "#", all
                gettoken v1 rest : pair_clean, parse("#")
                gettoken hash v2 : rest, parse("#")
                if (substr("`v2'", 1, 2) == "c.") local v2 = substr("`v2'", 3, .)
                if ("`v1'" == "" | "`v2'" == "" | "`slopevar'" == "") {
                    di as err "internal error: invalid heterogeneous slope absorb term"
                    exit 198
                }
                local absorb_list "`absorb_list' `v1'"
                local absorb_display "`absorb_display' `label'"
            }
            else if (strpos("`entry'", "#") > 0) {
                local pair "`entry'"
                gettoken v1 rest : pair, parse("#")
                gettoken hash v2 : rest, parse("#")
                if ("`v1'" == "" | "`v2'" == "") {
                    di as err "absorb() interactions must have exactly two components"
                    exit 198
                }
                if ("`alias_name'" != "") confirm new variable `alias_name'
                tempvar feint
                quietly egen long `feint' = group(`v1' `v2') if `touse'
                local absorb_list "`absorb_list' `feint'"
                local absorb_display "`absorb_display' `label'"
            }
            else {
                if ("`alias_name'" != "" & "`alias_name'" != "`entry'") confirm new variable `alias_name'
                local absorb_list "`absorb_list' `entry'"
                local absorb_display "`absorb_display' `label'"
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
                    di as err "cluster() interactions must have exactly two components"
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

    // Encode absorb() and cluster() IDs if needed (string / float / double).
    local nfe 0
    local fe_ids
    local fe_orig
    local fe_raw
    local individual_fe_pos 0
    local individual_id
    local nslope 0
    local slope_absorb_vars
    local slope_fe_map
    local slope_intercepts

    if ("`absorb_list'" != "") {
        local fe_idx = 0
        foreach v of varlist `absorb_list' {
            local ++nfe
            local ++fe_idx
            local fe_label "`v'"
            if ("`absorb_labels'" != "") {
                local fe_label : word `fe_idx' of `absorb_labels'
                if ("`fe_label'" == "") local fe_label "`v'"
            }
            local fe_orig "`fe_orig' `fe_label'"
            local t : type `v'
            local feid "`v'"
            if (substr("`t'", 1, 3) == "str") {
                tempvar feid_tmp
                quietly egen long `feid_tmp' = group(`v') if `touse'
                local feid "`feid_tmp'"
            }
            local fe_raw "`fe_raw' `v'"
            local fe_ids "`fe_ids' `feid'"
            local kind : word `fe_idx' of `absorb_kinds'
            if ("`kind'" == "slope") {
                local slopevar : word `fe_idx' of `absorb_slopevars'
                local slopeintercept : word `fe_idx' of `absorb_slopeintercepts'
                local ++nslope
                local slope_absorb_vars "`slope_absorb_vars' `slopevar'"
                local slope_idx0 = `nfe' - 1
                if ("`slope_fe_map'" == "") local slope_fe_map "`slope_idx0'"
                else local slope_fe_map "`slope_fe_map',`slope_idx0'"
                if ("`slope_intercepts'" == "") local slope_intercepts "`slopeintercept'"
                else local slope_intercepts "`slope_intercepts',`slopeintercept'"
            }
            if ("`individual'" != "" & "`v'" == "`individual'") {
                local individual_id "`feid'"
                local individual_fe_pos `nfe'
            }
        }
    }

    // Map duplicate absorb() id variables (e.g. two slope terms on the same
    // FE) so the plugin reads/dense-codes each distinct variable only once.
    local fe_dup_map
    local fe_has_dup 0
    if (`nfe' > 0) {
        local fe_seen
        foreach f of local fe_ids {
            local dpos : list posof "`f'" in fe_seen
            local dpos0 = `dpos' - 1
            if (`dpos0' >= 0) local fe_has_dup 1
            local fe_dup_map "`fe_dup_map' `dpos0'"
            local fe_seen "`fe_seen' `f'"
        }
    }

    local nclust 0
    local clust_ids
    local cluster_fe_map
    if ("`cluster_list'" != "") {
        foreach v of varlist `cluster_list' {
            local ++nclust
            local clid "`v'"
            local cluster_fe_pos -1

            // Reuse absorb() encodings when the same variable appears in absorb() and cluster().
            local pos : list posof "`v'" in fe_raw
            if (`pos' > 0) {
                local clid : word `pos' of `fe_ids'
                local cluster_fe_pos = `pos' - 1
            }
            else {
                local t : type `v'
                if (substr("`t'", 1, 3) == "str") {
                    tempvar clid_tmp
                    quietly egen long `clid_tmp' = group(`v') if `touse'
                    local clid "`clid_tmp'"
                }
            }

            local clust_ids "`clust_ids' `clid'"
            local cluster_fe_map "`cluster_fe_map' `cluster_fe_pos'"
        }
    }
    if (`__xhdfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=fe_cluster_ids ms=" %12.3f (1000 * r(t99))
        timer clear 99
        timer on 99
    }

    local group_mode 0
    local has_individual 0
    local group_id
    if ("`group'" != "") {
        local group_mode 1
        local t : type `group'
        tempvar gid
        if (substr("`t'", 1, 3) == "str" | inlist("`t'", "float", "double")) {
            quietly egen long `gid' = group(`group') if `touse'
        }
        else {
            quietly gen long `gid' = `group' if `touse'
        }
        local group_id "`gid'"
    }
    if ("`individual'" != "") {
        local has_individual 1
        if (`individual_fe_pos' == 0) {
            di as err "internal error: individual() must be present in absorb() list"
            exit 198
        }
    }
    if (`group_mode' & `nslope' > 0) {
        di as err "group()/individual() mode does not support heterogeneous slopes"
        exit 198
    }

    // Output variables.
    local store_resid 0
    local resid_var
    if ("`residuals2'" != "" & "`residuals'" != "") {
        di as err "only one of residuals and residuals() may be specified"
        exit 198
    }
    if ("`residuals2'" != "") {
        capture drop _reghdfe_resid
        local residuals "_reghdfe_resid"
    }
    if ("`residuals'" != "") {
        confirm new variable `residuals'
        gen double `residuals' = .
        local resid_var "`residuals'"
        local store_resid 1
    }

    local store_groupvar 0
    local groupvar_var
    if ("`groupvar'" != "") {
        confirm new variable `groupvar'
        gen long `groupvar' = .
        local groupvar_var "`groupvar'"
        local store_groupvar 1
    }

    local store_fes 0
    local retain_fes 0
    local fe_out_vars
    local fe_out_count 0
    local savefe_alias 0
    local fe_alias_out
    if ("`absorb_ordered'" != "") {
        local idx = 0
        foreach entry of local absorb_ordered {
            local ++idx
            local alias_name : word `idx' of `absorb_aliases'
            if ("`alias_name'" == ".") local alias_name ""
            if ("`alias_name'" == "`entry'") local alias_name ""
            if ("`alias_name'" != "") local savefe_alias 1
            local fe_alias_out "`fe_alias_out' `alias_name'"
        }
    }
    if (`savefe_absorb' & "`savefes'" != "") {
        di as err "absorb(..., savefe) cannot be combined with savefes()"
        exit 198
    }
    // Guard before generating any FE output variable: saving fixed-effect
    // contributions is not supported when both group() and individual() are
    // specified. Reject early so a failed command never mutates the dataset.
    if (`group_mode' & `has_individual' & (`savefe_absorb' | "`savefes'" != "" | `savefe_alias')) {
        di as err "savefe/savefes are not supported when both group() and individual() are specified"
        exit 198
    }
    if (`savefe_absorb') {
        local store_fes 1
        local retain_fes 1
        local k 0
        foreach fe of local fe_orig {
            local ++k
            local kind : word `k' of `absorb_kinds'
            local slopevar : word `k' of `absorb_slopevars'
            local slopeintercept : word `k' of `absorb_slopeintercepts'
            local fe_slope_label "`fe'"
            if ("`kind'" == "slope") {
                local fe_group_label "`fe'"
                local fe_group_label : subinstr local fe_group_label "##c.`slopevar'" "", all
                local fe_group_label : subinstr local fe_group_label "#c.`slopevar'" "", all
                local fe_slope_label "`fe_group_label'#c.`slopevar'"
            }
            local vname "__hdfe`k'__"
            if ("`kind'" == "slope" & "`slopeintercept'" == "0") {
                local slope_vname "`vname'Slope1"
                confirm new variable `slope_vname'
                gen double `slope_vname' = .
                label variable `slope_vname' "[FE] `k'.`fe_slope_label'"
                local fe_out_vars "`fe_out_vars' `slope_vname'"
            }
            else {
                confirm new variable `vname'
                gen double `vname' = .
                label variable `vname' "[FE] `k'.`fe'"
                local fe_out_vars "`fe_out_vars' `vname'"
                if ("`kind'" == "slope" & "`slopeintercept'" == "1") {
                    local slope_vname "`vname'Slope1"
                    confirm new variable `slope_vname'
                    gen double `slope_vname' = .
                    label variable `slope_vname' "[FE] `k'.`fe_slope_label'"
                    local fe_out_vars "`fe_out_vars' `slope_vname'"
                }
            }
        }
    }
    else if ("`savefes'" != "") {
        local store_fes 1
        local retain_fes 1
        local prefix "`savefes'"
        if (substr("`prefix'", -1, 1) != "_") {
            local prefix "`prefix'_"
        }
        local k 0
        foreach fe of local fe_orig {
            local ++k
            local kind : word `k' of `absorb_kinds'
            local slopevar : word `k' of `absorb_slopevars'
            local slopeintercept : word `k' of `absorb_slopeintercepts'
            local fe_slope_label "`fe'"
            if ("`kind'" == "slope") {
                local fe_group_label "`fe'"
                local fe_group_label : subinstr local fe_group_label "##c.`slopevar'" "", all
                local fe_group_label : subinstr local fe_group_label "#c.`slopevar'" "", all
                local fe_slope_label "`fe_group_label'#c.`slopevar'"
            }
            local reserve_suffix 0
            if ("`kind'" == "slope") local reserve_suffix 6
            local max_base = 32 - `reserve_suffix'
            local fe_safe "`fe'"
            local fe_safe : subinstr local fe_safe "#" "_" , all
            local fe_safe : subinstr local fe_safe "." "_" , all
            local vname "`prefix'`fe_safe'"
            if (strlen("`vname'") > `max_base') {
                local vname "`prefix'fe`k'"
            }
            if (strlen("`vname'") > `max_base') {
                local vname "xfe`k'"
            }
            capture confirm new variable `vname'
            if (_rc) {
                local vname "`prefix'fe`k'"
                if (strlen("`vname'") > `max_base') {
                    local vname "xfe`k'"
                }
                confirm new variable `vname'
            }
            if ("`kind'" == "slope" & "`slopeintercept'" == "0") {
                local slope_vname "`vname'Slope1"
                confirm new variable `slope_vname'
                gen double `slope_vname' = .
                label variable `slope_vname' "[FE] `k'.`fe_slope_label'"
                local fe_out_vars "`fe_out_vars' `slope_vname'"
            }
            else {
                gen double `vname' = .
                label variable `vname' "[FE] `k'.`fe'"
                local fe_out_vars "`fe_out_vars' `vname'"
                if ("`kind'" == "slope" & "`slopeintercept'" == "1") {
                    local slope_vname "`vname'Slope1"
                    confirm new variable `slope_vname'
                    gen double `slope_vname' = .
                    label variable `slope_vname' "[FE] `k'.`fe_slope_label'"
                    local fe_out_vars "`fe_out_vars' `slope_vname'"
                }
            }
        }
    }
    else if (`savefe_alias') {
        local store_fes 1
        local retain_fes 1
        local k 0
        foreach entry of local absorb_ordered {
            local ++k
            local kind : word `k' of `absorb_kinds'
            local slopevar : word `k' of `absorb_slopevars'
            local slopeintercept : word `k' of `absorb_slopeintercepts'
            local entry_slope_label "`entry'"
            if ("`kind'" == "slope") {
                local entry_group_label "`entry'"
                local entry_group_label : subinstr local entry_group_label "##c.`slopevar'" "", all
                local entry_group_label : subinstr local entry_group_label "#c.`slopevar'" "", all
                local entry_slope_label "`entry_group_label'#c.`slopevar'"
            }
            local alias_name : word `k' of `fe_alias_out'
            local vname "`alias_name'"
            if ("`vname'" == "") local vname "__hdfe`k'__"
            if ("`kind'" == "slope" & strlen("`vname'") > 26) {
                di as err "absorb() alias `vname' is too long to save heterogeneous slope variables"
                exit 198
            }
            if ("`kind'" == "slope" & "`slopeintercept'" == "0") {
                local slope_vname "`vname'Slope1"
                confirm new variable `slope_vname'
                gen double `slope_vname' = .
                label variable `slope_vname' "[FE] `k'.`entry_slope_label'"
                local fe_out_vars "`fe_out_vars' `slope_vname'"
            }
            else {
                confirm new variable `vname'
                gen double `vname' = .
                label variable `vname' "[FE] `k'.`entry'"
                local fe_out_vars "`fe_out_vars' `vname'"
                if ("`kind'" == "slope" & "`slopeintercept'" == "1") {
                    local slope_vname "`vname'Slope1"
                    confirm new variable `slope_vname'
                    gen double `slope_vname' = .
                    label variable `slope_vname' "[FE] `k'.`entry_slope_label'"
                    local fe_out_vars "`fe_out_vars' `slope_vname'"
                }
            }
        }
    }
    if (`store_fes') {
        local fe_out_count : word count `fe_out_vars'
    }

    // e(sample) marker (always created).
    tempvar esample
    if (`group_mode') {
        gen byte `esample' = 0
    }
    else {
        // Prefill with the candidate sample; the plugin will clear dropped observations.
        gen byte `esample' = `touse'
    }
    if (`__xhdfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=outputs_esample ms=" %12.3f (1000 * r(t99))
        timer clear 99
        timer on 99
    }

    // drop_singletons handling
    local drop_singletons 1
    if ("`keepsingletons'" != "") {
        local drop_singletons 0
        if ("`nowarn'" == "") {
            di as err "WARNING: Singleton observations not dropped; statistical significance is biased (http://scorreia.com/reghdfe/nested_within_cluster.pdf)"
        }
    }

    // constant / intercept handling
    local fit_intercept 1
    if ("`noconstant'" != "") {
        local fit_intercept 0
    }

    // Whether the fitted model contains an intercept anywhere: as a regressor
    // (no absorb), or absorbed through at least one FE level dimension.
    // Slope-only absorption (absorb(fe#c.z)) has no intercept: reghdfe reports
    // no _cons there and uses noconstant-style fit statistics.
    local model_has_cons = `fit_intercept'
    local nkinds : word count `absorb_kinds'
    if (`fit_intercept' & `nkinds' > 0 & `nendog' == 0) {
        local any_level 0
        forvalues d = 1/`nkinds' {
            local kind : word `d' of `absorb_kinds'
            local sint : word `d' of `absorb_slopeintercepts'
            if ("`kind'" != "slope") local any_level 1
            else if ("`sint'" == "1") local any_level 1
        }
        if (!`any_level') local model_has_cons 0
    }

    // absorption method default
    if ("`absorptionmethod'" == "") {
        local absorptionmethod "auto"
    }

    local convergence = lower(strtrim("`convergence'"))
    if ("`convergence'" == "") local convergence "auto"
    if (inlist("`convergence'", "norm", "norm_change", "current", "xhdfe")) {
        local convergence "normchange"
    }
    if (inlist("`convergence'", "vector", "vectors", "reghdfe-style", "reghdfe_style")) {
        local convergence "reghdfe"
    }
    if (!inlist("`convergence'", "auto", "normchange", "reghdfe", "both")) {
        di as err "convergence() must be auto, normchange, reghdfe, or both"
        exit 198
    }

    // Some Stata versions leave long option defaults empty; enforce defaults here.
    if ("`minparrows'" == "") local minparrows 20000
    if ("`targetrpt'" == "") local targetrpt 500000

    // symmetric sweep flag
    local symmetric_sweep 0
    if ("`symmetricsweep'" != "") local symmetric_sweep 1

    // SSC (fixest-style) small-sample adjustments
    local ssc_k_adj ""
    local ssc_k_fixef ""
    local ssc_k_exact ""
    local ssc_g_adj ""
    local ssc_g_df ""
    local ssc_t_df ""
    if ("`ssc'" != "") {
        local ssc_clean : subinstr local ssc "," " " , all
        local ssc_clean : subinstr local ssc_clean "(" " " , all
        local ssc_clean : subinstr local ssc_clean ")" " " , all
        local ssc_clean = strtrim("`ssc_clean'")
        foreach tok of local ssc_clean {
            if ("`tok'" == "") continue
            gettoken key val : tok, parse("=")
            local key = lower(strtrim("`key'"))
            local val : subinstr local val "=" "", all
            local val = lower(strtrim("`val'"))
            if (inlist("`val'", "true", "t", "yes", "y")) local val "1"
            if (inlist("`val'", "false", "f", "no", "n")) local val "0"

            if (inlist("`key'", "k.adj", "kadj")) {
                if ("`val'" == "") local val "1"
                local ssc_k_adj "`val'"
                continue
            }
            if (inlist("`key'", "k.fixef", "kfixef")) {
                local ssc_k_fixef "`val'"
                continue
            }
            if (inlist("`key'", "k.exact", "kexact")) {
                if ("`val'" == "") local val "1"
                local ssc_k_exact "`val'"
                continue
            }
            if (inlist("`key'", "g.adj", "gadj")) {
                if ("`val'" == "") local val "1"
                local ssc_g_adj "`val'"
                continue
            }
            if (inlist("`key'", "g.df", "gdf")) {
                local ssc_g_df "`val'"
                continue
            }
            if (inlist("`key'", "t.df", "tdf")) {
                local ssc_t_df "`val'"
                continue
            }
        }
    }
    local statstyle = lower(strtrim("`statstyle'"))
    if ("`statstyle'" == "") local statstyle "reghdfe"
    if (inlist("`statstyle'", "current", "xhdfe")) local statstyle "legacy"
    if (!inlist("`statstyle'", "reghdfe", "legacy")) {
        di as err "statstyle() must be reghdfe or legacy"
        exit 198
    }

    local tolerance_mode = lower(strtrim("`tolerancemode'"))
    if ("`tolerance_mode'" == "") local tolerance_mode "reghdfe-comparable"
    local tolerance_mode : subinstr local tolerance_mode "_" "-", all
    if (inlist("`tolerance_mode'", "xhdfe", "fast", "current")) local tolerance_mode "xhdfe-fast"
    if (inlist("`tolerance_mode'", "reghdfe", "comparable")) local tolerance_mode "reghdfe-comparable"
    if (inlist("`tolerance_mode'", "strict", "residual-certificate")) local tolerance_mode "strict-residual"
    if (!inlist("`tolerance_mode'", "xhdfe-fast", "reghdfe-comparable", "strict-residual")) {
        di as err "tolerancemode() must be xhdfe-fast, reghdfe-comparable, or strict-residual"
        exit 198
    }

    if ("`ssc_k_fixef'" == "") {
        if ("`statstyle'" == "legacy") {
            local ssc_k_fixef "nonnested"
        }
        else {
            local ssc_k_fixef "full"
        }
    }

    local gpu_backend ""
    if ("`gpubackend'" != "") {
        local gpu_backend = lower(strtrim("`gpubackend'"))
        if (!inlist("`gpu_backend'", "cpu", "cuda", "metal")) {
            di as err "gpubackend() must be one of cpu, cuda, or metal"
            exit 198
        }
    }

    local mobility_profile ""
    local mobility_profile_mode ""
    if ("`mobfile'" != "") {
        local mobility_profile = strtrim("`mobfile'")
        local mobility_profile_mode "auto"
    }
    if ("`mobilityprofile'" != "") {
        if ("`mobility_profile'" == "") {
            local mobility_profile "xhdfe_mobility_profile.txt"
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

    local absorption_cache ""
    local absorption_cache_mode ""
    if ("`abscachemode'" != "") {
        local absorption_cache_mode = lower(strtrim("`abscachemode'"))
        if (!inlist("`absorption_cache_mode'", "off", "auto", "read", "write")) {
            di as err "absorptioncachemode() must be off, auto, read, or write"
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
        // Strip surrounding quotes to avoid passing literal quotes to the plugin.
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
        capture findfile xhdfe.ado
        if (!_rc) local ado_path "`r(fn)'"
        local cache_dir ""
        if ("`ado_path'" != "") {
            local cache_dir = substr("`ado_path'", 1, strlen("`ado_path'") - strlen("xhdfe.ado"))
        }
        if ("`cache_dir'" == "") {
            local cache_dir "`c(pwd)'/"
        }
        local fe_structure_cache "`cache_dir'xhdfe_fe_structure_cache.bin"
        if ("`fe_structure_cache_mode'" == "") {
            local fe_structure_cache_mode "auto"
        }
    }
    if ("`fe_structure_cache_mode'" != "" & "`fe_structure_cache'" == "") {
        di as err "fescachemode() requires fescache() or festructurecache"
        exit 198
    }
    if ("`fe_structure_cache'" != "") {
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

    // Expand factor variables (including interactions) into concrete regressors for the plugin,
    // while preserving Stata-style coefficient names (e.g., 1.var, 2.var, 1.var#c.z).
    local indepvars "`indepvars_raw'"
    local plugin_indepvars
    local coef_indepvars
    local omit_priority_list
    local fv_stub "__xhdfe_"
    local fv_created 0
    if ("`indepvars'" != "") {
        local has_fv = (strpos("`indepvars'", ".") > 0) | (strpos("`indepvars'", "#") > 0)
        local raw_x "`indepvars'"
        if (`has_fv') {
            quietly fvrevar `indepvars' if `touse', stub(`fv_stub')
            local raw_x "`r(varlist)'"
            local fv_created 1
        }

        foreach v of local raw_x {
            local term : char `v'[fvrevar]
            if ("`term'" == "") local term "`v'"

            // Drop explicitly omitted terms and base categories.
            if (substr("`term'", 1, 2) == "o.") continue
            if (regexm("`term'", "([0-9]+b\.)")) continue

            // Match Stata naming conventions for continuous main effects.
            if (substr("`term'", 1, 2) == "c." & strpos("`term'", "#") == 0) {
                local term = substr("`term'", 3, .)
            }

            // Drop exact duplicates (e.g., specifying tenure and i.x##c.tenure together).
            local already : list posof "`term'" in coef_indepvars
            if (`already' > 0) continue

            local priority 0
            if (regexm("`term'", "^[0-9]+\.")) local priority 1
            local plugin_indepvars "`plugin_indepvars' `v'"
            local coef_indepvars "`coef_indepvars' `term'"
            local omit_priority_list "`omit_priority_list' `priority'"
        }
    }

    local p : word count `plugin_indepvars'
    tempname omit_priority
    if (`p' > 0) {
        matrix `omit_priority' = J(`p', 1, 0)
        local pi = 0
        foreach val of local omit_priority_list {
            local ++pi
            matrix `omit_priority'[`pi',1] = `val'
        }
    }

    // Build the plugin varlist in the exact order expected by the plugin.
    local plugin_varlist "`depvar' `plugin_indepvars' `fe_ids' `slope_absorb_vars' `clust_ids'"
    if (`has_weight') {
        local plugin_varlist "`plugin_varlist' `wvar'"
    }
    if (`group_mode') {
        local plugin_varlist "`plugin_varlist' `group_id'"
    }
    if (`has_individual') {
        local plugin_varlist "`plugin_varlist' `individual_id'"
    }
    if (`ninst' > 0) {
        local plugin_varlist "`plugin_varlist' `instruments'"
    }
    if (`store_resid') {
        local plugin_varlist "`plugin_varlist' `resid_var'"
    }
    if (`store_groupvar') {
        local plugin_varlist "`plugin_varlist' `groupvar_var'"
    }
    if (`store_fes') {
        local plugin_varlist "`plugin_varlist' `fe_out_vars'"
    }
    local plugin_varlist "`plugin_varlist' `esample'"

    // Matrices and scalars to receive results.
    tempname b V
    local k = `p' + `fit_intercept'
    matrix `b' = J(1, `k', .)
    matrix `V' = J(`k', `k', .)
    tempname omit_reason
    if (`k' > 0) {
        matrix `omit_reason' = J(`k', 1, 0)
    }

    // Name columns/rows to match Stata conventions.
    local coefnames "`coef_indepvars'"
    if (`fit_intercept') local coefnames "`coefnames' _cons"
    matrix colnames `b' = `coefnames'
    matrix colnames `V' = `coefnames'
    matrix rownames `V' = `coefnames'
    if (`k' > 0) {
        matrix rownames `omit_reason' = `coefnames'
        matrix colnames `omit_reason' = reason
    }

    // Cluster diagnostics matrix (counts for each cluster combination).
    local cluster_diag_mat ""
    if (`nclust' > 0) {
        local ncomb = 2^`nclust' - 1
        tempname cluster_diag
        matrix `cluster_diag' = J(`ncomb', 1, .)
        local rownames ""
        forvalues mask = 1/`ncomb' {
            local name ""
            forvalues j = 1/`nclust' {
                local bit = 2^(`j' - 1)
                if (mod(floor(`mask'/`bit'), 2)) {
                    local cname : word `j' of `cluster_display'
                    if ("`name'" == "") local name "`cname'"
                    else local name "`name'#`cname'"
                }
            }
            local rownames "`rownames' `name'"
        }
        matrix rownames `cluster_diag' = `rownames'
        local cluster_diag_mat "`cluster_diag'"
    }

    tempname sN sNfull sNsng sDFr sDFRUnadj sDFm sDFa sDFaLevels sDFaExact sDFaNested ///
        sR2 sR2w sSig2 sRss sTss sTssw sSat sIter sConv sThr sGpuUsed sGpuStatus ///
        sGpuAttempted sGpuAbsConv sGpuAbsIter sMeth sNclust sClustScale sVcvFix ///
        sFeRecConv sFeRecIter sFeRecMaxDelta
    scalar `sGpuUsed' = 0
    scalar `sGpuStatus' = 0
    scalar `sSat' = 0
    scalar `sGpuAttempted' = 0
    scalar `sGpuAbsConv' = .
    scalar `sGpuAbsIter' = .
    scalar `sVcvFix' = 0
    scalar `sFeRecConv' = .
    scalar `sFeRecIter' = .
    scalar `sFeRecMaxDelta' = .

    local have_dof_table 0
    tempname dof_table
    if (`nfe' > 0) {
        // Column 6 receives the per-dimension count of distinct FE levels in
        // e(sample) from the plugin (used by the slope DoF display, saving an
        // O(N log N) egen-tag recount on huge datasets).
        matrix `dof_table' = J(`nfe', 6, .)
        local have_dof_table 1
    }

    // Assemble plugin args as a single packed token to avoid `plugin call` argument limits.
    local cfg "cfg=b=`b';V=`V';p=`p';nfe=`nfe';nslope=`nslope';nclust=`nclust';ninst=`ninst';"
    local cfg "`cfg'has_weight=`has_weight';group_mode=`group_mode';has_individual=`has_individual';"
    if (`has_weight') {
        local cfg "`cfg'weight_type=`weight';"
    }
    local cfg "`cfg'store_resid=`store_resid';store_groupvar=`store_groupvar';store_fes=`store_fes';savefe_out_count=`fe_out_count';"
    local cfg "`cfg'se_type=`se_type';tol=`tolerance';convergence=`convergence';fetol=`fetolerance';max_iter=`maxiter';fit_intercept=`fit_intercept';"
    local cfg "`cfg'num_threads=`numthreads';default_threads=`defaultthreads';max_threads=`maxthreads';"
    local cfg "`cfg'min_parallel_rows=`minparrows';target_rows_per_thread=`targetrpt';"
    local cfg "`cfg'drop_singletons=`drop_singletons';retain_fes=`retain_fes';symmetric_sweep=`symmetric_sweep';"
    local cfg "`cfg'tolerance_mode=`tolerance_mode';"
    if (!`group_mode') {
        local cfg "`cfg'esample_prefilled=1;"
    }
    local cfg "`cfg'absorption_method=`absorptionmethod';jacobi_relaxation=`jacobirelaxation';level=`level';"
    local cfg "`cfg's_N=`sN';s_N_full=`sNfull';s_num_singletons=`sNsng';s_df_r=`sDFr';"
    local cfg "`cfg's_df_r_unadj=`sDFRUnadj';s_df_m=`sDFm';s_df_a=`sDFa';"
    local cfg "`cfg's_df_a_levels=`sDFaLevels';s_df_a_exact=`sDFaExact';s_df_a_nested=`sDFaNested';"
    local cfg "`cfg's_r2=`sR2';s_r2_within=`sR2w';s_sigma2=`sSig2';s_rss=`sRss';"
    local cfg "`cfg's_tss=`sTss';s_tss_within=`sTssw';s_saturated=`sSat';"
    local cfg "`cfg's_iterations=`sIter';s_converged=`sConv';"
    local cfg "`cfg's_fe_recovery_converged=`sFeRecConv';s_fe_recovery_iterations=`sFeRecIter';s_fe_recovery_max_delta=`sFeRecMaxDelta';"
    local cfg "`cfg's_threads_used=`sThr';s_gpu_used=`sGpuUsed';s_method_used=`sMeth';"
    local cfg "`cfg's_gpu_status_code=`sGpuStatus';s_gpu_attempted=`sGpuAttempted';"
    local cfg "`cfg's_gpu_absorption_converged=`sGpuAbsConv';s_gpu_absorption_iterations=`sGpuAbsIter';"
    local cfg "`cfg's_num_clusters=`sNclust';s_cluster_scale=`sClustScale';"
    local cfg "`cfg's_vcv_psd_fixed=`sVcvFix';"

    if (`ninst' > 0) {
        local cfg "`cfg'endogenous_idx=`endogenous_idx';"
    }
    if (`has_individual') {
        local cfg "`cfg'aggregation=`aggregation';individual_fe_pos=`individual_fe_pos';"
    }
    if ("`dofadjustments'" != "") {
        local dofadjustments_clean : subinstr local dofadjustments " " "," , all
        local cfg "`cfg'dofadjustments=`dofadjustments_clean';"
    }
    if ("`cluster_diag_mat'" != "") {
        local cfg "`cfg'cluster_diag=`cluster_diag_mat';"
    }
    if ("`cluster_fe_map'" != "") {
        local cluster_fe_map : list retokenize cluster_fe_map
        local cluster_fe_map_cfg : subinstr local cluster_fe_map " " "," , all
        local cfg "`cfg'cluster_fe_map=`cluster_fe_map_cfg';"
    }
    if (`fe_has_dup') {
        local fe_dup_map : list retokenize fe_dup_map
        local fe_dup_map_cfg : subinstr local fe_dup_map " " "," , all
        local cfg "`cfg'fe_dup_map=`fe_dup_map_cfg';"
    }
    if (`nslope' > 0) {
        local cfg "`cfg'slope_fe_map=`slope_fe_map';slope_intercepts=`slope_intercepts';"
    }
    if (`have_dof_table') {
        local cfg "`cfg'dof_table=`dof_table';"
    }
    if (`k' > 0) {
        local cfg "`cfg'omit_reason=`omit_reason';"
    }
    if (`p' > 0) {
        local cfg "`cfg'omit_priority=`omit_priority';"
    }
    if ("`ssc_k_adj'" != "") local cfg "`cfg'ssc_k_adj=`ssc_k_adj';"
    if ("`ssc_k_fixef'" != "") local cfg "`cfg'ssc_k_fixef=`ssc_k_fixef';"
    if ("`ssc_k_exact'" != "") local cfg "`cfg'ssc_k_exact=`ssc_k_exact';"
    if ("`ssc_g_adj'" != "") local cfg "`cfg'ssc_g_adj=`ssc_g_adj';"
    if ("`ssc_g_df'" != "") local cfg "`cfg'ssc_g_df=`ssc_g_df';"
    if ("`ssc_t_df'" != "" & "`ssc_t_df'" != "min" & "`ssc_t_df'" != "conventional") {
        local cfg "`cfg'ssc_t_df=`ssc_t_df';"
    }
    if ("`statstyle'" != "") local cfg "`cfg'statstyle=`statstyle';"
    if ("`ferecoverymethod'" != "") {
        local cfg "`cfg'fe_recovery_method=`ferecoverymethod';"
    }
    if ("`gpu_backend'" != "") {
        local cfg "`cfg'gpu_backend=`gpu_backend';"
    }
    if ("`mobility_profile'" != "") {
        local cfg "`cfg'mobility_profile=`mobility_profile';"
    }
    if ("`mobility_profile_mode'" != "") {
        local cfg "`cfg'mobility_profile_mode=`mobility_profile_mode';"
    }
    if ("`absorption_cache'" != "") {
        local cfg "`cfg'absorption_cache=`absorption_cache';"
    }
    if ("`absorption_cache_mode'" != "") {
        local cfg "`cfg'absorption_cache_mode=`absorption_cache_mode';"
    }
    if ("`fe_structure_cache'" != "") {
        local cfg "`cfg'fe_structure_cache=`fe_structure_cache';"
    }
    if ("`fe_structure_cache_mode'" != "") {
        local cfg "`cfg'fe_structure_cache_mode=`fe_structure_cache_mode';"
    }

    // Bind the plugin to the same directory as the active xhdfe.ado so we do not
    // silently pick up a stale CPU-only plugin from another adopath entry.
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
        di as err "xhdfe: the active session is still bound to an older xhdfe.plugin path"
        di as err "xhdfe: run discard (with no arguments) and rerun the command so Stata reloads the current plugin"
        exit 498
    }
    capture program `plugin_prog', plugin using("`plugin_path'")
    if (_rc & _rc != 110) {
        di as err "xhdfe.plugin could not be loaded from `plugin_path'"
        exit _rc
    }
    global XHDFE_PLUGIN_PATH_INTERNAL "`plugin_path'"

    if (`__xhdfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=pre_plugin ms=" %12.3f (1000 * r(t99))
        timer clear 99
        timer on 99
    }

    // Run plugin.
    capture noisily plugin call `plugin_prog' `plugin_varlist' if `touse', "`cfg'"
    local rc = _rc
    if (`__xhdfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=plugin_call ms=" %12.3f (1000 * r(t99))
        timer clear 99
        timer on 99
    }
    if (`rc') {
        if (`rc' == 2000) {
            quietly count if `touse'
            if (r(N) == 0) {
                di as err "xhdfe: no observations after applying if/in and removing missing values"
            }
            else {
                di as err "xhdfe: no observations left for estimation (all observations may have been dropped as singletons)"
            }
        }
        if (`fv_created') {
            capture drop `fv_stub'*
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

    // When the user explicitly requests a GPU backend, do not accept CPU
    // fallback results. GPU failures and non-convergence must be surfaced as
    // errors with diagnostics rather than silently changing backend.
    local reject_cpu_gpu = ("`gpu_backend'" != "" & "`gpu_backend'" != "cpu" & scalar(`sGpuUsed') < 0.5)
    if (`reject_cpu_gpu') {
        if (`fv_created') {
            capture drop `fv_stub'*
        }
        local gpu_backend_disp = upper("`gpu_backend'")
        if ("`gpu_status'" == "backend_unavailable") {
            di as err "xhdfe: requested gpubackend(`gpu_backend'), but `gpu_backend_disp' was not available at runtime"
            di as err "xhdfe: check the GPU device/runtime and that xhdfe.plugin was built with `gpu_backend' support"
            di as err "xhdfe: if you recently rebuilt or switched plugins, run discard (with no arguments) and rerun the command"
            if ("`gpu_backend'" == "cuda" & "`c(os)'" == "Unix") {
                di as err "xhdfe: the net-install plugin is CPU-only; on a machine with an NVIDIA GPU run {bf:xhdfegpu} once to build and install a CUDA plugin for this hardware"
            }
        }
        else if ("`gpu_status'" == "gpu_absorption_not_converged") {
            di as err "xhdfe: requested gpubackend(`gpu_backend'), but `gpu_backend_disp' absorption did not converge"
            if (scalar(`sGpuAbsIter') < .) {
                di as err "xhdfe: `gpu_backend_disp' absorption reached " %9.0g scalar(`sGpuAbsIter') " iterations without convergence"
            }
            if (scalar(`sConv') == 0 & scalar(`sIter') < .) {
                di as err "xhdfe: CPU fallback is disabled for explicit GPU requests"
            }
            else {
                di as err "xhdfe: CPU fallback is disabled for explicit GPU requests"
            }
        }
        else if ("`gpu_status'" == "gpu_backend_failed") {
            di as err "xhdfe: requested gpubackend(`gpu_backend'), but `gpu_backend_disp' absorption failed before producing a converged result"
            di as err "xhdfe: CPU fallback was rejected because a GPU backend was explicitly requested"
        }
        else if ("`gpu_status'" == "cpu_cache_or_profile_result") {
            di as err "xhdfe: requested gpubackend(`gpu_backend'), but the effective absorption result came from CPU cache/profile state"
            di as err "xhdfe: rerun with matching GPU cache/profile state, disable absorption cache, or clear stale mobility/profile files"
        }
        else {
            di as err "xhdfe: requested gpubackend(`gpu_backend') but the effective backend was CPU"
            di as err "xhdfe: rebuild the plugin with `gpu_backend' support and ensure the requested backend is available at runtime"
            di as err "xhdfe: if you recently rebuilt or switched plugins, run discard (with no arguments) and rerun the command"
        }
        exit 498
    }

    if (scalar(`sConv') == 0) {
        if (`fv_created') {
            capture drop `fv_stub'*
        }
        di as err "xhdfe: HDFE absorption did not converge"
        if (scalar(`sIter') < .) {
            di as err "xhdfe: absorption reached " %9.0g scalar(`sIter') " iterations without convergence"
        }
        di as err "xhdfe: refusing to post estimates from a non-converged absorption"
        exit 498
    }

    // Refuse to write fixed-effect contributions recovered from a non-converged
    // recovery loop. The estimation coefficients can converge while the FE
    // recovery sweep does not; saving those FEs would silently violate the
    // requested tolerance. Drop the (already-written) FE output variables so a
    // failed command leaves no imprecise output behind.
    if (`store_fes' & scalar(`sFeRecConv') == 0) {
        if (`fv_created') {
            capture drop `fv_stub'*
        }
        foreach v of local fe_out_vars {
            capture drop `v'
        }
        di as err "xhdfe: fixed-effect recovery did not converge"
        if (scalar(`sFeRecIter') < .) {
            di as err "xhdfe: FE recovery reached " %9.0g scalar(`sFeRecIter') " iterations without convergence"
        }
        if (scalar(`sFeRecMaxDelta') < .) {
            di as err "xhdfe: final FE recovery max change = " %12.4e scalar(`sFeRecMaxDelta')
        }
        di as err "xhdfe: refusing to save fixed effects from a non-converged recovery (raise maxiter() or fetolerance())"
        exit 498
    }

    // Stata postestimation (e.g., test/lincom) fails if e(V) has missing values.
    // The plugin uses missing for the intercept variance when it is not identified.
    mata: _xhdfe_fix_v("`V'")

    local dof_labels ""
    if (`have_dof_table') {
        local dof_display_rows 0
        forvalues i = 1/`nfe' {
            local kind : word `i' of `absorb_kinds'
            local slopeintercept : word `i' of `absorb_slopeintercepts'
            if ("`kind'" == "slope" & "`slopeintercept'" == "1") {
                local dof_display_rows = `dof_display_rows' + 2
            }
            else {
                local ++dof_display_rows
            }
        }

        tempname dof_display_table
        matrix `dof_display_table' = J(`dof_display_rows', 5, .)
        local dof_rowstripe ""
        local dof_new_i 0

        forvalues dof_i = 1/`nfe' {
            local entry : word `dof_i' of `absorb_ordered'
            local label : word `dof_i' of `absorb_display'
            if ("`label'" == "") local label "`entry'"
            local kind : word `dof_i' of `absorb_kinds'
            local slopevar : word `dof_i' of `absorb_slopevars'
            local slopeintercept : word `dof_i' of `absorb_slopeintercepts'

            if ("`kind'" == "slope") {
                local group_label "`entry'"
                local group_label : subinstr local group_label "##c.`slopevar'" "", all
                local group_label : subinstr local group_label "#c.`slopevar'" "", all
                local slope_label "`group_label'#c.`slopevar'"

                // Prefer the plugin-provided count of distinct FE levels in
                // e(sample) (dof_table col 6); the egen-tag recount below is
                // a fallback only (it costs a full-dataset sort per slope
                // term, ~84s on 56M rows).
                local base_levels = `dof_table'[`dof_i',6]
                if (missing(`base_levels')) {
                    local feid : word `dof_i' of `fe_ids'
                    tempvar dof_tag
                    quietly egen byte `dof_tag' = tag(`feid') if `esample'
                    quietly count if `dof_tag'
                    local base_levels = r(N)
                    capture drop `dof_tag'
                }

                local total_levels = `dof_table'[`dof_i',1]
                local total_redundant = `dof_table'[`dof_i',2]
                local inexact = `dof_table'[`dof_i',4]
                local nested = `dof_table'[`dof_i',5]

                if ("`slopeintercept'" == "1") {
                    local extra_levels = `total_levels' - `base_levels'
                    if (missing(`extra_levels') | `extra_levels' < 0) local extra_levels = `base_levels'
                    local slope_redundant = max(`base_levels' - `extra_levels', 0)
                    if (`slope_redundant' > `total_redundant') local slope_redundant = `total_redundant'
                    local base_redundant = `total_redundant' - `slope_redundant'
                    local base_coefs = `base_levels' - `base_redundant'
                    local slope_coefs = `base_levels' - `slope_redundant'

                    local ++dof_new_i
                    matrix `dof_display_table'[`dof_new_i',1] = `base_levels'
                    matrix `dof_display_table'[`dof_new_i',2] = `base_redundant'
                    matrix `dof_display_table'[`dof_new_i',3] = `base_coefs'
                    matrix `dof_display_table'[`dof_new_i',4] = `inexact'
                    matrix `dof_display_table'[`dof_new_i',5] = `nested'
                    local dof_labels "`dof_labels' `group_label'"
                    local rn "`group_label'"
                    local rn : subinstr local rn "##" "_X_" , all
                    local rn : subinstr local rn "#" "_X_" , all
                    local rn : subinstr local rn "." "_" , all
                    if (strlen("`rn'") > 28) local rn "FE`dof_new_i'"
                    capture confirm name `rn'
                    if (_rc) local rn "FE`dof_new_i'"
                    local dof_rowstripe "`dof_rowstripe' 1.`rn'"

                    local ++dof_new_i
                    matrix `dof_display_table'[`dof_new_i',1] = `base_levels'
                    matrix `dof_display_table'[`dof_new_i',2] = `slope_redundant'
                    matrix `dof_display_table'[`dof_new_i',3] = `slope_coefs'
                    matrix `dof_display_table'[`dof_new_i',4] = `inexact'
                    matrix `dof_display_table'[`dof_new_i',5] = `nested'
                    local dof_labels "`dof_labels' `slope_label'"
                    local rn "`slope_label'"
                    local rn : subinstr local rn "##" "_X_" , all
                    local rn : subinstr local rn "#" "_X_" , all
                    local rn : subinstr local rn "." "_" , all
                    if (strlen("`rn'") > 28) local rn "FE`dof_new_i'"
                    capture confirm name `rn'
                    if (_rc) local rn "FE`dof_new_i'"
                    local dof_rowstripe "`dof_rowstripe' 1.`rn'"
                }
                else {
                    local slope_coefs = `dof_table'[`dof_i',3]
                    local slope_redundant = max(`base_levels' - `slope_coefs', 0)

                    local ++dof_new_i
                    matrix `dof_display_table'[`dof_new_i',1] = `base_levels'
                    matrix `dof_display_table'[`dof_new_i',2] = `slope_redundant'
                    matrix `dof_display_table'[`dof_new_i',3] = `slope_coefs'
                    matrix `dof_display_table'[`dof_new_i',4] = `inexact'
                    matrix `dof_display_table'[`dof_new_i',5] = `nested'
                    local dof_labels "`dof_labels' `slope_label'"
                    local rn "`slope_label'"
                    local rn : subinstr local rn "##" "_X_" , all
                    local rn : subinstr local rn "#" "_X_" , all
                    local rn : subinstr local rn "." "_" , all
                    if (strlen("`rn'") > 28) local rn "FE`dof_new_i'"
                    capture confirm name `rn'
                    if (_rc) local rn "FE`dof_new_i'"
                    local dof_rowstripe "`dof_rowstripe' 1.`rn'"
                }
            }
            else {
                local ++dof_new_i
                forvalues c = 1/5 {
                    matrix `dof_display_table'[`dof_new_i',`c'] = `dof_table'[`dof_i',`c']
                }
                local dof_labels "`dof_labels' `label'"
                local rn "`label'"
                local rn : subinstr local rn "##" "_X_" , all
                local rn : subinstr local rn "#" "_X_" , all
                local rn : subinstr local rn "." "_" , all
                if (strlen("`rn'") > 28) local rn "FE`dof_new_i'"
                capture confirm name `rn'
                if (_rc) local rn "FE`dof_new_i'"
                local dof_rowstripe "`dof_rowstripe' 1.`rn'"
            }
        }

        matrix `dof_table' = `dof_display_table'
        matrix rownames `dof_table' = `dof_rowstripe'
        matrix colnames `dof_table' = Categories Redundant "Num Coefs" "Inexact?" "Nested?"
    }

    // Sum of weights on the estimation sample.
    tempname sumweights
    scalar `sumweights' = .
    if (`has_weight') {
        quietly summarize `wvar' if `esample', meanonly
        scalar `sumweights' = r(sum)
    }
    else {
        scalar `sumweights' = scalar(`sN')
    }

    // Force-omit regressors that are main effects of absorbed FE variables (reghdfe-style).
    local forced_omit 0
    if (`k' > 0 & "`absorb_markvars'" != "") {
        local cn : colnames `b'
        local kk = colsof(`b')
        forvalues j = 1/`kk' {
            local name : word `j' of `cn'
            if ("`name'" == "_cons") continue
            if (strpos("`name'", "#") > 0) continue
            if (substr("`name'", 1, 2) == "o.") continue
            local base "`name'"
            if (regexm("`name'", "^[0-9]+[^.]*\.(.+)$")) {
                local base = regexs(1)
            }
            local pos : list posof "`base'" in absorb_markvars
            if (`pos' == 0) continue

            matrix `b'[1,`j'] = 0
            forvalues jj = 1/`kk' {
                matrix `V'[`j',`jj'] = 0
                matrix `V'[`jj',`j'] = 0
            }
            matrix `omit_reason'[`j',1] = 1
            local forced_omit 1
        }
    }

    // Mark collinear regressors as omitted (Stata-style).
    _ms_findomitted `b' `V'

    // Recompute _cons after forced omissions so it matches reported coefficients.
    if (`forced_omit' & `fit_intercept') {
        local wgt ""
        if (`has_weight') local wgt " [`weight'=`wvar']"
        tempname xbarb
        scalar `xbarb' = 0
        local idx = 0
        foreach v of local plugin_indepvars {
            local ++idx
            local bval = `b'[1,`idx']
            if (`bval' != 0 & `bval' < .) {
                quietly summarize `v' `wgt' if `esample', meanonly
                scalar `xbarb' = scalar(`xbarb') + r(mean) * `bval'
            }
        }
        quietly summarize `depvar' `wgt' if `esample', meanonly
        matrix `b'[1,`k'] = r(mean) - scalar(`xbarb')
    }

    // reghdfe parity: a slope-only absorb() fit has no intercept anywhere, and
    // reghdfe reports no _cons row for it. Drop the recovered constant column
    // (the estimator itself never included it; df_r/df_m are unchanged).
    local report_cons = `fit_intercept'
    if (`fit_intercept' & !`model_has_cons' & `nendog' == 0 & `p' > 0) {
        matrix `b' = `b'[1, 1..`p']
        matrix `V' = `V'[1..`p', 1..`p']
        matrix `omit_reason' = `omit_reason'[1..`p', 1]
        local k = `p'
        local report_cons 0
    }

    // reghdfe/Stata parity for factor variables: re-insert base and omitted
    // levels as structural-zero columns with canonical stripe names, so e(b)
    // carries e.g. "1b.cat" (base) and "2.cat" keeps its real coefficient
    // instead of being re-marked as the base level by ereturn post.
    if (`fv_created') {
        quietly fvexpand `indepvars_raw' if `esample'
        local canon_terms "`r(varlist)'"
        mata: _xhdfe_expand_fv_base("`b'", "`V'", "`omit_reason'", "`canon_terms'")
    }

    if (`fv_created') {
        capture drop `fv_stub'*
    }

    local absorb_ordered = strtrim("`absorb_ordered'")
    local absorb_display = strtrim("`absorb_display'")
    local cluster_display = strtrim("`cluster_display'")
    local cluster_list = strtrim("`cluster_list'")

    // Effective model degrees of freedom (exclude omitted coefficients).
    local cn : colnames `b'
    local kk = colsof(`b')
    local df_m_eff 0
    forvalues j = 1/`kk' {
        local name : word `j' of `cn'
        if ("`name'" == "_cons") continue
        // Prefer explicit omission tracking from C++/forced omission pass.
        if (`omit_reason'[`j',1] > 0) continue
        // Guard against base/omitted FV terms that may still appear in colnames.
        if (regexm("`name'", "([0-9]+b\.)")) continue
        if (regexm("`name'", "([0-9]+bn\.)")) continue
        if (regexm("`name'", "^o\.")) continue
        if (regexm("`name'", "([0-9]+o\.)")) continue
        local ++df_m_eff
    }

    // Post results. ereturn post's dof() option requires a nonnegative value
    // and silently converts negatives to missing. When the model is saturated
    // (df_r <= 0), omit dof() here and set e(df_r) via ereturn scalar so
    // reghdfe-style negative residual df is reported as-is.
    local dof_raw = scalar(`sDFr')
    local dof_arg ""
    if (!missing(`dof_raw') & `dof_raw' >= 0) {
        local dof_arg "dof(`dof_raw')"
    }
    local esample_postopt "esample(`esample')"
    if ("`nosample'" != "") local esample_postopt ""
    ereturn post `b' `V', `esample_postopt' obs(`=scalar(`sN')') `dof_arg' depname(`depvar') buildfvinfo
    if (!missing(`dof_raw') & `dof_raw' < 0) {
        ereturn scalar df_r = `dof_raw'
    }
    ereturn local cmd "xhdfe"
    ereturn local cmdline "`cmdline'"
    ereturn local title "HDFE Linear regression"
    ereturn local depvar "`depvar'"
    ereturn local indepvars "`coef_indepvars'"
    ereturn local absorb "`absorb_display'"
    // Canonical absorbed-variable names (reghdfe-compatible): underlying variables, not aliases.
    ereturn local absvars "`absorb_ordered'"
    ereturn local extended_absvars "`absorb_ordered'"
    // Keep alias/display labels available for UI/reporting.
    ereturn local absorb_labels "`absorb_display'"
    if (`nfe' > 0) {
        local title2 "Absorbing `nfe' HDFE group"
        if (`nfe' != 1) local title2 "Absorbing `nfe' HDFE groups"
        ereturn local title2 "`title2'"
    }
    local dofmethod "pairwise clusters continuous"
    if ("`dofadjustments'" != "") {
        local dofmethod "`dofadjustments'"
    }
    local dofmethod : subinstr local dofmethod "," " ", all
    local dofmethod : list retokenize dofmethod
    ereturn local dofmethod "`dofmethod'"
    if ("`dof_labels'" != "") {
        ereturn local dof_labels "`dof_labels'"
    }
    ereturn local predict "xhdfe_p"
    ereturn local marginsnotok "Residuals SCore"
    ereturn local footnote "xhdfe__footnote"
    ereturn local estat_cmd "xhdfe_estat"
    ereturn local model "ols"
    ereturn local version "2.13.0 07jul2026"
    if ("`nowarn'" != "") {
        ereturn local nowarn "nowarn"
    }
    if (`store_resid') {
        ereturn local resid "`resid_var'"
    }
    if ("`group'" != "") {
        ereturn local group "`group'"
    }
    if ("`individual'" != "") {
        ereturn local individual "`individual'"
        ereturn local aggregation "`aggregation'"
    }
    if ("`cluster_list'" != "") {
        ereturn local clustvar "`cluster_display'"
        if ("`cluster_diag_mat'" != "") {
            ereturn scalar N_clustervars = `nclust'
            forvalues j = 1/`nclust' {
                tempname clust_count
                local mask = 2^(`j' - 1)
                scalar `clust_count' = `cluster_diag_mat'[`mask',1]
                ereturn scalar N_clust`j' = `clust_count'
            }
        }
    }
    if ("`se_type'" == "cluster") {
        ereturn local vce "cluster"
        ereturn local vcetype "Robust"
    }
    else if ("`se_type'" == "robust") {
        ereturn local vce "robust"
        ereturn local vcetype "Robust"
    }
    else {
        ereturn local vce "ols"
    }
    ereturn scalar N_full = scalar(`sNfull')
    ereturn scalar num_singletons = scalar(`sNsng')
    ereturn scalar sumweights = scalar(`sumweights')
    ereturn scalar drop_singletons = `drop_singletons'
    ereturn scalar df_r_unadj = scalar(`sDFRUnadj')
    ereturn scalar df_a = scalar(`sDFa')
    ereturn scalar df_a_levels = scalar(`sDFaLevels')
    ereturn scalar df_a_exact = scalar(`sDFaExact')
    ereturn scalar df_a_nested = scalar(`sDFaNested')
    if (missing(e(df_a_nested))) ereturn scalar df_a_nested = 0
    ereturn scalar df_a_initial = e(df_a_levels)
    ereturn scalar df_a_redundant = e(df_a_initial) - e(df_a)
    if (e(df_a_redundant) < 0) ereturn scalar df_a_redundant = 0
    ereturn scalar N_hdfe = `nfe'
    ereturn scalar N_hdfe_extended = `nfe'
    if (`have_dof_table') {
        ereturn matrix dof_table = `dof_table'
    }
    if (`k' > 0) {
        ereturn matrix omit_reason = `omit_reason'
    }
    ereturn scalar r2 = scalar(`sR2')
    ereturn scalar r2_within = scalar(`sR2w')
    ereturn scalar sigma2 = scalar(`sSig2')
    ereturn scalar rss = scalar(`sRss')
    ereturn scalar tss = scalar(`sTss')
    ereturn scalar tss_within = scalar(`sTssw')
    ereturn scalar saturated = scalar(`sSat')
    if (`nfe' == 0) {
        ereturn scalar r2_within = e(r2)
        ereturn scalar tss_within = e(tss)
    }
    ereturn scalar rmse = sqrt(e(sigma2))
    ereturn scalar iterations = scalar(`sIter')
    ereturn scalar ic = scalar(`sIter')
    ereturn scalar converged = scalar(`sConv')
    if (`store_fes') {
        ereturn scalar fe_recovery_converged = scalar(`sFeRecConv')
        ereturn scalar fe_recovery_iterations = scalar(`sFeRecIter')
        ereturn scalar fe_recovery_max_delta = scalar(`sFeRecMaxDelta')
    }
    ereturn scalar report_constant = `report_cons'
    ereturn scalar threads_used = scalar(`sThr')
    ereturn scalar gpu_used = scalar(`sGpuUsed')
    ereturn scalar gpu_status_code = scalar(`sGpuStatus')
    ereturn scalar gpu_attempted = scalar(`sGpuAttempted')
    ereturn scalar gpu_absorption_converged = scalar(`sGpuAbsConv')
    ereturn scalar gpu_absorption_iterations = scalar(`sGpuAbsIter')
    ereturn local gpu_status "`gpu_status'"
    ereturn local tolerance_mode "`tolerance_mode'"
    if ("`gpu_backend'" != "") {
        ereturn local gpu_backend_requested "`gpu_backend'"
        if (e(gpu_used) > 0.5) ereturn local gpu_backend "`gpu_backend'"
        else ereturn local gpu_backend "cpu"
    }
    else if (e(gpu_used) > 0.5) {
        // GPU was used but selected via the XHDFE_GPU_BACKEND environment
        // variable rather than the gpubackend() option; surface the backend
        // macros so diagnostics are consistent across both selection paths.
        local gpu_env_backend : environment XHDFE_GPU_BACKEND
        local gpu_env_backend = lower(strtrim("`gpu_env_backend'"))
        if (inlist("`gpu_env_backend'", "cuda", "metal")) {
            ereturn local gpu_backend_requested "`gpu_env_backend'"
            ereturn local gpu_backend "`gpu_env_backend'"
        }
        else {
            ereturn local gpu_backend "cuda"
        }
    }
    ereturn scalar absorption_method_used = scalar(`sMeth')
    if (scalar(`sNclust') < .) ereturn scalar N_clust = scalar(`sNclust')
    if (scalar(`sClustScale') < .) ereturn scalar cluster_scale = scalar(`sClustScale')
    ereturn scalar vcv_psd_fixed = cond(scalar(`sVcvFix') < ., scalar(`sVcvFix'), 0)
    if ("`cluster_diag_mat'" != "") {
        ereturn matrix cluster_diag = `cluster_diag_mat'
    }

    // Reghdfe-compat metadata fixes that do not alter the estimator:
    // - dofadjustments(none) still removes trivial intercept redundancies;
    // - clustered degenerate fits derive N_clust/df_r from the realized sample.
    if ("`dofmethod'" == "none" & `fit_intercept' & `nfe' > 0) {
        capture confirm matrix e(dof_table)
        if (!_rc) {
            tempname dof_fix V_fix
            matrix `dof_fix' = e(dof_table)
            local rows = rowsof(`dof_fix')
            local nonnested 0
            local current_redundant 0
            forvalues i = 1/`rows' {
                if (`dof_fix'[`i',5] == 1) continue
                local ++nonnested
                local current_redundant = `current_redundant' + `dof_fix'[`i',2]
            }
            local target_redundant = max(`nonnested' - 1, 0)
            local add_redundant = max(`target_redundant' - `current_redundant', 0)
            if (`add_redundant' > 0) {
                local skipped_first 0
                forvalues i = 1/`rows' {
                    if (`dof_fix'[`i',5] == 1) continue
                    if (`skipped_first' == 0) {
                        local skipped_first 1
                        continue
                    }
                    if (`add_redundant' <= 0) continue, break
                    if (`dof_fix'[`i',2] < 1) {
                        matrix `dof_fix'[`i',2] = 1
                        matrix `dof_fix'[`i',3] = `dof_fix'[`i',1] - `dof_fix'[`i',2]
                        local --add_redundant
                    }
                }

                local old_df_a = e(df_a)
                local old_df_r_unadj = e(df_r_unadj)
                local new_df_a 0
                forvalues i = 1/`rows' {
                    local new_df_a = `new_df_a' + `dof_fix'[`i',3]
                }
                ereturn matrix dof_table = `dof_fix'
                ereturn scalar df_a = `new_df_a'
                ereturn scalar df_a_exact = `new_df_a'
                ereturn scalar df_a_redundant = max(e(df_a_initial) - e(df_a), 0)

                if (!missing(`old_df_r_unadj')) {
                    local new_df_r_unadj = `old_df_r_unadj' + (`old_df_a' - `new_df_a')
                    ereturn scalar df_r_unadj = `new_df_r_unadj'
                    ereturn scalar df_r = `new_df_r_unadj'
                    if ("`e(vce)'" == "ols" & `old_df_r_unadj' > 0 & `new_df_r_unadj' > 0) {
                        matrix `V_fix' = e(V) * (`old_df_r_unadj' / `new_df_r_unadj')
                        mata: _xhdfe_fix_v("`V_fix'")
                        ereturn repost V = `V_fix'
                    }
                }
            }
        }
    }

    if ("`cluster_list'" != "" & (missing(e(N_clust)) | e(N_clust) <= 0)) {
        local derived_nclust .
        local have_derived 0
        foreach cl of varlist `cluster_list' {
            quietly levelsof `cl' if e(sample), local(_xhdfe_cluster_levels)
            local cluster_count : word count `_xhdfe_cluster_levels'
            if (!`have_derived' | `cluster_count' < `derived_nclust') {
                local derived_nclust `cluster_count'
                local have_derived 1
            }
        }
        if (`have_derived') {
            ereturn scalar N_clust = `derived_nclust'
        }
    }

    // Optional t df override (fixest-style).
    if ("`ssc_t_df'" == "min" & !missing(e(N_clust))) {
        ereturn scalar df_r = e(N_clust) - 1
    }
    else if ("`ssc_t_df'" == "conventional" & !missing(e(df_r_unadj))) {
        ereturn scalar df_r = e(df_r_unadj)
    }
    else if ("`cluster_list'" != "" & "`statstyle'" == "reghdfe" & !missing(e(N_clust)) & e(N_clust) > 0) {
        local df_r_base = e(df_r)
        if (!missing(e(df_r_unadj))) local df_r_base = e(df_r_unadj)
        ereturn scalar df_r = min(`df_r_base', e(N_clust) - 1)
    }

    ereturn scalar df_m = `df_m_eff'
    ereturn scalar rank = `df_m_eff'
    // Guard: treat missing e(df_r) as invalid (Stata orders missing as +infinity,
    // so the raw comparison e(df_r) > 0 would be true for missing).
    if (`df_m_eff' > 0 & !missing(e(df_r)) & e(df_r) > 0) {
        local keep_cols
        local idx = 0
        foreach name of local cn {
            local ++idx
            if ("`name'" == "_cons") continue
            if (e(omit_reason)[`idx',1] > 0) continue
            if (regexm("`name'", "([0-9]+b\.)")) continue
            if (regexm("`name'", "([0-9]+bn\.)")) continue
            if (regexm("`name'", "^o\.")) continue
            if (regexm("`name'", "([0-9]+o\.)")) continue
            local keep_cols "`keep_cols' `idx'"
        }

        tempname b1 V1 Vinv W Fstat
        matrix `b1' = J(1, `df_m_eff', .)
        matrix `V1' = J(`df_m_eff', `df_m_eff', .)

        local ii = 0
        foreach c of local keep_cols {
            local ++ii
            matrix `b1'[1,`ii'] = e(b)[1,`c']

            local jj = 0
            foreach c2 of local keep_cols {
                local ++jj
                matrix `V1'[`ii',`jj'] = e(V)[`c',`c2']
            }
        }

        matrix `Vinv' = syminv(`V1')
        local wald_df = rowsof(`Vinv') - diag0cnt(`Vinv')
        scalar `Fstat' = .
        if (`wald_df' > 0) {
            matrix `W' = `b1' * `Vinv' * `b1''
            scalar `Fstat' = `W'[1,1] / `wald_df'
        }
        if (`wald_df' > 0 & !missing(`Fstat')) {
            ereturn scalar F = `Fstat'
            ereturn scalar p = Ftail(`wald_df', e(df_r), `Fstat')
        }
    }

    ereturn scalar mss = .
    if (!missing(e(rss)) & !missing(e(tss))) {
        ereturn scalar mss = e(tss) - e(rss)
    }
    ereturn scalar ll = .
    ereturn scalar ll_0 = .
    if (!missing(e(rss)) & e(rss) > 0 & !missing(e(tss_within)) & e(tss_within) > 0) {
        ereturn scalar ll   = -0.5 * (e(N)*ln(2*_pi) + e(N)*ln(e(rss)/e(N)) + e(N))
        ereturn scalar ll_0 = -0.5 * (e(N)*ln(2*_pi) + e(N)*ln(e(tss_within)/e(N)) + e(N))
    }

    local used_df_r = e(df_r_unadj)
    if ("`statstyle'" == "reghdfe") {
        local df_a_nested = e(df_a_nested)
        if (missing(`df_a_nested')) local df_a_nested 0
        local used_df_r = max(`used_df_r' - `df_a_nested', 0)
    }
    if (`used_df_r' > 0 & !missing(e(rss)) & e(rss) >= 0) {
        ereturn scalar rmse = sqrt( e(rss) / `used_df_r' )
    }
    else if ("`statstyle'" == "reghdfe" & `used_df_r' == 0 & !missing(e(rss)) & e(rss) >= 0) {
        ereturn scalar rmse = sqrt( e(rss) )
    }
    if (`used_df_r' > 0 & !missing(e(tss)) & e(tss) > 0) {
        ereturn scalar r2_a = 1 - (e(rss)/`used_df_r') / ( e(tss) / (e(N)-`model_has_cons') )
        if (!missing(e(tss_within)) & e(tss_within) > 0) {
            ereturn scalar r2_a_within = 1 - (e(rss)/`used_df_r') / ( e(tss_within) / (`used_df_r'+e(df_m)) )
        }
    }
    else if (`used_df_r' <= 0 & !missing(e(rss)) & !missing(e(tss)) & e(tss) > 0 & ///
             e(rss) <= 1e-8 * e(tss)) {
        // Saturated model that fits within machine precision: report Adj R-squared = 1
        // to match reghdfe's display (mathematically, rss/df_r tends to 0 as rss -> 0).
        ereturn scalar r2_a = 1
        if (!missing(e(tss_within)) & e(tss_within) > 0) {
            ereturn scalar r2_a_within = 1
        }
    }

    if ("`compat_ignored'" != "" & "`nowarn'" == "") {
        local compat_ignored : list retokenize compat_ignored
        di as txt "(note: ignored reghdfe-compat options: `compat_ignored')"
    }

    if (`__xhdfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=post_plugin ms=" %12.3f (1000 * r(t99))
        timer clear 99
        timer on 99
    }

    local display_opts
    foreach opt in noheader notable nofootnote noomitted noempty noci nopvalues ///
        baselevels allbaselevels vsquish nolstretch nofvlabel {
        if ("``opt''" != "") local display_opts "`display_opts' `opt'"
    }
    if ("`cformat'" != "") local display_opts `"`display_opts' cformat(`cformat')"' 
    if ("`pformat'" != "") local display_opts `"`display_opts' pformat(`pformat')"' 
    if ("`sformat'" != "") local display_opts `"`display_opts' sformat(`sformat')"' 
    if (`fvwrap' > 0) local display_opts "`display_opts' fvwrap(`fvwrap')"
    // ereturn display in supported Stata versions accepts fvwrap() but not
    // fvwrapon(); parse fvwrapon() for compatibility and leave display stable.

    xhdfe_display, level(`level') `display_opts'
    if (`__xhdfe_profile') {
        timer off 99
        quietly timer list 99
        di as txt "ado_profile label=display ms=" %12.3f (1000 * r(t99))
        timer clear 99
    }
end

program define xhdfe__footnote
    version 16.0
    if (!missing(e(drop_singletons)) & e(drop_singletons) == 0 & !missing(e(num_singletons)) & e(num_singletons) > 0 & "`e(nowarn)'" == "") {
        di as err "WARNING: Singleton observations not dropped: statistical significance is biased (http://scorreia.com/reghdfe/nested_within_cluster.pdf)"
    }
    capture confirm matrix e(dof_table)
    if (!_rc) {
        xhdfe_footnote
    }
    exit 0
end

program define xhdfe_display
    version 16.0
    local optline `"`0'"'
    local optline : subinstr local optline ",fvwrapon(" ", fvwon(", all
    local optline : subinstr local optline ", fvwrapon(" ", fvwon(", all
    local optline : subinstr local optline " fvwrapon(" " fvwon(", all
    local 0 `"`optline'"'
    syntax [, Level(cilevel) NOHeader NOTABle NOFOOTnote NOOMITTED NOEMPty ///
        NOCI NOPVALUES BASElevels ALLBASElevels CFORMAT(string asis) ///
        PFORMAT(string asis) SFORMAT(string asis) FVWrap(integer 0) ///
        FVWON(string asis) VSQUISH NOLSTRetch NOFVLABEL]

    local printed_preamble 0
    if (!missing(e(drop_singletons)) & e(drop_singletons) == 1 & !missing(e(num_singletons)) & e(num_singletons) > 0) {
        local num_singletons : display %20.0fc e(num_singletons)
        local num_singletons = strtrim("`num_singletons'")
        di as txt "(dropped " as res "`num_singletons'" as txt " singleton observations)"
        local printed_preamble 1
    }
    if (!missing(e(iterations))) {
        local iterations : display %20.0fc e(iterations)
        local iterations = strtrim("`iterations'")
        if (!missing(e(converged)) & e(converged) == 1) {
            di as txt "(MWFE estimator converged in " as res "`iterations'" as txt " iterations)"
            local printed_preamble 1
        }
        else if (!missing(e(converged)) & e(converged) == 0) {
            di as txt "(MWFE estimator failed to converge in " as res "`iterations'" as txt " iterations)"
            local printed_preamble 1
        }
    }
    if (!missing(e(vcv_psd_fixed)) & e(vcv_psd_fixed) == 1) {
        di as txt "Warning: VCV matrix was non-positive semi-definite; adjustment from Cameron, Gelbach & Miller applied."
        local printed_preamble 1
    }
    // Saturated / perfect-fit design: when the absorbed FEs plus the regressors
    // span the data the within residuals are ~0 (within R-sq = 1), so the point
    // estimates are exact but the error variance — and hence the standard errors
    // — is not identified. Reporting it explicitly avoids misreading the ~0 SEs
    // as infinite precision (e.g. synthetic-zigzag: 10002 obs, 10000 absorbed FE
    // ranks + 2 slopes => 0 residual df). Informational only; never changes the
    // estimates or VCV.
    if (!missing(e(saturated)) & e(saturated) == 1) {
        di as txt "Note: perfect fit (within R-squared = 1); the design is saturated, so the reported standard errors are not identified."
        local printed_preamble 1
    }

    capture confirm matrix e(omit_reason)
    if (!_rc) {
        tempname omit_reason
        matrix `omit_reason' = e(omit_reason)
        local cn : colnames e(b)
        local k = colsof(e(b))
        forvalues j = 1/`k' {
            local name : word `j' of `cn'
            if ("`name'" == "_cons") continue
            if (substr("`name'", 1, 2) == "o.") {
                local name = substr("`name'", 3, .)
            }
            local reason = `omit_reason'[`j',1]
            if (`reason' == 1) {
                di as txt "note: `name' is probably collinear with the fixed effects (all partialled-out values are close to zero; tol = 1.0e-09)"
                local printed_preamble 1
            }
            else if (`reason' == 2) {
                di as txt "note: `name' omitted because of collinearity"
                local printed_preamble 1
            }
        }
    }

    local title = cond("`e(title)'" != "", "`e(title)'", "HDFE Linear regression")

    local N = e(N)
    local df_m = e(df_m)
    local df_r = e(df_r)
    local Fstat = e(F)
    local pF = e(p)

    local nfe 0
    if ("`e(absorb)'" != "") {
        local nfe : word count `e(absorb)'
    }

    local vce_desc ""
    if (inlist("`e(vce)'", "cluster", "robust")) {
        local vce_desc "Statistics robust to heteroskedasticity"
    }
    else {
        local vce_desc "Statistics based on homoskedasticity"
    }

    if ("`noheader'" == "") {
        if (`printed_preamble') di as txt ""
        di as txt "`title'" _col(50) "Number of obs" _col(67) "= " as res %10.0fc `N'

        // Show the F header whenever we have df_m and df_r available (even if df_r <= 0,
        // as in saturated models): in that case the F value itself is missing, matching
        // reghdfe's "F( df_m, df_r) = ." display.
        local have_F_header = (!missing(`df_m') & `df_m' > 0 & !missing(`df_r'))
        local show_F_value = (`have_F_header' & `df_r' > 0 & !missing(`Fstat'))

        if (`nfe' > 0 & `have_F_header') {
            if (`show_F_value') {
                di as txt "Absorbing `nfe' HDFE group(s)" _col(50) "F(" as res %4.0f `df_m' as txt "," as res %8.0f `df_r' as txt ")" ///
                    _col(67) "= " as res %10.2f `Fstat'
            }
            else {
                di as txt "Absorbing `nfe' HDFE group(s)" _col(50) "F(" as res %4.0f `df_m' as txt "," as res %8.0f `df_r' as txt ")" ///
                    _col(67) "= " as res %10.2f .
            }
        }
        else if (`nfe' > 0) {
            di as txt "Absorbing `nfe' HDFE group(s)"
        }
        else if (`have_F_header') {
            if (`show_F_value') {
                di as txt _col(50) "F(" as res %4.0f `df_m' as txt "," as res %8.0f `df_r' as txt ")" _col(67) "= " as res %10.2f `Fstat'
            }
            else {
                di as txt _col(50) "F(" as res %4.0f `df_m' as txt "," as res %8.0f `df_r' as txt ")" _col(67) "= " as res %10.2f .
            }
        }

        if ("`vce_desc'" != "") {
            if (`have_F_header') {
                if (`show_F_value') {
                    di as txt "`vce_desc'" _col(50) "Prob > F" _col(67) "= " as res %10.4f `pF'
                }
                else {
                    di as txt "`vce_desc'" _col(50) "Prob > F" _col(67) "= " as res %10.4f .
                }
            }
            else {
                di as txt "`vce_desc'"
            }
        }

        if (!missing(e(r2))) {
            di as txt _col(50) "R-squared" _col(67) "= " as res %10.4f e(r2)
        }
        if (!missing(e(r2_a))) {
            di as txt _col(50) "Adj R-squared" _col(67) "= " as res %10.4f e(r2_a)
        }
        if (!missing(e(r2_within))) {
            di as txt _col(50) "Within R-sq." _col(67) "= " as res %10.4f e(r2_within)
        }
        if (!missing(e(rmse))) {
            di as txt _col(50) "Root MSE" _col(67) "= " as res %10.4f e(rmse)
        }
    }

    local display_opts
    foreach opt in notable noomitted noempty noci nopvalues baselevels allbaselevels ///
        vsquish nolstretch nofvlabel {
        if ("``opt''" != "") local display_opts "`display_opts' `opt'"
    }
    if ("`cformat'" != "") local display_opts `"`display_opts' cformat(`cformat')"' 
    if ("`pformat'" != "") local display_opts `"`display_opts' pformat(`pformat')"' 
    if ("`sformat'" != "") local display_opts `"`display_opts' sformat(`sformat')"' 
    if (`fvwrap' > 0) local display_opts "`display_opts' fvwrap(`fvwrap')"
    // ereturn display in supported Stata versions rejects fvwrapon().

    if ("`notable'" == "") {
        if ("`noheader'" == "") di as txt ""
        ereturn display, level(`level') `display_opts'
    }

    if ("`nofootnote'" == "") {
        if ("`e(footnote)'" != "") {
            capture confirm program `e(footnote)'
            if (!_rc) {
                `e(footnote)'
            }
            else {
                capture confirm matrix e(dof_table)
                if (!_rc) {
                    xhdfe_footnote
                }
            }
        }
        else {
            capture confirm matrix e(dof_table)
            if (!_rc) {
                xhdfe_footnote
            }
            else if (!missing(e(df_a))) {
                di as txt "Absorbed degrees of freedom:" _col(30) %10.0f e(df_a)
            }
        }
    }
end

program define xhdfe_footnote
    syntax [, width(int 13)]

    if (`"`e(absvars)'"' == "_cons") {
        exit
    }

    tempname table
    matrix `table' = e(dof_table)
    local vars "`e(dof_labels)'"
    local have_dof_labels = ("`vars'" != "")
    if ("`vars'" == "") {
        mata: st_local("vars", invtokens(st_matrixrowstripe("`table'")[., 2]'))
    }
    local var_width 0
    foreach v of local vars {
        local v_clean "`v'"
        if (!`have_dof_labels') {
            local v_clean = subinstr("`v_clean'", "1.", "", .)
            local v_clean = subinstr("`v_clean'", "#c.", "#", .)
        }
        local var_width = max(`var_width', strlen("`v_clean'"))
    }
    if (`var_width' + 2 > `width') local width = `var_width' + 2
    local rows = rowsof("`table'")

    di as text _n "Absorbed degrees of freedom:"
    tempname mytab
    .`mytab' = ._tab.new, col(5) lmargin(0)
    .`mytab'.width `width'  | 12   12    14    1
    .`mytab'.pad   .        1     1     1     0
    .`mytab'.numfmt .       %9.0g %9.0g %9.0g .
    .`mytab'.numcolor .     text  text  result .
    .`mytab'.sep, top

    local explain_exact 0
    local explain_nested 0

    .`mytab'.titles "Absorbed FE" "Categories" " - Redundant" "  = Num. Coefs" ""
    .`mytab'.sep, middle

    forval i = 1/`rows' {
        local var : word `i' of `vars'
        if (!`have_dof_labels') {
            local var = subinstr("`var'", "1.", "", .)
            local var = subinstr("`var'", "#c.", "#", .)
        }
        local note " "
        if (`=`table'[`i', 4]' == 1) {
            local note "?"
            local explain_exact 1
        }
        if (`=`table'[`i', 5]' == 1) {
            local note "*"
            local explain_nested 1
        }

        .`mytab'.row "`var'" `=`table'[`i', 1]' `=`table'[`i', 2]' `=`table'[`i', 3]' "`note'"
    }

    .`mytab'.sep, bottom
    if (`explain_exact') di as text "? = number of redundant parameters may be higher"
    if (`explain_nested') di as text `"* = FE nested within cluster; treated as redundant for DoF computation"'
end

capture mata drop _xhdfe_fix_v()
mata:
void _xhdfe_fix_v(string scalar Vname)
{
    real matrix V
    string matrix rowstripe
    string matrix colstripe
    real scalar r, c, i, j, found
    real scalar eps

    V = st_matrix(Vname)
    rowstripe = st_matrixrowstripe(Vname)
    colstripe = st_matrixcolstripe(Vname)
    r = rows(V)
    c = cols(V)
    found = 0
    eps = 1e-12
    for (i = 1; i <= r; i++) {
        for (j = 1; j <= c; j++) {
            if (!(V[i, j] < .)) {
                V[i, j] = 0
                found = 1
            }
        }
    }
    if (r == c & r > 0) {
        V = 0.5 * (V + V')
        for (i = 1; i <= r; i++) {
            if (V[i, i] < 0 & V[i, i] > -eps) {
                V[i, i] = 0
            }
        }
        found = 1
    }
    if (found) {
        st_matrix(Vname, V)
        st_matrixrowstripe(Vname, rowstripe)
        st_matrixcolstripe(Vname, colstripe)
    }
}
end

capture mata drop _xhdfe_fv_cleanname()
capture mata drop _xhdfe_fv_normname()
capture mata drop _xhdfe_expand_fv_base()
mata:
// Drop "b"/"n" factor markers from a term name but keep omit markers, so
// matrix-colnames artifacts like "2bn.cat" -> "2.cat" and "2bno.race" ->
// "2o.race" once the true base column is re-inserted (reghdfe stripe).
string scalar _xhdfe_fv_cleanname(string scalar s)
{
    string rowvector parts
    string scalar out, p, pre, post, digits, markers
    real scalar i, dot, j

    parts = tokens(subinstr(s, "#", " # "))
    out = ""
    for (i = 1; i <= cols(parts); i++) {
        p = parts[i]
        if (p != "#") {
            dot = strpos(p, ".")
            if (dot > 1) {
                pre = substr(p, 1, dot - 1)
                post = substr(p, dot, .)
                j = 1
                while (j <= strlen(pre) & strpos("0123456789", substr(pre, j, 1)) > 0) {
                    j++
                }
                if (j > 1) {
                    digits = substr(pre, 1, j - 1)
                    markers = substr(pre, j, .)
                    p = digits + (strpos(markers, "o") ? "o" : "") + post
                }
            }
        }
        out = out + p
    }
    return(out)
}
// Strip factor-variable base/omit markers from a term name so that e.g.
// "4o.cat", "1b.cat", and "4.cat" all normalize to "4.cat" (interaction
// parts are normalized independently).
string scalar _xhdfe_fv_normname(string scalar s)
{
    string rowvector parts
    string scalar out, p, pre, post
    real scalar i, dot, j

    parts = tokens(subinstr(s, "#", " # "))
    out = ""
    for (i = 1; i <= cols(parts); i++) {
        p = parts[i]
        if (p != "#") {
            if (substr(p, 1, 2) == "o.") {
                p = substr(p, 3, .)
            }
            dot = strpos(p, ".")
            if (dot > 1) {
                pre = substr(p, 1, dot - 1)
                post = substr(p, dot, .)
                j = 1
                while (j <= strlen(pre) & strpos("0123456789", substr(pre, j, 1)) > 0) {
                    j++
                }
                if (j > 1) {
                    p = substr(pre, 1, j - 1) + post
                }
            }
        }
        out = out + p
    }
    return(out)
}

// Re-insert base/omitted factor-variable levels into b/V/omit_reason as
// structural-zero columns with canonical names (reghdfe/Stata stripe parity).
// canon holds the fvexpand expansion of the regressor list on the estimation
// sample; existing columns are matched by marker-normalized name and keep
// their (possibly o.-marked) names and values.
void _xhdfe_expand_fv_base(string scalar bname, string scalar Vname,
                           string scalar omitname, string scalar canon)
{
    real matrix b, V, om, bn, Vn, omn
    string matrix stripe
    string colvector names, norm_names, target
    string rowvector want
    string scalar t, nt, nm
    real colvector src, used
    real scalar i, j, kx, kn, pos, have_cons, same

    b = st_matrix(bname)
    V = st_matrix(Vname)
    om = st_matrix(omitname)
    stripe = st_matrixcolstripe(bname)
    names = stripe[., 2]
    kx = cols(b)
    if (kx <= 0) {
        return
    }
    have_cons = (names[kx] == "_cons")

    norm_names = J(kx, 1, "")
    for (i = 1; i <= kx; i++) {
        norm_names[i] = _xhdfe_fv_normname(names[i])
    }

    want = tokens(canon)
    target = J(0, 1, "")
    src = J(0, 1, .)
    used = J(kx, 1, 0)
    for (i = 1; i <= cols(want); i++) {
        t = want[i]
        nt = _xhdfe_fv_normname(t)
        pos = 0
        for (j = 1; j <= kx; j++) {
            if (!used[j] & norm_names[j] == nt) {
                pos = j
                break
            }
        }
        if (pos > 0) {
            used[pos] = 1
            // matrix colnames canonicalizes a base-less factor set with "bn"
            // markers (2.cat -> 2bn.cat); strip them now that the true base
            // column is being re-inserted. Omit markers survive (2bno.cat ->
            // 2o.cat), matching the reghdfe stripe.
            target = target \ _xhdfe_fv_cleanname(names[pos])
            src = src \ pos
        }
        else {
            // Structural zero column (base level or user-omitted term). Mark
            // continuous parts of base interactions as omitted ("#co.") the
            // way Stata/reghdfe stripe them.
            nm = t
            if (strpos(nm, "b.") > 0 & strpos(nm, "#") > 0) {
                nm = subinstr(nm, "#c.", "#co.")
            }
            target = target \ nm
            src = src \ 0
        }
    }
    // Keep any leftover estimated columns (defensive; preserves order).
    for (j = 1; j <= kx; j++) {
        if (!used[j] & names[j] != "_cons") {
            target = target \ _xhdfe_fv_cleanname(names[j])
            src = src \ j
            used[j] = 1
        }
    }
    if (have_cons) {
        target = target \ "_cons"
        src = src \ kx
    }

    kn = rows(target)
    if (kn == kx) {
        // Nothing to insert; keep the plugin stripe untouched.
        same = 1
        for (i = 1; i <= kn; i++) {
            if (src[i] != i) {
                same = 0
                break
            }
        }
        if (same) {
            return
        }
    }

    bn = J(1, kn, 0)
    Vn = J(kn, kn, 0)
    omn = J(kn, 1, 0)
    for (i = 1; i <= kn; i++) {
        if (src[i] > 0) {
            bn[1, i] = b[1, src[i]]
            omn[i, 1] = om[src[i], 1]
            for (j = 1; j <= kn; j++) {
                if (src[j] > 0) {
                    Vn[i, j] = V[src[i], src[j]]
                }
            }
        }
    }

    st_matrix(bname, bn)
    st_matrixcolstripe(bname, (J(kn, 1, ""), target))
    st_matrix(Vname, Vn)
    st_matrixcolstripe(Vname, (J(kn, 1, ""), target))
    st_matrixrowstripe(Vname, (J(kn, 1, ""), target))
    st_matrix(omitname, omn)
    st_matrixrowstripe(omitname, (J(kn, 1, ""), target))
    st_matrixcolstripe(omitname, (J(1, 1, ""), J(1, 1, "reason")))
}
end
