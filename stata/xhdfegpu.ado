*! version 2.35.1  25nov2026
*! Build and install the CUDA plugins, or report the build metadata.
program define xhdfegpu, rclass
    version 14.0
    syntax [, STATUS CPUTARGET(string) ZIP(string) SOURCE(string) ARCH(integer 0) ///
              KEEP FORCE URL(string)]

    local cputarget = lower(strtrim(`"`cputarget'"'))
    if (!inlist(`"`cputarget'"', "", "portable", "native")) {
        di as err "cputarget() must be portable or native"
        exit 198
    }
    if ("`status'" != "") {
        if (`arch' != 0 | "`keep'`force'`cputarget'" != "" | ///
            `"`zip'`source'`url'"' != "") {
            di as err "status must be specified on its own"
            exit 198
        }
        foreach name in xhdfe xfepout {
            _xhdfegpu_target `name'
            local path `"`r(plugin)'"'
            if ("`name'" == "xhdfe") local bound `"$XHDFE_PLUGIN_PATH_INTERNAL"'
            else local bound `"$XFEPOUT_PLUGIN_PATH_INTERNAL"'
            _xhdfegpu_status `name' `"`path'"' `"`bound'"'
            return local `name'_plugin `"`path'"'
            foreach key in bound_plugin build_source cputarget sms compiler build_version {
                return local `name'_`key' `"`r(`key')'"'
            }
            return scalar `name'_cuda = r(cuda)
            return scalar `name'_openmp = r(openmp)
        }
        exit
    }

    if (`arch' < 0 | (`arch' > 0 & `arch' < 75)) {
        di as err "arch() must be 75 or higher, or omitted for automatic detection"
        exit 198
    }
    if ("`c(os)'" != "Unix") {
        di as err "xhdfegpu installs CUDA plugins on Linux only"
        exit 198
    }

    _xhdfegpu_target xhdfe
    local xado `"`r(ado)'"'
    local xplugin `"`r(plugin)'"'
    if (`"`xplugin'"' == "" | !fileexists(`"`xplugin'"')) {
        di as err "xhdfe.plugin not found alongside the active xhdfe.ado"
        exit 601
    }
    _xhdfegpu_target xfepout
    local fado `"`r(ado)'"'
    local fplugin `"`r(plugin)'"'
    if (`"`fplugin'"' != "" & !fileexists(`"`fplugin'"')) {
        di as err "xfepout.plugin not found alongside the active xfepout.ado"
        exit 601
    }
    _xhdfegpu_target xhdfegpu
    local installer_ado `"`r(ado)'"'
    local installer_dir = substr(`"`installer_ado'"', 1, strlen(`"`installer_ado'"') - 12)
    local helper `"`installer_dir'xhdfegpu-install.sh"'
    local smoke `"`installer_dir'xhdfegpu-smoke.do"'
    if (!fileexists(`"`helper'"') | !fileexists(`"`smoke'"')) {
        di as err "xhdfegpu installation helpers are missing alongside xhdfegpu.ado"
        exit 601
    }

    if (`"`url'"' == "") {
        local url "https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata/xhdfe-src.zip"
    }
    local stata "`c(sysdir_stata)'stata"
    if (c(MP)) local stata "`stata'-mp"
    else if (c(SE)) local stata "`stata'-se"
    local tmpdir "`c(tmpdir)'"
    tempfile request receipt
    tempname spec result
    foreach field in source zip url xado xplugin fado fplugin smoke stata tmpdir {
        if (strpos(`"``field''"', char(10)) | strpos(`"``field''"', char(13))) {
            di as err "a path or URL contains a line break"
            exit 198
        }
    }
    file open `spec' using "`request'", write text
    foreach field in cputarget arch source zip url xado xplugin fado fplugin smoke stata tmpdir keep force {
        file write `spec' `"``field''"' _n
    }
    file close `spec'

    * The shell receives only three filenames. The request fields are literal data.
    foreach field in helper request receipt {
        local q_`field' = char(39) + subinstr(`"``field''"', char(39), ///
            char(39) + char(34) + char(39) + char(34) + char(39), .) + char(39)
    }
    capture noisily shell bash `q_helper' `q_request' `q_receipt'
    capture file open `result' using "`receipt'", read text
    if (_rc) {
        di as err "xhdfegpu: installer did not return a result"
        exit 198
    }
    file read `result' code
    if ("`code'" != "0") {
        file close `result'
        di as err "xhdfegpu: installation did not complete"
        if ("`code'" == "601") exit 601
        if ("`code'" == "603") exit 603
        exit 198
    }
    foreach field in target sm installed_xhdfe installed_xfepout backup workdir {
        file read `result' `field'
    }
    file close `result'

    di as txt "xhdfe.plugin (CPU `target', CUDA sm_`sm'): " as res `"`installed_xhdfe'"'
    if (`"`installed_xfepout'"' != "") {
        di as txt "xfepout.plugin (CPU `target', CUDA sm_`sm'): " as res `"`installed_xfepout'"'
    }
    if (`"`backup'"' != "") di as txt "Recovery journal: " as res `"`backup'"'
    if (`"`workdir'"' != "") di as txt "Build files: " as res `"`workdir'"'
    di as txt "Open a new Stata session to load the installed plugin files."
    return local plugin `"`installed_xhdfe'"'
    return local xfepout_plugin `"`installed_xfepout'"'
    return local arch "sm_`sm'"
    return local cputarget "`target'"
    return local backup `"`backup'"'
    return local workdir `"`workdir'"'
    return scalar gpu_ready = 1
end

program define _xhdfegpu_target, rclass
    args name
    capture quietly findfile `name'.ado
    if (_rc) {
        return local ado ""
        return local plugin ""
        exit
    }
    local path `"`r(fn)'"'
    if (substr(`"`path'"', 1, 1) == "~") {
        local homedir : env HOME
        local path = `"`homedir'"' + substr(`"`path'"', 2, .)
    }
    if (!inlist(substr(`"`path'"', 1, 1), "/", char(92)) & substr(`"`path'"', 2, 1) != ":") {
        local path = c(pwd) + "/" + `"`path'"'
    }
    return local ado `"`path'"'
    local plugin = substr(`"`path'"', 1, strlen(`"`path'"') - 4) + ".plugin"
    return local plugin `"`plugin'"'
end

program define _xhdfegpu_status, rclass
    args name path bound
    local dispatch "__`name'_plugin_dispatch"
    foreach key in protocol role cpu cuda sms openmp compiler version {
        local xhdfe_buildinfo_`key' ""
    }
    local build_source "missing"
    local call_rc = 199
    if (`"`bound'"' != "") {
        * Plug-in subprograms are callable only from their owning ado.
        capture quietly program list `name'._`name'_plugin_buildinfo
        if (!_rc) {
            capture quietly `name', _xhdfegpu_buildinfo
            local call_rc = _rc
            if (`call_rc' == 199) local bound ""
        }
        else {
            * A top-level dispatcher is directly callable. A legacy active
            * ado has no bridge, so leave its build metadata unrecorded.
            capture quietly plugin call `dispatch', "task=buildinfo"
            local call_rc = _rc
            if (`call_rc' == 199) {
                capture quietly program list `name'
                if (_rc) local bound ""
            }
        }
        if (`"`bound'"' == "") {
            foreach key in protocol role cpu cuda sms openmp compiler version {
                local xhdfe_buildinfo_`key' ""
            }
        }
        else local build_source "memory"
    }
    if (`"`bound'"' == "" & `"`path'"' != "" & fileexists(`"`path'"')) {
        local build_source "by_path"
        * Stata keeps plugin subprograms registered in the enclosing ado.
        * Use one probe per role and record the path for subsequent queries.
        local probe "__xhdfegpu_probe_`name'"
        local probe_path "XHDFEGPU_PROBE_`name'"
        capture program `probe', plugin using(`"`path'"')
        local registered = _rc
        if (`registered' == 0) global `probe_path' `"`path'"'
        if (`registered' == 110) {
            local bound `"${`probe_path'}"'
            local build_source = cond(`"`bound'"' == "", "memory_unrecorded", "memory")
        }
        if (`registered' == 0 | "`build_source'" == "memory") {
            capture quietly plugin call `probe', "task=buildinfo"
            local call_rc = _rc
        }
    }
    local declared = (`call_rc' == 0 & "`xhdfe_buildinfo_protocol'" == "1" ///
        & "`xhdfe_buildinfo_role'" == "`name'")
    foreach key in cpu sms compiler version {
        if (!`declared' | `"`xhdfe_buildinfo_`key''"' == "") local xhdfe_buildinfo_`key' "unrecorded"
    }
    foreach key in cuda openmp {
        if (!`declared' | !inlist("`xhdfe_buildinfo_`key''", "0", "1")) local xhdfe_buildinfo_`key' "."
    }

    if (`"`path'"' != "") di as txt "`name' on disk: " as res `"`path'"'
    else di as txt "`name': no active ado found"
    if ("`build_source'" == "missing" & `"`path'"' != "") di as txt "  plugin file missing"
    if ("`build_source'" == "memory") {
        di as txt "  metadata from the loaded plugin: " as res `"`bound'"'
        if (!inlist(substr(`"`bound'"', 1, 1), "/", char(92)) & substr(`"`bound'"', 2, 1) != ":") {
            di as txt "  The binding was recorded relative to the directory used at load time."
        }
        di as txt "  Restart Stata to read the plugin currently on disk."
    }
    if ("`build_source'" == "by_path") di as txt "  metadata queried by path"
    if ("`build_source'" == "memory_unrecorded") {
        di as txt "  loaded probe with unrecorded provenance. Restart Stata to read the file on disk."
    }
    if ("`build_source'" != "missing") {
        di as txt "  CPU: " as res "`xhdfe_buildinfo_cpu'" as txt "   CUDA: " as res "`xhdfe_buildinfo_cuda'" ///
            as txt "   SM: " as res "`xhdfe_buildinfo_sms'" as txt "   OpenMP: " as res "`xhdfe_buildinfo_openmp'"
    }
    if ("`name'" == "xhdfe") di as txt "  To build for this CPU: " as res "xhdfegpu, cputarget(native)"
    return local bound_plugin `"`bound'"'
    return local build_source "`build_source'"
    return local cputarget "`xhdfe_buildinfo_cpu'"
    return local sms "`xhdfe_buildinfo_sms'"
    return local compiler `"`xhdfe_buildinfo_compiler'"'
    return local build_version "`xhdfe_buildinfo_version'"
    return scalar cuda = `xhdfe_buildinfo_cuda'
    return scalar openmp = `xhdfe_buildinfo_openmp'
end
