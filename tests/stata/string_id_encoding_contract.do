* String identifiers coded in a narrow frame (_xhdfe_string_group) must give
* exactly what egen group() gives, and the estimates of the ado that ran egen
* at those sites.
*
* Five fresh Stata processes, all launched from one scratch directory, which
* receives the files (synthetic data only, numthreads(1), CPU):
*
*   XHDFE_STATA_ADOPATH=<package>/stata stata-mp -b do string_id_encoding_contract.do codes
*   XHDFE_STATA_ADOPATH=<package>/stata XHDFE_PROFILE_CPU=1 stata-mp -b do string_id_encoding_contract.do profile
*   XHDFE_STATA_ADOPATH=<package>/stata stata-mp -b do string_id_encoding_contract.do battery patched
*   XHDFE_STATA_ADOPATH=<package>/stata stata-mp -b do string_id_encoding_contract.do battery native
*   stata-mp -b do string_id_encoding_contract.do compare
*
* XHDFE_PROFILE_CPU must be unset in every process but profile. The native arm
* is the same package and plugin, loaded the same way (run), with
* _xhdfe_string_group replaced, in this test only, by a program that runs
* "egen long target = group(varlist) if touse", which is what the ado did at
* those sites before the helper. Any other arm name loads the package of
* XHDFE_STATA_ADOPATH unchanged (for instance a frozen release, by hand), and
* "compare <left> <right>" compares any two arms (default: patched native).
*
* The helper falls back to egen group() when it fails; every mode sets
* $XHDFE_STRING_GROUP_STRICT to 1 so that a broken helper stops the test instead
* of hiding behind that fallback, and clears it only where the fallback itself
* is tested.
*
* codes:   runs the package's ado and compares the helper's codes, type, label
*          and notes with egen group() on every list, sample and order the
*          battery uses (and more: extended missing values, abbreviations on
*          and off, caller versions 16 to 19, set type double), from the
*          default and from another frame; shows that the comparison detects a
*          relabelling; at the frame limit (frame put fails for real) and with a
*          failure injected after each statement of the body, and in a Mata
*          function that stores half of the codes and then fails, without
*          strict mode the call falls back to egen's codes and estimates, in
*          strict mode it stops, Break (rc 1) always stops, and nothing is left
*          behind; prints what the helper and egen leave in r(); shows which
*          calls reach the helper.
* profile: under XHDFE_PROFILE_CPU, the fallback prints its one profile line.
* battery: the same fits in each arm, with the return code each must give and,
*          after a successful fit, e(converged) 1 and e(gpu_used) 0; written in
*          %21x: e(b), the whole e(V), N, df_r, rss, rank, df_a, df_m, the
*          sample, residuals and saved fixed effects, and after every call,
*          successful or not, c(rc), what the call leaves in r() and s(), the
*          EGEN_* globals and the characteristics of the data and of every
*          variable; order, sortedby, frames and Mata globals are checked after
*          every call, and the Mata functions the session gains must all be the
*          ado's own (_xhdfe_*). The native arm counts the calls of its program
*          against the number each case must make, and its Mata scatter stops
*          any call that reaches the frame-based body. One fit is logged as
*          printed.
* compare: the two battery files must be identical byte for byte, and so must
*          the printed fits, line by line with nothing skipped, and the
*          random-number states; the sort-tiebreak states are reported, not
*          required to match.
* Every log the test reads is opened with nomsg: without the header and footer
* of log using, no line depends on the path, the date or the time. What the
* test deliberately does not run is printed as STRING_ID_NOT_RUN lines in
* every log.
version 16
clear all
set more off
set varabbrev off
args mode arm arm2
local adopath : env XHDFE_STATA_ADOPATH
* Without a trailing slash, so that adopath/xhdfe.ado is what findfile returns.
while (strlen(`"`adopath'"') > 1 & substr(`"`adopath'"', -1, 1) == "/") {
    local adopath = substr(`"`adopath'"', 1, strlen(`"`adopath'"') - 1)
}
global XHDFE_STRING_GROUP_STRICT 1

* One line of the log, unwrapped.
capture program drop s4_say
program define s4_say
    local ls = c(linesize)
    set linesize 255
    display `"`0'"'
    set linesize `ls'
end

s4_say STRING_ID_NOT_RUN long_double_switch: more than 2,147,483,620 observations or groups, where the codes become double, needs data of that size
s4_say STRING_ID_NOT_RUN stata_16_to_18: only Stata 19.5 is installed here; the caller's version 16 to 19 blocks are run
s4_say STRING_ID_NOT_RUN windows_macos: the test runs on Linux only
s4_say STRING_ID_NOT_RUN break_during_fallback: a real Break needs a signal from outside Stata; one-off evidence in the S4 report
s4_say STRING_ID_NOT_RUN memory_failure: a bounded attempt with set max_memory produced no allocation failure inside the helper; frame put fails for real at the frame limit, the other stages by injection

mata:
// Bytes of a string in hexadecimal, so that any text is written safely.
string scalar s4_hexstr(string scalar s)
{
    if (strlen(s) == 0) return("-")
    return(invtokens(inbase(16, ascii(s)), "."))
}

// A Stata matrix (r() included): size, row and column names, elements in %21x.
string scalar s4_hexmat(string scalar name)
{
    real matrix M
    string matrix R, C
    string scalar out
    real scalar i, j
    M = st_matrix(name)
    R = st_matrixrowstripe(name)
    C = st_matrixcolstripe(name)
    out = strofreal(rows(M)) + "x" + strofreal(cols(M))
    if (rows(R)) out = out + " rows=" + s4_hexstr(invtokens((R[., 1] :+ ":" :+ R[., 2])'))
    if (rows(C)) out = out + " cols=" + s4_hexstr(invtokens((C[., 1] :+ ":" :+ C[., 2])'))
    for (i = 1; i <= rows(M); i++) {
        for (j = 1; j <= cols(M); j++) out = out + " " + strofreal(M[i, j], "%21x")
    }
    return(out)
}

// The Mata functions listed by a log of "mata describe", sorted.
string scalar s4_function_names(string scalar file)
{
    string colvector L, N
    string rowvector t
    string scalar last
    real scalar i
    L = cat(file)
    N = J(0, 1, "")
    for (i = 1; i <= rows(L); i++) {
        t = tokens(L[i])
        if (cols(t) == 0) continue
        last = t[cols(t)]
        if (strlen(last) > 2) {
            if (substr(last, strlen(last) - 1, 2) == "()") N = N \ substr(last, 1, strlen(last) - 2)
        }
    }
    if (rows(N) == 0) return("")
    return(invtokens(sort(N, 1)'))
}

// The statement of the helper after which a failure is injected.
real scalar s4_stage_line(string scalar t, string scalar stage)
{
    if (stage == "mark") return(substr(t, 1, 5) == "mark ")
    if (stage == "markout") return(substr(t, 1, 8) == "markout ")
    if (stage == "count") return(substr(t, 1, 14) == "quietly count ")
    if (stage == "frameput") return(substr(t, 1, 10) == "frame put ")
    if (stage == "genidx") return(substr(t, 1, 17) == "quietly generate " & strpos(t, "= _n") > 0)
    if (stage == "sort") return(substr(t, 1, 5) == "sort ")
    if (stage == "byfirst") return(substr(t, 1, 11) == "quietly by ")
    if (stage == "gencode") return(substr(t, 1, 17) == "quietly generate " & strpos(t, "= sum(") > 0)
    if (stage == "drop") return(substr(t, 1, 5) == "drop ")
    if (stage == "gentarget") return(substr(t, 1, 17) == "quietly generate " & substr(t, strlen(t) - 2, 3) == "= .")
    if (stage == "label") return(substr(t, 1, 15) == "label variable " & strpos(t, "group(") > 0)
    _error(3498, "s4: unknown stage " + stage)
}

// The helper of the ado file, as a do-file that redefines it; with a stage,
// "error 3498" follows that statement, which must occur exactly once.
void s4_helper_copy(string scalar adofile, string scalar stage, string scalar out)
{
    string colvector L, B
    string scalar pad
    real scalar i, a, z, hits, at, fh
    L = cat(adofile)
    a = 0
    z = 0
    for (i = 1; i <= rows(L); i++) {
        if (a == 0) {
            if (strtrim(L[i]) == "program define _xhdfe_string_group") a = i
        }
        else if (z == 0 & L[i] == "end") z = i
    }
    if (a == 0 | z == 0) _error(3498, "s4: the helper was not found in " + adofile)
    B = L[|a \ z|]
    if (stage != "") {
        hits = 0
        at = 0
        for (i = 1; i <= rows(B); i++) {
            if (s4_stage_line(strtrim(B[i]), stage)) {
                hits = hits + 1
                at = i
            }
        }
        if (hits != 1) _error(3498, "s4: stage " + stage + " matched " + strofreal(hits) + " statements")
        pad = substr(B[at], 1, strlen(B[at]) - strlen(strltrim(B[at])))
        B = B[|1 \ at|] \ (pad + "error 3498") \ B[|at + 1 \ rows(B)|]
    }
    if (fileexists(out)) unlink(out)
    fh = fopen(out, "w")
    fput(fh, "capture program drop _xhdfe_string_group")
    for (i = 1; i <= rows(B); i++) fput(fh, B[i])
    fclose(fh)
}

// The Mata function of the ado file, as a do-file that redefines it.
void s4_scatter_copy(string scalar adofile, string scalar out)
{
    string colvector L
    real scalar i, a, z, fh
    L = cat(adofile)
    a = 0
    z = 0
    for (i = 1; i <= rows(L); i++) {
        if (a == 0) {
            if (strtrim(L[i]) == "capture mata: mata drop _xhdfe_group_scatter()") a = i
        }
        else if (z == 0 & strtrim(L[i]) == "end") z = i
    }
    if (a == 0 | z == 0) _error(3498, "s4: the Mata scatter was not found in " + adofile)
    if (fileexists(out)) unlink(out)
    fh = fopen(out, "w")
    for (i = a; i <= z; i++) fput(fh, L[i])
    fclose(fh)
}

// The native arm's program: what the ado ran at the string sites before the
// helper, and a counter of its calls.
void s4_native_file(string scalar out)
{
    string scalar bq, sq
    real scalar fh
    bq = char(96)
    sq = char(39)
    if (fileexists(out)) unlink(out)
    fh = fopen(out, "w")
    fput(fh, "capture program drop _xhdfe_string_group")
    fput(fh, "program define _xhdfe_string_group")
    fput(fh, "    version 16.0")
    fput(fh, "    gettoken target 0 : 0")
    fput(fh, "    gettoken touse 0 : 0")
    fput(fh, "    syntax varlist")
    fput(fh, "    scalar s4_native_calls = scalar(s4_native_calls) + 1")
    fput(fh, "    quietly egen long " + bq + "target" + sq + " = group(" + bq + "varlist" + sq + ") if " + bq + "touse" + sq)
    fput(fh, "end")
    fclose(fh)
}
end

* Number of lines of a text file that contain a string, in r(found).
capture program drop s4_log_count
program define s4_log_count, rclass
    args file text
    tempname h
    file open `h' using `"`file'"', read text
    local found 0
    file read `h' line
    while (!r(eof)) {
        if (strpos(`"`macval(line)'"', `"`text'"')) local ++found
        file read `h' line
    }
    file close `h'
    return scalar found = `found'
end

capture program drop s4_data
program define s4_data
    clear
    set seed 20261113
    quietly set obs 3000
    generate long ord = _n
    generate int firm = 1 + mod(7 * _n, 250)
    generate byte year = 1 + mod(floor((_n - 1) / 11), 8)
    generate byte cnum = 1 + mod(_n, 5)
    quietly replace cnum = . if mod(_n, 97) == 7
    * The same with extended missing values.
    generate byte cext = cnum
    quietly replace cext = .a if mod(_n, 89) == 3
    quietly replace cext = .z if mod(_n, 83) == 5
    generate double u = runiform()
    generate double x1 = rnormal()
    generate double x2 = rnormal()
    generate double y = 1 + x1 - .5 * x2 + firm / 100 + year / 10 + rnormal()
    * Few levels: empty (missing), leading, trailing and inner spaces, case.
    local m = "mod(_n, 7)"
    generate str3 s_few = cond(`m' == 0, "", cond(`m' == 1, " a", cond(`m' == 2, "a", ///
        cond(`m' == 3, "a ", cond(`m' == 4, "a b", cond(`m' == 5, "A", "b"))))))
    * Many levels with duplicates, some empty.
    generate str6 s_many = "id" + string(floor(700 * runiform()))
    quietly replace s_many = "" if mod(_n, 113) == 50
    * Unicode.
    local m = "mod(_n, 9)"
    generate str8 s_uni = cond(`m' == 0, "é", cond(`m' == 1, "e", cond(`m' == 2, "É", ///
        cond(`m' == 3, "ü", cond(`m' == 4, "日本", cond(`m' == 5, "😀", ///
        cond(`m' == 6, "Ω", cond(`m' == 7, "z", "Z"))))))))
    * strL, some values longer than a str2045 could hold.
    generate strL s_L = s_many
    quietly replace s_L = s_L + "x" * 3000 if mod(_n, 11) == 0 & s_L != ""
    generate str1 s_empty = ""
    generate str3 syear = "y" + string(year)
end

capture program drop s4_group_data
program define s4_group_data
    clear
    set seed 20261114
    quietly set obs 400
    generate long g = _n
    generate str6 sgroup = "G" + string(1000 - _n)
    generate double x = rnormal()
    generate double z = rnormal()
    generate double noise = rnormal() / 10
    generate byte area = 1 + mod(_n - 1, 3)
    generate byte sector = 1 + mod(floor((_n - 1) / 3), 4)
    quietly expand 3
    sort g, stable
    by g: generate byte slot = _n
    generate long pid = 1 + mod(17 * g + 7 * slot, 97)
    generate str5 spid = "p" + string(pid)
    generate double y = .6 * x - .2 * z + area * sector / 10 + noise
    egen long joint = group(area sector)
    sort g slot
    generate long ord = _n
end

* The state a call must leave alone, kept in globals so that s4_after can compare.
capture program drop s4_before
program define s4_before
    global S4_SORTEDBY : sortedby
    global S4_FRAME = c(frame)
    global S4_K = c(k)
    mata: st_global("S4_FRAMES", invtokens(sort(st_framedir(), 1)'))
    mata: st_global("S4_MATA", strofreal(rows(direxternal("*"))))
    global S4_RNG = c(rngstate)
    global S4_SORTRNG = c(sortrngstate)
    capture drop s4_pos
    quietly generate long s4_pos = _n
end

capture program drop s4_after
program define s4_after
    syntax [, KEEPORDER NEWVARS(integer 0)]
    local changed
    if ("`c(frame)'" != "$S4_FRAME") local changed "`changed' current-frame"
    if ("`keeporder'" == "") {
        capture assert s4_pos == _n
        if (_rc) local changed "`changed' order"
    }
    drop s4_pos
    local sb : sortedby
    if ("`sb'" != "$S4_SORTEDBY") local changed "`changed' sortedby"
    if (c(k) != $S4_K + `newvars') local changed "`changed' variables"
    mata: st_local("frames", invtokens(sort(st_framedir(), 1)'))
    mata: st_local("matag", strofreal(rows(direxternal("*"))))
    if ("`frames'" != "$S4_FRAMES") local changed "`changed' frames"
    if (`matag' != $S4_MATA) local changed "`changed' mata-globals"
    if ("`changed'" != "") {
        display as error "state changed by the call:`changed'"
        exit 9
    }
end

* ---------------------------------------------------------------------------
* codes
* ---------------------------------------------------------------------------
capture program drop s4_codes_one
program define s4_codes_one
    args touse list
    capture drop s4_e s4_h
    s4_before
    quietly egen long s4_e = group(`list') if `touse'
    s4_after, newvars(1)
    s4_before
    _xhdfe_string_group s4_h `touse' `list'
    s4_after, newvars(1)
    assert s4_e == s4_h
    assert "`: type s4_e'" == "`: type s4_h'"
    assert `"`: variable label s4_e'"' == `"`: variable label s4_h'"'
    assert `"`: char s4_e[note0]'"' == `"`: char s4_h[note0]'"'
    assert `"`: char s4_e[note1]'"' == `"`: char s4_h[note1]'"'
    drop s4_e s4_h
    global S4_CODES = $S4_CODES + 1
end

capture program drop s4_codes_all
program define s4_codes_all
    local lists `""s_few" "s_many" "s_uni" "s_L" "s_empty" "s_many cnum" "cnum s_many""'
    local lists `"`lists' "s_few syear" "s_many syear" "s_few s_few" "s_many s_uni s_L""'
    * Extended missing values (.a, .z) in the numeric component.
    local lists `"`lists' "s_many cext" "cext s_few syear""'
    * A label longer than 80 characters: egen writes it as a note.
    local lists `"`lists' "long_identifier_variable_name_a1 long_identifier_variable_name_b2 long_identifier_variable_name_c3""'
    foreach order in ord byid random reverse {
        if ("`order'" == "ord") sort ord
        else if ("`order'" == "byid") sort s_many ord
        else if ("`order'" == "random") sort u ord
        else {
            tempvar rev
            quietly generate long `rev' = -ord
            sort `rev'
            drop `rev'
        }
        foreach touse in t_all t_pos t_in t_none t_miss t_one {
            foreach list of local lists {
                s4_codes_one `touse' "`list'"
            }
        }
    }
    sort ord
end

* Replaces the helper's Mata function by one that switches to the temporary
* frame and then fails (error 3498) or exits with the Break code (rc 1), or by
* one that stores the codes of half the selected observations and then fails
* (partial; it leaves the number stored in the scalar s4_partial). A Mata block
* cannot sit inside a program, so the definition is written to a do-file in the
* current directory and run.
capture program drop s4_mata_fault
program define s4_mata_fault
    args kind
    local f "string_id_encoding_fault_`kind'.do"
    tempname h
    file open `h' using `"`f'"', write text replace
    file write `h' "mata: mata drop _xhdfe_group_scatter()" _n "mata:" _n
    file write `h' "void _xhdfe_group_scatter(string scalar frame, string scalar idxvar," _n
    file write `h' "    string scalar codevar, string scalar target, string scalar selvar)" _n
    file write `h' "{" _n
    if ("`kind'" == "partial") {
        file write `h' "    string scalar origin" _n "    real colvector out, sel" _n "    real scalar n, h" _n
        file write `h' "    origin = st_framecurrent()" _n "    st_framecurrent(frame)" _n "    n = st_nobs()" _n
        file write `h' "    out = J(n, 1, .)" _n "    out[st_data(., idxvar)] = st_data(., codevar)" _n
        file write `h' "    st_framecurrent(origin)" _n "    sel = selectindex(st_data(., selvar))" _n
        file write `h' "    h = ceil(n / 2)" _n "    st_store(sel[|1 \ h|], target, out[|1 \ h|])" _n
        file write `h' `"    st_numscalar("s4_partial", h)"' _n
        file write `h' `"    _error(3498, "injected fault after a partial scatter")"' _n
    }
    else {
        local stmt `"_error(3498, "injected fault")"'
        if ("`kind'" == "break") local stmt "exit(1)"
        file write `h' "    st_framecurrent(frame)" _n `"    `stmt'"' _n
    }
    file write `h' "}" _n "end" _n
    file close `h'
    run `"`f'"'
    erase `"`f'"'
end

* Redefines the helper from the ado file, with "error 3498" after the statement
* of a stage, or as it is when the stage is empty.
capture program drop s4_stage_fault
program define s4_stage_fault
    args adofile stage
    local f "string_id_encoding_stage.do"
    mata: s4_helper_copy(st_local("adofile"), st_local("stage"), st_local("f"))
    run `"`f'"'
    erase `"`f'"'
end

* Redefines the helper's Mata function from the ado file.
capture program drop s4_scatter_restore
program define s4_scatter_restore
    args adofile
    local f "string_id_encoding_scatter.do"
    mata: s4_scatter_copy(st_local("adofile"), st_local("f"))
    run `"`f'"'
    erase `"`f'"'
end

if ("`mode'" == "codes") {
    if (`"`adopath'"' == "") error 601
    adopath ++ `"`adopath'"'
    quietly findfile xhdfe.ado
    local adofile `"`r(fn)'"'
    assert `"`adofile'"' == `"`adopath'/xhdfe.ado"'
    run `"`adofile'"'
    global S4_CODES 0
    s4_data
    generate str6 long_identifier_variable_name_a1 = s_many
    generate str3 long_identifier_variable_name_b2 = s_few
    generate str8 long_identifier_variable_name_c3 = s_uni
    generate byte t_all = 1
    generate byte t_pos = x1 > 0
    generate byte t_in = inrange(_n, 101, 2900)
    generate byte t_none = 0
    generate byte t_miss = x2 > -1
    generate byte t_one = (_n == 17)
    s4_codes_all
    * The same comparisons from a frame that is not default.
    frame copy default s4_other
    frame s4_other: s4_codes_all
    frame drop s4_other
    * Abbreviations: with varabbrev on the helper takes an abbreviated name as
    * egen does; with it off both refuse it.
    set varabbrev on
    s4_codes_one t_all "s_ma"
    s4_codes_one t_pos "s_ma syear"
    set varabbrev off
    capture _xhdfe_string_group s4_h t_all s_ma
    local rc_helper = _rc
    capture egen long s4_e = group(s_ma) if t_all
    assert `rc_helper' == _rc & _rc == 111
    * The caller's version (those this Stata knows) and the default storage
    * type do not change the codes.
    foreach v in 16 17 18 19 {
        if (c(stata_version) < `v') continue
        quietly egen long s4_e = group(s_many cext) if t_pos
        version `v': _xhdfe_string_group s4_h t_pos s_many cext
        assert s4_e == s4_h
        assert "`: type s4_e'" == "`: type s4_h'"
        drop s4_e s4_h
    }
    set type double
    s4_codes_one t_all "s_many"
    s4_codes_one t_miss "s_few syear"
    set type float
    * Group-mode identifiers.
    preserve
    s4_group_data
    generate byte t_all = 1
    foreach list in sgroup spid "joint spid" "spid joint" {
        s4_codes_one t_all "`list'"
    }
    restore
    display "STRING_ID_CODES comparisons=$S4_CODES"
    * What the helper and egen leave in r() and in the EGEN_* globals (printed,
    * not compared: the battery compares what the command leaves).
    quietly egen long s4_e = group(s_many) if t_all
    local rs : r(scalars)
    local rv
    foreach s of local rs {
        local rv "`rv' r(`s')=`=r(`s')'"
    }
    s4_say STRING_ID_STATE egen: r(scalars)=`rs'`rv' EGEN_Varname=[$EGEN_Varname] EGEN_SVarname=[$EGEN_SVarname] EGEN_Message=[$EGEN_Message]
    _xhdfe_string_group s4_h t_all s_many
    local rs : r(scalars)
    local rv
    foreach s of local rs {
        local rv "`rv' r(`s')=`=r(`s')'"
    }
    s4_say STRING_ID_STATE helper: r(scalars)=`rs'`rv' EGEN_Varname=[$EGEN_Varname] EGEN_SVarname=[$EGEN_SVarname] EGEN_Message=[$EGEN_Message]
    drop s4_e s4_h
    * A value the user left in EGEN_Varname: egen clears it, the helper does not.
    global EGEN_Varname "s4_user"
    quietly egen long s4_e = group(s_many) if t_all
    s4_say STRING_ID_STATE EGEN_Varname set to s4_user before egen: after it [$EGEN_Varname]
    global EGEN_Varname "s4_user"
    _xhdfe_string_group s4_h t_all s_many
    s4_say STRING_ID_STATE EGEN_Varname set to s4_user before the helper: after it [$EGEN_Varname]
    global EGEN_Varname
    drop s4_e s4_h
    * The comparison detects a relabelling of two levels.
    s4_data
    generate byte t_all = 1
    quietly egen long s4_e = group(s_many) if t_all
    _xhdfe_string_group s4_h t_all s_many
    quietly replace s4_h = cond(s4_h == 1, 2, cond(s4_h == 2, 1, s4_h))
    capture assert s4_e == s4_h
    assert _rc == 9
    display "STRING_ID_CODES relabelling detected"
    drop s4_e s4_h
    * A target that exists is refused as egen refuses it.
    capture _xhdfe_string_group x1 t_all s_many
    local rc_helper = _rc
    capture egen long x1 = group(s_many)
    assert `rc_helper' == _rc & _rc == 110
    * References with the working helper: egen's codes and one fit.
    quietly egen long s4_eref = group(s_many) if t_all
    quietly xhdfe y x1 x2, absorb(s_many year) vce(cluster s_many) numthreads(1)
    matrix s4_bref = e(b)
    matrix s4_Vref = e(V)
    scalar s4_nref = e(N)
    scalar s4_dfref = e(df_r)
    scalar s4_rssref = e(rss)
    generate byte s4_sref = e(sample)
    * The frame limit: with every frame Stata allows in use, frame put fails
    * for real. Without strict mode the helper and the command fall back to
    * egen; in strict mode they stop with the code of frame put; the frames of
    * the session and the data are left as they were.
    local filled 0
    local full_rc 0
    while (`full_rc' == 0) {
        capture frame create s4_fill`filled'
        local full_rc = _rc
        if (`full_rc' == 0) local ++filled
    }
    display "STRING_ID_CODES frame limit: `filled' frames added, frame create rc=`full_rc'"
    assert `filled' > 0
    foreach strict in 1 0 {
        global XHDFE_STRING_GROUP_STRICT = cond(`strict', "1", "")
        s4_before
        capture noisily _xhdfe_string_group s4_f t_all s_many
        local rc = _rc
        s4_after, newvars(`=!`strict'')
        if (`strict') {
            assert `rc' == `full_rc'
            capture confirm variable s4_f
            assert _rc == 111
        }
        else {
            assert `rc' == 0
            assert s4_f == s4_eref
            assert "`: type s4_f'" == "`: type s4_eref'"
            assert `"`: variable label s4_f'"' == `"`: variable label s4_eref'"'
            drop s4_f
        }
        s4_before
        capture noisily xhdfe y x1 x2, absorb(s_many year) vce(cluster s_many) numthreads(1)
        local rc = _rc
        s4_after
        if (`strict') {
            assert `rc' == `full_rc'
            assert `"`e(cmd)'"' == ""
        }
        else {
            assert `rc' == 0
            assert mreldif(e(b), s4_bref) == 0 & mreldif(e(V), s4_Vref) == 0
            assert e(N) == scalar(s4_nref) & e(df_r) == scalar(s4_dfref) & e(rss) == scalar(s4_rssref)
            assert e(sample) == s4_sref
        }
        display "STRING_ID_CODES frame limit strict=`strict' rc=`rc' frames and data intact"
    }
    forvalues i = 0/`=`filled' - 1' {
        frame drop s4_fill`i'
    }
    global XHDFE_STRING_GROUP_STRICT 1
    * Faults at the deepest point: frame built, target created, current frame
    * switched to the temporary frame inside Mata. Without strict mode an error
    * falls back to egen (codes and estimates as egen's, and with the profile
    * switch off no profile line, logged through capture noisily); in strict
    * mode it stops; Break (rc 1) stops in both. Never a residue. They run from
    * a frame that is not default: a program that ends with its temporary frame
    * current leaves Stata in the default frame, so only from another frame does
    * the check tell a return to the frame of origin apart from that.
    frame copy default s4_work
    frame change s4_work
    foreach kind in error break {
        s4_mata_fault `kind'
        foreach strict in 1 0 {
            global XHDFE_STRING_GROUP_STRICT = cond(`strict', "1", "")
            local falls = ("`kind'" == "error" & !`strict')
            local expect = cond("`kind'" == "error", 3498, 1)
            s4_before
            log using "string_id_encoding_fallback.log", text replace name(s4fallback) nomsg
            capture noisily _xhdfe_string_group s4_f t_all s_many
            local rc = _rc
            log close s4fallback
            s4_after, newvars(`falls')
            s4_log_count "string_id_encoding_fallback.log" "string_group_fallback"
            assert r(found) == 0
            erase "string_id_encoding_fallback.log"
            if (`falls') {
                assert `rc' == 0
                assert s4_f == s4_eref
                assert "`: type s4_f'" == "`: type s4_eref'"
                assert `"`: variable label s4_f'"' == `"`: variable label s4_eref'"'
                drop s4_f
            }
            else {
                assert `rc' == `expect'
                capture confirm variable s4_f
                assert _rc == 111
            }
            * The same fault through the command, called as a user calls it.
            s4_before
            log using "string_id_encoding_fallback.log", text replace name(s4fallback) nomsg
            capture noisily xhdfe y x1 x2, absorb(s_many year) vce(cluster s_many) numthreads(1)
            local rc = _rc
            log close s4fallback
            s4_after
            s4_log_count "string_id_encoding_fallback.log" "string_group_fallback"
            assert r(found) == 0
            erase "string_id_encoding_fallback.log"
            if (`falls') {
                assert `rc' == 0
                assert mreldif(e(b), s4_bref) == 0 & mreldif(e(V), s4_Vref) == 0
                assert e(N) == scalar(s4_nref) & e(df_r) == scalar(s4_dfref) & e(rss) == scalar(s4_rssref)
                assert e(sample) == s4_sref
            }
            else {
                assert `rc' == `expect'
                assert `"`e(cmd)'"' == ""
            }
            display "STRING_ID_CODES fault `kind' strict=`strict' rc=`rc' fallback=`falls' no residue, frame `c(frame)'"
        }
    }
    * A failure after each statement of the body, injected into a copy of the
    * helper of this package (the stages before the Mata function have no
    * deterministic real failure), and a Mata function that stores the codes of
    * half the selected observations and then fails: the target never survives,
    * the fallback gives all of egen's codes, strict mode stops.
    s4_scatter_restore `"`adofile'"'
    foreach stage in mark markout count frameput genidx sort byfirst gencode drop gentarget label partial {
        if ("`stage'" == "partial") s4_mata_fault partial
        else s4_stage_fault `"`adofile'"' `stage'
        foreach strict in 1 0 {
            global XHDFE_STRING_GROUP_STRICT = cond(`strict', "1", "")
            scalar s4_partial = .
            s4_before
            capture noisily _xhdfe_string_group s4_f t_all s_many
            local rc = _rc
            s4_after, newvars(`=!`strict'')
            if (`strict') {
                assert `rc' == 3498
                capture confirm variable s4_f
                assert _rc == 111
            }
            else {
                assert `rc' == 0
                assert s4_f == s4_eref
                assert "`: type s4_f'" == "`: type s4_eref'"
                assert `"`: variable label s4_f'"' == `"`: variable label s4_eref'"'
                drop s4_f
            }
            if ("`stage'" == "partial") assert scalar(s4_partial) > 0 & scalar(s4_partial) < _N
            display "STRING_ID_CODES stage `stage' strict=`strict' rc=`rc' no residue, frame `c(frame)'"
        }
        if ("`stage'" == "partial") s4_scatter_restore `"`adofile'"'
        else s4_stage_fault `"`adofile'"' ""
    }
    frame change default
    frame drop s4_work
    global XHDFE_STRING_GROUP_STRICT 1
    * Which calls reach the helper: with the failing Mata function every string
    * site stops with 3498 in strict mode and falls back without it; the
    * numeric paths (egen or gen) never reach it.
    s4_mata_fault error
    local stop `""absorb(s_many year)" "absorb(s_many#cnum year)""'
    local stop `"`stop' "absorb(firm year) vce(cluster s_many#syear)""'
    local stop `"`stop' "absorb(firm year) vce(cluster s_many)""'
    foreach strict in 1 0 {
        global XHDFE_STRING_GROUP_STRICT = cond(`strict', "1", "")
        foreach spec of local stop {
            capture xhdfe y x1 x2, `spec' numthreads(1)
            assert _rc == cond(`strict', 3498, 0)
            display "STRING_ID_CODES site reached: `spec' strict=`strict' rc=" _rc
        }
        foreach spec in "absorb(firm#cnum year) vce(cluster firm#cnum)" "absorb(firm year) vce(cluster firm)" {
            capture xhdfe y x1 x2, `spec' numthreads(1)
            assert _rc == 0
            display "STRING_ID_CODES numeric path kept: `spec' strict=`strict'"
        }
    }
    s4_group_data
    generate float fgroup = g
    foreach strict in 1 0 {
        global XHDFE_STRING_GROUP_STRICT = cond(`strict', "1", "")
        capture xhdfe y x z, absorb(joint pid) group(sgroup) individual(pid) numthreads(1)
        assert _rc == cond(`strict', 3498, 0)
        display "STRING_ID_CODES site reached: group(sgroup) strict=`strict' rc=" _rc
        capture xhdfe y x z, absorb(joint pid) group(fgroup) individual(pid) numthreads(1)
        assert _rc == 0
        display "STRING_ID_CODES numeric path kept: group(fgroup), float strict=`strict'"
    }
    global XHDFE_STRING_GROUP_STRICT 1
    display "STRING_ID_ENCODING_CODES_PASS"
    exit
}

* ---------------------------------------------------------------------------
* profile
* ---------------------------------------------------------------------------
if ("`mode'" == "profile") {
    local profile : environment XHDFE_PROFILE_CPU
    if (`"`adopath'"' == "" | "`profile'" != "1") error 198
    adopath ++ `"`adopath'"'
    run `"`adopath'/xhdfe.ado"'
    s4_data
    generate byte t_all = 1
    quietly egen long s4_eref = group(s_many) if t_all
    s4_mata_fault error
    global XHDFE_STRING_GROUP_STRICT ""
    * The helper alone, and the command as a user calls it (noisily).
    log using "string_id_encoding_profile.log", text replace name(s4profile) nomsg
    capture noisily _xhdfe_string_group s4_f t_all s_many
    local rc = _rc
    log close s4profile
    assert `rc' == 0
    assert s4_f == s4_eref
    s4_log_count "string_id_encoding_profile.log" "ado_profile label=string_group_fallback rc=3498"
    assert r(found) == 1
    log using "string_id_encoding_profile_fit.log", text replace name(s4profile) nomsg
    capture noisily xhdfe y x1 x2, absorb(s_many year) numthreads(1)
    local rc = _rc
    log close s4profile
    assert `rc' == 0
    s4_log_count "string_id_encoding_profile_fit.log" "ado_profile label=string_group_fallback rc=3498"
    assert r(found) == 1
    display "STRING_ID_ENCODING_PROFILE_PASS"
    exit
}

* ---------------------------------------------------------------------------
* battery
* ---------------------------------------------------------------------------
capture program drop s4_record
program define s4_record
    syntax name(name=tag), rc(integer) [OUTPUTS(varlist) ORDER]
    file write S4OUT "CASE `tag' rc=`rc'" _n
    file write S4RNG "`tag' rng_before=$S4_RNG" _n
    file write S4RNG "`tag' rng_after=`c(rngstate)'" _n
    file write S4SRT "`tag' sortrng_before=$S4_SORTRNG" _n
    file write S4SRT "`tag' sortrng_after=`c(sortrngstate)'" _n
    if (`rc' != 0) exit
    file write S4OUT `"`tag' absvars=`e(absvars)' clustvar=`e(clustvar)' cmdline=`e(cmdline)'"' _n
    foreach s in N df_r rss rank df_a df_m N_clust N_hdfe r2 F {
        file write S4OUT "`tag' `s'=" %21x (e(`s')) _n
    }
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    file write S4OUT "`tag' names=`names'" _n
    local k = colsof(`b')
    forvalues j = 1/`k' {
        file write S4OUT "`tag' b`j'=" %21x (`b'[1, `j']) _n
        forvalues i = 1/`k' {
            file write S4OUT "`tag' V`i'_`j'=" %21x (`V'[`i', `j']) _n
        }
    }
    tempvar es
    quietly generate byte `es' = e(sample)
    mata: st_local("bits", invtokens(strofreal(st_data(., "`es'"))', ""))
    file write S4OUT "`tag' sample=`bits'" _n
    if ("`order'" != "") {
        mata: st_local("seq", invtokens(strofreal(st_data(., "ord"))', ","))
        file write S4OUT "`tag' order=`seq'" _n
    }
    foreach v of local outputs {
        local type : type `v'
        file write S4OUT "`tag' `v' type=`type'" _n
        forvalues i = 1/`=_N' {
            file write S4OUT "`tag' `v'[`i']=" %21x (`v'[`i']) _n
        }
    }
end

* What a call leaves behind, recorded before anything else runs: c(rc), r()
* and s() (names and values), and the EGEN_* globals of egen.
capture program drop s4_state
program define s4_state
    args tag crc
    file write S4OUT "`tag' crc=`crc'" _n
    foreach kind in scalars macros matrices {
        local names : r(`kind')
        file write S4OUT "`tag' r_`kind'=`names'" _n
        foreach n of local names {
            if ("`kind'" == "scalars") {
                file write S4OUT "`tag' r(`n')=" %21x (r(`n')) _n
            }
            else if ("`kind'" == "macros") {
                mata: st_local("hx", s4_hexstr(st_global("r(`n')")))
                file write S4OUT "`tag' r(`n')=`hx'" _n
            }
            else {
                mata: st_local("hx", s4_hexmat("r(`n')"))
                file write S4OUT "`tag' r(`n')=`hx'" _n
            }
        }
    }
    local names : s(macros)
    file write S4OUT "`tag' s_macros=`names'" _n
    foreach n of local names {
        mata: st_local("hx", s4_hexstr(st_global("s(`n')")))
        file write S4OUT "`tag' s(`n')=`hx'" _n
    }
    mata: st_local("hx", s4_hexstr(st_global("EGEN_Varname") + "|" + st_global("EGEN_SVarname") + ///
        "|" + st_global("EGEN_Message") + "|" + st_global("EGEN_GROUP_debug")))
    file write S4OUT "`tag' egen_globals=`hx'" _n
end

* The characteristics of the data and of every variable.
capture program drop s4_chars
program define s4_chars
    args tag
    local names : char _dta[]
    file write S4OUT "`tag' char _dta[]=`names'" _n
    foreach c of local names {
        mata: st_local("hx", s4_hexstr(st_global("_dta[`c']")))
        file write S4OUT "`tag' char _dta[`c']=`hx'" _n
    }
    foreach v of varlist _all {
        local names : char `v'[]
        if ("`names'" == "") continue
        file write S4OUT "`tag' char `v'[]=`names'" _n
        foreach c of local names {
            mata: st_local("hx", s4_hexstr(st_global("`v'[`c']")))
            file write S4OUT "`tag' char `v'[`c']=`hx'" _n
        }
    }
end

* The Mata functions defined in the session, sorted, in a global.
capture program drop s4_mata_inventory
program define s4_mata_inventory
    args gname
    local f "string_id_encoding_inventory.log"
    quietly log using `"`f'"', text replace nomsg name(s4inv)
    mata: mata describe
    quietly log close s4inv
    mata: st_global(st_local("gname"), s4_function_names(st_local("f")))
    erase `"`f'"'
end

* s4_case <tag> [, outputs(<new variables the call creates>) record(<existing
*   variables it refills>) keeporder expect(<return code>) calls(<string-site
*   calls>)] : <command>
* The arm itself fails when the return code is not the expected one, when a
* successful fit does not report e(converged) 1 and e(gpu_used) 0, and, in the
* native arm, when its program did not run exactly calls() times.
capture program drop s4_case
program define s4_case
    _on_colon_parse `0'
    local command `"`s(after)'"'
    local 0 `"`s(before)'"'
    syntax name(name=tag) [, OUTPUTS(namelist) RECORD(varlist) KEEPORDER EXPECT(integer 0) CALLS(integer -1)]
    local nout : word count `outputs'
    scalar s4_native_calls = 0
    s4_before
    capture noisily quietly `command'
    local rc = _rc
    local crc = c(rc)
    s4_state `tag' `crc'
    s4_after, newvars(`nout') `keeporder'
    local order = cond("`keeporder'" != "", "order", "")
    local written `outputs' `record'
    s4_record `tag', rc(`rc') outputs(`written') `order'
    s4_chars `tag'
    if ("`outputs'" != "") capture drop `outputs'
    if (`rc' != `expect') {
        display as error "case `tag': return code `rc', expected `expect'"
        exit 9
    }
    if (`rc' == 0) {
        if (e(converged) != 1 | e(gpu_used) != 0) {
            display as error "case `tag': e(converged)=" e(converged) ", e(gpu_used)=" e(gpu_used)
            exit 9
        }
    }
    if ("$S4_ARM" == "native") {
        display "STRING_ID_NATIVE `tag' calls=" scalar(s4_native_calls) " expected=`calls'"
        if (scalar(s4_native_calls) != `calls') {
            display as error "case `tag': the native program ran " scalar(s4_native_calls) " times, expected `calls'"
            exit 9
        }
    }
end

if ("`mode'" == "battery") {
    if (`"`adopath'"' == "" | `"`arm'"' == "") error 198
    global S4_ARM "`arm'"
    adopath ++ `"`adopath'"'
    quietly findfile xhdfe.ado
    local adofile `"`r(fn)'"'
    assert `"`adofile'"' == `"`adopath'/xhdfe.ado"'
    * Every arm runs the ado file, so that its programs are ordinary programs
    * (a subroutine of an autoloaded ado file is private to it); the native
    * arm then replaces the helper and installs a Mata function that stops any
    * call reaching the frame-based body.
    s4_mata_inventory S4_INV0
    run `"`adofile'"'
    if ("`arm'" == "native") {
        mata: s4_native_file("string_id_encoding_native.do")
        run "string_id_encoding_native.do"
        erase "string_id_encoding_native.do"
        s4_mata_fault error
        program list _xhdfe_string_group
    }
    which xhdfe
    file open S4OUT using "string_id_encoding_`arm'.txt", write text replace
    file open S4RNG using "string_id_encoding_`arm'_rng.txt", write text replace
    file open S4SRT using "string_id_encoding_`arm'_sortrng.txt", write text replace
    local o "numthreads(1)"
    s4_data
    s4_case A01, calls(1) : xhdfe y x1 x2, absorb(s_few) `o'
    s4_case A02, calls(1) : xhdfe y x1 x2, absorb(s_many year) vce(cluster s_many) `o'
    s4_case A03, calls(1) : xhdfe y x1 x2 if x1 > 0, absorb(s_many year) `o'
    s4_case A04, calls(1) : xhdfe y x1 x2 in 101/2900, absorb(s_many year) `o'
    s4_case A05, calls(1) : xhdfe y x1 x2, absorb(s_uni year) `o'
    s4_case A06, calls(1) : xhdfe y x1 x2, absorb(s_L year) `o'
    s4_case A07, calls(1) : xhdfe y x1 x2, absorb(s_many#cnum year) `o'
    s4_case A08, calls(1) : xhdfe y x1 x2, absorb(s_few#syear firm) `o'
    s4_case A09, calls(1) : xhdfe y x1 x2, absorb(firm year) vce(cluster s_many) `o'
    s4_case A10, calls(1) : xhdfe y x1 x2, absorb(firm year) vce(cluster s_many#syear) `o'
    s4_case A11, calls(0) : xhdfe y x1 x2, absorb(firm year) vce(cluster firm) `o'
    s4_case A12, calls(0) : xhdfe y x1 x2, absorb(firm#cnum year) vce(cluster firm#cnum) `o'
    s4_case A13, outputs(__hdfe1__ __hdfe2__ s4_res) calls(1) : ///
        xhdfe y x1 x2, absorb(s_many year, savefe) residuals(s4_res) `o'
    s4_case A14, calls(1) : xhdfe y x1 x2, absorb(s_many year) vce(cluster s_many) `o'
    s4_case A15, calls(1) : xhdfe y x1 x2, absorb(s_many year) vce(cluster s_many) `o'
    s4_case A16, calls(2) : xhdfe y x1 x2, absorb(s_L#s_uni) vce(cluster s_few#s_uni) `o'
    s4_case A17, expect(2000) calls(1) : xhdfe y x1 x2, absorb(s_empty) `o'
    * Extended missing values in the numeric component of an interaction.
    s4_case E01, calls(1) : xhdfe y x1 x2, absorb(s_many#cext year) `o'
    s4_case E02, calls(1) : xhdfe y x1 x2, absorb(firm year) vce(cluster s_few#cext) `o'
    * An abbreviated name with varabbrev on, and the same call with it off.
    set varabbrev on
    s4_case V01, calls(1) : xhdfe y x1 x2, absorb(s_ma year) vce(cluster s_ma) `o'
    set varabbrev off
    s4_case V02, expect(111) calls(0) : xhdfe y x1 x2, absorb(s_ma year) `o'
    * The caller's version (those this Stata knows), and the default storage type.
    foreach v in 16 17 18 19 {
        if (c(stata_version) < `v') continue
        s4_case R`v', calls(1) : version `v': xhdfe y x1 x2, absorb(s_many year) vce(cluster s_many) `o'
    }
    set type double
    s4_case T01, calls(2) : xhdfe y x1 x2, absorb(s_many#cnum year) vce(cluster s_few) `o'
    set type float
    * Several observation orders, with tie-free sorts.
    sort s_many ord
    s4_case B01, calls(1) : xhdfe y x1 x2, absorb(s_many year) vce(cluster s_many) `o'
    sort u ord
    s4_case B02, calls(1) : xhdfe y x1 x2, absorb(s_many year) vce(cluster s_many) `o'
    tempvar rev
    generate long `rev' = -ord
    sort `rev'
    drop `rev'
    s4_case B03, calls(3) : xhdfe y x1 x2, absorb(s_many#syear s_uni) vce(cluster s_many) `o'
    sort ord
    * From a frame that is not default.
    frame copy default s4_work
    frame s4_work {
        s4_case C01, calls(1) : xhdfe y x1 x2, absorb(s_many year) vce(cluster s_many) `o'
    }
    frame drop s4_work
    * The sortpreserve runner: data sorted by an automatic output of this call.
    quietly xhdfe y x1 x2, absorb(s_many year, savefe) `o'
    sort __hdfe1__ ord
    s4_case C02, keeporder record(__hdfe1__ __hdfe2__) calls(1) : xhdfe y x1 x2, absorb(s_many year, savefe) `o'
    capture drop __hdfe1__ __hdfe2__
    sort ord
    * A string alias variable (frames, Stata 18 and later).
    if (c(stata_version) >= 18) {
        frame put ord s_many, into(s4_alias)
        frlink 1:1 ord, frame(s4_alias) generate(s4_link)
        fralias add s_al = s_many, from(s4_link)
        * frlink records the time it ran in a characteristic; no time is compared.
        char s4_link[frlink_date]
        s4_case C03, calls(1) : xhdfe y x1 x2, absorb(s_al year) vce(cluster s_al) `o'
        drop s_al s4_link
        frame drop s4_alias
    }
    * group() as a string, with individual() numeric and as a string.
    s4_group_data
    s4_case D01, calls(1) : xhdfe y x z, absorb(joint pid) group(sgroup) individual(pid) aggregation(mean) `o'
    s4_case D02, calls(2) : xhdfe y x z, absorb(joint spid) group(sgroup) individual(spid) aggregation(mean) `o'
    s4_case D03, calls(1) : xhdfe y x z, absorb(joint pid) group(sgroup) individual(pid) aggregation(sum) `o'
    file close S4OUT
    file close S4RNG
    file close S4SRT
    * One fit as printed.
    s4_data
    log using "string_id_encoding_`arm'_printed.log", text replace name(s4printed) nomsg
    noisily xhdfe y x1 x2, absorb(s_many#syear s_uni) vce(cluster s_many) `o'
    log close s4printed
    * The Mata functions the session gained must all be the ado's own.
    s4_mata_inventory S4_INV1
    local inv0 "$S4_INV0"
    local inv1 "$S4_INV1"
    local added : list inv1 - inv0
    local lost : list inv0 - inv1
    s4_say STRING_ID_MATA_FUNCTIONS added=`added' removed=`lost'
    assert "`lost'" == ""
    foreach f of local added {
        if (substr("`f'", 1, 7) != "_xhdfe_") {
            display as error "Mata function `f'() was added and is not the ado's own"
            exit 9
        }
    }
    display "STRING_ID_ENCODING_BATTERY_DONE `arm'"
    exit
}

* ---------------------------------------------------------------------------
* compare
* ---------------------------------------------------------------------------
capture program drop s4_same_file
program define s4_same_file
    args left right
    tempname a b
    file open `a' using `"`left'"', read text
    file open `b' using `"`right'"', read text
    local n 0
    local diff 0
    file read `a' la
    local ea = r(eof)
    file read `b' lb
    local eb = r(eof)
    while (!`ea' | !`eb') {
        if (`ea' != `eb') {
            local ++diff
            display as error "`left' and `right' have different numbers of lines"
            continue, break
        }
        local ++n
        if (`"`macval(la)'"' != `"`macval(lb)'"') {
            local ++diff
            if (`diff' == 1) display as error "first difference at line `n' of `left'"
        }
        file read `a' la
        local ea = r(eof)
        file read `b' lb
        local eb = r(eof)
    }
    file close `a'
    file close `b'
    display "STRING_ID_COMPARE `left' lines=`n' differences=`diff'"
    if (`diff') exit 9
end

* The lines of two cases of one battery file, without their tags, must be equal:
* A02, A14 and A15 are the same call made three times in one session.
capture program drop s4_same_tags
program define s4_same_tags
    args file t1 t2
    tempname h
    file open `h' using `"`file'"', read text
    local n1 0
    local n2 0
    file read `h' line
    while (!r(eof)) {
        gettoken first rest : line
        if ("`first'" == "`t1'") {
            local ++n1
            local a`n1' `"`macval(rest)'"'
        }
        else if ("`first'" == "`t2'") {
            local ++n2
            local b`n2' `"`macval(rest)'"'
        }
        file read `h' line
    }
    file close `h'
    assert `n1' == `n2' & `n1' > 0
    forvalues i = 1/`n1' {
        assert `"`macval(a`i')'"' == `"`macval(b`i')'"'
    }
    display "STRING_ID_COMPARE `file' `t1'=`t2' lines=`n1'"
end

if ("`mode'" == "compare") {
    local left = cond(`"`arm'"' == "", "patched", `"`arm'"')
    local right = cond(`"`arm2'"' == "", "native", `"`arm2'"')
    set linesize 255
    foreach a in `left' `right' {
        s4_same_tags string_id_encoding_`a'.txt A02 A14
        s4_same_tags string_id_encoding_`a'.txt A02 A15
    }
    s4_same_file string_id_encoding_`left'.txt string_id_encoding_`right'.txt
    s4_same_file string_id_encoding_`left'_rng.txt string_id_encoding_`right'_rng.txt
    s4_same_file string_id_encoding_`left'_printed.log string_id_encoding_`right'_printed.log
    capture noisily s4_same_file string_id_encoding_`left'_sortrng.txt string_id_encoding_`right'_sortrng.txt
    display "STRING_ID_COMPARE sort-tiebreak states identical=" (_rc == 0)
    display "STRING_ID_ENCODING_CONTRACT_PASS"
    exit
}

display as error "usage: do string_id_encoding_contract.do codes | profile | battery <arm> | compare [<left> <right>]"
exit 198
