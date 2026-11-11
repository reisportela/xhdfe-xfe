version 16.0
clear all
set more off

local ROOT "/home/mangelo/Documents/GitHub/xhdfe"
local ROOT_ENV : environment XHDFE_REPO_ROOT
if ("`ROOT_ENV'" != "") local ROOT "`ROOT_ENV'"
adopath ++ "`ROOT'/stata"
discard

set obs 600000
generate long row = _n - 1
generate int fe1 = mod(row, 1200)
generate int fe2 = mod(floor(row / 1200), 500)
generate double x1 = sin(row * .001) + mod(row * 17, 101) / 101
generate double x2 = cos(row * .0013) + mod(row * 29, 103) / 103
generate double y = .6*x1 - .25*x2 + .001*fe1 - .0005*fe2 + sin(row * .0021)

foreach threads in 1 2 8 16 24 48 {
    quietly xhdfe y x1 x2, absorb(fe1 fe2) vce(cluster fe1) keepsingletons ///
        tolerance(1e-10) tolerancemode(reghdfe-comparable) ///
        numthreads(`threads') gpubackend(cpu)
    assert e(converged) == 1
    assert e(threads_requested) == `threads'
    assert e(threads_effective) == `threads'
    assert e(threads_used) == `threads'
    assert e(gpu_used) == 0

    if (`threads' == 1) {
        matrix b_ref = e(b)
        matrix V_ref = e(V)
        scalar iterations_ref = e(iterations)
    }
    else {
        mata: st_numscalar("threads_b_diff", ///
            max(abs(st_matrix("e(b)") :- st_matrix("b_ref"))))
        mata: st_numscalar("threads_V_diff", ///
            max(abs(st_matrix("e(V)") :- st_matrix("V_ref"))))
        assert scalar(threads_b_diff) <= 1e-13
        assert scalar(threads_V_diff) <= 1e-13
        assert e(iterations) == scalar(iterations_ref)
    }
    display as result "XHDFE_THREADS_CONTRACT|cpu|requested=`threads'|used=" ///
        e(threads_used)
}

local TEST_CUDA : environment XHDFE_THREADS_TEST_CUDA
if ("`TEST_CUDA'" == "1") {
    foreach threads in 1 2 8 16 24 48 {
        quietly xhdfe y x1 x2, absorb(fe1 fe2) vce(cluster fe1) keepsingletons ///
            tolerance(1e-10) tolerancemode(reghdfe-comparable) ///
            numthreads(`threads') gpubackend(cuda)
        assert e(converged) == 1
        assert e(threads_requested) == `threads'
        assert e(threads_effective) == `threads'
        assert e(threads_used) == `threads'
        assert e(gpu_used) == 1
        assert "`e(gpu_backend)'" == "cuda"
        assert "`e(gpu_status)'" == "used"
        mata: st_numscalar("threads_b_diff", ///
            max(abs(st_matrix("e(b)") :- st_matrix("b_ref"))))
        mata: st_numscalar("threads_V_diff", ///
            max(abs(st_matrix("e(V)") :- st_matrix("V_ref"))))
        assert scalar(threads_b_diff) <= 1e-9
        assert scalar(threads_V_diff) <= 1e-9
        display as result "XHDFE_THREADS_CONTRACT|cuda|requested=`threads'|used=" ///
            e(threads_used)
    }
}

* Automatic team (numthreads(0)) at the row-count boundaries of
* HdfeRegressorV11::resolve_threads, compared with one thread on the same data.
* Rows per thread: kAutoRowsPerThreadAbsorb in the canonical core, or
* XHDFE_AUTO_ROWS_PER_THREAD when it holds a positive integer. Run the file once
* more in a fresh Stata with XHDFE_AUTO_ROWS_PER_THREAD=500000 to check the
* override.
local OMP_NUM_THREADS : environment OMP_NUM_THREADS
if ("`OMP_NUM_THREADS'" != "") {
    display as error "unset OMP_NUM_THREADS: the automatic team starts from the default OpenMP team"
    exit 198
}

mata:
string scalar auto_rows_per_thread(string scalar path)
{
    string colvector lines
    real scalar i

    lines = cat(path)
    for (i = 1; i <= rows(lines); i++) {
        if (regexm(lines[i], "kAutoRowsPerThreadAbsorb = ([0-9]+);")) {
            return(regexs(1))
        }
    }
    _error(3498, "kAutoRowsPerThreadAbsorb not found in " + path)
    return("")
}
end

local auto_rpt : environment XHDFE_AUTO_ROWS_PER_THREAD
if (!regexm("`auto_rpt'", "^[0-9]+$")) local auto_rpt
else if (real("`auto_rpt'") < 1 | real("`auto_rpt'") > 2147483647) local auto_rpt
if ("`auto_rpt'" == "") {
    mata: st_local("auto_rpt", auto_rows_per_thread("`ROOT'/src/hdfe_regressor_v11.cpp"))
}

* With XHDFE_PARALLEL_READ=0 the plugin reads serially, e(threads_used) is the
* fit's own team and the automatic checks keep the bound they had before the
* parallel read; otherwise the sample selection may scan the whole range.
local parallel_read : environment XHDFE_PARALLEL_READ
local range_bound = ("`parallel_read'" != "0")

* Team resolve_threads selects for numthreads(0) before the runtime capacity;
* capped: fixed effects absorbed without being saved.
program define automatic_team, rclass
    args rows capped rows_per_thread
    local team = max(1, floor(`rows' / `rows_per_thread'))
    if (`capped') {
        if (`rows' < 100000) local team 1
        else if (`rows' < 250000) local team = min(`team', 8)
        else local team = min(`team', 16)
    }
    if (`rows' < 20000) local team 1
    return scalar team = `team'
end

program define check_automatic_fit
    syntax, rows(integer) team(integer) rangeteam(integer) fit(string asis)
    local expected = min(`team', e(thread_capacity))
    * The parallel input read selects the sample on the whole data range, with
    * the team the same rule gives for that range, and e(threads_used) counts it.
    local range_expected = min(`rangeteam', e(thread_capacity))
    assert e(N) == `rows'
    assert e(converged) == 1
    assert e(threads_requested) == 0
    assert e(threads_effective) == `expected'
    * The absorber may narrow an automatic team by fixed-effect shape.
    assert inrange(e(threads_used), min(`expected', 2), max(`expected', `range_expected'))
    assert e(gpu_used) == 0
    display as result "XHDFE_THREADS_CONTRACT|auto|rows=`rows'|expected=`expected'" ///
        "|effective=" e(threads_effective) "|used=" e(threads_used) ///
        "|workers=" e(parallel_workers_active) "|capacity=" e(thread_capacity) ///
        "|reason=`e(thread_limit_reason)'"
    if (`expected' > 1) {
        matrix auto_b = e(b)
        matrix auto_V = e(V)
        scalar auto_iterations = e(iterations)
        quietly `fit' numthreads(1)
        mata: st_numscalar("auto_b_diff", ///
            max(abs(st_matrix("e(b)") :- st_matrix("auto_b"))))
        mata: st_numscalar("auto_V_diff", ///
            max(abs(st_matrix("e(V)") :- st_matrix("auto_V"))))
        assert scalar(auto_b_diff) <= 1e-13
        assert scalar(auto_V_diff) <= 1e-13
        assert e(iterations) == scalar(auto_iterations)
    }
end

drop _all
set obs 1000000
generate long row = _n - 1
generate int fe1 = mod(row, 1200)
generate int fe2 = mod(floor(row / 1200), 500)
generate double x1 = sin(row * .001) + mod(row * 17, 101) / 101
generate double x2 = cos(row * .0013) + mod(row * 29, 103) / 103
generate double y = .6*x1 - .25*x2 + .001*fe1 - .0005*fe2 + sin(row * .0021)
local auto_opts "vce(cluster fe1) keepsingletons tolerance(1e-10) tolerancemode(reghdfe-comparable) gpubackend(cpu)"

* Largest sizes first, so a team carried over from the previous call could not
* push the next fit below its expected team; the last check tests carry-over.
foreach rows in 1000000 600000 400000 250000 249999 100000 99999 {
    automatic_team `rows' 1 `auto_rpt'
    local team = r(team)
    automatic_team `=_N' 1 `auto_rpt'
    local range_team = cond(`range_bound', r(team), `team')
    local fit "xhdfe y x1 x2 in 1/`rows', absorb(fe1 fe2) `auto_opts'"
    quietly `fit'
    check_automatic_fit, rows(`rows') team(`team') rangeteam(`range_team') fit(`fit')
}

* Saved fixed effects keep one thread per 500,000 rows.
automatic_team 1000000 0 500000
local team = r(team)
local fit "xhdfe y x1 x2, absorb(fe1 fe2, savefe) `auto_opts'"
quietly `fit'
check_automatic_fit, rows(1000000) team(`team') rangeteam(`team') fit(`fit')
drop __hdfe*

* An explicit density restores the 2.30.1 rule.
automatic_team 600000 1 500000
local team = r(team)
automatic_team `=_N' 1 500000
local range_team = cond(`range_bound', r(team), `team')
local fit "xhdfe y x1 x2 in 1/600000, absorb(fe1 fe2) `auto_opts' targetrowsperthread(500000)"
quietly `fit'
check_automatic_fit, rows(600000) team(`team') rangeteam(`range_team') fit(`fit')

* After smaller teams: the automatic team must not depend on the previous call.
automatic_team 1000000 1 `auto_rpt'
local team = r(team)
local fit "xhdfe y x1 x2, absorb(fe1 fe2) `auto_opts'"
quietly `fit'
check_automatic_fit, rows(1000000) team(`team') rangeteam(`team') fit(`fit')

display as result "PASS: Stata automatic thread rule (rows per thread `auto_rpt')"
display as result "PASS: Stata explicit thread contract"
