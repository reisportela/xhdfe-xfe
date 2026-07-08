* Run the public xhdfe core-23 replication benchmarks (Stata front-end).
*
* Mirrors registry.json (Stata cannot parse JSON): each entry below is
* "name|file|depvar|regressors|absorb|cluster". Sergio datasets are the .dta
* downloaded by get_sergio_data.sh; the simulated/pyfixest datasets need the
* .dta copies written by the generators' --stata flag.
*
* Environment knobs (same semantics as run_python.py):
*   SPECS=credit,credit2   comma list of spec names (default: all available)
*   REPS=3                 timed repetitions per spec (default 1)
*   MODE=comparable|fast   tolerance mode (default comparable)
*   GPU=1                  request the CUDA backend (fail-closed)
*   OTHERS=1               also time reghdfe on each spec, if installed
*
* Run from this directory:  stata-mp -b do run_stata.do
* Requires xhdfe on the adopath (net install xhdfe, from(...) replace).

version 16
clear all
set more off

local here "`c(pwd)'"
capture mkdir "`here'/output"

local specs_env : env SPECS
local reps_env : env REPS
local mode_env : env MODE
local gpu_env : env GPU
local others_env : env OTHERS

local reps = cond("`reps_env'" == "", 1, real("`reps_env'"))
local mode = cond("`mode_env'" == "", "comparable", "`mode_env'")
local tolmode = cond("`mode'" == "fast", "xhdfe-fast", "reghdfe-comparable")
local gpuopt ""
local gpuflag 0
if ("`gpu_env'" != "" & "`gpu_env'" != "0") {
    local gpuopt "gpubackend(cuda)"
    local gpuflag 1
}

* name|file|depvar|regressors|absorb|cluster  (mirror of registry.json)
local all_specs ///
    "credit|data/sergio/credit.dta|y|x1 x2|id1 id2|id1" ///
    "credit2|data/sergio/credit2.dta|y|x1 x2|id1 id2|id1" ///
    "directors|data/sergio/directors.dta|y|x1 x2|id1 id2|id1" ///
    "enron|data/sergio/enron.dta|y|x1 x2|id1 id2|id1" ///
    "github|data/sergio/github.dta|y|x1 x2|id1 id2 id3|id1" ///
    "patents|data/sergio/patents.dta|y|x1 x2|id1 id2|id1" ///
    "schools|data/sergio/schools.dta|y|x1 x2|id1 id2 id3|id1" ///
    "soccer|data/sergio/soccer.dta|y|x1 x2|id1 id2|id1" ///
    "synthetic-assortative|data/sergio/synthetic-assortative.dta|y|x1 x2|id1 id2 year|id1" ///
    "synthetic-complete|data/sergio/synthetic-complete.dta|y|x1 x2|id1 id2|id1" ///
    "synthetic-uniform-easy|data/sergio/synthetic-uniform-easy.dta|y|x1 x2|id1 id2|id1" ///
    "synthetic-uniform-hard|data/sergio/synthetic-uniform-hard.dta|y|x1 x2|id1 id2|id1" ///
    "synthetic-uniform-harder|data/sergio/synthetic-uniform-harder.dta|y|x1 x2|id1 id2|id1" ///
    "synthetic-zigzag|data/sergio/synthetic-zigzag.dta|y|x1 x2|id1 id2|id1" ///
    "workers|data/sergio/workers.dta|y|x1 x2|id1 id2 id3 id4|id1" ///
    "simulated_panel|data/simulated_panel.dta|ln_wage|education experience experience_sq union|worker_id firm_id occupation_id year|worker_id" ///
    "pf_difficult_10m_3fe|data/benchmark_difficult_n10000000_k10.dta|y|x1 x2 x3 x4 x5 x6 x7 x8 x9 x10|indiv_id firm_id year|indiv_id" ///
    "pf_difficult_10m_2fe|data/benchmark_difficult_n10000000_k10.dta|y|x1 x2 x3 x4 x5 x6 x7 x8 x9 x10|indiv_id year|indiv_id" ///
    "pf_simple_10m_3fe|data/benchmark_simple_n10000000_k10.dta|y|x1 x2 x3 x4 x5 x6 x7 x8 x9 x10|indiv_id firm_id year|indiv_id" ///
    "pf_simple_10m_2fe|data/benchmark_simple_n10000000_k10.dta|y|x1 x2 x3 x4 x5 x6 x7 x8 x9 x10|indiv_id year|indiv_id"

* output CSVs (append; header only when new)
local runs_csv "`here'/output/stata_runs.csv"
local coef_csv "`here'/output/stata_coefficients.csv"
capture confirm file "`runs_csv'"
if (_rc) {
    file open rh using "`runs_csv'", write text replace
    file write rh "spec,engine,mode,gpu,rep,elapsed_seconds,n_obs,converged" _n
    file close rh
}
capture confirm file "`coef_csv'"
if (_rc) {
    file open ch using "`coef_csv'", write text replace
    file write ch "spec,engine,mode,variable,coef,se" _n
    file close ch
}

foreach entry of local all_specs {
    tokenize `"`entry'"', parse("|")
    local name "`1'"
    local dfile "`3'"
    local depvar "`5'"
    local xs "`7'"
    local fes "`9'"
    local clvar "`11'"

    if ("`specs_env'" != "" & strpos(",`specs_env',", ",`name',") == 0) {
        continue
    }
    capture confirm file "`here'/`dfile'"
    if (_rc) {
        di as txt "skip `name': `dfile' not found (download/generate it first)"
        continue
    }

    di as txt _n "=== `name' ==="
    use "`here'/`dfile'", clear

    forvalues rep = 1/`reps' {
        timer clear 1
        timer on 1
        xhdfe `depvar' `xs', absorb(`fes') vce(cluster `clvar') ///
            tolerancemode(`tolmode') `gpuopt'
        timer off 1
        quietly timer list 1
        local secs = r(t1)
        file open rh using "`runs_csv'", write text append
        file write rh "`name',xhdfe_stata,`mode',`gpuflag',`rep',`secs',`=e(N)',`=e(converged)'" _n
        file close rh
        di as txt "  xhdfe rep`rep': " %8.2f `secs' "s"
        if (`rep' == 1) {
            file open ch using "`coef_csv'", write text append
            foreach v of local xs {
                file write ch "`name',xhdfe_stata,`mode',`v',`=_b[`v']',`=_se[`v']'" _n
            }
            file close ch
        }
    }

    if ("`others_env'" != "" & "`others_env'" != "0") {
        capture which reghdfe
        if (!_rc) {
            forvalues rep = 1/`reps' {
                timer clear 2
                timer on 2
                reghdfe `depvar' `xs', absorb(`fes') vce(cluster `clvar')
                timer off 2
                quietly timer list 2
                local secs = r(t2)
                file open rh using "`runs_csv'", write text append
                file write rh "`name',reghdfe,`mode',0,`rep',`secs',`=e(N)'," _n
                file close rh
                di as txt "  reghdfe rep`rep': " %8.2f `secs' "s"
                if (`rep' == 1) {
                    file open ch using "`coef_csv'", write text append
                    foreach v of local xs {
                        file write ch "`name',reghdfe,`mode',`v',`=_b[`v']',`=_se[`v']'" _n
                    }
                    file close ch
                }
            }
        }
    }
}

di as txt _n "done -> `runs_csv' / `coef_csv'"
exit
