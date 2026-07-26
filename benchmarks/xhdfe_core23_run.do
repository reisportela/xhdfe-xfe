* xhdfe_core23_run.do — standalone xhdfe (Stata plugin) runner for the core23
* encompassing-benchmark group: the 22 core specs plus akm_v02_secondreg
* (Tiago's Table-7 AKM second regression with firm-specific heterogeneous
* seniority slopes: NPC_FIC##c.firm_seniority_obs + NPC_FIC#c.firm_seniority_spline10,
* two-way cluster idtrab NPC_FIC).
*
* Specs mirror benchmarks/encompassing/registry.json (group core23) exactly.
* Reference run: _akm_opt/encompassing_core23_xhdfe_20260701/ (REPS=3 medians).
*
* Usage:
*   stata-mp -q -b do benchmarks/xhdfe_core23_run.do
*   CORE23_OUT=/path/to/out bash benchmarks/run_xhdfe_core23_with_sentinel.sh
* The wrapper is preferred for performance runs: it writes non-gating host
* provenance JSON before and after this unchanged Stata measurement runner.
*
* Env overrides (all optional):
*   XHDFE_REPO_ROOT  repo root (default /home/mangelo/Documents/GitHub/xhdfe)
*   CORE23_BACKENDS  "cpu cuda" (default) | "cpu" | "cuda"
*   CORE23_MODES     "fast comparable" (default) | "fast" | "comparable"
*   CORE23_REPS      repetitions per cell (default 1; the 1jul table used 3)
*   CORE23_THREADS   numthreads() (default 16)
*   CORE23_ONLY      space-separated dataset names to subset (default: all 23)
*   CORE23_OUT       output dir (default benchmarks/_out/xhdfe_core23_stata)
*
* Full default sweep (cpu+cuda x fast+comparable x 1 rep) takes roughly an
* hour on the shared 48-core server, dominated by akm_v02_secondreg
* (~55.9M rows; Stata cpu comparable median was ~925s under load on 1jul).
* Requires the pq package for parquet loading.

version 18.0
clear all
set more off
set rmsg off

* ---------------------------------------------------------------- config ----
local ROOT "/home/mangelo/Documents/GitHub/xhdfe"
local ev : environment XHDFE_REPO_ROOT
if ("`ev'" != "") local ROOT "`ev'"
adopath ++ "`ROOT'/stata"

local BACKENDS "cpu cuda"
local ev : environment CORE23_BACKENDS
if ("`ev'" != "") local BACKENDS "`ev'"

local MODES "fast comparable"
local ev : environment CORE23_MODES
if ("`ev'" != "") local MODES "`ev'"

local REPS 1
local ev : environment CORE23_REPS
if ("`ev'" != "") local REPS `ev'

local THREADS 16
local ev : environment CORE23_THREADS
if ("`ev'" != "") local THREADS `ev'

local ONLY ""
local ev : environment CORE23_ONLY
if ("`ev'" != "") local ONLY "`ev'"

local OUT "`ROOT'/benchmarks/_out/xhdfe_core23_stata"
local ev : environment CORE23_OUT
if ("`ev'" != "") local OUT "`ev'"
capture mkdir "`ROOT'/benchmarks/_out"
capture mkdir "`OUT'"

* ------------------------------------------- registry path_vars (1:1) ------
local SERGIO "/home/mangelo/Documents/BigData/Sergio/all-dta"
local READY  "/home/mangelo/Documents/BigData/Paulo/main_95_21_ready.parquet"
local SIM    "/home/mangelo/Documents/BigData/Simulated_Data/simulated_panel.parquet"
local AKM    "`ROOT'/benchmarks/_out/akm_v02_firstreg_full_fast_comparable_5x_20260619_163305/akm_v02_firstreg_common.parquet"
local AKM2   "/srv/projetos/P008/04_to_test_xhdfe/data/coauthor_base/derived/akm_v02_r_julia_full/akm_v02_r_julia.parquet"
local PF     "/srv/projetos/P008/dataset/pyfixest"

* --------------------------------------------------------- core23 specs ----
* abs`i' holds the full absorb() contents, including heterogeneous-slope
* terms (fe##c.var keeps the FE intercept, fe#c.var is slope-only).
local i = 0

local ++i
local nm`i'  "credit"
local pth`i' "`SERGIO'/credit.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2"
local cl`i'  "id1"

local ++i
local nm`i'  "credit2"
local pth`i' "`SERGIO'/credit2.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2"
local cl`i'  "id1"

local ++i
local nm`i'  "directors"
local pth`i' "`SERGIO'/directors.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2"
local cl`i'  "id1"

local ++i
local nm`i'  "enron"
local pth`i' "`SERGIO'/enron.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2"
local cl`i'  "id1"

local ++i
local nm`i'  "github"
local pth`i' "`SERGIO'/github.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2 id3"
local cl`i'  "id1"

local ++i
local nm`i'  "patents"
local pth`i' "`SERGIO'/patents.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2"
local cl`i'  "id1"

local ++i
local nm`i'  "schools"
local pth`i' "`SERGIO'/schools.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2 id3"
local cl`i'  "id1"

local ++i
local nm`i'  "soccer"
local pth`i' "`SERGIO'/soccer.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2"
local cl`i'  "id1"

local ++i
local nm`i'  "synthetic-assortative"
local pth`i' "`SERGIO'/synthetic-assortative.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2 year"
local cl`i'  "id1"

local ++i
local nm`i'  "synthetic-complete"
local pth`i' "`SERGIO'/synthetic-complete.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2"
local cl`i'  "id1"

local ++i
local nm`i'  "synthetic-uniform-easy"
local pth`i' "`SERGIO'/synthetic-uniform-easy.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2"
local cl`i'  "id1"

local ++i
local nm`i'  "synthetic-uniform-hard"
local pth`i' "`SERGIO'/synthetic-uniform-hard.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2"
local cl`i'  "id1"

local ++i
local nm`i'  "synthetic-uniform-harder"
local pth`i' "`SERGIO'/synthetic-uniform-harder.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2"
local cl`i'  "id1"

local ++i
local nm`i'  "synthetic-zigzag"
local pth`i' "`SERGIO'/synthetic-zigzag.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2"
local cl`i'  "id1"

local ++i
local nm`i'  "workers"
local pth`i' "`SERGIO'/workers.parquet"
local y`i'   "y"
local x`i'   "x1 x2"
local abs`i' "id1 id2 id3 id4"
local cl`i'  "id1"

local ++i
local nm`i'  "main_95_21_ready"
local pth`i' "`READY'"
local y`i'   "log_rhwage"
local x`i'   "age_sq tenure tenure_sq educ"
local abs`i' "w_id f_id year"
local cl`i'  "w_id"

local ++i
local nm`i'  "simulated_panel"
local pth`i' "`SIM'"
local y`i'   "ln_wage"
local x`i'   "education experience experience_sq union"
local abs`i' "worker_id firm_id occupation_id year"
local cl`i'  "worker_id"

local ++i
local nm`i'  "akm_v02_firstreg"
local pth`i' "`AKM'"
local y`i'   "ln_wgain_hour_wz"
local x`i'   "exp_akm exp_akm_sq_100 exp_akm_cu_1000 exp_akm_qt_10000 firm_seniority_obs firm_seniority_spline10"
local abs`i' "idtrab ano NPC_FIC"
local cl`i'  "idtrab"

local ++i
local nm`i'  "pf_difficult_10m_3fe"
local pth`i' "`PF'/benchmark_difficult_n10000000_k10.parquet"
local y`i'   "y"
local x`i'   "x1 x2 x3 x4 x5 x6 x7 x8 x9 x10"
local abs`i' "indiv_id firm_id year"
local cl`i'  "indiv_id"

local ++i
local nm`i'  "pf_simple_10m_2fe"
local pth`i' "`PF'/benchmark_simple_n10000000_k10.parquet"
local y`i'   "y"
local x`i'   "x1 x2 x3 x4 x5 x6 x7 x8 x9 x10"
local abs`i' "indiv_id year"
local cl`i'  "indiv_id"

local ++i
local nm`i'  "pf_simple_10m_3fe"
local pth`i' "`PF'/benchmark_simple_n10000000_k10.parquet"
local y`i'   "y"
local x`i'   "x1 x2 x3 x4 x5 x6 x7 x8 x9 x10"
local abs`i' "indiv_id firm_id year"
local cl`i'  "indiv_id"

local ++i
local nm`i'  "pf_difficult_10m_2fe"
local pth`i' "`PF'/benchmark_difficult_n10000000_k10.parquet"
local y`i'   "y"
local x`i'   "x1 x2 x3 x4 x5 x6 x7 x8 x9 x10"
local abs`i' "indiv_id year"
local cl`i'  "indiv_id"

* Tiago AKM second regression (Table 7): firm-specific seniority slopes.
* Canonical command per registry.json — firm_seniority_obs carries the firm
* intercept ('##'), firm_seniority_spline10 is slope-only ('#').
local ++i
local nm`i'  "akm_v02_secondreg"
local pth`i' "`AKM2'"
local y`i'   "ln_wgain_hour_wz"
local x`i'   "exp_akm exp_akm_sq_100 exp_akm_cu_1000 exp_akm_qt_10000"
local abs`i' "idtrab ano NPC_FIC##c.firm_seniority_obs NPC_FIC#c.firm_seniority_spline10"
local cl`i'  "idtrab NPC_FIC"

local NDS = `i'

* ------------------------------------------------------------ run sweep ----
di as text "core23 xhdfe runner: backends=[`BACKENDS'] modes=[`MODES'] reps=`REPS' threads=`THREADS'"
if ("`ONLY'" != "") di as text "subset: `ONLY'"
di as text "output: `OUT'"

tempname PO
postfile `PO' str40 dataset str32 engine str8 backend str12 mode int run ///
    double elapsed_seconds int success double gpu_used double iterations ///
    double nobs str200 note using "`OUT'/runs.dta", replace
tempname CO
postfile `CO' str40 dataset str32 engine str32 term double estimate ///
    double std_error using "`OUT'/coefficients.dta", replace

local nfallback = 0
forvalues i = 1/`NDS' {
    if ("`ONLY'" != "" & strpos(" `ONLY' ", " `nm`i'' ") == 0) continue
    di as text _n "===== core23 [`i'/`NDS'] `nm`i'' ====="
    capture noisily pq use using "`pth`i''", clear
    if (_rc) {
        local lrc = _rc
        foreach bk of local BACKENDS {
            foreach md of local MODES {
                forvalues r = 1/`REPS' {
                    post `PO' ("`nm`i''") ("xhdfe_stata_`bk'_`md'") ("`bk'") ("`md'") (`r') (.) (0) (.) (.) (.) ("load_rc=`lrc'")
                }
            }
        }
        continue
    }
    foreach bk of local BACKENDS {
        foreach md of local MODES {
            local eng = "xhdfe_stata_`bk'_`md'"
            local tolmode = cond("`md'"=="fast", "xhdfe-fast", "reghdfe-comparable")
            forvalues r = 1/`REPS' {
                timer clear 1
                timer on 1
                capture noisily xhdfe `y`i'' `x`i'', absorb(`abs`i'') vce(cluster `cl`i'') ///
                    gpubackend(`bk') tolerancemode(`tolmode') numthreads(`THREADS')
                local rc = _rc
                timer off 1
                quietly timer list 1
                local el = r(t1)
                local gu = .
                capture local gu = e(gpu_used)
                local it = .
                capture local it = e(iterations)
                local nob = .
                capture local nob = e(N)
                local note "rc=`rc'"
                if ("`bk'" == "cuda" & `rc' == 0 & `gu' != 1) {
                    local note "rc=`rc' SILENT_CPU_FALLBACK gpu_used=`gu'"
                    local ++nfallback
                    di as error "WARNING: `nm`i'' cuda run `r' did not use the GPU (e(gpu_used)=`gu')"
                }
                post `PO' ("`nm`i''") ("`eng'") ("`bk'") ("`md'") (`r') (`el') (`=(`rc'==0)') (`gu') (`it') (`nob') ("`note'")
                if (`rc' == 0 & `r' == 1) {
                    foreach t in `x`i'' {
                        capture post `CO' ("`nm`i''") ("`eng'") ("`t'") (_b[`t']) (_se[`t'])
                    }
                }
                di as result "CORE23_RESULT dataset=`nm`i'' engine=`eng' run=`r' rc=`rc' elapsed=`el' gpu_used=`gu' iterations=`it'"
            }
        }
    }
}
postclose `PO'
postclose `CO'

* -------------------------------------------------------------- summary ----
use "`OUT'/runs.dta", clear
export delimited using "`OUT'/runs.csv", replace
quietly count if success == 0
local nfail = r(N)
capture use "`OUT'/coefficients.dta", clear
capture export delimited using "`OUT'/coefficients.csv", replace

use "`OUT'/runs.dta", clear
keep if success == 1
if (_N > 0) {
    collapse (median) elapsed_seconds (min) success, by(dataset backend mode)
    gen cell = backend + "_" + mode
    keep dataset cell elapsed_seconds
    reshape wide elapsed_seconds, i(dataset) j(cell) string
    rename elapsed_seconds* *
    capture format %9.2f cpu_*
    capture format %9.2f cuda_*
    di as text _n "core23 median elapsed seconds by cell:"
    list, noobs abbreviate(24)
}
di as text _n "CORE23_DONE datasets=`NDS' failed_runs=`nfail' silent_gpu_fallbacks=`nfallback'"
di as text "runs: `OUT'/runs.csv"
di as text "coefficients: `OUT'/coefficients.csv"
