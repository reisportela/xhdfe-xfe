program define _up_xhdfe, eclass
    version 16
    local call `"`0'"'
    local backend : env XHDFE_UPSTREAM_BACKEND
    local equivalent : env XHDFE_UPSTREAM_EQUIVALENT
    local route "`backend'"
    local eligibility "fe_supported"
    if ("`backend'" == "cuda" & "`equivalent'" == "1") {
        syntax [anything] [if] [in] [aw fw pw iw] [, Absorb(string asis) *]
        if (strtrim(`"`absorb'"') == "") {
            local route cpu
            local eligibility "cpu_only_noabsorb_control"
        }
    }
    local delimiter
    if (strpos(`"`call'"', ",") == 0) local delimiter ","
    local requested_fe_tol : env XHDFE_UPSTREAM_FETOL
    local feoption
    if ("`requested_fe_tol'" != "") local feoption "fetolerance(`requested_fe_tol')"
    capture noisily xhdfe `call' `delimiter' gpubackend(`route') numthreads(2) `feoption'
    local result_rc = _rc
    tempname saved_r
    _return hold `saved_r'
    local actual_gpu = e(gpu_used)
    local threads = e(threads_used)
    local observed_backend "`e(gpu_backend)'"
    local gpu_status "`e(gpu_status)'"
    local nfe = e(N_hdfe)
    // N_hdfe includes the virtual constant in noabsorb/reghdfe-style output.
    // The canonical absorbed-variable list identifies actual absorption.
    local actual_absorbed = strtrim(`"`e(absvars)'"') != ""
    local backend_ok 1
    if (!`result_rc' & "`backend'" == "cuda") {
        if (missing(`nfe')) local backend_ok 0
        else if (`actual_absorbed' | "`route'" == "cuda") {
            local backend_ok = `actual_gpu' == 1 & "`observed_backend'" == "cuda" & "`gpu_status'" == "used"
        }
        else if (`actual_gpu' != 0 | "`eligibility'" != "cpu_only_noabsorb_control") local backend_ok 0
    }
    local receipt : env XHDFE_UPSTREAM_CALLS
    tempname stream
    file open `stream' using "`receipt'", write append text
    file write `stream' "rc=" (`result_rc') " requested=`backend' route=`route' eligibility=`eligibility' gpu=" (`actual_gpu') " backend=`observed_backend' status=`gpu_status' threads=" (`threads') " nfe=" (`nfe') " actual_absorbed=" (`actual_absorbed') " backend_ok=" (`backend_ok') _n
    file close `stream'
    _return restore `saved_r'
    assert `backend_ok' == 1
    exit `result_rc'
end
