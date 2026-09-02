clear all
set more off

di as result "STATA_ID version=" c(stata_version) " flavor=" c(flavor) " os=" c(os) " machine=" c(machine_type)

foreach df in 1 2 10 100 49347 112742 118119 148356 206623 {
    scalar critical = invttail(`df', 0.025)
    di as result "STATA_CRITICAL df=`df' value=" %24.17g critical
}

foreach tail in 0.0000005 1e-12 1e-50 1e-100 1e-300 {
    scalar cauchy = invttail(1, `tail')
    di as result "STATA_CAUCHY tail=`tail' value=" %24.17g cauchy
}

scalar public_level = 99.9999
scalar public_tail = (100 - public_level) / 200
scalar public_critical = invttail(1, public_tail)
di as result "STATA_PUBLIC_LEVEL level=" %24.17g public_level " tail=" %24.17g public_tail " value=" %24.17g public_critical

local names patents schools uniform_easy uniform_hard uniform_harder
local dfs 49347 206623 148356 112742 118119
local ts 0.50916929119588794 1.473485721671121 0.93501225146834288 0.0074594128213933608 1.9594791142984223
forvalues i = 1/5 {
    local name : word `i' of `names'
    local df : word `i' of `dfs'
    local t : word `i' of `ts'
    scalar probability = 2 * ttail(`df', `t')
    di as result "STATA_P name=`name' df=`df' t=" %24.17g `t' " value=" %24.17g probability
}

exit, clear
