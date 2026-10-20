version 16
clear all
local task_root : env XHDFE_UPSTREAM_ROOT
include "`task_root'/env.do"
sysuse auto, clear
foreach option in "absorb(turn)" "a(turn)" "ab(turn)" "abs(turn)" {
    quietly _up_xhdfe price weight length, `option'
    assert e(N_hdfe) == 1
    assert e(gpu_used) == 1 & "`e(gpu_backend)'" == "cuda" & "`e(gpu_status)'" == "used"
}
quietly _up_xhdfe price weight length, noabsorb
assert strtrim(`"`e(absvars)'"') == "" & e(gpu_used) == 0
quietly _up_xhdfe price weight length
assert strtrim(`"`e(absvars)'"') == "" & e(gpu_used) == 0
display "CORE_ROUTING_CONTROLS_PASS"
