noi di as text "xhdfe certification: transactional lifecycle"

clear
set obs 240
set seed 20260828
gen long row = _n - 1
gen int fe1 = mod(row, 17)
gen int fe2 = mod(row, 13)
gen double x = rnormal()
gen double y = .7*x + .1*fe1 - .05*fe2 + rnormal()

program define lifecycle_assert_empty
    local cmd "`e(cmd)'"
    if ("`cmd'" != "") {
        di as error "expected empty e(), found e(cmd)=`cmd'"
        exit 9
    }
end

quietly regress y x
assert "`e(cmd)'" == "regress"
capture noisily xhdfe y x, absorb(fe1 fe2) lifecycle_bad_option
assert _rc == 198
lifecycle_assert_empty

quietly xhdfe y x, absorb(fe1 fe2) keepsingletons numthreads(1)
assert "`e(cmd)'" == "xhdfe"
quietly xhdfe
assert "`e(cmd)'" == "xhdfe"

local old_xhdfe_path "$XHDFE_PLUGIN_PATH_INTERNAL"
global XHDFE_PLUGIN_PATH_INTERNAL "/tmp/xhdfe-lifecycle-stale.plugin"
capture noisily xhdfe y x, absorb(fe1 fe2) keepsingletons numthreads(1)
assert _rc == 498
lifecycle_assert_empty
global XHDFE_PLUGIN_PATH_INTERNAL "`old_xhdfe_path'"

capture noisily xhdfe y x, absorb(fe1 fe2) keepsingletons numthreads(1) ///
    gpubackend(metal)
assert _rc != 0
lifecycle_assert_empty

quietly xhdfe y x, absorb(fe1 fe2) keepsingletons numthreads(1)
assert "`e(cmd)'" == "xhdfe"

capture drop lc_*
quietly xfepout y x, absorb(fe1 fe2) generate(lc_) sample(lc_sample) ///
    keepsingletons numthreads(1)
assert "`e(cmd)'" == "xfepout"

capture noisily xfepout y x, absorb(fe1 fe2) lifecycle_bad_option
assert _rc == 198
lifecycle_assert_empty

local old_xfepout_path "$XFEPOUT_PLUGIN_PATH_INTERNAL"
global XFEPOUT_PLUGIN_PATH_INTERNAL "/tmp/xfepout-lifecycle-stale.plugin"
capture noisily xfepout y x, absorb(fe1 fe2) generate(lc2_) sample(lc2_sample) ///
    keepsingletons numthreads(1)
assert _rc == 498
lifecycle_assert_empty
global XFEPOUT_PLUGIN_PATH_INTERNAL "`old_xfepout_path'"

capture noisily xfepout y x, absorb(fe1 fe2) generate(lc3_) sample(lc3_sample) ///
    keepsingletons numthreads(1) gpubackend(metal)
assert _rc != 0
lifecycle_assert_empty

quietly xfepout y x, absorb(fe1 fe2) generate(lc4_) sample(lc4_sample) ///
    keepsingletons numthreads(1)
assert "`e(cmd)'" == "xfepout"
quietly xfepout
assert "`e(cmd)'" == "xfepout"

capture program drop lifecycle_assert_empty
