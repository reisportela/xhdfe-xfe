version 16.0
clear all
set more off

local pkg : environment XHDFE_STATA_ADOPATH
if (`"`pkg'"' == "") local pkg "/home/mangelo/Documents/GitHub/xhdfe/stata"
local plugin `"`pkg'/xhdfe.plugin"'

set obs 50000
gen long worker = ceil(_n / 2)
gen long firm = cond(mod(_n, 2) == 1, worker, mod(worker, 25000) + 1)
gen double y = sin(_n)
gen double keep = .

capture program drop __xakm_bounded_note_probe
program define __xakm_bounded_note_probe, rclass
    version 16.0
    args plugin_path
    capture program __xhdfe_bounded_note_plugin, plugin using(`"`plugin_path'"')
    if (_rc & _rc != 110) exit _rc

    local cfg "cfg=task=akm_kss;ncontrols=0;store_effects=0;has_fweight=0;"
    local cfg "`cfg'leave_out_level=match;leverages=jla;jla_draws=2;seed=1;prune=1;"
    local cfg "`cfg'exact_max_rows=0;direct_max_firms=0;cg_tol=1e-30;cg_max_iter=1;"
    local cfg "`cfg'num_threads=4;"

    plugin call __xhdfe_bounded_note_plugin y worker firm keep, "`cfg'"
    local note_len : length local xakm_notes
    assert scalar(__xakm_converged) == 0
    assert `note_len' < 4096
    assert strpos(`"`xakm_notes'"', "affected working rows:") > 0
    di as err "  details: `xakm_notes'"

    // Exercise the same returned-local expansion used by xhdfeakm.ado.
    return local notes `"`xakm_notes'"'
    return scalar note_length = `note_len'
end

__xakm_bounded_note_probe `"`plugin'"'
assert r(note_length) < 4096
assert strpos(`"`r(notes)'"', "affected working rows:") > 0
di as result "AKM_BOUNDED_NONCONVERGENCE_NOTES_SMOKE_OK"
