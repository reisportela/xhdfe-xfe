*! Absorbed-target allocation: a worker-invariant descriptive group gap.

version 16
clear
set more off
set seed 20260719

local n_workers 200
local periods 5
set obs `=`n_workers' * `periods''
gen long worker = ceil(_n / `periods')
bysort worker: gen byte group = runiform() > 0.5 if _n == 1
bysort worker: replace group = group[1]
bysort worker: gen double experience = _n - 1 + rnormal(0, 0.2)
gen double job = 0.3 * group + 0.2 * experience + rnormal()
bysort worker: gen double worker_pay = rnormal() if _n == 1
bysort worker: replace worker_pay = worker_pay[1]
gen double y = 0.25 * group + 0.08 * experience + 0.5 * job + ///
    worker_pay + rnormal()

* group is absorbed by worker FE. Its full coefficient is imposed at zero;
* this is neither an estimated within-worker coefficient nor causal mediation.
xhdfegelbach y, x1(group experience) ///
    x2groups("job_covariate = job") fes(worker) ///
    absorbedtargets(group) vce(cluster) cluster(worker)

assert "`r(estimand)'" == "absorbed_target_allocation"
assert "`r(absorbed_targets)'" == "0"
assert "`r(absorbed_target_names)'" == "group"
assert "`r(b_full_status)'" == "imposed_zero estimated"
assert "`r(inference_status)'" == "clustered_at_absorbing_fe"
matrix b_full = r(b_full)
assert b_full[1, 1] == 0
