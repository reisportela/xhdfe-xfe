* Categorical interactions and abbreviations in genuine group/individual data.
version 16
clear all
set more off
set varabbrev on
args backend
if "`backend'" == "" local backend cpu
local package : env XHDFE_STATA_ADOPATH
if `"`package'"' != "" adopath ++ `"`package'"'

set seed 19102026
set obs 600
generate long group_id = _n
generate byte area_code = 1 + mod(_n-1,3)
generate byte sector_code = 1 + mod(floor((_n-1)/3),4)
generate byte cohort_code = 1 + mod(floor((_n-1)/12),5)
generate double x = rnormal()
generate double z = rnormal()
generate double noise = rnormal()/10
expand 3
bysort group_id: generate byte slot = _n
generate long person = 1 + mod(17*group_id+7*slot,97)
generate double person_signal = sin(person/7)
bysort group_id: egen double group_signal = total(person_signal)
generate double y = .6*x-.2*z+group_signal/5+area_code*sector_code*cohort_code/30+noise
replace sector_code = . if group_id == 17
egen long joint = group(area_code sector_code cohort_code)

local opts "group(group_id) individual(person) numthreads(2) gpubackend(`backend')"
foreach aggregation in sum mean {
    quietly xhdfe y x z, absorb(joint person) aggregation(`aggregation') `opts'
    matrix bref = e(b)
    matrix Vref = e(V)
    local nref = e(N)
    local dfref = e(df_a)
    generate byte reference_sample = e(sample)
    foreach term in "area_code#sector_code#cohort_code" "are#sec#coh" "ib2.are#bn.sec#ibn.coh" {
        quietly xhdfe y x z, absorb(`term' person) aggregation(`aggregation') `opts'
        assert mreldif(bref,e(b)) < 1e-10
        assert mreldif(Vref,e(V)) < 1e-10
        assert e(N) == `nref' & e(df_a) == `dfref'
        assert e(sample) == reference_sample
        assert e(converged) == 1
        assert e(gpu_used) == ("`backend'" == "cuda")
        if "`backend'" == "cuda" {
            assert "`e(gpu_backend)'" == "cuda"
            assert "`e(gpu_status)'" == "used"
        }
        display "GROUP_INTERACTION_CASE_PASS `backend' `aggregation' `term'"
    }
    drop reference_sample
}
display "GROUP_INTERACTION_CONTRACT_PASS `backend'"
