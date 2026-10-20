* Adapted from reghdfe/test/part1/cluster-after-expand.do (MIT, Sergio Correia).
* Replication preserves b and the unscaled sandwich. N, RSS and the declared
* finite-sample corrections change; upstream intentionally flags full equality.
version 16
clear
sysuse auto, clear
bysort turn: generate t = _n
tsset turn t
drop if missing(rep78)
generate long original_row = _n
quietly regress price weight length i.turn
local design_rank = e(rank)
local original_N = e(N)
local hc1 = `original_N'/(`original_N'-`design_rank')

quietly _up_xhdfe price weight length, absorb(turn) keepsingletons vce(robust)
assert e(N) == `original_N'
matrix original_b = e(b)
matrix original_V = e(V)
scalar original_rss = e(rss)

expand 50
quietly _up_xhdfe price weight length, absorb(turn) keepsingletons vce(cluster original_row)
assert e(N) == 50*`original_N'
assert e(N_clust) == `original_N'
assert reldif(e(rss), 50*original_rss) <= 1e-9
assert mreldif(e(b), original_b) <= 1e-9
local expanded_N = 50*`original_N'
local cluster_scale = (`expanded_N'-1)/(`expanded_N'-`design_rank') * `original_N'/(`original_N'-1)
matrix expected_V = original_V * (`cluster_scale'/`hc1')
matrix actual_V = e(V)
forvalues i=1/`=rowsof(expected_V)' {
    forvalues j=1/`=colsof(expected_V)' {
        assert abs(actual_V[`i',`j']-expected_V[`i',`j']) <= ///
            1e-8*sqrt(expected_V[`i',`i'])*sqrt(expected_V[`j',`j'])
    }
}
display "CORE_EXPECTED_DIFFERENCE_CONFIRMED expansion_N_RSS_SSC"
