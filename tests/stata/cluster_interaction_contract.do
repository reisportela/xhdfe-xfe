* Multi-component categorical cluster() and xfepout interactions: grouped-ID references.
version 16
clear all
set more off
set linesize 160
args backend
if ("`backend'" == "") local backend cpu
local adopath : env XHDFE_STATA_ADOPATH
if (`"`adopath'"' != "") adopath ++ `"`adopath'"'
which xhdfe
which xfepout
which reghdfe

set seed 16102026
set obs 6000
generate int a = 1 + mod(_n-1,7)
generate int b = 1 + mod(floor((_n-1)/7),5)
generate int c = 1 + mod(floor((_n-1)/35),4)
generate int firm = 1 + mod(floor((_n-1)/12),50)
generate int time = 1 + mod(_n,11)
generate double x1 = rnormal()
generate double x2 = rnormal()
generate double y = .7*x1 - .3*x2 + firm/40 + a*b*c/90 + rnormal()
replace b = . in 11/12
generate str6 sa = string(a)
generate str6 sb = cond(missing(b), "", string(b))
generate str6 sc = string(c)
egen long abc = group(a b c)
local opts "numthreads(2) gpubackend(`backend') tolerance(1e-12) tolerancemode(reghdfe-comparable)"

* One-way cluster on a three-component interaction equals the grouped ID.
quietly xhdfe y x1 x2, absorb(firm time) vce(cluster abc) `opts'
matrix b0 = e(b)
matrix V0 = e(V)
local N0 = e(N)
local nc0 = e(N_clust)
generate byte s0 = e(sample)
foreach term in "a#b#c" "i.a#i.b#i.c" "ib2.a#bn.b#ibn.c" "c#b#a" "sa#sb#sc" {
    quietly xhdfe y x1 x2, absorb(firm time) vce(cluster `term') `opts'
    display "CLUSTER_PARITY `term' b=" mreldif(b0,e(b)) " V=" mreldif(V0,e(V))
    assert mreldif(b0,e(b)) < 1e-12 & mreldif(V0,e(V)) < 1e-12
    assert e(N) == `N0' & e(N_clust) == `nc0'
    assert e(sample) == s0
    * Factor prefixes are stripped from the canonical cluster label, as for two components.
    local shown "`term'"
    foreach prefix in "ibn." "ib2." "bn." "i." {
        local shown : subinstr local shown "`prefix'" "", all
    }
    assert "`e(clustvar)'" == "`shown'"
    assert e(gpu_used) == ("`backend'" == "cuda")
    display "CLUSTER_INTERACTION_CASE_PASS `backend' `term'"
}
* The documented factor prefixes also parse in absorb() interactions (ib#. and b#. were
* rejected with r(101) up to 2.29.1 because of a double-escaped regex).
quietly xhdfe y x1 x2, absorb(abc time) vce(cluster abc) `opts'
matrix bp = e(b)
quietly xhdfe y x1 x2, absorb(ib2.a#bn.b#ibn.c time) vce(cluster b2.a#c#i.b) `opts'
assert mreldif(bp,e(b)) < 1e-12 & e(N) == `N0'
assert "`e(absvars)'" == "a#b#c time" & "`e(clustvar)'" == "a#c#b"
display "CLUSTER_INTERACTION_CASE_PASS `backend' prefixes"
* reghdfe comparator on the grouped identifier; D-C constant policy differs only for multiway.
quietly reghdfe y x1 x2, absorb(firm time) vce(cluster abc) tolerance(1e-12)
assert e(N) == `N0' & e(N_clust) == `nc0'
foreach v in x1 x2 {
    display "REFERENCE_PRECISION `v' b=" abs(_b[`v']-b0[1,"`v'"]) " se=" abs(_se[`v']-sqrt(V0["`v'","`v'"]))
    assert abs(_b[`v']-b0[1,"`v'"]) < 1e-9*max(1,abs(_b[`v']))
    assert abs(_se[`v']-sqrt(V0["`v'","`v'"])) < 1e-7*max(1,abs(_se[`v']))
}

* Multiway clustering with an interaction component.
quietly xhdfe y x1 x2, absorb(firm) vce(cluster abc time) `opts'
matrix b1 = e(b)
matrix V1 = e(V)
local ncv1 = e(N_clustervars)
quietly xhdfe y x1 x2, absorb(firm) vce(cluster a#b#c time) `opts'
assert mreldif(b1,e(b)) < 1e-12 & mreldif(V1,e(V)) < 1e-12
assert e(N_clustervars) == `ncv1'
assert "`e(clustvar)'" == "a#b#c time"
display "CLUSTER_INTERACTION_CASE_PASS `backend' multiway"

* Two-component ## keeps its documented expansion to main effects plus interaction.
quietly xhdfe y x1 x2, absorb(firm) vce(cluster a##c) `opts'
assert "`e(clustvar)'" == "a c a#c"
assert e(N_clustervars) == 3
display "CLUSTER_INTERACTION_CASE_PASS `backend' double-hash"

* Negative controls: ## beyond two components and continuous components are refused.
capture noisily xhdfe y x1 x2, absorb(firm) vce(cluster a##b##c) numthreads(1)
assert _rc == 198
assert "`e(cmd)'" == ""
capture noisily xhdfe y x1 x2, absorb(firm) vce(cluster a#b#c.x1) numthreads(1)
assert _rc == 198
assert "`e(cmd)'" == ""

* xfepout: absorb() and clustervars() interactions equal the grouped identifiers.
quietly xfepout y x1 x2, absorb(abc time) clustervars(abc) generate(g_) sample(g_s) ///
    numthreads(2) gpubackend(`backend') tolerance(1e-12)
local gN = e(N)
local gdfa = e(df_a)
quietly xfepout y x1 x2, absorb(a#b#c time) clustervars(a#b#c) generate(k_) sample(k_s) ///
    numthreads(2) gpubackend(`backend') tolerance(1e-12)
assert e(N) == `gN' & e(df_a) == `gdfa'
assert g_s == k_s
foreach v in y x1 x2 {
    quietly count if abs(g_`v' - k_`v') > 1e-10*max(1,abs(g_`v')) & !missing(g_`v')
    assert r(N) == 0
    quietly count if missing(g_`v') != missing(k_`v')
    assert r(N) == 0
}
display "CLUSTER_INTERACTION_CASE_PASS `backend' xfepout"
* String components: the reference groups the same strings, because xfepout's
* sample marking treats an empty string differently from a numeric missing.
egen long sabc = group(sa sb sc)
quietly xfepout y x1 x2, absorb(sabc time) generate(u_) sample(u_s) ///
    numthreads(2) gpubackend(`backend') tolerance(1e-12)
local uN = e(N)
local udfa = e(df_a)
quietly xfepout y x1 x2, absorb(sa#sb#sc time) generate(t_) sample(t_s) ///
    numthreads(2) gpubackend(`backend') tolerance(1e-12)
display "XFEPOUT_STRINGS N=" e(N) "/" `uN' " numeric-grouped N=" `gN'
assert e(N) == `uN' & e(df_a) == `udfa'
assert t_s == u_s
quietly count if abs(u_y - t_y) > 1e-10*max(1,abs(u_y)) & !missing(u_y)
assert r(N) == 0
display "CLUSTER_INTERACTION_CASE_PASS `backend' xfepout-strings"
quietly xfepout y x1 x2, absorb(ib2.a#bn.b#ibn.c time) clustervars(b2.a#c) generate(v_) sample(v_s) ///
    numthreads(2) gpubackend(`backend') tolerance(1e-12)
assert e(N) == `gN' & e(df_a) == `gdfa'
assert v_s == g_s
quietly count if abs(g_y - v_y) > 1e-10*max(1,abs(g_y)) & !missing(g_y)
assert r(N) == 0
display "CLUSTER_INTERACTION_CASE_PASS `backend' xfepout-prefixes"
capture noisily xfepout y x1, absorb(abc) clustervars(a##b##c) generate(z_) numthreads(1)
assert _rc == 198
assert "`e(cmd)'" == ""
capture noisily xfepout y x1, absorb(a#b#c.x1) generate(z_) numthreads(1)
assert _rc == 198
display "CLUSTER_INTERACTION_CONTRACT_PASS `backend'"
exit
