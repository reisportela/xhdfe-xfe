// Stable explicit OLS basis for the patent chain; no iterative oracle.
run "fixtures/toy-patents-chain.do"
local backend : env XHDFE_GPU_BACKEND

foreach nodes in 64 128 {
    toy_dificil `nodes'
    bysort patent_id: generate byte team_size = _N
    tempfile long
    save `long'
    foreach aggregation in sum mean {
        use `long', clear
        preserve
        bysort patent_id: keep if _n == 1
        generate int pattern = floor((patent_id - 1)/2) + 1
        local patterns = 4*`nodes'
        local last = `patterns' - 1
        local reference_z = cond("`aggregation'" == "mean", 6, 1)
        // The sole row-pattern restriction is z'FE=0. For mean, z=(7,-7,-6,6).
        forvalues j = 1/`last' {
            local type = mod(`j'-1,4)
            local zj = cond(inlist(`type',0,3),1,-1)
            if "`aggregation'" == "mean" local zj = `zj'*cond(`type'<2,7,6)
            generate double gfe`j' = (pattern==`j') - (`zj'/`reference_z')*(pattern==`patterns')
        }
        quietly regress citations funding lab_size gfe*, noconstant
        scalar gi_b1 = _b[funding]
        scalar gi_b2 = _b[lab_size]
        matrix gi_V = e(V)
        scalar gi_rss = e(rss)
        scalar gi_df = e(df_r)
        assert gi_df == 4*`nodes'-1
        predict double oracle_residual, residuals
        quietly summarize oracle_residual
        scalar gi_sd = r(sd)
        keep patent_id oracle_residual
        tempfile reference
        save `reference'
        restore
        merge m:1 patent_id using `reference', assert(match) nogen

        foreach order in "inventor_id year" "year inventor_id" {
            quietly xhdfe citations funding lab_size, absorb(`order') ///
                group(patent_id) individual(inventor_id) aggregation(`aggregation') ///
                keepsingletons dofadjustments(exact) tolerancemode(strict-residual) ///
                tolerance(1e-12) numthreads(2) residuals(residual)
            assert e(converged)==1 & e(precision_certified)==1 & e(df_r)==gi_df
            assert abs(_b[funding]-gi_b1)<1e-8 & abs(_b[lab_size]-gi_b2)<1e-8
            assert abs(e(rss)-gi_rss)<1e-9
            assert abs(_se[funding]^2-gi_V[1,1])<1e-10
            assert abs(_se[lab_size]^2-gi_V[2,2])<1e-10
            if "`backend'" == "cuda" {
                assert e(gpu_used)==1 & "`e(gpu_backend)'"=="cuda" & "`e(gpu_status)'"=="used"
            }
            generate double difference = abs(residual-oracle_residual)/gi_sd if e(sample)
            quietly summarize difference, meanonly
            assert r(max)<1e-7 & r(N)==8*`nodes'
            bysort patent_id: egen byte residual_count = count(residual)
            assert residual_count==1
            drop residual difference residual_count
        }
    }
}
display "GROUP_INDIVIDUAL_ANALYTIC_64_128_PASS"
