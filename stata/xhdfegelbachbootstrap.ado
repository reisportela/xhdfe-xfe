*! version 1.0.0  25jul2026
*! Full pairs/cluster-pairs bootstrap for xhdfegelbach point functionals.

program define xhdfegelbachbootstrap, rclass sortpreserve
    version 14.0
    syntax varname(numeric) [if] [in] [aweight fweight /], ///
        x1(varlist numeric) METHOD(string) ///
        [ x2groups(string) fes(varlist numeric) COMMONFES(varlist numeric) ///
          vce(string) cluster(varname numeric) ///
          gamma0 cov0 ABSorbedtargets(varlist numeric) tol(real 1e-8) ///
          threads(integer 0) GPU VERBose FOCAL(varlist numeric) ///
          CONNected(string) CONNectivityfes(varlist numeric min=2 max=2) ///
          BOOTCLuster(varname numeric) REPS(integer 999) SEED(integer 0) ///
          MINValid(integer 0) BOOTCI(string) Level(cilevel) ///
          REQUIREGPU SHARETOL(real 1e-12) ]

    local y `varlist'
    local method = lower(trim("`method'"))
    if (!inlist("`method'", "pairs", "cluster_pairs")) {
        di as err "method() must be pairs or cluster_pairs"
        exit 198
    }
    if ("`method'" == "pairs" & "`bootcluster'" != "") {
        di as err "bootcluster() is only valid with method(cluster_pairs)"
        exit 198
    }
    if ("`method'" == "cluster_pairs" & "`bootcluster'" == "") {
        di as err "method(cluster_pairs) requires bootcluster(); the resampling unit is never inferred from vce(cluster)"
        exit 198
    }
    if (`reps' < 2) {
        di as err "reps() must be at least 2"
        exit 198
    }
    if (`seed' < 0) {
        di as err "seed() must be nonnegative"
        exit 198
    }
    if (`minvalid' == 0) local minvalid = ceil(0.9 * `reps')
    if (`minvalid' < 2 | `minvalid' > `reps') {
        di as err "minvalid() must lie between 2 and reps()"
        exit 198
    }
    local bootci = lower(trim("`bootci'"))
    if ("`bootci'" == "") local bootci "percentile"
    if (!inlist("`bootci'", "percentile", "basic")) {
        di as err "bootci() must be percentile or basic"
        exit 198
    }
    if (`sharetol' < 0 | missing(`sharetol')) {
        di as err "sharetol() must be finite and nonnegative"
        exit 198
    }
    if ("`requiregpu'" != "" & "`gpu'" == "") {
        di as err "requiregpu requires the gpu option"
        exit 198
    }
    if ("`weight'" == "fweight") {
        di as err "frequency-weight pairs bootstrap is not exposed because record-level and expanded-sample resampling are different estimands; expand the data explicitly before bootstrapping"
        exit 198
    }
    if ("`vce'" == "") local vce "unadjusted"
    if (!inlist("`vce'", "unadjusted", "robust", "cluster")) {
        di as err "vce() must be unadjusted, robust or cluster"
        exit 198
    }
    if ("`vce'" == "cluster" & "`cluster'" == "") {
        di as err "vce(cluster) requires cluster()"
        exit 198
    }
    if ("`vce'" != "cluster" & "`cluster'" != "") {
        di as err "cluster() requires vce(cluster)"
        exit 198
    }
    if (`threads' < 0) {
        di as err "threads() must be nonnegative"
        exit 198
    }
    local connected = lower(trim("`connected'"))
    if ("`connected'" == "") local connected "diagnose"

    * Recover the x2 varlist for one common retained sample.
    local x2vars
    if ("`x2groups'" != "") {
        tokenize "`x2groups'", parse(":")
        while ("`1'" != "") {
            if ("`1'" != ":") {
                gettoken __gname __rest : 1, parse("=")
                gettoken __eq __gvars : __rest, parse("=")
                local __gvars = trim("`__gvars'")
                if ("`__gvars'" == "") {
                    di as err "each x2groups() block must be name = varlist"
                    exit 198
                }
                unab __gvars : `__gvars'
                local x2vars "`x2vars' `__gvars'"
            }
            macro shift
        }
    }

    marksample touse
    markout `touse' `y' `x1' `x2vars' `commonfes' `fes' ///
        `cluster' `bootcluster'
    tempvar boot_weight
    local point_weight
    if ("`weight'" != "") {
        quietly gen double `boot_weight' = `exp' if `touse'
        markout `touse' `boot_weight'
        local point_weight "[aweight=`boot_weight']"
    }
    quietly count if `touse'
    if (r(N) == 0) error 2000
    if ("`method'" == "cluster_pairs") {
        quietly levelsof `bootcluster' if `touse', local(__boot_levels)
        local __n_boot_clusters : word count `__boot_levels'
        if (`__n_boot_clusters' < 2) {
            di as err "cluster-pairs bootstrap requires at least two clusters"
            exit 198
        }
    }

    local point_options `"x1(`x1')"'
    if ("`x2groups'" != "") ///
        local point_options `"`point_options' x2groups(`"`x2groups'"')"'
    if ("`commonfes'" != "") ///
        local point_options "`point_options' commonfes(`commonfes')"
    if ("`fes'" != "") local point_options "`point_options' fes(`fes')"
    local point_options "`point_options' vce(`vce') tol(`tol') threads(`threads')"
    if ("`cluster'" != "") ///
        local point_options "`point_options' cluster(`cluster')"
    if ("`gamma0'" != "") local point_options "`point_options' gamma0"
    if ("`cov0'" != "") local point_options "`point_options' cov0"
    if ("`absorbedtargets'" != "") ///
        local point_options "`point_options' absorbedtargets(`absorbedtargets')"
    if ("`focal'" != "") ///
        local point_options "`point_options' focal(`focal')"
    if ("`gpu'" != "") local point_options "`point_options' gpu"
    if ("`verbose'" != "") local point_options "`point_options' verbose"
    local point_options "`point_options' connected(`connected')"
    if ("`connectivityfes'" != "") ///
        local point_options "`point_options' connectivityfes(`connectivityfes')"

    quietly xhdfegelbach `y' if `touse' `point_weight', `point_options'
    if (r(converged) != 1) {
        di as err "the point decomposition did not converge; bootstrap aborted"
        exit 430
    }
    if ("`requiregpu'" != "" & r(gpu_used) != 1) {
        di as err "GPU use was required but the point estimate used a fallback"
        exit 459
    }

    tempname POINTD POINTSE POINTT POINTB POINTF
    tempname POINTCOV POINTTCOV POINTBASECOV POINTCOVDB POINTCOVTB
    matrix `POINTD' = r(delta)
    matrix `POINTSE' = r(se)
    matrix `POINTT' = r(total)
    matrix `POINTB' = r(b_base)
    matrix `POINTF' = r(b_full)
    matrix `POINTCOV' = r(cov)
    matrix `POINTTCOV' = r(total_cov)
    matrix `POINTBASECOV' = r(base_cov)
    matrix `POINTCOVDB' = r(cov_delta_bbase)
    matrix `POINTCOVTB' = r(cov_total_bbase)
    local groups "`r(groups)'"
    local x1_names "`r(x1_names)'"
    local focal_names "`r(focal_names)'"
    local identity_status "`r(identity_status)'"
    local point_vce "`r(vce)'"
    local point_notes "`r(notes)'"
    local point_gpu_used = r(gpu_used)
    local point_gpu_backend "`r(gpu_backend)'"
    local point_gpu_status "`r(gpu_status)'"
    local p = colsof(`POINTB')
    local k1 = rowsof(`POINTD')
    local G = colsof(`POINTD')
    local kd = `k1' * `G'

    tempname DRAWD DRAWT DRAWB DRAWF LEDGER
    tempname PDELTA PTOTAL PBASE PFULL
    matrix `DRAWD' = J(`reps', `kd', .)
    matrix `DRAWT' = J(`reps', `k1', .)
    matrix `DRAWB' = J(`reps', `p', .)
    matrix `DRAWF' = J(`reps', `p', .)
    matrix `LEDGER' = J(`reps', 7, .)
    matrix `PDELTA' = J(1, `kd', .)
    matrix `PTOTAL' = J(1, `k1', .)
    matrix `PBASE' = `POINTB'
    matrix `PFULL' = `POINTF'

    local flatnames
    local coefnames "`x1_names' _cons"
    local position 0
    forvalues g = 1/`G' {
        local gname : word `g' of `groups'
        forvalues r = 1/`k1' {
            local ++position
            local xname : word `r' of `coefnames'
            local flatnames "`flatnames' `gname':`xname'"
            matrix `PDELTA'[1, `position'] = `POINTD'[`r', `g']
        }
    }
    forvalues r = 1/`k1' {
        matrix `PTOTAL'[1, `r'] = `POINTT'[`r', 1]
    }
    matrix colnames `DRAWD' = `flatnames'
    matrix colnames `PDELTA' = `flatnames'
    matrix colnames `DRAWT' = `x1_names' _cons
    matrix colnames `PTOTAL' = `x1_names' _cons
    matrix colnames `DRAWB' = `x1_names'
    matrix colnames `DRAWF' = `x1_names'
    matrix colnames `PBASE' = `x1_names'
    matrix colnames `PFULL' = `x1_names'
    matrix colnames `LEDGER' = ///
        valid rc n_rows n_obs n_singletons_dropped identity_gap gpu_used
    local repnames
    forvalues b = 1/`reps' {
        local repnames "`repnames' b`b'"
    }
    foreach M in DRAWD DRAWT DRAWB DRAWF LEDGER {
        matrix rownames ``M'' = `repnames'
    }

    local replicate_options `"x1(`x1')"'
    if ("`x2groups'" != "") ///
        local replicate_options `"`replicate_options' x2groups(`"`x2groups'"')"'
    if ("`commonfes'" != "") ///
        local replicate_options "`replicate_options' commonfes(`commonfes')"
    if ("`fes'" != "") ///
        local replicate_options "`replicate_options' fes(`fes')"
    local replicate_options "`replicate_options' vce(unadjusted) tol(`tol') threads(`threads')"
    if ("`gamma0'" != "") ///
        local replicate_options "`replicate_options' gamma0"
    if ("`cov0'" != "") local replicate_options "`replicate_options' cov0"
    if ("`absorbedtargets'" != "") ///
        local replicate_options "`replicate_options' absorbedtargets(`absorbedtargets')"
    if ("`focal'" != "") ///
        local replicate_options "`replicate_options' focal(`focal')"
    if ("`gpu'" != "") local replicate_options "`replicate_options' gpu"
    local replicate_options "`replicate_options' connected(`connected')"
    if ("`connectivityfes'" != "") ///
        local replicate_options "`replicate_options' connectivityfes(`connectivityfes')"

    local rngstate "`c(rngstate)'"
    set seed `seed'
    preserve
    quietly keep if `touse'
    tempfile __xgel_boot_base
    quietly save `__xgel_boot_base'
    local valid 0
    tempname REPD REPT REPB REPF
    forvalues b = 1/`reps' {
        quietly use `__xgel_boot_base', clear
        if ("`method'" == "pairs") {
            capture quietly bsample
        }
        else {
            capture quietly bsample, cluster(`bootcluster')
        }
        local rc = _rc
        matrix `LEDGER'[`b', 2] = `rc'
        matrix `LEDGER'[`b', 3] = _N
        if (`rc' == 0) {
            capture quietly xhdfegelbach `y' `point_weight', ///
                `replicate_options'
            local rc = _rc
        }
        if (`rc' == 0 & r(converged) != 1) local rc 430
        if (`rc' == 0 & "`requiregpu'" != "" & ///
            r(gpu_used) != 1) local rc 459
        if (`rc' == 0) {
            matrix `REPD' = r(delta)
            matrix `REPT' = r(total)
            matrix `REPB' = r(b_base)
            matrix `REPF' = r(b_full)
            if (rowsof(`REPD') != `k1' | colsof(`REPD') != `G') ///
                local rc 459
            forvalues r = 1/`p' {
                if (`rc' == 0 & missing(`REPT'[`r', 1])) local rc 459
                if (`rc' == 0 & missing(`REPB'[1, `r'])) local rc 459
                if (`rc' == 0 & missing(`REPF'[1, `r'])) local rc 459
                forvalues g = 1/`G' {
                    if (`rc' == 0 & missing(`REPD'[`r', `g'])) local rc 459
                }
            }
        }
        matrix `LEDGER'[`b', 2] = `rc'
        if (`rc' == 0) {
            local ++valid
            matrix `LEDGER'[`b', 1] = 1
            matrix `LEDGER'[`b', 4] = r(n_obs)
            matrix `LEDGER'[`b', 5] = r(n_singletons_dropped)
            matrix `LEDGER'[`b', 6] = r(identity_gap)
            matrix `LEDGER'[`b', 7] = r(gpu_used)
            local position 0
            forvalues g = 1/`G' {
                forvalues r = 1/`k1' {
                    local ++position
                    matrix `DRAWD'[`b', `position'] = `REPD'[`r', `g']
                }
            }
            forvalues r = 1/`k1' {
                matrix `DRAWT'[`b', `r'] = `REPT'[`r', 1]
            }
            forvalues r = 1/`p' {
                matrix `DRAWB'[`b', `r'] = `REPB'[1, `r']
                matrix `DRAWF'[`b', `r'] = `REPF'[1, `r']
            }
        }
        else matrix `LEDGER'[`b', 1] = 0
    }
    restore
    quietly set rngstate `rngstate'

    local gpu_all_valid 0
    if ("`gpu'" != "") {
        local gpu_all_valid 1
        forvalues b = 1/`reps' {
            if (`LEDGER'[`b', 1] == 1 & `LEDGER'[`b', 7] != 1) {
                local gpu_all_valid 0
            }
        }
    }

    if (`valid' < `minvalid') {
        di as err "Gelbach bootstrap failed closed: `valid'/`reps' valid replications, fewer than minvalid(`minvalid')"
        di as err "Inspect failure return codes by rerunning with a lower temporary minvalid() only for diagnosis."
        exit 498
    }
    if (`valid' < `reps') {
        di as err "warning: retained `valid'/`reps' valid replications; inspect r(bootstrap_ledger)"
    }

    tempname DRAWSB DRAWSM DRAWFS DRAWTS PSB PSM PFS PTS
    mata: xgel_boot_shares_20260725( ///
        "`DRAWD'", "`DRAWT'", "`DRAWB'", "`DRAWF'", ///
        "`PDELTA'", "`PTOTAL'", "`PBASE'", "`PFULL'", ///
        "`DRAWSB'", "`DRAWSM'", "`DRAWFS'", "`DRAWTS'", ///
        "`PSB'", "`PSM'", "`PFS'", "`PTS'", `p', `k1', `sharetol')
    matrix colnames `DRAWSB' = `flatnames'
    matrix colnames `DRAWSM' = `flatnames'
    matrix colnames `DRAWFS' = `x1_names'
    matrix colnames `DRAWTS' = `x1_names'
    matrix colnames `PSB' = `flatnames'
    matrix colnames `PSM' = `flatnames'
    matrix colnames `PFS' = `x1_names'
    matrix colnames `PTS' = `x1_names'
    matrix rownames `DRAWSB' = `repnames'
    matrix rownames `DRAWSM' = `repnames'
    matrix rownames `DRAWFS' = `repnames'
    matrix rownames `DRAWTS' = `repnames'

    tempname CID CIT CIB CIF CISB CISM CIFS CITS
    mata: xgel_boot_ci_20260725( ///
        "`DRAWD'", "`PDELTA'", "`CID'", `level' / 100, "`bootci'")
    mata: xgel_boot_ci_20260725( ///
        "`DRAWT'", "`PTOTAL'", "`CIT'", `level' / 100, "`bootci'")
    mata: xgel_boot_ci_20260725( ///
        "`DRAWB'", "`PBASE'", "`CIB'", `level' / 100, "`bootci'")
    mata: xgel_boot_ci_20260725( ///
        "`DRAWF'", "`PFULL'", "`CIF'", `level' / 100, "`bootci'")
    mata: xgel_boot_ci_20260725( ///
        "`DRAWSB'", "`PSB'", "`CISB'", `level' / 100, "`bootci'")
    mata: xgel_boot_ci_20260725( ///
        "`DRAWSM'", "`PSM'", "`CISM'", `level' / 100, "`bootci'")
    mata: xgel_boot_ci_20260725( ///
        "`DRAWFS'", "`PFS'", "`CIFS'", `level' / 100, "`bootci'")
    mata: xgel_boot_ci_20260725( ///
        "`DRAWTS'", "`PTS'", "`CITS'", `level' / 100, "`bootci'")
    foreach M in CID CIT CIB CIF CISB CISM CIFS CITS {
        matrix rownames ``M'' = low high se valid_count
    }
    matrix colnames `CID' = `flatnames'
    matrix colnames `CIT' = `x1_names' _cons
    matrix colnames `CIB' = `x1_names'
    matrix colnames `CIF' = `x1_names'
    matrix colnames `CISB' = `flatnames'
    matrix colnames `CISM' = `flatnames'
    matrix colnames `CIFS' = `x1_names'
    matrix colnames `CITS' = `x1_names'

    di as txt _n "{hline 78}"
    di as res "Gelbach `method' bootstrap"
    di as txt "{hline 78}"
    di as txt "Valid replications: " as res "`valid'/`reps'" ///
        as txt "    Seed: " as res "`seed'"
    di as txt "Interval: " as res "`bootci' (`level'%)"
    if ("`method'" == "cluster_pairs") {
        di as txt "Resampling unit: " as res "`bootcluster' (explicit)"
    }
    else di as txt "Resampling unit: " as res "observation"
    di as txt "Each valid replication fully re-estimates base, full, auxiliaries, and added FE contributions."
    di as txt "Bootstrap intervals do not by themselves cure product nonregularity."
    di as txt "{hline 78}"

    return scalar reps_requested = `reps'
    return scalar reps_valid = `valid'
    return scalar reps_failed = `reps' - `valid'
    return scalar min_valid_reps = `minvalid'
    return scalar seed = `seed'
    return scalar conf_level = `level' / 100
    return scalar share_tol = `sharetol'
    return scalar gpu_requested = ("`gpu'" != "")
    return scalar gpu_required = ("`requiregpu'" != "")
    return scalar gpu_used_point = `point_gpu_used'
    return scalar gpu_used_all_valid = `gpu_all_valid'
    return local method "`method'"
    local resampling_unit "declared_cluster"
    if ("`method'" == "pairs") local resampling_unit "observation"
    return local resampling_unit "`resampling_unit'"
    return local bootstrap_cluster "`bootcluster'"
    return local ci_method "`bootci'"
    return local interval_status ///
        "resampling_based_not_a_nonregularity_cure"
    return local version "1.0.0 25jul2026"
    return local rng "Stata_rng_reproducible_seed"
    return local point_vce "`point_vce'"
    return local point_gpu_backend "`point_gpu_backend'"
    return local point_gpu_status "`point_gpu_status'"
    return local replicate_vce "unadjusted_point_functional_only"
    return local identity_status "`identity_status'"
    return local groups "`groups'"
    return local x1_names "`x1_names'"
    return local focal_names "`focal_names'"
    if ("`point_notes'" != "") return local notes "`point_notes'"
    return matrix bootstrap_total_share_base_ci = `CITS'
    return matrix bootstrap_full_share_base_ci = `CIFS'
    return matrix bootstrap_share_movement_ci = `CISM'
    return matrix bootstrap_share_base_ci = `CISB'
    return matrix bootstrap_full_ci = `CIF'
    return matrix bootstrap_base_ci = `CIB'
    return matrix bootstrap_total_ci = `CIT'
    return matrix bootstrap_delta_ci = `CID'
    return matrix bootstrap_total_share_base_draws = `DRAWTS'
    return matrix bootstrap_full_share_base_draws = `DRAWFS'
    return matrix bootstrap_share_movement_draws = `DRAWSM'
    return matrix bootstrap_share_base_draws = `DRAWSB'
    return matrix bootstrap_full_draws = `DRAWF'
    return matrix bootstrap_base_draws = `DRAWB'
    return matrix bootstrap_total_draws = `DRAWT'
    return matrix bootstrap_delta_draws = `DRAWD'
    return matrix bootstrap_ledger = `LEDGER'
    return matrix b_full = `POINTF'
    return matrix b_base = `POINTB'
    return matrix total = `POINTT'
    return matrix se = `POINTSE'
    return matrix delta = `POINTD'
    return matrix cov_delta_bbase = `POINTCOVDB'
    return matrix cov_total_bbase = `POINTCOVTB'
    return matrix base_cov = `POINTBASECOV'
    return matrix total_cov = `POINTTCOV'
    return matrix cov = `POINTCOV'
end

capture mata: mata drop xgel_boot_q_20260725()
capture mata: mata drop xgel_boot_ci_20260725()
capture mata: mata drop xgel_boot_shares_20260725()
mata:
real scalar xgel_boot_q_20260725(
    real colvector values,
    real scalar probability)
{
    real scalar n, h, lower, upper, fraction
    values = select(values, values :< .)
    n = rows(values)
    if (n == 0) return(.)
    values = sort(values, 1)
    if (n == 1) return(values[1])
    h = 1 + (n - 1) * probability
    lower = floor(h)
    upper = ceil(h)
    fraction = h - lower
    return(values[lower] + fraction * (values[upper] - values[lower]))
}

void xgel_boot_ci_20260725(
    string scalar draws_name,
    string scalar point_name,
    string scalar output_name,
    real scalar conf_level,
    string scalar method)
{
    real matrix draws, output
    real rowvector point
    real scalar column, alpha, qlow, qhigh, n
    real colvector values
    draws = st_matrix(draws_name)
    point = st_matrix(point_name)
    output = J(4, cols(draws), .)
    alpha = 1 - conf_level
    for (column = 1; column <= cols(draws); column++) {
        values = select(draws[, column], draws[, column] :< .)
        n = rows(values)
        output[4, column] = n
        if (n == 0 | point[column] >= .) continue
        qlow = xgel_boot_q_20260725(
            values, alpha / 2)
        qhigh = xgel_boot_q_20260725(
            values, 1 - alpha / 2)
        if (method == "percentile") {
            output[1, column] = qlow
            output[2, column] = qhigh
        }
        else {
            output[1, column] = 2 * point[column] - qhigh
            output[2, column] = 2 * point[column] - qlow
        }
        if (n > 1) {
            output[3, column] =
                sqrt(sum((values :- mean(values)):^2) / (n - 1))
        }
    }
    st_matrix(output_name, output)
}

void xgel_boot_shares_20260725(
    string scalar delta_name,
    string scalar total_name,
    string scalar base_name,
    string scalar full_name,
    string scalar point_delta_name,
    string scalar point_total_name,
    string scalar point_base_name,
    string scalar point_full_name,
    string scalar share_base_name,
    string scalar share_movement_name,
    string scalar full_share_name,
    string scalar total_share_name,
    string scalar point_share_base_name,
    string scalar point_share_movement_name,
    string scalar point_full_share_name,
    string scalar point_total_share_name,
    real scalar p,
    real scalar k1,
    real scalar tolerance)
{
    real matrix delta, total, base, full, share_base, share_movement
    real matrix point_delta, point_total, point_base, point_full
    real matrix full_share, total_share
    real matrix point_share_base, point_share_movement
    real matrix point_full_share, point_total_share
    real scalar replication, column, row, denominator
    delta = st_matrix(delta_name)
    total = st_matrix(total_name)
    base = st_matrix(base_name)
    full = st_matrix(full_name)
    point_delta = st_matrix(point_delta_name)
    point_total = st_matrix(point_total_name)
    point_base = st_matrix(point_base_name)
    point_full = st_matrix(point_full_name)
    share_base = J(rows(delta), cols(delta), .)
    share_movement = J(rows(delta), cols(delta), .)
    full_share = J(rows(full), cols(full), .)
    total_share = J(rows(full), cols(full), .)
    point_share_base = J(1, cols(point_delta), .)
    point_share_movement = J(1, cols(point_delta), .)
    point_full_share = J(1, cols(point_full), .)
    point_total_share = J(1, cols(point_full), .)
    for (replication = 1; replication <= rows(delta); replication++) {
        for (column = 1; column <= cols(delta); column++) {
            row = mod(column - 1, k1) + 1
            if (row <= p) {
                denominator = base[replication, row]
                if (denominator < . & abs(denominator) > tolerance) {
                    share_base[replication, column] =
                        delta[replication, column] / denominator
                }
            }
            denominator = total[replication, row]
            if (denominator < . & abs(denominator) > tolerance) {
                share_movement[replication, column] =
                    delta[replication, column] / denominator
            }
        }
        for (row = 1; row <= p; row++) {
            denominator = base[replication, row]
            if (denominator < . & abs(denominator) > tolerance) {
                full_share[replication, row] =
                    full[replication, row] / denominator
                total_share[replication, row] =
                    total[replication, row] / denominator
            }
        }
    }
    for (column = 1; column <= cols(point_delta); column++) {
        row = mod(column - 1, k1) + 1
        if (row <= p) {
            denominator = point_base[1, row]
            if (denominator < . & abs(denominator) > tolerance) {
                point_share_base[1, column] =
                    point_delta[1, column] / denominator
            }
        }
        denominator = point_total[1, row]
        if (denominator < . & abs(denominator) > tolerance) {
            point_share_movement[1, column] =
                point_delta[1, column] / denominator
        }
    }
    for (row = 1; row <= p; row++) {
        denominator = point_base[1, row]
        if (denominator < . & abs(denominator) > tolerance) {
            point_full_share[1, row] =
                point_full[1, row] / denominator
            point_total_share[1, row] =
                point_total[1, row] / denominator
        }
    }
    st_matrix(share_base_name, share_base)
    st_matrix(share_movement_name, share_movement)
    st_matrix(full_share_name, full_share)
    st_matrix(total_share_name, total_share)
    st_matrix(point_share_base_name, point_share_base)
    st_matrix(point_share_movement_name, point_share_movement)
    st_matrix(point_full_share_name, point_full_share)
    st_matrix(point_total_share_name, point_total_share)
}
end
