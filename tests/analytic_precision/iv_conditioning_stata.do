version 16.0
clear all
set more off
args package_dir backend log_path control

// PACKAGE_DIR contains xhdfe.ado and xhdfe.plugin; use a fresh Stata process.
if (substr(`"`package_dir'"', 1, 1) != "/" | ///
    substr(`"`log_path'"', 1, 1) != "/" | ///
    !inlist("`backend'", "cpu", "cuda") | !inlist("`control'", "", "control")) {
    display as error "Expected absolute PACKAGE_DIR LOG_PATH, cpu|cuda, and optional control"
    exit 198
}
log using `"`log_path'"', text name(ivcond)
adopath ++ `"`package_dir'"'
which xhdfe
quietly findfile xhdfe.ado
if (`"`r(fn)'"' != `"`package_dir'/xhdfe.ado"') {
    display as error "The loaded xhdfe ado path differs from PACKAGE_DIR"
    exit 9
}

// Independent inference on the well-conditioned Walsh score space [w,t].
mata:
void ivcond_oracle(real scalar weighted, string scalar vce, real scalar fe_rank)
{
    real matrix Q, X, bread, meat, score, totals, Vref, V, scale
    real colvector weight, u, cluster, se
    real rowvector beta
    real scalar n, df, inference_df, rss, correction, g

    Q = st_data(., ("w", "t"))
    X = st_data(., ("w", "x"))
    u = st_data(., "e")
    n = rows(Q)
    weight = weighted ? st_data(., "aw") : J(n, 1, 1)
    weight = weight * (n / sum(weight))
    bread = luinv(quadcross(Q, weight, X))
    df = n - 2 - fe_rank
    inference_df = df
    rss = quadcross(u, weight, u)
    if (vce == "unadjusted") {
        Vref = (rss / df) * bread
    }
    else {
        score = Q :* ((weight :* u) * J(1, 2, 1))
        if (vce == "robust") {
            meat = quadcross(score, score)
            correction = n / df
        }
        else {
            cluster = st_data(., "cl")
            totals = J(7, 2, 0)
            for (g = 0; g < 7; g++) {
                totals[g+1, .] = colsum(select(score, cluster :== g))
            }
            meat = quadcross(totals, totals)
            correction = (7/6) * (n-1) / df
            inference_df = 6
        }
        Vref = correction * bread * meat * bread'
    }
    beta = st_matrix("e(b)")
    V = st_matrix("e(V)")
    assert(cols(beta) == 2 & rows(V) == 2 & cols(V) == 2)
    assert(max(abs(beta)) < . & max(abs(V)) < .)
    assert(min(diagonal(Vref)) > 0)
    se = sqrt(diagonal(Vref))
    scale = se * se'
    st_numscalar("ivcond_beta_error", max(abs(beta - (2,3)) :/ (2,3)))
    st_numscalar("ivcond_v_error", max(abs(V - Vref) :/ scale))
    st_numscalar("ivcond_se_error", max(abs(sqrt(diagonal(V)) - se) :/ se))
    st_numscalar("ivcond_rss_ref", rss)
    st_numscalar("ivcond_df_ref", inference_df)
}
end

program define ivcond_check
    syntax, Weighted(integer) VCE(string) FERank(integer) BACKEND(string)
    assert e(N) == _N
    assert e(sample)
    assert e(df_a) == `ferank'
    assert e(converged) == 1 & e(precision_certified) == 1
    assert e(gpu_used) == ("`backend'" == "cuda")
    if ("`backend'" == "cuda") {
        assert e(gpu_status_code) == 1
        assert "`e(gpu_backend)'" == "cuda"
    }
    mata: ivcond_oracle(`weighted', "`vce'", `ferank')
    assert scalar(ivcond_beta_error) <= 1e-9
    assert scalar(ivcond_v_error) <= 1e-8
    assert scalar(ivcond_se_error) <= 1e-8
    assert abs(r - e) <= 1e-7
    assert abs(e(rss) - scalar(ivcond_rss_ref)) <= 1e-8
    assert e(df_r) == scalar(ivcond_df_ref)
end

local n = cond("`backend'" == "cuda", 3072, 128)
set obs `n'
generate long row = _n - 1
generate double w = 2*mod(row, 2) - 1
generate double t = 2*mod(floor(row/2), 2) - 1
generate double v = 2*mod(floor(row/4), 2) - 1
generate double e = 2*mod(floor(row/8), 2) - 1
generate double x = t + v
generate double y = 2*w + 3*x + e
generate double aw = 1 + floor(row/16)
generate byte cl = mod(row, 7)
generate byte fe = floor(row/384)
generate double z = .
generate double zcopy = .
local absorb ""
local fe_rank 0
if ("`backend'" == "cuda") {
    replace y = y + .5*fe
    local absorb "absorb(fe)"
    local fe_rank 8
}
local exponents "0 10 20 25 27"
if ("`control'" == "control") local exponents "0"
local passed 0

foreach exponent of local exponents {
    quietly replace z = w + 2^(-`exponent')*t
    forvalues weighted = 0/1 {
        local weight ""
        if (`weighted') local weight "[aw=aw]"
        foreach mode in xhdfe-fast reghdfe-comparable {
            foreach vce in unadjusted robust cluster {
                local variance "vce(`vce')"
                if ("`vce'" == "cluster") local variance "vce(cluster cl)"
                quietly xhdfe y w x `weight', endogenous(x) instruments(z) ///
                    noconstant `absorb' `variance' keepsingletons numthreads(2) ///
                    tolerancemode(`mode') gpubackend(`backend') residuals(r) ///
                    noheader notable nofootnote
                ivcond_check, weighted(`weighted') vce(`vce') ///
                    ferank(`fe_rank') backend(`backend')
                display "PASS IV backend=`backend' delta=2^-`exponent' aw=`weighted' mode=`mode' vce=`vce'" ///
                    " b=" %12.5e scalar(ivcond_beta_error) " V=" %12.5e scalar(ivcond_v_error)
                drop r
                local ++passed
            }
        }
    }
}

// A failed instrument-rank check must revoke estimates and permit a valid refit.
quietly replace z = w + t
quietly replace zcopy = z
foreach mode in xhdfe-fast reghdfe-comparable {
    quietly xhdfe y w x, endogenous(x) instruments(z) noconstant `absorb' ///
        keepsingletons numthreads(2) tolerancemode(`mode') ///
        gpubackend(`backend') residuals(r) noheader notable nofootnote
    ivcond_check, weighted(0) vce(unadjusted) ferank(`fe_rank') backend(`backend')
    drop r
    capture noisily xhdfe y w x, endogenous(x) instruments(z zcopy) ///
        noconstant `absorb' keepsingletons numthreads(2) ///
        tolerancemode(`mode') gpubackend(`backend') noheader notable nofootnote
    local failed_rc = _rc
    assert `failed_rc' != 0
    assert "`e(cmd)'" == ""
    capture matrix list e(b)
    assert _rc != 0
    quietly xhdfe y w x, endogenous(x) instruments(z) noconstant `absorb' ///
        keepsingletons numthreads(2) tolerancemode(`mode') ///
        gpubackend(`backend') residuals(r) noheader notable nofootnote
    ivcond_check, weighted(0) vce(unadjusted) ferank(`fe_rank') backend(`backend')
    drop r
    display "PASS failed_refit backend=`backend' mode=`mode' rejected_rc=`failed_rc'"
    local ++passed
}

if ("`backend'" == "cuda") {
    local exponent = cond("`control'" == "control", 0, 27)
    quietly replace z = w + 2^(-`exponent')*t
    forvalues weighted = 0/1 {
        local weight ""
        if (`weighted') local weight "[aw=aw]"
        foreach mode in xhdfe-fast reghdfe-comparable {
            quietly xhdfe y w x `weight', endogenous(x) instruments(z) ///
                noconstant absorb(fe, savefe) keepsingletons numthreads(2) ///
                tolerancemode(`mode') gpubackend(cuda) residuals(r) ///
                noheader notable nofootnote
            ivcond_check, weighted(`weighted') vce(unadjusted) ferank(8) backend(cuda)
            assert e(fe_recovery_converged) == 1
            assert abs(y - _b[w]*w - _b[x]*x - __hdfe1__ - r) <= 1e-7
            drop r __hdfe1__
            display "PASS savefe_reconstruction backend=cuda aw=`weighted' mode=`mode'"
            local ++passed
        }
    }
}

display "IV_CONDITIONING_STATA PASS backend=`backend' control=`control' cases=`passed'"
log close ivcond
