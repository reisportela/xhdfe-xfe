* Shared helpers for xhdfe Stata certification tests.

version 16

capture program drop xcert_setup
program define xcert_setup
    version 16
    discard
    local adopath : env XHDFE_STATA_ADOPATH
    if (`"`adopath'"' != "") {
        adopath ++ `"`adopath'"'
        di as text `"Added xhdfe adopath: `adopath'"'
    }

    capture which xhdfe
    if (c(rc)) {
        di as error "xhdfe not found. Build/install the Stata package or set XHDFE_STATA_ADOPATH."
        exit 601
    }
    which xhdfe

    capture which xhdfe_p
    if (c(rc)) {
        di as error "xhdfe_p not found next to xhdfe."
        exit 601
    }
    which xhdfe_p
end

capture program drop xcert_require_reghdfe
program define xcert_require_reghdfe
    version 16

    capture which reghdfe
    if (c(rc)) {
        di as error "reghdfe not found; install it to run this certification layer."
        exit 601
    }
    which reghdfe
end

capture program drop xcert_require_ivreghdfe
program define xcert_require_ivreghdfe
    version 16

    capture which ivreghdfe
    if (c(rc)) {
        di as error "ivreghdfe not found; install it to run this certification layer."
        exit 601
    }
    which ivreghdfe
end

capture program drop xcert_store_estimates
program define xcert_store_estimates
    version 16
    syntax, Prefix(name) [SCALARS(string)]

    if (`"`scalars'"' == "") {
        local scalars "N rmse tss rss mss r2 r2_a F df_r df_m ll ll_0"
    }

    matrix `prefix'_b = e(b)
    matrix `prefix'_V = e(V)

    foreach s of local scalars {
        capture scalar `prefix'_`s' = e(`s')
        if (c(rc)) {
            scalar `prefix'_`s' = .
        }
    }
end

capture program drop xcert_subset_estimates
program define xcert_subset_estimates
    version 16
    syntax, INprefix(name) OUTprefix(name) COLS(string) [SCALARS(string)]

    local ncols : word count `cols'
    if (`ncols' == 0) {
        di as error "xcert_subset_estimates requires at least one column"
        exit 198
    }

    matrix `outprefix'_b = J(1, `ncols', .)
    matrix colnames `outprefix'_b = `cols'
    matrix rownames `outprefix'_b = y1
    matrix `outprefix'_V = J(`ncols', `ncols', .)
    matrix rownames `outprefix'_V = `cols'
    matrix colnames `outprefix'_V = `cols'

    forvalues i = 1/`ncols' {
        local ci_name : word `i' of `cols'
        local ci = colnumb(`inprefix'_b, "`ci_name'")
        if (`ci' == .) {
            di as error "column `ci_name' not found in `inprefix'_b"
            matrix list `inprefix'_b
            exit 503
        }
        matrix `outprefix'_b[1, `i'] = `inprefix'_b[1, `ci']

        forvalues j = 1/`ncols' {
            local cj_name : word `j' of `cols'
            local cj = colnumb(`inprefix'_b, "`cj_name'")
            if (`cj' == .) {
                di as error "column `cj_name' not found in `inprefix'_b"
                matrix list `inprefix'_b
                exit 503
            }
            matrix `outprefix'_V[`i', `j'] = `inprefix'_V[`ci', `cj']
        }
    }

    foreach s of local scalars {
        capture scalar `outprefix'_`s' = scalar(`inprefix'_`s')
        if (c(rc)) {
            scalar `outprefix'_`s' = .
        }
    }
end

capture program drop xcert_assert_matrix_close
program define xcert_assert_matrix_close
    version 16
    syntax namelist(min=2 max=2), TOL(real) [NAME(string)]

    gettoken left right : namelist
    if (`"`name'"' == "") local name "`left' vs `right'"

    local left_rows : rownames `left'
    local right_rows : rownames `right'
    local left_cols : colnames `left'
    local right_cols : colnames `right'

    if (`"`left_rows'"' != `"`right_rows'"') {
        di as error "matrix row names differ for `name'"
        di as error "left:  `left_rows'"
        di as error "right: `right_rows'"
        exit 503
    }
    if (`"`left_cols'"' != `"`right_cols'"') {
        di as error "matrix column names differ for `name'"
        di as error "left:  `left_cols'"
        di as error "right: `right_cols'"
        exit 503
    }

    tempname maxdiff maxreldiff onesided
    mata: {                                                                                  ///
        A = st_matrix("`left'");                                                             ///
        B = st_matrix("`right'");                                                            ///
        st_numscalar("`onesided'", sum((A :>= .) :!= (B :>= .)));                            ///
        D = abs(A :- B);                                                                      ///
        S = (abs(A) :> abs(B)) :* abs(A) + (abs(A) :<= abs(B)) :* abs(B);                    ///
        S = (S :< 1) :* 1 + (S :>= 1) :* S;                                                   ///
        st_numscalar("`maxdiff'", max(D));                                                    ///
        st_numscalar("`maxreldiff'", max(D :/ S));                                            ///
    }
    // Mata's max() drops missing entries, so a position that is finite in one
    // matrix and missing in the other would otherwise be invisible here (the
    // scalar and variable comparators already guard one-sided missing).
    if (`onesided' > 0) {
        di as error "matrix comparison failed for `name': " `onesided' " position(s) missing on one side only"
        matrix list `left'
        matrix list `right'
        exit 9
    }
    if (missing(`maxdiff')) {
        scalar `maxdiff' = 0
    }
    if (missing(`maxreldiff')) {
        scalar `maxreldiff' = 0
    }

    di as text "  matrix `name': maxdiff=" %21.9g `maxdiff' " maxreldiff=" %21.9g `maxreldiff' " tol=" %21.9g `tol'
    if (`maxreldiff' > `tol') {
        di as error "matrix comparison failed for `name'"
        matrix list `left'
        matrix list `right'
        exit 9
    }
end

capture program drop xcert_assert_scalars_close
program define xcert_assert_scalars_close
    version 16
    syntax, LEFTprefix(name) RIGHTprefix(name) SCALARS(string) TOL(real)

    foreach s of local scalars {
        capture confirm scalar `leftprefix'_`s'
        if (c(rc)) {
            di as error "missing scalar `leftprefix'_`s'"
            exit 498
        }
        capture confirm scalar `rightprefix'_`s'
        if (c(rc)) {
            di as error "missing scalar `rightprefix'_`s'"
            exit 498
        }

        if (missing(scalar(`leftprefix'_`s')) & missing(scalar(`rightprefix'_`s'))) {
            di as text "  scalar `s': both missing"
        }
        else if (missing(scalar(`leftprefix'_`s')) | missing(scalar(`rightprefix'_`s'))) {
            di as error "scalar `s' missing mismatch: left=" scalar(`leftprefix'_`s') " right=" scalar(`rightprefix'_`s')
            exit 9
        }
        else {
            tempname diff scale limit
            scalar `diff' = abs(scalar(`leftprefix'_`s') - scalar(`rightprefix'_`s'))
            scalar `scale' = max(1, abs(scalar(`leftprefix'_`s')), abs(scalar(`rightprefix'_`s')))
            scalar `limit' = `tol' * `scale'
            di as text "  scalar `s': diff=" %21.9g `diff' " limit=" %21.9g `limit'
            if (`diff' > `limit') {
                di as error "scalar comparison failed for `s': left=" %21.9g scalar(`leftprefix'_`s') ///
                    " right=" %21.9g scalar(`rightprefix'_`s')
                exit 9
            }
        }
    }
end

capture program drop xcert_compare_estimates
program define xcert_compare_estimates
    version 16
    syntax, REFprefix(name) TESTprefix(name) [SCALARS(string) BTOL(real 1e-10) VTOL(real 1e-10) SCALTOL(real 1e-10)]

    if (`"`scalars'"' == "") {
        local scalars "N rmse tss rss mss r2 r2_a F df_r df_m ll ll_0"
    }

    xcert_assert_matrix_close `refprefix'_b `testprefix'_b, tol(`btol') name("e(b)")
    xcert_assert_matrix_close `refprefix'_V `testprefix'_V, tol(`vtol') name("e(V)")
    xcert_assert_scalars_close, leftprefix(`refprefix') rightprefix(`testprefix') scalars("`scalars'") tol(`scaltol')
end

capture program drop xcert_assert_var_close
program define xcert_assert_var_close
    version 16
    syntax varlist(min=2 max=2 numeric) [if] [in], TOL(real) [NAME(string)]

    marksample touse, novarlist
    gettoken left right : varlist
    if (`"`name'"' == "") local name "`left' vs `right'"

    quietly count if `touse' & (missing(`left') != missing(`right'))
    if (r(N) > 0) {
        di as error "variable missing-value mismatch for `name': " r(N) " rows"
        exit 9
    }

    tempvar diff
    quietly gen double `diff' = abs(`left' - `right') if `touse' & !missing(`left') & !missing(`right')
    quietly summarize `diff', meanonly
    tempname maxdiff
    scalar `maxdiff' = cond(r(N) == 0, 0, r(max))
    di as text "  variable `name': maxdiff=" %21.9g `maxdiff' " tol=" %21.9g `tol'
    if (`maxdiff' > `tol') {
        di as error "variable comparison failed for `name'"
        exit 9
    }
end

* ===========================================================================
* reghdfe convention-parity helpers, shared by every parity layer.
* ===========================================================================

* Report every divergence in a spec before failing, so one run yields the whole
* list instead of one item per re-run.
capture program drop xcert_parity_check
program define xcert_parity_check, rclass
    version 16
    syntax, Name(string) [BTOL(real 1e-10) BFLOOR(real .01) ///
        SETOL(real 1e-8) SCALTOL(real 1e-8) SCALFLOOR(real 1) SKIPF ///
        EXCEPT(string) KNOWN(string) KNOWNSE(real 0)]

    * EXCEPT = a divergence that is correct and justified at the call site.
    * KNOWN  = a real, open divergence from the canonical package: reported on
    *          every run so it cannot be forgotten, but not failed, so that a
    *          NEW divergence is what turns this suite red.
    local fails 0
    local known_hits 0

    * ---- discrete: exact equality, no tolerance ----------------------------
    foreach s in N df_r df_m df_a N_clust rank {
        * An exception must be named per scalar and justified at the call site;
        * it is never a blanket skip of the spec.
        if (strpos(" `except' ", " `s' ")) {
            di as text "  [`name'] e(`s') exempt: see call site"
            continue
        }
        local xv = scalar(px_`s')
        local rv = scalar(pr_`s')
        local xm = missing(`xv')
        local rm = missing(`rv')
        if (`xm' != `rm' | (!`xm' & !`rm' & `xv' != `rv')) {
            if (strpos(" `known' ", " `s' ")) {
                di as text "XHDFE_KNOWN_OPEN|`name'|e(`s')|xhdfe=`xv'|reghdfe=`rv'"
                local known_hits = `known_hits' + 1
            }
            else {
                di as error "  [`name'] e(`s') EXACT mismatch: xhdfe=`xv' reghdfe=`rv'"
                local fails = `fails' + 1
            }
        }
    }

    * ---- reported numerical statistics: missingness and value --------------
    * The scale floor is explicit: below it the test is an absolute tolerance
    * of scaltol*scalfloor, rather than the previous implicit unit floor.
    foreach s in r2 r2_a rmse tss rss mss {
        local xv = scalar(px_`s')
        local rv = scalar(pr_`s')
        local xm = missing(`xv')
        local rm = missing(`rv')
        if (`xm' != `rm') {
            di as error "  [`name'] e(`s') missingness differs: " ///
                "xhdfe missing=`xm' reghdfe missing=`rm'"
            local fails = `fails' + 1
        }
        else if (!`xm' & !`rm') {
            local scale = max(abs(`rv'), `scalfloor')
            local reldiff = abs(`xv' - `rv') / `scale'
            if (`reldiff' > `scaltol') {
                di as error "  [`name'] e(`s') scaled diff " ///
                    %9.2e `reldiff' " > `scaltol' (floor=`scalfloor')"
                local fails = `fails' + 1
            }
        }
    }

    * ---- coefficient stripe and omission pattern ---------------------------
    * A dropped term carries a zero (or missing) standard error in both
    * engines; the pattern, not the value, is the convention under test.
    local xcols : colnames px_b
    local rcols : colnames pr_b
    local nx : word count `xcols'
    local nr : word count `rcols'
    if (`nx' != `nr') {
        di as error "  [`name'] e(b) column count differs: xhdfe=`nx' reghdfe=`nr'"
        local fails = `fails' + 1
    }
    foreach c of local xcols {
        local ci = colnumb(pr_b, "`c'")
        if (`ci' == .) {
            di as error "  [`name'] coefficient `c' absent from reghdfe e(b)"
            local fails = `fails' + 1
            continue
        }
        local xj = colnumb(px_b, "`c'")
        local xse = sqrt(px_V[`xj', `xj'])
        local rse = sqrt(pr_V[`ci', `ci'])
        local xdrop = (missing(`xse') | `xse' == 0)
        local rdrop = (missing(`rse') | `rse' == 0)
        if (`xdrop' != `rdrop') {
            di as error "  [`name'] omission differs for `c': " ///
                "xhdfe dropped=`xdrop' reghdfe dropped=`rdrop'"
            local fails = `fails' + 1
        }
    }
    foreach c of local rcols {
        if (colnumb(px_b, "`c'") == .) {
            di as error "  [`name'] coefficient `c' absent from xhdfe e(b)"
            local fails = `fails' + 1
        }
    }

    * ---- continuous: solver tolerance --------------------------------------
    foreach c of local xcols {
        local ci = colnumb(pr_b, "`c'")
        if (`ci' == .) continue
        local xj = colnumb(px_b, "`c'")
        local xb = px_b[1, `xj']
        local rb = pr_b[1, `ci']
        local xse = sqrt(px_V[`xj', `xj'])
        local rse = sqrt(pr_V[`ci', `ci'])
        if (missing(`xb') != missing(`rb')) {
            di as error "  [`name'] b[`c'] missingness differs"
            local fails = `fails' + 1
        }
        else if (!missing(`xb') & !missing(`rb')) {
            local scale = max(abs(`rb'), `bfloor')
            if (abs(`xb' - `rb') / `scale' > `btol') {
                di as error "  [`name'] b[`c'] rel diff " ///
                    %9.2e (abs(`xb' - `rb') / `scale') ///
                    " > `btol' (floor=`bfloor')"
                local fails = `fails' + 1
            }
        }
        if (!missing(`xse') & !missing(`rse') & `rse' > 0) {
            local sedev = abs(`xse' - `rse') / `rse'
            if (`sedev' > `setol') {
                if (`knownse' > 0 & `sedev' <= `knownse') {
                    di as text "XHDFE_KNOWN_OPEN|`name'|se[`c']|reldiff=" %9.2e `sedev'
                    local known_hits = `known_hits' + 1
                }
                else {
                    di as error "  [`name'] se[`c'] rel diff " ///
                        %9.2e `sedev' " > `setol'"
                    local fails = `fails' + 1
                }
            }
        }
    }

    * e(F) is excluded by default: the engines post different F conventions on
    * some clustered designs (documented in dof-ssc.do at roughly 0.2%).
    if ("`skipf'" == "") {
        if (!missing(scalar(px_F)) & !missing(scalar(pr_F)) & scalar(pr_F) > 0) {
            if (abs(scalar(px_F) - scalar(pr_F)) / scalar(pr_F) > 1e-6) {
                di as error "  [`name'] e(F) rel diff " ///
                    %9.2e (abs(scalar(px_F) - scalar(pr_F)) / scalar(pr_F))
                local fails = `fails' + 1
            }
        }
    }

    if (`fails' == 0 & `known_hits' > 0) {
        di as result "  PASS  `name' (`known_hits' known open divergence(s))"
    }
    else if (`fails' == 0) {
        di as result "  PASS  `name'"
    }
    else {
        di as error "  FAIL  `name' (`fails' divergences)"
    }
    return scalar fails = `fails'
end

* Run one specification through both engines and compare.
capture program drop xcert_parity_spec
program define xcert_parity_spec, rclass
    version 16
    syntax anything(everything) [aweight fweight pweight iweight], ///
        Name(string) [XOPTS(string) ROPTS(string) ///
        BTOL(real 1e-10) SETOL(real 1e-8) SKIPF EXCEPT(string) ///
        KNOWN(string) KNOWNSE(real 0)]

    quietly xhdfe `anything' [`weight'`exp'], ///
        tolerancemode(reghdfe-comparable) tolerance(1e-12) ///
        `xopts' noheader notable nofootnote
    xcert_store_estimates, prefix(px) ///
        scalars(N rmse tss rss mss r2 r2_a F df_r df_m df_a N_clust rank)

    quietly reghdfe `anything' [`weight'`exp'], `ropts'
    xcert_store_estimates, prefix(pr) ///
        scalars(N rmse tss rss mss r2 r2_a F df_r df_m df_a N_clust rank)

    xcert_parity_check, name("`name'") btol(`btol') setol(`setol') `skipf' ///
        except(`except') known(`known') knownse(`knownse')
    return scalar fails = r(fails)
end

capture program drop xcert_parity_tally
program define xcert_parity_tally
    global PARITY_FAILS = $PARITY_FAILS + r(fails)
end
