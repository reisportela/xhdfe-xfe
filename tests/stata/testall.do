* ===========================================================================
* Run all xhdfe Stata certification tests
* ===========================================================================

version 16
clear all
set more off
set varabbrev on
set linesize 120
capture log close _all

local testdir : env XHDFE_STATA_TEST_DIR
if (`"`testdir'"' == "") {
    local testdir "`c(pwd)'/tests/stata"
}

cd `"`testdir'"'
do "_helpers.do"

xcert_setup

global XHDFE_TESTS_RUN 0

local tests ///
    "part1/noabsorb.do" ///
    "part1/unadjusted.do" ///
    "part1/robust.do" ///
    "part1/cluster.do" ///
    "part1/weights.do" ///
    "part1/factor-vars.do" ///
    "part1/multiway-fe.do" ///
    "part1/multiway-cluster.do" ///
    "part1/heterogeneous-slopes.do" ///
    "part1/savefes.do" ///
    "part1/savefe-alias.do" ///
    "part1/iv.do" ///
    "part1/weights-extended.do" ///
    "part1/group-individual.do" ///
    "part1/sample-missing.do" ///
    "part1/dof-ssc.do" ///
    "part1/string-interactions.do" ///
    "part1/collinearity.do" ///
    "part1/noconstant.do" ///
    "part1/estat.do" ///
    "part1/nosample.do" ///
    "part1/methods.do" ///
    "part1/postestimation.do" ///
    "part1/predict.do" ///
    "part1/options.do" ///
    "part1/companions.do" ///
    "../../examples/gelbach_example.do" ///
    "../../examples/gelbach_absorbed_target.do"

foreach test of local tests {
    di as text _n "{hline 72}"
    di as text "Running `test'"
    di as text "{hline 72}"
    capture noisily do "`test'"
    if (c(rc)) {
        local test_rc = c(rc)
        di as error "FAILED: " as result "`test'" as error " returned rc=" as result `test_rc'
        exit `test_rc'
    }
    global XHDFE_TESTS_RUN = $XHDFE_TESTS_RUN + 1
}

di as text _n "{hline 72}"
di as text "XHDFE STATA CERTIFICATION TESTS COMPLETED SUCCESSFULLY"
di as text "tests_run = " $XHDFE_TESTS_RUN
di as text "{hline 72}"

clear
exit
