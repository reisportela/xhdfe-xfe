version 16
clear all
local task_root : env XHDFE_UPSTREAM_ROOT
include "`task_root'/env.do"
sysuse auto, clear
quietly regress price weight length
program define _core_set_control, eclass
    args value
    ereturn scalar diagnostic_with_a_long_name = `value'
end
_core_set_control 1
_core_results save long_identifier_test e()
_core_results compare long_identifier_test e(), tolerance(0)
_core_results compare long_identifier_test e(), include(macros: wexp wtype)
_core_set_control 2
capture noisily _core_results compare long_identifier_test e(), tolerance(0)
assert _rc == 9
_core_set_control 1
program define _core_change_covariance, eclass
    tempname wrong
    matrix `wrong' = e(V)
    matrix `wrong'[1,1] = `wrong'[1,1] + 1
    ereturn repost V = `wrong'
end
_core_change_covariance
capture noisily _core_results compare long_identifier_test e(), include(matrix: V) tolerance(1e-12)
assert _rc == 9
_core_results drop long_identifier_test
capture noisily _core_results compare long_identifier_test e()
assert _rc != 0
quietly summarize price
_core_results save returns r()
_core_results compare returns r(), tolerance(0)
quietly summarize weight
capture noisily _core_results compare returns r(), include(scalar: mean)
assert _rc == 9
_core_results drop returns
local dk_contract : env XHDFE_CORE_DK_CONTRACT
if ("`dk_contract'" == "1") {
    quietly regress price weight length
    matrix variance_original = e(V)
    _core_results save variance_control e()
    program define _core_variance_mutation, eclass
        args fraction
        tempname changed
        matrix `changed' = variance_original
        matrix `changed'[1,1] = variance_original[1,1]*(1+`fraction')
        ereturn repost V = `changed'
    end
    _core_variance_mutation 5e-9
    _core_results compare variance_control e(), include(matrix: V) tolerance(1e-12)
    _core_variance_mutation 2e-8
    capture noisily _core_results compare variance_control e(), include(matrix: V) tolerance(1e-12)
    assert _rc == 9
    _core_results drop variance_control
}
local repeat_contract : env XHDFE_CORE_REPEAT_CONTRACT
if ("`repeat_contract'" == "1") {
    quietly regress price weight length
    matrix repeat_b = e(b)
    matrix repeat_V = e(V)
    scalar repeat_F = e(F)
    _core_results save repeat_control e()
    program define _core_repeat_mutation, eclass
        args kind fraction
        tempname b V
        matrix `b' = repeat_b
        matrix `V' = repeat_V
        if ("`kind'" == "b") matrix `b'[1,1] = repeat_b[1,1] + `fraction'*max(1,abs(repeat_b[1,1]))
        if ("`kind'" == "V") matrix `V'[1,1] = repeat_V[1,1]*(1+`fraction')
        ereturn repost b=`b' V=`V'
        ereturn scalar F = repeat_F
        if ("`kind'" == "F") ereturn scalar F = repeat_F + `fraction'*max(1,abs(repeat_F))
    end
    foreach kind in b V F {
        local limit = cond("`kind'" == "b", 1e-9, 1e-8)
        local type = cond("`kind'" == "F", "scalar", "matrix")
        _core_repeat_mutation `kind' `=0.5*`limit''
        _core_results compare repeat_control e(), include(`type': `kind')
        _core_repeat_mutation `kind' `=2*`limit''
        capture noisily _core_results compare repeat_control e(), include(`type': `kind')
        assert _rc == 9
    }
    _core_results drop repeat_control
}
display "CORE_SNAPSHOT_CONTROLS_PASS"
