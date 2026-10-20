"""Preserve the ordinary-fit workload across native and Stata timing cells.

Derived from tests/analytic_precision/benchmark_interfaces_workload_contract.py.
Coefficient parity cannot justify adding residual recovery to only one arm.
"""


def same_workload(native, stata):
    assert native['interface'] == 'native' and stata['interface'] == 'stata'
    assert native['retain_fes'] is False and stata['retain_fes'] is False
    assert stata['store_residuals'] is False, 'Stata requests an extra residual absorption'
    assert stata['workflow'] == 'ordinary'
    for key in ['dataset', 'mode', 'backend', 'threads', 'tol', 'max_iter',
                'weights', 'nobs', 'num_singletons', 'terms']:
        assert native[key] == stata[key], key
