"""Numerical gates plus the required error/no-estimates rejection contract."""
from evaluate import assess as numerical_assessment
from evaluate import selftest as numerical_selftest
import numpy as np


def assess(job, result, fixture):
    if result.get('timeout') or result.get('harness_error'):
        return numerical_assessment(job, result, fixture)
    if job.get('expected_rejection') and result.get('rc', 0) == 0 and not result.get('error'):
        return dict(job, result=result, verdict='FAILED_TO_REJECT')
    if job.get('expected_rejection'):
        for name in ('beta', 'V', 'residuals'):
            if name in result and np.isfinite(np.asarray(result[name], dtype=float)).any():
                return dict(job, result=result, verdict='ERROR_RETAINED_ESTIMATES')
    return numerical_assessment(job, result, fixture)


def selftest(oracle, metadata, frame, fixture):
    numerical_selftest(oracle, metadata, frame, fixture)
    job = dict(engine='xhdfe', expected_rejection=True,
               reject_pattern='deliberate unsupported mode',
               rejection_status=dict(converged=0, iterations=0))
    assert assess(job, dict(rc=0, converged=0, iterations=0), fixture)['verdict'] == 'FAILED_TO_REJECT'
    failed = dict(rc=198, error='deliberate unsupported mode')
    assert assess(job, failed, fixture)['verdict'] == 'UNSUPPORTED_EXPECTED'
    for key, value in [('beta', oracle['beta']), ('V', oracle['V_contract']), ('residuals', [0.1])]:
        assert assess(job, dict(failed, **{key: value}), fixture)['verdict'] == 'ERROR_RETAINED_ESTIMATES'
