"""Public OLS range regressions with independent finite-sample Walsh oracles.

Run once per pinned module into a new JSON file. --compare-normal-from accepts
the other module's receipt from this unchanged worker. Known broken range rows
remain FAIL in a baseline receipt; only its normal rows enter the comparison.

For N=128 orthogonal Walsh columns, beta=0, RSS=N*sy^2 and HC1/Homosk
V=(sy/sx)^2 I/(N-p). For [a,a+delta*b], replace I by the analytic inverse
[[1+delta^-2,-delta^-2],[-delta^-2,delta^-2]]. Constant aweights leave V
unchanged and reported RSS is normalized back to N*sy^2. The four-cluster
positive-score fixture has V=(sy/sx)^2/3; balanced within-cluster products
give exact zero. Disjoint X/residual support also gives exact HC zero.

The limits retain Comparable beta 1e-9, diagonal-scaled full V 1e-8,
residual 1e-7 in fixture units, and the existing RSS 1e-8+1e-10*|RSS| rule.
Exact zero assertions apply only to these explicitly exact dyadic zero cases.
Their undefined t/p/CI retain the existing NaN convention.
No empirical rounding margin, alternative estimator oracle or timing gate is
introduced. The unweighted representable fixtures must succeed; extreme
aweights may instead fail informatively with all estimates revoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

N = 128
BETA_TOL = 1e-9
V_TOL = 1e-8
RESIDUAL_TOL = 1e-7
EMPTY_FIELDS = ('coef_', 'covariance_', 'stderr_', 'tvalues_', 'pvalues_',
                'conf_int_', 'residuals_', 'sample_index_')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def walsh(mask):
    return np.fromiter((1.0 if (i & mask).bit_count() % 2 else -1.0
                        for i in range(N)), dtype=np.float64, count=N)


def make_case(name, *, p=1, ex=0, ey=0, se='robust', kind='orthogonal',
              weight_exp=None, near=False, permuted=False, normal=False,
              allow_range_error=False):
    sx, sy = math.ldexp(1.0, ex), math.ldexp(1.0, ey)
    X = np.column_stack([walsh(j + 1) for j in range(p)]) * sx
    y = walsh(16) * sy
    inverse = [[Fraction(int(j == k)) for k in range(p)] for j in range(p)]
    if near:
        assert p == 2
        X[:, 1] = (walsh(1) + math.ldexp(1.0, -16) * walsh(2)) * sx
        d = Fraction(2) ** 32
        inverse = [[1 + d, -d], [-d, d]]
    clusters = None
    zero = kind in ('cluster_cancel', 'disjoint')
    if kind.startswith('cluster_'):
        assert p == 1 and se == 'cluster'
        clusters = np.arange(N, dtype=np.int64) // 32
        y = (walsh(1) * walsh(32) if kind == 'cluster_positive' else walsh(2)) * sy
    elif kind == 'disjoint':
        X[N // 2:, :] = 0.0
        y[:N // 2] = 0.0
    weight = None if weight_exp is None else np.full(N, math.ldexp(1.0, weight_exp))
    factor = Fraction(2) ** (2 * (ey - ex)) / (3 if clusters is not None else N - p)
    V = np.asarray([[float(0 if zero else factor * inverse[j][k])
                     for k in range(p)] for j in range(p)])
    rss = float(Fraction(N // 2 if kind == 'disjoint' else N) * Fraction(2) ** (2 * ey))
    if permuted:
        order = (37 * np.arange(N) + 11) % N
        X, y = X[order], y[order]
        if clusters is not None:
            clusters = clusters[order]
    X = np.asfortranarray(X)
    signature = hashlib.sha256(X.tobytes(order='F') + y.tobytes() +
        (b'' if clusters is None else clusters.tobytes()) +
        (b'' if weight is None else weight.tobytes())).hexdigest()
    return dict(name=name, X=X, y=y, weights=weight, clusters=clusters, V=V,
                rss=rss, p=p, se=se, residual_scale=sy, exact_zero=zero,
                df=3 if clusters is not None else N-p, normal=normal,
                allow_range_error=allow_range_error, input_sha256=signature)


def cases():
    result = [make_case('hc_p1_unit', normal=True),
              make_case('hc_p1_small', ex=-400, ey=-400),
              make_case('hc_p1_mixed', ex=-200, ey=-400),
              make_case('hc_p9_unit', p=9, normal=True),
              make_case('hc_p9_small', p=9, ex=-400, ey=-400),
              make_case('homo_p1_unit', se='unadjusted', normal=True),
              make_case('homo_p1_small', se='unadjusted', ex=-400, ey=-400),
              make_case('aw_p1_extreme', ex=200, ey=300, weight_exp=-600, allow_range_error=True),
              make_case('aw_p2_extreme', p=2, near=True, ex=200, ey=300,
                        weight_exp=-600, allow_range_error=True),
              make_case('aw_p2_unit', p=2, near=True, weight_exp=0, normal=True)]
    for kind in ('cluster_positive', 'cluster_cancel'):
        for exponent, scale in ((0, 'unit'), (-400, 'small')):
            for permuted in (False, True):
                result.append(make_case(f'{kind}_{scale}_{"permuted" if permuted else "sorted"}',
                    se='cluster', kind=kind, ex=exponent, ey=exponent,
                    permuted=permuted, normal=exponent == 0))
    result.extend([make_case('hc_zero_disjoint_unit', kind='disjoint', normal=True),
                   make_case('hc_zero_disjoint_small', kind='disjoint', ex=-400, ey=-400)])
    return result


def snapshot(model):
    arrays = {name: np.asarray(getattr(model, name)).tolist() for name in EMPTY_FIELDS}
    return dict(arrays=arrays, converged=bool(model.converged_),
                certified=bool(model.precision_certified_), gpu_used=bool(model.gpu_used_),
                nobs=float(model.nobs_), df=float(model.df_resid_),
                df_m=float(model.df_m_), df_a=float(model.df_a_), rss=float(model.rss_),
                omitted=[bool(v) for v in model.omitted_],
                threads_used=int(model.threads_used_))


def empty_state(model):
    sizes = {name: int(np.asarray(getattr(model, name)).size) for name in EMPTY_FIELDS}
    return dict(sizes=sizes, converged=bool(model.converged_),
                certified=bool(model.precision_certified_),
                empty=not any(sizes.values()) and not model.converged_ and not model.precision_certified_)


def range_error(message):
    value = message.lower()
    return 'no estimates' in value and any(word in value for word in ('range', 'overflow', 'underflow'))


def scaled_covariance_error(actual, reference):
    if np.all(reference == 0):
        return 0.0 if np.all(actual == 0) else math.inf
    scale = np.sqrt(np.diag(reference))
    return float(np.max(np.abs(actual-reference) / scale[:, None] / scale[None, :]))


def verify(case, actual):
    arrays = {name: np.asarray(value, dtype=float) for name, value in actual['arrays'].items()}
    assert arrays['coef_'].shape == (case['p'],), 'coefficient shape/rank'
    assert arrays['covariance_'].shape == (case['p'], case['p']), 'covariance shape'
    required_finite = ('coef_', 'covariance_', 'stderr_', 'residuals_', 'sample_index_')
    assert all(np.isfinite(arrays[name]).all() for name in required_finite), 'nonfinite output'
    for name, shape in (('tvalues_', (case['p'],)), ('pvalues_', (case['p'],)),
                        ('conf_int_', (case['p'], 2))):
        assert arrays[name].shape == shape, (name, 'shape')
        if case['exact_zero']:
            assert np.isnan(arrays[name]).all(), (name, 'expected NaN for exact zero variance')
        else:
            assert np.isfinite(arrays[name]).all(), (name, 'nonfinite positive-variance inference')
    assert actual['converged'] and actual['certified'] and not actual['gpu_used'], 'status/backend'
    assert actual['nobs'] == N and actual['df'] == case['df'], 'sample size/residual df'
    assert actual['df_m'] == case['p'] and actual['df_a'] == 0, 'model/absorbed df'
    assert len(actual['omitted']) == case['p'] and not any(actual['omitted']), 'unexpected omission'
    np.testing.assert_array_equal(arrays['sample_index_'], np.arange(N))
    beta_error = float(np.max(np.abs(arrays['coef_'])))
    covariance_error = scaled_covariance_error(arrays['covariance_'], case['V'])
    residual_error = float(np.max(np.abs((arrays['residuals_']-case['y']) / case['residual_scale'])))
    assert beta_error <= BETA_TOL, ('beta', beta_error)
    assert covariance_error <= V_TOL, ('scaled full V', covariance_error)
    assert residual_error <= RESIDUAL_TOL, ('scaled residual', residual_error)
    assert actual['rss'] > 0 and math.isfinite(actual['rss']), 'positive RSS lost'
    assert abs(actual['rss']-case['rss']) <= 1e-8 + 1e-10*abs(case['rss']), 'RSS'
    np.testing.assert_allclose(arrays['stderr_'], np.sqrt(np.diag(arrays['covariance_'])), rtol=V_TOL, atol=0)
    return dict(beta_error=beta_error, scaled_V_error=covariance_error, scaled_residual_error=residual_error)


def fit_case(cpp, case):
    model = cpp.HdfeRegressor(num_threads=2, se_type=case['se'], fit_intercept=False,
        drop_singletons=False, tol=1e-8, tolerance_mode='reghdfe-comparable', max_iter=1000)
    kwargs = {}
    if case['weights'] is not None:
        kwargs['weights'] = case['weights']
    if case['clusters'] is not None:
        kwargs['clusters'] = [case['clusters']]
    row = {key: case[key] for key in ('name', 'normal', 'allow_range_error', 'input_sha256')}
    row['oracle'] = dict(beta=[0.0]*case['p'], V=case['V'].tolist(), rss=case['rss'],
                         n=N, df=case['df'], rank=case['p'], exact_zero=case['exact_zero'])
    try:
        model.fit(case['y'], case['X'], **kwargs)
    except Exception as error:
        state = empty_state(model)
        allowed = case['allow_range_error'] and range_error(str(error)) and state['empty']
        row.update(status='SAFE_RANGE_ERROR' if allowed else 'FAIL', error=str(error), state=state)
        return row
    row['actual'] = snapshot(model)
    try:
        row.update(verify(case, row['actual']), status='PASS')
    except Exception as error:
        row.update(status='FAIL', error=str(error))
    return row


def failed_refit(cpp):
    good = make_case('prior_good', p=2, near=True)
    model = cpp.HdfeRegressor(num_threads=2, se_type='robust', fit_intercept=False,
        drop_singletons=False, tol=1e-8, tolerance_mode='reghdfe-comparable', max_iter=1000)
    row = dict(name='range_failure_after_success', normal=False)
    try:
        model.fit(good['y'], good['X'])
        verify(good, snapshot(model))
        row['prior_good_fit'] = True
        # RSS=128*2^1200=2^1207 is unrepresentable in the public FP64 result.
        try:
            model.fit(walsh(16)*math.ldexp(1.0, 600), good['X'])
        except Exception as error:
            state = empty_state(model)
            row.update(status='PASS' if range_error(str(error)) and state['empty'] else 'FAIL',
                       error=str(error), state=state)
        else:
            row.update(status='FAIL', error='unrepresentable numerical refit returned estimates', actual=snapshot(model))
    except Exception as error:
        row.update(status='FAIL', error=f'prior good fit failed: {error}')
    return row


def compare_normal(rows, baseline, catalog):
    previous = {row['name']: row for row in baseline['rows']}
    cases_by_name = {case['name']: case for case in catalog}
    comparisons = []
    for row in rows:
        if not row['normal']:
            continue
        result = dict(name=row['name'], status='PASS')
        try:
            old = previous[row['name']]
            assert row['status'] == old['status'] == 'PASS', 'normal row did not pass both runs'
            assert row['input_sha256'] == old['input_sha256'], 'fixture mismatch'
            a, b = row['actual'], old['actual']
            for key in ('nobs', 'df', 'df_m', 'df_a', 'omitted', 'converged', 'certified', 'gpu_used'):
                assert a[key] == b[key], key
            aa = {key: np.asarray(value, dtype=float) for key, value in a['arrays'].items()}
            bb = {key: np.asarray(value, dtype=float) for key, value in b['arrays'].items()}
            np.testing.assert_array_equal(aa['sample_index_'], bb['sample_index_'])
            np.testing.assert_allclose(aa['coef_'], bb['coef_'], rtol=0, atol=BETA_TOL)
            reference = cases_by_name[row['name']]['V']
            assert scaled_covariance_error(np.asarray(aa['covariance_']), np.asarray(bb['covariance_']) if np.any(reference) else reference) <= V_TOL
            for key in ('stderr_', 'tvalues_', 'pvalues_', 'conf_int_'):
                np.testing.assert_allclose(aa[key], bb[key], rtol=V_TOL, atol=V_TOL, equal_nan=True)
            np.testing.assert_allclose(aa['residuals_'], bb['residuals_'], rtol=0, atol=RESIDUAL_TOL)
            assert abs(a['rss']-b['rss']) <= 1e-8 + 1e-10*abs(b['rss']), 'RSS'
        except Exception as error:
            result.update(status='FAIL', error=str(error))
        comparisons.append(result)
    return comparisons


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return value.hex()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--module-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compare-normal-from', type=Path)
    args = parser.parse_args()
    module = args.module.resolve()
    assert module.is_file() and sha(module) == args.module_sha256.lower(), 'module pin mismatch'
    assert not args.out.exists() and args.out.parent.is_dir(), 'output must be a new file in an existing directory'
    worker_sha = sha(__file__)
    baseline = None
    if args.compare_normal_from:
        baseline = json.loads(args.compare_normal_from.read_text())
        assert baseline['worker_sha256'] == worker_sha, 'baseline used another worker'
    for name in tuple(os.environ):
        if name.startswith('XHDFE_'):
            del os.environ[name]
    settings = dict(XHDFE_GPU_BACKEND='cpu', XHDFE_ABSORPTION_CACHE_MODE='off',
        XHDFE_MOBILITY_MODE='off', XHDFE_FE_STRUCTURE_MODE='off', OMP_NUM_THREADS='2',
        OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    os.environ.update(settings)
    sys.dont_write_bytecode = True
    global np
    import numpy as np
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', module)
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    assert Path(cpp.__file__).resolve() == module, 'loaded module differs'
    catalog = cases()
    rows = [fit_case(cpp, case) for case in catalog]
    rows.append(failed_refit(cpp))
    comparison = compare_normal(rows, baseline, catalog) if baseline else []
    custody = sha(module) == args.module_sha256.lower() and sha(__file__) == worker_sha
    passed = custody and all(row['status'] in ('PASS', 'SAFE_RANGE_ERROR') for row in rows)
    passed = passed and all(row['status'] == 'PASS' for row in comparison)
    report = dict(passed=passed, module=str(module), module_sha256=sha(module), worker_sha256=worker_sha,
        custody_unchanged=custody, settings=settings, rows=rows, normal_comparison=comparison,
        baseline_sha256=sha(args.compare_normal_from) if baseline else None,
        limits=dict(beta=BETA_TOL, scaled_full_V=V_TOL, scaled_residual=RESIDUAL_TOL,
                    rss='1e-8 + 1e-10*abs(reference RSS)'),
        scope='Public CPU OLS range fixtures only; no FE/IV/GPU/performance certification')
    with args.out.open('x') as output:
        json.dump(json_safe(report), output, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=passed, outcomes=dict(Counter(row['status'] for row in rows)),
                         normal_comparisons=len(comparison), output=str(args.out))))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
