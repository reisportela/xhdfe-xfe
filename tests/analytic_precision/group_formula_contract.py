"""Grouped formula snapshots and sample/weight contracts against analytic truth."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from cases import BETA, generate


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--formula-source', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.environ['XHDFE_GPU_BACKEND'] = args.backend
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    native = load('py_hdfe_v11', args.module)
    sys.modules['xhdfe.py_hdfe_v11'] = native
    import xhdfe
    formula = load('xhdfe._formula', args.formula_source)
    options = dict(num_threads=2, drop_singletons=False, max_iter=100000,
                   tolerance_mode='strict-residual', tol=1e-12)
    records = []

    def record(name, work):
        try:
            work()
            records.append(dict(name=name, status='PASS'))
        except Exception as error:
            records.append(dict(name=name, status='FAIL', error=repr(error)))

    def verify(model, case, exact_dof=True):
        oracle = case.oracle()
        np.testing.assert_allclose(np.asarray(model.coef_)[:3], BETA, atol=1e-10, rtol=0)
        if exact_dof:
            np.testing.assert_allclose(np.asarray(model.covariance_)[:3, :3],
                                       oracle['V_exact'], atol=1e-12, rtol=1e-7)
            assert model.df_a_ == oracle['rank_D']
        selected = np.asarray(model.sample_index_, dtype=int)
        groups = case.frame.group.to_numpy()[selected]
        expected = case.group_frame.set_index('group').loc[groups, 'exact_residual'].to_numpy()
        np.testing.assert_allclose(model.residuals_, expected, atol=1e-7, rtol=0)
        assert len(np.unique(groups)) == len(case.group_frame)
        assert model.converged_ and model.precision_certified_
        assert bool(model.gpu_used_) == (args.backend == 'cuda')
        if args.backend == 'cuda':
            assert model.gpu_status_code_ == 1

    for aggregation in ('sum', 'mean'):
        case = generate('group_' + aggregation)
        for vectors in (False, True):
            def snapshot(case=case, vectors=vectors, aggregation=aggregation):
                data = case.frame.copy(deep=True)
                group = data.group.to_numpy(copy=True) if vectors else 'group'
                individual = data.individual.to_numpy(copy=True) if vectors else 'individual'
                prepared = formula.prepare_formula(
                    'y ~ x1 + x2 + x3 | own', data, group=group, individual=individual,
                    aggregation=aggregation, dofadjustments='exact', **options)
                assert not prepared.group.flags.writeable and not prepared.individual.flags.writeable
                first = prepared.fit()
                verify(first, case)
                data.loc[:, ['y', 'x1', 'own', 'group', 'individual']] = -12345
                if vectors:
                    group[:] = -8
                    individual[:] = -9
                repeated = prepared.fit()
                verify(repeated, case)
                np.testing.assert_array_equal(first.sample_index_, repeated.sample_index_)
                np.testing.assert_allclose(first.coef_, repeated.coef_, atol=1e-13, rtol=0)
            record(f'{aggregation}_snapshot_{"vectors" if vectors else "names"}', snapshot)

    def group_only():
        case = generate('group_only')
        model = formula.feols('y ~ x1 + x2 + x3 | g1 + g2', case.frame,
                              group='group', **options)
        verify(model, case, exact_dof=False)
    record('group_only_original_sample_map', group_only)

    case = generate('group_sum')
    for field in ('group', 'individual', 'y'):
        def missing(field=field):
            data = case.frame.copy(deep=True)
            data.loc[data.index[1], field] = np.nan
            try:
                formula.feols('y ~ x1 + x2 + x3 | own', data,
                              group='group', individual='individual', **options)
            except (ValueError, RuntimeError) as error:
                assert any(word in str(error).lower() for word in ('missing', 'nan', 'null', 'non-finite'))
            else:
                raise AssertionError('missing input was silently accepted')
        record(f'missing_{field}_rejected', missing)

    for value in (0, -1, 1.5, np.inf, np.nan, 2):
        def invalid_weight(value=value):
            weights = np.ones(len(case.frame))
            weights[1] = value
            try:
                formula.feols('y ~ x1 + x2 + x3 | own', case.frame, weights=weights,
                              fweights=True, group='group', individual='individual', **options)
            except (ValueError, RuntimeError) as error:
                assert any(word in str(error).lower() for word in ('weight', 'finite', 'missing'))
            else:
                raise AssertionError('invalid or inconsistent member weight was accepted')
        record(f'frequency_member_{value}_rejected', invalid_weight)

    def large_frequency_total():
        weights = np.full(len(case.frame), float(2**54))
        assert len(case.frame)*2**54 > 2**63-1
        assert len(case.group_frame)*2**54 < 2**63-1
        model = formula.feols('y ~ x1 + x2 + x3 | own', case.frame,
            weights=weights, fweights=True, group='group', individual='individual',
            aggregation='sum', dofadjustments='exact', **options)
        assert model.nobs_ == float(len(case.group_frame)*2**54)
        verify(model, case, exact_dof=False)
    record('frequency_total_counts_groups_once', large_frequency_total)

    receipt = dict(backend=args.backend, rows=records,
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.module, args.formula_source, Path(__file__))})
    with args.out.open('x') as output:
        json.dump(receipt, output, indent=2)
    print(json.dumps(dict(passed=sum(r['status']=='PASS' for r in records), rows=len(records))))
    for row in records:
        if row['status'] != 'PASS':
            print(json.dumps(row))
    return int(any(r['status'] != 'PASS' for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
