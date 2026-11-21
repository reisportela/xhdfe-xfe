"""Directed CPU/CUDA contract for conditional intercept reconstruction.

Synthetic data only. By default the reference fit drops the literal zero
columns, so it uses the unchanged full-width reconstruction. An optional
--reference-module compares identical inputs with a frozen functional module.
The positive-intercept/indefinite fixture is pinned by dummy-WLS arithmetic.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys

import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / 'tests'))
sys.path.insert(0, str(WT / 'tests/analytic_precision'))
from cluster_shared_passes_contract import panel
from omitted_multiway_covariance_contract import group_panel
from multiway_intercept_ssc_contract import fixture as positive_fixture


def scenarios():
    cases = []
    for mode in ('reghdfe-comparable', 'xhdfe-fast'):
        for weight in ('none', 'aw', 'fw', 'pw', 'iw'):
            for gdf in ('min', 'conventional'):
                for adjusted in (False, True):
                    cases.append(dict(name=f'positive_{mode}_{weight}_{gdf}_{adjusted}',
                        kind='positive', mode=mode, weight=weight, gdf=gdf,
                        adjusted=adjusted, layout=len(cases) % 3,
                        expected_fallback=False))
        for weight in ('none', 'aw', 'fw'):
            for scale in (1e-140, 1e140):
                for offset in (0.0, 1e6):
                    cases.append(dict(name=f'positive_range_{mode}_{weight}_{scale}_{offset}',
                        kind='positive', mode=mode, weight=weight, scale=scale,
                        offset=offset, layout=len(cases) % 3,
                        expected_fallback=False))
        for aggregation in ('sum', 'mean'):
            for weight in ('none', 'aw', 'fw'):
                cases.append(dict(name=f'grouped_{mode}_{aggregation}_{weight}',
                    kind='grouped', mode=mode, weight=weight, aggregation=aggregation,
                    layout=len(cases) % 3))
        for weight in ('none', 'aw', 'fw'):
            for scale in (1.0, 1e-140, 1e140):
                cases.append(dict(name=f'nested_range_{mode}_{weight}_{scale}',
                    kind='nested', mode=mode, weight=weight, scale=scale,
                    layout=len(cases) % 3))
        cases.append(dict(name=f'positive_intercept_indefinite_{mode}',
            kind='indefinite', mode=mode, weight='none', layout=1,
            expected_fallback=False))
    cases += [dict(name='full_width_keeps_reconstruction', kind='positive',
                   weight='none', layout=-1, expected_fallback=False),
              dict(name='repeat_before', kind='positive', weight='none', layout=1,
                   expected_fallback=False, reuse=True),
              dict(name='zero_fallback_without_positive_result', kind='zero',
                   weight='none', layout=1, expected_fallback=True, reuse=True),
              dict(name='repeat_after', kind='positive', weight='none', layout=1,
                   expected_fallback=False, reuse=True)]
    return cases


def data(case):
    if case['kind'] in ('positive', 'zero'):
        y, x, fe, clusters, w = positive_fixture()
        fes = [fe]
    elif case['kind'] == 'indefinite':
        rng = np.random.default_rng(22)
        n = 640
        fe = rng.integers(0, 12, n).astype(np.int32)
        c1 = rng.integers(0, 6, n).astype(np.int32)
        c2 = rng.integers(0, 7, n).astype(np.int32)
        x = rng.normal(size=(n, 2)) + [.5, -.3]
        y = x @ [.7, -.4] + np.sin(fe) + rng.normal(size=6)[c1]
        y += rng.normal(size=7)[c2] + rng.normal(size=n)
        fes, clusters, w = [fe], [c1, c2], np.ones(n)
    elif case['kind'] == 'grouped':
        d = group_panel()
        y, x = d['y'], d['X']
        fes, clusters = [d['individual']], [d['c1'], d['c2']]
        w = 1.0 + d['group'] % 3
    else:
        d = panel(5)
        y, x = d['y'], d['X']
        fes = [d['worker'], d['firm'], d['year']]
        clusters = [d['worker'], d['firm_sparse']]
        w = d['fw'] if case['weight'] == 'fw' else d['w']
    if case['kind'] == 'zero':
        y = np.zeros_like(y)
    x = x + np.array([case.get('offset', 0.0), -2 * case.get('offset', 0.0)])
    layout = case['layout']
    if layout == 0:
        X = np.column_stack((np.zeros(len(y)), x))
    elif layout == 1:
        X = np.column_stack((x[:, 0], np.zeros(len(y)), x[:, 1]))
    elif layout == 2:
        X = np.column_stack((np.zeros(len(y)), x, np.zeros(len(y))))
    else:
        X = x
    kw = dict(fes=fes, clusters=clusters)
    if case['weight'] != 'none':
        kw['weights'] = w
        if case['weight'] == 'fw': kw['fweights'] = True
        if case['weight'] == 'pw': kw['pweights'] = True
        if case['weight'] == 'iw': kw['iweights'] = True
    if case['kind'] == 'grouped':
        kw.update(group=d['group'], individual=d['individual'],
                  aggregation=case['aggregation'])
    return y * case.get('scale', 1.0), np.asfortranarray(X), kw


def worker(module, kept_reference=False):
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rows, reusable = [], None
    for case in scenarios():
        print('CGATE_BEGIN ' + case['name'], file=sys.stderr, flush=True)
        row = dict(name=case['name'])
        try:
            y, X, kw = data(case)
            input_sha256 = hashlib.sha256(y.tobytes() + X.tobytes(order='F')).hexdigest()
            if kept_reference:
                X = np.asfortranarray(X[:, np.any(X != 0.0, axis=0)])
            opts = dict(se_type='cluster', num_threads=4,
                tolerance_mode=case.get('mode', 'reghdfe-comparable'),
                ssc_k_adj=False, ssc_g_df=case.get('gdf', 'min'),
                ssc_g_adj=case.get('adjusted', False))
            if case.get('reuse'):
                if reusable is None: reusable = core.HdfeRegressor(**opts)
                model = reusable
            else:
                model = core.HdfeRegressor(**opts)
            model.fit(y, X, **kw)
            assert model.converged_ and model.precision_certified_
            assert bool(model.gpu_used_) == (os.environ['XHDFE_GPU_BACKEND'] == 'cuda')
            omitted = np.asarray(model.omitted_reason_)[:X.shape[1]] != 0
            kept = [int(i) for i in np.flatnonzero(~omitted)] + [X.shape[1]]
            V = np.asarray(model.covariance_)
            assert np.all(V[np.flatnonzero(omitted)] == 0)
            assert np.all(V[:, np.flatnonzero(omitted)] == 0)
            for name in ('stderr_', 'tvalues_', 'pvalues_', 'conf_int_'):
                assert np.isnan(np.asarray(getattr(model, name))[np.flatnonzero(omitted)]).all()
            row.update(status='ESTIMATES', b=np.asarray(model.coef_)[kept].tolist(),
                V=V[np.ix_(kept, kept)].tolist(), se=np.asarray(model.stderr_)[kept].tolist(),
                kept=kept, omitted=np.flatnonzero(omitted).tolist(),
                input_sha256=input_sha256, kept_reference=kept_reference,
                sample_sha256=hashlib.sha256(np.asarray(model.sample_index_).tobytes()).hexdigest(),
                iterations=int(model.num_iterations_), method=int(model.absorption_method_used),
                gpu_used=bool(model.gpu_used_),
                psd_fixed=getattr(model, 'vcv_psd_fixed_', None))
        except Exception as error:
            row.update(status='ERROR', error_type=type(error).__name__, error=str(error))
        rows.append(row)
        print('CGATE_END ' + case['name'], file=sys.stderr, flush=True)
    print(json.dumps(rows, allow_nan=False))


def metrics(actual, reference):
    b, br = np.asarray(actual['b']), np.asarray(reference['b'])
    V, vr = np.asarray(actual['V']), np.asarray(reference['V'])
    s = np.sqrt(np.maximum(0.0, np.diag(vr)))
    if np.any(s == 0):
        zero = s == 0
        assert np.array_equal(V[zero], vr[zero]) and np.array_equal(V[:, zero], vr[:, zero])
        s[zero] = 1
    dv = float(np.max(np.abs((V-vr)/s[:, None]/s[None, :])))
    positive = np.asarray(reference['se']) > 0
    ds = float(np.max(np.abs(np.asarray(actual['se'])[positive] /
                               np.asarray(reference['se'])[positive] - 1))) if positive.any() else 0.0
    return dict(b_error=float(np.max(np.abs(b-br)/np.maximum(1.0, np.abs(br)))),
                V_error=dv, se_error=ds)


def indefinite_fixture_proof():
    case = next(c for c in scenarios() if c['kind'] == 'indefinite')
    y, X, kw = data(case)
    x = X[:, np.any(X != 0.0, axis=0)]
    fe = kw['fes'][0]
    D = np.eye(int(fe.max()) + 1)[fe]
    design = np.column_stack((x, D))
    q, r = np.linalg.qr(design, mode='reduced')
    beta = np.linalg.solve(r, q.T @ y)
    residual = y - design @ beta
    transform = np.zeros((3, design.shape[1]))
    transform[:2, :2] = np.eye(2)
    transform[2, 2:] = D.mean(axis=0)
    score = (transform @ np.linalg.solve(r, q.T)).T * residual[:, None]
    raw = np.zeros((3, 3))
    c1, c2 = kw['clusters']
    for sign, labels in ((1, c1[:, None]), (1, c2[:, None]),
                         (-1, np.column_stack((c1, c2)))):
        _, inverse = np.unique(labels, axis=0, return_inverse=True)
        totals = np.zeros((int(inverse.max()) + 1, 3))
        np.add.at(totals, inverse, score)
        raw += sign * (totals.T @ totals)
    eigenvalues = np.linalg.eigvalsh(raw)
    slope_eigenvalues = np.linalg.eigvalsh(raw[:2, :2])
    assert raw[2, 2] > 0 and eigenvalues.min() < 0 and slope_eigenvalues.min() > 0
    return dict(raw_covariance=raw.tolist(), eigenvalues=eigenvalues.tolist(),
                slope_eigenvalues=slope_eigenvalues.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=Path)
    ap.add_argument('--kept-reference', action='store_true')
    ap.add_argument('--module', type=Path)
    ap.add_argument('--reference-module', type=Path)
    ap.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.worker:
        worker(args.worker, args.kept_reference)
        return
    if args.module is None or args.output is None:
        ap.error('--module and --output are required')
    fixture_proof = indefinite_fixture_proof()
    out = args.output
    out.mkdir()
    arms = {}
    env = {k:v for k,v in os.environ.items() if not k.startswith('XHDFE_') and k != 'LD_LIBRARY_PATH'}
    env.update(XHDFE_GPU_BACKEND=args.backend, XHDFE_PROFILE_CPU='1',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    reference_module = args.reference_module or args.module
    for arm, module in [('reference', reference_module), ('candidate', args.module)]:
        command = [sys.executable, '-B', __file__, '--worker', str(module)]
        if arm == 'reference' and args.reference_module is None:
            command.append('--kept-reference')
        result = subprocess.run(command,
            cwd=WT, env=env, text=True, capture_output=True, timeout=420)
        (out/(arm+'.stderr')).write_text(result.stderr)
        (out/(arm+'.stdout')).write_text(result.stdout)
        assert result.returncode == 0, (arm, result.returncode, result.stderr[-1000:])
        arms[arm] = json.loads(result.stdout)
        assert len(arms[arm]) == len(scenarios()), (arm, 'case count')
    segments, active = {}, None
    for line in (out/'candidate.stderr').read_text().splitlines():
        if line.startswith('CGATE_BEGIN '):
            active = line[len('CGATE_BEGIN '):]; segments[active] = []
        elif line.startswith('CGATE_END '): active = None
        elif active is not None: segments[active].append(line)
    rows = []
    for case, ref, got in zip(scenarios(), arms['reference'], arms['candidate']):
        row = dict(name=case['name'])
        try:
            assert case['name'] == ref['name'] == got['name']
            assert ref['status'] == 'ESTIMATES', ('reference', ref)
            assert got['status'] == 'ESTIMATES', ('candidate', got)
            assert ref['input_sha256'] == got['input_sha256']
            assert ref['sample_sha256'] == got['sample_sha256']
            if args.reference_module is not None:
                assert ref['kept'] == got['kept']
            error = metrics(got, ref)
            assert error['b_error'] <= 1e-8 and error['se_error'] <= 1e-7
            assert error['V_error'] <= 1e-8, error
            lines = segments[case['name']]
            gates = [line for line in lines if 'label=multiway_reconstruction_gate ' in line]
            assert gates, 'gate trace absent'
            values = [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in gates]
            if 'expected_fallback' in case:
                assert all(int(v['preliminary_fallback']) == int(case['expected_fallback']) for v in values), values
            for value in values:
                full_width = value['kept'] == value['full']
                assert int(value['required']) == int(full_width or int(value['preliminary_fallback']))
            actual_reconstruction = any('label=cluster_passes site=recon ' in line for line in lines)
            assert actual_reconstruction == any(int(v['required']) for v in values)
            row.update(status='PASS', **error, gates=values, reconstruction=actual_reconstruction)
        except Exception as failure:
            row.update(status='FAIL', error_type=type(failure).__name__, error=str(failure),
                       reference_status=ref['status'], candidate_status=got['status'])
        rows.append(row)
    report = dict(module=str(args.module), reference_module=str(reference_module),
                  reference_kind='same_inputs' if args.reference_module else 'kept_columns',
                  backend=args.backend,
                  indefinite_fixture_proof=fixture_proof,
                  hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [args.module, reference_module, Path(__file__)]}, rows=rows)
    with (out/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    failed = [row for row in rows if row['status'] == 'FAIL']
    print(f'{len(rows)-len(failed)}/{len(rows)} extended cases passed; {len(failed)} failed')
    for row in failed: print(row['name'], row['error_type'], row['error'])
    raise SystemExit(bool(failed))


if __name__ == '__main__':
    main()
