"""Grade the corrected ordinary CUDA projections against independent WLS."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
sys.path.insert(0, str(SRC / 'tests'))
sys.path.insert(0, str(SRC / 'tests/analytic_precision'))
from cuda_weighted_cpu_route_contract import Runner, cov240k, grade, spec, weight_vector, weighted_oracle


def run(args):
    args.output.mkdir(mode=0o700)
    fixtures = (
        spec('aw2_savefe', nfe=2, weights='aw', retain_fes=True),
        spec('aw3', nfe=3, weights='aw'),
        spec('aw3_savefe', nfe=3, weights='aw', retain_fes=True),
        spec('fw3', nfe=3, weights='fw'),
        spec('pw3', nfe=3, weights='pw'),
        spec('iw3', nfe=3, weights='iw'),
        spec('aw3_group', nfe=3, weights='aw', group='group'),
        spec('aw3_gs', nfe=3, weights='aw', method='gauss-seidel'),
        spec('aw4', nfe=4, weights='aw'),
        spec('aw4_savefe', nfe=4, weights='aw', retain_fes=True),
        spec('uw4', nfe=4, weights='none'),
        spec('uw3', nfe=3, weights='none'),
        spec('aw3_strict', nfe=3, weights='aw', mode='strict-residual'),
        spec('uw4_strict', nfe=4, weights='none', mode='strict-residual'),
    )
    if args.select:
        names = set(args.select.split(','))
        fixtures = tuple(case for case in fixtures if case['name'] in names)
        assert len(fixtures) == len(names)
    if args.capture:
        assert any(case['name'] == 'aw3_strict' for case in fixtures)
        fixtures = tuple(sorted(fixtures, key=lambda case: case['name'] == 'aw3_strict'))
    module = args.module.resolve()
    run = Runner(args.output, args.threads)
    by_case = {}
    by_oracle = {}
    for repeat in range(args.repeats):
        capture = args.output.resolve() / f'strict_{repeat}.bin'
        env = (dict(XHDFE_ABSORPTION_CACHE_MODE='write', XHDFE_ABSORPTION_CACHE=str(capture))
               if args.capture else None)
        receipt = run.run(module, f'fit_{repeat}', fixtures, backend='cuda', env=env)
        for case, fit in zip(fixtures, receipt['fits']):
            label = case['name']
            if fit['refused']:
                reduced = {key: fit[key] for key in ('refused', 'error', 'estimates_left')}
                row = dict(name=label, repeat=repeat, status='REFUSED', **reduced)
            else:
                assert fit['gpu_used'] and fit['gpu_status_code'] == 1, (label, fit)
                assert fit['converged'] and fit['certified'], (label, fit)
                key = case['nfe'], case['weights']
                if key not in by_oracle:
                    y, x, fes = cov240k(case['nfe'])
                    w = weight_vector(case['weights'], len(y))
                    by_oracle[key] = weighted_oracle(
                        y, x, fes, np.ones(len(y)) if w is None else w, case['weights'] == 'fw')
                oracle = by_oracle[key]
                accuracy = grade(oracle, fit)
                assert 'ungraded' not in accuracy, (label, accuracy)
                sd = np.sqrt(np.diag(oracle['V']))
                accuracy['scaled_V'] = float(np.max(np.abs(oracle['V'] - fit['V']) / np.outer(sd, sd)))
                passed = (accuracy['b_scaled'] <= 1e-8 and accuracy['se_rel'] <= 1e-7 and
                          accuracy['scaled_V'] <= 1e-8 and fit['df_a'] == oracle['df_a'])
                fields = ('iterations', 'method', 'cuda_tail', 'df_a', 'df_resid', 'sample_sha',
                          'fe_recovery', 'gpu_used', 'gpu_status_code')
                row = dict(name=label, repeat=repeat, status='PASS' if passed else 'FAIL',
                           **{field: fit[field] for field in fields}, **accuracy)
            print(json.dumps(row), flush=True)
            by_case.setdefault(label, []).append(row)
    with (args.output / 'summary.json').open('x') as stream:
        json.dump(dict(module=str(module), sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                       cases=by_case), stream, indent=2)
    assert all(row['status'] == 'PASS' for rows in by_case.values() for row in rows), args.output
    print('cuda_projection_repair_contract PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--select')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--capture', action='store_true')
    run(parser.parse_args())
