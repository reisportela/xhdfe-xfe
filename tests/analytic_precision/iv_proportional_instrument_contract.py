#!/usr/bin/env python3
"""Exact duplicates, tiny distinct directions and underidentification on CPU/CUDA.

Independent Walsh score-space oracle from the conditioning contract. None of
these references is another estimator fit. Exercise both tolerance modes,
weights, all three covariance types and multiple endogenous variables.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import iv_projection_conditioning_contract as contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    for key in tuple(os.environ):
        if key.startswith('XHDFE_'):
            del os.environ[key]
    os.environ['XHDFE_GPU_BACKEND'] = args.backend
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rows = []
    for delta in (1.0, 2.0**-27):
        for weighted in (False, True):
            for mode in contract.MODES:
                for vce in ('unadjusted', 'robust', 'cluster'):
                    data = contract.fixture(delta, weighted, args.backend, multiple=True)
                    # The multiple-endogenous fixture uses six Walsh bits.
                    # Hold weights constant over a complete 64-row cycle so
                    # its analytic score remains orthogonal to absorbed FEs.
                    if weighted:
                        data['weights'] = 1.0 + np.arange(data['y'].size)//64
                    weights = data['weights'] if weighted else np.ones(data['y'].size)
                    for fe in data['fes']:
                        for level in np.unique(fe):
                            assert np.max(np.abs(data['score'].T @ (weights*(fe == level)))) == 0
                    assert np.max(np.abs(data['score'].T @
                                         (weights[:, None]*(data['X']-data['score'])))) == 0
                    assert np.max(np.abs(data['score'].T @ (weights*data['residual']))) == 0
                    z, a = data['Z'].T
                    # The first two columns are nearly proportional when delta
                    # is small. The last two are exactly proportional; a must
                    # survive to identify the second endogenous variable.
                    w = data['X'][:, 0]
                    data['Z'] = np.column_stack((w, z, a, 2*z, -.5*a))
                    rows.append(contract.evaluate(core, data, args.backend, mode, vce,
                        dict(kind='distinct_and_exact', delta=delta, weighted=weighted)))
    for mode in contract.MODES:
        data = contract.fixture(2.0**-25, False, args.backend, zero_weight=True)
        z = data['Z'][:, 0]
        duplicate = 2*z
        duplicate[0] += 1.0  # differs only on an observation outside the sample
        data['Z'] = np.column_stack((z, duplicate))
        rows.append(contract.evaluate(core, data, args.backend, mode, 'unadjusted',
                                      dict(kind='proportional_on_positive_weight_sample')))
    for mode in contract.MODES:
        data = contract.fixture(1.0, False, args.backend, multiple=True)
        z = data['Z'][:, 0]
        model = core.HdfeRegressor(num_threads=2, fit_intercept=False,
                                  tolerance_mode=mode)
        model.fit(data['y'], data['X'], **contract.fit_arguments(data, 'unadjusted'))
        data['Z'] = np.column_stack((z, 2*z, -.5*z))
        row = dict(kind='duplicates_do_not_identify_second_endogenous', mode=mode)
        try:
            model.fit(data['y'], data['X'], **contract.fit_arguments(data, 'unadjusted'))
        except RuntimeError as error:
            row.update(status='PASS' if (np.asarray(model.coef_).size == 0 and
                           np.asarray(model.covariance_).size == 0 and
                           not model.converged_ and model.lifecycle_state_ == 'failed' and
                           ('rank' in str(error) or 'instrument' in str(error))) else 'FAIL',
                       error=str(error))
        else:
            row['status'] = 'FAIL'
        rows.append(row)
    failures = sum(row['status'] != 'PASS' for row in rows)
    report = dict(status='FAIL' if failures else 'PASS', backend=args.backend,
                  cases=len(rows), passed=len(rows)-failures, failed=failures,
                  module=str(args.module.resolve()),
                  module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
                  oracle='unchanged conditioning contract, independent Walsh scores',
                  results=rows)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','cases','passed','failed')}))
    return int(failures != 0)


if __name__ == '__main__':
    raise SystemExit(main())
