"""Full OLS/IV covariance against MP120 on a public synthetic CG counterexample."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    os.environ.update(XHDFE_GPU_BACKEND=args.backend,
                      XHDFE_ABSORPTION_CACHE_MODE='off')
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    fixtures = Path(__file__).with_name('fixtures')
    oracle = json.loads((fixtures/'cg_seed1_2017_oracle.json').read_text())
    data_path = fixtures/'cg_seed1_2017.csv'
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == oracle['csv_sha256']
    data = pd.read_csv(data_path, float_precision='round_trip')
    rows = []
    for kind in ('ols', 'iv'):
        ref = oracle['references'][kind]
        b = np.array([float(x) for x in ref['b']])
        V = np.array([[float(x) for x in row] for row in ref['V']])
        margin = np.asarray(ref['covariance_arithmetic_margin'])
        diagonal_scale = np.sqrt(np.outer(V.diagonal(), V.diagonal()))
        for order in ('original', 'reverse', 'shuffled'):
            frame = data.sort_values(['sid', 'year'], ascending=order != 'reverse')
            if order == 'shuffled':
                frame = data.sort_values('shuffle')
            y = frame.dagg.to_numpy()
            x = frame['x' if kind == 'ols' else 'expo'].to_numpy()
            z = frame['x' if kind == 'ols' else 'z'].to_numpy()
            fes = [pd.factorize(frame[k], sort=True)[0].astype(np.int32)
                   for k in ('sid', 'year', 'entry_age')]
            clusters = pd.factorize(frame.did, sort=True)[0].astype(np.int32)
            for mode in ('reghdfe-comparable', 'xhdfe-fast', 'strict-residual'):
                row = dict(kind=kind, order=order, mode=mode)
                reg = core.HdfeRegressor(se_type='cluster', num_threads=1,
                                        stats_style='legacy', ssc_k_adj=False,
                                        ssc_g_adj=False, tolerance_mode=mode)
                kwargs = dict(fes=fes, clusters=clusters[:, None])
                if kind == 'iv':
                    kwargs.update(instruments=z[:, None], endogenous_idx=[0])
                try:
                    reg.fit(y, x[:, None], **kwargs)
                    rb, rv = np.asarray(reg.coef_), np.asarray(reg.covariance_)
                    # Match the permanent battery's distinct mode contracts.
                    # T01's Comparable full-V limit is not a StrictResidual
                    # work target (precision_contract.mode_and_tolerance).
                    if mode == 'reghdfe-comparable':
                        b_limit = 1e-9*np.maximum(1, abs(b))
                        v_limit = 1e-8*diagonal_scale+margin
                    else:
                        b_limit = np.full_like(b, 1e-10 if mode == 'strict-residual' else 1e-8)
                        v_limit = 1e-12+1e-7*abs(V)
                    row.update(b_ratio=float(np.max(abs(rb-b)/b_limit)),
                               V_ratio=float(np.max(abs(rv-V)/v_limit)),
                               V_scaled_error=float(np.max(abs(rv-V)/diagonal_scale)),
                               iterations=int(reg.num_iterations_),
                               method=int(reg.absorption_method_used),
                               gpu_used=bool(reg.gpu_used_),
                               gpu_status=int(reg.gpu_status_code_),
                               converged=bool(reg.converged_),
                               certified=bool(reg.precision_certified_))
                    passed = (np.isfinite(rb).all() and np.isfinite(rv).all()
                              and row['b_ratio'] <= 1 and row['V_ratio'] <= 1
                              and reg.converged_ and reg.precision_certified_
                              and reg.nobs_ == oracle['rows']
                              and bool(reg.gpu_used_) == (args.backend == 'cuda'))
                    row['status'] = 'PASS' if passed else 'FAIL'
                except Exception as error:
                    row.update(status='ERROR', error=repr(error))
                rows.append(row)
    result = dict(status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL',
                  module=str(args.module.resolve()),
                  module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
                  backend=args.backend, rows=rows)
    if args.output:
        with args.output.open('x') as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write('\n')
    print(json.dumps({'status': result['status'], 'cases': len(rows),
                      'failed': [r for r in rows if r['status'] != 'PASS']}))
    return result['status'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
