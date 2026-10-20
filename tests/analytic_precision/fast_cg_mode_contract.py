"""Fast CG return/precision controls with analytic and optional core24 inputs.

The optional github reproducer reads its supplied dataset in place. Only
aggregates are written. Its covariance error is reported without imposing a
new global Fast forward-error target; the original raw comparison is retained.
"""
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
    parser.add_argument('--github-data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(XHDFE_GPU_BACKEND=args.backend, XHDFE_ABSORPTION_CACHE_MODE='off')
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rows = []
    # Stay above the established CPU Auto acceleration gate; below it the
    # dense small graph uses plain sweeps and does not exercise CG policy.
    for groups, cells in ((32, 32768), (1024, 32768)):
        cell = np.repeat(np.arange(cells), 8)
        signs = np.tile(np.arange(8), cells)
        a = cell % groups
        b = np.where(cell < groups, (a+1) % groups, a)
        c = (cell//groups) % 17
        X = np.asfortranarray(np.column_stack((np.where(signs & 1, 1., -1.),
                                               np.where(signs & 2, 1., -1.))))
        noise = np.where(signs & 4, .125, -.125)
        effects = 3+.125*a-.125*b+.25*c
        y = 1.25*X[:, 0]-.5*X[:, 1]+effects+noise
        # The first two dimensions form one cycle; the third contributes16
        # independent contrasts. Walsh columns/noise sum to zero in every cell.
        rank_fe = 2*groups+17-2
        expected = np.r_[1.25, -.5, effects.mean()]
        V = np.eye(3)*.015625/(len(y)-rank_fe-2)
        for mode in ('xhdfe-fast', 'reghdfe-comparable'):
            row = dict(case='walsh_near_nested', groups=groups, mode=mode)
            reg = core.HdfeRegressor(se_type='unadjusted', tol=1e-8, max_iter=1000,
                num_threads=1, drop_singletons=False, tolerance_mode=mode, convergence='reghdfe')
            try:
                reg.fit(y, X, fes=[f.astype(np.int32) for f in (a, b, c)])
                db = float(np.max(abs(reg.coef_-expected)))
                dv = float(np.max(abs(reg.covariance_-V))/V[0, 0])
                ok = (db <= (1e-8 if mode == 'xhdfe-fast' else 1e-9) and
                      np.allclose(reg.covariance_, V, atol=1e-12, rtol=1e-7) and
                      reg.converged_ and reg.precision_certified_ and
                      reg.nobs_ == len(y) and reg.df_resid_ == len(y)-rank_fe-2 and
                      bool(reg.gpu_used_) == (args.backend == 'cuda'))
                row.update(status='PASS' if ok else 'FAIL', b_error=db, V_scaled=dv,
                           iterations=int(reg.num_iterations_), gpu_used=bool(reg.gpu_used_))
            except RuntimeError as error:
                row.update(status='FAIL', error=str(error))
            rows.append(row)
    github = dict(status='NOT_RUN', reason='External core24 dataset not supplied')
    if args.github_data:
        reference_path = Path(__file__).with_name('fixtures')/'github_fast_aggregate_reference.json'
        ref = json.loads(reference_path.read_text())
        frame = pd.read_parquet(args.github_data, columns=ref['columns'])
        digest = hashlib.sha256()
        for name in ref['columns']:
            values = np.asarray(frame[name], dtype='<f8' if name in ('y', 'x1', 'x2') else '<i8')
            digest.update(name.encode())
            digest.update(np.asarray(values.shape, dtype='<i8').tobytes())
            digest.update(values.tobytes())
        assert digest.hexdigest() == ref['canonical_columns_sha256'], 'Different reference input'
        fes = [pd.factorize(frame[name], sort=True)[0].astype(np.int32)
               for name in ('id1', 'id2', 'id3')]
        reg = core.HdfeRegressor(se_type='cluster', tol=1e-8, max_iter=1000,
            num_threads=16, drop_singletons=True, tolerance_mode='xhdfe-fast',
            ssc_k_adj=False, ssc_g_adj=False)
        try:
            reg.fit(frame.y.to_numpy(), np.asfortranarray(frame[['x1', 'x2']].to_numpy()),
                    fes=fes, clusters=fes[0][:, None])
            b, V = np.asarray(ref['b']), np.asarray(ref['V_no_ssc'])
            db = float(np.max(abs(reg.coef_-b)))
            dv = float(np.max(abs(reg.covariance_-V)/np.sqrt(np.outer(V.diagonal(), V.diagonal()))))
            ok = (reg.converged_ and reg.precision_certified_ and db <= 1e-8 and
                  reg.nobs_ == ref['n_estimation'] and reg.num_iterations_ <= 1000 and
                  np.isfinite(reg.covariance_).all() and
                  np.all(np.diag(reg.covariance_) >= 0) and
                  bool(reg.gpu_used_) == (args.backend == 'cuda'))
            github = dict(status='PASS_RETURN_AND_COEFFICIENTS' if ok else 'FAIL',
                b_error=db, V_scaled_diagnostic=dv, iterations=int(reg.num_iterations_),
                gpu_used=bool(reg.gpu_used_),
                covariance_scope='DIAGNOSTIC_ONLY; Fast forward-V bounds are not redefined')
        except RuntimeError as error:
            github = dict(status='FAIL', error=str(error))
    passed = all(row['status'] == 'PASS' for row in rows) and github['status'] != 'FAIL'
    result = dict(status='PASS_SCOPED_FAST_POLICY' if passed else 'FAIL', backend=args.backend,
        module=str(args.module.resolve()), sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
        analytic_rows=rows, github=github)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))
    return not passed


if __name__ == '__main__':
    raise SystemExit(main())
