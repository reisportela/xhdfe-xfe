"""Independent full-parameter checks for the near-redundant CG design.

The oracle removes school effects exactly, then projects onto the remaining
small dummy design by SVD. It never calls an xhdfe or reghdfe solver. The
cluster sandwich includes the identified constant and its cross-covariances.
Small-sample adjustments are explicitly disabled in both the requested
estimator contract and the oracle; existing analytic cases cover default SSC.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


def fixture(outcome: str, weighted: bool):
    sid = np.repeat(np.arange(1536), 5)
    year = np.tile(np.arange(2015, 2020), 1536)
    first = np.select([sid < 1515, sid < 1522, sid < 1526, sid < 1532],
                      [0, 2015, 2016, 2017], default=2018)
    keep = year >= first
    sid, year, first = sid[keep], year[keep], first[keep]
    age = np.where(first == 0, 0, year-first+1)
    rng = np.random.default_rng(16)
    X = rng.normal(size=(len(sid), 2)) + [0.5, -0.3]
    if outcome == 'rare':
        y = ((sid < 2) | ((sid == 2) & (year != 2017)) |
             ((sid == 3) & (year == 2017))).astype(float)
    else:
        y = 10 + rng.normal(size=1536)[sid] + .03*(year-2017)
        y += .02*rng.normal(size=len(sid))
    weights = 1 + ((sid + 3*year) % 5)/4 if weighted else np.ones(len(sid))
    clusters = (sid % 53).astype(np.int64)
    fes = [v.astype(np.int32) for v in (sid, year-2015, age)]
    return y, X, fes, weights, clusters


def reference(y, X, fes, weights, clusters):
    sid, year, age = fes
    wsum = np.bincount(sid, weights=weights)

    def demean(values):
        means = np.column_stack([
            np.bincount(sid, weights=weights*values[:, j])/wsum
            for j in range(values.shape[1])])
        return values-means[sid]

    dummy = np.column_stack((np.eye(year.max()+1)[year],
                             np.eye(age.max()+1)[age]))
    rootw = np.sqrt(weights)
    reduced = rootw[:, None]*demean(dummy)
    U, singular, _ = np.linalg.svd(reduced, full_matrices=False)
    rank = np.count_nonzero(singular > singular[0]*1e-12)
    Q = U[:, :rank]
    values = rootw[:, None]*demean(np.column_stack((y, X)))
    projected = (values-Q@(Q.T@values))/rootw[:, None]
    yr, xr = projected[:, 0], projected[:, 1:]
    beta = np.linalg.lstsq(rootw[:, None]*xr, rootw*yr, rcond=None)[0]
    residual = yr-xr@beta
    means = np.average(X, axis=0, weights=weights)
    constant = np.average(y-X@beta, weights=weights)
    gram_inv = np.linalg.inv(xr.T@(weights[:, None]*xr))
    slope_influence = xr@gram_inv
    influence = np.column_stack((slope_influence,
                                  1/weights.sum()-slope_influence@means))
    scores = np.zeros((clusters.max()+1, X.shape[1]+1))
    np.add.at(scores, clusters, (weights*residual)[:, None]*influence)
    covariance = scores.T@scores
    # Necessary oracle checks and the original-input reconstruction.
    assert np.max(np.abs(xr.T@(weights*residual))) < 1e-9
    assert np.max(np.abs(np.bincount(sid, weights=weights*residual))) < 1e-9
    assert np.max(np.abs(dummy.T@(weights*residual))) < 1e-9
    assert np.isfinite(covariance).all() and np.all(covariance.diagonal() > 0)
    return np.r_[beta, constant], covariance, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', required=True, type=Path)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    os.environ.update(XHDFE_GPU_BACKEND=args.backend,
                      XHDFE_ABSORPTION_CACHE_MODE='off')
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rows = []
    for outcome in ('rare', 'aggregate'):
        for weighted in (False, True):
            y, X, fes, weights, clusters = fixture(outcome, weighted)
            expected, covariance, residual = reference(y, X, fes, weights, clusters)
            for permuted in (False, True):
                order = (np.random.default_rng(27).permutation(len(y))
                         if permuted else np.arange(len(y)))
                for retain in (False, True):
                    row = dict(outcome=outcome, weighted=weighted,
                               permuted=permuted, retain_fes=retain)
                    model = core.HdfeRegressor(
                        se_type='cluster', num_threads=args.threads,
                        tolerance_mode='reghdfe-comparable',
                        fit_intercept=True, drop_singletons=False,
                        retain_fes=retain, ssc_k_adj=False, ssc_g_adj=False)
                    kwargs = dict(fes=[f[order] for f in fes],
                                  clusters=clusters[order, None])
                    if weighted:
                        kwargs['weights'] = weights[order]
                    try:
                        model.fit(y[order], X[order], **kwargs)
                        actual_b = np.asarray(model.coef_)
                        actual_v = np.asarray(model.covariance_)
                        actual_r = np.asarray(model.residuals_)
                        b_error = float(np.max(np.abs(actual_b-expected)/
                                               np.maximum(1, np.abs(expected))))
                        v_error = float(np.max(np.abs(actual_v-covariance)/
                                               np.sqrt(np.outer(covariance.diagonal(),
                                                                covariance.diagonal()))))
                        r_error = float(np.max(np.abs(actual_r-residual[order])))
                        row.update(b_error=b_error, V_error=v_error,
                                   residual_error=r_error,
                                   iterations=int(model.num_iterations_),
                                   converged=bool(model.converged_),
                                   certified=bool(model.precision_certified_),
                                   gpu_used=bool(model.gpu_used_),
                                   gpu_status=int(model.gpu_status_code_),
                                   threads_used=int(model.threads_used_))
                        if retain:
                            effects = np.asarray(model.fe_effects_).sum(axis=0)
                            reconstruction = (y[order]-X[order]@actual_b[:-1]
                                              -actual_b[-1]-actual_r-effects)
                            row['reconstruction_error'] = float(np.max(np.abs(reconstruction)))
                            row['recovery_converged'] = bool(model.fe_recovery_converged_)
                        passed = (model.converged_ and model.precision_certified_
                                  and model.nobs_ == len(y)
                                  and bool(model.gpu_used_) == (args.backend == 'cuda')
                                  and b_error <= 1e-9 and v_error <= 1e-8
                                  and r_error <= 1e-7
                                  and (not retain or (row['recovery_converged']
                                                      and row['reconstruction_error'] <= 1e-6)))
                        row['status'] = 'PASS' if passed else 'FAIL'
                    except Exception as error:
                        row.update(status='ERROR', error=repr(error))
                    rows.append(row)
    result = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
                  backend=args.backend, requested_threads=args.threads, rows=rows)
    result['status'] = 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL'
    if args.output is not None:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'status': result['status'], 'cases': len(rows),
                      'failed': [r for r in rows if r['status'] != 'PASS']}))
    return result['status'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
