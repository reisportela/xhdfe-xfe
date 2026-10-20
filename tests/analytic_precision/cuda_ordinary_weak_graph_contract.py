"""CUDA04: ordinary weak FE graphs against explicit dummy OLS, offline only.

The deterministic 200000-row design comes from tests/audit-20260804.R.
Frozen 2.29.1 CUDA returns a certified fit with coefficient error 2.7e-9,
above the existing 1e-9 contract. Neither package supplies the oracle.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

import mpmath as mp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import splu


def fixture():
    row = np.arange(200000)
    f = (row % 20000).astype(np.int32)
    g = ((row * 37) % 17003).astype(np.int32)
    X = np.asfortranarray(np.column_stack((np.sin(row*.001), np.cos(row*.0013))))
    y = .6*X[:, 0] - .25*X[:, 1] + .001*f - .0005*g + np.sin(row*.0021)
    return y, X, [f, g]


def oracle(y, X, fes):
    """Anchored sparse dummy solve with long-double residual refinement."""
    f, g = fes
    n = len(y)
    row = np.arange(n)
    cross = sp.coo_matrix((np.ones(n), (f, g)), shape=(20000, 17003)).tocsr()
    adjacency = sp.bmat([[None, cross], [cross.T, None]], format='csr')
    assert connected_components(adjacency, directed=False, return_labels=False) == 1
    radius = int(shortest_path(adjacency, directed=False, unweighted=True,
                               indices=37002).max())
    keep = g != 17002
    D = sp.coo_matrix((np.ones(n + sum(keep)),
                      (np.r_[row, row[keep]], np.r_[f, 20000 + g[keep]])),
                     shape=(n, 37002)).tocsr()
    S = (D.T @ D).tocsc()
    lu = splu(S, permc_spec='MMD_AT_PLUS_A')
    DL, SL = D.astype(np.longdouble), S.astype(np.longdouble)
    A = np.column_stack((y, X)).astype(np.longdouble)
    rhs = DL.T @ A
    alpha = lu.solve(np.asarray(rhs, float)).astype(np.longdouble)
    for _ in range(4):
        alpha += lu.solve(np.asarray(rhs - SL @ alpha, float)).astype(np.longdouble)
    Q = A - DL @ alpha
    x, response = Q[:, 1:], Q[:, 0]
    G, h = x.T @ x, x.T @ response
    with mp.workdps(80):
        gram = mp.matrix([[mp.mpf(str(v)) for v in r] for r in G])
        bread = gram**-1
        solution = bread * mp.matrix([mp.mpf(str(v)) for v in h])
        b = np.array([np.longdouble(str(v)) for v in solution])
        residual = response - x @ b
        df = n - 37002 - X.shape[1]
        sigma = residual @ residual / df
        covariance = mp.mpf(str(sigma)) * bread
        V = np.zeros((3, 3), np.longdouble)
        for i in range(2):
            for j in range(2):
                V[i, j] = np.longdouble(str(covariance[i, j]))
    means = A[:, 1:].mean(axis=0)
    intercept = A[:, 0].mean() - means @ b
    V[:2, 2] = V[2, :2] = -V[:2, :2] @ means
    V[2, 2] = sigma/n + means @ V[:2, :2] @ means
    # Bound the dummy projection's uncertainty independently of any package
    # discrepancy: lambda_min(D'D) >= 1/(rank * anchored graph radius).
    unit_roundoff = np.finfo(np.longdouble).eps/2
    operations = int(max(np.asarray(D.sum(axis=0)).ravel().max(),
                         np.diff(S.tocsr().indptr).max())) + 2
    gamma = operations*unit_roundoff/(1-operations*unit_roundoff)
    arithmetic = gamma*(abs(DL).T @ abs(A) + abs(SL) @ abs(alpha))
    arithmetic += unit_roundoff*(abs(rhs) + abs(SL @ alpha))
    defect = abs(rhs - SL @ alpha) + arithmetic
    uncertainty = np.sqrt(np.sum(defect**2, axis=0))*np.sqrt(np.longdouble(37002*radius))
    assert np.max(uncertainty) < 1e-8, 'Oracle projection unresolved'
    return np.r_[b, intercept].astype(float), V.astype(float), dict(
        n=n, rank_fe=37002, df=df, graph_radius=radius,
        projection_uncertainty=[float(v) for v in uncertainty])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(XHDFE_GPU_BACKEND=args.backend, XHDFE_ABSORPTION_CACHE_MODE='off')
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    y, X, fes = fixture()
    b, V, reference = oracle(y, X, fes)
    rows = []
    # The explicit CG criterion reaches the same weak graph through its cold
    # restart. It must also meet the ordinary output contract.
    reg = core.HdfeRegressor(se_type='unadjusted', tol=1e-8, num_threads=1,
        max_iter=100000, drop_singletons=False,
        tolerance_mode='reghdfe-comparable', convergence='reghdfe')
    reg.fit(y, X, fes=fes)
    db = float(np.max(abs(reg.coef_ - b)/np.maximum(1, abs(b))))
    dv = float(np.max(abs(reg.covariance_ - V)/np.sqrt(
        np.outer(V.diagonal(), V.diagonal()))))
    ok = (db <= 1e-9 and dv <= 1e-8 and reg.converged_ and
          reg.precision_certified_ and reg.df_resid_ == reference['df'] and
          bool(reg.gpu_used_) == (args.backend == 'cuda'))
    rows.append(dict(case='explicit_cg_restart', b_scaled=db, V_scaled=dv,
        iterations=int(reg.num_iterations_), method=int(reg.absorption_method_used),
        gpu_used=bool(reg.gpu_used_), status='PASS' if ok else 'FAIL'))
    for threads in (1, 2, 8, 16, 24, 48):
        for name, ys, xs, order in [('original', 1., np.ones(2), [0, 1]),
                                    ('rescaled_reordered', 8., np.array([.5, 4.]), [1, 0])]:
            reg = core.HdfeRegressor(se_type='unadjusted', tol=1e-8,
                num_threads=threads, max_iter=100000, drop_singletons=False,
                tolerance_mode='reghdfe-comparable')
            reg.fit(y*ys, np.asfortranarray(X*xs), fes=[fes[i] for i in order])
            scale = np.r_[ys/xs, ys]
            expected_b, expected_V = b*scale, V*np.outer(scale, scale)
            db = float(np.max(abs(reg.coef_ - expected_b)/np.maximum(1, abs(expected_b))))
            dv = float(np.max(abs(reg.covariance_ - expected_V)/np.sqrt(
                np.outer(expected_V.diagonal(), expected_V.diagonal()))))
            ok = (db <= 1e-9 and dv <= 1e-8 and reg.converged_ and
                  reg.precision_certified_ and reg.nobs_ == reference['n'] and
                  reg.df_resid_ == reference['df'] and
                  bool(reg.gpu_used_) == (args.backend == 'cuda'))
            rows.append(dict(case=name, threads=threads, b_scaled=db, V_scaled=dv,
                iterations=int(reg.num_iterations_), method=int(reg.absorption_method_used),
                gpu_used=bool(reg.gpu_used_), df=int(reg.df_resid_),
                status='PASS' if ok else 'FAIL'))
    # A duplicated-edge tree has an exact cell-demeaning solution. Keep that
    # existing shortcut ahead of iterative sparse-graph routing.
    cells = np.r_[np.arange(10000), [0, 2, 4]]
    f = (cells//2).astype(np.int32)
    g = ((cells+1)//2).astype(np.int32)
    row = np.arange(len(f))
    tx = np.column_stack((np.sin(row*.071), np.cos(row*.093)))
    ty = .7*tx[:, 0] - .2*tx[:, 1] + f*.01 - g*.003 + np.sin(row*.131)
    raw = np.column_stack((ty, tx))
    count = np.bincount(cells)
    means = np.column_stack([np.bincount(cells, weights=raw[:, j])/count
                             for j in range(3)])
    centred = raw - means[cells]
    expected = np.linalg.lstsq(centred[:, 1:], centred[:, 0], rcond=None)[0]
    expected = np.r_[expected, ty.mean() - tx.mean(axis=0) @ expected]
    reg = core.HdfeRegressor(tol=1e-8, num_threads=1, drop_singletons=False,
                            tolerance_mode='reghdfe-comparable')
    reg.fit(ty, np.asfortranarray(tx), fes=[f, g])
    error = float(np.max(abs(reg.coef_ - expected)))
    ok = (error <= 1e-9 and reg.converged_ and reg.precision_certified_ and
          bool(reg.gpu_used_) == (args.backend == 'cuda') and
          (args.backend != 'cuda' or int(reg.absorption_method_used) != 5))
    rows.append(dict(case='exact_tree_shortcut', error=error,
        method=int(reg.absorption_method_used), iterations=int(reg.num_iterations_),
        gpu_used=bool(reg.gpu_used_), status='PASS' if ok else 'FAIL'))
    # A budget refusal is a separate outcome; no stale estimates may survive.
    budgets = [('auto', 1)] + ([('reghdfe', 128)] if args.backend == 'cuda' else [])
    for criterion, maximum in budgets:
        reg = core.HdfeRegressor(tol=1e-8, num_threads=1, max_iter=maximum,
            drop_singletons=False, tolerance_mode='reghdfe-comparable', convergence=criterion)
        try:
            reg.fit(y, X, fes=fes)
            rows.append(dict(case='budget_'+criterion, status='FAIL',
                             reason='Unexpected successful fit'))
        except RuntimeError as error:
            message = str(error)
            match = re.search(r'iterations=(\d+)/(\d+)', message)
            stale = any(np.asarray(getattr(reg, key)).size for key in
                        ('coef_', 'covariance_', 'residuals_'))
            ok = (match is not None and int(match[1]) <= maximum and
                  int(match[2]) == maximum and not stale and not reg.converged_
                  and not reg.precision_certified_)
            rows.append(dict(case='budget_'+criterion, max_iter=maximum,
                status='EXPECTED_REFUSAL' if ok else 'FAIL', error=message, stale_estimates=stale))
    result = dict(module=str(args.module.resolve()),
        module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
        backend=args.backend, oracle=reference, rows=rows,
        status='FAIL' if any(r['status'] == 'FAIL' for r in rows) else 'PASS')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))
    return result['status'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
