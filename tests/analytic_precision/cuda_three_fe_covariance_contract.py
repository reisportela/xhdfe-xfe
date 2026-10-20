"""Ordinary three-FE cluster covariance against explicit dummy OLS, offline only.

The frozen 2.30.0 CUDA Comparable route fails the rescaled synthetic fixture
by roughly 5e-6 in diagonal-scaled full covariance. No package is the oracle.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def fixture():
    rng = np.random.default_rng(20261020)
    workers, periods, firms = 6000, 40, 80
    n = workers*periods
    worker = np.repeat(np.arange(workers, dtype=np.int32), periods)
    year = np.tile(np.arange(periods, dtype=np.int32), workers)
    mover = (worker//firms)%5 == 0
    firm = ((worker%firms)+mover*(year >= 16)).astype(np.int32)%firms
    age = rng.uniform(20, 45, workers)[worker]+year/2
    tenure = rng.uniform(0, 10, workers)[worker]+year/2-8*mover*(year >= 16)
    education = (rng.integers(6, 16, workers)[worker]
                 +.2*(year >= rng.integers(8, 30, workers)[worker])
                 +.01*rng.normal(size=n))
    x = np.column_stack([age**2, tenure, tenure**2, education])
    y = 2+x@np.array([-.0003, .01, -.0002, .007])+rng.normal(size=workers)[worker]
    y += 2*np.sin(2*np.pi*firm/firms)+.3*np.cos(year/4)+.2*rng.normal(size=n)
    return y, x, [worker, firm, year]


def oracle(y, x, fes):
    """Anchored original dummy system, wide residual refinement and sandwich."""
    w, f, t = fes
    n, p = x.shape
    nw, nf, nt = (int(v.max())+1 for v in fes)
    rows = np.arange(n)
    fi, ti = f != 0, t != 0
    d = sp.coo_matrix((np.ones(n+sum(fi)+sum(ti)),
        (np.r_[rows, rows[fi], rows[ti]],
         np.r_[w, nw+f[fi]-1, nw+nf-1+t[ti]-1])),
         shape=(n, nw+nf+nt-2)).tocsr()
    s = (d.T@d).tocsc()
    lu = splu(s, permc_spec='MMD_AT_PLUS_A')
    dl, sl = d.astype(np.longdouble), s.astype(np.longdouble)
    a = np.column_stack([y, x]).astype(np.longdouble)
    rhs = dl.T@a
    alpha = lu.solve(np.asarray(rhs, float)).astype(np.longdouble)
    for _ in range(4):
        alpha += lu.solve(np.asarray(rhs-sl@alpha, float)).astype(np.longdouble)
    within = a-dl@alpha
    defects = dl.T@within
    counts = np.asarray(dl.sum(axis=0)).ravel()
    relative = np.sqrt(np.sum(defects**2/counts[:, None], axis=0)/np.sum(a*a, axis=0))
    assert np.max(relative) < 5e-14, 'Independent projection unresolved'
    qx, qy = within[:, 1:], within[:, 0]
    gram, h = qx.T@qx, qx.T@qy
    with mp.workdps(80):
        g = mp.matrix([[mp.mpf(str(v)) for v in row] for row in gram])
        inverse = g**-1
        beta = inverse*mp.matrix([mp.mpf(str(v)) for v in h])
        b = np.array([np.longdouble(str(v)) for v in beta])
        residual = qy-qx@b
        scores = np.zeros((nw, p+1), np.longdouble)
        np.add.at(scores, w, np.column_stack([qx*residual[:, None], residual]))
        meat = scores.T@scores
        m = mp.matrix([[mp.mpf(str(v)) for v in row] for row in meat])
        means = a.mean(axis=0)
        transport = mp.zeros(p+1)
        for j in range(p):
            for k in range(p):
                transport[j, k] = inverse[j, k]
                transport[p, k] -= mp.mpf(str(means[j+1]))*inverse[j, k]
        transport[p, p] = mp.mpf(1)/n
        df_a = nf+nt-2
        adjustment = mp.mpf(nw)/(nw-1)*mp.mpf(n-1)/(n-p-df_a-1)
        v = adjustment*transport*m*transport.T
        constant = mp.mpf(str(means[0]))-sum(mp.mpf(str(means[j+1]))*beta[j] for j in range(p))
        full_b = np.array([float(vv) for vv in beta]+[float(constant)])
        full_v = np.array([[float(v[j,k]) for k in range(p+1)] for j in range(p+1)])
    return full_b, full_v, dict(n=n, df_a=df_a, df_resid=nw-1,
        rank_fe=nw+nf+nt-2, relative_projection=[float(v) for v in relative],
        solver='Explicit sparse dummy normal equations, four wide refinements, MP80 inference')


def metrics(b, v, expected_b, expected_v):
    if not np.isfinite(b).all() or not np.isfinite(v).all():
        return dict(b_scaled=float('inf'), V_scaled=float('inf'))
    return dict(b_scaled=float(np.max(abs(b-expected_b)/np.maximum(1, abs(expected_b)))),
        V_scaled=float(np.max(abs(v-expected_v)/np.sqrt(np.outer(expected_v.diagonal(), expected_v.diagonal())))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', default='1,2,8,16,24,48')
    args = parser.parse_args()
    assert not args.output.exists()
    for key in list(os.environ):
        if key.startswith('XHDFE_'):
            del os.environ[key]
    os.environ.update(XHDFE_GPU_BACKEND=args.backend, XHDFE_ABSORPTION_CACHE_MODE='off',
                      XHDFE_MOBILITY_MODE='off', XHDFE_FE_STRUCTURE_MODE='off')
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    y, x, fes = fixture()
    b, v, reference = oracle(y, x, fes)
    bad = v.copy(); bad[-1, -1] *= 1+1e-6
    assert metrics(b, bad, b, v)['V_scaled'] > 1e-8
    permutation = np.random.default_rng(2718).permutation(len(y))
    cases = [('original', 1., np.ones(4), [0,1,2], None),
             ('balanced', .5, np.array([2**-11, 2**-4, 2**-8, 2**-3]), [0,1,2], None),
             ('permuted_reordered', 8., np.array([.25,16.,2**-6,32.]), [2,0,1], permutation)]
    rows = []
    for threads in [int(v) for v in args.threads.split(',')]:
        for name, sy, sx, order, perm in cases:
            indices = np.arange(len(y)) if perm is None else perm
            reg = core.HdfeRegressor(se_type='cluster', tol=1e-8, max_iter=100000,
                num_threads=threads, drop_singletons=True, retain_fes=False,
                symmetric_sweep=False, absorption_method='auto',
                tolerance_mode='reghdfe-comparable', jacobi_relaxation=0.)
            error = None
            try:
                reg.fit(y[indices]*sy, np.asfortranarray(x[indices]*sx),
                        fes=[fes[k][indices] for k in order], clusters=fes[0][indices])
            except Exception as exc:
                error = str(exc)
            row = dict(case=name, threads=threads, error=error, status='FAIL')
            if error is None:
                undo = np.r_[sx/sy, 1/sy]
                difference = metrics(np.asarray(reg.coef_)*undo,
                    np.asarray(reg.covariance_)*np.outer(undo, undo), b, v)
                ok = (difference['b_scaled'] <= 1e-9 and difference['V_scaled'] <= 1e-8
                    and reg.converged_ and reg.precision_certified_
                    and reg.nobs_ == reference['n'] and reg.df_a_ == reference['df_a']
                    and reg.df_resid_ == reference['df_resid']
                    and bool(reg.gpu_used_) == (args.backend == 'cuda'))
                row.update(difference, iterations=int(reg.num_iterations_),
                    threads_used=int(reg.threads_used_), gpu_used=bool(reg.gpu_used_),
                    status='PASS' if ok else 'FAIL')
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
        reference=reference, oracle_b=b.tolist(), oracle_V=v.tolist(), rows=rows,
        backend=args.backend, module=str(args.module.resolve()),
        module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
        negative_control='Injected recovered-constant variance error rejected')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
