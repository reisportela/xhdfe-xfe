#!/usr/bin/env python3
"""Recovered-constant covariance when one cluster nests an absorbed intercept FE.

If the only cluster dimension is a union of the levels of an absorbed intercept
dimension (a plain FE, or a slope term that keeps its level dummies, i.d##c.z),
the normal equations of those level dummies make every weighted cluster
residual sum exactly zero. The covariance of the recovered constant
_cons = ybar - xbar'b then satisfies, in exact arithmetic,

    V(_cons,_cons) = xbar' V_bb xbar    and    V(_cons,b) = -V_bb xbar,

with V_bb the reported slope covariance and xbar the weighted means on the
estimation sample; both are exactly zero with no regressors or xbar = 0. The
oracle is this analytic identity evaluated exactly (dyadic rationals for the
reported V_bb and the sample means); no package is the oracle. The acceptance
margin is the arithmetic allowance of the operations that form the reported
values, under a predeclared model (Higham's gamma_m with binary64 u = 2^-53,
and the long double means of fixed_chunk_weighted_mean with u = 2^-64 on
x86_64). It contains no iterative error and is never derived from observed
deviations. Exact-zero rows (xbar = 0 by construction, or no identified
regressor) require V(_cons,.) = 0, SE(_cons) = 0 and missing t, p and CI.

Every row requires a converged and precision-certified fit. A refusal in either
build is never a PASS: identical refusals are UNSUPPORTED, others FAIL. On
deterministic one-thread CPU routes the slope block, b and the DoF must equal
the reference build bit for bit, and controls (non-nested cluster, robust,
multi-way, IV, xhdfe-fast, slope-only nested dimension, a saturated design)
must reproduce the whole reference fit. On CUDA, which is never
bit-reproducible, and on multithreaded rows, b and V_bb are graded by the
project's Comparable contracts against the reference (|db_j| <= 1e-9 max(1,|b_j|),
|dV_jk| <= 1e-8 sqrt(V_jj V_kk)), the DoF exactly, and a CUDA control must not
satisfy the identity where it does not hold. Fewer than two clusters keep the
missing-inference convention (diagonal sentinel -1e-10, missing SE/t/p/CI).
The reference build must violate the identity on the public
synthetic-assortative cluster(year) case at tol 1e-8 (positive control, error
recorded on 08nov2026), so the check is live. Data are read in place; only
coefficients, covariances, means and diagnostics are persisted.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True

U = 2.0 ** -53
U_LD = 2.0 ** -64 if platform.machine() in ('x86_64', 'AMD64') else U
DEFAULT_DATA = Path('/home/mangelo/Documents/BigData/Sergio/all-dta')
B_CONTRACT = 1e-9      # Comparable coefficient contract, |db_j| <= 1e-9 max(1,|b_j|)
V_CONTRACT = 1e-8      # full-covariance contract, |dV_jk| <= 1e-8 sqrt(V_jj V_kk)
SENTINEL = -1.0e-10    # mark_inference_unavailable diagonal


def gamma(m, unit=U):
    return m * unit / (1.0 - m * unit)


# ----------------------------------------------------------------------------
# Fixtures (regenerated identically in every worker).


def panel(np, seed=20261108, workers=4000, periods=10, firms=120):
    rng = np.random.default_rng(seed)
    n = workers * periods
    worker = np.repeat(np.arange(workers, dtype=np.int64), periods)
    year = np.tile(np.arange(periods, dtype=np.int64), workers)
    mover = (worker // 7) % 3 == 0
    firm = ((worker % firms) + mover * (year >= periods // 2) * (1 + worker % 11)) % firms
    a = rng.normal(size=workers)[worker]
    f = rng.normal(size=firms)[firm]
    t = rng.normal(size=periods)[year]
    x1 = .5 * a + .3 * f + .2 * t + rng.normal(size=n)
    x2 = .002 + .01 * f + rng.normal(size=n)
    z = rng.uniform(.5, 1.5, size=n)
    y = (1. + .7 * x1 - .4 * x2 + a + 2. * f + .5 * t + .3 * z * a
         + rng.normal(size=n) * (.5 + .5 * (year % 3)))
    inst = x1 + rng.normal(size=n)
    # Exactly balanced +-1 regressors outside the FE span (products of worker
    # and year functions): their sums are exact integers, zero on every sample.
    e1 = np.where((worker + year) % 2 == 0, 1., -1.)
    e2 = np.where(((worker // 2) + (year // 5)) % 2 == 0, 1., -1.)
    assert e1.sum() == 0 and e2.sum() == 0
    return dict(
        y=y, X=np.column_stack([x1, x2]), X_zero_mean=np.column_stack([e1, e2]),
        X_empty=np.empty((n, 0)),
        fes=[worker, firm, year],
        clusters=dict(worker=worker, year=year, firmgroup=firm // 10,
                      cross=(7 * worker + 3 * year) % 50, single=np.zeros(n, dtype=np.int64)),
        aweight=rng.uniform(.5, 2., size=n),
        fweight=rng.integers(1, 4, size=n).astype(float),
        z=z, inst=inst[:, None],
        absorbed_constant=a[:, None])


def tiny(np):
    """Saturated with dofadjustments(pairwise): N = 6 = k + rank(D) = 2 + 4."""
    rng = np.random.default_rng(611)
    g1 = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    g2 = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    return dict(y=rng.normal(size=6), X=rng.normal(size=(6, 2)), fes=[g1, g2],
                clusters=dict(g1=g1))


def public(np, data_dir, name):
    import pandas as pd
    path = data_dir / f'{name}.parquet'
    frame = pd.read_parquet(path, columns=(['y', 'x1', 'x2', 'id1', 'id2'] +
                                           (['year'] if name == 'synthetic-assortative' else [])))
    fes = [frame[c].to_numpy(np.int64) for c in
           (['id1', 'id2', 'year'] if name == 'synthetic-assortative' else ['id1', 'id2'])]
    clusters = {c: frame[c].to_numpy(np.int64) for c in
                (['id1', 'id2', 'year'] if name == 'synthetic-assortative' else ['id1', 'id2'])}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(y=frame['y'].to_numpy(float),
                X=np.column_stack([frame['x1'].to_numpy(float), frame['x2'].to_numpy(float)]),
                fes=fes, clusters=clusters, sha256=digest)


# ----------------------------------------------------------------------------
# Rows. kind: 'identity' (the identity must hold), 'control' (generic VCE kept)
# or 'one_cluster' (missing-inference convention for fewer than two clusters).


def synthetic_rows(backend):
    rows = []
    def add(name, kind, cluster='worker', method='auto', tol=1e-8, threads=1, **extra):
        rows.append(dict(name=name, kind=kind, data=extra.pop('data', 'panel'), cluster=cluster,
                         method=method, tol=tol, threads=threads, **extra))
    if backend == 'cpu':
        for cluster in ['worker', 'year']:
            for method in ['gs', 'mlsmr', 'lsmr', 'auto']:
                for tol in [1e-8, 1e-10]:
                    add(f'nested_{cluster}_{method}_{tol:g}', 'identity', cluster, method, tol)
        for method in ['gs', 'auto']:
            for tol in [1e-8, 1e-10]:
                add(f'nested_firmgroup_{method}_{tol:g}', 'identity', 'firmgroup', method, tol)
        add('slopes_on_other_dim', 'identity', slopes=[(1, 'z', False)])
        # i.d##c.z keeps the level dummies of the nested dimension (round 2).
        for tol in [1e-8, 1e-10]:
            add(f'intercept_slope_nested_worker_{tol:g}', 'identity', 'worker', 'auto', tol,
                slopes=[(0, 'z', True)])
        add('intercept_slope_nested_firmgroup', 'identity', 'firmgroup', slopes=[(1, 'z', True)])
        add('aweights', 'identity', weights='aweight')
        add('aweights_lsmr', 'identity', 'worker', 'lsmr', 1e-10, weights='aweight')
        # Forced GS with these analytic or frequency weights is refused by the
        # absorption certificate in both builds (e.g. 1.8e-8 > 8e-9 at tol 1e-8 with
        # fweights; 5.2e-7 at tol 1e-10 with aweights; 08nov2026): not a P1b case.
        add('fweights', 'identity', 'year', weights='fweight', fweights=True)
        add('pweights', 'identity', 'firmgroup', 'mlsmr', weights='aweight', pweights=True)
        add('savefe', 'identity', retain_fes=True)
        add('dof_without_cluster_scan', 'identity', dofadjustments='pairwise')
        add('strict_residual', 'identity', 'year', mode='strict-residual')
        add('threads16', 'identity', 'year', threads=16)
        # Exact zeros: xbar = 0 by construction, and no identified regressor.
        add('xbar_zero_worker_auto', 'identity', regressors='X_zero_mean', exact_zero=True)
        add('xbar_zero_year_gs_1e-10', 'identity', 'year', 'gs', 1e-10, regressors='X_zero_mean',
            exact_zero=True)
        add('k0_no_regressors', 'identity', regressors='X_empty', exact_zero=True)
        add('k0_omitted_regressor', 'identity', regressors='absorbed_constant', exact_zero=True)
        for method in ['gs', 'mlsmr', 'lsmr', 'auto']:
            for tol in [1e-8, 1e-10]:
                add(f'control_cross_{method}_{tol:g}', 'control', 'cross', method, tol,
                    identity_false=True)
        add('control_robust', 'control', se='robust', identity_false=True)
        add('control_multiway', 'control', cluster=['worker', 'year'])
        add('control_fast', 'control', mode='xhdfe-fast')
        add('control_iv', 'control', iv=True)
        add('control_slope_only_nested', 'control', slopes=[(0, 'z', False)])
        add('control_saturated', 'control', 'g1', data='tiny', dofadjustments='pairwise')
        add('one_cluster_k0', 'one_cluster', 'single', regressors='X_empty')
        add('one_cluster', 'one_cluster', 'single')
    else:
        # absorptionmethod(lsmr/mlsmr) is CPU-only and refused under the CUDA backend.
        for cluster in ['worker', 'year']:
            for method in ['auto', 'gs']:
                for tol in [1e-8, 1e-10]:
                    add(f'cuda_nested_{cluster}_{method}_{tol:g}', 'identity', cluster, method, tol,
                        threads=0)
        for tol in [1e-8, 1e-10]:
            add(f'cuda_nested_firmgroup_auto_{tol:g}', 'identity', 'firmgroup', 'auto', tol, threads=0)
        add('cuda_aweights', 'identity', weights='aweight', threads=0)
        add('cuda_savefe', 'identity', retain_fes=True, threads=0)
        add('cuda_aweights_year', 'identity', 'year', weights='aweight', threads=0)
        add('cuda_fweights_year', 'identity', 'year', weights='fweight', fweights=True, threads=0)
        add('cuda_pweights_year', 'identity', 'year', weights='aweight', pweights=True, threads=0)
        add('cuda_xbar_zero_worker', 'identity', regressors='X_zero_mean', exact_zero=True, threads=0)
        add('cuda_xbar_zero_year', 'identity', 'year', regressors='X_zero_mean', exact_zero=True,
            threads=0)
        add('cuda_k0_no_regressors', 'identity', regressors='X_empty', exact_zero=True, threads=0)
        add('cuda_k0_omitted_regressor', 'identity', regressors='absorbed_constant',
            exact_zero=True, threads=0)
        for tol in [1e-8, 1e-10]:
            add(f'cuda_control_cross_auto_{tol:g}', 'control', 'cross', 'auto', tol, threads=0,
                identity_false=True)
        add('cuda_control_robust', 'control', se='robust', threads=0, identity_false=True)
    return rows


def public_rows(backend):
    rows = []
    def add(data, cluster, method, tol, positive=False):
        rows.append(dict(name=f'{"cuda_" if backend == "cuda" else ""}{data}_{cluster}_{method}_{tol:g}',
                         kind='identity', data=data, cluster=cluster, method=method, tol=tol,
                         threads=1 if backend == 'cpu' else 0, positive_control=positive))
    a, p = 'synthetic-assortative', 'patents'
    if backend == 'cpu':
        add(a, 'year', 'auto', 1e-8, positive=True)
        add(a, 'year', 'auto', 1e-10)
        add(a, 'year', 'mlsmr', 1e-8)
        add(a, 'id1', 'auto', 1e-8)
        add(a, 'id1', 'auto', 1e-10)
        add(a, 'id2', 'auto', 1e-8)
        add(p, 'id1', 'auto', 1e-8)
        add(p, 'id1', 'auto', 1e-10)
        add(p, 'id2', 'auto', 1e-8)
        add(p, 'id2', 'auto', 1e-10)
    else:
        add(a, 'year', 'auto', 1e-8, positive=True)
        add(a, 'year', 'auto', 1e-10)
        add(a, 'id1', 'auto', 1e-8)
        add(p, 'id1', 'auto', 1e-8)
        add(p, 'id2', 'auto', 1e-8)
    return rows


# ----------------------------------------------------------------------------
# Exact dyadic arithmetic for the sample means.


def exact_sum(np, values):
    """Exact sum of binary64 values as a Fraction."""
    values = np.asarray(values, dtype=np.float64)
    mantissa, exponent = np.frexp(values)
    m = (mantissa * 2.0 ** 53).astype(np.int64)
    e = exponent.astype(np.int64) - 53
    keep = m != 0
    m, e = m[keep], e[keep]
    if m.size == 0:
        return Fraction(0)
    base = int(e.min())
    total = 0
    for exp in np.unique(e):
        total += sum(m[e == exp].tolist()) << int(exp - base)
    return Fraction(total) * (Fraction(2) ** base)


def exact_dot(np, a, b):
    """Exact sum of products of two binary64 vectors."""
    ma, ea = np.frexp(np.asarray(a, dtype=np.float64))
    mb, eb = np.frexp(np.asarray(b, dtype=np.float64))
    ia = (ma * 2.0 ** 53).astype(np.int64)
    ib = (mb * 2.0 ** 53).astype(np.int64)
    e = ea.astype(np.int64) + eb.astype(np.int64) - 106
    keep = (ia != 0) & (ib != 0)
    ia, ib, e = ia[keep], ib[keep], e[keep]
    if ia.size == 0:
        return Fraction(0)
    base = int(e.min())
    total = 0
    for exp in np.unique(e):
        sel = e == exp
        total += sum(x * y for x, y in zip(ia[sel].tolist(), ib[sel].tolist())) << int(exp - base)
    return Fraction(total) * (Fraction(2) ** base)


def frac_str(value):
    return f'{value.numerator}/{value.denominator}'


def parse_frac(text):
    num, den = text.split('/')
    return Fraction(int(num), int(den))


def hexes(values):
    return [float(v).hex() for v in values]


# ----------------------------------------------------------------------------
# Worker: one module per process (pybind11 registers its types once).


def worker(spec_path, result_path):
    import importlib.util
    import numpy as np
    spec = json.loads(Path(spec_path).read_text())
    for key in list(os.environ):
        if key.startswith('XHDFE_'):
            del os.environ[key]
    os.environ.update(XHDFE_GPU_BACKEND=spec['backend'], XHDFE_ABSORPTION_CACHE_MODE='off',
                      XHDFE_MOBILITY_MODE='off', XHDFE_FE_STRUCTURE_MODE='off',
                      XHDFE_AUTO_TEAM_COORDINATION='0', CUDA_CACHE_DISABLE='1',
                      OPENBLAS_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
    module = importlib.util.spec_from_file_location('py_hdfe_v11', spec['module'])
    core = importlib.util.module_from_spec(module)
    module.loader.exec_module(core)
    cache = {}
    out = []
    for row in spec['rows']:
        if row['data'] not in cache:
            cache.clear()
            cache[row['data']] = (panel(np) if row['data'] == 'panel' else tiny(np)
                                  if row['data'] == 'tiny'
                                  else public(np, Path(spec['data_dir']), row['data']))
        data = cache[row['data']]
        X = data[row.get('regressors', 'X')]
        cl = row['cluster']
        clusters = (np.column_stack([data['clusters'][c] for c in cl]) if isinstance(cl, list)
                    else data['clusters'][cl])
        weights = data[row['weights']] if row.get('weights') else None
        slopes = [(i, data[v], inc) for i, v, inc in row.get('slopes', [])] or None
        kwargs = dict(se_type=row.get('se', 'cluster'), tol=row['tol'], max_iter=100000,
                      num_threads=row['threads'], drop_singletons=True,
                      retain_fes=bool(row.get('retain_fes', False)), symmetric_sweep=False,
                      absorption_method=row['method'],
                      tolerance_mode=row.get('mode', 'reghdfe-comparable'),
                      jacobi_relaxation=0.)
        if row.get('dofadjustments'):
            kwargs['dofadjustments'] = row['dofadjustments']
        result = dict(name=row['name'])
        started = time.time()
        try:
            reg = core.HdfeRegressor(**kwargs)
            fit = dict(fes=data['fes'], clusters=clusters, weights=weights, slopes=slopes,
                       fweights=bool(row.get('fweights', False)),
                       pweights=bool(row.get('pweights', False)))
            if row.get('iv'):
                fit.update(instruments=data['inst'], endogenous_idx=[0])
            reg.fit(data['y'], np.asfortranarray(X), **fit)
        except Exception as exc:  # recorded and graded, never swallowed silently
            result.update(error=f'{type(exc).__name__}: {exc}')
            out.append(result)
            continue
        coef = np.asarray(reg.coef_, dtype=float)
        cov = np.asarray(reg.covariance_, dtype=float)
        omitted = list(map(bool, getattr(reg, 'omitted_', [False] * X.shape[1])))
        sample = np.asarray(reg.sample_index_, dtype=np.int64)
        w = None if weights is None else np.asarray(weights, dtype=float)[sample]
        means, abs_means = [], []
        sum_w = exact_sum(np, w) if w is not None else Fraction(len(sample))
        for j in range(X.shape[1]):
            xj = np.asarray(X[:, j], dtype=float)[sample]
            num = exact_dot(np, xj, w) if w is not None else exact_sum(np, xj)
            means.append(frac_str(num / sum_w))
            mag = (math.fsum(np.abs(xj * w)) if w is not None else math.fsum(np.abs(xj)))
            abs_means.append(mag / float(sum_w) * (1 + 1e-12))
        ci = np.asarray(reg.conf_int_, dtype=float).reshape(len(coef), -1)
        result.update(
            coef=hexes(coef), cov=[hexes(r) for r in cov],
            se=hexes(np.asarray(reg.stderr_, dtype=float)),
            t=hexes(np.asarray(reg.tvalues_, dtype=float)),
            p=hexes(np.asarray(reg.pvalues_, dtype=float)),
            ci=[hexes(r) for r in ci],
            omitted=omitted, means=means, abs_means=abs_means, rows=int(len(sample)),
            weighted=w is not None,
            nobs=int(reg.nobs_), df_resid=float(reg.df_resid_).hex(), df_a=float(reg.df_a_).hex(),
            df_a_nested=float(reg.df_a_nested_).hex(), num_clusters=int(reg.num_clusters_),
            cluster_scale=float(reg.cluster_scale_).hex(),
            converged=bool(reg.converged_), precision_certified=bool(reg.precision_certified_),
            iterations=int(reg.num_iterations_), abs_residual_rel=float(reg.abs_residual_rel_),
            method=str(reg.absorption_method_used), gpu_used=bool(reg.gpu_used_),
            cuda_tail=str(getattr(reg, 'cuda_tail_', 'n/a')), threads_used=int(reg.threads_used_),
            data_sha256=data.get('sha256'),
            seconds_informative_only=round(time.time() - started, 3))
        out.append(result)
    Path(result_path).write_text(json.dumps(out, indent=1) + '\n')


def run_worker(module, backend, rows, data_dir, scratch, label):
    spec = scratch / f'{label}.spec.json'
    result = scratch / f'{label}.result.json'
    spec.write_text(json.dumps(dict(module=str(module), backend=backend, rows=rows,
                                    data_dir=str(data_dir))))
    env = {k: v for k, v in os.environ.items() if k != 'LD_LIBRARY_PATH'}
    proc = subprocess.run([sys.executable, '-B', __file__, '--worker', str(spec), str(result)],
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'worker {label} failed rc={proc.returncode}:\n{proc.stderr[-4000:]}')
    return {r['name']: r for r in json.loads(result.read_text())}


# ----------------------------------------------------------------------------
# Grading.


def floats(values):
    return [float.fromhex(v) for v in values]


def identity_check(fit, exact_zero=False):
    """Exact identity deviations and their predeclared arithmetic allowances."""
    cov = [floats(r) for r in fit['cov']]
    p = len(cov) - 1
    kept = [j for j in range(p) if not fit['omitted'][j]]
    k = len(kept)
    xbar = [parse_frac(fit['means'][j]) for j in kept]
    absx = [abs(float(v)) for v in xbar]
    # Model of the C++ mean (fixed_chunk_weighted_mean): long double products and
    # sums, one long double division, one rounding to binary64; weights possibly
    # rescaled once before the fit (2u). Any summation order: gamma_{n+2}.
    n = fit['rows']
    g_ld = gamma(n + 2, U_LD) * 1.01
    delta = []
    for pos, j in enumerate(kept):
        d = U * absx[pos] + g_ld * (absx[pos] + fit['abs_means'][j])
        if fit['weighted']:
            d += 2.02 * U * (absx[pos] + fit['abs_means'][j])
        delta.append(d)
    V = [[Fraction(cov[kept[a]][kept[b]]) for b in range(k)] for a in range(k)]
    absV = [[abs(cov[kept[a]][kept[b]]) for b in range(k)] for a in range(k)]
    vx = [sum(V[a][b] * xbar[b] for b in range(k)) for a in range(k)]
    q = sum(xbar[a] * vx[a] for a in range(k))
    vcc = cov[p][p]
    dev_cc = abs(Fraction(vcc) - q)
    xd = [absx[a] + delta[a] for a in range(k)]
    quad = lambda left, right: sum(left[a] * absV[a][b] * right[b] for a in range(k) for b in range(k))
    allow_cc = (gamma(k + 4) * quad(xd, xd) + quad(delta, absx) + quad(absx, delta)
                + quad(delta, delta)) * (1 + 1e-9)
    dev_cb, allow_cb = [], []
    for a in range(k):
        for value in (cov[p][kept[a]], cov[kept[a]][p]):
            dev_cb.append(float(abs(Fraction(value) + vx[a])))
            allow_cb.append((gamma(k + 2) * sum(absV[a][b] * xd[b] for b in range(k))
                             + sum(absV[a][b] * delta[b] for b in range(k))) * (1 + 1e-9))
    cons_row = [cov[p][j] for j in range(p)] + [cov[j][p] for j in range(p)]
    zeros = vcc == 0.0 and all(value == 0.0 for value in cons_row)
    if k == 0 or exact_zero:
        # No identified regressor or xbar = 0 exactly: the identity is an exact zero.
        ok = zeros and q == 0
    else:
        ok = float(dev_cc) <= allow_cc and all(d <= m for d, m in zip(dev_cb, allow_cb))
    scale = abs(vcc) if vcc else float('nan')
    return dict(k=k, holds=bool(ok), exact_zero_row=bool(k == 0 or exact_zero), zeros=bool(zeros),
                V_cons=vcc, xbar_V_xbar=float(q),
                dev_cc=float(dev_cc), allow_cc=allow_cc,
                dev_cc_rel=float(dev_cc) / scale if vcc else (0. if dev_cc == 0 else float('inf')),
                allow_cc_rel=allow_cc / scale if vcc else 0.,
                dev_cb_max=max(dev_cb) if dev_cb else 0., allow_cb_min=min(allow_cb) if allow_cb else 0.,
                cb_ratio_max=max((d / m if m else (0. if d == 0 else float('inf')))
                                 for d, m in zip(dev_cb, allow_cb)) if dev_cb else 0.)


def same_bits(a, b, keys):
    return all(a.get(key) == b.get(key) for key in keys)


def slope_block_bits(a, b):
    p = len(a['cov']) - 1
    return (a['coef'] == b['coef'] and a['omitted'] == b['omitted'] and
            all(a['cov'][i][:p] == b['cov'][i][:p] for i in range(p)))


def scaled_difference(a, b, block='all'):
    """max |dV_jk|/sqrt(V_jj V_kk) over 'all', the 'slopes' block or the '_cons' row."""
    ca = [floats(r) for r in a['cov']]
    cb = [floats(r) for r in b['cov']]
    p = len(ca) - 1
    worst = 0.
    for i in range(len(ca)):
        for j in range(len(ca)):
            if block == 'slopes' and (i == p or j == p):
                continue
            if block == 'cons' and i != p:
                continue
            s = math.sqrt(abs(cb[i][i] * cb[j][j]))
            if s > 0:
                worst = max(worst, abs(ca[i][j] - cb[i][j]) / s)
            elif ca[i][j] != cb[i][j]:
                worst = float('inf')
    return worst


def coefficient_difference(a, b):
    """max |db_j|/max(1,|b_j|) over every coefficient (Comparable contract metric)."""
    ca, cb = floats(a['coef']), floats(b['coef'])
    return max((abs(x - y) / max(1., abs(y)) for x, y in zip(ca, cb)), default=0.)


def inference_missing(fit, index, standard_error=True):
    """t, p and CI missing; with standard_error, the standard error as well."""
    se, t, p = floats(fit['se']), floats(fit['t']), floats(fit['p'])
    ci = [floats(r) for r in fit['ci']]
    return ((not standard_error or math.isnan(se[index])) and math.isnan(t[index])
            and math.isnan(p[index]) and all(math.isnan(v) for v in ci[index]))


FIT_KEYS = ['coef', 'cov', 'se', 't', 'p', 'ci', 'omitted', 'nobs', 'df_resid', 'df_a',
            'df_a_nested', 'num_clusters', 'cluster_scale', 'iterations', 'method', 'converged',
            'precision_certified']
DOF_KEYS = ['nobs', 'df_resid', 'df_a', 'df_a_nested', 'num_clusters', 'cluster_scale']


def grade(row, cand, ref, backend):
    out = dict(name=row['name'], kind=row['kind'], data=row['data'], cluster=row['cluster'],
               method_requested=row['method'], tol=row['tol'], threads=row['threads'])
    if 'error' in cand or (ref is not None and 'error' in ref):
        # A refusal is never a PASS: identical refusals in both builds are reported
        # apart as UNSUPPORTED, any other combination as FAIL.
        same = ('error' in cand and ref is not None and ref.get('error') == cand.get('error'))
        out.update(status='UNSUPPORTED' if same else 'FAIL', error=cand.get('error'),
                   reference_error=ref.get('error') if ref else None)
        return out
    out.update({k: cand[k] for k in ['method', 'iterations', 'converged', 'precision_certified',
                                     'abs_residual_rel', 'gpu_used', 'cuda_tail', 'threads_used',
                                     'nobs', 'num_clusters', 'data_sha256']})
    check = identity_check(cand, row.get('exact_zero', False))
    out['identity'] = check
    deterministic = backend == 'cpu' and row['threads'] == 1
    gpu_ok = backend == 'cpu' or cand['gpu_used']
    certified = cand['converged'] and cand['precision_certified']
    p = len(cand['coef']) - 1
    if ref:
        out['reference_identity'] = identity_check(ref, row.get('exact_zero', False))
        out['reference_converged_certified'] = bool(ref['converged'] and ref['precision_certified'])
        out['scaled_V_difference_vs_reference'] = scaled_difference(cand, ref)
        out['scaled_slope_block_difference_vs_reference'] = scaled_difference(cand, ref, 'slopes')
        out['scaled_cons_row_change_vs_reference'] = scaled_difference(cand, ref, 'cons')
        out['coefficient_difference_vs_reference'] = coefficient_difference(cand, ref)
        out['V_cons_relative_change_vs_reference'] = (
            (check['V_cons'] - out['reference_identity']['V_cons']) / out['reference_identity']['V_cons']
            if out['reference_identity']['V_cons'] else None)
        out['slope_block_bit_identical'] = slope_block_bits(cand, ref)
        out['dof_identical'] = same_bits(cand, ref, DOF_KEYS)
        out['fit_bit_identical'] = same_bits(cand, ref, FIT_KEYS)
        out['contract_b_V_slopes'] = (out['coefficient_difference_vs_reference'] <= B_CONTRACT and
                                      out['scaled_slope_block_difference_vs_reference'] <= V_CONTRACT)
    if row.get('exact_zero'):
        se, t = floats(cand['se']), floats(cand['t'])
        out['exact_zero_output'] = dict(V_cons=check['V_cons'], zeros=check['zeros'], se_cons=se[p],
                                        t_cons=t[p], t_p_ci_missing=inference_missing(cand, p, False))
    if row['kind'] == 'identity':
        ok = certified and check['holds'] and gpu_ok and bool(ref)
        if row.get('exact_zero'):
            ok = ok and floats(cand['se'])[p] == 0.0 and inference_missing(cand, p, False)
        if ref:
            ok = ok and out['dof_identical'] and (out['slope_block_bit_identical'] if deterministic
                                                  else out['contract_b_V_slopes'])
    elif row['kind'] == 'one_cluster':
        # Fewer than two clusters: every identified coefficient keeps missing
        # inference and the diagonal sentinel, exactly as the reference build.
        cov = [floats(r) for r in cand['cov']]
        identified = [j for j in range(p + 1) if j == p or not cand['omitted'][j]]
        convention = (cand['num_clusters'] == 1 and
                      all(inference_missing(cand, j) and cov[j][j] == SENTINEL for j in identified))
        out['missing_inference_convention'] = convention
        ok = certified and convention and bool(ref) and out['dof_identical']
        if ref:
            rcov = [floats(r) for r in ref['cov']]
            out['reference_convention'] = all(inference_missing(ref, j) and rcov[j][j] == SENTINEL
                                              for j in identified)
            ok = ok and out['reference_convention'] and out['slope_block_bit_identical']
            if check['k'] == 0:
                ok = ok and out['fit_bit_identical']
    else:  # control: the generic computation is kept
        ok = certified and gpu_ok and bool(ref)
        if ref:
            if deterministic:
                ok = ok and out['fit_bit_identical']
            else:
                ok = (ok and out['dof_identical'] and out['coefficient_difference_vs_reference'] <= B_CONTRACT
                      and out['scaled_V_difference_vs_reference'] <= V_CONTRACT)
        if row.get('identity_false') and not deterministic:
            # The identity does not hold here; the candidate must not impose it.
            ok = ok and not check['holds'] and check['dev_cc'] > 100 * check['allow_cc']
    out['gpu_required_and_used'] = gpu_ok
    out['status'] = 'PASS' if ok else 'FAIL'
    return out


def grader_self_test():
    """Mutations of the constant's variance or covariance, and refusals, must be rejected."""
    xbar = [Fraction(3, 7), Fraction(-5, 11)]
    V = [[2.5e-3, -1.25e-4], [-1.25e-4, 4.0e-3]]
    q = float(sum(xbar[a] * Fraction(V[a][b]) * xbar[b] for a in range(2) for b in range(2)))
    vx = [float(sum(Fraction(V[a][b]) * xbar[b] for b in range(2))) for a in range(2)]
    def fit(vcc, vc0):
        cov = [[V[0][0], V[0][1], vc0], [V[1][0], V[1][1], -vx[1]], [vc0, -vx[1], vcc]]
        return dict(cov=[hexes(r) for r in cov], omitted=[False, False],
                    means=[frac_str(v) for v in xbar], abs_means=[1., 1.], rows=1000,
                    weighted=False)
    exact = identity_check(fit(q, -vx[0]))
    bad_cc = identity_check(fit(q * (1 + 1e-12), -vx[0]))
    bad_cb = identity_check(fit(q, -vx[0] * (1 + 1e-12)))
    zero = dict(cov=[hexes([0.])], omitted=[], means=[], abs_means=[], rows=10, weighted=False)
    zero_ok = identity_check(zero)
    zero_bad = identity_check(dict(zero, cov=[hexes([1e-300])]))
    refusal = grade(dict(name='self', kind='control', data='panel', cluster='worker', method='auto',
                         tol=1e-8, threads=1), dict(error='RuntimeError: x'),
                    dict(error='RuntimeError: x'), 'cpu')
    ok = (exact['holds'] and not bad_cc['holds'] and not bad_cb['holds'] and zero_ok['holds']
          and not zero_bad['holds'] and refusal['status'] == 'UNSUPPORTED')
    return dict(exact_accepted=exact['holds'], mutated_variance_rejected=not bad_cc['holds'],
                mutated_covariance_rejected=not bad_cb['holds'],
                exact_zero_accepted=zero_ok['holds'], nonzero_rejected=not zero_bad['holds'],
                identical_refusal_status=refusal['status'], status='PASS' if ok else 'FAIL')


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3])
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--reference-module', type=Path, required=True,
                        help='previous build: graded bit for bit (CPU, one thread) or by contract')
    parser.add_argument('--backend', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA,
                        help='public synthetic-assortative/patents parquet files')
    parser.add_argument('--skip-public', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists(), f'{args.output} exists'
    module = args.module.resolve(strict=True)
    reference = args.reference_module.resolve(strict=True)
    rows = synthetic_rows(args.backend)
    public_available = not args.skip_public and all(
        (args.data_dir / f'{d}.parquet').exists() for d in ['synthetic-assortative', 'patents'])
    if public_available:
        rows += public_rows(args.backend)
    scratch = args.output.parent / (args.output.stem + '_work')
    scratch.mkdir(parents=True, exist_ok=False)
    started = time.time()
    cand = run_worker(module, args.backend, rows, args.data_dir, scratch, 'candidate')
    ref = run_worker(reference, args.backend, rows, args.data_dir, scratch, 'reference')
    graded = [grade(row, cand[row['name']], ref.get(row['name']), args.backend) for row in rows]
    for g in graded:
        print(json.dumps({k: g.get(k) for k in ['name', 'status', 'method', 'iterations', 'gpu_used']}
                         | {'dev_rel': g.get('identity', {}).get('dev_cc_rel'),
                            'allow_rel': g.get('identity', {}).get('allow_cc_rel'),
                            'ref_dev_rel': g.get('reference_identity', {}).get('dev_cc_rel'),
                            'db': g.get('coefficient_difference_vs_reference'),
                            'dVb': g.get('scaled_slope_block_difference_vs_reference')}),
              flush=True)
    positive = [g for g, row in zip(graded, rows) if row.get('positive_control')]
    positive_live = bool(positive) and all(
        'reference_identity' in g and not g['reference_identity']['holds'] for g in positive)
    self_test = grader_self_test()
    statuses = {s: sum(g['status'] == s for g in graded) for s in ('PASS', 'FAIL', 'UNSUPPORTED')}
    status = 'PASS' if (statuses['PASS'] == len(graded) and self_test['status'] == 'PASS'
                        and (positive_live or not public_available)) else 'FAIL'
    report = dict(
        status=status, backend=args.backend,
        module=str(module), module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
        reference_module=str(reference),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        public_data=dict(available=public_available, dir=str(args.data_dir)),
        positive_control=dict(required=public_available, live=positive_live,
                              rows=[g['name'] for g in positive]),
        grader_self_test=self_test,
        contracts=dict(coefficients=B_CONTRACT, covariance=V_CONTRACT, sentinel=SENTINEL),
        arithmetic_model=dict(u=U, u_long_double=U_LD, machine=platform.machine(),
                              identity='gamma_{k+4} (xbar\'V xbar) and gamma_{k+2} (V xbar) plus '
                                       'mean error: u|xbar| + 1.01 gamma^ld_{n+2}(|xbar| + mean|w x|) '
                                       '[+ 2.02u(|xbar| + mean|w x|) when weighted]; exact zeros '
                                       'when k = 0 or xbar = 0'),
        counts=dict(rows=len(graded), **{s.lower(): c for s, c in statuses.items()},
                    identity_rows=sum(r['kind'] == 'identity' for r in rows),
                    control_rows=sum(r['kind'] == 'control' for r in rows),
                    one_cluster_rows=sum(r['kind'] == 'one_cluster' for r in rows)),
        wall_seconds_informative_only=round(time.time() - started, 1),
        rows=graded)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=status, counts=report['counts'],
                          positive_control=report['positive_control'],
                          grader_self_test=self_test['status'])))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
