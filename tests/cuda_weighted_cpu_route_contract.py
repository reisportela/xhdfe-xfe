#!/usr/bin/env python3
"""Package C (18nov2026): weighted fits requested on CUDA run on the CPU and say so (offline; CUDA machine).

A fit with a weights vector (all ones and zero weights included) and two or more absorbed effect dimensions as
received, in reghdfe-comparable or strict-residual mode (a group() fit without individual() is the collapsed
weighted fit and counts; with individual() it keeps the device route), whose effective backend (thread request,
XHDFE_GPU_BACKEND, else the build default) is CUDA, runs as the CPU call and publishes GPU
status code 7 (weighted_cpu_route) when the fit requests it; every other GPU field is the CPU call's.
By default the fit uses the device route. Design: Verifications/corrigir_erros_20261117/c/DESENHO_C.md and
c/RESPOSTA_DESENHO_C.md. Synthetic data from fixed seeds; one fresh process per arm unless the case is a session;
every arm with an explicit thread count (--threads, default 8) and every XHDFE_ variable purged first.

Cases (--only):
* identity: a routed call (XHDFE_GPU_BACKEND=cuda) against the same call with XHDFE_GPU_BACKEND=cpu, same module,
  threads and initial state: b (hex), the full covariance, SE, residuals, RSS, iterations, saved effects, method,
  tier status and reason, the retry diagnostics, df, sample map, threads, and the GPU fields other than the code
  (as produced: gpu_used and gpu_attempted false, zero CUDA iterations, no tail); the code 7 against 0. fw/aw/pw/iw
  x 2/3/4 effects x comparable/strict, saved effects, instruments, slopes (strict + slopes: the same error before
  estimation), zero weights, the weighted path-like two-effect graph the device refused, saved effects with
  instruments and with slopes, weighted group() fits without individual() with two and three effects in both
  modes, and the strict rows with the CPU's method recorded: the absolute check is claimed only for a sweep method
  (an explicit gauss-seidel strict cell), never for the weighted 2x2 direct projection (a control cell).
* retry: the weighted CPU retry fires in the CPU arm (eligible, fired, status 2, MLSMR) and the routed arm repeats
  it exactly: the discriminating identity, which also catches a 7 published too early.
* boundaries: one effect, no weights, all ones, fes=[a, a], fast, group() without individual() (routed) and with
  two distinct individuals per group (the device: converged, GPU used, code 1).
* errors: the CPU-only module keeps today's unavailable error; explicit lsmr/mlsmr keep theirs; an explicit schwarz
  with weights returns what the CPU returns (the ordinary refusal; with a slope in comparable the mixed absorber's
  answer), and with the switch at 0 the device's answer is recorded.
* switch: an omitted request and False use the device; True requests CPU. The default device fit is checked
  against the reference CUDA module with A/A beside A/B.
* sessions: one object alternating routed and device fits in both orders; refusals raised inside the CPU path (the
  ordinary schwarz refusal, a max_iter refusal) leave no estimates and the next device fit really on the GPU.
* caches: with the absorption cache on, a routed call writes the CPU's entry and hits it (code 7 on the hit), and a
  CPU call hits the routed call's entry; with profile writing on, the routed and the CPU call write byte-identical
  profiles, and device fits that read each agree (A/A beside A/B).
* internal: a weighted Gelbach decomposition with gpu=True on the fast path reports 7 and weighted_cpu_route, its
  bootstrap with require_gpu_used=True refuses; through the environment with zero, one and two common effects the
  outcomes are required (zero: today's refusal; one and two: 7); an AKM fit reports what the reference module
  reports, both processes succeeding.
* contract: cov240k with the weights and fourth effect of levantamento/p1_domain.py:54-76, clusters on the first
  effect: the routed fits against an independent weighted oracle written here (anchored sparse dummy LU with four
  long-double refinements, MP80 slopes, weighted cluster sandwich; the nested-in-cluster first effect not counted
  in the degrees of freedom, asserted against df_a_): b 1e-8 max(1,|b|), SE 1e-7 relative. The two CPU gaps (four
  effects with saved effects; an explicit gauss-seidel) are cells expected outside, each with a stated bound; the
  switch-0 device cells are recorded, not graded. The weighted group() fits (exact duplicate rows per group) are
  graded on their coefficients against the oracle of the ungrouped data.
* nochange: fits the predicate does not route (unweighted, one effect, xhdfe-fast, an unweighted group(), and every
  fit with the CPU backend) are bit for bit the reference modules' on the CPU backend (the CPU-only and the CUDA builds), and the
  unrouted device fits agree with the reference CUDA module (A/A beside A/B, AB_PROCESSES per side).
* default (--default-module, CUDA on and XHDFE_GPU_BACKEND_DEFAULT=cuda): with a device an unrequested weighted
  two-effect fit is routed (7); without a visible device (CUDA_VISIBLE_DEVICES empty) today's behaviour of fit
  stays (the CPU fallback, status 2); the unweighted fits are recorded (the reduced tier, which reads the request
  and not the build default, serves them on the CPU; attributed by reading, with no reference CUDA-default build).
Exit code 0 PASS, 1 FAIL, 3 only labels of missing coverage.
usage: cuda_weighted_cpu_route_contract.py --module CUDA.so [--cpu-module CPU.so] [--reference-module REF_CUDA.so]
       [--reference-cpu-module REF_CPU.so] [--default-module CUDA_DEFAULT.so] [--only case,...] [--threads 8]
       [--output result.json] [--keep DIR]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AP = HERE / 'analytic_precision'
sys.path.insert(0, str(AP))
from cuda_fast_restart_contract import cuda_device_visible, gpu_apps  # noqa: E402

TOL = 1e-8
LIMIT_B, LIMIT_SE = 1e-8, 1e-7
ROUTE = 7
# Fresh processes per side of every A/A beside A/B comparison of device fits (never bit-reproducible). With two per
# side the rule "A/B median <= A/A max" fails 6 times in 15 when nothing changed (2 A/A against 4 A/B distances,
# exchangeable); with five (20 against 25) almost never. Five, as package B's switch-0 arm.
AB_PROCESSES = 5
CASES = ('identity', 'retry', 'boundaries', 'errors', 'switch', 'sessions', 'caches', 'internal', 'contract',
         'nochange', 'default')
# Stated bounds of the two cells expected outside the contract on the CPU (survey: 1.6e-7/1.3e-5 and 3.2e-7/2.0e-5).
CPU_GAP_BOUND = dict(b_scaled=1e-5, se_rel=1e-3)
IDENTITY_KEYS = ('refused', 'error', 'b_hex', 'V_sha', 'se_hex', 'e_sha', 'rss_hex', 'iterations', 'converged',
                 'certified', 'fe_sha', 'fe_recovery', 'method', 'tier_status', 'tier_reason', 'retry', 'df_a',
                 'df_resid', 'nobs', 'sample_sha', 'threads_effective', 'threads_used', 'omitted', 'gpu_used',
                 'gpu_attempted', 'gpu_absorption_converged', 'gpu_absorption_iterations', 'cuda_tail')


# ---------------------------------------------------------------- fixtures (synthetic, fixed seeds)
def cov240k(nfe):
    """The public 240,000-row fixture of cuda_three_fe_covariance_contract.py (worker, firm, year) and the fourth
    effect of levantamento/p1_domain.py:54-76 (25 levels, not nested), first nfe effects."""
    from cuda_three_fe_covariance_contract import fixture as make
    y, x, fes = make()
    worker, year = fes[0], fes[2]
    workers = int(worker.max()) + 1
    rng = np.random.default_rng(20261117)
    base = rng.integers(0, 25, workers)[worker]
    switch = rng.integers(5, 35, workers)[worker]
    fes = list(fes) + [((base + (year >= switch)) % 25).astype(np.int32)]
    return y, np.asfortranarray(x), [np.asarray(f, np.int32) for f in fes[:nfe]]


def pathlike():
    from cuda_strict_restart_contract import pathlike_fixture
    y, x, fes = pathlike_fixture()
    return y, np.asfortranarray(x), [np.asarray(f, np.int32) for f in fes]


def two_by_two():
    """Two effects with two levels each: the design of the weighted 2x2 direct projection (weighted_two_by_two.hpp),
    which returns before the sweep (src/fe_absorption.cpp, try_cpu)."""
    rng = np.random.default_rng(20261126)
    n = 4000
    a, b = rng.integers(0, 2, n).astype(np.int32), rng.integers(0, 2, n).astype(np.int32)
    x = rng.normal(size=(n, 2))
    y = x @ np.array([.5, -.25]) + .3 * a - .2 * b + rng.normal(size=n)
    return y, np.asfortranarray(x), [a, b]


def data(name, nfe):
    if name == 'cov240k':
        return cov240k(nfe)
    if name == 'pathlike':
        return pathlike()
    if name == 'two_by_two':
        return two_by_two()
    raise ValueError(name)


def weight_vector(kind, n):
    """p1_domain.py:65-70: fw integers 1..4 (seed 20261110); aw, pw, iw U[0.5, 2) (seed 20261111)."""
    if kind == 'none':
        return None
    if kind == 'fw':
        return np.random.default_rng(20261110).integers(1, 5, n).astype(float)
    if kind == 'ones':
        return np.ones(n)
    w = np.random.default_rng(20261111).uniform(.5, 2., n)
    if kind == 'aw_zeros':
        w[np.random.default_rng(20261118).random(n) < .1] = 0.0
    elif kind == 'negative':
        w[7] = -1.0
    return w


# ---------------------------------------------------------------- the independent weighted oracle
def weighted_oracle(y, x, fes, w, frequency):
    """b (slopes and constant) and the CR1 cluster covariance (clusters: the first effect, nested in them) of the
    weighted OLS on the anchored dummy system: D'WD solved by sparse LU with four long-double refinements, MP80
    slopes, the constant's weighted transport. Degrees of freedom: N (rows, or the frequency total), the
    regressors, the constant and the levels of every effect after the first minus one each."""
    import mpmath as mp
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    n, p = x.shape
    rows = np.arange(n)
    levels = [int(v.max()) + 1 for v in fes]
    row_idx, col_idx, offset = [rows], [np.asarray(fes[0])], levels[0]
    for fe, size in zip(fes[1:], levels[1:]):
        keep = fe != 0
        row_idx.append(rows[keep])
        col_idx.append(offset + fe[keep] - 1)
        offset += size - 1
    r, c = np.concatenate(row_idx), np.concatenate(col_idx)
    d = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, offset)).tocsr()
    dw = sp.diags(w) @ d
    s = (d.T @ dw).tocsc()
    lu = splu(s, permc_spec='MMD_AT_PLUS_A')
    dl, sl = d.astype(np.longdouble), s.astype(np.longdouble)
    wl = np.asarray(w, np.longdouble)
    a = np.column_stack([y, x]).astype(np.longdouble)
    rhs = dl.T @ (a * wl[:, None])
    alpha = lu.solve(np.asarray(rhs, float)).astype(np.longdouble)
    for _ in range(4):
        alpha += lu.solve(np.asarray(rhs - sl @ alpha, float)).astype(np.longdouble)
    within = a - dl @ alpha
    defects = dl.T @ (within * wl[:, None])
    counts = np.asarray(dl.T @ wl).ravel()
    relative = np.sqrt(np.sum(defects ** 2 / counts[:, None], axis=0) / np.sum(wl[:, None] * a * a, axis=0))
    qx, qy = within[:, 1:], within[:, 0]
    gram, h = qx.T @ (qx * wl[:, None]), qx.T @ (qy * wl)
    total_w = wl.sum()
    with mp.workdps(80):
        g = mp.matrix([[mp.mpf(str(v)) for v in row] for row in gram])
        inverse = g ** -1
        beta = inverse * mp.matrix([mp.mpf(str(v)) for v in h])
        b = np.array([np.longdouble(str(v)) for v in beta])
        residual = qy - qx @ b
        scores = np.zeros((levels[0], p + 1), np.longdouble)
        np.add.at(scores, fes[0], np.column_stack([qx * (wl * residual)[:, None], wl * residual]))
        meat = scores.T @ scores
        m = mp.matrix([[mp.mpf(str(v)) for v in row] for row in meat])
        means = (a * wl[:, None]).sum(axis=0) / total_w
        transport = mp.zeros(p + 1)
        for j in range(p):
            for k in range(p):
                transport[j, k] = inverse[j, k]
                transport[p, k] -= mp.mpf(str(means[j + 1])) * inverse[j, k]
        transport[p, p] = mp.mpf(1) / mp.mpf(str(total_w))
        df_a = sum(levels[1:]) - (len(levels) - 1)
        big_n = mp.mpf(str(total_w)) if frequency else mp.mpf(n)
        clusters = levels[0]
        adjustment = mp.mpf(clusters) / (clusters - 1) * (big_n - 1) / (big_n - p - df_a - 1)
        v = adjustment * transport * m * transport.T
        constant = mp.mpf(str(means[0])) - sum(mp.mpf(str(means[j + 1])) * beta[j] for j in range(p))
        full_b = np.array([float(vv) for vv in beta] + [float(constant)])
        full_v = np.array([[float(v[j, k]) for k in range(p + 1)] for j in range(p + 1)])
    return dict(b=full_b, V=full_v, df_a=df_a, defect=float(np.max(relative)))


def well_formed(b, V, k=None):
    """The reason a fit cannot be graded, or None: full dimensions (k coefficients, the constant included, and a k x k
    covariance), finite values and positive variances."""
    b, V = np.asarray(b, float), np.asarray(V, float)
    k = len(b) if k is None else k
    if b.shape != (k,) or V.shape != (k, k):
        return f'dimensions b {b.shape} V {V.shape}, expected {k}'
    if not (np.isfinite(b).all() and np.isfinite(V).all()):
        return 'non-finite estimates'
    if not (np.diag(V) > 0).all():
        return 'a variance is not positive'
    return None


def grade(oracle, fit):
    """Distances from the oracle; infinite (never inside a limit) when the fit or the oracle is not well formed."""
    ob, oV = np.asarray(oracle['b'], float), np.asarray(oracle['V'], float)
    bad = well_formed(ob, oV) or well_formed(fit.get('b', []), fit.get('V', []), len(ob))
    if bad:
        return dict(b_scaled=float('inf'), se_rel=float('inf'), ungraded=bad)
    b, V = np.asarray(fit['b'], float), np.asarray(fit['V'], float)
    se, ose = np.sqrt(np.diag(V)), np.sqrt(np.diag(oV))
    return dict(b_scaled=float(np.max(np.abs(b - ob) / np.maximum(1.0, np.abs(ob)))),
                se_rel=float(np.max(np.abs(se - ose) / ose)))


def inside(value, limit):
    """value <= limit, false for NaN."""
    return bool(np.isfinite(value) and value <= limit)


# ---------------------------------------------------------------- the worker (one process)
def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def fit_kwargs(f, spec, n):
    y, x, fes = f
    kw = dict(fes=list(fes))
    if spec.get('dup_fe'):
        kw['fes'] = [fes[0], fes[0]]
    w = weight_vector(spec.get('weights', 'none'), n)
    if w is not None:
        kw['weights'] = w
        kind = spec['weights']
        kw.update(fweights=kind == 'fw', iweights=kind == 'iw', pweights=kind == 'pw')
    if spec.get('se', 'cluster') == 'cluster':
        kw['clusters'] = np.asarray(fes[0]).reshape(-1, 1)
    if spec.get('iv'):
        z = x[:, 1] + 0.5 * np.random.default_rng(20261121).normal(size=n)
        kw.update(instruments=z.reshape(-1, 1), endogenous_idx=[1])
    if spec.get('slopes'):
        s = np.random.default_rng(20261122).normal(size=n)
        kw['slopes'] = [(1 if len(kw['fes']) > 1 else 0, np.ascontiguousarray(s), True)]
    if spec.get('group'):
        idx = np.repeat(np.arange(n), 2)
        kw['fes'] = [fe[idx] for fe in kw['fes']]
        for key in ('weights', 'clusters'):
            if key in kw:
                kw[key] = kw[key][idx]
        kw['group'] = idx
        if spec['group'] == 'individual':
            # Two distinct individuals in each group ((group, individual) unique), drawn from 5000 as in the Stata
            # leg's groups_ind fixture; the individual replaces the first absorbed effect.
            r = np.random.default_rng(20261124)
            first = r.integers(0, 5000, n)
            second = (first + 1 + r.integers(0, 4999, n)) % 5000
            kw['fes'][0] = np.column_stack([first, second]).reshape(-1).astype(np.int32)
            kw['individual'] = kw['fes'][0]
        return y[idx], np.asfortranarray(x[idx]), kw
    return y, x, kw


def record(reg, spec, error=None):
    rec = dict(name=spec['name'])
    if error is not None:
        rec.update(refused=True, error=str(error)[:1500],
                   estimates_left=int(np.asarray(reg.coef_).size) if reg is not None else None)
    else:
        b = np.asarray(reg.coef_, np.float64).ravel()
        V = np.asarray(reg.covariance_, np.float64)
        se = np.asarray(reg.stderr_, np.float64).ravel()
        e = np.asarray(reg.residuals_, np.float64).ravel()
        fe = reg.fe_effects_
        fe_sha = hashlib.sha256(b''.join(np.ascontiguousarray(np.asarray(v, np.float64)).tobytes() for v in fe)).hexdigest() \
            if fe is not None and len(fe) else None
        rec.update(refused=False, error=None, b=b.tolist(), V=V.tolist(), b_hex=[v.hex() for v in b],
                   V_sha=sha(V), se_hex=[float(v).hex() for v in se], e_sha=sha(e), rss_hex=float(reg.rss_).hex(),
                   iterations=int(reg.num_iterations_), converged=bool(reg.converged_),
                   certified=bool(reg.precision_certified_), fe_sha=fe_sha,
                   fe_recovery=[bool(reg.fe_recovery_converged_), int(reg.fe_recovery_iterations_)],
                   method=str(reg.absorption_method_used).replace('AbsorptionMethod.', ''),
                   tier_status=int(reg.reduced_pcg_status_), tier_reason=str(reg.reduced_pcg_reason_),
                   retry=[bool(reg.auto_routing_retry_eligible_), bool(reg.auto_routing_retry_fired_),
                          int(reg.auto_routing_retry_status_), int(reg.auto_routing_retry_primary_method_),
                          int(reg.auto_routing_retry_primary_iterations_), int(reg.auto_routing_retry_iterations_)],
                   df_a=float(reg.df_a_), df_resid=float(reg.df_resid_), nobs=float(reg.nobs_effective_),
                   sample_sha=sha(np.asarray(reg.sample_index_, np.int64)),
                   threads_effective=int(reg.threads_effective_), threads_used=int(reg.threads_used_),
                   omitted=[bool(v) for v in reg.omitted_])
    if reg is not None:
        rec.update(gpu_used=bool(reg.gpu_used_), gpu_status_code=int(reg.gpu_status_code_),
                   gpu_attempted=bool(reg.gpu_attempted_), gpu_absorption_converged=bool(reg.gpu_absorption_converged_),
                   gpu_absorption_iterations=int(reg.gpu_absorption_iterations_), cuda_tail=int(reg.cuda_tail_code_))
    return rec


def worker(job_path):
    import importlib.util
    job = json.loads(Path(job_path).read_text())
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', job['module'])
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.modules['py_hdfe_v11'] = core
    cache = {}
    fits, reg = [], None
    for k, s in enumerate(job['fits']):
        sys.stderr.write(f"CWR_FIT {k} {s['name']}\n")
        sys.stderr.flush()
        if s.get('kind') in ('gelbach', 'gelbach_bootstrap', 'akm'):
            fits.append(internal_call(core, s))
            continue
        key = (s.get('data', 'cov240k'), s.get('nfe', 3))
        if key not in cache:
            cache[key] = data(*key)
        f = cache[key]
        y, x, kw = fit_kwargs(f, s, len(f[0]))
        if reg is None or not job.get('same_object'):
            constructor = {'weighted_cpu_route': s['constructor_route']} if 'constructor_route' in s else {}
            reg = core.HdfeRegressor(se_type=s.get('se', 'cluster'), tol=float.fromhex(s.get('tol_hex', TOL.hex())),
                                     max_iter=s.get('max_iter', 100000), num_threads=job['threads'],
                                     drop_singletons=True, retain_fes=bool(s.get('retain_fes')),
                                     absorption_method=s.get('method', 'auto'),
                                     tolerance_mode=s.get('mode', 'reghdfe-comparable'),
                                     convergence=s.get('convergence', 'auto'), **constructor)
        route = s.get('weighted_cpu_route')
        if route is not None:
            kw['weighted_cpu_route'] = route
        t0 = time.perf_counter()
        try:
            reg.fit(y, x, **kw)
        except Exception as error:  # noqa: BLE001 - refusals are graded
            rec = record(reg, s, error)
        else:
            rec = record(reg, s)
        rec['seconds'] = time.perf_counter() - t0
        rec['spec'] = s
        fits.append(rec)
    Path(job['out']).write_text(json.dumps(dict(fits=fits)))


def internal_call(core, s):
    """Gelbach and AKM through the binding and the package wrapper (xhdfe.gelbach)."""
    rec = dict(name=s['name'], spec=s)
    rng = np.random.default_rng(20261123)
    n = 4000
    worker_id = rng.integers(0, 400, n).astype(np.int32)
    firm = rng.integers(0, 60, n).astype(np.int32)
    year = rng.integers(0, 8, n).astype(np.int32)
    x1 = rng.normal(size=(n, 1))
    x2 = 0.5 * x1[:, 0] + rng.normal(size=n)
    y = 0.7 * x1[:, 0] + 0.4 * x2 + 0.3 * rng.normal(size=400)[worker_id] + 0.2 * rng.normal(size=60)[firm] \
        + rng.normal(size=n)
    w = rng.uniform(.5, 2., n)
    try:
        if s['kind'] in ('gelbach', 'gelbach_bootstrap'):
            sys.path.insert(0, str(ROOT))
            import xhdfe.gelbach as gel
            common = {f'c{j}': v for j, v in zip(range(s.get('common', 0)), (year, firm))}
            fes = {'worker': worker_id, 'firm': firm} if s.get('common', 0) == 0 else {'worker': worker_id}
            kwargs = dict(x2_groups={'x2': x2}, fes=fes, weights=w, gpu=bool(s.get('gpu')), tol=1e-8)
            if 'weighted_cpu_route' in s:
                kwargs['weighted_cpu_route'] = s['weighted_cpu_route']
            if common:
                kwargs['common_fes'] = common
            if s['kind'] == 'gelbach':
                r = gel.decompose(y, x1, **kwargs)
            else:
                r = gel.bootstrap(y, x1, reps=3, seed=1, require_gpu_used=bool(s.get('require_gpu_used')), **kwargs)
                boot = r.get('bootstrap', {})
                rec.update(reps_valid=boot.get('reps_valid'),
                           replicate_gpu=[entry['gpu_used'] for entry in boot.get('ledger', [])])
                r = r.get('point', r) if isinstance(r, dict) else r
            r = dict(r) if isinstance(r, dict) else {}
            rec.update(refused=False, gpu_used=bool(r.get('gpu_used')), gpu_status_code=r.get('gpu_status_code'),
                       gpu_status=r.get('gpu_status'), gpu_backend=r.get('gpu_backend'))
        else:
            # The binding's AKM entry (an earlier core.akm call named no entry: both arms refused alike and the
            # row passed without a result; c/RESPOSTA_2_C.md 2.3).
            r = core.akm_kss(y, worker_id, firm, X=x1, fweights=np.ones(n), num_threads=4, gpu=True)
            rec.update(refused=False, **{k: (v if isinstance(v, (int, float, str, bool)) else None)
                                         for k, v in dict(r).items() if 'gpu' in k or 'status' in k})
    except Exception as error:  # noqa: BLE001
        rec.update(refused=True, error=str(error)[:600])
    return rec


# ---------------------------------------------------------------- the driver
def clean_env(backend='cuda', extra=None):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('XHDFE_') and k not in ('LD_LIBRARY_PATH', 'OMP_NUM_THREADS')}
    if backend is not None:
        env['XHDFE_GPU_BACKEND'] = backend
    env.update(XHDFE_ABSORPTION_CACHE_MODE='off', XHDFE_MOBILITY_MODE='off', XHDFE_FE_STRUCTURE_MODE='off',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for k, v in (extra or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


class Runner:
    def __init__(self, workdir, threads):
        self.workdir, self.threads, self.count = workdir, threads, 0

    def run(self, module, label, fits, backend='cuda', env=None, same_object=False, route=None):
        if route is not None:
            fits = [dict(weighted_cpu_route=route, **s) if 'weighted_cpu_route' not in s else s for s in fits]
        job = dict(module=str(module), fits=fits, threads=self.threads, same_object=same_object,
                   out=str(self.workdir / f'{label}.json'))
        with (self.workdir / f'{label}.job.json').open('x') as stream:
            json.dump(job, stream)
        apps = gpu_apps() if backend == 'cuda' or (env or {}).get('CUDA_VISIBLE_DEVICES') is not None else None
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                                   str(self.workdir / f'{label}.job.json')], env=clean_env(backend, env),
                                  capture_output=True, text=True, timeout=3600)
            stderr, rc = proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stderr, rc = str(exc.stderr), 'timeout'
        (self.workdir / f'{label}.err').write_text(stderr)
        out = self.workdir / f'{label}.json'
        if rc != 0 or not out.is_file():
            raise RuntimeError(f'{label}: worker did not complete (rc={rc}); see {self.workdir / (label + ".err")}')
        rec = json.loads(out.read_text())
        received = rec.get('fits', [])
        if [f['name'] for f in received] != [s['name'] for s in fits]:
            raise RuntimeError(f'{label}: missing or incorrect fit receipts')
        self.count += len(received)
        return dict(label=label, module=str(module), backend=backend, env=env or {}, rc=rc, wall=time.time() - t0,
                    fits=rec.get('fits', []), gpu_apps_before=apps, stderr_tail=stderr[-1500:] if rc != 0 else '')


def spec(name, **kw):
    s = dict(name=name)
    s.update(kw)
    return s


def identity_cells():
    cells = []
    for kind in ('fw', 'aw', 'pw', 'iw'):
        for mode in ('reghdfe-comparable', 'strict-residual'):
            cells.append(spec(f'{kind}_3fe_{mode[:6]}', weights=kind, nfe=3, mode=mode))
    for nfe in (2, 4):
        for mode in ('reghdfe-comparable', 'strict-residual'):
            cells.append(spec(f'aw_{nfe}fe_{mode[:6]}', weights='aw', nfe=nfe, mode=mode))
    for nfe in (2, 3, 4):
        cells.append(spec(f'aw_{nfe}fe_savefe', weights='aw', nfe=nfe, retain_fes=True))
    cells += [spec('aw_3fe_iv', weights='aw', iv=True), spec('aw_3fe_iv_strict', weights='aw', iv=True,
                                                             mode='strict-residual'),
              spec('aw_3fe_slopes', weights='aw', slopes=True),
              spec('aw_3fe_slopes_strict', weights='aw', slopes=True, mode='strict-residual'),
              spec('aw_3fe_iv_savefe', weights='aw', iv=True, retain_fes=True),
              spec('aw_3fe_slopes_savefe', weights='aw', slopes=True, retain_fes=True),
              spec('aw_zeros_3fe', weights='aw_zeros'), spec('aw_zeros_3fe_strict', weights='aw_zeros',
                                                               mode='strict-residual'),
              spec('fw_2fe_robust', weights='fw', nfe=2, se='robust'),
              spec('aw_pathlike', data='pathlike', nfe=2, weights='aw', se='robust'),
              spec('aw_pathlike_savefe', data='pathlike', nfe=2, weights='aw', se='robust', retain_fes=True)]
    for nfe in (2, 3):
        for mode in ('reghdfe-comparable', 'strict-residual'):
            cells.append(spec(f'group_aw_{nfe}fe_{mode[:6]}', weights='aw', nfe=nfe, mode=mode, group='group'))
    # The strict absolute check: a sweep method (explicit gauss-seidel) and the weighted 2x2 direct projection, which
    # returns before the sweep (the control: never claimed).
    cells += [spec('aw_3fe_gs_strict', weights='aw', nfe=3, method='gauss-seidel', mode='strict-residual'),
              spec('aw_2x2_strict', data='two_by_two', nfe=2, weights='aw', se='robust', mode='strict-residual')]
    return cells


def compare(routed, cpu):
    """Identity of a routed record and its CPU record: every key of IDENTITY_KEYS; then 7 against 0."""
    p = []
    if not routed or not cpu:
        return ['missing record']
    if routed.get('refused') or cpu.get('refused'):
        # A refusal publishes nothing: the throw clears the GPU fields (fail_attempt), so only the same refusal
        # and no estimates left are required.
        if routed.get('refused') != cpu.get('refused') or routed.get('error') != cpu.get('error'):
            p.append(f"refusals differ: routed {routed.get('error')} vs cpu {cpu.get('error')}")
        if routed.get('estimates_left') or cpu.get('estimates_left'):
            p.append('estimates left after a refusal')
        return p
    for k in IDENTITY_KEYS:
        if routed.get(k) != cpu.get(k):
            p.append(f'{k}: routed {str(routed.get(k))[:120]} vs cpu {str(cpu.get(k))[:120]}')
    if routed.get('gpu_status_code') != ROUTE:
        p.append(f"routed code {routed.get('gpu_status_code')} (expected 7)")
    if cpu.get('gpu_status_code') != 0:
        p.append(f"cpu code {cpu.get('gpu_status_code')} (expected 0)")
    for k, v in (('gpu_used', False), ('gpu_attempted', False), ('gpu_absorption_iterations', 0), ('cuda_tail', 0)):
        if routed.get(k) != v:
            p.append(f'routed {k} {routed.get(k)} (the CPU call produces {v})')
    return p


def scaled_distance(a, b):
    """Infinite when either fit is not well formed (dimensions, finite values, positive variances) or differs in
    dimensions from the other."""
    ba = np.asarray(a.get('b', []), float)
    if well_formed(a.get('b', []), a.get('V', [])) or well_formed(b.get('b', []), b.get('V', []), len(ba)):
        return float('inf')
    bb, Va, Vb = np.asarray(b['b'], float), np.asarray(a['V'], float), np.asarray(b['V'], float)
    scale = np.sqrt(np.outer(np.diag(Va), np.diag(Va)))
    return float(max(np.max(np.abs(ba - bb) / np.maximum(1.0, np.abs(ba))), np.max(np.abs(Va - Vb) / scale)))


def aa_ab(new, ref):
    """A/A beside A/B; a non-finite distance never agrees."""
    aa = [scaled_distance(x, y) for grp in (new, ref) for i, x in enumerate(grp) for y in grp[i + 1:]]
    ab = sorted(scaled_distance(x, y) for x in new for y in ref)
    finite = bool(aa) and bool(ab) and bool(np.isfinite(aa).all() and np.isfinite(ab).all())
    return dict(aa=max(aa) if aa else None, ab=max(ab) if ab else None, ab_median=ab[len(ab) // 2] if ab else None,
                agree=finite and ab[len(ab) // 2] <= max(aa))


def write_stata_data(folder):
    """The synthetic fixtures of the Stata leg (tests/stata/cuda_weighted_cpu_route_contract.do): cov240k with
    the fourth effect, its weights, a copy of the second effect, an instrument and a slope variable; two group()
    fixtures (rows duplicated within each group; and two individuals per group)."""
    import pandas as pd
    folder.mkdir(parents=True, exist_ok=True)
    y, x, fes = cov240k(4)
    n = len(y)
    frame = pd.DataFrame({'y': y, **{f'x{j + 1}': x[:, j] for j in range(x.shape[1])},
                          **{f'fe{j + 1}': fes[j].astype(np.int32) for j in range(4)}})
    frame['fe2c'] = frame['fe2']
    frame['w_fw'] = weight_vector('fw', n)
    frame['w_aw'] = weight_vector('aw', n)
    frame['z'] = x[:, 1] + 0.5 * np.random.default_rng(20261121).normal(size=n)
    frame.to_stata(folder / 'cov240k.dta', write_index=False, version=118)
    rng = np.random.default_rng(20261124)
    groups = 20000
    fe_a = rng.integers(0, 2000, groups).astype(np.int32)
    fe_b = rng.integers(0, 10, groups).astype(np.int32)
    x1 = rng.normal(size=groups)
    yg = 0.6 * x1 + 0.3 * rng.normal(size=2000)[fe_a] + 0.2 * rng.normal(size=10)[fe_b] + rng.normal(size=groups)
    wg = rng.uniform(.5, 2., groups)
    idx = np.repeat(np.arange(groups), 2)
    dup = pd.DataFrame(dict(gid=idx.astype(np.int32), y=yg[idx], x1=x1[idx], fe_a=fe_a[idx], fe_b=fe_b[idx],
                            w=wg[idx]))
    dup.to_stata(folder / 'groups_dup.dta', write_index=False, version=118)
    # Two distinct individuals in each group (group() with individual() needs (gid, ind) unique), drawn from 5000.
    ind = dup.copy()
    first = rng.integers(0, 5000, groups)
    second = (first + 1 + rng.integers(0, 4999, groups)) % 5000
    ind['ind'] = np.column_stack([first, second]).reshape(-1).astype(np.int32)
    ind.to_stata(folder / 'groups_ind.dta', write_index=False, version=118)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == '--write-stata-data':
        write_stata_data(Path(sys.argv[2]))
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--module', type=Path, required=True)
    ap.add_argument('--cpu-module', type=Path)
    ap.add_argument('--reference-module', type=Path)
    ap.add_argument('--reference-cpu-module', type=Path)
    ap.add_argument('--default-module', type=Path)
    ap.add_argument('--only', default=','.join(CASES))
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--keep', type=Path)
    args = ap.parse_args()
    want = set(args.only.split(','))
    if not want or not want <= set(CASES):
        ap.error("--only must name valid contract cases")
    if not cuda_device_visible():
        print(json.dumps(dict(status='NOT_EXERCISED', labels=['no CUDA device visible'])))
        return 3
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=False)
        workdir, cleanup = args.keep, None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix='cwr_contract_')
        workdir = Path(cleanup.name)
    module = args.module.resolve()
    runner = Runner(workdir, args.threads)
    rows, failures, labels, notes = [], [], [], []
    started = time.time()

    def row(case, name, problems, **extra):
        r = dict(case=case, name=name, problems=problems, **extra)
        rows.append(r)
        if problems:
            failures.append(dict(case=case, name=name, problems=problems))
        print(f"{case:10s} {name:28s} {'FAIL ' + '; '.join(problems)[:300] if problems else 'ok'}", flush=True)

    def by_name(run):
        return {f['name']: f for f in run['fits']}

    # ---- identity (and the report fields), the strict rows' method
    if 'identity' in want:
        cells = identity_cells()
        routed = by_name(runner.run(module, 'identity_routed', cells, backend='cuda', route=True))
        cpu = by_name(runner.run(module, 'identity_cpu', cells, backend='cpu'))
        for c in cells:
            r, k = routed.get(c['name']), cpu.get(c['name'])
            p = compare(r, k)
            extra = dict(method=(k or {}).get('method'), retry=(k or {}).get('retry'),
                         refused=(k or {}).get('refused'), error=((k or {}).get('error') or '')[:200])
            if c.get('mode') == 'strict-residual':
                # The absolute check of strict is made on the CPU only by the sweep methods' polish (the binding's
                # names), never by the weighted 2x2 direct projection, which returns before the sweep.
                extra['absolute_check'] = (k or {}).get('method') in ('gauss_seidel', 'symmetric_gauss_seidel',
                                                                      'jacobi') and c.get('data') != 'two_by_two'
                extra['iterations'] = (k or {}).get('iterations')
                if c['name'] == 'aw_3fe_gs_strict' and not extra['absolute_check']:
                    p.append(f"the gauss-seidel strict cell did not end on a sweep method: {(k or {}).get('method')}")
            if c['name'].endswith('slopes_strict') and not (k or {}).get('refused'):
                p.append('strict with slopes was not refused before estimation')
            if not c['name'].endswith('slopes_strict') and (k or {}).get('refused'):
                p.append(f"the CPU call refused: {(k or {}).get('error', '')[:200]}")
            row('identity', c['name'], p, **extra)
    # ---- the weighted CPU retry fires, and the routed call repeats it
    if 'retry' in want:
        # The saved-effects projection of 16nov (E1) runs the comparable route from its sweep, where the weighted
        # CPU retry fires on cov240k with three effects (run 1: eligible, fired, status 2, Gauss-Seidel primary
        # of 380 iterations, MLSMR retry of 9); the plain fits go to MLSMR at once and do not reach it.
        cells = [spec('retry_aw_3fe_savefe', weights='aw', nfe=3, retain_fes=True),
                 spec('retry_fw_3fe_savefe', weights='fw', nfe=3, retain_fes=True),
                 spec('retry_aw_2fe', weights='aw', nfe=2), spec('retry_aw_pathlike', data='pathlike', nfe=2,
                                                                 weights='aw', se='robust')]
        routed = by_name(runner.run(module, 'retry_routed', cells, backend='cuda', route=True))
        cpu = by_name(runner.run(module, 'retry_cpu', cells, backend='cpu'))
        fired = [c['name'] for c in cells if (cpu.get(c['name']) or {}).get('retry', [False, False])[1]]
        for c in cells:
            row('retry', c['name'], compare(routed.get(c['name']), cpu.get(c['name'])),
                retry=(cpu.get(c['name']) or {}).get('retry'), method=(cpu.get(c['name']) or {}).get('method'))
        if not fired:
            labels.append('NOT_EXERCISED: no retry cell fired the weighted CPU retry in the CPU arm')
        else:
            notes.append(f'retry fired in the CPU arm of: {fired}')
    # ---- boundaries
    if 'boundaries' in want:
        cells = [spec('one_effect_aw', weights='aw', nfe=1, se='robust'), spec('two_effects_aw', weights='aw', nfe=2),
                 spec('no_weights_2fe', weights='none', nfe=2), spec('ones_2fe', weights='ones', nfe=2),
                 spec('dup_fe_aw', weights='aw', nfe=1, dup_fe=True, se='robust'),
                 spec('fast_aw_2fe', weights='aw', nfe=2, mode='xhdfe-fast'),
                 spec('group_aw_2fe', weights='aw', nfe=2, group='group', se='robust'),
                 spec('group_individual_aw', weights='aw', nfe=2, group='individual', se='robust')]
        b = by_name(runner.run(module, 'boundaries_cuda', cells, backend='cuda', route=True))
        ref = by_name(runner.run(args.reference_module.resolve(), 'boundaries_ref', cells, backend='cuda')) \
            if args.reference_module else {}
        expect_route = {'two_effects_aw': True, 'ones_2fe': True, 'dup_fe_aw': True, 'group_aw_2fe': True}
        for c in cells:
            f = b.get(c['name']) or {}
            p = []
            if expect_route.get(c['name']):
                if f.get('gpu_status_code') != ROUTE or f.get('gpu_used'):
                    p.append(f"expected routed: code {f.get('gpu_status_code')} gpu_used {f.get('gpu_used')}")
            else:
                if f.get('gpu_status_code') == ROUTE:
                    p.append('routed, expected the device route')
                if c['name'] in ('one_effect_aw', 'no_weights_2fe', 'fast_aw_2fe', 'group_individual_aw') and not \
                        (f and not f.get('refused') and f.get('gpu_used') and f.get('gpu_status_code') == 1 and
                         f.get('converged')):
                    p.append(f"expected the GPU used, converged, code 1: code {f.get('gpu_status_code')} "
                             f"{f.get('error')}")
                if ref and c['name'] in ref:
                    rf = ref[c['name']]
                    if (rf.get('gpu_used'), rf.get('gpu_status_code'), rf.get('refused')) != \
                            (f.get('gpu_used'), f.get('gpu_status_code'), f.get('refused')):
                        p.append(f"not today's outcome: new {(f.get('gpu_used'), f.get('gpu_status_code'))} "
                                 f"reference {(rf.get('gpu_used'), rf.get('gpu_status_code'))}")
            row('boundaries', c['name'], p, code=f.get('gpu_status_code'), gpu_used=f.get('gpu_used'))
        if not ref:
            labels.append('NOT_EXERCISED: boundaries against the reference module (no --reference-module)')
    # ---- errors that stay, and schwarz
    if 'errors' in want:
        if args.cpu_module:
            for mod, label in ((args.cpu_module, 'cpu_only_new'), (args.reference_cpu_module, 'cpu_only_ref')):
                if mod is None:
                    continue
                e = runner.run(mod.resolve(), label, [spec('unavailable', weights='aw', nfe=2)],
                               backend='cuda', route=True if label == 'cpu_only_new' else None)
                f = (e['fits'] or [{}])[0]
                p = [] if f.get('refused') and 'Requested GPU backend is unavailable' in (f.get('error') or '') \
                    else [f"expected today's unavailable error: {f.get('error')}"]
                row('errors', label, p, error=(f.get('error') or '')[:160])
        else:
            labels.append('NOT_EXERCISED: the CPU-only module (no --cpu-module)')
        cells = [spec('lsmr_aw', weights='aw', nfe=2, method='lsmr'), spec('mlsmr_aw', weights='aw', nfe=2,
                                                                           method='mlsmr')]
        e = by_name(runner.run(module, 'explicit_krylov', cells, backend='cuda', route=True))
        for c in cells:
            f = e.get(c['name']) or {}
            row('errors', c['name'], [] if f.get('refused') and 'CPU-only' in (f.get('error') or '')
                else [f"expected today's CPU-only error: {f.get('error')}"])
        cells = [spec('schwarz_aw', weights='aw', nfe=2, method='schwarz'),
                 spec('schwarz_aw_slopes', weights='aw', nfe=2, method='schwarz', slopes=True)]
        routed = by_name(runner.run(module, 'schwarz_routed', cells, backend='cuda', route=True))
        cpu = by_name(runner.run(module, 'schwarz_cpu', cells, backend='cpu'))
        device = by_name(runner.run(module, 'schwarz_switch0', cells, backend='cuda', route=False))
        for c in cells:
            r, k = routed.get(c['name']) or {}, cpu.get(c['name']) or {}
            p = []
            if c['name'] == 'schwarz_aw' and not k.get('refused'):
                p.append('the CPU call did not refuse the ordinary explicit schwarz with weights')
            if k.get('refused'):
                if not r.get('refused') or r.get('error') != k.get('error'):
                    p.append(f"routed {r.get('error')} vs cpu {k.get('error')}")
                if r.get('estimates_left'):
                    p.append('estimates left after the refusal')
            else:
                p += compare(r, k)
            d = device.get(c['name']) or {}
            row('errors', c['name'], p, cpu_answer=('refused: ' + (k.get('error') or '')[:160]) if k.get('refused')
                else f"estimates, method {k.get('method')}",
                device_answer_switch0=('refused: ' + (d.get('error') or '')[:160]) if d.get('refused')
                else f"estimates, gpu_used {d.get('gpu_used')} code {d.get('gpu_status_code')} method {d.get('method')}")
    # ---- the switch
    if 'switch' in want:
        value_cells = [spec('switch_probe', weights='aw', nfe=2)]
        for value, routes in ((None, False), (False, False), (True, True)):
            s = runner.run(module, f'switch_{value}', value_cells, backend='cuda', route=value)
            f = (s['fits'] or [{}])[0]
            ok = (f.get('gpu_status_code') == ROUTE and not f.get('gpu_used')) if routes else \
                (f.get('gpu_used') is True and f.get('gpu_status_code') == 1)
            row('switch', f'value {value!r}', [] if ok else
                [f"code {f.get('gpu_status_code')} gpu_used {f.get('gpu_used')} (expected {'7' if routes else 'device'})"])
        three = [spec('switch0_aw_3fe', weights='aw', nfe=3)]
        new = [(runner.run(module, f'switch0_new_{i}', three, backend='cuda')['fits']
                or [{}])[0] for i in range(AB_PROCESSES)]
        p = [f"process {i}: gpu_used {f.get('gpu_used')} code {f.get('gpu_status_code')}" for i, f in enumerate(new)
             if not (f.get('gpu_used') is True and f.get('gpu_status_code') == 1)]
        d = None
        if args.reference_module:
            ref = [(runner.run(args.reference_module.resolve(), f'switch0_ref_{i}', three, backend='cuda')['fits']
                    or [{}])[0] for i in range(AB_PROCESSES)]
            p += [f"reference {i}: gpu_used {f.get('gpu_used')}" for i, f in enumerate(ref) if not f.get('gpu_used')]
            if not p:
                d = aa_ab(new, ref)
                if not d['agree']:
                    p.append(f"A/B median {d['ab_median']} above A/A {d['aa']}")
            control = ref[0]
            row('switch', 'baseline_power_control',
                [] if not control.get('refused') and control.get('converged') and control.get('certified') and
                control.get('gpu_used') and control.get('gpu_status_code') == 1
                else ['the old interface must complete a valid fit on CUDA with status 1'])
        else:
            labels.append('NOT_EXERCISED: switch 0 against the reference module (no --reference-module)')
        row('switch', 'switch0_vs_reference', p, aa_ab=d)
    # ---- sessions
    if 'sessions' in want:
        seq = [spec('s_routed_1', weights='aw', nfe=3, weighted_cpu_route=True),
               spec('s_device_1', weights='aw', nfe=3),
               spec('s_routed_2', weights='aw', nfe=3, weighted_cpu_route=True),
               spec('s_device_2', weights='aw', nfe=3, weighted_cpu_route=False)]
        for order, fits in (('routed_first', seq), ('device_first', seq[1:] + seq[:1])):
            run_ = runner.run(module, f'session_{order}', fits, backend='cuda', same_object=True)
            s = by_name(run_)
            p = [] if run_['rc'] == 0 else [f"the session process failed: rc {run_['rc']}"]
            p += [f'{c["name"]}: no record' for c in fits if c['name'] not in s]
            p += [f'{n}: refused {f.get("error")}' for n, f in s.items() if f.get('refused')]
            for name, f in s.items():
                want_route = name.startswith('s_routed')
                if want_route and (f.get('gpu_status_code') != ROUTE or f.get('gpu_used')):
                    p.append(f"{name}: code {f.get('gpu_status_code')}")
                if not want_route and not (f.get('gpu_used') and f.get('gpu_status_code') == 1):
                    p.append(f"{name}: gpu_used {f.get('gpu_used')} code {f.get('gpu_status_code')}")
            if s.get('s_routed_1', {}).get('b_hex') != s.get('s_routed_2', {}).get('b_hex'):
                p.append('the two routed fits of one object differ')
            row('sessions', order, p)
        throws = [spec('t_schwarz', weights='aw', nfe=2, method='schwarz', weighted_cpu_route=True),
                  spec('t_maxiter', weights='aw', nfe=3, max_iter=3, weighted_cpu_route=True)]
        # Each refusal is raised inside the CPU path of a routed call; the next fit of the same object (and
        # thread) must really run on the GPU, so the override was released by the throw.
        # One process and one thread, a new object per fit (an object's method and max_iter are fixed): the
        # override is thread-local, so the device fit after the refusal is the test of its release.
        s = by_name(runner.run(module, 'session_throws_schwarz', throws[:1] +
                               [spec('t_device_a', weights='none', nfe=2)], backend='cuda'))
        s2 = by_name(runner.run(module, 'session_throws_maxiter', throws[1:] +
                                [spec('t_device_b', weights='none', nfe=3)], backend='cuda'))
        # An invalid per-fit request fails without leaving its previous fit posted.
        invalid = [spec('t_start', weights='aw', nfe=2, weighted_cpu_route=True),
                   spec('t_invalid', weights='aw', nfe=2, weighted_cpu_route=1),
                   spec('t_after', weights='aw', nfe=2)]
        si = by_name(runner.run(module, 'session_invalid_request', invalid, backend='cuda', same_object=True))
        first, bad, after = (si.get(c['name'], {}) for c in invalid)
        row('sessions', 'invalid_request', [] if first.get('gpu_status_code') == ROUTE and bad.get('refused')
            and 'True or False' in bad.get('error', '') and not bad.get('estimates_left')
            and after.get('gpu_status_code') == 1 and after.get('gpu_used') else ['invalid request lifecycle failed'])
        defaults = [spec('ctor_on', weights='aw', nfe=2, constructor_route=True),
                    spec('override_off', weights='aw', nfe=2, weighted_cpu_route=False),
                    spec('ctor_restored', weights='aw', nfe=2)]
        d = runner.run(module, 'session_constructor_default', defaults, backend='cuda', same_object=True)
        states = [(f.get('gpu_status_code'), f.get('gpu_used')) for f in d['fits']]
        row('sessions', 'constructor_default', [] if states == [(7, False), (1, True), (7, False)] else
            [f'wrong states across constructor default and override: {states}'])
        p = []
        for name, f in (('t_schwarz', s.get('t_schwarz', {})), ('t_maxiter', s2.get('t_maxiter', {}))):
            if not f.get('refused') or f.get('estimates_left'):
                p.append(f"{name}: refused {f.get('refused')} estimates_left {f.get('estimates_left')}")
        for name, f in (('t_device_a', s.get('t_device_a', {})), ('t_device_b', s2.get('t_device_b', {}))):
            if not (f.get('gpu_used') and f.get('gpu_status_code') == 1):
                p.append(f"{name}: gpu_used {f.get('gpu_used')} code {f.get('gpu_status_code')} {f.get('error')}")
        row('sessions', 'refusals_inside_cpu_path', p, schwarz=(s.get('t_schwarz') or {}).get('error', '')[:120],
            maxiter=(s2.get('t_maxiter') or {}).get('error', '')[:120])
    # ---- caches and the profile
    if 'caches' in want:
        for nfe in (2, 3):
            cache = workdir / f'abs_cache_{nfe}.bin'
            env = {'XHDFE_ABSORPTION_CACHE': str(cache), 'XHDFE_ABSORPTION_CACHE_MODE': 'auto'}
            cells = [spec(f'c_write_{nfe}', weights='aw', nfe=nfe), spec(f'c_hit_{nfe}', weights='aw', nfe=nfe)]
            r = by_name(runner.run(module, f'cache_routed_{nfe}', cells, backend='cuda', env=env,
                                   same_object=True, route=True))
            k = by_name(runner.run(module, f'cache_cpu_{nfe}', [spec(f'c_cpu_hit_{nfe}', weights='aw', nfe=nfe)],
                                   backend='cpu', env=env))
            w, h, c = r.get(f'c_write_{nfe}', {}), r.get(f'c_hit_{nfe}', {}), k.get(f'c_cpu_hit_{nfe}', {})
            p = []
            if w.get('tier_reason') == 'cache':
                p.append('the first routed fit hit an entry (expected a write)')
            if h.get('tier_reason') != 'cache' or h.get('gpu_status_code') != ROUTE:
                p.append(f"routed hit: reason {h.get('tier_reason')} code {h.get('gpu_status_code')}")
            if c.get('tier_reason') != 'cache' or c.get('gpu_status_code') != 0:
                p.append(f"CPU call on the routed entry: reason {c.get('tier_reason')} code {c.get('gpu_status_code')}")
            if h.get('b_hex') != c.get('b_hex') or h.get('V_sha') != c.get('V_sha'):
                p.append('the routed hit and the CPU hit differ')
            if not cache.is_file():
                p.append('no cache file written')
            row('caches', f'absorption_cache_{nfe}fe', p)
            prof = {arm: workdir / f'profile_{nfe}_{arm}.txt' for arm in ('routed', 'cpu')}
            for arm, backend in (('routed', 'cuda'), ('cpu', 'cpu')):
                runner.run(module, f'profile_write_{nfe}_{arm}', [spec(f'p_{arm}', weights='aw', nfe=nfe)],
                           backend=backend, route=True if arm == 'routed' else None,
                           env={'XHDFE_MOBILITY_PROFILE': str(prof[arm]), 'XHDFE_MOBILITY_MODE': 'write'})
            same = all(prof[a].is_file() for a in prof) and prof['routed'].read_bytes() == prof['cpu'].read_bytes()
            reads = {arm: [(runner.run(module, f'profile_read_{nfe}_{arm}_{i}',
                                       [spec('p_device', weights='none', nfe=nfe)], backend='cuda',
                                       env={'XHDFE_MOBILITY_PROFILE': str(prof[arm]), 'XHDFE_MOBILITY_MODE': 'read'})
                            ['fits'] or [{}])[0] for i in range(AB_PROCESSES)] for arm in prof}
            p = [] if same else ['the routed and the CPU call wrote different profiles']
            p += [f'{arm} reader {i}: gpu_used {f.get("gpu_used")}' for arm, fs in reads.items()
                  for i, f in enumerate(fs) if not f.get('gpu_used')]
            d = aa_ab(reads['routed'], reads['cpu']) if not p else None
            if d and not d['agree']:
                p.append(f"device readers: A/B median {d['ab_median']} above A/A {d['aa']}")
            row('caches', f'profile_{nfe}fe', p, aa_ab=d)
    # ---- internal callers
    if 'internal' in want:
        g = by_name(runner.run(module, 'gelbach_gpu', [spec('gel', kind='gelbach', gpu=True),
                                                       spec('gel_boot_required', kind='gelbach_bootstrap', gpu=True,
                                                            require_gpu_used=True)], backend=None, route=True))
        f, bt = g.get('gel', {}), g.get('gel_boot_required', {})
        p = []
        if f.get('refused') or f.get('gpu_status_code') != ROUTE or f.get('gpu_status') != 'weighted_cpu_route':
            p.append(f"gelbach: {f.get('gpu_status_code')} {f.get('gpu_status')} {f.get('error')}")
        if not bt.get('refused'):
            p.append('the bootstrap with require_gpu_used=True did not refuse')
        row('internal', 'gelbach_gpu_true', p, bootstrap_error=(bt.get('error') or '')[:160])
        boots = [spec(f'boot_{route}', kind='gelbach_bootstrap', gpu=True, common=2,
                      weighted_cpu_route=route) for route in (True, False)]
        h = by_name(runner.run(module, 'gelbach_boot_gpu', boots, backend=None))
        for route in (True, False):
            col = h[f'boot_{route}']
            row('internal', f'bootstrap_route_{route}', [] if not col.get('refused') and
                col.get('gpu_status_code') == (7 if route else 1) and col.get('reps_valid') == 3 and
                col.get('replicate_gpu') == [not route] * 3 else [f'incorrect point or replicate backends: {col}'])
        env_cells = [spec(f'gel_env_{c}', kind='gelbach', gpu=False, common=c) for c in (0, 1, 2)]
        run_ = runner.run(module, 'gelbach_env', env_cells, backend='cuda', route=True)
        e = by_name(run_)
        p = [] if run_['rc'] == 0 else [f"the process failed: rc {run_['rc']}"]
        z = e.get('gel_env_0')
        # No common effect: the base regression has none, and an environment request for CUDA without effects is
        # refused as today (validation before the route); one and two common effects: the full model is routed.
        if not z or not z.get('refused') or 'at least one fixed-effect dimension' not in (z.get('error') or ''):
            p.append(f"gel_env_0: expected today's refusal, got {z}")
        for name in ('gel_env_1', 'gel_env_2'):
            v = e.get(name)
            if not v or v.get('refused') or (v.get('gpu_status_code'), v.get('gpu_status')) != (ROUTE, 'weighted_cpu_route'):
                p.append(f'{name}: expected 7 weighted_cpu_route, got {v}')
        row('internal', 'gelbach_environment', p,
            outcomes={k: (v.get('error')[:120] if v.get('refused') else (v.get('gpu_status_code'), v.get('gpu_status')))
                      for k, v in e.items()})
        run_a = runner.run(module, 'akm_new', [spec('akm', kind='akm')], backend=None)
        a = by_name(run_a)
        if args.reference_module:
            run_r = runner.run(args.reference_module.resolve(), 'akm_ref', [spec('akm', kind='akm')], backend=None)
            ra = by_name(run_r)
            na, rr = a.get('akm', {}), ra.get('akm', {})
            keys = sorted(k for k in set(na) | set(rr) if 'gpu' in k or 'status' in k)
            p = [f'{k}: new {na.get(k)} ref {rr.get(k)}' for k in keys if na.get(k) != rr.get(k)]
            for arm, run_x, rec in (('new', run_a, na), ('ref', run_r, rr)):
                if run_x['rc'] != 0 or not rec or rec.get('refused') or not any('gpu' in k for k in rec):
                    p.append(f"{arm}: rc {run_x['rc']}, refused {rec.get('refused')} {rec.get('error')}")
            row('internal', 'akm_as_today', p)
        else:
            labels.append('NOT_EXERCISED: AKM against the reference module (no --reference-module)')
    # ---- the contract
    if 'contract' in want:
        sys.path.insert(0, str(AP))
        cells = [spec(f'k_{kind}_3fe', weights=kind, nfe=3) for kind in ('fw', 'aw', 'pw', 'iw')]
        cells += [spec('k_aw_2fe', weights='aw', nfe=2), spec('k_aw_4fe', weights='aw', nfe=4),
                  spec('k_aw_3fe_strict', weights='aw', nfe=3, mode='strict-residual'),
                  spec('k_aw_3fe_savefe', weights='aw', nfe=3, retain_fes=True),
                  spec('k_aw_4fe_savefe', weights='aw', nfe=4, retain_fes=True),
                  spec('k_gap_aw_3fe_gs', weights='aw', nfe=3, method='gauss-seidel'),
                  spec('k_group_aw_2fe', weights='aw', nfe=2, group='group'),
                  spec('k_group_aw_3fe', weights='aw', nfe=3, group='group')]
        routed = by_name(runner.run(module, 'contract_routed', cells, backend='cuda', route=True))
        device = by_name(runner.run(module, 'contract_switch0', cells[:2] + cells[5:6], backend='cuda', route=False))
        oracles = {}
        for c in cells:
            key = (c['weights'], c['nfe'])
            if key not in oracles:
                y, x, fes = cov240k(c['nfe'])
                oracles[key] = weighted_oracle(y, x, fes, weight_vector(c['weights'], len(y)), c['weights'] == 'fw')
            o, f = oracles[key], routed.get(c['name']) or {}
            p = [] if o['defect'] < 5e-14 else [f"oracle projection unresolved ({o['defect']})"]
            if f.get('refused') or f.get('gpu_status_code') != ROUTE:
                p.append(f"routed fit: refused {f.get('refused')} code {f.get('gpu_status_code')} {f.get('error')}")
                row('contract', c['name'], p)
                continue
            if abs(f.get('df_a', -1) - o['df_a']) > 0:
                p.append(f"df_a {f.get('df_a')} against the oracle's {o['df_a']}")
            gr = grade(o, f)
            gap = c['name'].startswith('k_gap')
            if gr.get('ungraded'):
                p.append(f"not gradable: {gr['ungraded']}")
            elif gap:
                if not (inside(gr['b_scaled'], CPU_GAP_BOUND['b_scaled']) and
                        inside(gr['se_rel'], CPU_GAP_BOUND['se_rel'])):
                    p.append(f'expected-outside cell beyond its stated bound {CPU_GAP_BOUND}: {gr}')
                notes.append(f"EXPECTED_OUTSIDE {c['name']}: {gr} (inside the contract: "
                             f"{inside(gr['b_scaled'], LIMIT_B) and inside(gr['se_rel'], LIMIT_SE)})")
            elif c.get('group'):
                # The collapsed fit of exact duplicate rows is the ungrouped fit: its coefficients are graded
                # against the ungrouped oracle; its standard errors are recorded.
                if not inside(gr['b_scaled'], LIMIT_B):
                    p.append(f'coefficients outside the contract: {gr}')
            elif not (inside(gr['b_scaled'], LIMIT_B) and inside(gr['se_rel'], LIMIT_SE)):
                p.append(f'outside the contract: {gr}')
            d = device.get(c['name'])
            row('contract', c['name'], p, vs_oracle=gr, method=f.get('method'),
                switch0=(grade(o, d) if d and not d.get('refused') else None))
    # ---- what must not change: the CPU backend bit for bit against the reference modules (CPU-only and CUDA
    # builds), and the unrouted device fits A/A beside A/B against the reference CUDA module
    if 'nochange' in want:
        cpu_cells = [spec('n_none_3fe', weights='none'), spec('n_aw_3fe', weights='aw'),
                     spec('n_fw_2fe_strict', weights='fw', nfe=2, mode='strict-residual'),
                     spec('n_aw_1fe', weights='aw', nfe=1, se='robust'), spec('n_aw_3fe_fast', weights='aw',
                                                                             mode='xhdfe-fast'),
                     spec('n_aw_3fe_savefe', weights='aw', retain_fes=True),
                     spec('n_group_aw_2fe', weights='aw', nfe=2, group='group', se='robust')]
        keys = IDENTITY_KEYS + ('gpu_status_code',)
        pairs = ((args.cpu_module, args.reference_cpu_module, 'cpu_build'),
                 (args.module, args.reference_module, 'cuda_build'))
        for mod, ref, tag in pairs:
            if mod is None or ref is None:
                labels.append(f'NOT_EXERCISED: nochange on the CPU backend of the {tag} (a module is missing)')
                continue
            a = by_name(runner.run(mod.resolve(), f'nochange_{tag}_new', cpu_cells, backend='cpu'))
            b = by_name(runner.run(ref.resolve(), f'nochange_{tag}_ref', cpu_cells, backend='cpu'))
            for c in cpu_cells:
                fa, fb = a.get(c['name']) or {}, b.get(c['name']) or {}
                p = [f'{k}: new {str(fa.get(k))[:100]} ref {str(fb.get(k))[:100]}' for k in keys
                     if fa.get(k) != fb.get(k)]
                if not fa or fa.get('refused'):
                    p.append(f"no estimates: {fa.get('error')}")
                row('nochange', f'{tag}:{c["name"]}', p)
        device_cells = [spec('d_none_3fe', weights='none'), spec('d_aw_1fe', weights='aw', nfe=1, se='robust'),
                        spec('d_aw_3fe_fast', weights='aw', mode='xhdfe-fast'),
                        spec('d_group_none_2fe', weights='none', nfe=2, group='group', se='robust')]
        if args.reference_module:
            new = [by_name(runner.run(module, f'nochange_device_new_{i}', device_cells, backend='cuda'))
                   for i in range(AB_PROCESSES)]
            ref = [by_name(runner.run(args.reference_module.resolve(), f'nochange_device_ref_{i}', device_cells,
                                      backend='cuda')) for i in range(AB_PROCESSES)]
            for c in device_cells:
                fn, fr = [r.get(c['name']) or {} for r in new], [r.get(c['name']) or {} for r in ref]
                p = [f"{arm} {i}: gpu_used {f.get('gpu_used')} code {f.get('gpu_status_code')} {f.get('error')}"
                     for arm, fs in (('new', fn), ('ref', fr)) for i, f in enumerate(fs)
                     if f.get('refused') or not f.get('gpu_used') or f.get('gpu_status_code') != 1]
                d = None
                if not p:
                    d = aa_ab(fn, fr)
                    if not d['agree']:
                        p.append(f"A/B median {d['ab_median']} above A/A {d['aa']}")
                row('nochange', f'device:{c["name"]}', p, aa_ab=d)
        else:
            labels.append('NOT_EXERCISED: nochange on the device (no --reference-module)')
    # ---- the CUDA-default module
    if 'default' in want:
        if args.default_module:
            dm = args.default_module.resolve()
            cells = [spec('d_aw_2fe', weights='aw', nfe=2, weighted_cpu_route=True),
                     spec('d_default_aw_2fe', weights='aw', nfe=2)]
            with_dev = by_name(runner.run(dm, 'default_device', cells, backend=None))
            no_dev = by_name(runner.run(dm, 'default_no_device', cells, backend=None, env={'CUDA_VISIBLE_DEVICES': ''}))
            p = []
            a = with_dev.get('d_aw_2fe', {})
            if a.get('refused') or a.get('gpu_status_code') != ROUTE or a.get('gpu_used'):
                p.append(f"default build, device: weighted code {a.get('gpu_status_code')} gpu_used {a.get('gpu_used')}")
            # Without a device the predicate is false (the availability term), so the fit is today's: the CPU
            # fallback with status 2 (hdfe_regressor_v11.cpp, the unavailable-backend fallback), never 7.
            f = no_dev.get('d_aw_2fe', {})
            if f.get('refused') or f.get('gpu_status_code') != 2 or f.get('gpu_used'):
                p.append(f"default build, no device, weighted: refused {f.get('refused')} code "
                         f"{f.get('gpu_status_code')} {f.get('error')}")
            # The unweighted fits are recorded only: in a CUDA-default build with no request the reduced tier,
            # which reads the request and not the build default, serves them on the CPU (by reading; no reference build).
            row('default', 'cuda_default_module', p,
                device={k: (v.get('gpu_status_code'), v.get('gpu_used'), v.get('method'), v.get('tier_reason'))
                        for k, v in with_dev.items()},
                no_device={k: (v.get('gpu_status_code'), v.get('gpu_used'), v.get('method'), v.get('refused'))
                           for k, v in no_dev.items()})
        else:
            labels.append('NOT_EXERCISED: the CUDA-default module (no --default-module)')
    status = 'FAIL' if failures else ('NOT_EXERCISED' if labels else 'PASS')
    report = dict(status=status, failures=failures, labels=labels, notes=notes, rows=rows, fits=runner.count,
                  module=str(module), module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                  seconds=time.time() - started, gpu_apps_at_start=gpu_apps())
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'failures', 'labels', 'notes', 'fits', 'seconds')},
                     indent=1)[:20000])
    if cleanup:
        cleanup.cleanup()
    return 1 if failures else (3 if labels else 0)


if __name__ == '__main__':
    sys.exit(main())
