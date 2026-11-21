"""Package B (18nov2026): the strict-residual acceptance of CUDA fits in scope, graded against independent
oracles (offline, CUDA only; the Stata leg with --stata-package).

A strict-residual fit on CUDA is in scope when it has the automatic method and criterion, no weights, no saved
effects and no group() (instruments are in scope). It takes the reghdfe-comparable CUDA route and is accepted
when it is certified, the existing checks pass and D(e) <= tol: the largest absolute group mean of the delivered
residual (y~ - X~b; with instruments the structural residual), in the outcome's units, at the public tolerance,
read by the core on the host in long double. One refinement per fit (the existing N1 refinement or the strict
continuation); a second failure, or an exhausted budget, refuses with no estimates. On a sweep exit the
sweep-exit step (b0 from the Gram of [X~|y~], counted sweeps of the combination) runs inside the primary; its
direct form hands the state over when its last three power-of-two readings are bit for bit the same
(XHDFE_STRICT_FROZEN_STOP=0 turns that frozen stop off; any other value, unset included, is the default).
XHDFE_STRICT_CUDA_LEGACY_ACCEPTANCE=1 restores the earlier
branch (the negative control). XHDFE_STRICT_ACCEPT_TRACE=1 (2: every fresh reading) prints the trace graded here.

Limits, pre-registered (Verifications/corrigir_erros_20261117/b/DESENHO_B.md section 9, not adjusted afterwards):
coefficients 1e-8 max(1,|b|); standard errors 1e-7 relative; the complete covariance 1e-8 sqrt(V_jj V_kk); the
RSS 1e-8 relative (source: the covariance limit, since the unadjusted V is proportional to rss/df); D(e) <= tol
measured here with long-double group sums on the delivered residual. The distance of the residuals from the
oracle is recorded, never graded. With four effects only coefficients are graded (their covariance needs the
exact redundancy of four effects; the Stata leg grades it against reghdfe at 1e-12 when reghdfe is installed).

Oracles (independent of xhdfe): the within transform of the anchored dummy system (sparse LU, four long-double
refinements; cuda_fast_restart_contract.within_transform), MP80 slopes, the constant from the means, the
unadjusted covariance with the constant's transport (Verifications/perf_fereg_gap_20261106/p3fix/
p3fix_unit_weight_probe55.py:110 ols_full; the degrees of freedom of the absorbed effects counted from their
levels on one connected component, asserted); 2SLS in MP80 on the same within data for instruments; per
component for the disconnected fixture; the public cov240k oracle (b, cluster V, e; --v3-npz, sha256 checked).

Cases (one fresh process per fit or per session; tol 1e-8 unless stated; --threads, default 16):
* scope: F0 to F4 (path-like cycles, p7p6_triage.py:95-141), T3 and W0 (p6_fixtures.py:116-145), cov240k
  (three-way): accepted, converged, certified, GPU status 1, iterations <= max_iter, the trace's limit equal to
  the tolerance and its last decision `accept`, the tier's reason `tolerance_mode`, D <= tol, the limits, and
  the delivered residual is the fit's own (its sum of squares equals rss_ to 1e-10 relative).
* continuation: cov240k with the outcome times 2^17 (exact), and with one instrument: pass 1 certified with the
  checks passed and D > tol, the continuation, pass 2 D <= tol, accepted, graded against the scaled oracle.
* continuation_short (one object, max_iter = the calibrated pass-1 iterations of cov240k x 2^17 + 24, package C):
  easy (accepted), cov240k x 2^17 (the continuation runs to the budget: refused at max_iter/max_iter with no
  estimates left and the reason "the fit's single refinement was exhausted"; run again, up to three times, only on
  the expected refusal before the continuation), easy again (accepted, equal to the first within the limit). The
  Stata leg's short budget is the same; without a calibration both record missing coverage.
* instruments: cov240k with one instrument (unadjusted V, 2SLS oracle); the legacy arm is reported, not graded.
* legacy (negative control): F4 with the legacy switch reproduces the reference module's numbers (A/B within
  A/A) and lies outside the limits (today: about 1.5e-7 in b and 1.7e-4 in the SE).
* no_regressors (cov240k), omitted (cov240k with a duplicated column), disconnected (two components).
* rescaled: x1 times 2^10 and 2^-10: the same decision, D <= tol, the trace's limit equal to the tolerance,
  the limits after undoing the scale.
* tolerances 1e-8+1ulp and 2e-8; maxiter at the boundary (N-1, N, N+1 from a calibration fit): each fit either
  accepted within its budget and inside the limits, or refused at exactly its budget with no estimates.
* session: easy, cov240k with max_iter 30 (refused 30/30, no estimates), easy (equal to the first).
* step rows (sweep exits): T3 without regressors (exit=confirmed, one pass, no continuation); cov240k with a
  fourth effect and an instrument (skipped=instruments) and with a duplicated column (skipped=rank), each accepted;
  the symmetric form (no power-of-two reading in loop=step at trace level 2); XHDFE_STRICT_FROZEN_STOP=0 on T3
  without regressors (no power-of-two reading in loop=step; readings equal confirmations).
* large scale (a deterministic row; W0 with the outcome times 2^LARGE_EXPONENT, whose floor the trace shows above
  ten tolerances at pass 1, certified with the checks passed): one object with max_iter LARGE_BUDGET: refused at
  exactly the budget, no estimates, the text carries D(e), the iterations and the reason; W0 then accepted on the
  same object (also in the Stata leg, where the text is read from the log).
* all_is_default (stage 2c addendum): XHDFE_STRICT_FROZEN_STOP=all now means the default (the value `all` and the
  continuation's frozen exit were removed): T3 without regressors accepted as by default, its step line printing
  frozen_stop=step.
* excluded (--reference-module; --cpu-module with --reference-cpu-module): weights (package C: the routed arm, the
  same module with the CUDA and the CPU backend, code 7 against 0 and then exact, runs without a reference too;
  the comparison with the reference module uses the default device route), auto-mlsmr, an explicit
  method (gauss-seidel), an explicit criterion (reghdfe), saved effects, a group() fit without individual(), slopes
  with the automatic and with an explicit criterion
  (the error before estimation) on CUDA, EXCLUDED_RUNS processes per arm, A/B within A/A; CPU strict bit for bit.
  Before any distance every process of both arms has the outcome its case expects (the error before estimation
  for slopes; otherwise accepted, converged, certified, finite b and V and, on CUDA, the GPU used with status 1;
  in Stata the session's SCA_OK line with its estimates and stored residuals), or the row fails, naming the arm
  and index; a non-finite distance never agrees. A CUDA row whose largest within-arm distance is above the
  coefficient limit (1e-8) gets the note WITHIN_DISPERSION_ONLY: its agreement shows only that the median A/B lies
  within the run-to-run dispersion of the two binaries, not numerical equivalence (no change of the exit code).
  (Five per arm since stage 2c, a change after a result: with three, the Stata comparison of criterion_reghdfe
  failed with median A/B 1.0293e-11 against largest A/A 1.0261e-11; the residual variable set every distance
  (6.1e-12 to 1.07e-11; b and V at most 7.8e-13). Looked at after that failure, the observed split was the only
  one of the ten splits of those six sessions into two arms that fails the rule (b/stage2c/aa_permutation.txt);
  that is a description of those six sessions, not a false-alarm rate. With five, the median A/B exceeds the
  largest A/A only when 13 of the 25 cross distances exceed all 20 within-arm ones.)
Stata leg: as above per leg, plus the four-effect fit (cov240k with the 50-level fourth effect) at tolerance(1e-8):
D, the diagnostics and the coefficients (oracle; reghdfe at 1e-12) asserted; its standard errors against reghdfe
are the known defect KNOWN_DEFECT_FOUR_EFFECT_COVARIANCE_CUDA (stage 2c, changed after its result): a demonstrated
estimator error (an MP80 oracle sides with reghdfe), earlier than package B, present in the reference binaries and
still observable in scope (better than before, not fixed). Exit code 4 when that distance lies in the declared band
(1e-7, 2e-7] (source: the in-scope occurrences so far, 1.57e-7 against reghdfe and 4.5e-8 to 1.44e-7 against the
oracle), a failure above it, nothing at or below 1e-7; whenever reghdfe ran for it (SCA_R with reghdfe_db), the
scope leg must print exactly one SCA_K line for it, or the row fails. On that fixture the ratio of the
standard-error distance to the oracle over D(e) was 13.2 to 14.4 in nine fits (six comparable, three new strict) and
26.1 in the three reference
strict fits, whose state comes from a CG stage; no bound follows (the largest new strict distance, 1.441e-7, is above
14 x 1e-8) (Verifications/corrigir_erros_20261117/b/stage2b/fourfe_run1/). Beside it, the same fit at
tolerance(1e-10) in three sessions (pre-registered in b/stage2b/PRE_REGISTO_2B_20261118.md): accepted, D <= 1e-10
on the stored residuals, coefficients within 1e-8 (oracle and reghdfe), standard errors within 1e-7 of reghdfe at
1e-12.
Exit code 0 PASS, 1 FAIL, 3 only labels of missing coverage, 4 a known defect inside its declared band; notes
(WITHIN_DISPERSION_ONLY) are reported and printed but do not change it.
usage: strict_cuda_acceptance_contract.py --module PATH.so [--v3-npz PATH] [--reference-module PATH.so]
       [--cpu-module PATH.so --reference-cpu-module PATH.so] [--stata-package DIR
       [--reference-stata-package DIR] [--reghdfe-adopath DIR]] [--only case,...] [--fits 2] [--threads 16]
       [--output result.json] [--keep DIR]
"""
import argparse
import hashlib
import json
import math
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from cuda_fast_restart_contract import (TOLERANCES, V3_NPZ, V3_NPZ_SHA256, cuda_device_visible,  # noqa: E402
                                        gpu_apps, within_transform)

TOL = 1e-8
MAX_ITER = 100000
LIMITS = dict(b_scaled=1e-8, se_rel=1e-7, V_scaled=1e-8, rss_rel=1e-8)
OWN_RESIDUAL_REL = 1e-10
Y_SCALE_EXPONENT = 17
EXCLUDED_RUNS = 5  # processes (Stata: sessions) per arm in the comparisons with the reference binaries
SHORT_MARGIN = 24  # package C: the short budget is the calibrated pass 1 + 24 (no calibration: missing coverage)
LARGE_EXPONENT = 34
LARGE_BUDGET = 600
LARGE_FLOOR_FACTOR = 10.0
TRACE_PREFIXES = ('strict_accept_trace', 'p7_refine_trace', 'cuda_tail_restart', 'gi_lsmr_solve',
                  'cuda_strict_restart', 'SCA_FIT')


# ---------------------------------------------------------------- fixtures (synthetic, fixed seeds)
def latent():
    """Verifications/perf_fereg_gap_20261106/p7p6/p7p6_triage.py:95-107 (the path-like base)."""
    from cuda_strict_restart_contract import pathlike_fixture
    y, x, fes = pathlike_fixture()
    m = (len(y) - 2) // 2
    rng = np.random.default_rng(20261109)
    alpha, beta = rng.normal(size=m), rng.normal(size=m)
    return (np.asarray(y, float), np.asarray(x[:, 0], float), np.asarray(fes[0], np.int64),
            np.asarray(fes[1], np.int64), alpha, beta, m)


def rows_like_base(rng, a, b, alpha, beta):
    """p7p6_triage.py:110-113."""
    x = rng.normal(size=len(a)) + 0.3 * alpha[a] - 0.2 * beta[b]
    y = 0.5 * x + alpha[a] + beta[b] + 0.1 * rng.normal(size=len(a))
    return y, x


def path_fixtures():
    """p7p6_triage.py:116-141: F0 and the four cycle fixtures F1 to F4."""
    y0, x0, a0, b0, alpha, beta, m = latent()
    out = {'F0': (y0, x0[:, None], a0, b0)}
    j = np.arange(40)
    la = (150 * j + 7) % m
    lb = (la + 2 + (j % 3)) % m
    y1, x1 = rows_like_base(np.random.default_rng(20261111), la, lb, alpha, beta)
    a1, b1 = np.r_[a0, la], np.r_[b0, lb]
    out['F1'] = (np.r_[y0, y1], np.r_[x0, x1][:, None], a1, b1)
    rng = np.random.default_rng(20261112)
    ra, rb = rng.integers(0, m, 40), rng.integers(0, m, 40)
    y2, x2 = rows_like_base(rng, ra, rb, alpha, beta)
    out['F2'] = (np.r_[y0, y2], np.r_[x0, x2][:, None], np.r_[a0, ra], np.r_[b0, rb])
    da, db = a0[:-2], b0[:-2]
    y3, x3 = rows_like_base(np.random.default_rng(20261113), da, db, alpha, beta)
    out['F3'] = (np.r_[y0, y3], np.r_[x0, x3][:, None], np.r_[a0, da], np.r_[b0, db])
    rng = np.random.default_rng(20261114)
    xa = np.r_[x0, x1]
    nn = len(a1)
    z = rng.normal(size=(3, nn))
    x2c = 1000.0 * (z[0] + 0.3 * alpha[a1] - 0.2 * beta[b1])
    x3c = z[1] - 0.5 * alpha[a1] + 0.4 * beta[b1]
    x4c = z[2]
    y4 = 0.5 * xa + 0.0003 * x2c - 0.2 * x3c + 0.1 * x4c + alpha[a1] + beta[b1] + 0.1 * rng.normal(size=nn)
    out['F4'] = (y4, np.column_stack([xa, x2c, x3c, x4c]), a1, b1)
    return out


def w0_fixture():
    """Verifications/perf_fereg_gap_20261106/p7p6/p6_fixtures.py:116-123."""
    rng = np.random.default_rng(20261115)
    n, la, lb = 20000, 400, 300
    a, b = rng.integers(0, la, n), rng.integers(0, lb, n)
    alpha, beta = rng.normal(size=la), rng.normal(size=lb)
    x = rng.normal(size=n) + 0.3 * alpha[a] - 0.2 * beta[b]
    y = 0.5 * x + alpha[a] + beta[b] + 0.1 * rng.normal(size=n)
    return y, x[:, None], [a.astype(np.int32), b.astype(np.int32)]


def t3_fixture():
    """p6_fixtures.py:126-145: 4 000 workers x 8 years in 12, 200 firms on a cycle of movers."""
    rng = np.random.default_rng(20261116)
    nw, nf, ny, spell = 4000, 200, 12, 8
    home = np.r_[np.arange(nf), rng.integers(0, nf, nw - nf)]
    start = rng.integers(0, ny - spell + 1, nw)
    random_mover = np.r_[np.zeros(nf, bool), rng.random(nw - nf) < 0.02]
    dest = np.where(np.arange(nw) < nf, (home + 1) % nf, rng.integers(0, nf, nw))
    w = np.repeat(np.arange(nw), spell)
    t = (np.repeat(start, spell) + np.tile(np.arange(spell), nw)).astype(np.int64)
    k = np.tile(np.arange(spell), nw)
    moved = (np.repeat((np.arange(nw) < nf) | random_mover, spell)) & (k >= spell // 2)
    f = np.where(moved, np.repeat(dest, spell), np.repeat(home, spell))
    alpha, beta, gamma = rng.normal(size=nw), rng.normal(size=nf), 0.3 * rng.normal(size=ny)
    n = len(w)
    x1 = rng.normal(size=n) + 0.3 * alpha[w] - 0.2 * beta[f] + 0.1 * gamma[t]
    x2 = 100.0 * (rng.normal(size=n) + 0.2 * alpha[w] + 0.3 * beta[f])
    y = 0.4 * x1 + 0.002 * x2 + alpha[w] + beta[f] + gamma[t] + 0.1 * rng.normal(size=n)
    return y, np.column_stack([x1, x2]), [w.astype(np.int32), f.astype(np.int32), t.astype(np.int32)]


def disconnected_fixture():
    """Two components of a random two-way graph (10 000 rows each; 200 x 150 levels per component)."""
    rng = np.random.default_rng(20261121)
    parts = []
    for c in range(2):
        n, la, lb = 10000, 200, 150
        a, b = rng.integers(0, la, n), rng.integers(0, lb, n)
        alpha, beta = rng.normal(size=la), rng.normal(size=lb) + 3.0 * c
        x = rng.normal(size=n) + 0.3 * alpha[a] - 0.2 * beta[b]
        y = 0.5 * x + alpha[a] + beta[b] + 0.1 * rng.normal(size=n)
        parts.append((y, x, a + c * la, b + c * lb))
    y, x, a, b = (np.concatenate(v) for v in zip(*parts))
    return y, x[:, None], [a.astype(np.int32), b.astype(np.int32)]


def cov240k():
    from cuda_three_fe_covariance_contract import fixture
    y, x, fes = fixture()
    return y, x, list(fes)


def fixture(name):
    """dict(y, x, fes, clusters, inst, endog, se) of one named fixture (suffixes: _2pK the outcome times 2^K,
    _x1pK / _x1mK the first regressor times 2^K / 2^-K, _nox no regressors, _iv one instrument for the second
    regressor, _dup a duplicated first column, _4fe the fourth effect of cuda_fast_restart_contract)."""
    base = name.split('_')[0]
    clusters, inst, endog, se = None, None, [], 'unadjusted'
    if base in ('F0', 'F1', 'F2', 'F3', 'F4'):
        y, x, a, b = path_fixtures()[base]
        fes = [a.astype(np.int32), b.astype(np.int32)]
    elif base == 'W0':
        y, x, fes = w0_fixture()
    elif base == 'T3':
        y, x, fes = t3_fixture()
    elif base == 'disc':
        y, x, fes = disconnected_fixture()
    elif base == 'easy':
        from cuda_strict_restart_contract import easy_fixture
        y, x, fes = easy_fixture()
    elif base == 'cov240k':
        y, x, fes = cov240k()
        if '_4fe' in name:  # cuda_fast_restart_contract.structure('4fe'): seed 20261109, 50 levels
            fes.append(np.random.default_rng(20261109).integers(0, 50, size=len(y)).astype(np.int32))
        if '_iv' not in name and '_nox' not in name:
            clusters, se = fes[0], 'cluster'
    else:
        raise ValueError(name)
    y = np.asarray(y, float).copy()
    x = np.asarray(x, float).copy()
    m = re.search(r'_2p(\d+)', name)
    if m:
        y *= 2.0 ** int(m.group(1))
    m = re.search(r'_x1([pm])(\d+)', name)
    if m:
        x[:, 0] *= 2.0 ** (int(m.group(2)) * (1 if m.group(1) == 'p' else -1))
    if '_iv' in name:  # one instrument for the second regressor, from a fixed seed
        inst = (x[:, 1] + 0.5 * np.random.default_rng(20261118).normal(size=len(y)))[:, None]
        endog = [1]
    if '_dup' in name:
        x = np.column_stack([x, x[:, 0]])
    if '_nox' in name:
        x = np.zeros((len(y), 0))
    return dict(y=y, x=np.asfortranarray(x), fes=[np.asarray(f, np.int32) for f in fes], clusters=clusters,
                inst=inst, endog=endog, se=se)


# ---------------------------------------------------------------- oracles
def components(fes):
    parent = {}

    def find(u):
        while parent.setdefault(u, u) != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    for d in range(1, len(fes)):
        for u, v in set(zip(fes[0].tolist(), fes[d].tolist())):
            ru, rv = find((0, u)), find((d, v))
            if ru != rv:
                parent[ru] = rv
    return len({find((0, int(u))) for u in np.unique(fes[0])})


def absorbed_dof(fes):
    assert components(fes) == 1, 'the oracle needs one connected component'
    for f in fes:
        assert np.bincount(f).min() >= 2, 'the oracle needs no singleton level'
    levels = [int(f.max()) + 1 for f in fes]
    return levels[0] + sum(v - 1 for v in levels[1:])


def mp_matrix(a):
    import mpmath as mp
    return mp.matrix([[mp.mpf(str(v)) for v in row] for row in np.atleast_2d(a)])


def transport_v(inv, s2, means, n, p):
    """The full covariance with the constant recovered from the means (ols_full)."""
    import mpmath as mp
    v = mp.zeros(p + 1)
    for j in range(p):
        for k in range(p):
            v[j, k] = s2 * inv[j, k]
    for k in range(p):
        v[p, k] = v[k, p] = -sum(mp.mpf(str(means[j + 1])) * s2 * inv[j, k] for j in range(p))
    v[p, p] = sum(mp.mpf(str(means[j + 1])) * mp.mpf(str(means[k + 1])) * s2 * inv[j, k]
                  for j in range(p) for k in range(p)) + s2 / n
    return np.array([[float(v[j, k]) for k in range(p + 1)] for j in range(p + 1)])


def within_by_component(y, x, fes):
    """Two effects on several components: each component's anchored within transform, stacked."""
    a, b = fes
    comp = np.full(len(y), -1)
    parent = list(range(int(a.max()) + 1 + int(b.max()) + 1))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    off = int(a.max()) + 1
    for u, v in set(zip(a.tolist(), b.tolist())):
        ru, rv = find(u), find(off + v)
        if ru != rv:
            parent[ru] = rv
    roots = np.array([find(u) for u in a.tolist()])
    within = np.zeros((len(y), 1 + x.shape[1]), np.longdouble)
    df_a, worst = 0, 0.0
    for r in np.unique(roots):
        rows = np.flatnonzero(roots == r)
        ca = np.unique(a[rows], return_inverse=True)[1].astype(np.int32)
        cb = np.unique(b[rows], return_inverse=True)[1].astype(np.int32)
        w, defect = within_transform(y[rows], x[rows], [ca, cb])
        within[rows] = w
        df_a += int(ca.max()) + 1 + int(cb.max())
        worst = max(worst, defect)
        comp[rows] = r
    return within, df_a, worst, len(np.unique(roots))


def ols_oracle(y, x, fes, disconnected=False):
    """MP80 slopes, the constant from the means, the full unadjusted V (dof without the constant), e, rss."""
    import mpmath as mp
    n, p = x.shape
    if disconnected:
        within, df_a, defect, ncomp = within_by_component(y, x, fes)
        assert ncomp > 1
    else:
        df_a = absorbed_dof(fes)
        within, defect = within_transform(y, x, fes)
    assert defect < 5e-14, f'independent projection unresolved {defect}'
    qx, qy = within[:, 1:], within[:, 0]
    means = np.column_stack([y, x]).astype(np.longdouble).mean(axis=0)
    with mp.workdps(80):
        if p:
            inv = mp_matrix(qx.T @ qx) ** -1
            beta = inv * mp.matrix([mp.mpf(str(v)) for v in qx.T @ qy])
            b = np.array([np.longdouble(str(v)) for v in beta])
        else:
            inv, beta, b = mp.matrix(0, 0), [], np.zeros(0, np.longdouble)
        e = qy - qx @ b
        rss = mp.mpf(str(np.sum(e * e)))
        s2 = rss / (n - p - df_a)
        cons = mp.mpf(str(means[0])) - sum(mp.mpf(str(means[j + 1])) * beta[j] for j in range(p))
        V = transport_v(inv, s2, means, n, p)
        full_b = np.array([float(t) for t in beta] + [float(cons)])
    return dict(b=full_b, V=V, e=np.asarray(e, np.float64), rss=float(rss), df_a=df_a, defect=defect)


def iv_oracle(y, x, z, endog, fes, se_v=True):
    """2SLS in MP80 on the long-double within data of [y x z]; the unadjusted V with the constant's transport."""
    import mpmath as mp
    n, p = x.shape
    df_a = absorbed_dof(fes)
    within, defect = within_transform(y, np.column_stack([x, z]), fes)
    assert defect < 5e-14, f'independent projection unresolved {defect}'
    qy, qx = within[:, 0], within[:, 1:p + 1]
    qz = np.column_stack([qx[:, [j for j in range(p) if j not in endog]], within[:, p + 1:]])
    means = np.column_stack([y, x]).astype(np.longdouble).mean(axis=0)
    with mp.workdps(80):
        zz_inv = mp_matrix(qz.T @ qz) ** -1
        zx = mp_matrix(qz.T @ qx)
        zy = mp.matrix([mp.mpf(str(v)) for v in qz.T @ qy])
        inv = (zx.T * zz_inv * zx) ** -1
        beta = inv * (zx.T * zz_inv * zy)
        b = np.array([np.longdouble(str(v)) for v in beta])
        e = qy - qx @ b
        rss = mp.mpf(str(np.sum(e * e)))
        s2 = rss / (n - p - df_a)
        cons = mp.mpf(str(means[0])) - sum(mp.mpf(str(means[j + 1])) * beta[j] for j in range(p))
        V = transport_v(inv, s2, means, n, p) if se_v else None
        full_b = np.array([float(t) for t in beta] + [float(cons)])
    return dict(b=full_b, V=V, e=np.asarray(e, np.float64), rss=float(rss), df_a=df_a, defect=defect)


def oracle_for(base, v3_npz):
    """The oracle of an unscaled fixture name (scales and duplicated columns are undone on the estimates)."""
    if base in ('cov240k',):
        o = np.load(v3_npz)
        return dict(b=o['b'], V=o['V'], e=o['e'], rss=float(np.sum(np.asarray(o['e'], np.longdouble) ** 2)))
    f = fixture(base)
    if base.endswith('_4fe_iv'):
        return iv_oracle(f['y'], f['x'], f['inst'], f['endog'], f['fes'], se_v=False)
    if base.endswith('_iv'):
        return iv_oracle(f['y'], f['x'], f['inst'], f['endog'], f['fes'])
    o = ols_oracle(f['y'], f['x'], f['fes'], disconnected=base == 'disc')
    if base.endswith('_4fe'):
        o['V'] = None  # four effects: coefficients only (module docstring)
    return o


# ---------------------------------------------------------------- one fit in a fresh process
def group_mean_max(e, fes):
    out = 0.0
    for fe in fes:
        fe = np.asarray(fe, np.int64)
        s = np.zeros(int(fe.max()) + 1, np.longdouble)
        np.add.at(s, fe, np.asarray(e, np.longdouble))
        c = np.bincount(fe)
        keep = c > 0
        out = max(out, float(np.max(np.abs(s[keep] / c[keep]))))
    return out


def undo_scales(name, b, V, e):
    b, V, e = np.array(b, float), np.array(V, float), np.array(e, float)
    m = re.search(r'_2p(\d+)', name)
    if m:
        s = 2.0 ** int(m.group(1))
        b, V, e = b / s, V / (s * s), e / s
    m = re.search(r'_x1([pm])(\d+)', name)
    if m:
        s = 2.0 ** (int(m.group(2)) * (1 if m.group(1) == 'p' else -1))
        b[0] *= s
        V[0, :] *= s
        V[:, 0] *= s
    if '_dup' in name:  # the duplicated column is the last regressor; it must be omitted (0, zero row/column)
        p = len(b) - 1
        dup = dict(b=float(b[p - 1]), V=float(np.max(np.abs(V[p - 1]))))
        keep = [j for j in range(len(b)) if j != p - 1]
        b, V = b[keep], V[np.ix_(keep, keep)]
        return b, V, e, dup
    return b, V, e, None


def grade(o, b, V, e, rss, scale):
    out = {}
    ob, oV, oe = o['b'], o['V'], o['e']
    if b.shape != ob.shape:
        return dict(shape_mismatch=True)
    out['b_scaled'] = float(np.max(np.abs(b - ob) / np.maximum(1.0, np.abs(ob))))
    if e.shape == oe.shape:
        out['resid_rel'] = float(np.linalg.norm(e - oe) / np.linalg.norm(oe))
    if oV is not None:
        if V.shape != oV.shape or not np.isfinite(V).all():
            return dict(out, V_mismatch=True)
        ose = np.sqrt(np.diag(oV))
        out.update(se_rel=float(np.max(np.abs(np.sqrt(np.maximum(np.diag(V), 0.0)) / ose - 1.0))),
                   V_scaled=float(np.max(np.abs(V - oV) / np.sqrt(np.outer(np.diag(oV), np.diag(oV))))))
    if o.get('rss') is not None:
        out['rss_rel'] = float(abs(rss / (scale * scale) / o['rss'] - 1.0))
    return out


def worker(job_path):
    import importlib.util
    job = json.loads(Path(job_path).read_text())
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', job['module'])
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    opts = job['options']
    reg = core.HdfeRegressor(se_type=opts.get('se', 'unadjusted'), tol=float.fromhex(job['tol_hex']),
                             max_iter=job['max_iter'], num_threads=job['threads'], drop_singletons=True,
                             retain_fes=bool(opts.get('retain_fes')),
                             symmetric_sweep=bool(opts.get('symmetric')),
                             absorption_method=opts.get('method', 'auto'), tolerance_mode=opts.get('mode', 'strict-residual'),
                             jacobi_relaxation=0., convergence=opts.get('convergence', 'auto'))
    fits = []
    for k, name in enumerate(job['fits']):
        sys.stderr.write(f'SCA_FIT {k} {name}\n')
        sys.stderr.flush()
        f = fixture(name)
        kwargs = dict(fes=f['fes'])
        if f['clusters'] is not None and opts.get('se', 'unadjusted') == 'cluster':
            kwargs['clusters'] = f['clusters']
        if f['inst'] is not None:
            kwargs.update(instruments=f['inst'], endogenous_idx=f['endog'])
        if opts.get('weights'):
            kwargs['weights'] = np.random.default_rng(20261119).uniform(0.5, 2.0, size=len(f['y']))
        if opts.get('weighted_cpu_route'):
            kwargs['weighted_cpu_route'] = True
        y, x = f['y'], f['x']
        if opts.get('group'):  # each row twice, one group per original row (group() without individual())
            idx = np.repeat(np.arange(len(y)), 2)
            y, x = y[idx], np.asfortranarray(x[idx])
            kwargs['fes'] = [fe[idx] for fe in f['fes']]
            kwargs['group'] = idx
        if opts.get('slopes'):  # a slope on the first regressor within the first effect
            kwargs['slopes'] = [(0, np.ascontiguousarray(x[:, 0]), True)]
        rec = dict(name=name)
        t0 = time.perf_counter()
        try:
            reg.fit(y, x, **kwargs)
        except Exception as error:  # noqa: BLE001 - the refusal and today's slope error are both graded
            m = re.search(r'iterations=(\d+)/(\d+)', str(error))
            rec.update(refused=True, error_type=type(error).__name__, error=str(error)[:1200],
                       message_iterations=int(m.group(1)) if m else None,
                       message_max_iter=int(m.group(2)) if m else None,
                       estimates_left={a: int(np.asarray(getattr(reg, a)).size)
                                       for a in ('coef_', 'covariance_', 'residuals_')})
        else:
            e = np.asarray(reg.residuals_, np.float64).ravel()
            b = np.asarray(reg.coef_, np.float64).ravel()
            V = np.asarray(reg.covariance_, np.float64)
            fes_s = kwargs['fes']
            if len(e) != len(fes_s[0]):
                ix = np.asarray(reg.sample_index_, int)
                fes_s = [fe[ix] for fe in fes_s]
            rss = float(reg.rss_)
            rec.update(refused=False, iterations=int(reg.num_iterations_), converged=bool(reg.converged_),
                       certified=bool(reg.precision_certified_), gpu_used=bool(reg.gpu_used_),
                       gpu_status_code=int(reg.gpu_status_code_), cuda_tail=str(reg.cuda_tail_),
                       method=str(reg.absorption_method_used).replace('AbsorptionMethod.', ''),
                       tier_reason=str(reg.reduced_pcg_reason_), rss=rss,
                       ss_e=float(np.sum(np.asarray(e, np.longdouble) ** 2)), D=group_mean_max(e, fes_s),
                       omitted=[bool(v) for v in reg.omitted_], b=b.tolist(), b_hex=[v.hex() for v in b],
                       V=V.tolist(), V_sha=hashlib.sha256(V.tobytes()).hexdigest(),
                       e_sha=hashlib.sha256(e.tobytes()).hexdigest())
            oracle_npz = job.get('oracles', {}).get(name)
            if oracle_npz and not opts.get('group'):
                z = np.load(oracle_npz, allow_pickle=False)
                o = dict(b=z['b'], V=z['V'] if z['V'].size else None, e=z['e'],
                         rss=float(z['rss']) if z['rss'].size else None)
                bu, Vu, eu, dup = undo_scales(name, b, V, e)
                m = re.search(r'_2p(\d+)', name)
                rec['vs_oracle'] = grade(o, bu, Vu, eu, rss, 2.0 ** int(m.group(1)) if m else 1.0)
                if dup is not None:
                    rec['omitted_column'] = dup
        rec['seconds'] = time.perf_counter() - t0
        fits.append(rec)
    Path(job['out']).write_text(json.dumps(dict(fits=fits)))


# ---------------------------------------------------------------- the driver
def fields(line):
    return dict(item.split('=', 1) for item in line.split() if '=' in item)


def clean_env(extra=None, backend='cuda'):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('XHDFE_') and k not in ('LD_LIBRARY_PATH', 'OMP_NUM_THREADS')}
    env.update(XHDFE_GPU_BACKEND=backend, XHDFE_ABSORPTION_CACHE_MODE='off', XHDFE_MOBILITY_MODE='off',
               XHDFE_FE_STRUCTURE_MODE='off', XHDFE_STRICT_ACCEPT_TRACE='1', XHDFE_GI_LSMR_TRACE='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.update(extra or {})
    return env


class Runner:
    def __init__(self, workdir, threads, v3_npz):
        self.workdir, self.threads, self.v3_npz = workdir, threads, v3_npz
        self.oracles = {}
        self.count = 0

    def oracle(self, name):
        """The cached oracle npz of the unscaled base of a fixture name."""
        base = re.sub(r'_(2p\d+|x1[pm]\d+)', '', name)
        if '_dup' in base:
            base = base.replace('_dup', '')
        if base not in self.oracles:
            o = oracle_for(base, self.v3_npz)
            path = self.workdir / f'oracle_{base}.npz'
            np.savez(path, b=o['b'], V=o['V'] if o['V'] is not None else np.zeros(0), e=o['e'],
                     rss=np.array(o['rss']) if o.get('rss') is not None else np.zeros(0))
            self.oracles[base] = str(path)
        return self.oracles[base]

    def run(self, module, label, fits, options=None, env=None, max_iter=MAX_ITER, tol=TOL, graded=True,
            backend='cuda'):
        options = dict(options or {})
        if 'se' not in options:
            options['se'] = fixture_se(fits[0])
        job = dict(module=str(module), fits=fits, options=options, max_iter=max_iter, tol_hex=tol.hex(),
                   threads=self.threads, out=str(self.workdir / f'{label}.json'),
                   oracles={n: self.oracle(n) for n in fits if graded and n.split('_')[0] != 'easy'})
        (self.workdir / f'{label}.job.json').write_text(json.dumps(job))
        apps = gpu_apps() if backend == 'cuda' else None
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                                   str(self.workdir / f'{label}.job.json')], env=clean_env(env, backend),
                                  capture_output=True, text=True, timeout=3600)
            stderr, rc = proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr.decode(errors='replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')
            rc = 'timeout'
        (self.workdir / f'{label}.err').write_text(stderr)
        out = self.workdir / f'{label}.json'
        rec = json.loads(out.read_text()) if out.is_file() else {}
        blocks, current = [], None
        for line in stderr.splitlines():
            if line.startswith('SCA_FIT '):
                current = []
                blocks.append(current)
            elif current is not None and line.startswith(TRACE_PREFIXES):
                current.append(line)
        fits_out = rec.get('fits', [])
        for k, f in enumerate(fits_out):
            f['trace'] = blocks[k] if k < len(blocks) else []
        self.count += len(fits)
        return dict(label=label, module=str(module), fits=fits_out, env=env or {}, max_iter=max_iter, tol=tol,
                    options=options, rc=rc, wall=time.time() - t0, gpu_apps_before=apps,
                    stderr_tail=stderr[-2000:] if rc != 0 else '')


def fixture_se(name):
    base = name.split('_')[0]
    return 'cluster' if base == 'cov240k' and not any(s in name for s in ('_iv', '_nox')) else 'unadjusted'


def trace_parts(fit):
    lines = [l for l in fit.get('trace', []) if l.startswith('strict_accept_trace')]
    step = next((fields(l) for l in lines if ' sweep_exit ' in l), None)
    passes = [fields(l) for l in lines if re.search(r' pass=\d+ ', l) and ' D=' in l]
    cont = next((fields(l) for l in lines if 'refinement=continuation' in l), None)
    readings = [fields(l) for l in lines if l.startswith('strict_accept_trace reading ')]
    return step, passes, cont, readings


def accepted_problems(fit, max_iter, tol=TOL, limits=True, oracle_keys=('b_scaled', 'se_rel', 'V_scaled', 'rss_rel')):
    p = []
    if fit.get('refused') is not False:
        return [f"refused or no result: {str(fit.get('error'))[:300]}"]
    if not (fit['converged'] and fit['certified'] and fit['gpu_used'] and fit['gpu_status_code'] == 1):
        p.append('not converged, certified and on the GPU with status 1')
    if fit['iterations'] > max_iter:
        p.append(f"iterations {fit['iterations']} above max_iter {max_iter}")
    if not fit['D'] <= tol:
        p.append(f"D {fit['D']:.6g} above the tolerance {tol:.6g}")
    if fit['tier_reason'] != 'tolerance_mode':
        p.append(f"tier reason {fit['tier_reason']}")
    if not abs(fit['ss_e'] - fit['rss']) <= OWN_RESIDUAL_REL * max(fit['rss'], 1e-300):
        p.append(f"residuals_ is not the fit's own (sum of squares {fit['ss_e']!r} against rss {fit['rss']!r})")
    step, passes, cont, _ = trace_parts(fit)
    if not passes:
        p.append('no strict_accept_trace pass line (not in scope?)')
    else:
        last = passes[-1]
        if last.get('decision') != 'accept' or float(last['limit']) != tol:
            p.append(f"last pass line decision={last.get('decision')} limit={last.get('limit')}")
        if not (last.get('certified') == '1' and last.get('checks') == 'pass'):
            p.append('last pass line not certified with the checks passed')
    if limits:
        vs = fit.get('vs_oracle') or {}
        for k in oracle_keys:
            if k in vs and not vs[k] <= LIMITS[k]:
                p.append(f"{k} {vs[k]:.4g} above the limit {LIMITS[k]:.0e}")
        if not vs or any(v for k, v in vs.items() if k.endswith('mismatch')):
            p.append(f'no oracle comparison: {vs}')
        if 'b_scaled' not in vs:
            p.append('coefficients not graded')
    return p


def refusal_text_problems(fit, budget):
    """The refusal text of a strict CUDA fit in scope: D(e), the iterations within the budget and one of the
    code's two reasons (no budget left after the absorption, or the single refinement exhausted)."""
    p, error = [], fit.get('error', '')
    if 'the largest absolute group mean of the delivered residual is' not in error:
        p.append('the refusal text does not carry D(e)')
    if fit.get('message_iterations') is None or fit.get('message_max_iter') != budget or \
            fit['message_iterations'] > budget:
        p.append(f"the refusal text's iterations {fit.get('message_iterations')}/{fit.get('message_max_iter')} "
                 f'are not within the budget {budget}')
    if not any(r in error for r in ('no absorption iteration budget remains', "the fit's single refinement was exhausted")):
        p.append('the refusal text does not give one of the two reasons')
    return p


def refusal_problems(fit, budget=None):
    p = []
    if fit.get('refused') is not True:
        return ['not refused']
    if any(fit.get('estimates_left', {}).values()) or 'No estimates returned' not in fit.get('error', ''):
        p.append('estimates left behind or no refusal notice')
    if 'did not converge' not in fit.get('error', ''):
        p.append('no non-convergence diagnosis')
    if budget is not None and (fit.get('message_iterations'), fit.get('message_max_iter')) != (budget, budget):
        p.append(f"message iterations {fit.get('message_iterations')}/{fit.get('message_max_iter')} != {budget}/{budget}")
    return p


def groups_frame():
    """A synthetic group()/individual() fixture (as tests/stata/fused_read_contract.do: 800 patents of three
    inventor slots, 197 inventors, nine years), from a fixed seed."""
    import pandas as pd
    rng = np.random.default_rng(20261122)
    n = 2400
    patent = np.arange(n) // 3 + 1
    slot = np.arange(n) % 3 + 1
    inventor = 1 + (patent * 5 + slot * 17) % 197
    year = 2000 + patent % 9
    funding = rng.normal(size=patent.max() + 1)[patent]
    citations = funding + 0.1 * year + rng.normal(size=patent.max() + 1)[patent]
    return pd.DataFrame(dict(y=citations, x1=funding, patent=patent.astype(np.int32),
                             inventor=inventor.astype(np.int32), year=year.astype(np.int32)))


def stata_hex(text):
    """A value printed with Stata's %21x format (hexadecimal mantissa and exponent)."""
    m = re.match(r'^([+-])([0-9a-f]+\.[0-9a-f]*)X([+-])([0-9a-f]+)$', text.strip())
    if not m:
        return math.nan
    return float.fromhex(f'{m.group(1)}0x{m.group(2)}p{m.group(3)}{int(m.group(4), 16)}')


def run_stata(args, workdir, runner, short_budget):
    """The Stata leg: tests/stata/strict_cuda_acceptance_contract.do in fresh stata-mp sessions."""
    folder = workdir / 'stata'
    folder.mkdir()
    for name in ('F0', 'F1', 'F2', 'F3', 'F4', 'T3', 'W0', 'cov240k', 'cov240k_4fe', 'easy'):
        import pandas as pd
        f = fixture(name)
        frame = pd.DataFrame(dict(y=f['y']))
        for j in range(f['x'].shape[1]):
            frame[f'x{j + 1}'] = f['x'][:, j]
        for d, fe in enumerate(f['fes']):
            frame[f'fe{d + 1}'] = fe
        frame.to_stata(folder / f'{name}.dta', write_index=False, version=118)
        if name != 'easy':
            z = np.load(runner.oracle(name))
            lines = ['matrix oracle_b = (' + ', '.join(format(v, '.17g') for v in z['b']) + ')']
            if z['V'].size:
                lines.append('matrix oracle_V = (' + ' \\ '.join(', '.join(format(v, '.17g') for v in r)
                                                                for r in z['V']) + ')')
            lines.append(f"scalar oracle_rss = {float(z['rss']) if z['rss'].size else -1:.17g}")
            (folder / f'oracle_{name}.do').write_text('\n'.join(lines) + '\n')
    groups_frame().to_stata(folder / 'groups.dta', write_index=False, version=118)
    script = ROOT / 'tests' / 'stata' / 'strict_cuda_acceptance_contract.do'
    stata = shutil.which('stata-mp') or '/usr/local/stata/stata-mp'
    legs = [('scope', args.stata_package, {}),
            ('legacy_accept', args.stata_package, {'XHDFE_STRICT_CUDA_LEGACY_ACCEPTANCE': '1'}),
            ('legacy_resid', args.stata_package, {'XHDFE_STATA_LEGACY_RESIDUAL_REFINE': '1'}),
            ('both', args.stata_package, {'XHDFE_STRICT_CUDA_LEGACY_ACCEPTANCE': '1',
                                          'XHDFE_STATA_LEGACY_RESIDUAL_REFINE': '1'})]
    legs += [(f'fourfe_tight_{i}', args.stata_package, {}) for i in range(3)]  # stage 2c: three sessions
    legs += [(f'excluded_new_{i}', args.stata_package, {}) for i in range(EXCLUDED_RUNS)]
    if args.reference_stata_package:
        legs.append(('reference', args.reference_stata_package, {}))
        legs += [(f'excluded_ref_{i}', args.reference_stata_package, {}) for i in range(EXCLUDED_RUNS)]
    results = {}
    for leg, package, extra in legs:
        sub = folder / leg
        sub.mkdir()
        tmp = sub / 'statatmp'
        tmp.mkdir()
        kind = re.sub(r'_(new|ref)_\d+$', '', leg)
        kind = 'fourfe_tight' if kind.startswith('fourfe_tight') else kind
        reghdfe = str(args.reghdfe_adopath) if (args.reghdfe_adopath and kind in ('scope', 'fourfe_tight')) else ''
        (sub / 'driver.do').write_text(
            f'capture noisily do "{script}" "{Path(package).resolve()}" "{folder}" "{kind}" "{sub}" '
            f'{LARGE_EXPONENT} {LARGE_BUDGET} {short_budget} "{reghdfe}"\n'
            'local driver_rc = _rc\ndisplay "DRIVER_RC " `driver_rc\'\nexit, clear\n')
        env = clean_env(dict(extra, XHDFE_P7_REFINE_TRACE='1'))
        env.pop('XHDFE_GPU_BACKEND', None)
        env['STATATMP'] = str(tmp)
        apps = gpu_apps()
        with (sub / 'process.log').open('x') as log:
            try:
                rc = subprocess.run([stata, '-q', '-b', 'do', 'driver.do'], cwd=sub, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=7200).returncode
            except subprocess.TimeoutExpired:
                rc = 'timeout'
        text = (sub / 'driver.log').read_text(errors='replace') if (sub / 'driver.log').is_file() else ''
        trace = (sub / 'process.log').read_text(errors='replace')
        results[leg] = dict(returncode=rc, text=text, trace=trace, folder=sub,
                            statatmp_leftovers=sorted(p.name for p in tmp.iterdir()), gpu_apps_before=apps)
    return grade_stata(results, short_budget)


def sca_fields(text):
    """key=value pairs of a SCA_* line: a value may follow `=` after spaces (Stata's %21.14e pads positive
    numbers), never contains `=`, and an empty value stays empty (stage 2c addendum: `D= 9.99...` was read empty)."""
    return dict(re.findall(r'(\w+)=\s*([^\s=]*)(?=\s|$)', text))


def stata_rows(text):
    """The SCA_* lines of one leg, by fit name, with the order of the fits (SCA_OK / SCA_REFUSED)."""
    rows, order = {}, []
    for m in re.finditer(r'^SCA_(OK|REFUSED|M|X|R|K) (.*)$', text, re.M):
        kv = sca_fields(m.group(2))
        name = kv.get('name')
        row = rows.setdefault(name, {})
        row.update(kv)
        if m.group(1) in ('OK', 'REFUSED'):
            row['kind'] = m.group(1)
            order.append((name, m.group(1)))
    return rows, order


def stata_blocks(trace, order):
    """Trace lines per fit. Every fit in scope prints one `strict_accept_trace sweep_exit` line right after its
    absorption (before its p7_refine_trace line, which the plugin prints at each pass, and before its pass lines),
    so a fit's block runs from its sweep_exit line to the next one. A leg whose fits are all out of scope has none:
    its single fit (legacy_resid, both) takes the whole trace. A count that does not match the fits is reported."""
    lines = [l for l in trace.splitlines() if l.startswith(TRACE_PREFIXES)]
    starts = [i for i, l in enumerate(lines) if l.startswith('strict_accept_trace sweep_exit')]
    if not starts:
        return {order[0][0]: lines} if len(order) == 1 else {}
    if len(starts) != len(order):
        return {'__misaligned__': [f'{len(starts)} sweep_exit lines for {len(order)} fits']}
    blocks = [lines[a:b] for a, b in zip(starts, starts[1:] + [len(lines)])]
    return {name: block for (name, _), block in zip(order, blocks)}


def stata_errors(text):
    """The refusal text printed before each SCA_REFUSED line, as a fit record (Stata wraps lines at the line size
    and starts each continuation with '> ')."""
    lines = text.replace('\n> ', '').splitlines()
    out = {}
    for i, line in enumerate(lines):
        m = re.match(r'^SCA_REFUSED name=(\S+)', line)
        if m:
            error = next((lines[j] for j in range(i - 1, max(i - 8, -1), -1) if 'did not converge' in lines[j]), '')
            it = re.search(r'iterations=(\d+)/(\d+)', error)
            out[m.group(1)] = dict(error=error, message_iterations=int(it.group(1)) if it else None,
                                   message_max_iter=int(it.group(2)) if it else None)
    return out


def estimates(folder, name):
    p = folder / f'est_{name}.txt'
    return p.read_text() if p.is_file() else None


def grade_stata(results, short_budget):
    problems, labels, summary, notes = [], [], {}, []
    for leg, r in results.items():
        if r['returncode'] != 0 or not re.search(r'^DRIVER_RC 0\s*$', r['text'], re.M):
            problems.append(f"{leg}: Stata run failed (returncode {r['returncode']})")
        if r['statatmp_leftovers']:
            problems.append(f"{leg}: STATATMP not empty: {r['statatmp_leftovers'][:5]}")
        if not re.search(r'^SCA_PASS\s*$', r['text'], re.M):
            problems.append(f"{leg}: no SCA_PASS line ({re.findall(r'^SCA_FAIL (.*)$', r['text'], re.M)[:8]})")
    rows = {leg: stata_rows(r['text']) for leg, r in results.items()}
    blocks = {leg: stata_blocks(r['trace'], rows[leg][1]) for leg, r in results.items()}
    summary['rows'] = {leg: v[0] for leg, v in rows.items()}
    if 'scope' in results:
        b = blocks['scope']
        if '__misaligned__' in b:
            problems.append(f"stata scope: trace misaligned ({b['__misaligned__']})")
        for name in ('F0', 'F1', 'F2', 'F3', 'F4', 'T3', 'W0', 'cov240k', 'cont_2p17') + \
                (('easy_after_short',) if short_budget else ()) + ('W0_after_large', 'cov240k_4fe'):
            lines = b.get(name, [])
            p7 = [fields(l) for l in lines if l.startswith('p7_refine_trace')]
            if not p7 or p7[-1].get('refined') != '0' or p7[-1].get('fit_state') != '1':
                problems.append(f'stata scope {name}: p7_refine_trace {p7[-1:] or "missing"} (expected refined=0 '
                                'fit_state=1)')
            passes = [fields(l) for l in lines if ' pass=' in l and ' D=' in l]
            if not passes or passes[-1].get('decision') != 'accept' or float(passes[-1].get('limit', 'nan')) != TOL:
                problems.append(f'stata scope {name}: no accepting pass line at the tolerance ({passes[-1:]})')
        if not any('refinement=continuation' in l for l in b.get('cont_2p17', [])):
            problems.append('stata scope cont_2p17: no continuation in the trace')
        # Package C (18nov2026): each of the three short fits is either the expected refusal before the
        # continuation (no budget left after the primary) or a refusal in the continuation, at exactly the budget,
        # with the exact reason; at least one must reach the continuation. A missing calibration skipped them.
        if short_budget:
            short_errors = stata_errors(results['scope']['text'])
            exercised = 0
            for k in (1, 2, 3):
                cont_k = [fields(l) for l in b.get(f'cont_short_{k}', []) if 'refinement=continuation' in l]
                err = short_errors.get(f'cont_short_{k}', {})
                text_k = err.get('error', '')
                if cont_k:
                    exercised += 1
                    if cont_k[0].get('exit') != 'budget':
                        problems.append(f"stata cont_short_{k}: the continuation ended by {cont_k[0].get('exit')}, "
                                        'not its budget')
                    if "the fit's single refinement was exhausted" not in text_k:
                        problems.append(f"stata cont_short_{k}: the refusal text does not say the fit's single "
                                        'refinement was exhausted')
                    if (err.get('message_iterations'), err.get('message_max_iter')) != (short_budget, short_budget):
                        problems.append(f"stata cont_short_{k}: refused at {err.get('message_iterations')}/"
                                        f"{err.get('message_max_iter')}, not at the budget {short_budget}")
                    if 'the largest absolute group mean of the delivered residual is' not in text_k:
                        problems.append(f'stata cont_short_{k}: the refusal text does not carry D(e)')
                elif 'no absorption iteration budget remains' not in text_k:
                    problems.append(f'stata cont_short_{k}: neither the refusal before the continuation nor one in '
                                    f'it: {text_k[:200]}')
            if not exercised:
                labels.append('NOT_EXERCISED: stata cont_short refused before the continuation in its three fits')
        large = [fields(l) for l in b.get('large_default', []) if ' pass=1 ' in l and ' D=' in l]
        if not large or not float(large[0]['D']) > LARGE_FLOOR_FACTOR * TOL or large[0].get('certified') != '1':
            problems.append(f'stata large_default: pass 1 not certified with a floor above ten tolerances: {large[:1]}')
        # The large row's refusal text in Stata, tied to its trace (stage 2c): D(e), exactly the budget, a reason.
        errors = stata_errors(results['scope']['text'])
        f = errors.get('large_default', {})
        problems += [f'stata large_default: {q}' for q in refusal_text_problems(f, LARGE_BUDGET)]
        if f.get('message_iterations') != LARGE_BUDGET:
            problems.append(f"stata large_default: refused at {f.get('message_iterations')}, not at the budget")
        r4 = rows['scope'][0].get('cov240k_4fe', {})
        if 'reghdfe_db' not in r4:
            labels.append(f"NOT_EXERCISED: stata four-effect fit against reghdfe ({r4.get('reghdfe', r4.get('reghdfe_rc'))})")
        else:
            # Round 10: that fit's standard errors are graded only through its SCA_K line, so whenever reghdfe
            # ran for it the line must be there, exactly once.
            k4 = [m for m in re.findall(r'^SCA_K (.*)$', results['scope']['text'], re.M)
                  if sca_fields(m).get('name') == 'cov240k_4fe']
            if len(k4) != 1:
                problems.append(f'stata scope cov240k_4fe: {len(k4)} SCA_K lines with reghdfe present (exactly one '
                                'expected; its standard errors are graded only through that line)')
    # The four-effect covariance at tolerance(1e-8) (SCA_K): a known defect inside its declared band (1e-7, 2e-7]
    # of the standard-error distance to reghdfe, a failure above it, nothing at or below 1e-7 (stage 2c addendum).
    known_defects = []
    summary['known'] = {leg: [sca_fields(m) for m in re.findall(r'^SCA_K (.*)$', r['text'], re.M)]
                        for leg, r in results.items() if re.search(r'^SCA_K ', r['text'], re.M)}
    for leg, lines in summary['known'].items():
        for k in lines:
            dse, dv = float(k.get('reghdfe_dse') or 'nan'), k.get('reghdfe_dv')
            r4 = rows[leg][0].get(k.get('name'), {})
            numbers = (f"SE {dse:.3g} and covariance {dv} against reghdfe at 1e-12, b {r4.get('reghdfe_db')} against "
                       f"it and {r4.get('db')} against the oracle, D {r4.get('D')}")
            if not dse <= 2e-7:
                problems.append(f"stata {leg} {k.get('name')}: {numbers}: above the declared band (1e-7, 2e-7]")
            elif dse > 1e-7:
                known_defects.append(f"KNOWN_DEFECT_FOUR_EFFECT_COVARIANCE_CUDA ({leg} {k.get('name')}): {numbers}; a "
                                     'demonstrated estimator error (an MP80 oracle sides with reghdfe), earlier than '
                                     'package B, present in the reference binaries, inside the declared band (1e-7, 2e-7]')
    # The four-effect fit at tolerance(1e-10), three sessions (pre-registered): each accepted at that tolerance in
    # scope, with the stored residuals' own pass line, and its slopes against reghdfe (the do-file asserts the
    # limits); a session without reghdfe is missing coverage.
    tight = sorted(leg for leg in results if leg.startswith('fourfe_tight'))
    summary['fourfe_tight'] = {}
    for leg in tight:
        r = rows[leg][0].get('cov240k_4fe_1e10', {})
        lines = blocks[leg].get('cov240k_4fe_1e10', [])
        passes = [fields(l) for l in lines if ' pass=' in l and ' D=' in l]
        summary['fourfe_tight'][leg] = dict(it=r.get('it'), D=r.get('D'), db=r.get('db'), reghdfe_db=r.get('reghdfe_db'),
                                            reghdfe_dse=r.get('reghdfe_dse'), reghdfe_dv=r.get('reghdfe_dv'))
        if r.get('kind') != 'OK' or not passes or passes[-1].get('decision') != 'accept' or \
                float(passes[-1].get('limit', 'nan')) != 1e-10:
            problems.append(f'stata {leg}: the four-effect fit at tolerance(1e-10) not accepted at that limit ({passes[-1:]})')
        if 'reghdfe_dse' not in r:
            labels.append(f"NOT_EXERCISED: stata {leg} against reghdfe ({r.get('reghdfe', r.get('reghdfe_rc'))})")
    for leg in ('legacy_resid', 'both'):
        if leg in results:
            p7 = [fields(l) for l in blocks[leg].get('F0', []) if l.startswith('p7_refine_trace')]
            if not p7 or p7[-1].get('refined') != '1':
                problems.append(f'stata {leg} F0: the stored vector was not absorbed again ({p7[-1:]})')
    if 'scope' in results and 'legacy_resid' in results:
        a, c = estimates(results['scope']['folder'], 'F0'), estimates(results['legacy_resid']['folder'], 'F0')
        if a is None or a != c:
            problems.append('stata legacy_resid F0: e(b), e(V) or e(rss) differ from the scope leg (the corrected fit)')
    if 'reference' in results:
        for leg, name in (('legacy_accept', 'F4'), ('both', 'F0')):
            if leg in results:
                a, c = estimates(results[leg]['folder'], name), estimates(results['reference']['folder'], name)
                if a is None or a != c:
                    problems.append(f'stata {leg} {name}: e(b), e(V) or e(rss) differ from the reference package')
        summary['excluded'] = {}
        legs = {arm: [f'excluded_{arm}_{i}' for i in range(EXCLUDED_RUNS)] for arm in ('new', 'ref')}
        for name in ('group', 'group_individual', 'cpu_strict', 'criterion_reghdfe'):
            # Round 10: every session of both arms has the outcome the fit expects before any distance; a
            # session that does not is a problem of the row, and nothing is computed for it.
            pre = [q for arm in ('new', 'ref') for q in stata_outcome_problems(name, arm, legs[arm], results, rows)]
            if pre:
                summary['excluded'][name] = dict(aa=None, ab=None, ab_median=None, agree=False, bit_identical=False)
                problems += [f'stata excluded {name}: {q}' for q in pre]
                continue
            d = stata_aa_ab(name, [results[leg]['folder'] for leg in legs['new']],
                            [results[leg]['folder'] for leg in legs['ref']])
            summary['excluded'][name] = d
            if name == 'cpu_strict':
                if not d['bit_identical']:
                    problems.append('stata excluded cpu_strict: not bit for bit against the reference package')
            elif not d['agree']:
                problems.append(f"stata excluded {name}: A/B (median {d['ab_median']}, max {d['ab']}) against A/A "
                                f"(max {d['aa']})")
            else:
                note = within_dispersion_note(f'stata {name}', d)
                if note:
                    notes.append(note)
    else:
        labels.append('NOT_EXERCISED: Stata comparisons with the reference package (no --reference-stata-package)')
    return dict(problems=problems, labels=labels, known_defects=known_defects, summary=summary, notes=notes,
                known=summary.get('known', {}),
                legs={k: dict(returncode=v['returncode'], statatmp_leftovers=v['statatmp_leftovers'],
                              lines=re.findall(r'^SCA_.*$', v['text'], re.M)) for k, v in results.items()})


def stata_load(folder, name):
    """One Stata fit's kept output: the estimates text, e(b), e(V) and the stored residual variable (None when a
    file is missing; a value Stata printed as missing reads as NaN)."""
    import pandas as pd

    est = estimates(folder, name)
    p = folder / f'resid_{name}.dta'
    if est is None or not p.is_file():
        return None
    vals = [stata_hex(line.split()[1]) for line in est.splitlines()]
    kinds = [line.split()[0] for line in est.splitlines()]
    b = np.array([v for v, k in zip(vals, kinds) if k == 'b'])
    V = np.array([v for v, k in zip(vals, kinds) if k == 'V'])
    r = pd.read_stata(p)['r'].to_numpy(dtype=float)
    return est, b, V, r


def stata_outcome_problems(name, arm, legs, results, rows):
    """Round 10: the outcome each session of one arm must have for an excluded Stata fit before any distance: its
    SCA_OK line (no excluded Stata fit expects a refusal), converged and certified with the backend's flags
    (cpu_strict on the CPU, the others on CUDA with the GPU used), and its estimates and stored residuals loaded,
    with finite e(b) and e(V). Each failure names its arm and index."""
    want = dict(conv='1', cert='1', gpu='0', be='cpu') if name == 'cpu_strict' else \
        dict(conv='1', cert='1', gpu='1', be='cuda', st='used')
    p = []
    for i, leg in enumerate(legs):
        where = f'{arm} {i}'
        if leg not in results:
            p.append(f'{where}: no session')
            continue
        r = rows[leg][0].get(name, {})
        if r.get('kind') != 'OK':
            p.append(f"{where}: {'SCA_' + r['kind'] if r.get('kind') else 'no SCA_OK line'}")
            continue
        bad = {k: r.get(k) for k, v in want.items() if r.get(k) != v}
        if bad:
            p.append(f'{where}: {bad}')
        u = stata_load(results[leg]['folder'], name)
        if u is None:
            p.append(f'{where}: estimates or stored residuals not loaded')
        elif not (u[1].size and u[2].size and np.all(np.isfinite(u[1])) and np.all(np.isfinite(u[2]))
                  and np.any(~np.isnan(u[3]))):
            p.append(f'{where}: e(b) or e(V) not finite, or no stored residual')
    return p


def stata_aa_ab(name, new, ref):
    """e(b), e(V) and the stored residuals of one excluded fit across sessions whose outcome
    stata_outcome_problems accepted: bit identity of all (CPU), or the largest scaled distance within each arm
    (A/A) and across (A/B); a distance with a part that is not finite is infinite."""
    def dist(u, w):
        if u is None or w is None or u[1].shape != w[1].shape or u[2].shape != w[2].shape or \
                u[3].shape != w[3].shape:
            return math.inf
        mask = ~np.isnan(u[3]) & ~np.isnan(w[3])
        if not mask.any():
            return math.inf
        scale = np.sqrt(np.abs(u[2])).max() if u[2].size else 1.0
        parts = [float(np.max(np.abs(u[1] - w[1]) / np.maximum(1.0, np.abs(u[1])))),
                 float(np.max(np.abs(u[2] - w[2])) / max(scale * scale, 1e-300)),
                 float(np.max(np.abs(u[3][mask] - w[3][mask])) / max(float(np.max(np.abs(u[3][mask]))), 1e-300))]
        return max(parts) if all(math.isfinite(v) for v in parts) else math.inf
    n, o = [stata_load(f, name) for f in new], [stata_load(f, name) for f in ref]
    same = bool(n) and bool(o) and all(
        x is not None and y is not None and x[0] == y[0] and np.array_equal(x[3], y[3], equal_nan=True)
        for x in n for y in o)
    aa = [dist(x[i], x[j]) for x in (n, o) for i in range(len(x)) for j in range(i + 1, len(x))]
    ab = [dist(u, w) for u in n for w in o]
    return dict(aa_ab_summary(aa, ab), bit_identical=same)


def cuda_aa_ab(rows_new, rows_ref):
    """The largest scaled distance of b and V between processes whose outcome excluded_outcome_problems accepted:
    A/A within each arm, A/B across; a distance with a part that is not finite is infinite."""
    def dist(a, b):
        if a.get('refused') or b.get('refused'):
            return 0.0 if (a.get('refused') and b.get('refused') and a.get('error') == b.get('error')) else math.inf
        ba, bb = np.array(a['b'], float), np.array(b['b'], float)
        Va, Vb = np.array(a['V'], float), np.array(b['V'], float)
        if ba.shape != bb.shape or Va.shape != Vb.shape:
            return math.inf
        d = np.sqrt(np.outer(np.abs(np.diag(Va)), np.abs(np.diag(Va))))
        d[d == 0] = 1.0
        parts = [float(np.max(np.abs(ba - bb) / np.maximum(1.0, np.abs(ba)))), float(np.max(np.abs(Va - Vb) / d))]
        return max(parts) if all(math.isfinite(v) for v in parts) else math.inf
    aa = [dist(x[i], x[j]) for x in (rows_new, rows_ref) for i in range(len(x)) for j in range(i + 1, len(x))]
    ab = [dist(a, b) for a in rows_new for b in rows_ref]
    return aa_ab_summary(aa, ab)


def aa_ab_summary(aa, ab):
    """A/B against A/A. The largest cross distance is not the statistic: most pairs are cross pairs (9 of 15 with
    three runs per arm, 25 of 45 with five), so it can exceed the largest within-arm distance when the arms do not
    differ. The rule compares the median cross distance with the largest within-arm distance (five runs per arm
    since stage 2c); runs identical within both arms need identity; a distance that is not finite never agrees
    (round 10: with refusals in one arm both sides were infinite and inf <= inf agreed)."""
    if not aa or not ab:
        return dict(aa=None, ab=None, ab_median=None, agree=False)
    top, median = float(np.max(aa)), float(np.median(ab))
    finite = all(math.isfinite(v) for v in list(aa) + list(ab))
    agree = finite and ((float(np.max(ab)) == 0.0) if top == 0.0 else bool(median <= top))
    return dict(aa=top, aa_min=float(np.min(aa)), ab=float(np.max(ab)), ab_median=median, agree=agree,
                finite=finite)


def excluded_outcome_problems(fits, arm, refusal_expected, gpu=True):
    """Round 10: the outcome every process of an excluded comparison must have before any distance: the error
    before estimation where the case expects it (slopes); otherwise accepted, converged, certified, finite b and V
    and, on CUDA, the GPU used with status 1. Each failure names its arm and index."""
    p = []
    for i, f in enumerate(fits):
        where = f'{arm} {i}'
        if not f:
            p.append(f'{where}: no result')
            continue
        if refusal_expected:
            if not f.get('refused'):
                p.append(f'{where}: not refused (the case expects the error before estimation)')
            continue
        if f.get('refused'):
            p.append(f"{where}: refused ({str(f.get('error'))[:160]})")
            continue
        if not (f.get('converged') is True and f.get('certified') is True):
            p.append(f"{where}: converged={f.get('converged')} certified={f.get('certified')}")
        if gpu and not (f.get('gpu_used') is True and f.get('gpu_status_code') == 1):
            p.append(f"{where}: gpu_used={f.get('gpu_used')} gpu_status_code={f.get('gpu_status_code')}")
        b = np.asarray(f.get('b') if f.get('b') is not None else [], float)
        V = np.asarray(f.get('V') if f.get('V') is not None else [], float)
        if not (b.size and V.size and np.all(np.isfinite(b)) and np.all(np.isfinite(V))):
            p.append(f'{where}: b or V missing or not finite')
    return p


def within_dispersion_note(label, d):
    """Round 10: an agreeing excluded CUDA row whose largest within-arm distance is above the coefficient limit
    shows only that the median A/B lies within the run-to-run dispersion (a note; the exit code does not change)."""
    if d.get('agree') and d.get('aa') is not None and d['aa'] > LIMITS['b_scaled']:
        return (f"WITHIN_DISPERSION_ONLY: {label}: the code excludes this fit from the route; the median A/B "
                f"({d['ab_median']:.4g}) lies within the run-to-run dispersion of the reference and new binaries "
                f"({d['aa']:.4g}), which is above the limit 1e-8; numerical equivalence not demonstrated")
    return None


def grade_excluded_cuda(case, fn, fr):
    """One excluded CUDA case against the reference module (fn, fr: one fit record or None per process): the
    outcome of every process first, then today's error (slopes) or the A/A against A/B rule. Returns the summary,
    the problems and the notes; no distance is computed for a row whose precondition fails."""
    slopes = case.startswith('slopes')
    p = excluded_outcome_problems(fn, 'new', slopes) + excluded_outcome_problems(fr, 'ref', slopes)
    if len(fn) != EXCLUDED_RUNS or len(fr) != EXCLUDED_RUNS:
        p.append(f'{len(fn)} new and {len(fr)} reference processes (expected {EXCLUDED_RUNS} each)')
    if p:
        return dict(aa=None, ab=None, ab_median=None, agree=False), p, []
    d = cuda_aa_ab(fn, fr)
    if slopes:
        if len({f.get('error') for f in fn + fr}) != 1:
            p.append(f"slopes: not today's error before estimation: {[f.get('error') for f in fn + fr][:2]}")
        return d, p, []
    if not d['agree']:
        p.append(f'A/B (median {d["ab_median"]}, max {d["ab"]}) against A/A (max {d["aa"]})')
        return d, p, []
    note = within_dispersion_note(case, d)
    return d, p, [note] if note else []


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--module', type=Path, required=True)
    ap.add_argument('--v3-npz', type=Path, default=V3_NPZ)
    ap.add_argument('--reference-module', type=Path)
    ap.add_argument('--cpu-module', type=Path)
    ap.add_argument('--reference-cpu-module', type=Path)
    ap.add_argument('--stata-package', type=Path)
    ap.add_argument('--reference-stata-package', type=Path)
    ap.add_argument('--reghdfe-adopath', type=Path)
    ap.add_argument('--only', default='')
    ap.add_argument('--fits', type=int, default=2)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--keep', type=Path)
    args = ap.parse_args()
    if not cuda_device_visible():
        print(json.dumps(dict(status='SKIP', reason='no visible CUDA device')))
        return 0
    if hashlib.sha256(args.v3_npz.read_bytes()).hexdigest() != V3_NPZ_SHA256:
        raise SystemExit('v3 oracle npz sha256 mismatch')
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=False)
        workdir, cleanup = args.keep, None
    else:
        cleanup = tempfile.TemporaryDirectory()
        workdir = Path(cleanup.name)
    only = set(c for c in args.only.split(',') if c)
    want = (lambda case: not only or case in only)
    runner = Runner(workdir, args.threads, args.v3_npz)
    module = args.module.resolve()
    failures, labels, rows, notes = [], [], [], []
    started, apps = time.time(), gpu_apps()

    def record(case, row, problems):
        row.update(case=case, problems=problems)
        rows.append(row)
        if problems:
            failures.append(dict(label=row['label'], problems=problems))
        fit = row['fits'][0] if row['fits'] else {}
        print(json.dumps(dict(label=row['label'], problems=problems, refused=[f.get('refused') for f in row['fits']],
                              it=[f.get('iterations', f.get('message_iterations')) for f in row['fits']],
                              D=fit.get('D'), vs=fit.get('vs_oracle'))), flush=True)

    if want('scope'):
        for name in ('F0', 'F1', 'F2', 'F3', 'F4', 'T3', 'W0', 'cov240k'):
            for i in range(args.fits):
                row = runner.run(module, f'scope_{name}_{i}', [name])
                record('scope', row, accepted_problems(row['fits'][0], MAX_ITER) if row['fits'] else ['no result'])
    # The continuation's primary (pass-1 iterations of cov240k x 2^17), the base of the short budget: pass 1 +
    # SHORT_MARGIN. Package C (18nov2026), a change made after a result: with pass 1 + 2 the row read
    # NOT_EXERCISED in one of three full runs on package B's final binaries (b_audit/final_checks/contract.json,
    # budget 204), because the primary takes 201-215 iterations depending on the process (15 kept fits) and a
    # calibration that draws low leaves no budget for the continuation. The shortest confirming continuation on
    # this fixture in the kept legs is 124 sweeps (Stata, b/stage2b/contract_final/stata/scope/process.log:75;
    # the Python legs confirmed in 563-2528, the 3015 is the instrument variant), so a margin of 24 leaves the
    # continuation 10 to 38 sweeps over that spread, well below 124. Pass 1's D varies per process (57-375
    # tolerances).
    short = dict(primary=None)

    def calibrate_short():
        if short['primary'] is None:
            row = runner.run(module, 'continuation_calibration', [f'cov240k_2p{Y_SCALE_EXPONENT}'])
            _, passes, _, _ = trace_parts(row['fits'][0] if row['fits'] else {})
            short['primary'] = int(passes[0]['iterations']) if passes else None
        return None if short['primary'] is None else short['primary'] + SHORT_MARGIN
    if want('continuation'):
        for name in (f'cov240k_2p{Y_SCALE_EXPONENT}', f'cov240k_2p{Y_SCALE_EXPONENT}_iv'):
            for i in range(args.fits):
                row = runner.run(module, f'continuation_{name}_{i}', [name])
                fit = row['fits'][0] if row['fits'] else {}
                p = accepted_problems(fit, MAX_ITER) if fit else ['no result']
                _, passes, cont, _ = trace_parts(fit)
                if not (len(passes) == 2 and passes[0].get('decision') == 'continue'
                        and passes[0].get('certified') == '1' and float(passes[0]['D']) > TOL and cont):
                    p.append(f'expected pass 1 (certified, D > tol, continue), the continuation, pass 2: '
                             f'{[(q.get("D"), q.get("decision")) for q in passes]} continuation={bool(cont)}')
                elif short['primary'] is None and '_iv' not in name:
                    short['primary'] = int(passes[0]['iterations'])
                record('continuation', row, p)
    if want('continuation_short'):
        name = f'cov240k_2p{Y_SCALE_EXPONENT}'
        budget = calibrate_short()
        if budget is None:
            # Package C: no calibration is missing coverage, never the fixed budget (with 400 the Stata leg confirmed
            # after 325 iterations, b/stage2b/contract_final/stata/scope/driver.log:591-592).
            labels.append('NOT_EXERCISED: continuation_short without a calibrated pass 1')
        else:
            def easy_ok(f):
                return (not f.get('refused') and f.get('converged') and f.get('certified') and f.get('gpu_used') and
                        f.get('iterations', budget + 1) <= budget)

            for attempt in range(3):
                row = runner.run(module, f'continuation_short_{attempt}', ['easy', name, 'easy'],
                                 options=dict(se='unadjusted'), max_iter=budget, graded=False)
                fits = row['fits']
                # Package C (c/RESPOSTA_2_C.md 2.6): an attempt is run again only when the whole sequence is as
                # expected (both easy fits accepted on the GPU within the budget, the middle fit refused with no
                # estimates left) and the refusal is the expected one before the continuation (the primary used
                # the budget up, no continuation in the trace); any other outcome is the attempt that counts.
                if not (len(fits) == 3 and easy_ok(fits[0]) and easy_ok(fits[2]) and fits[1].get('refused') and
                        not any(fits[1].get('estimates_left', {}).values()) and trace_parts(fits[1])[2] is None and
                        'no absorption iteration budget remains' in fits[1].get('error', '')):
                    break
            p = []
            row['budget'] = budget
            if [f.get('refused') for f in fits] != [False, True, False]:
                p.append(f"sequence {[f.get('refused') for f in fits]} (budget {budget})")
            else:
                _, passes, cont, _ = trace_parts(fits[1])
                expected_before = 'no absorption iteration budget remains' in fits[1].get('error', '')
                if not cont and not expected_before:
                    # A refusal before the continuation with any other reason is a failure, not missing coverage.
                    p.append(f"refused before the continuation with another reason: {fits[1].get('error', '')[:200]}")
                elif not cont:
                    if any(fits[1].get('estimates_left', {}).values()):
                        p.append('estimates left behind by the refusal')
                    labels.append('NOT_EXERCISED: continuation_short refused before the continuation in three '
                                  f'attempts (primary above the budget {budget})')
                else:
                    p += refusal_problems(fits[1], budget)
                    if cont.get('exit') != 'budget':
                        p.append(f"the continuation ended by {cont.get('exit')}, not by its budget")
                    # Package C: the exact reason of a refusal in the continuation.
                    if "the fit's single refinement was exhausted" not in fits[1].get('error', ''):
                        p.append("the refusal text does not say the fit's single refinement was exhausted")
                if 'the largest absolute group mean of the delivered residual is' not in fits[1].get('error', ''):
                    p.append('the refusal text does not carry D(e)')
                b0, b2 = np.array(fits[0]['b']), np.array(fits[2]['b'])
                if not np.max(np.abs(b0 - b2) / np.maximum(1.0, np.abs(b0))) <= LIMITS['b_scaled']:
                    p.append('the third fit differs from the first beyond the coefficient limit')
                for f in (fits[0], fits[2]):
                    if not (f['converged'] and f['certified'] and f['gpu_used'] and f['iterations'] <= budget):
                        p.append(f"{f['name']} not converged, certified and on the GPU within the budget")
            record('continuation_short', row, p)
    if want('instruments'):
        for i in range(args.fits):
            row = runner.run(module, f'instruments_{i}', ['cov240k_iv'])
            record('instruments', row, accepted_problems(row['fits'][0], MAX_ITER) if row['fits'] else ['no result'])
        row = runner.run(module, 'instruments_legacy', ['cov240k_iv'],
                         env={'XHDFE_STRICT_CUDA_LEGACY_ACCEPTANCE': '1'})
        record('instruments_legacy', row, [])  # reported, not graded (brief: do not presume it shows the defect)
    if want('legacy'):
        new = [runner.run(module, f'legacy_F4_{i}', ['F4'], env={'XHDFE_STRICT_CUDA_LEGACY_ACCEPTANCE': '1'})
               for i in range(2)]
        p = []
        for row in new:
            fit = row['fits'][0] if row['fits'] else {}
            vs = fit.get('vs_oracle') or {}
            if fit.get('refused') is not False or not (vs.get('b_scaled', 0) > LIMITS['b_scaled']
                                                        or vs.get('se_rel', 0) > LIMITS['se_rel']):
                p.append(f'the legacy arm does not show the defect: {vs}')
            if any(l.startswith('strict_accept_trace') for l in fit.get('trace', [])):
                p.append('the legacy arm printed the new acceptance trace')
        if args.reference_module:
            ref = [runner.run(args.reference_module.resolve(), f'legacy_F4_reference_{i}', ['F4']) for i in range(2)]
            fn = [r['fits'][0] if r['fits'] else None for r in new]
            fr = [r['fits'][0] if r['fits'] else None for r in ref]
            # Round 10: the same precondition as the excluded rows before the distance (both arms accepted,
            # converged, certified, on the GPU with status 1, finite b and V).
            pre = excluded_outcome_problems(fn, 'legacy', False) + excluded_outcome_problems(fr, 'reference', False)
            if pre:
                d = dict(aa=None, ab=None, ab_median=None, agree=False)
                p += [f'the legacy comparison: {q}' for q in pre]
            else:
                d = cuda_aa_ab(fn, fr)
                if not d['agree']:
                    p.append(f'the legacy arm differs from the reference module beyond A/A: {d}')
            new[0]['aa_ab'] = d
            for r in ref:
                record('legacy_reference', r, [])
        else:
            labels.append('NOT_EXERCISED: legacy arm against the reference module (no --reference-module)')
        record('legacy', new[0], p)
        record('legacy', new[1], [])
    if want('shapes'):
        for name, label in (('cov240k_nox', 'no_regressors'), ('cov240k_dup', 'omitted'), ('disc', 'disconnected')):
            row = runner.run(module, f'shapes_{label}', [name])
            fit = row['fits'][0] if row['fits'] else {}
            p = accepted_problems(fit, MAX_ITER) if fit else ['no result']
            if label == 'omitted' and fit and not fit.get('refused'):
                # omitted_ lists the regressors and then the constant: the duplicate is the last regressor.
                if fit.get('omitted', [])[-2:-1] != [True] or (fit.get('omitted_column') or {}).get('b') != 0.0:
                    p.append(f"the duplicated column is not omitted: {fit.get('omitted')} {fit.get('omitted_column')}")
            record('shapes', row, p)
    if want('rescaled'):
        for name in ('cov240k_x1p10', 'cov240k_x1m10'):
            row = runner.run(module, f'rescaled_{name}', [name])
            record('rescaled', row, accepted_problems(row['fits'][0], MAX_ITER) if row['fits'] else ['no result'])
    if want('tolerances'):
        for tname in ('1e-8+1ulp', '2e-8'):
            tol = TOLERANCES[tname]
            row = runner.run(module, f"tolerance_{tname.replace('+', 'p')}", ['cov240k'], tol=tol)
            record('tolerances', row, accepted_problems(row['fits'][0], MAX_ITER, tol=tol) if row['fits'] else ['no result'])
    if want('boundary'):
        calib = runner.run(module, 'boundary_calibration', ['cov240k'])
        fit = calib['fits'][0] if calib['fits'] else {}
        record('boundary', calib, accepted_problems(fit, MAX_ITER) if fit else ['no result'])
        if fit and fit.get('refused') is False:
            n = fit['iterations']
            for m in (n - 1, n, n + 1):
                row = runner.run(module, f'boundary_{m}', ['cov240k'], max_iter=m)
                f = row['fits'][0] if row['fits'] else {}
                p = (accepted_problems(f, m) if f.get('refused') is False else refusal_problems(f, m)) if f \
                    else ['no result']
                row['outcome'] = 'accepted' if f.get('refused') is False else 'refused'
                record('boundary', row, p)
    if want('session'):
        row = runner.run(module, 'session', ['easy', 'cov240k', 'easy'], options=dict(se='unadjusted'), max_iter=30,
                         graded=False)
        fits = row['fits']
        p = []
        if [f.get('refused') for f in fits] != [False, True, False]:
            p.append(f"sequence {[f.get('refused') for f in fits]}")
        else:
            p += refusal_problems(fits[1], 30)
            b0, b2 = np.array(fits[0]['b']), np.array(fits[2]['b'])
            if not np.max(np.abs(b0 - b2) / np.maximum(1.0, np.abs(b0))) <= LIMITS['b_scaled']:
                p.append('the third fit differs from the first beyond the coefficient limit')
        record('session', row, p)
    if want('step'):
        row = runner.run(module, 'step_confirmed', ['T3_nox'])
        fit = row['fits'][0] if row['fits'] else {}
        p = accepted_problems(fit, MAX_ITER) if fit else ['no result']
        step, passes, cont, _ = trace_parts(fit)
        if not (step and step.get('reached') == '1' and step.get('exit') == 'confirmed' and int(step['sweeps']) > 0
                and len(passes) == 1 and cont is None):
            p.append(f'expected the step confirmed after sweeps and one pass: {step} passes={len(passes)}')
        record('step', row, p)
        for name, reason in (('cov240k_4fe_iv', 'instruments'), ('cov240k_4fe_dup', 'rank')):
            row = runner.run(module, f'step_skip_{reason}', [name])
            fit = row['fits'][0] if row['fits'] else {}
            p = accepted_problems(fit, MAX_ITER, oracle_keys=('b_scaled',)) if fit else ['no result']
            step, _, _, _ = trace_parts(fit)
            if not (step and step.get('reached') == '1' and step.get('skipped') == reason):
                p.append(f'expected the step skipped for {reason}: {step}')
            record('step', row, p)
        for label, name, opts, env in (('step_symmetric', 'cov240k_4fe', dict(symmetric=True),
                                        {'XHDFE_STRICT_ACCEPT_TRACE': '2'}),
                                       ('step_frozen_off', 'T3_nox', {},
                                        {'XHDFE_STRICT_ACCEPT_TRACE': '2', 'XHDFE_STRICT_FROZEN_STOP': '0'})):
            row = runner.run(module, label, [name], options=opts, env=env)
            fit = row['fits'][0] if row['fits'] else {}
            p = accepted_problems(fit, MAX_ITER, oracle_keys=('b_scaled',)) if fit else ['no result']
            step, _, _, readings = trace_parts(fit)
            powers = [r for r in readings if r.get('loop') == 'step' and r.get('kind') in ('milestone', 'both')]
            if not step or step.get('reached') != '1' or step.get('frozen_stop') != '0':
                p.append(f'expected the step reached with frozen_stop=0: {step}')
            if powers:
                p.append(f'{len(powers)} power-of-two readings in loop=step')
            if step and step.get('readings') != step.get('confirmations'):
                p.append(f"step readings {step.get('readings')} != confirmations {step.get('confirmations')}")
            record('step', row, p)
    if want('large'):
        # The default row (pre-registered, b/stage2b/PRE_REGISTO_2B_20261118.md): deterministic on W0 x 2^34.
        name = f'W0_2p{LARGE_EXPONENT}'
        row = runner.run(module, 'large_default', [name, 'W0'], max_iter=LARGE_BUDGET, graded=False)
        fits = row['fits']
        p = []
        if [f.get('refused') for f in fits] != [True, False]:
            p.append(f"sequence {[f.get('refused') for f in fits]}")
        else:
            f = fits[0]
            step, passes, cont, _ = trace_parts(f)
            p += refusal_problems(f, LARGE_BUDGET)
            if not (step and step.get('reached') == '1' and step.get('formed') == '1'):
                p.append(f'expected the step reached and formed: {step}')
            if not (passes and passes[0].get('certified') == '1' and passes[0].get('checks') == 'pass'
                    and float(passes[0]['D']) > LARGE_FLOOR_FACTOR * TOL):
                p.append(f'pass 1 not certified with the checks passed and a floor above {LARGE_FLOOR_FACTOR} '
                         f'tolerances: {passes[:1]}')
            if 'the largest absolute group mean of the delivered residual is' not in f.get('error', ''):
                p.append('the refusal text does not carry D(e)')
            if not any(r in f.get('error', '') for r in ('no absorption iteration budget remains',
                                                         "the fit's single refinement was exhausted")):
                p.append('the refusal text does not give the reason')
            g = fits[1]
            if not (g['converged'] and g['certified'] and g['gpu_used'] and g['D'] <= TOL):
                p.append('W0 after the refusal not accepted on the same object')
        record('large', row, p)
    if want('all_is_default'):
        # Stage 2c addendum (b/RESPOSTA_2B_B_ADENDA.md): the value `all` was removed, so XHDFE_STRICT_FROZEN_STOP=all
        # means the default: T3 without regressors accepted as by default, its step line printing frozen_stop=step.
        row = runner.run(module, 'all_is_default_T3_nox', ['T3_nox'], env={'XHDFE_STRICT_FROZEN_STOP': 'all'})
        fit = row['fits'][0] if row['fits'] else {}
        p = accepted_problems(fit, MAX_ITER) if fit else ['no result']
        step, passes, cont, _ = trace_parts(fit)
        if not (step and step.get('frozen_stop') == 'step' and step.get('exit') == 'confirmed' and len(passes) == 1
                and cont is None):
            p.append(f'expected the default (frozen_stop=step, the step confirmed, one pass): {step}')
        record('all_is_default', row, p)
    excluded = []
    if want('excluded'):
        cases = [('weights', 'cov240k', dict(weights=True)), ('auto_mlsmr', 'cov240k', dict(method='auto-mlsmr')),
                 ('method_gauss_seidel', 'cov240k', dict(method='gauss-seidel')),
                 ('criterion_reghdfe', 'cov240k', dict(convergence='reghdfe')),
                 ('saved_effects', 'cov240k', dict(retain_fes=True)), ('group', 'W0', dict(group=True)),
                 ('slopes', 'W0', dict(slopes=True)),
                 ('slopes_criterion', 'W0', dict(slopes=True, convergence='reghdfe'))]
        for case, name, opts in cases:
            if case == 'weights':
                # Package C (18nov2026): a weighted strict fit with two or more effects requested on CUDA is the CPU
                # call. Routed arm, with or without a reference: the same module with the CUDA and the CPU backend,
                # code 7 against 0, both converged, certified and finite, then exact; the CPU's method is recorded
                # (on the CPU only the sweep methods make the absolute check). The switch-0 arm below is today's
                # comparison with the reference module (five processes per side, status 1, the note).
                ra = runner.run(module, 'excluded_weights_routed', [name],
                                options=dict(opts, weighted_cpu_route=True), graded=False)
                ca = runner.run(module, 'excluded_weights_cpu_backend', [name], options=opts, graded=False,
                                backend='cpu')
                fa, fc = (ra['fits'] or [None])[0], (ca['fits'] or [None])[0]
                pre = (excluded_outcome_problems([fa], 'routed', False, gpu=False) +
                       excluded_outcome_problems([fc], 'cpu backend', False, gpu=False))
                if not pre and not (fa.get('gpu_status_code') == 7 and fa.get('gpu_used') is False):
                    pre.append(f"routed: gpu_used {fa.get('gpu_used')} code {fa.get('gpu_status_code')} (expected 7)")
                if not pre and (fc.get('gpu_status_code') != 0 or fc.get('gpu_used') is not False):
                    pre.append(f"cpu backend: gpu_used {fc.get('gpu_used')} code {fc.get('gpu_status_code')} "
                               '(expected false, 0)')
                keys = ('refused', 'iterations', 'converged', 'certified', 'b_hex', 'V_sha', 'e_sha', 'rss', 'error',
                        'method', 'tier_reason', 'cuda_tail')
                same = not pre and all(fa.get(k) == fc.get(k) for k in keys)
                excluded.append(dict(case='weights_routed', bit_identical=same, cpu_method=(fc or {}).get('method')))
                record('excluded', ra, pre or ([] if same else ['the routed weighted strict fit is not the CPU call, '
                                                                'bit for bit']))
            if args.reference_module:
                new = [runner.run(module, f'excluded_{case}_new_{i}', [name], options=opts, graded=False)
                       for i in range(EXCLUDED_RUNS)]
                ref = [runner.run(args.reference_module.resolve(), f'excluded_{case}_ref_{i}', [name], options=opts,
                                  graded=False) for i in range(EXCLUDED_RUNS)]
                fn = [r['fits'][0] if r['fits'] else None for r in new]
                fr = [r['fits'][0] if r['fits'] else None for r in ref]
                d, p, row_notes = grade_excluded_cuda(case, fn, fr)
                notes.extend(row_notes)
                if any(l.startswith('strict_accept_trace') for f in fn if f for l in f.get('trace', [])):
                    p.append('an excluded fit printed the acceptance trace (in scope?)')
                new[0]['aa_ab'] = d
                excluded.append(dict(case=case, aa_ab=d))
                record('excluded', new[0], p)
            else:
                labels.append(f'NOT_EXERCISED: excluded {case} against the reference (no --reference-module)')
        if args.cpu_module and args.reference_cpu_module:
            for case, name, opts in [('cpu_strict', 'cov240k', {}), ('cpu_strict_F0', 'F0', {})] + \
                    [(f'cpu_{c}', n, o) for c, n, o in cases]:
                a = runner.run(args.cpu_module.resolve(), f'excluded_{case}_new', [name], options=opts, graded=False,
                               backend='cpu')
                b = runner.run(args.reference_cpu_module.resolve(), f'excluded_{case}_ref', [name], options=opts,
                               graded=False, backend='cpu')
                fa, fb = (a['fits'] or [None])[0], (b['fits'] or [None])[0]
                keys = ('refused', 'iterations', 'converged', 'certified', 'b_hex', 'V_sha', 'e_sha', 'error')
                # Round 10: each process has the outcome its case expects before the comparison (no GPU fields).
                slopes = case.startswith('cpu_slopes')
                pre = (excluded_outcome_problems([fa], 'new', slopes, gpu=False) +
                       excluded_outcome_problems([fb], 'ref', slopes, gpu=False))
                same = not pre and all(fa.get(k) == fb.get(k) for k in keys)
                excluded.append(dict(case=case, bit_identical=same))
                record('excluded_cpu', a, pre or ([] if same else ['CPU not bit for bit against the reference module']))
        else:
            labels.append('NOT_EXERCISED: CPU bit for bit (no --cpu-module/--reference-cpu-module)')
    stata = None
    known_defects = []
    if args.stata_package and want('stata'):
        short_budget = calibrate_short()
        if short_budget is None:
            labels.append('NOT_EXERCISED: stata cont_short without a calibrated pass 1')
        stata = run_stata(args, workdir, runner, short_budget or 0)
        if stata['problems']:
            failures.append(dict(label='stata', problems=stata['problems']))
        labels.extend(stata['labels'])
        known_defects.extend(stata['known_defects'])
        notes.extend(stata['notes'])
    status = 'FAIL' if failures else ('KNOWN_DEFECT' if known_defects else ('NOT_EXERCISED' if labels else 'PASS'))
    known = (stata or {}).get('known', {})  # the SCA_K lines as read
    report = dict(status=status, failures=failures, known_defects=known_defects, labels=labels, notes=notes,
                  known=known, excluded=excluded, stata=stata,
                  module=str(module), module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                  limits=LIMITS, gpu_apps_at_start=apps, seconds=time.time() - started, fits=runner.count,
                  rows=[{k: v for k, v in r.items() if k != 'stderr_tail'} for r in rows])
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'failures', 'known_defects', 'labels', 'notes', 'fits',
                                             'seconds')},
                     indent=1)[:20000])
    if cleanup:
        cleanup.cleanup()
    return 1 if failures else (4 if known_defects else (3 if labels else 0))


if __name__ == '__main__':
    sys.exit(main())
