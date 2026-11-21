"""P3-fix (3'): the strict-residual CUDA recovery after a broken CG stage, graded against
independent oracles (offline, CUDA only).

Reproducer (Verifications/perf_fereg_gap_20261106/RELATORIO_P3FIX_20261109.md sections 7.4, 10, 11):
the public cov240k fixture (240 000 rows, three FEs, cluster on the worker), tolerance_mode
strict-residual, CUDA, 16 threads. The regressor's native CUDA attempt of a strict fit runs the CG
stage (xhdfe-fast options, update criterion); about one process in ten breaks down at iteration 46
(alpha above the 1/sqrt(eps) ceiling). Up to 2.30.1 the broken stage restarted with the cold MAP under
the Comparable stopping rule and the strict certificate accepted that state (b 3.55e-7, V 1.11e-4,
SE 2.91e-5, residuals 4.73e-4 against the oracle: outside the 2.28.1 strict envelope).
The fix (rounds 20-21, hypothesis P3-S): a finite sequence under the strict provenance
(HdfeOptions::strict_provenance, internal): the first breakdown in a strict stage (the native attempt
or the strict repair) uses the one cold device LSMR restart from the original data with the strict
check (fe_absorption_cuda.cu restart_strict_after_breakdown); an LSMR state that fails the strict
acceptance with budget left uses the one strict repair (run_strict_cuda_repair); a breakdown after
the LSMR was used ends with an honest refusal; never the 2.30.1 cold MAP.

Cases (one fresh process per fit; tol 1e-8 unless stated; max_iter 100 000):
* natural (--natural, default 40): no hook. Accepted, converged, certified, GPU status 1, inside the
  strict envelope, the strict check D <= max(tol,64 eps)(1 + |b_slopes|_1), no cold MAP; a fit whose CG
  broke by itself must show the LSMR restart approved. No natural breakdown: NOT_EXERCISED_ROUTE.
* forced (--forced, default 8 per tolerance; the release gate runs 40): XHDFE_TEST_CUDA_CG_BREAKDOWN=30.
  One forced breakdown, one LSMR restart approved (lsmr-strict or lsmr-polished), accepted, inside
  the envelope of the tolerance, stage sum exact.
* late (--late, default 8): hook 45 (the iteration at which the natural breakdowns happen): as forced;
  a fit whose CG converged before 45 is NOT_EXERCISED.
* budget (--budget-fits, default 2 each, hook 30): max_iter 30 (no LSMR, refusal 30/30), 31 (LSMR
  stopped after one iteration, refusal 31/31), 119 (LSMR stopped one short, refusal 119/119), 120 (LSMR
  approved at exactly the budget, accepted). Refusals leave no estimates.
* polish (--polish-fits, default 2, hook 30): y scaled by 2^17 (exact; the oracle scales exactly), so
  the LSMR state misses the absolute strict condition 1e-10 and the existing strict polish runs:
  accepted (lsmr-polished) inside the envelope; then max_iter = 120 + polish - 1 (the polish iterations
  of the first fit): refusal (polish-stopped) with every stage counted.
* sequence (--sequence-fits, default 3 each; test hook XHDFE_TEST_CUDA_STRICT_LSMR_REJECT=1 rejects an
  approved LSMR state: flow control, not frequency): B2 cov240k hook 30 -> LSMR rejected -> the strict
  repair (one MAP stage) -> accepted inside the envelope; B3 the path-like two-FE fixture (12 000 rows,
  an even cycle plus two chords: the repair's own CG gate) hook 25:all -> LSMR rejected -> repair -> CG
  breaks again -> refusal, iterations = CG + LSMR + CG from the stage lines; B4 one process, one
  regressor object: an easy fixture (success), the B3 fixture (refusal, no estimate left), the easy
  fixture again (success, equal to the first).
* explicit criterion (scope addendum 2; --reghdfe-natural, default 40; --reghdfe-forced, default 8):
  convergence reghdfe keeps the strict primary outside the regressor's strict branch; the same
  sequence must cover it (natural and forced fits as above; B2 and B3 also with convergence reghdfe).
* negative control (--reference-module, e.g. the 13b1 build or b4c5bbd9; --reference-fits, default 8
  per entry, hook 30, entries auto and reghdfe): on an entry without the sequence the reference must
  accept the cold MAP state outside the envelope (detected); at least one entry must be detected in
  full; hook never fired: NEGATIVE_CONTROL_NOT_EXERCISED (missing coverage).
* stata (--stata-package DIR): one Stata session with the hook 30:all: success (120 iterations), a
  refusal with maxiter(31) that leaves no e() estimates, success again; e(gpu_used)==1,
  e(gpu_backend)=="cuda", e(gpu_status)=="used". A second session takes the strict MAP route (hook 30
  and the reject hook): flow, b, V and SE graded; the residuals() variable, re-absorbed by
  refine_stored_residuals by MAP at the public tolerance (older than 2.28.1; natural in every strict CUDA
  Stata fit with residuals() outside the LSMR route), is labelled
  KNOWN_DEFECT_STATA_RESIDUALS_CUDA_REFINE while it misses the strict envelope (inherited, P7).
Invariants checked on every fit (mandatory): at most one LSMR restart and at most one strict repair
stage, the reported (or refusal message) iterations equal the exact sum of the stage lines, never more
than max_iter. Costs (iterations, seconds) are recorded, never graded: the fixture's cost limit is
frozen only after the campaign and the user's decision (round 20, condition 8).
Always labelled (missing coverage, coordinator 9 nov 2026): the generic CUDA repair of WEIGHTED
xhdfe-fast / reghdfe-comparable fits also calls run_strict_cuda_repair, without the strict provenance;
its path-like CG can break and fall into the 2.30.1 cold MAP; acceptance is then that of the public mode;
not measured here (NOT_EXERCISED_WEIGHTED_GENERIC_REPAIR).

Strict envelope (round 20, condition 5; Verifications/perf_fereg_gap_20261106/p3fix/
ENVELOPE_STRICT_20261109.json, sha256 ENVELOPE_STRICT_SHA256): per tolerance, the maxima of 40
accepted 2.28.1 strict CUDA fits; embedded below.
Package B (18nov2026; pre-registered in Verifications/corrigir_erros_20261117/b/DESENHO_B.md section 9 and
b/stage2b/PRE_REGISTO_2B_20261118.md): a strict-residual CUDA fit with the automatic method and criterion, no
weights, no saved effects and no group() is now in scope of the strict acceptance of the delivered residual: it
takes the reghdfe-comparable route (on cov240k no CG stage: sweeps and the device LSMR completion), and the strict
sequence above no longer runs for it. The automatic entry is graded from each process's own trace
(XHDFE_STRICT_ACCEPT_TRACE=1): the reported iterations equal the last strict pass line's, pass 2 = pass 1 + the
continuation's sweeps, a refusal before the check equals its stage lines; no cuda_strict_restart line; the
cold-MAP ban is lifted for these rows (the comparable route's own MAP stage). natural, forced, late and B2: accepted
inside the envelope (the hooks do not fire: NOT_EXERCISED for this entry); budgets 30, 31, 119 and 120: refused at
B/B; polish: accepted after the continuation, then exhausted at pass-1 iterations + 2: refused after reaching the
continuation; B3: accepted (device LSMR); B4: three successes. New in scope with the hook firing: cgseed_iv (the
public CG counterexample, its IV case, hook 3): the comparable tail restart to the device LSMR, accepted, graded
against its MP120 oracle at the package-B limits. The explicit-criterion rows and the negative control are
unchanged. Stata: session 1 accepted / refused 31/31 / accepted with no strict restart; session 2 (automatic entry)
stores the fit's own residuals (p7_refine_trace refined=0), inside the envelope: the label
KNOWN_DEFECT_STATA_RESIDUALS_CUDA_REFINE is a failure there; session 3 (the same with convergence(reghdfe)) keeps
that label inside its band. --v3-npz names the public oracle file (default unchanged).
Stage 2c (b/RESPOSTA_2B_B.md): the earlier strict branch still ships, so the hooks keep a route that fires them:
the legacy entry, the automatic entry under XHDFE_STRICT_CUDA_LEGACY_ACCEPTANCE=1, graded exactly as HEAD graded
the automatic entry (forced, late, the four budgets 30/31/119/120 with their outcomes, the polish and its
exhaustion, B2, B3, B4, and HEAD's two Stata sessions: A and C at 120 with lsmr-strict, lsmr-stopped, lsmr-strict;
the strict MAP session with its P7 band). Where the legacy entry fired the hook, the in-scope "the hook does not
fire" is reported as a note, not a label.
Exit code 0 PASS, 1 FAIL, 3 only labels of missing coverage, 4 a known defect (KNOWN_DEFECT inside the
declared band of its known occurrence; missing coverage listed apart in not_exercised): never a clean PASS.
usage: cuda_strict_restart_contract.py --module PATH.so [--natural 40] [--forced 8] [--late 8]
       [--budget-fits 2] [--polish-fits 2] [--sequence-fits 3] [--tolerances 1e-8,1e-8+1ulp,2e-8]
       [--reference-module PATH.so] [--reference-fits 8] [--stata-package DIR] [--output result.json]
       [--keep DIR] [--v3-npz PATH] [--cgseed-fits 2]
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
                                        envelope_metrics, fields, gpu_apps, mp_slopes, parse_trace,
                                        structure, wilson, within_transform)

MAX_ITER = 100000
THREADS = 16
FORCED = 30
LATE = 45
SEQUENCE_HOOK, SEQUENCE_HOOK_ITERATION = '25:all', 25
Y_SCALE = 2.0 ** 17
EPS = 2.0 ** -52
ENVELOPE_STRICT_SOURCE = 'Verifications/perf_fereg_gap_20261106/p3fix/ENVELOPE_STRICT_20261109.json'
ENVELOPE_STRICT_SHA256 = 'b866e11aad3fd29366e1d83c53c885d12c1828bc68190b39cd8a42374b717021'
ENVELOPE_STRICT = {
    '1e-8': dict(b_scaled=4.4538619299963927e-07, V_scaled=6.81264079092126e-07,
                 se_rel=2.029804002656732e-07, resid_rel=9.632214715117729e-08),
    '1e-8+1ulp': dict(b_scaled=3.820877725943689e-07, V_scaled=1.2882468450939105e-06,
                      se_rel=6.441232152099019e-07, resid_rel=4.939097398529736e-07),
    '2e-8': dict(b_scaled=5.000185621805129e-07, V_scaled=8.797758689474726e-07,
                 se_rel=2.810373187500659e-07, resid_rel=8.919289432472049e-08)}
METRICS = ('b_scaled', 'V_scaled', 'se_rel', 'resid_rel')
ORACLE_NPZ = [V3_NPZ]  # the public cov240k oracle; --v3-npz replaces it
# Package B: the limits of a fit in scope against an oracle (DESENHO_B.md section 9).
SCOPE_LIMITS = dict(b_scaled=1e-8, se_rel=1e-7, V_scaled=1e-8)
# Declared band of the known P7 occurrence in the Stata strict MAP session (reviewer round 24): the values
# reproduced on 9 nov 2026 (stored residuals 2.07788e-6 to 2.07792e-6, D 9.99528e-9 to 9.99546e-9 in 2.28.1,
# 2.30.1 and the candidate) with a +-0.5 % margin; an aggravation beyond it is a new failure.
P7_STRICT_BAND = dict(resid=(2.067e-6, 2.089e-6), D=(9.945e-9, 1.0045e-8))
APPROVED = ('lsmr-strict', 'lsmr-polished')
STRICT_RESTART = re.compile(r'^cuda_strict_restart (.*)$', re.M)
STRICT_ACCEPT = re.compile(r'^strict_accept_trace (.*)$', re.M)
TAIL_RESTART_LINE = re.compile(r'^cuda_tail_restart (.*)$', re.M)
CGSEED_HOOK = 3  # package B: the cgseed_iv case's CG stage breaks naturally at 7 in comparable
WEIGHTED_LABEL = ('NOT_EXERCISED_WEIGHTED_GENERIC_REPAIR: the generic CUDA repair of weighted xhdfe-fast/'
                  'reghdfe-comparable fits calls run_strict_cuda_repair without the strict provenance; its '
                  'path-like CG can break and fall into the 2.30.1 cold MAP; not measured')


def pathlike_fixture(n=12000, seed=20261109):
    """An even cycle of n - 2 edges on (n - 2)/2 levels per FE plus two chords: no singleton,
    n - 2 levels (the path-like CG gate), a within space of dimension 3; one regressor."""
    m = (n - 2) // 2
    k = np.arange(n - 2)
    fe_a = np.r_[k // 2, [0, m // 3]].astype(np.int32)
    fe_b = np.r_[((k + 1) // 2) % m, [m // 2, (2 * m) // 3]].astype(np.int32)
    rng = np.random.default_rng(seed)
    alpha, beta = rng.normal(size=m), rng.normal(size=m)
    x = rng.normal(size=n) + 0.3 * alpha[fe_a] - 0.2 * beta[fe_b]
    y = 0.5 * x + alpha[fe_a] + beta[fe_b] + 0.1 * rng.normal(size=n)
    return y, x.reshape(-1, 1), [fe_a, fe_b]


def easy_fixture():
    rng = np.random.default_rng(20261110)
    n = 20000
    fes = [rng.integers(0, k, size=n).astype(np.int32) for k in (20, 25, 30)]
    x = rng.normal(size=(n, 2))
    y = x @ np.array([0.7, -1.3]) + sum(rng.normal(size=k)[f] for k, f in zip((20, 25, 30), fes)) \
        + rng.normal(size=n)
    return y, x, fes


def group_mean_max(e, fes):
    out = 0.0
    for fe in fes:
        sums, counts = np.bincount(fe, weights=e), np.bincount(fe)
        out = max(out, float(np.max(np.abs(sums[counts > 0] / counts[counts > 0]))))
    return out


def regressor(core, tol, max_iter, se_type, convergence='auto'):
    return core.HdfeRegressor(se_type=se_type, tol=tol, max_iter=max_iter, num_threads=THREADS,
                              drop_singletons=True, retain_fes=False, symmetric_sweep=False,
                              absorption_method='auto', tolerance_mode='strict-residual',
                              jacobi_relaxation=0., convergence=convergence)


def estimates_left(reg):
    return {a: int(np.asarray(getattr(reg, a)).size) for a in ('coef_', 'covariance_', 'residuals_')}


def fit_record(reg, y, x, fes, clusters=None):
    rec = {}
    t0 = time.perf_counter()
    try:
        if clusters is None:
            reg.fit(y, np.asfortranarray(x), fes=fes)
        else:
            reg.fit(y, np.asfortranarray(x), fes=fes, clusters=clusters)
    except RuntimeError as error:
        m = re.search(r'iterations=(\d+)/(\d+)', str(error))
        rec.update(refused=True, error=str(error)[:800], message_iterations=int(m.group(1)) if m else None,
                   message_max_iter=int(m.group(2)) if m else None)
    else:
        e = np.asarray(reg.residuals_, np.float64).ravel()
        rec.update(refused=False, iterations=int(reg.num_iterations_), converged=bool(reg.converged_),
                   certified=bool(reg.precision_certified_), gpu_used=bool(reg.gpu_used_),
                   gpu_status_code=int(reg.gpu_status_code_),
                   abs_residual_rel=float(getattr(reg, 'abs_residual_rel_', math.nan)),
                   group_mean_max=group_mean_max(e, fes) if e.shape == (len(y),) else None,
                   b1=1.0 + float(np.sum(np.abs(np.asarray(reg.coef_, np.float64).ravel()[:x.shape[1]]))))
    rec.update(seconds=time.perf_counter() - t0, estimates_left=estimates_left(reg))
    return rec


def worker(job_path):
    import importlib.util
    job = json.loads(Path(job_path).read_text())
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', job['module'])
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    tol = float.fromhex(job['tol_hex'])
    conv = job.get('convergence', 'auto')
    out = {}
    if job['fixture'] in ('cov240k', 'cov240k_scaled'):
        y, x, fes = structure('3fe')
        scale = Y_SCALE if job['fixture'] == 'cov240k_scaled' else 1.0
        y = y * scale
        reg = regressor(core, tol, job['max_iter'], 'cluster', conv)
        out = fit_record(reg, y, x, fes, clusters=fes[0])
        if not out['refused']:
            o = np.load(job['oracle_npz'])
            out.update(envelope_metrics(reg.coef_, reg.covariance_, reg.residuals_, o['b'] * scale,
                                        o['V'] * scale * scale, o['e'] * scale))
    elif job['fixture'] == 'pathlike':
        y, x, fes = pathlike_fixture()
        out = fit_record(regressor(core, tol, job['max_iter'], 'unadjusted', conv), y, x, fes)
    elif job['fixture'] == 'cgseed_iv':
        # Package B: the public CG counterexample (tests/analytic_precision/fixtures/cg_seed1_2017.csv), its IV
        # case with the options of cg_final_step_contract.py, graded against its MP120 oracle.
        import pandas as pd
        fixtures = HERE / 'fixtures'
        oracle = json.loads((fixtures / 'cg_seed1_2017_oracle.json').read_text())
        data_path = fixtures / 'cg_seed1_2017.csv'
        assert hashlib.sha256(data_path.read_bytes()).hexdigest() == oracle['csv_sha256']
        frame = pd.read_csv(data_path, float_precision='round_trip').sort_values(['sid', 'year'])
        y, x, z = frame.dagg.to_numpy(), frame['expo'].to_numpy(), frame['z'].to_numpy()
        fes = [pd.factorize(frame[k], sort=True)[0].astype(np.int32) for k in ('sid', 'year', 'entry_age')]
        clusters = pd.factorize(frame.did, sort=True)[0].astype(np.int32)
        reg = core.HdfeRegressor(se_type='cluster', num_threads=1, stats_style='legacy', ssc_k_adj=False,
                                 ssc_g_adj=False, tolerance_mode='strict-residual', tol=tol, max_iter=job['max_iter'])
        t0 = time.perf_counter()
        try:
            reg.fit(y, x[:, None], fes=fes, clusters=clusters[:, None], instruments=z[:, None], endogenous_idx=[0])
        except RuntimeError as error:
            m = re.search(r'iterations=(\d+)/(\d+)', str(error))
            out = dict(refused=True, error=str(error)[:800], message_iterations=int(m.group(1)) if m else None,
                       message_max_iter=int(m.group(2)) if m else None)
        else:
            ref = oracle['references']['iv']
            ob = np.array([float(v) for v in ref['b']])
            oV = np.array([[float(v) for v in row] for row in ref['V']])
            b, V = np.asarray(reg.coef_, np.float64), np.asarray(reg.covariance_, np.float64)
            e = np.asarray(reg.residuals_, np.float64).ravel()
            out = dict(refused=False, iterations=int(reg.num_iterations_), converged=bool(reg.converged_),
                       certified=bool(reg.precision_certified_), gpu_used=bool(reg.gpu_used_),
                       gpu_status_code=int(reg.gpu_status_code_),
                       group_mean_max=group_mean_max(e, fes) if e.shape == y.shape else None,
                       b_scaled=float(np.max(np.abs(b - ob) / np.maximum(1.0, np.abs(ob)))),
                       se_rel=float(np.max(np.abs(np.sqrt(np.diag(V)) / np.sqrt(np.diag(oV)) - 1.0))),
                       V_scaled=float(np.max(np.abs(V - oV) / np.sqrt(np.outer(np.diag(oV), np.diag(oV))))))
        out.update(seconds=time.perf_counter() - t0, estimates_left=estimates_left(reg))
    else:  # session: easy, path-like, easy on one regressor object
        reg = regressor(core, tol, job['max_iter'], 'unadjusted', conv)
        fits = []
        for name, data in (('easy_a', easy_fixture()), ('pathlike', pathlike_fixture()), ('easy_c', easy_fixture())):
            rec = fit_record(reg, *data)
            if not rec['refused']:
                rec.update(coef=np.asarray(reg.coef_, np.float64).ravel().tolist(),
                           resid_sha256=hashlib.sha256(np.asarray(reg.residuals_, np.float64).tobytes()).hexdigest())
            fits.append(dict(rec, name=name))
        out = dict(fits=fits)
    Path(job['out']).write_text(json.dumps(out))


def trace_env(hooks):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('XHDFE_') and k not in ('LD_LIBRARY_PATH', 'OMP_NUM_THREADS')}
    env.update(XHDFE_GPU_BACKEND='cuda', XHDFE_ABSORPTION_CACHE_MODE='off', XHDFE_MOBILITY_MODE='off',
               XHDFE_FE_STRUCTURE_MODE='off', XHDFE_GI_LSMR_TRACE='1', XHDFE_ACCEL_DEBUG='1',
               XHDFE_CUDA_ACCEL_TRACE='1', XHDFE_STRICT_ACCEPT_TRACE='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    env.update(hooks)
    return env


def run_fit(module, directory, label, case, fixture, tol_name, max_iter, hooks, convergence='auto'):
    job = dict(module=str(module), tol_hex=TOLERANCES[tol_name].hex(), max_iter=max_iter, fixture=fixture,
               oracle_npz=str(ORACLE_NPZ[0]), out=str(directory / f'{label}.json'), convergence=convergence)
    (directory / f'{label}.job.json').write_text(json.dumps(job))
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                               str(directory / f'{label}.job.json')], env=trace_env(hooks),
                              capture_output=True, text=True, timeout=3600)
        stderr, rc = proc.stderr, proc.returncode
    except (OSError, subprocess.SubprocessError) as error:
        stderr, rc = f'{type(error).__name__}: {error}', None
    (directory / f'{label}.err').write_text(stderr)
    rec = json.loads((directory / f'{label}.json').read_text()) if (directory / f'{label}.json').is_file() else {}
    trace = parse_trace(stderr)
    row = dict(label=label, case=case, fixture=fixture, tolerance=tol_name, max_iter=max_iter, hooks=hooks,
               convergence=convergence,
               rc=rc, wall_seconds=time.time() - t0, record=rec, breakdowns=trace['breakdowns'],
               cg_end=trace['cg_end'], accel=trace['accel'], lsmr=trace['lsmr'],
               other_restarts=len(trace['fast_restart']) + trace['tail_restart_lines'] + len(trace['strict_handback']),
               strict_restart=[fields(m.group(1)) for m in STRICT_RESTART.finditer(stderr)],
               strict_accept=[fields(m.group(1)) for m in STRICT_ACCEPT.finditer(stderr)],
               tail_restart=[fields(m.group(1)) for m in TAIL_RESTART_LINE.finditer(stderr)])
    if rc != 0:
        row['stderr_tail'] = stderr[-3000:]
    return row


def invariant_problems(row, rec, tol):
    """At most one LSMR restart and one strict repair stage; the exact stage sum; <= max_iter.
    cuda_accel_trace lines: the native attempt's stage first, then one line per strict polish (one per
    ledger with polished=1) and per strict repair stage; the LSMR prints gi_lsmr_solve lines."""
    del tol
    p = []
    restarts = row['strict_restart']
    if sum(1 for s in restarts if s.get('lsmr_available') == '1') > 1:
        p.append('more than one LSMR restart')
    if any(a.get('honest') == '1' for a in row['accel']):
        p.append('a cold MAP (honest=1) ran')
    if row['other_restarts']:
        p.append('a Fast, Comparable-tail or option-3 restart ran')
    polished = sum(1 for s in restarts if s.get('polished') == '1')
    repair_stages = len(row['accel']) - 1 - polished
    if repair_stages > 1:
        p.append(f'{repair_stages} strict repair stages (one at most)')
    stage_sum = (sum(int(a.get('iterations', 0)) for a in row['accel'])
                 + sum(int(x.get('iterations', 0)) for x in row['lsmr'])
                 + sum(int(s.get('uncounted_it', 0)) for s in restarts))
    reported = rec.get('message_iterations') if rec.get('refused') else rec.get('iterations')
    if reported != stage_sum:
        p.append(f'iterations {reported} != the stage lines {stage_sum}')
    if reported is not None and reported > row['max_iter']:
        p.append(f"{reported} iterations exceed max_iter {row['max_iter']}")
    return p


def scope_invariant_problems(row, rec, allow_tail_restart=False):
    """Package B, the automatic entry (in scope): no strict restart, the iterations exact from the process's own
    trace (the last strict pass line; pass 2 = pass 1 + the continuation's sweeps; a refusal before the check:
    the stage lines), within max_iter. The comparable route's own MAP stage may print honest=1."""
    p = []
    if row['strict_restart']:
        p.append(f"{len(row['strict_restart'])} cuda_strict_restart lines (the strict sequence ran in scope)")
    if row['other_restarts'] - (len(row['tail_restart']) if allow_tail_restart else 0):
        p.append('a Fast, Comparable-tail or strict hand-back restart ran')
    reported = rec.get('message_iterations') if rec.get('refused') else rec.get('iterations')
    passes = [a for a in row['strict_accept'] if 'pass' in a and 'D' in a]
    cont = [a for a in row['strict_accept'] if a.get('refinement') == 'continuation']
    if passes:
        if reported != int(passes[-1]['iterations']):
            p.append(f"iterations {reported} != the last strict pass line's {passes[-1]['iterations']}")
        firsts = [a for a in passes if a.get('pass') == '1']
        seconds = [a for a in passes if a.get('pass') == '2']
        if seconds and cont and int(seconds[0]['iterations']) != int(firsts[0]['iterations']) + int(cont[0]['sweeps']):
            p.append('pass 2 iterations != pass 1 + the continuation sweeps')
    else:
        stage_sum = (sum(int(a.get('iterations', 0)) for a in row['accel'])
                     + sum(int(x.get('iterations', 0)) for x in row['lsmr']))
        if reported != stage_sum:
            p.append(f'iterations {reported} != the stage lines {stage_sum} (no strict pass line)')
    if reported is not None and reported > row['max_iter']:
        p.append(f"{reported} iterations exceed max_iter {row['max_iter']}")
    return p


def accepted_problems(row, rec, tol_name):
    p = []
    if rec.get('refused') is not False:
        return [f"refused or no result: {str(rec.get('error'))[:200]}"]
    if not (rec.get('converged') and rec.get('certified') and rec.get('gpu_used') and rec.get('gpu_status_code') == 1):
        p.append('not converged, certified and on the GPU with status 1')
    tol = TOLERANCES[tol_name]
    if rec.get('group_mean_max') is None or not rec['group_mean_max'] <= max(tol, 64 * EPS) * rec['b1']:
        p.append(f"strict check D {rec.get('group_mean_max')} > max(tol,64eps)*B1 ({rec.get('b1')})")
    for k in METRICS:
        if not rec.get(k, math.inf) <= ENVELOPE_STRICT[tol_name][k]:
            p.append(f"{k} {rec.get(k)} outside the strict envelope {ENVELOPE_STRICT[tol_name][k]:.4g}")
    return p


def refusal_problems(rec, expected=None, max_iter=None):
    p = []
    if rec.get('refused') is not True:
        return ['not refused']
    if any(rec.get('estimates_left', {}).values()) or 'No estimates returned' not in rec.get('error', ''):
        p.append('estimates left behind or no refusal notice')
    if expected is not None and rec.get('message_iterations') != expected:
        p.append(f"message iterations {rec.get('message_iterations')} != {expected}")
    if max_iter is not None and rec.get('message_max_iter') != max_iter:
        p.append(f"message max_iter {rec.get('message_max_iter')} != {max_iter}")
    return p


def forced_problems(row, iteration, outcomes):
    forced = [b for b in row['breakdowns'] if b['stage'] == 'test-forced']
    got = [s.get('outcome') for s in row['strict_restart']]
    p = []
    if not forced or forced[0]['iter'] != iteration:
        p.append(f"forced breakdowns {[(b['stage'], b['iter']) for b in row['breakdowns']]}")
    if got not in outcomes:
        p.append(f'strict restart outcomes {got}')
    return p


def stata_session(folder, package, data, oracle_do, convergence, hooks, script_args=''):
    """One fresh stata-mp session of cuda_strict_restart_stata_fit.do (the strict MAP route: hook 30 and the
    reject hook) with the P7 trace; returns the session's result line fields and its trace."""
    fit_do = HERE / 'cuda_strict_restart_stata_fit.do'
    folder.mkdir()
    (folder / 'driver.do').write_text(f'capture noisily do "{fit_do}" "{Path(package).resolve()}" "{data}" '
                                      f'"{oracle_do}" 1e-08 {MAX_ITER} strict-residual {convergence}\n'
                                      'display "DRIVER_RC " _rc\nexit, clear\n')
    env = trace_env(dict(hooks, XHDFE_P7_REFINE_TRACE='1'))
    env.pop('XHDFE_GPU_BACKEND', None)
    env['STATATMP'] = str(folder)
    stata = shutil.which('stata-mp') or '/usr/local/stata/stata-mp'
    with (folder / 'process.log').open('x') as log:
        try:
            rc = subprocess.run([stata, '-q', '-b', 'do', 'driver.do'], cwd=folder, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=3600).returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
    text = (folder / 'driver.log').read_text(errors='replace') if (folder / 'driver.log').is_file() else ''
    trace = (folder / 'process.log').read_text(errors='replace')
    head = re.search(r'^P3S_STATA (.*)$', text, re.M)
    metrics = re.search(r'^P3S_STATA_M (.*)$', text, re.M)
    res = None
    if rc == 0 and head and metrics:
        res = dict(re.findall(r'(\w+)=\s*(\S+)', head.group(1)), **dict(re.findall(r'(\w+)=\s*(\S+)', metrics.group(1))))
    return dict(returncode=rc, result=res, trace=trace,
                outcomes=[fields(m.group(1)).get('outcome') for m in STRICT_RESTART.finditer(trace)],
                p7=[fields(m.group(1)) for m in re.finditer(r'^p7_refine_trace (.*)$', trace, re.M)])


def p7_label(resid, dmax, problems, labels, where):
    """The known P7 occurrence (reviewer round 24): a label inside its declared band, a failure outside it."""
    if P7_STRICT_BAND['resid'][0] <= resid <= P7_STRICT_BAND['resid'][1] and \
            P7_STRICT_BAND['D'][0] <= dmax <= P7_STRICT_BAND['D'][1]:
        labels.append(f'KNOWN_DEFECT_STATA_RESIDUALS_CUDA_REFINE (strict, {where}): the refinement of the '
                      'stored residuals (Stata residuals(), CUDA) re-absorbs y - Xb by MAP at the public tolerance and '
                      f'returns {resid:.6g} against the oracle (strict envelope {ENVELOPE_STRICT["1e-8"]["resid_rel"]:.3g}; '
                      f'D {dmax:.6g}); inside the band of the known occurrence')
    else:
        problems.append(f'{where}: stored residuals {resid:.6g} (D {dmax:.6g}) outside the envelope and outside the '
                        f'band of the known P7 occurrence {P7_STRICT_BAND}')


def stata_session1(folder, package, data, oracle_do, extra_env):
    """One fresh stata-mp session of cuda_strict_restart_stata.do with the hook 30:all (A, B with maxiter(31), C)."""
    script = HERE / 'cuda_strict_restart_stata.do'
    folder.mkdir()
    (folder / 'driver.do').write_text(f'capture noisily do "{script}" "{Path(package).resolve()}" "{data}" '
                                      f'"{oracle_do}"\nlocal driver_rc = _rc\ndisplay "DRIVER_RC " `driver_rc\'\n'
                                      'exit, clear\n')
    env = trace_env(dict(extra_env, XHDFE_TEST_CUDA_CG_BREAKDOWN=f'{FORCED}:all'))
    env.pop('XHDFE_GPU_BACKEND', None)
    env['STATATMP'] = str(folder)
    stata = shutil.which('stata-mp') or '/usr/local/stata/stata-mp'
    with (folder / 'process.log').open('x') as log:
        try:
            rc = subprocess.run([stata, '-q', '-b', 'do', 'driver.do'], cwd=folder, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=3600).returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
    text = (folder / 'driver.log').read_text(errors='replace') if (folder / 'driver.log').is_file() else ''
    trace = (folder / 'process.log').read_text(errors='replace')
    problems = []
    lines = text.splitlines()
    if rc != 0 or not re.search(r'^DRIVER_RC 0\s*$', text, re.M):
        problems.append(f'Stata run failed (returncode {rc})')
    for marker in ('P3S_STATA_A_OK', 'P3S_STATA_B_REFUSED', 'P3S_STATA_C_OK', 'P3S_STATA_PASS'):
        if not any(line.startswith(marker) for line in lines):
            problems.append(f'marker {marker} missing')
    if 'iterations=31/31' not in text:
        problems.append('the refusal does not report iterations=31/31')
    return dict(returncode=rc, text=text, trace=trace, lines=lines, problems=problems,
                outcomes=[fields(m.group(1)).get('outcome') for m in STRICT_RESTART.finditer(trace)],
                iterations={m.group(1): int(m.group(2))
                            for m in re.finditer(r'^P3S_STATA_(A|C) iterations=(\d+)', text, re.M)})


def strict_map_head_problems(s, where, problems, labels):
    """HEAD's grading of the strict MAP session (hook 30 and the reject hook): the rejected LSMR, the e()
    diagnostics, b, V and SE inside the envelope; the stored residuals inside it or inside the P7 band (a label)."""
    if s['result'] is None:
        problems.append(f"{where} failed (returncode {s['returncode']})")
        return
    res = s['result']
    if s['outcomes'] != ['lsmr-rejected-by-test-hook']:
        problems.append(f"{where} outcomes {s['outcomes']}")
    if not (res.get('converged') == '1' and res.get('certified') == '1' and res.get('gpu_used') == '1'
            and res.get('gpu_status') == 'used'):
        problems.append(f'{where} e() diagnostics {res}')
    for key, short in (('b_scaled', 'b'), ('V_scaled', 'V'), ('se_rel', 'se')):
        if not float(res[short]) <= ENVELOPE_STRICT['1e-8'][key]:
            problems.append(f'{where} {key} {res[short]} outside the strict envelope')
    if not float(res['resid']) <= ENVELOPE_STRICT['1e-8']['resid_rel']:
        p7_label(float(res['resid']), float(res['D']), problems, labels, where)


def run_stata(package, workdir):
    import pandas as pd
    folder = workdir / 'stata'
    folder.mkdir()
    o = np.load(ORACLE_NPZ[0])
    y, x, fes = structure('3fe')
    data = folder / 'cov240k_strict.dta'
    pd.DataFrame(dict(y=y, x1=x[:, 0], x2=x[:, 1], x3=x[:, 2], x4=x[:, 3], worker=fes[0], firm=fes[1],
                      year=fes[2], e_oracle=o['e'])).to_stata(data, write_index=False, version=118)
    oracle_do = folder / 'oracle.do'
    oracle_do.write_text('matrix oracle_b = (' + ', '.join(format(v, '.17g') for v in o['b']) + ')\n'
                         'matrix oracle_V = (' + ' \\ '.join(', '.join(format(v, '.17g') for v in r)
                                                            for r in o['V']) + ')\n' +
                         ''.join(f'scalar envelope_{k} = {ENVELOPE_STRICT["1e-8"][k]!r}\n' for k in METRICS))
    problems, labels = [], []
    # Session 1, in scope (automatic entry): no strict restart; A's and C's iterations equal their trace's
    # strict pass lines (the first and the last of the session; B is refused before the check).
    s1 = stata_session1(folder / 'session1', package, data, oracle_do, {})
    problems += s1['problems']
    if s1['outcomes']:
        problems.append(f"cuda_strict_restart outcomes {s1['outcomes']} (the strict sequence ran in scope)")
    passes = [fields(m.group(1)) for m in STRICT_ACCEPT.finditer(s1['trace'])]
    passes = [a for a in passes if 'pass' in a and 'D' in a and a.get('decision') == 'accept']
    its = s1['iterations']
    if len(passes) != 2 or its.get('A') != int(passes[0]['iterations']) or its.get('C') != int(passes[1]['iterations']):
        problems.append(f"session 1 iterations {its} against the trace's accepting pass lines "
                        f"{[a.get('iterations') for a in passes]}")
    # Session 1 on the legacy entry (XHDFE_STRICT_CUDA_LEGACY_ACCEPTANCE=1), graded as HEAD graded the automatic
    # entry: the LSMR restarts lsmr-strict, lsmr-stopped, lsmr-strict; A and C at 30 + 90 = 120; no cold MAP.
    legacy = {'XHDFE_STRICT_CUDA_LEGACY_ACCEPTANCE': '1'}
    s1l = stata_session1(folder / 'session1_legacy', package, data, oracle_do, legacy)
    problems += [f'legacy session 1: {q}' for q in s1l['problems']]
    if s1l['outcomes'] != ['lsmr-strict', 'lsmr-stopped', 'lsmr-strict']:
        problems.append(f"legacy session 1: cuda_strict_restart outcomes {s1l['outcomes']}")
    if s1l['iterations'] != {'A': 120, 'C': 120}:
        problems.append(f"legacy session 1: iterations {s1l['iterations']} (expected A and C at 120)")
    if any(fields(m.group(1)).get('honest') == '1' for m in re.finditer(r'^cuda_accel_trace (.*)$', s1l['trace'], re.M)):
        problems.append('legacy session 1: a cold MAP ran')
    # Session 2: the strict MAP hooks on the automatic entry, now in scope: the comparable route, the fit's own
    # residuals stored (refined=0), inside the strict envelope; the P7 label is a failure here.
    hooks2 = {'XHDFE_TEST_CUDA_CG_BREAKDOWN': str(FORCED), 'XHDFE_TEST_CUDA_STRICT_LSMR_REJECT': '1'}
    s2 = stata_session(workdir / 'stata_strict_map', package, data, oracle_do, 'auto', hooks2)
    if s2['result'] is None:
        problems.append(f"session 2 failed (returncode {s2['returncode']})")
    else:
        res = s2['result']
        if s2['outcomes']:
            problems.append(f"session 2 outcomes {s2['outcomes']} (the strict sequence ran in scope)")
        if not (res.get('converged') == '1' and res.get('certified') == '1' and res.get('gpu_used') == '1'
                and res.get('gpu_status') == 'used'):
            problems.append(f'session 2 e() diagnostics {res}')
        for key, short in (('b_scaled', 'b'), ('V_scaled', 'V'), ('se_rel', 'se'), ('resid_rel', 'resid')):
            if not float(res[short]) <= ENVELOPE_STRICT['1e-8'][key]:
                problems.append(f'session 2 {key} {res[short]} outside the strict envelope (a failure in scope)')
        if not s2['p7'] or s2['p7'][-1].get('refined') != '0':
            problems.append(f"session 2 p7_refine_trace {s2['p7'][-1:]} (expected refined=0)")
    # Session 2 on the legacy entry: HEAD's grading, the P7 label inside its band.
    s2l = stata_session(workdir / 'stata_strict_map_legacy', package, data, oracle_do, 'auto', dict(hooks2, **legacy))
    strict_map_head_problems(s2l, 'legacy session 2 (automatic entry, legacy switch)', problems, labels)
    # Session 3: the explicit control (convergence reghdfe): unchanged, the P7 label inside its band.
    s3 = stata_session(workdir / 'stata_strict_map_reghdfe', package, data, oracle_do, 'reghdfe', hooks2)
    strict_map_head_problems(s3, 'session 3, explicit criterion', problems, labels)
    return dict(case='stata', returncode=s1['returncode'], problems=problems, outcomes=s1['outcomes'], labels=labels,
                legacy_session1=dict(outcomes=s1l['outcomes'], iterations=s1l['iterations'],
                                     returncode=s1l['returncode']),
                strict_map={k: v for k, v in s2.items() if k != 'trace'},
                strict_map_legacy={k: v for k, v in s2l.items() if k != 'trace'},
                strict_map_reghdfe={k: v for k, v in s3.items() if k != 'trace'},
                result_lines=[line for line in s1['lines'] + s1l['lines'] if line.startswith('P3S_STATA')])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--module', type=Path, required=True)
    ap.add_argument('--natural', type=int, default=40)
    ap.add_argument('--forced', type=int, default=8)
    ap.add_argument('--late', type=int, default=8)
    ap.add_argument('--budget-fits', type=int, default=2)
    ap.add_argument('--polish-fits', type=int, default=2)
    ap.add_argument('--sequence-fits', type=int, default=3)
    ap.add_argument('--reghdfe-natural', type=int, default=40,
                    help='natural fits of the explicit-criterion entry (convergence reghdfe)')
    ap.add_argument('--reghdfe-forced', type=int, default=8,
                    help='forced fits (hook 30, 1e-8) of the explicit-criterion entry')
    ap.add_argument('--tolerances', default=','.join(TOLERANCES))
    ap.add_argument('--reference-module', type=Path)
    ap.add_argument('--reference-fits', type=int, default=8)
    ap.add_argument('--stata-package', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--keep', type=Path)
    ap.add_argument('--v3-npz', type=Path, default=V3_NPZ, help='the public cov240k oracle (sha256 checked)')
    ap.add_argument('--cgseed-fits', type=int, default=2, help='package B: cgseed_iv in scope with hook 3')
    args = ap.parse_args()
    if not cuda_device_visible():
        print(json.dumps(dict(status='SKIP', reason='no visible CUDA device')))
        return 0
    if hashlib.sha256(args.v3_npz.read_bytes()).hexdigest() != V3_NPZ_SHA256:
        raise SystemExit('v3 oracle npz sha256 mismatch')
    ORACLE_NPZ[0] = args.v3_npz.resolve()
    source = ROOT / ENVELOPE_STRICT_SOURCE
    if source.is_file():
        if hashlib.sha256(source.read_bytes()).hexdigest() != ENVELOPE_STRICT_SHA256:
            raise SystemExit('strict envelope source changed')
        recorded = json.loads(source.read_text())
        for name, limits in ENVELOPE_STRICT.items():
            if any(recorded[name][k] != v for k, v in limits.items()):
                raise SystemExit(f'embedded strict envelope differs from {source} at {name}')
    module = args.module.resolve()
    tolerances = [t for t in args.tolerances.split(',') if t]
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=False)
        workdir, cleanup = args.keep, None
    else:
        cleanup = tempfile.TemporaryDirectory()
        workdir = Path(cleanup.name)
    labels, failures, rows = [WEIGHTED_LABEL], [], []

    def record(row, problems):
        row['problems'] = problems
        rows.append(row)
        if problems:
            failures.append(dict(label=row['label'], problems=problems))
        print(json.dumps(dict(label=row['label'], problems=problems,
                              outcomes=[s.get('outcome') for s in row['strict_restart']],
                              iterations=row['record'].get('iterations', row['record'].get('message_iterations')))),
              flush=True)

    started = time.time()
    apps = gpu_apps()
    natural_breakdowns = 0
    # Package B: the automatic entry is in scope (module docstring); graded from each process's own trace.
    for i in range(args.natural):
        row = run_fit(module, workdir, f'natural_{i:02d}', 'natural', 'cov240k', '1e-8', MAX_ITER, {})
        rec = row['record']
        p = accepted_problems(row, rec, '1e-8') + scope_invariant_problems(row, rec)
        if row['breakdowns']:
            natural_breakdowns += 1
        record(row, p)
    if args.natural and not natural_breakdowns:
        labels.append('NOT_EXERCISED_ROUTE: no natural breakdown among the natural fits')
    hook_unfired = 0
    for name in tolerances:
        for i in range(args.forced):
            row = run_fit(module, workdir, f"forced_{name.replace('+', 'p')}_{i:02d}", 'forced', 'cov240k', name,
                          MAX_ITER, {'XHDFE_TEST_CUDA_CG_BREAKDOWN': str(FORCED)})
            rec = row['record']
            hook_unfired += 0 if [b for b in row['breakdowns'] if b['stage'] == 'test-forced'] else 1
            record(row, accepted_problems(row, rec, name) + scope_invariant_problems(row, rec))
    reghdfe_breakdowns = 0
    for i in range(args.reghdfe_natural):
        row = run_fit(module, workdir, f'reghdfe_natural_{i:02d}', 'reghdfe_natural', 'cov240k', '1e-8', MAX_ITER,
                      {}, 'reghdfe')
        rec = row['record']
        p = accepted_problems(row, rec, '1e-8') + invariant_problems(row, rec, 1e-8)
        if row['breakdowns']:
            reghdfe_breakdowns += 1
            if [s.get('outcome') for s in row['strict_restart']] not in ([o] for o in APPROVED):
                p.append('a natural breakdown without an approved LSMR restart')
        record(row, p)
    if args.reghdfe_natural and not reghdfe_breakdowns:
        labels.append('NOT_EXERCISED_ROUTE: no natural breakdown among the explicit-criterion fits')
    for i in range(args.reghdfe_forced):
        row = run_fit(module, workdir, f'reghdfe_forced_{i:02d}', 'reghdfe_forced', 'cov240k', '1e-8', MAX_ITER,
                      {'XHDFE_TEST_CUDA_CG_BREAKDOWN': str(FORCED)}, 'reghdfe')
        rec = row['record']
        record(row, forced_problems(row, FORCED, [[o] for o in APPROVED]) + accepted_problems(row, rec, '1e-8')
               + invariant_problems(row, rec, 1e-8))
    for i in range(args.late):
        row = run_fit(module, workdir, f'late_{i:02d}', 'late', 'cov240k', '1e-8', MAX_ITER,
                      {'XHDFE_TEST_CUDA_CG_BREAKDOWN': str(LATE)})
        rec = row['record']
        hook_unfired += 0 if [b for b in row['breakdowns'] if b['stage'] == 'test-forced'] else 1
        record(row, accepted_problems(row, rec, '1e-8') + scope_invariant_problems(row, rec))
    # Package B, stage 2c: a label only if the legacy entry below does not fire the hook either; otherwise a note.
    scope_unfired = []
    if hook_unfired:
        scope_unfired.append(f'the CG hook did not fire in {hook_unfired} forced/late fits of the automatic '
                             'entry (in scope: the comparable route of cov240k has no CG stage)')
    # In scope the route needs its sweeps and the device LSMR (176-178 on cov240k): every budget below refuses.
    for max_iter in (30, 31, 119, 120):
        for i in range(args.budget_fits):
            row = run_fit(module, workdir, f'budget_{max_iter}_{i:02d}', f'budget_{max_iter}', 'cov240k', '1e-8',
                          max_iter, {'XHDFE_TEST_CUDA_CG_BREAKDOWN': str(FORCED)})
            rec = row['record']
            record(row, refusal_problems(rec, max_iter, max_iter) + scope_invariant_problems(row, rec))
    # The continuation (y x 2^17): accepted after it; then exhausted at that process's pass-1 iterations + 2.
    polish_primary = None
    for i in range(args.polish_fits):
        row = run_fit(module, workdir, f'polish_{i:02d}', 'polish', 'cov240k_scaled', '1e-8', MAX_ITER,
                      {'XHDFE_TEST_CUDA_CG_BREAKDOWN': str(FORCED)})
        rec = row['record']
        p = accepted_problems(row, rec, '1e-8') + scope_invariant_problems(row, rec)
        passes = [a for a in row['strict_accept'] if 'pass' in a and 'D' in a]
        if not (len(passes) == 2 and float(passes[0]['D']) > TOLERANCES['1e-8']
                and any(a.get('refinement') == 'continuation' for a in row['strict_accept'])):
            p.append(f"expected pass 1 above the tolerance, the continuation and pass 2: {[a.get('D') for a in passes]}")
        elif polish_primary is None:
            polish_primary = int(passes[0]['iterations'])
        record(row, p)
    if args.polish_fits:
        if polish_primary is None:
            labels.append('NOT_EXERCISED: the continuation (no polish fit to calibrate its exhaustion)')
        else:
            short = polish_primary + 2
            for attempt in range(3):
                row = run_fit(module, workdir, f'polish_exhausted_{attempt}', 'polish_exhausted', 'cov240k_scaled',
                              '1e-8', short, {'XHDFE_TEST_CUDA_CG_BREAKDOWN': str(FORCED)})
                rec = row['record']
                cont = [a for a in row['strict_accept'] if a.get('refinement') == 'continuation']
                if cont or attempt == 2:
                    break
            p = refusal_problems(rec, short, short) + scope_invariant_problems(row, rec)
            if not cont or cont[0].get('exit') != 'budget':
                p.append(f'the refusal did not come after the continuation ended by its budget: {cont}')
            record(row, p)
    reject = {'XHDFE_TEST_CUDA_STRICT_LSMR_REJECT': '1'}
    for i in range(args.sequence_fits):
        # The explicit-criterion entry (scope addendum 2): B2 and B3 with convergence reghdfe.
        row = run_fit(module, workdir, f'B2r_{i:02d}', 'B2_reghdfe', 'cov240k', '1e-8', MAX_ITER,
                      dict(reject, XHDFE_TEST_CUDA_CG_BREAKDOWN=str(FORCED)), 'reghdfe')
        rec = row['record']
        p = forced_problems(row, FORCED, [['lsmr-rejected-by-test-hook']]) + accepted_problems(row, rec, '1e-8')
        if len(row['accel']) != 2:
            p.append(f"{len(row['accel'])} stages (expected the CG stage and the strict repair)")
        record(row, p + invariant_problems(row, rec, 1e-8))
        row = run_fit(module, workdir, f'B3r_{i:02d}', 'B3_reghdfe', 'pathlike', '1e-8', MAX_ITER,
                      dict(reject, XHDFE_TEST_CUDA_CG_BREAKDOWN=SEQUENCE_HOOK), 'reghdfe')
        rec = row['record']
        cg = [int(c['cg_iterations']) for c in row['cg_end']]
        lsmr = [int(x['iterations']) for x in row['lsmr']]
        p = refusal_problems(rec, sum(cg) + sum(lsmr), MAX_ITER)
        if [s.get('outcome') for s in row['strict_restart']] != ['lsmr-rejected-by-test-hook', 'lsmr-already-consumed']:
            p.append(f"outcomes {[s.get('outcome') for s in row['strict_restart']]}")
        if cg != [SEQUENCE_HOOK_ITERATION] * 2 or len(lsmr) != 1:
            p.append(f'CG stages {cg}, LSMR {lsmr}')
        record(row, p + invariant_problems(row, rec, 1e-8))
    # Package B: the automatic entry of B2, B3 and B4 is in scope (pre-registered expectations).
    scope_b2_unfired = []
    for i in range(args.sequence_fits):
        row = run_fit(module, workdir, f'B2_{i:02d}', 'B2', 'cov240k', '1e-8', MAX_ITER,
                      dict(reject, XHDFE_TEST_CUDA_CG_BREAKDOWN=str(FORCED)))
        rec = row['record']
        if not [b for b in row['breakdowns'] if b['stage'] == 'test-forced']:
            scope_b2_unfired.append(f"{row['label']} (in scope: no CG stage on cov240k, the hook did not fire)")
        record(row, accepted_problems(row, rec, '1e-8') + scope_invariant_problems(row, rec))
        row = run_fit(module, workdir, f'B3_{i:02d}', 'B3', 'pathlike', '1e-8', MAX_ITER,
                      dict(reject, XHDFE_TEST_CUDA_CG_BREAKDOWN=SEQUENCE_HOOK))
        rec = row['record']
        p = scope_invariant_problems(row, rec)
        if rec.get('refused') is not False:
            p.append(f"refused or no result: {str(rec.get('error'))[:200]}")
        elif not (rec.get('converged') and rec.get('certified') and rec.get('gpu_used') and rec.get('gpu_status_code') == 1
                  and rec.get('group_mean_max') is not None and rec['group_mean_max'] <= TOLERANCES['1e-8']):
            p.append(f"not converged, certified, on the GPU with D <= tol: D {rec.get('group_mean_max')}")
        record(row, p)
        row = run_fit(module, workdir, f'B4_{i:02d}', 'B4', 'session', '1e-8', MAX_ITER,
                      dict(reject, XHDFE_TEST_CUDA_CG_BREAKDOWN=SEQUENCE_HOOK))
        fits = row['record'].get('fits', [])
        p = []
        if [f.get('refused') for f in fits] != [False, False, False]:
            p.append(f"sequence {[f.get('refused') for f in fits]} (expected three successes)")
        else:
            b0, b2 = np.array(fits[0]['coef']), np.array(fits[2]['coef'])
            if not np.max(np.abs(b0 - b2) / np.maximum(1.0, np.abs(b0))) <= SCOPE_LIMITS['b_scaled']:
                p.append('the third fit differs from the first beyond 1e-8 max(1,|b|)')
            for f in fits:
                if not (f.get('converged') and f.get('certified') and f.get('gpu_used') and f.get('gpu_status_code') == 1):
                    p.append(f"{f['name']} not converged, certified and on the GPU")
        if row['strict_restart']:
            p.append('the strict sequence ran in scope')
        record(row, p)
    # Package B, stage 2c: the legacy entry, the automatic entry under XHDFE_STRICT_CUDA_LEGACY_ACCEPTANCE=1 (the
    # earlier strict branch, which still ships), graded exactly as HEAD graded the automatic entry: forced, late,
    # the four budgets, the polish and its exhaustion, B2, B3, B4 (the Stata sessions in run_stata).
    legacy = {'XHDFE_STRICT_CUDA_LEGACY_ACCEPTANCE': '1'}
    legacy_fired = dict(forced=0, b2=0)
    for name in tolerances:
        for i in range(args.forced):
            row = run_fit(module, workdir, f"legacy_forced_{name.replace('+', 'p')}_{i:02d}", 'legacy_forced',
                          'cov240k', name, MAX_ITER, dict(legacy, XHDFE_TEST_CUDA_CG_BREAKDOWN=str(FORCED)))
            rec = row['record']
            legacy_fired['forced'] += 1 if [b for b in row['breakdowns'] if b['stage'] == 'test-forced'] else 0
            record(row, forced_problems(row, FORCED, [[o] for o in APPROVED]) + accepted_problems(row, rec, name)
                   + invariant_problems(row, rec, TOLERANCES[name]))
    for i in range(args.late):
        row = run_fit(module, workdir, f'legacy_late_{i:02d}', 'legacy_late', 'cov240k', '1e-8', MAX_ITER,
                      dict(legacy, XHDFE_TEST_CUDA_CG_BREAKDOWN=str(LATE)))
        rec = row['record']
        if not row['breakdowns'] and not row['strict_restart']:
            labels.append(f"NOT_EXERCISED: {row['label']} (the CG converged before iteration {LATE})")
            record(row, accepted_problems(row, rec, '1e-8') + invariant_problems(row, rec, 1e-8))
            continue
        legacy_fired['forced'] += 1
        record(row, forced_problems(row, LATE, [[o] for o in APPROVED]) + accepted_problems(row, rec, '1e-8')
               + invariant_problems(row, rec, 1e-8))
    for max_iter, outcome in ((30, 'budget-exhausted'), (31, 'lsmr-stopped'), (119, 'lsmr-stopped'), (120, None)):
        for i in range(args.budget_fits):
            row = run_fit(module, workdir, f'legacy_budget_{max_iter}_{i:02d}', f'legacy_budget_{max_iter}',
                          'cov240k', '1e-8', max_iter, dict(legacy, XHDFE_TEST_CUDA_CG_BREAKDOWN=str(FORCED)))
            rec = row['record']
            if outcome is None:
                p = forced_problems(row, FORCED, [[o] for o in APPROVED]) + accepted_problems(row, rec, '1e-8')
            else:
                p = forced_problems(row, FORCED, [[outcome]]) + refusal_problems(rec, max_iter, max_iter)
            record(row, p + invariant_problems(row, rec, 1e-8))
    polish_iterations = None
    for i in range(args.polish_fits):
        row = run_fit(module, workdir, f'legacy_polish_{i:02d}', 'legacy_polish', 'cov240k_scaled', '1e-8', MAX_ITER,
                      dict(legacy, XHDFE_TEST_CUDA_CG_BREAKDOWN=str(FORCED)))
        rec = row['record']
        record(row, forced_problems(row, FORCED, [['lsmr-polished']]) + accepted_problems(row, rec, '1e-8')
               + invariant_problems(row, rec, 1e-8))
        if row['strict_restart'] and row['strict_restart'][0].get('outcome') == 'lsmr-polished' and polish_iterations is None:
            polish_iterations = int(row['strict_restart'][0]['polish_iterations'])
            lsmr_iterations = int(row['strict_restart'][0]['lsmr_iterations'])
    if args.polish_fits:
        if polish_iterations is None:
            labels.append('NOT_EXERCISED: the legacy strict polish (no lsmr-polished fit to calibrate its exhaustion)')
        else:
            short = FORCED + lsmr_iterations + polish_iterations - 1
            row = run_fit(module, workdir, 'legacy_polish_exhausted', 'legacy_polish_exhausted', 'cov240k_scaled',
                          '1e-8', short, dict(legacy, XHDFE_TEST_CUDA_CG_BREAKDOWN=str(FORCED)))
            rec = row['record']
            record(row, forced_problems(row, FORCED, [['polish-stopped']]) + refusal_problems(rec, short, short)
                   + invariant_problems(row, rec, 1e-8))
    for i in range(args.sequence_fits):
        row = run_fit(module, workdir, f'legacy_B2_{i:02d}', 'legacy_B2', 'cov240k', '1e-8', MAX_ITER,
                      dict(legacy, **reject, XHDFE_TEST_CUDA_CG_BREAKDOWN=str(FORCED)))
        rec = row['record']
        legacy_fired['b2'] += 1 if [b for b in row['breakdowns'] if b['stage'] == 'test-forced'] else 0
        p = forced_problems(row, FORCED, [['lsmr-rejected-by-test-hook']]) + accepted_problems(row, rec, '1e-8')
        if len(row['accel']) != 2:
            p.append(f"{len(row['accel'])} stages (expected the CG stage and the strict repair)")
        record(row, p + invariant_problems(row, rec, 1e-8))
        row = run_fit(module, workdir, f'legacy_B3_{i:02d}', 'legacy_B3', 'pathlike', '1e-8', MAX_ITER,
                      dict(legacy, **reject, XHDFE_TEST_CUDA_CG_BREAKDOWN=SEQUENCE_HOOK))
        rec = row['record']
        cg = [int(c['cg_iterations']) for c in row['cg_end']]
        lsmr = [int(x['iterations']) for x in row['lsmr']]
        p = refusal_problems(rec, sum(cg) + sum(lsmr), MAX_ITER)
        if [s.get('outcome') for s in row['strict_restart']] != ['lsmr-rejected-by-test-hook', 'lsmr-already-consumed']:
            p.append(f"outcomes {[s.get('outcome') for s in row['strict_restart']]}")
        if cg != [SEQUENCE_HOOK_ITERATION] * 2 or len(lsmr) != 1:
            p.append(f'CG stages {cg}, LSMR {lsmr}')
        record(row, p + invariant_problems(row, rec, 1e-8))
        row = run_fit(module, workdir, f'legacy_B4_{i:02d}', 'legacy_B4', 'session', '1e-8', MAX_ITER,
                      dict(legacy, **reject, XHDFE_TEST_CUDA_CG_BREAKDOWN=SEQUENCE_HOOK))
        fits = row['record'].get('fits', [])
        p = []
        if [f.get('refused') for f in fits] != [False, True, False]:
            p.append(f"sequence {[f.get('refused') for f in fits]}")
        else:
            p += refusal_problems(fits[1], max_iter=MAX_ITER)
            if 'did not converge' not in fits[1].get('error', ''):
                p.append('refusal diagnosis: ' + fits[1].get('error', '')[:200])
            if fits[0].get('coef') != fits[2].get('coef') or fits[0].get('resid_sha256') != fits[2].get('resid_sha256'):
                p.append('the third fit differs from the first')
            for f in (fits[0], fits[2]):
                if not (f.get('converged') and f.get('certified') and f.get('gpu_used') and f.get('gpu_status_code') == 1):
                    p.append(f"{f['name']} not converged, certified and on the GPU")
        if any(a.get('honest') == '1' for a in row['accel']):
            p.append('a cold MAP ran')
        record(row, p)
    # Where the legacy entry fired the hook, the in-scope "the hook does not fire" is a note, not a label.
    notes = []
    for messages, fired in ((scope_unfired, legacy_fired['forced']), (scope_b2_unfired, legacy_fired['b2'])):
        for message in messages:
            (notes if fired else labels).append(('' if fired else 'NOT_EXERCISED: ') + message)
    # Package B: cgseed_iv in scope with the hook firing (its CG stage breaks naturally at 7 in comparable).
    for i in range(args.cgseed_fits):
        row = run_fit(module, workdir, f'cgseed_iv_{i:02d}', 'cgseed_iv', 'cgseed_iv', '1e-8', MAX_ITER,
                      {'XHDFE_TEST_CUDA_CG_BREAKDOWN': str(CGSEED_HOOK)})
        rec = row['record']
        p = scope_invariant_problems(row, rec, allow_tail_restart=True)
        forced = [b for b in row['breakdowns'] if b['stage'] == 'test-forced']
        if not forced or forced[0]['iter'] != CGSEED_HOOK:
            p.append(f"forced breakdowns {[(b['stage'], b['iter']) for b in row['breakdowns']]}")
        if [t.get('outcome') for t in row['tail_restart']] != ['restart->device-lsmr']:
            p.append(f"tail restarts {[t.get('outcome') for t in row['tail_restart']]}")
        if rec.get('refused') is not False:
            p.append(f"refused or no result: {str(rec.get('error'))[:200]}")
        else:
            lsmr = sum(int(x.get('iterations', 0)) for x in row['lsmr'])
            if rec['iterations'] != CGSEED_HOOK + lsmr:
                p.append(f"iterations {rec['iterations']} != {CGSEED_HOOK} + LSMR {lsmr}")
            if not (rec['converged'] and rec['certified'] and rec['gpu_used'] and rec['gpu_status_code'] == 1):
                p.append('not converged, certified and on the GPU with status 1')
            if rec.get('group_mean_max') is None or not rec['group_mean_max'] <= TOLERANCES['1e-8']:
                p.append(f"D {rec.get('group_mean_max')} above the tolerance")
            for k, limit in SCOPE_LIMITS.items():
                if not rec.get(k, math.inf) <= limit:
                    p.append(f'{k} {rec.get(k)} above {limit}')
        record(row, p)
    control = None
    if args.reference_module:
        # A reference module without the strict sequence on an entry (13b1: both; b4c5bbd9: the
        # explicit criterion) must accept the cold MAP state outside the envelope there; an entry on
        # which the reference already has the sequence is reported, not graded.
        control = dict(module_sha256=hashlib.sha256(args.reference_module.read_bytes()).hexdigest(), entries={})
        for convergence in ('auto', 'reghdfe'):
            detected = exercised = 0
            for i in range(args.reference_fits):
                row = run_fit(args.reference_module.resolve(), workdir, f'reference_{convergence}_{i:02d}',
                              f'reference_{convergence}', 'cov240k', '1e-8', MAX_ITER,
                              {'XHDFE_TEST_CUDA_CG_BREAKDOWN': str(FORCED)}, convergence)
                rec = row['record']
                row['problems'] = []
                rows.append(row)
                if [b for b in row['breakdowns'] if b['stage'] == 'test-forced']:
                    exercised += 1
                    if rec.get('refused') is False and accepted_problems(row, rec, '1e-8') and \
                            any(a.get('honest') == '1' for a in row['accel']):
                        detected += 1
            control['entries'][convergence] = dict(fits=args.reference_fits, exercised=exercised, detected=detected)
        if not any(e['exercised'] for e in control['entries'].values()):
            labels.append('NEGATIVE_CONTROL_NOT_EXERCISED: the hook never fired in the reference module')
        elif not any(e['detected'] == e['exercised'] > 0 for e in control['entries'].values()):
            failures.append(dict(label='negative_control', problems=[f"no entry detected: {control['entries']}"]))
    stata = None
    if args.stata_package:
        stata = run_stata(args.stata_package, workdir)
        if stata['problems']:
            failures.append(dict(label='stata', problems=stata['problems']))
        labels.extend(stata.get('labels', []))
    summary = {}
    for case in sorted(set(r['case'] for r in rows)):
        sel = [r for r in rows if r['case'] == case]
        summary[case] = dict(fits=len(sel), failed=sum(1 for r in sel if r['problems']),
                             wilson95_failure=wilson(sum(1 for r in sel if r['problems']), len(sel)),
                             iterations=sorted(set(r['record'].get('iterations', r['record'].get('message_iterations'))
                                                   for r in sel if r['record'])),
                             outcomes=sorted(set(s.get('outcome') for r in sel for s in r['strict_restart'])),
                             seconds=sorted(round(r['record'].get('seconds', -1), 2) for r in sel))
    known_defects = [label for label in labels if label.startswith('KNOWN_DEFECT')]
    not_exercised = [label for label in labels if not label.startswith('KNOWN_DEFECT')]
    status = 'FAIL' if failures else ('KNOWN_DEFECT' if known_defects else
                                      ('NOT_EXERCISED' if labels else 'PASS'))
    report = dict(status=status, failures=failures, labels=labels, known_defects=known_defects,
                  not_exercised=not_exercised, notes=notes, summary=summary, negative_control=control,
                  stata=stata, module=str(module), module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                  gpu_apps_at_start=apps, seconds=time.time() - started,
                  envelope_strict=ENVELOPE_STRICT, envelope_source=ENVELOPE_STRICT_SOURCE,
                  envelope_sha256=ENVELOPE_STRICT_SHA256, natural_breakdowns=natural_breakdowns,
                  rows=[{k: v for k, v in r.items() if k != 'stderr_tail'} for r in rows])
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'failures', 'known_defects', 'not_exercised', 'notes',
                                             'summary', 'negative_control')},
                     indent=1)[:20000])
    if cleanup:
        cleanup.cleanup()
    return 1 if failures else (4 if known_defects else (3 if labels else 0))


if __name__ == '__main__':
    sys.exit(main())
