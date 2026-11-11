"""P3-fix: the xhdfe-fast CUDA restart after a broken CG stage, graded against an
independent oracle (offline, CUDA only).

Reproducer (lesson FEREG-10; Verifications/perf_fereg_gap_20261106/RELATORIO_P3_20261108.md):
the public cov240k fixture of cuda_three_fe_covariance_contract.py (240 000 rows,
three FEs, cluster on the worker), xhdfe-fast, CUDA, 16 threads. The primary MAP
stops uncertified after 75 iterations; the CUDA repair is the CG stage, which
converges at iteration 45 in most processes and breaks down at iteration 46 in about
one in six (a lane at roundoff, alpha above the 1/sqrt(eps) ceiling). Up to 2.30.1 the
broken stage restarted with a cold MAP under the Comparable stopping rule; that state
was refused at tol 1e-8 by the 0.5*tol repair gate and returned certified above 1e-8,
43x (SE), 82x (V) and about 790x (residuals) worse than the worst 2.28.1 fit.
The fix (P3-fix (B), fe_absorption_cuda.cu restart_fast_cg_after_breakdown): up to five
cold restarts of the CG stage from the original data with the fit's own xhdfe-fast
options, then (kFastCgRestartLsmrTier, LSMR_TIER below) the device LSMR at the public
tolerance; (C) a refusal message for a converged, publicly certified repair short of the
0.5*tol target. The restart applies only when the public request is xhdfe-fast without
saved FEs (HdfeOptions::fast_intent).

Cases, at tol 1e-8, 1e-8 + 1 ulp and 2e-8, one fresh process per fit:
* natural (--fits, default 40 per tolerance): no hook. Every fit must return (no
  refusal), converged, certified, on the GPU (status 1), inside the envelope, with the
  accounting identity, without the Comparable cold MAP (no cuda_accel_trace line with
  honest=1) and without the three-FE Comparable restart (no cuda_tail_restart line).
  Fits whose CG stage broke down by itself exercise the restart; with none at a
  tolerance, the natural restart route of that tolerance is labelled
  NOT_EXERCISED_ROUTE (missing coverage, not a pass).
* forced (--forced, default 8 per tolerance): XHDFE_TEST_CUDA_CG_BREAKDOWN=30 breaks
  down the first CG stage at its iteration 30 (test hook, inert unless set). The hook
  must fire once; the fit must restart (one cuda_fast_restart line) and meet every
  requirement of the natural case.
* forced_all (--forced-all, default 8 per tolerance): XHDFE_TEST_CUDA_CG_BREAKDOWN=30:all
  breaks down every CG stage: six attempts of 30 iterations. With LSMR_TIER, every fit
  must be an LSMR run (ledger outcome lsmr-converged) that is ACCEPTED, on the GPU, inside
  the envelope; any refusal or other outcome fails (this case, at 40 fits per tolerance,
  decides whether the tier stays: RELATORIO_P3FIX_20261109.md section 4). Without it,
  every fit must be refused honestly (no estimates, "did not converge", ledger outcome
  cg-restarts-exhausted, iterations = primary + 6 x 30).
* budget (--budget-fits, default 2 per case, tol 1e-8; the primary's iteration count P
  is read from the natural fits at 1e-8, or two calibration fits): budget_zero (hook 30,
  max_iter P + 30: no budget left after the broken stage), budget_one (hook 30, P + 31:
  exactly one iteration left), budget_between (hook 30:all, P + 75: exhausted during the
  second cold restart), budget_lsmr (hook 30:all, P + 185: exhausted during the LSMR;
  without LSMR_TIER, budget_exhausted_restarts with P + 190: every restart broken with
  budget left). Every one must be refused with no estimates; the message's iteration
  count is checked against the configuration (max_iter, or P + 180), independently of
  the trace lines, which are checked as well. The number of forced breakdowns is not
  checked here: the ":all" hook cannot fire in an attempt that receives fewer than 30
  iterations (budget_one, the third attempt of budget_between).
  Accepted fits of every case are also checked without the trace: 0 < iterations <=
  max_iter, and more than 30 (forced) or 180 (forced_all) iterations counted.
* boundary (--boundary-fits, default 4 per case, tol 1e-8, hook 10): strict
  (tolerance_mode strict-residual, repaired natively as xhdfe-fast) and savefe (xhdfe-fast
  with retain_fes): outside the restart's shape by public intent, so no cuda_fast_restart
  line may appear (2.30.1 restart kept); their outcome is recorded and an accepted fit is
  measured against the oracle; an accepted state outside the envelope after a broken CG
  stage is a demonstrated error on a route the restart does not cover by public intent:
  label KNOWN_DEFECT_STRICT_RESTART for strict (classified as a demonstrated estimator
  error on 9 nov 2026, correction pending; RELATORIO_P3FIX_20261109.md), and
  OUTSIDE_SHAPE_OUT_OF_ENVELOPE otherwise: neither a PASS nor a failure of the restart,
  they keep the run from PASS. two_fe (worker, firm; convergence reghdfe, which
  opens the CG stage at the gate) and four_fe (plus a fourth FE drawn once per row from a
  fixed generator; the bounded four-FE xhdfe-fast primary opens the CG stage): inside the
  shape, so they must restart and return converged, certified, on the GPU, with the
  accounting identity; their b and residuals against their own oracle are recorded next to
  the same structure's unhooked fits (no 2.28.1 envelope exists for these structures, so
  accuracy is reported, not graded; V not computed: its small-sample factor follows the
  package's FE degrees of freedom). A boundary case whose hook never fired is
  NOT_EXERCISED_ROUTE.
* stata (--stata-package DIR): cuda_fast_restart_stata.do through the plugin of DIR with
  the hook at 30: a CPU fit, the forced restart graded against the oracle (b, V, SE; the
  stored residuals come from refine_stored_residuals, a CUDA MAP re-absorption at the public
  tolerance, and are labelled KNOWN_DEFECT_STATA_RESIDUALS_CUDA_REFINE when outside the
  envelope: 1.69e-3 in 2.28.1, 2.30.1 and the candidate on 9 nov 2026, hypothesis P7), a refusal
  right after it that must leave no e() estimates
  (max_iter P + 5), and a recovery fit; the session's trace must show one forced
  breakdown and the cuda_fast_restart outcomes cg-restart-converged and budget-exhausted.

Accounting identity (75 + CG + restart): iterations = the primary's iterations (first
cuda_accel_trace line) + every CG stage (cuda_cg_end lines) + the LSMR (cuda_fast_restart
lsmr_iterations); the cuda_fast_restart ledger must agree with the stage lines.
Restart statistics: per attempt index, attempts reached and broken (Wilson 95 %),
natural first attempts apart; independence of attempts is an inference these counts
test, not an assumption.

Envelope (brief P3_FIX_REINICIO_FAST_CUDA.md section 3.1): not worse than the worst of
the 52 fits of the published 2.28.1 on the same fixture, route and tolerance (1e-8),
against the P3 oracle; source benchmarks/_out/perf_fereg_gap_20261106/p3_tests/
P3_FINAL_TABLES.json, key accepted_fast_cuda_public_tol.ref_2281 (sha256 ENVELOPE_SHA256):
b_floor max 694.7005304453463 (b_scaled = 1e-9 * b_floor), V_floor max
135.73082083745382 (V_scaled = 1e-8 * V_floor), se_rel max 6.786538739866899e-07,
resid_rel max 5.992717823201028e-07. The same envelope grades 1e-8 + 1 ulp and 2e-8
(2.28.1 was measured at 1e-8 only; a looser tolerance graded against the tighter
envelope can only be stricter). These are the brief's acceptance limits for this test,
not product contracts. All four are graded: the refused 2.30.1 restart state is inside
the b envelope (b_scaled 3.55e-7) and outside the other three.

Oracle: the within transform by anchored sparse dummies (the first level of every FE
after the first anchored; one connected component), SuperLU and four long-double
refinements, as cuda_three_fe_covariance_contract.oracle(), whose b and V (MP80) are
reused for three FEs; the residuals are e = y~ - X~ b with b from an MP80 solve of the
same long-double within data. It must agree with the P3 v3 oracle (independent
construction, certified bounds b 3.8e-14, V 7.5e-12): b, V and ||e|| against the values
embedded below (oracle_v3/oracle_cov240k.json, sha256 V3_SHA256), and the residual
vector against oracle_v3/oracle_cov240k.npz (--v3-npz, sha256 V3_NPZ_SHA256); the run
stops if they differ, and without the npz the vector check is NOT_EXECUTED (label).

Negative control (--reference-module, e.g. the frozen cand_f2b_r2 module): the natural
case at 1e-8 + 1 ulp (--reference-fits, default 40). A module with the 2.30.1 restart
returns its restart state certified there, so the grader must see accepted fits outside
the envelope (NEGATIVE_CONTROL_DETECTED); none (NEGATIVE_CONTROL_NOT_EXERCISED) is missing
coverage and keeps the run from PASS.

Exit code 0 PASS, 1 FAIL, 3 only labels of missing coverage (no failure), 4 a known defect
(KNOWN_DEFECT label, inside the declared band of its known occurrence; missing coverage listed
apart in not_exercised): never a clean PASS.
usage: cuda_fast_restart_contract.py --module PATH.so [--fits 40] [--forced 8]
       [--forced-all 8] [--budget-fits 2] [--boundary-fits 4]
       [--tolerances 1e-8,1e-8+1ulp,2e-8] [--reference-module PATH.so] [--reference-fits 40]
       [--stata-package DIR] [--v3-npz PATH] [--output result.json] [--keep DIR]
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
MAX_ITER = 100000
THREADS = 16
FORCED_ITERATION = 30
BOUNDARY_ITERATION = 10  # early, so a structure whose CG stage converges sooner still breaks
FAST_CG_COLD_RESTARTS = 5  # fe_absorption_cuda.cu kFastCgColdRestarts
LSMR_TIER = True           # fe_absorption_cuda.cu kFastCgRestartLsmrTier (flip both together)
TOL_1E8 = 1e-8
TOL_1E8_ULP = float.fromhex('0x1.5798ee2308c3bp-27')  # the P3 instrument, 1e-8 + 1 ulp
TOLERANCES = {'1e-8': TOL_1E8, '1e-8+1ulp': TOL_1E8_ULP, '2e-8': 2e-8}
FOURTH_FE_SEED, FOURTH_FE_LEVELS = 20261109, 50

ENVELOPE_SOURCE = ('benchmarks/_out/perf_fereg_gap_20261106/p3_tests/P3_FINAL_TABLES.json'
                   ' accepted_fast_cuda_public_tol.ref_2281 (52 fits of 2.28.1, tol 1e-8)')
ENVELOPE_SHA256 = '38dac541742eac4e7fecd40b288fbf34b4095502d47e5a077e2b4c703381b8b2'
ENVELOPE = dict(b_scaled=1e-9 * 694.7005304453463,
                V_scaled=1e-8 * 135.73082083745382,
                se_rel=6.786538739866899e-07,
                resid_rel=5.992717823201028e-07)
B_E_KEYS = ('b_scaled', 'resid_rel')  # graded on the two- and four-FE boundary cases

V3_SOURCE = 'benchmarks/_out/perf_fereg_gap_20261106/p3_tests/oracle_v3/oracle_cov240k.json'
V3_SHA256 = '9bf0930762a03bf058be9fa1db90acf4875d0eaf348c188a422709ff74d1c4bd'
V3_NPZ = ROOT/'benchmarks/_out/perf_fereg_gap_20261106/p3_tests/oracle_v3/oracle_cov240k.npz'
V3_NPZ_SHA256 = '239cf08ebd193d517f8a157f80c4d526948c0699db102668dc77d4aea8b8d739'
V3_B = np.array([-0.0002948626519448652, 0.009769101805411089, -0.00019701207627696257,
                 0.005710104126964007, 2.0163786744917984])
V3_V = np.array([
    [2.3969634728750218e-11, 1.5822660003795225e-11, -1.909539552093635e-12,
     4.779833193650597e-10, -4.953046720981099e-08],
    [1.5822660003795225e-11, 1.7797559908890765e-07, -3.803263089711565e-09,
     -5.131166373516145e-09, -1.5520020610804782e-06],
    [-1.909539552093635e-12, -3.803263089711565e-09, 1.3017226291186248e-10,
     4.2887759983224207e-10, 2.1396459784690703e-08],
    [4.779833193650597e-10, -5.131166373516145e-09, 4.2887759983224207e-10,
     5.980634093061954e-05, -0.0006354416759481467],
    [-4.953046720981099e-08, -1.5520020610804782e-06, 2.1396459784690703e-08,
     -0.0006354416759481467, 0.0068505970854657325]])
V3_NORM_E = 96.89279697938592
# Agreement demanded between this oracle and v3 (both far below every envelope limit).
ORACLE_AGREEMENT = dict(b_scaled=1e-13, V_scaled=1e-11, norm_e_rel=1e-12, e_rel=1e-12)

C_MESSAGE = 'missed the bounded xhdfe-fast repair target'
# Declared band of the known P7 occurrence in the Stata session (reviewer round 24): the values reproduced on
# 9 nov 2026 (stored residuals 1.68840e-3 to 1.68842e-3, D 1.20481e-5 to 1.20483e-5 in 2.28.1, 2.30.1 and the
# candidate) with a +-0.5 % margin; an aggravation beyond it is a new failure.
P7_FAST_BAND = dict(resid=(1.680e-3, 1.697e-3), D=(1.199e-5, 1.211e-5))
CG_END = re.compile(r'^cuda_cg_end (.*)$', re.M)
FAST_RESTART = re.compile(r'^cuda_fast_restart (.*)$', re.M)
STRICT_HANDBACK = re.compile(r'^cuda_strict_handback (.*)$', re.M)
TAIL_RESTART = re.compile(r'^cuda_tail_restart ', re.M)
ACCEL = re.compile(r'^cuda_accel_trace (.*)$', re.M)
LSMR = re.compile(r'^gi_lsmr_solve (.*)$', re.M)
BREAKDOWN = re.compile(r'\[ACCEL cuda CG breakdown stage=(\S+) iter=(\d+) lane=(-?\d+) '
                       r'a=(\S+) b=(\S+) c=([^\]\s]+)\]')


def fields(text):
    """key=value pairs of one trace line, values kept as strings."""
    return dict(item.split('=', 1) for item in text.split() if '=' in item)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k/n
    centre = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.0, centre-half), min(1.0, centre+half)]


def structure(name):
    """The fixture and its FE list for one structure (3fe, 2fe, 4fe)."""
    sys.path.insert(0, str(HERE))
    from cuda_three_fe_covariance_contract import fixture
    y, x, fes = fixture()
    if name == '2fe':
        fes = fes[:2]
    elif name == '4fe':
        occupation = np.random.default_rng(FOURTH_FE_SEED).integers(
            0, FOURTH_FE_LEVELS, size=len(y)).astype(np.int32)
        fes = list(fes)+[occupation]
    return y, x, list(fes)


def within_transform(y, x, fes):
    """[y x] minus its projection on the FE dummies (first level of every FE after the
    first anchored), sparse LU and four long-double refinements, as the contract oracle;
    returns the long-double within data and its relative normal-equation defect."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    n = len(y)
    rows = np.arange(n)
    levels = [int(v.max())+1 for v in fes]
    row_idx, col_idx, offset = [rows], [np.asarray(fes[0])], levels[0]
    for fe, size in zip(fes[1:], levels[1:]):
        keep = fe != 0
        row_idx.append(rows[keep])
        col_idx.append(offset+fe[keep]-1)
        offset += size-1
    r, c = np.concatenate(row_idx), np.concatenate(col_idx)
    d = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, offset)).tocsr()
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
    return within, float(np.max(relative))


def mp_slopes(within):
    """MP80 OLS of y~ on X~ from the long-double within data."""
    import mpmath as mp
    qx, qy = within[:, 1:], within[:, 0]
    gram, h = qx.T@qx, qx.T@qy
    with mp.workdps(80):
        g = mp.matrix([[mp.mpf(str(v)) for v in row] for row in gram])
        beta = g**-1*mp.matrix([mp.mpf(str(v)) for v in h])
        return np.array([np.longdouble(str(v)) for v in beta])


def oracle_with_residuals(y, x, fes):
    """b and V of the contract's MP80 oracle, and the within residuals e = y~ - X~ b."""
    sys.path.insert(0, str(HERE))
    from cuda_three_fe_covariance_contract import oracle as contract_oracle
    b_full, v_full, reference = contract_oracle(y, x, fes)
    within, defect = within_transform(y, x, fes)
    assert defect < 5e-14, 'Independent projection unresolved'
    b = mp_slopes(within)
    e = np.asarray(within[:, 0]-within[:, 1:]@b, dtype=np.float64)
    p = x.shape[1]
    agreement = dict(
        b_scaled=float(np.max(np.abs(b_full-V3_B)/np.maximum(1.0, np.abs(V3_B)))),
        V_scaled=float(np.max(np.abs(v_full-V3_V)/np.sqrt(np.outer(np.diag(V3_V), np.diag(V3_V))))),
        norm_e_rel=float(abs(np.linalg.norm(e)/V3_NORM_E-1.0)),
        b_slopes_vs_contract=float(np.max(np.abs(np.asarray(b, np.float64)-b_full[:p]) /
                                          np.maximum(1.0, np.abs(b_full[:p])))))
    return b_full, v_full, e, dict(reference=reference, projection_defect=defect,
                                   agreement_with_v3=agreement)


def oracle_b_e(y, x, fes):
    """b (slopes and the constant recovered from the means, as the contract) and the
    within residuals, for any number of FEs on one connected component."""
    within, defect = within_transform(y, x, fes)
    assert defect < 5e-14, 'Independent projection unresolved'
    b = mp_slopes(within)
    means = np.column_stack([y, x]).astype(np.longdouble).mean(axis=0)
    constant = means[0]-np.sum(means[1:]*b)
    b_full = np.asarray(np.r_[b, constant], dtype=np.float64)
    e = np.asarray(within[:, 0]-within[:, 1:]@b, dtype=np.float64)
    return b_full, e, defect


def envelope_metrics(b, V, e, ob, oV, oe):
    """The graded quantities against the oracle (oV None: b and residuals only)."""
    b, e = np.asarray(b, np.float64), np.asarray(e, np.float64)
    keys = list(ENVELOPE)+['db_over_se'] if oV is not None else list(B_E_KEYS)
    bad = {k: math.inf for k in keys}
    if b.shape != ob.shape or e.shape != oe.shape:
        return dict(bad, shape_mismatch=True)
    if not (np.isfinite(b).all() and np.isfinite(e).all()):
        return dict(bad, non_finite=True)
    out = dict(b_scaled=float(np.max(np.abs(b-ob)/np.maximum(1.0, np.abs(ob)))),
               resid_rel=float(np.linalg.norm(e-oe)/np.linalg.norm(oe)))
    if oV is None:
        return out
    V = np.asarray(V, np.float64)
    if V.shape != oV.shape or not np.isfinite(V).all():
        return dict(bad, V_mismatch=True)
    ose = np.sqrt(np.diag(oV))
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    out.update(V_scaled=float(np.max(np.abs(V-oV)/np.sqrt(np.outer(np.diag(oV), np.diag(oV))))),
               se_rel=float(np.max(np.abs(se/ose-1.0))),
               db_over_se=float(np.max(np.abs(b-ob)/ose)))
    return out


def inside_envelope(metrics, keys=tuple(ENVELOPE)):
    return all(metrics.get(key, math.inf) <= ENVELOPE[key] for key in keys)


def worker(job_path):
    """One fit in this (fresh) process; the caller sets the environment."""
    import importlib.util
    job = json.loads(Path(job_path).read_text())
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', job['module'])
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    y, X, fes = structure(job['structure'])
    X = np.asfortranarray(X)
    reg = core.HdfeRegressor(se_type='cluster', tol=float.fromhex(job['tol_hex']),
                             max_iter=int(job['max_iter']), num_threads=THREADS,
                             drop_singletons=True, retain_fes=bool(job['retain_fes']),
                             symmetric_sweep=False, absorption_method='auto',
                             tolerance_mode=job['mode'], jacobi_relaxation=0.,
                             convergence=job['convergence'])
    record = dict(job=job)
    start = time.perf_counter()
    try:
        reg.fit(y, X, fes=fes, clusters=fes[0])
    except RuntimeError as error:
        message = str(error)
        it = re.search(r'iterations=(\d+)/(\d+)', message)
        stale = any(np.asarray(getattr(reg, name)).size
                    for name in ('coef_', 'covariance_', 'residuals_'))
        record.update(refused=True, error=message, seconds=time.perf_counter()-start,
                      message_iterations=int(it.group(1)) if it else None,
                      message_max_iter=int(it.group(2)) if it else None,
                      stale_estimates=bool(stale))
        Path(job['out']).write_text(json.dumps(record))
        return
    seconds = time.perf_counter()-start
    o = np.load(job['oracle_npz'])
    record.update(
        refused=False, seconds=seconds, converged=bool(reg.converged_),
        certified=bool(reg.precision_certified_), iterations=int(reg.num_iterations_),
        method=int(reg.absorption_method_used), gpu_used=bool(reg.gpu_used_),
        gpu_status_code=int(reg.gpu_status_code_),
        abs_residual_rel=float(getattr(reg, 'abs_residual_rel_', math.nan)),
        nobs=int(reg.nobs_), num_clusters=int(getattr(reg, 'num_clusters_', -1)))
    record.update(envelope_metrics(reg.coef_, reg.covariance_, reg.residuals_, o['b'],
                                   o['V'] if 'V' in o.files else None, o['e']))
    # The documented strict-residual check in its necessary form on the delivered
    # residual vector (RELATORIO_P3FIX_20261109.md section 10.3, S4): the largest absolute
    # group mean of e over the FEs, and 1 + the l1 norm of the slopes.
    e_fit = np.asarray(reg.residuals_, np.float64).ravel()
    if e_fit.shape == (len(y),) and np.isfinite(e_fit).all():
        means = []
        for fe in fes:
            sums, counts = np.bincount(fe, weights=e_fit), np.bincount(fe)
            means.append(float(np.max(np.abs(sums[counts > 0]/counts[counts > 0]))))
        record['group_mean_max'] = max(means)
    record['b1'] = 1.0+float(np.sum(np.abs(np.asarray(reg.coef_, np.float64).ravel()[:X.shape[1]])))
    Path(job['out']).write_text(json.dumps(record))


def cuda_device_visible():
    smi = shutil.which('nvidia-smi')
    if smi is None:
        return False
    try:
        run = subprocess.run([smi, '-L'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return run.returncode == 0 and 'GPU' in run.stdout


def gpu_apps():
    try:
        return subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                               '--format=csv,noheader'], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'n/a'


def parse_trace(stderr):
    accel = [fields(m.group(1)) for m in ACCEL.finditer(stderr)]
    return dict(
        accel=[dict(honest=a.get('honest'), iterations=a.get('iterations'),
                    converged=a.get('converged'), eff_tol=a.get('eff_tol'),
                    phases=a.get('phases')) for a in accel],
        cg_end=[fields(m.group(1)) for m in CG_END.finditer(stderr)],
        fast_restart=[fields(m.group(1)) for m in FAST_RESTART.finditer(stderr)],
        strict_handback=[fields(m.group(1)) for m in STRICT_HANDBACK.finditer(stderr)],
        tail_restart_lines=len(TAIL_RESTART.findall(stderr)),
        lsmr=[fields(m.group(1)) for m in LSMR.finditer(stderr)],
        breakdowns=[dict(stage=m.group(1), iter=int(m.group(2)), lane=int(m.group(3)),
                         alpha=m.group(4), recent=m.group(5), next_improvement=m.group(6))
                    for m in BREAKDOWN.finditer(stderr)])


def trace_env(hook=None):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('XHDFE_') and k not in ('LD_LIBRARY_PATH', 'OMP_NUM_THREADS')}
    env.update(XHDFE_GPU_BACKEND='cuda', XHDFE_ABSORPTION_CACHE_MODE='off',
               XHDFE_MOBILITY_MODE='off', XHDFE_FE_STRUCTURE_MODE='off',
               XHDFE_GI_LSMR_TRACE='1', XHDFE_ACCEL_DEBUG='1', XHDFE_CUDA_ACCEL_TRACE='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if hook is not None:
        env['XHDFE_TEST_CUDA_CG_BREAKDOWN'] = hook
    return env


def run_fit(module, tol, directory, label, oracle_npz, hook=None, structure_name='3fe',
            mode='xhdfe-fast', retain_fes=False, convergence='auto', max_iter=MAX_ITER):
    """One fit in a fresh process. Never raises: a failed worker is returned with
    worker_failed, its exit code and the tail of its standard error."""
    out = directory/f'{label}.json'
    job_path = directory/f'{label}.job.json'
    job = dict(module=str(module), tol_hex=tol.hex(), out=str(out), oracle_npz=str(oracle_npz),
               structure=structure_name, mode=mode, retain_fes=retain_fes,
               convergence=convergence, max_iter=int(max_iter))
    job_path.write_text(json.dumps(job))
    row = dict(label=label, tol_hex=tol.hex(), hook=hook, structure=structure_name, mode=mode,
               retain_fes=retain_fes, convergence=convergence, max_iter=int(max_iter),
               worker_failed=False, refused=False)
    stderr = ''
    started = time.time()
    try:
        run = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                              str(job_path)], env=trace_env(hook), capture_output=True,
                             text=True, timeout=1800)
        stderr = run.stderr
        row['returncode'] = run.returncode
        if run.returncode != 0:
            row.update(worker_failed=True, error=f'exit code {run.returncode}')
        elif not out.is_file():
            row.update(worker_failed=True, error='no result file')
        else:
            record = json.loads(out.read_text())
            record.pop('job', None)
            row.update(record)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        row.update(worker_failed=True, returncode=None, error=f'{type(error).__name__}: {error}')
    row['wall_seconds'] = time.time()-started
    (directory/f'{label}.err').write_text(stderr)
    row.update(parse_trace(stderr))
    if row['worker_failed']:
        row['stderr_tail'] = stderr[-3000:]
    return row


def route_of(row):
    if row.get('worker_failed'):
        return 'worker-failed'
    if row['fast_restart']:
        return 'restart:'+row['fast_restart'][0].get('outcome', '?')
    if row.get('strict_handback'):
        return 'strict-handback'
    stages = row['cg_end']
    if not stages:
        return 'primary'
    if len(stages) == 1 and stages[0].get('converged') == '1':
        return 'cg'
    return 'other'


def primary_iterations(row):
    """The primary MAP's iterations (its cuda_accel_trace line has phases > 0); 0 when
    the fit opens with the CG stage (explicit update criterion, bounded four-FE
    xhdfe-fast primary), whose line has phases=0."""
    if not row['accel']:
        return None
    first = row['accel'][0]
    return 0 if first.get('phases') == '0' else int(first['iterations'])


def accounting_problems(row):
    """iterations = primary + every CG stage + LSMR; the ledger agrees with the stages."""
    problems = []
    primary = primary_iterations(row)
    if primary is None:
        return ['no cuda_accel_trace line']
    stages = row['cg_end']
    restart = row['fast_restart']
    stage_iterations = [int(c['iterations']) for c in stages]
    lsmr_iterations = 0
    if restart:
        if len(restart) != 1:
            return [f'{len(restart)} cuda_fast_restart lines (expected 1)']
        r = restart[0]
        lsmr_iterations = int(r['lsmr_iterations'])
        attempts = [] if r['restart_iterations'] == '-' else [
            int(v) for v in r['restart_iterations'].split(',')]
        if not stages or int(r['broken_iterations']) != stage_iterations[0]:
            problems.append(f"broken stage {r['broken_iterations']} != first CG stage "
                            f"{stage_iterations[:1]}")
        if attempts != stage_iterations[1:]:
            problems.append(f'restart attempts {attempts} != CG stages {stage_iterations[1:]}')
        if len(attempts) != int(r['cold_restarts']) or len(attempts) > FAST_CG_COLD_RESTARTS:
            problems.append(f"cold_restarts {r['cold_restarts']} inconsistent with {attempts}")
        if int(r['iterations']) != sum(stage_iterations)+lsmr_iterations:
            problems.append(f"ledger iterations {r['iterations']} != CG {sum(stage_iterations)}"
                            f' + LSMR {lsmr_iterations}')
        if int(r['lsmr']) != (1 if row['lsmr'] else 0):
            problems.append(f"ledger lsmr={r['lsmr']} but {len(row['lsmr'])} gi_lsmr_solve lines")
        if int(r['max_iter']) != row['max_iter']-primary:
            problems.append(f"restart max_iter {r['max_iter']} != {row['max_iter']} - primary {primary}")
        broken = stages[:-1] if r.get('outcome') == 'cg-restart-converged' else stages
        for stage in broken:
            if stage.get('converged') != '0' or stage.get('restart_required') != '1':
                problems.append(f'a broken stage reports converged={stage.get("converged")} '
                                f'restart_required={stage.get("restart_required")}')
        if r.get('outcome') == 'cg-restart-converged' and stages[-1].get('converged') != '1':
            problems.append('the last CG stage of a converged restart did not converge')
    elif len(stages) > 1:
        problems.append(f'{len(stages)} CG stages without a cuda_fast_restart line')
    expected = primary+sum(stage_iterations)+lsmr_iterations
    reported = row.get('message_iterations') if row.get('refused') else row.get('iterations')
    if reported != expected:
        problems.append(f'iterations {reported} != primary {primary} + CG {sum(stage_iterations)}'
                        f' + LSMR {lsmr_iterations} = {expected}')
    if expected > row['max_iter']:
        problems.append(f"{expected} iterations exceed max_iter {row['max_iter']}")
    return problems


def accepted_problems(row, keys=tuple(ENVELOPE)):
    """Every requirement of an accepted fit (keys empty: the envelope is not graded)."""
    problems = []
    if not (row.get('converged') and row.get('certified') and row.get('gpu_used') and
            row.get('gpu_status_code') == 1):
        problems.append('not converged, certified and on the GPU with status 1: '
                        f"{row.get('converged')} {row.get('certified')} {row.get('gpu_used')} "
                        f"{row.get('gpu_status_code')}")
    if keys and not inside_envelope(row, keys):
        problems.append('outside the 2.28.1 Fast envelope: '+', '.join(
            f'{k} {row.get(k, math.inf):.3g} (limit {ENVELOPE[k]:.3g})' for k in keys))
    # Trace-free: the public counter within the requested budget.
    if not 0 < (row.get('iterations') or 0) <= row['max_iter']:
        problems.append(f"iterations {row.get('iterations')} outside (0, {row['max_iter']}]")
    if any(a.get('honest') == '1' for a in row['accel']):
        problems.append('a MAP ran under the Comparable stopping rule (honest=1) in a Fast fit')
    if row['tail_restart_lines']:
        problems.append('the three-FE Comparable restart ran in a Fast fit')
    return problems+accounting_problems(row)


def refusal_problems(row, expected_iterations=None):
    """A refusal must leave no estimates and report its iterations honestly."""
    problems = []
    message = row.get('error', '')
    if row.get('stale_estimates') or 'No estimates returned' not in message:
        problems.append('estimates left behind or no refusal notice')
    if expected_iterations is not None and row.get('message_iterations') != expected_iterations:
        problems.append(f"message iterations {row.get('message_iterations')} != "
                        f'{expected_iterations} (from the configuration)')
    if row.get('message_max_iter') != row['max_iter']:
        problems.append(f"message max_iter {row.get('message_max_iter')} != {row['max_iter']}")
    return problems+accounting_problems(row)


def restart_statistics(rows):
    """Attempts reached and broken per attempt index. Natural first attempts apart;
    restart attempts (index >= 2) pooled from natural and forced fits (a forced
    first attempt is not a natural draw; its restarts are)."""
    first = dict(reached=0, broken=0)
    by_index = {}
    for row in rows:
        if row.get('worker_failed') or row['case'] not in ('natural', 'forced'):
            continue
        if row['case'] == 'natural':
            first['reached'] += 1
            if row['fast_restart']:
                first['broken'] += 1
        r = row['fast_restart'][0] if row['fast_restart'] else None
        if r is None:
            continue
        restarts = int(r['cold_restarts'])
        converged = r.get('outcome') == 'cg-restart-converged'
        for index in range(2, restarts+2):
            s = by_index.setdefault(index, dict(reached=0, broken=0))
            s['reached'] += 1
            if index < restarts+1 or not converged:
                s['broken'] += 1
    out = dict(natural_first_attempt=dict(first, wilson95=wilson(first['broken'], first['reached'])))
    out['restart_attempts'] = {str(k): dict(v, wilson95=wilson(v['broken'], v['reached']))
                               for k, v in sorted(by_index.items())}
    return out


def run_stata(package, workdir, oracle, refusal_maxiter):
    """cuda_fast_restart_stata.do through the plugin of package; returns a row."""
    import pandas as pd
    ob, oV, oe = oracle
    folder = workdir/'stata'
    folder.mkdir()
    y, x, fes = structure('3fe')
    data = folder/'cov240k_fast_restart.dta'
    pd.DataFrame(dict(y=y, x1=x[:, 0], x2=x[:, 1], x3=x[:, 2], x4=x[:, 3], worker=fes[0],
                      firm=fes[1], year=fes[2], e_oracle=oe)).to_stata(
        data, write_index=False, version=118)
    oracle_do = folder/'oracle.do'
    with oracle_do.open('x') as stream:
        stream.write('matrix oracle_b = ('+', '.join(format(v, '.17g') for v in ob)+')\n')
        stream.write('matrix oracle_V = ('+' \\ '.join(', '.join(format(v, '.17g') for v in r)
                                                         for r in oV)+')\n')
        for key, name in (('b_scaled', 'envelope_b'), ('V_scaled', 'envelope_V'),
                          ('se_rel', 'envelope_se'), ('resid_rel', 'envelope_resid')):
            stream.write(f'scalar {name} = {ENVELOPE[key]!r}\n')
    script = HERE/'cuda_fast_restart_stata.do'
    arguments = ' '.join(f'"{v}"' for v in (Path(package).resolve(), data, oracle_do,
                                             refusal_maxiter))
    with (folder/'driver.do').open('x') as stream:
        stream.write(f'capture noisily do "{script}" {arguments}\n'
                     'local driver_rc = _rc\ndisplay "DRIVER_RC " `driver_rc\'\nexit, clear\n')
    env = trace_env(f'{FORCED_ITERATION}')
    env.pop('XHDFE_GPU_BACKEND', None)  # the do-file asks for its backend per fit
    env['STATATMP'] = str(folder)
    stata = shutil.which('stata-mp') or '/usr/local/stata/stata-mp'
    with (folder/'process.log').open('x') as log:
        try:
            returncode = subprocess.run([stata, '-q', '-b', 'do', 'driver.do'], cwd=folder,
                                        env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=1800).returncode
        except subprocess.TimeoutExpired:
            returncode = 'timeout'
    text = (folder/'driver.log').read_text(errors='replace') if (folder/'driver.log').is_file() else ''
    trace = (folder/'process.log').read_text(errors='replace')
    parsed = parse_trace(trace)
    forced = [b for b in parsed['breakdowns'] if b['stage'] == 'test-forced']
    outcomes = [r.get('outcome') for r in parsed['fast_restart']]
    lines = text.splitlines()
    problems = []
    if returncode != 0 or not re.search(r'^DRIVER_RC 0\s*$', text, re.M):
        problems.append(f'Stata run failed (returncode {returncode})')
    for marker in ('P3FIX_STATA_A_OK', 'P3FIX_STATA_B_OK', 'P3FIX_STATA_D_OK', 'P3FIX_STATA_PASS'):
        if not any(line.startswith(marker) for line in lines):
            problems.append(f'marker {marker} missing')
    if not any(line.startswith('P3FIX_STATA_C_REFUSED') for line in lines):
        problems.append('the refusal after a successful fit did not happen')
    if f'iterations={refusal_maxiter}/{refusal_maxiter}' not in text:
        problems.append(f'the refusal does not report iterations={refusal_maxiter}/{refusal_maxiter}')
    if len(forced) != 1 or forced[0]['iter'] != FORCED_ITERATION:
        problems.append(f'{len(forced)} forced breakdowns in the session (expected 1)')
    if 'cg-restart-converged' not in outcomes or 'budget-exhausted' not in outcomes:
        problems.append(f'cuda_fast_restart outcomes {outcomes}')
    labels = []
    outside = re.search(r'^P3FIX_STATA_B_RESIDUALS_OUTSIDE residuals=\s*(\S+)\s+D=\s*(\S+)', text, re.M)
    if outside:
        # Only the known occurrence is a known defect (reviewer round 24): the stored residuals and their
        # largest absolute group mean D must lie in the declared band around the values reproduced on
        # 9 nov 2026 in 2.28.1, 2.30.1 and the candidate (1.6884e-3, D 1.2048e-5; the Stata residuals()
        # refinement, not the restart; RELATORIO_P3FIX_20261109.md section 11). Any other excess,
        # including an aggravation, is a new failure.
        resid, dmax = float(outside.group(1)), float(outside.group(2))
        if P7_FAST_BAND['resid'][0] <= resid <= P7_FAST_BAND['resid'][1] and \
                P7_FAST_BAND['D'][0] <= dmax <= P7_FAST_BAND['D'][1]:
            labels.append('KNOWN_DEFECT_STATA_RESIDUALS_CUDA_REFINE (Fast): the stored residuals of the Stata '
                          'session (refine_stored_residuals, a CUDA MAP re-absorption of y - Xb at the public '
                          f'tolerance) are {resid:.6g} from the oracle (envelope {ENVELOPE["resid_rel"]:.3g}), '
                          f'D {dmax:.6g}, inside the band of the known occurrence; reproduced in 2.28.1 and '
                          '2.30.1, present at least since 2.28.1, not the restart; hypothesis P7')
        else:
            problems.append(f'stored residuals {resid:.6g} (D {dmax:.6g}) outside the envelope and outside the band '
                            f"of the known P7 occurrence {P7_FAST_BAND}: a new failure")
    return dict(case='stata', label='stata', returncode=returncode, problems=problems, labels=labels,
                outcomes=outcomes, forced_breakdowns=len(forced),
                result_lines=[line for line in lines if line.startswith('P3FIX_STATA')])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
        return 0
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--fits', type=int, default=40)
    parser.add_argument('--forced', type=int, default=8)
    parser.add_argument('--forced-all', type=int, default=8)
    parser.add_argument('--budget-fits', type=int, default=2)
    parser.add_argument('--boundary-fits', type=int, default=4)
    parser.add_argument('--tolerances', default=','.join(TOLERANCES))
    parser.add_argument('--reference-module', type=Path)
    parser.add_argument('--reference-fits', type=int, default=40)
    parser.add_argument('--stata-package', type=Path)
    parser.add_argument('--v3-npz', type=Path, default=V3_NPZ)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--keep', type=Path, help='keep every fit record and trace here')
    args = parser.parse_args()
    if TOL_1E8_ULP != math.nextafter(TOL_1E8, 1.0):
        raise SystemExit('1e-8 + 1 ulp constant inconsistent')
    module = args.module.resolve()
    if not cuda_device_visible():
        report = dict(status='SKIP', reason='no visible CUDA device')
        if args.output:
            args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))
        return 0
    tolerances = [name for name in args.tolerances.split(',') if name]
    unknown = [name for name in tolerances if name not in TOLERANCES]
    if unknown:
        raise SystemExit(f'unknown tolerances {unknown}; choose from {list(TOLERANCES)}')
    labels = []
    oracle_started = time.time()
    y, x, fes = structure('3fe')
    ob, oV, oe, oracle_info = oracle_with_residuals(y, x, fes)
    agreement = oracle_info['agreement_with_v3']
    if args.v3_npz.is_file():
        digest = hashlib.sha256(args.v3_npz.read_bytes()).hexdigest()
        if digest != V3_NPZ_SHA256:
            raise SystemExit(f'{args.v3_npz}: sha256 {digest} != {V3_NPZ_SHA256}')
        e_v3 = np.load(args.v3_npz)['e']
        agreement['e_rel'] = float(np.linalg.norm(oe-e_v3)/np.linalg.norm(e_v3))
    else:
        labels.append(f'NOT_EXECUTED: residual vector against the v3 oracle ({args.v3_npz} absent)')
    failed = {k: agreement[k] for k, v in ORACLE_AGREEMENT.items() if k in agreement and not agreement[k] <= v}
    if failed:
        raise SystemExit(f'oracle disagrees with the P3 v3 oracle: {failed} (all: {agreement})')
    boundary_oracles = {}
    if args.boundary_fits > 0:
        for name in ('2fe', '4fe'):
            yb, xb, fb = structure(name)
            b_full, e, defect = oracle_b_e(yb, xb, fb)
            boundary_oracles[name] = (b_full, e)
            oracle_info[f'projection_defect_{name}'] = defect
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=False)
        workdir, cleanup = args.keep, None
    else:
        cleanup = tempfile.TemporaryDirectory()
        workdir = Path(cleanup.name)
    oracle_npz = workdir/'oracle_cov240k.npz'
    np.savez(oracle_npz, b=ob, V=oV, e=oe)
    boundary_npz = {}
    for name, (b_full, e) in boundary_oracles.items():
        boundary_npz[name] = workdir/f'oracle_{name}.npz'
        np.savez(boundary_npz[name], b=b_full, e=e)
    failures, rows = [], []
    summary = dict(oracle_seconds=time.time()-oracle_started, oracle=oracle_info,
                   envelope=ENVELOPE, envelope_source=ENVELOPE_SOURCE,
                   envelope_sha256=ENVELOPE_SHA256, v3_source=V3_SOURCE, v3_sha256=V3_SHA256,
                   v3_npz=str(args.v3_npz), v3_npz_sha256=V3_NPZ_SHA256, lsmr_tier=LSMR_TIER,
                   module=str(module), module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                   gpu_apps_at_start=gpu_apps(), tolerances={}, cases={})

    def fail(row, message):
        failures.append(f"{row['label']}: {message}")
        row.setdefault('problems', []).append(message)

    def record(row, case, tolerance_name):
        row['case'], row['tolerance'] = case, tolerance_name
        rows.append(row)
        if row['worker_failed']:
            fail(row, f"worker failed ({row.get('error')})")
        return not row['worker_failed']

    def forced_count(row):
        return [b for b in row['breakdowns'] if b['stage'] == 'test-forced']

    for name in tolerances:
        tol = TOLERANCES[name]
        tag = name.replace('+', 'p').replace('-', 'm')
        stats = {case: dict(fits=0, refused=0, accepted=0, routes={}, seconds=[], max_metrics={})
                 for case in ('natural', 'forced', 'forced_all')}
        stats['natural']['breakdowns'] = 0

        def note(case, row):
            s = stats[case]
            s['fits'] += 1
            route = route_of(row)
            s['routes'][route] = s['routes'].get(route, 0)+1
            if row.get('refused'):
                s['refused'] += 1
            elif not row.get('worker_failed'):
                s['accepted'] += 1
                s['seconds'].append(row.get('seconds'))
                for key in list(ENVELOPE)+['db_over_se']:
                    s['max_metrics'][key] = max(s['max_metrics'].get(key, 0.0),
                                                row.get(key, math.inf))

        for i in range(args.fits):
            row = run_fit(module, tol, workdir, f'natural_{tag}_{i+1:02d}', oracle_npz)
            if not record(row, 'natural', name):
                note('natural', row)
                continue
            note('natural', row)
            stats['natural']['breakdowns'] += 1 if row['fast_restart'] else 0
            if forced_count(row):
                fail(row, 'the test hook fired without XHDFE_TEST_CUDA_CG_BREAKDOWN')
            if row['refused']:
                fail(row, f"refused: {row.get('error', '')[:300]}")
                continue
            for problem in accepted_problems(row):
                fail(row, problem)
        if args.fits and stats['natural']['breakdowns'] == 0:
            labels.append(f'NOT_EXERCISED_ROUTE: natural restart at {name} '
                          f'(no CG stage broke down in {args.fits} fits)')

        for i in range(args.forced):
            row = run_fit(module, tol, workdir, f'forced_{tag}_{i+1:02d}', oracle_npz,
                          f'{FORCED_ITERATION}')
            note('forced', row)
            if not record(row, 'forced', name):
                continue
            forced = forced_count(row)
            if len(forced) != 1 or forced[0]['iter'] != FORCED_ITERATION:
                fail(row, f'{len(forced)} forced breakdowns (expected 1 at {FORCED_ITERATION})')
            if len(row['fast_restart']) != 1:
                fail(row, f"{len(row['fast_restart'])} cuda_fast_restart lines (expected 1)")
            if row['refused']:
                fail(row, f"refused: {row.get('error', '')[:300]}")
                continue
            # Trace-free: the broken stage and at least one restart iteration are counted.
            if (row.get('iterations') or 0) <= FORCED_ITERATION:
                fail(row, f"iterations {row.get('iterations')} <= {FORCED_ITERATION}")
            for problem in accepted_problems(row):
                fail(row, problem)

        for i in range(args.forced_all):
            row = run_fit(module, tol, workdir, f'forced_all_{tag}_{i+1:02d}', oracle_npz,
                          f'{FORCED_ITERATION}:all')
            note('forced_all', row)
            if not record(row, 'forced_all', name):
                continue
            forced = forced_count(row)
            if len(forced) != 1+FAST_CG_COLD_RESTARTS or any(b['iter'] != FORCED_ITERATION
                                                              for b in forced):
                fail(row, f'{len(forced)} forced breakdowns (expected {1+FAST_CG_COLD_RESTARTS})')
            outcome = row['fast_restart'][0].get('outcome') if len(row['fast_restart']) == 1 else None
            if LSMR_TIER:
                # Decision rule: an accepted LSMR run on the GPU inside the envelope, always.
                if outcome != 'lsmr-converged':
                    fail(row, f'outcome {outcome} (expected lsmr-converged)')
                if row['refused']:
                    fail(row, f"refused: {row.get('error', '')[:300]}")
                    continue
                # Trace-free: six broken stages of 30 iterations and the LSMR are counted.
                if (row.get('iterations') or 0) <= (1+FAST_CG_COLD_RESTARTS)*FORCED_ITERATION:
                    fail(row, f"iterations {row.get('iterations')} <= "
                              f'{(1+FAST_CG_COLD_RESTARTS)*FORCED_ITERATION}')
                for problem in accepted_problems(row):
                    fail(row, problem)
            else:
                if outcome != 'cg-restarts-exhausted' or not row['refused'] or row['lsmr']:
                    fail(row, f"outcome {outcome}, refused {row['refused']}, "
                              f"{len(row['lsmr'])} LSMR lines (expected a refusal after "
                              'cg-restarts-exhausted, no LSMR)')
                    continue
                if 'did not converge' not in row.get('error', ''):
                    fail(row, 'the refusal does not say "did not converge"')
                primary = primary_iterations(row)
                expected = (primary+(1+FAST_CG_COLD_RESTARTS)*FORCED_ITERATION
                            if primary is not None else None)
                for problem in refusal_problems(row, expected):
                    fail(row, problem)
        summary['tolerances'][name] = dict(tol_hex=tol.hex(), **stats)

    # Budget boundaries (tol 1e-8): the primary's iterations from the natural fits at 1e-8.
    if args.budget_fits > 0:
        observed = [primary_iterations(r) for r in rows
                    if r['case'] == 'natural' and r['tolerance'] == '1e-8' and
                    not r['worker_failed'] and primary_iterations(r) is not None]
        for i in range(2 if not observed else 0):
            row = run_fit(module, TOL_1E8, workdir, f'calibration_{i+1}', oracle_npz)
            record(row, 'calibration', '1e-8')
            if primary_iterations(row) is not None:
                observed.append(primary_iterations(row))
        primaries = sorted(set(observed))
        summary['cases']['budget_primary_iterations'] = primaries
        if len(primaries) != 1:
            failures.append(f'budget: the primary iterations vary or are missing ({primaries}); '
                            'budget cases not run')
        else:
            P = primaries[0]
            budget_cases = [('budget_zero', f'{FORCED_ITERATION}', P+FORCED_ITERATION, 0, 0),
                            ('budget_one', f'{FORCED_ITERATION}', P+FORCED_ITERATION+1, 1, 0)]
            budget_cases.append(('budget_between', f'{FORCED_ITERATION}:all',
                                 P+2*FORCED_ITERATION+15, 2, 0))
            if LSMR_TIER:
                budget_cases.append(('budget_lsmr', f'{FORCED_ITERATION}:all',
                                     P+(1+FAST_CG_COLD_RESTARTS)*FORCED_ITERATION+5,
                                     FAST_CG_COLD_RESTARTS, 1))
            else:
                budget_cases.append(('budget_exhausted_restarts', f'{FORCED_ITERATION}:all',
                                     P+(1+FAST_CG_COLD_RESTARTS)*FORCED_ITERATION+10,
                                     FAST_CG_COLD_RESTARTS, 0))
            for case, hook, max_iter, restarts, lsmr in budget_cases:
                case_stats = dict(fits=0, refused=0, max_iter=max_iter)
                for i in range(args.budget_fits):
                    row = run_fit(module, TOL_1E8, workdir, f'{case}_{i+1:02d}', oracle_npz,
                                  hook, max_iter=max_iter)
                    case_stats['fits'] += 1
                    if not record(row, case, '1e-8'):
                        continue
                    if primary_iterations(row) != P:
                        fail(row, f'primary iterations {primary_iterations(row)} != {P}')
                        continue
                    if not row['refused']:
                        fail(row, 'accepted although the budget cannot complete the fit')
                        continue
                    case_stats['refused'] += 1
                    exhausted_restarts = case == 'budget_exhausted_restarts'
                    expected = (P+(1+FAST_CG_COLD_RESTARTS)*FORCED_ITERATION
                                if exhausted_restarts else max_iter)
                    for problem in refusal_problems(row, expected):
                        fail(row, problem)
                    if not exhausted_restarts and 'The iteration limit was reached' not in row.get('error', ''):
                        fail(row, 'the refusal does not report the iteration limit')
                    r = row['fast_restart'][0] if len(row['fast_restart']) == 1 else {}
                    expected_outcome = ('cg-restarts-exhausted' if exhausted_restarts else
                                        'lsmr-stopped' if lsmr else 'budget-exhausted')
                    if (r.get('outcome') != expected_outcome or int(r.get('cold_restarts', -1)) != restarts
                            or int(r.get('lsmr', -1)) != lsmr):
                        fail(row, f'ledger {r} (expected outcome {expected_outcome}, '
                                  f'{restarts} cold restarts, lsmr {lsmr})')
                summary['cases'][case] = case_stats

    # Boundaries of the restart's shape (tol 1e-8, hook at BOUNDARY_ITERATION).
    if args.boundary_fits > 0:
        boundary_cases = [('strict', '3fe', 'strict-residual', False, 'auto', False),
                          ('savefe', '3fe', 'xhdfe-fast', True, 'auto', False),
                          ('two_fe', '2fe', 'xhdfe-fast', False, 'reghdfe', True),
                          ('four_fe', '4fe', 'xhdfe-fast', False, 'auto', True)]
        for case, struct, mode, retain, convergence, in_shape in boundary_cases:
            case_stats = dict(fits=0, hook_fired=0, restarted=0, refused=0, accepted=0,
                              routes={}, in_shape=in_shape, max_metrics={},
                              outside_envelope=0)
            npz = boundary_npz.get(struct, oracle_npz)
            keys = tuple(ENVELOPE) if struct == '3fe' else B_E_KEYS
            if in_shape:
                # The same structure without the hook: the accuracy of its own route, the
                # reference the restarted fits are reported against (no 2.28.1 envelope
                # exists for these structures, so accuracy is recorded, not graded).
                reference = dict(fits=0, refused=0, accepted=0, routes={}, max_metrics={})
                for i in range(args.boundary_fits):
                    ref = run_fit(module, TOL_1E8, workdir, f'{case}_unhooked_{i+1:02d}', npz,
                                  None, struct, mode, retain, convergence)
                    reference['fits'] += 1
                    if not record(ref, case+'_unhooked', '1e-8'):
                        continue
                    route = route_of(ref)
                    reference['routes'][route] = reference['routes'].get(route, 0)+1
                    if ref['refused']:
                        reference['refused'] += 1
                        fail(ref, f"refused: {ref.get('error', '')[:300]}")
                        continue
                    reference['accepted'] += 1
                    for key in keys:
                        reference['max_metrics'][key] = max(reference['max_metrics'].get(key, 0.0),
                                                            ref.get(key, math.inf))
                    for problem in accepted_problems(ref, ()):
                        fail(ref, problem)
                case_stats['unhooked_reference'] = reference
            for i in range(args.boundary_fits):
                row = run_fit(module, TOL_1E8, workdir, f'{case}_{i+1:02d}', npz,
                              f'{BOUNDARY_ITERATION}', struct, mode, retain, convergence)
                case_stats['fits'] += 1
                if not record(row, case, '1e-8'):
                    continue
                route = route_of(row)
                case_stats['routes'][route] = case_stats['routes'].get(route, 0)+1
                fired = len(forced_count(row)) == 1
                case_stats['hook_fired'] += 1 if fired else 0
                case_stats['restarted'] += 1 if row['fast_restart'] else 0
                if row['refused']:
                    case_stats['refused'] += 1
                else:
                    case_stats['accepted'] += 1
                    for key in keys:
                        case_stats['max_metrics'][key] = max(case_stats['max_metrics'].get(key, 0.0),
                                                             row.get(key, math.inf))
                    # Counted only where a CG stage broke down (the hook fired): elsewhere
                    # the number describes the route without any restart (e.g. savefe's
                    # primary), for which no 2.28.1 envelope was measured.
                    if fired and not inside_envelope(row, keys):
                        case_stats['outside_envelope'] += 1
                if not in_shape:
                    if row['fast_restart']:
                        fail(row, 'outside the shape by public intent, yet cuda_fast_restart ran')
                    if row['refused']:
                        if row.get('stale_estimates') or 'No estimates returned' not in row.get('error', ''):
                            fail(row, 'a refusal left estimates or no refusal notice')
                    elif not (row.get('converged') and row.get('certified') and row.get('gpu_used')):
                        fail(row, 'accepted without convergence, certificate or GPU')
                    continue
                if not fired:
                    continue
                if len(row['fast_restart']) != 1:
                    fail(row, f"{len(row['fast_restart'])} cuda_fast_restart lines (expected 1)")
                if row['refused']:
                    fail(row, f"refused: {row.get('error', '')[:300]}")
                    continue
                if (row.get('iterations') or 0) <= BOUNDARY_ITERATION:
                    fail(row, f"iterations {row.get('iterations')} <= {BOUNDARY_ITERATION}")
                for problem in accepted_problems(row, ()):
                    fail(row, problem)
            if case_stats['hook_fired'] == 0:
                labels.append(f'NOT_EXERCISED_ROUTE: boundary {case} (the hook never fired in '
                              f"{case_stats['fits']} fits)")
            if not in_shape and case_stats['outside_envelope']:
                # A demonstrated estimator error on a route the restart does not cover by public
                # intent (RELATORIO_P3FIX_20261109.md sections 2.5 and 7): neither a PASS nor a
                # failure of the P3-fix restart; it keeps the run from PASS until it is decided.
                tag = ('KNOWN_DEFECT_STRICT_RESTART' if case == 'strict'
                       else 'OUTSIDE_SHAPE_OUT_OF_ENVELOPE')
                labels.append(f"{tag}: {case}: {case_stats['outside_envelope']} of "
                              f"{case_stats['accepted']} returned fits with a broken CG stage lie "
                              'outside the 2.28.1 envelope against the oracle ('+', '.join(
                                  f"{k} {case_stats['max_metrics'].get(k, math.inf):.3g}"
                                  for k in keys)+')')
            summary['cases'][case] = case_stats

    summary['restart_statistics'] = restart_statistics(rows)

    if args.reference_module:
        reference = args.reference_module.resolve()
        control = dict(module=str(reference),
                       module_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                       fits=0, refused=0, accepted=0, outside_envelope=0, max_metrics={},
                       worker_failed=0)
        for i in range(args.reference_fits):
            row = run_fit(reference, TOL_1E8_ULP, workdir, f'reference_1e8p1ulp_{i+1:02d}',
                          oracle_npz)
            row['case'], row['tolerance'] = 'reference_negative_control', '1e-8+1ulp'
            rows.append(row)
            control['fits'] += 1
            if row['worker_failed']:
                control['worker_failed'] += 1
                continue
            if row['refused']:
                control['refused'] += 1
                continue
            control['accepted'] += 1
            for key in ENVELOPE:
                control['max_metrics'][key] = max(control['max_metrics'].get(key, 0.0),
                                                  row.get(key, math.inf))
            if not inside_envelope(row):
                control['outside_envelope'] += 1
        control['label'] = ('NEGATIVE_CONTROL_DETECTED' if control['outside_envelope']
                            else 'NEGATIVE_CONTROL_NOT_EXERCISED')
        if not control['outside_envelope']:
            labels.append(f"NEGATIVE_CONTROL_NOT_EXERCISED: no accepted reference fit outside "
                          f"the envelope in {control['fits']} fits")
        summary['negative_control'] = control

    if args.stata_package:
        observed = [primary_iterations(r) for r in rows
                    if r.get('case') == 'natural' and r.get('tolerance') == '1e-8' and
                    not r['worker_failed'] and primary_iterations(r) is not None]
        P = max(set(observed), key=observed.count) if observed else 75
        stata_row = run_stata(args.stata_package, workdir, (ob, oV, oe), P+5)
        summary['stata'] = stata_row
        for problem in stata_row['problems']:
            failures.append(f'stata: {problem}')
        labels.extend(stata_row.get('labels', []))

    known_defects = [label for label in labels if label.startswith('KNOWN_DEFECT')]
    not_exercised = [label for label in labels if not label.startswith('KNOWN_DEFECT')]
    status = 'FAIL' if failures else ('KNOWN_DEFECT' if known_defects else
                                      ('NOT_EXERCISED_ROUTE' if labels else 'PASS'))
    report = dict(status=status, failures=failures, labels=labels, known_defects=known_defects,
                  not_exercised=not_exercised, summary=summary,
                  rows=[{k: v for k, v in r.items() if k != 'stderr_tail' or r['worker_failed']}
                        for r in rows])
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text+'\n')
    print(json.dumps(dict(status=status, failures=failures[:20], known_defects=known_defects,
                          not_exercised=not_exercised,
                          tolerances={k: {c: {kk: vv for kk, vv in v[c].items() if kk != 'seconds'}
                                          for c in ('natural', 'forced', 'forced_all')}
                                      for k, v in summary['tolerances'].items()},
                          cases=summary['cases'], restart_statistics=summary['restart_statistics'],
                          negative_control=summary.get('negative_control'),
                          stata=summary.get('stata')), indent=1, default=str))
    if cleanup is not None:
        cleanup.cleanup()
    return {'PASS': 0, 'NOT_EXERCISED_ROUTE': 3, 'KNOWN_DEFECT': 4}.get(status, 1)


if __name__ == '__main__':
    raise SystemExit(main())
