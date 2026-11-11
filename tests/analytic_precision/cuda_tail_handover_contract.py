"""F2-A: the three-FE Comparable CUDA completion may be handed to the host MLSMR.

The CUDA Comparable path with three ordinary fixed effects completes the
projection after its sweeps (the "tail"). A probe of the device LSMR predicts
its remaining work; when it is long (work >= 8192 device RHS-iterations, with
budget left), the fit has at least 750 000 rows, the public request was Auto
and the CPU's own route selector (select_auto_mlsmr_method, evaluated once per
fit) picks MLSMR, the host MLSMR completes the projection from the same IT
state. XHDFE_CUDA_TAIL_HANDOVER=0 keeps the 2.30.1 device tail; =force hands
over after the first device iteration (tests and diagnostics only).

Every probe line (one per checkpoint) is parsed and its decision rebuilt from
its own inputs: q from norm_ar and norm_ar_half, m0 and m1 from tol and test2,
m_rem and work from m1, the work threshold, the budget guard, and the selector
labels (not-asked, auto-mlsmr, auto-sweep = the selector's sweep vote, n/a =
not applicable with its reason, exception). The selector is evaluated at most
once per fit.

Fixtures:
* pyfixest's public "difficult" design (benchmark DGP; model
  y ~ x1..x10 | indiv_id + firm_id + year of benchmark_difficult_1m_3fe.py),
  10 M rows, read from /srv/projetos/P008/dataset/pyfixest with its absolute
  path; nothing is copied. Its cases are SKIPPED when the file is absent.
* the synthetic 240 000-row three-FE fixture of
  cuda_three_fe_covariance_contract.py, built in memory, with its independent
  MP80 oracle (explicit dummies, wide refinement).
The whole test is SKIPPED (not failed) when no CUDA device is visible.

Every fit runs in a fresh process (cache, mobility and FE-structure caches off,
XHDFE_GI_LSMR_TRACE=1 to read the hand-over trace). A worker that fails (an
exception other than a refusal, a non-zero exit code, a missing result) is
recorded as a failure of its case with its exit code and the tail of its trace,
and the independent cases still run; a refusal (RuntimeError, no estimates) is
captured by the worker and is a failure unless the case expects it. A
comparison whose reference fit is missing is recorded as NOT_EXECUTED, never as
a pass. Floors against the CPU fit at the same tolerance (graded in every case,
the 2.30.1 device routes included): |db_j| <= 1e-9 max(1,|b_j|),
|dV_jk| <= 1e-8 sqrt(V_jj V_kk). On the 10 M design the absorbed residuals are
compared with the CPU fit's (relative L2): a hard check, <= 1e-8 (a bound of
this harness, not a product contract), on the host routes, which promise the
CPU's projection (the F2-A tail and the F2-B restart, cuda_tail host-mlsmr);
on the device routes (the 2.30.1 device tail and device restart, and a CG stage
that converged) the value is recorded without PASS/FAIL, labelled
DEVICE_ROUTE_UNGRADED_RESIDUALS (coordinator, 08nov2026: missing coverage, the
device routes deviate from the CPU by 3-4e-7, brief P2/R4).

F2-A cases, tol 1e-8:
* cpu_d10m3: CPU Comparable, the reference; cuda_tail none.
* cuda_d10m3: CUDA default, n >= 750 000 and the selector applicable;
  host-mlsmr, converged, certified, gpu_used, status 1, absorption method 6
  (MLSMR); the probe hands over with the selector auto-mlsmr, evaluated once;
  iterations = IT + probe + host <= max_iter; floors and residuals.
* cov240k_default (threads 1 and 16): n < 750 000, the F2 guard: never a
  hand-over without force (a decided probe reports n/a, below-nmin); oracle
  floors.
* cov240k_force (threads 1 and 16): host-mlsmr, method 6, gpu_used, status 1,
  converged, certified, the accounting identity, and the MP80 oracle floors
  (b_scaled <= 1e-9, V_scaled <= 1e-8, the contract's metric).
* regressions, unchanged behaviour: cov240k_2fe (two FE, CUDA04) and
  cpu_cov240k: no completion, cuda_tail none; cov240k_fast (xhdfe-fast): no
  completion, none, converged on the GPU, and (P3-fix) b, V, standard errors and
  residuals against the oracle within the 2.28.1 Fast envelope of
  cuda_fast_restart_contract.py (never the Comparable floors); cov240k_explicit
  (absorptionmethod gauss-seidel): never a hand-over; oracle floors.
* with --slow (about 95 s each), the 2.30.1 device tail: cuda_d10m3_off
  (switch 0: device-lsmr, method 5, no probe); cuda_d10m3_sweepvote (negative
  control, XHDFE_AUTO_MLSMR_RHO_THRESHOLD=1 in this test only: long
  prediction, one evaluation, sweep vote with a finite rho < 1, keep-selector,
  device-lsmr, method 5); cuda_d10m3_explicit (absorptionmethod gauss-seidel:
  n/a, not-auto, device-lsmr). Floors and residuals graded (round 2).

F2-B (the cold restart after a broken CG stage), 10 M difficult design at tol
1e-10, where the IT sweeps hand over to the CG stage (iteration cap). The CG
stage either converges or stops (slow-mode-stalled) and the fit restarts cold
from the original data; the route is not deterministic on CUDA. The restart is
probed like the tail (probe lines with stage=restart); one cuda_it_to_cg, one
cuda_cg_end and one cuda_tail_restart line per fit give the stages.
Convention under the default switch, in the three-FE family: the budget is
cumulative in every stage (the CG stage receives max_iter - IT, the restart
max_iter - IT - CG, the host what the probe left) and iterations = IT + CG +
device probe + host (device restart: + device LSMR) after a restart; a CG
stage that converges counts only its own iterations (2.30.1, CPU and CUDA).
XHDFE_CUDA_TAIL_HANDOVER=0 keeps the 2.30.1 budget and accounting
(iterations = CG + device LSMR, budget max_iter - CG, the CG stage not
charged with the sweeps).
Route-dependent variants run up to RESTART_ATTEMPTS = 5 fits each and stop at
the restarts they need (default: 3; the others: 1). Every fit, whatever its
route, must be converged, certified, on the GPU, within the floors and the
residual bound; the F2-B contract (route, cuda_tail, method, ledger, probe
lines rebuilt, one selector evaluation) is checked on the fits that restarted;
the route of every fit is recorded. A variant with no restart in its 5 fits is
labelled NOT_EXERCISED_ROUTE (missing coverage, not a pass).
* cpu_d10m3_1e10: CPU Comparable at tol 1e-10, the reference.
* cuda_d10m3_1e10 (default): restart->host-mlsmr, host-mlsmr (2), method 6,
  the selector auto-mlsmr applicable and evaluated once.
* cuda_d10m3_1e10_budget_small (max_iter 1500): the CG stage is limited to
  1500 - IT and stops there; the restart has no budget: refused, no
  estimates, the message reports iterations=1500/1500, the ledger
  IT + CG = 1500 (restart->budget-exhausted).
* cuda_d10m3_1e10_budget_small_off (max_iter 1500, switch 0): the 2.30.1
  budget (CG stage up to 1500, not charged with the sweeps), refused with
  iterations=1500/1500.
* cuda_d10m3_1e10_budget_enough (max_iter 6000): the CG stage gets 6000 - IT;
  on a restart, the probe and the host fit in what is left, iterations
  <= 6000.
* with --slow: cuda_d10m3_1e10_off (switch 0: the 2.30.1 device restart, no
  probe, none, method 5, iterations = CG + device LSMR); _force (force:
  restart->host-mlsmr after one device iteration, method 6); _sweepvote
  (XHDFE_AUTO_MLSMR_RHO_THRESHOLD=1: long prediction, one evaluation, sweep
  vote, keep-selector, device restart, none, method 5); _explicit
  (absorptionmethod gauss-seidel: n/a, not-auto, device restart).

Exit code 0 PASS, 1 FAIL, 3 only NOT_EXERCISED_ROUTE labels (no failure).
Usage: cuda_tail_handover_contract.py --module PATH.so [--data PARQUET]
                                      [--output result.json] [--slow]
The module must be a CUDA-enabled build (it also runs the CPU fits).
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = '/srv/projetos/P008/dataset/pyfixest/benchmark_difficult_n10000000_k10.parquet'
TOL = 1e-8
MAX_ITER = 100000
B_FLOOR = 1e-9
V_FLOOR = 1e-8
RESID_FLOOR = 1e-8
METHOD_LSMR = 5
METHOD_MLSMR = 6
THETA = 8192.0
TOL_RESTART = 1e-10
RESTART_REPEATS = 3
RESTART_ATTEMPTS = 5
BUDGET_SMALL = 1500
BUDGET_ENOUGH = 6000
HANDOVER = re.compile(r'^cuda_tail_handover (.*)$', re.M)
PROBE = re.compile(r'^cuda_tail_probe (.*)$', re.M)
RESTART = re.compile(r'^cuda_tail_restart (.*)$', re.M)
IT_TO_CG = re.compile(r'^cuda_it_to_cg (.*)$', re.M)
CG_END = re.compile(r'^cuda_cg_end (.*)$', re.M)


def fields(text):
    """key=value pairs of one trace line, values kept as strings."""
    return dict(item.split('=', 1) for item in text.split())


def worker(module, source, threads, out, mode='reghdfe-comparable', method='auto',
           tol=repr(TOL), max_iter=str(MAX_ITER), resid_mode='none', resid_path='-'):
    import numpy as np
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    if source.startswith('cov240k'):
        sys.path.insert(0, str(HERE))
        from cuda_three_fe_covariance_contract import fixture
        y, X, fes = fixture()
        X = np.asfortranarray(X)
        clusters = fes[0]
        if source == 'cov240k_2fe':
            fes = fes[:2]
    else:
        import pandas as pd
        xs = ['x%d' % i for i in range(1, 11)]
        frame = pd.read_parquet(source, columns=['y', *xs, 'indiv_id', 'firm_id', 'year'])
        fac = lambda s: pd.factorize(s, sort=True)[0].astype(np.int32)
        y = frame['y'].to_numpy(np.float64)
        X = np.asfortranarray(frame[xs].to_numpy(np.float64))
        fes = [fac(frame[c]) for c in ('indiv_id', 'firm_id', 'year')]
        clusters = fac(frame['indiv_id'])
        del frame
    reg = core.HdfeRegressor(se_type='cluster', tol=float(tol), max_iter=int(max_iter),
                             num_threads=int(threads), drop_singletons=True,
                             retain_fes=False, symmetric_sweep=False,
                             absorption_method=method,
                             tolerance_mode=mode, jacobi_relaxation=0.)
    try:
        reg.fit(y, X, fes=fes, clusters=clusters)
    except RuntimeError as error:
        # A refusal: no estimates may be left behind. The caller decides whether
        # the case expects it.
        stale = any(np.asarray(getattr(reg, name)).size
                    for name in ('coef_', 'covariance_', 'residuals_'))
        Path(out).write_text(json.dumps(dict(refused=True, error=str(error),
                                             stale_estimates=bool(stale))))
        return
    result = dict(
        refused=False,
        b=[float(v) for v in np.asarray(reg.coef_)],
        V=np.asarray(reg.covariance_).tolist(),
        converged=bool(reg.converged_), certified=bool(reg.precision_certified_),
        iterations=int(reg.num_iterations_), method=int(reg.absorption_method_used),
        gpu_used=bool(reg.gpu_used_), gpu_status_code=int(reg.gpu_status_code_),
        cuda_tail=str(reg.cuda_tail_), cuda_tail_code=int(reg.cuda_tail_code_),
        threads_used=int(reg.threads_used_))
    residuals = np.asarray(reg.residuals_, dtype=np.float64)
    if resid_mode == 'save':
        np.save(resid_path, residuals)
    elif resid_mode == 'compare':
        if Path(resid_path).is_file():
            reference = np.load(resid_path)
            if reference.shape == residuals.shape and np.linalg.norm(reference) > 0:
                result['residuals_rel'] = float(np.linalg.norm(residuals - reference) /
                                                np.linalg.norm(reference))
            else:
                result['residuals_rel'] = 'SHAPE_MISMATCH'
        else:
            result['residuals_rel'] = 'NOT_EXECUTED'
    Path(out).write_text(json.dumps(result))


def cuda_device_visible():
    smi = shutil.which('nvidia-smi')
    if smi is None:
        return False
    try:
        run = subprocess.run([smi, '-L'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return run.returncode == 0 and 'GPU' in run.stdout


def fit(module, backend, source, threads, directory, label, extra=None,
        mode='reghdfe-comparable', method='auto', tol=TOL, max_iter=MAX_ITER,
        resid_mode='none', resid_path='-'):
    """One fit in a fresh process. Never raises: a failed worker is returned
    with worker_failed, its exit code and the tail of its standard error."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('XHDFE_')}
    env.update(XHDFE_GPU_BACKEND=backend, XHDFE_ABSORPTION_CACHE_MODE='off',
               XHDFE_MOBILITY_MODE='off', XHDFE_FE_STRUCTURE_MODE='off',
               XHDFE_GI_LSMR_TRACE='1')
    env.update(extra or {})
    out = directory / f'{label}.json'
    result = dict(label=label, tol=float(tol), max_iter=int(max_iter), worker_failed=False,
                  refused=False, extra=dict(extra or {}), method_requested=method,
                  mode_requested=mode)
    stderr = ''
    try:
        run = subprocess.run([sys.executable, '-B', __file__, '--worker', str(module), source,
                              str(threads), str(out), mode, method, repr(float(tol)),
                              str(int(max_iter)), resid_mode, str(resid_path)],
                             env=env, capture_output=True, text=True, timeout=1800)
        stderr = run.stderr
        result['returncode'] = run.returncode
        if run.returncode != 0:
            result.update(worker_failed=True, error=f'exit code {run.returncode}')
        elif not out.is_file():
            result.update(worker_failed=True, error='no result file')
        else:
            result.update(json.loads(out.read_text()))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        result.update(worker_failed=True, returncode=None,
                      error=f'{type(error).__name__}: {error}')
    result.update(
        handovers=[fields(m.group(1)) for m in HANDOVER.finditer(stderr)],
        probes=[fields(m.group(1)) for m in PROBE.finditer(stderr)],
        restarts=[fields(m.group(1)) for m in RESTART.finditer(stderr)],
        it_to_cg=[fields(m.group(1)) for m in IT_TO_CG.finditer(stderr)],
        cg_end=[fields(m.group(1)) for m in CG_END.finditer(stderr)])
    if result['worker_failed']:
        result['stderr_tail'] = stderr[-3000:]
    return result


def usable(result):
    return not result.get('worker_failed') and not result.get('refused')


def floors(reference, candidate):
    import numpy as np
    b0, b = np.asarray(reference['b']), np.asarray(candidate['b'])
    V0, V = np.asarray(reference['V']), np.asarray(candidate['V'])
    scale = np.sqrt(np.outer(np.diag(V0), np.diag(V0)))
    return (float(np.max(np.abs(b - b0) / (B_FLOOR * np.maximum(1.0, np.abs(b0))))),
            float(np.max(np.abs(V - V0) / (V_FLOOR * scale))))


def accounting(result):
    """IT + device probe + host iterations, from the hand-over trace."""
    if len(result['handovers']) != 1:
        return False
    h = result['handovers'][0]
    return (h['outcome'] == 'host-mlsmr' and
            result['iterations'] == (int(h['completed']) + int(h['device_iterations']) +
                                     int(h['host_iterations'])) and
            int(h['host_budget']) >= int(h['host_iterations']) and
            0 < result['iterations'] <= MAX_ITER)


def route_of(result):
    """restart (a cuda_tail_restart line), cg_converged (the CG stage converged,
    no restart), it_tail (no CG stage: the IT sweeps converged), or other."""
    if result['restarts']:
        return 'restart'
    if result['cg_end'] and all(c.get('converged') == '1' for c in result['cg_end']):
        return 'cg_converged'
    if not result['cg_end'] and not result['it_to_cg']:
        return 'it_tail'
    return 'other'


def restart_problems(result, mode, outcome, max_iter=MAX_ITER):
    """F2-B: rebuild the restart ledger from the stage lines.

    mode 'adaptive'/'force': the CG stage was given max_iter - IT,
    counted_before = IT + CG, the restart budget is max_iter - IT - CG and every
    stage adds to `iterations`; mode 'off': the 2.30.1 budget and accounting
    (the CG stage given max_iter, counted_before = CG, IT sweeps not counted)."""
    problems = []
    if len(result['restarts']) != 1:
        return [f"{len(result['restarts'])} cuda_tail_restart lines (expected 1)"]
    r = result['restarts'][0]
    if r['mode'] != mode or r['outcome'] != outcome:
        problems.append(f"mode {r['mode']} outcome {r['outcome']} (expected {mode}, {outcome})")
    if len(result['it_to_cg']) != 1 or len(result['cg_end']) != 1:
        return problems + [f"{len(result['it_to_cg'])} cuda_it_to_cg and "
                           f"{len(result['cg_end'])} cuda_cg_end lines (expected 1 and 1)"]
    i, c = result['it_to_cg'][0], result['cg_end'][0]
    it, cg = int(r['it_iterations']), int(r['cg_iterations'])
    device, host = int(r['device_iterations']), int(r['host_iterations'])
    counted, budget = int(r['counted_before']), int(r['budget'])
    if not (it == int(i['it_iterations']) == int(i['uncounted']) > 0 and
            r['it_reason'] == i['reason'] and int(i['max_iter']) == max_iter):
        problems.append(f'IT sweeps {it} / reason {r["it_reason"]} differ from the hand-off line {i}')
    cg_budget = max_iter if mode == 'off' else max_iter - it
    if not (int(i['cg_budget']) == cg_budget == int(c['cg_budget'])):
        problems.append(f"CG budget {i.get('cg_budget')}/{c.get('cg_budget')} != {cg_budget}")
    if not (c['entry'] == 'sweeps' and c['converged'] == '0' and c['restart_required'] == '1' and
            cg == int(c['iterations']) == int(c['cg_iterations']) and 0 <= cg <= cg_budget):
        problems.append(f'CG stage {cg} differs from the stage line {c} or exceeds its budget')
    if int(r['max_iter']) != max_iter:
        problems.append(f"max_iter {r['max_iter']} != {max_iter}")
    expected_counted = cg if mode == 'off' else it + cg
    if counted != expected_counted or budget != max(0, max_iter - counted):
        problems.append(f'counted_before {counted} / budget {budget}: expected '
                        f'{expected_counted} / {max(0, max_iter - expected_counted)}')
    if outcome == 'restart->host-mlsmr':
        if not (int(r['host_budget']) == budget - device and
                0 < host <= int(r['host_budget']) and r['host_converged'] == '1' and
                int(r['iterations']) == counted + device + host):
            problems.append(f'host ledger: device {device} host {host}/{r["host_budget"]} '
                            f'iterations {r["iterations"]} != {counted} + {device} + {host}')
        restart_probes = [p for p in result['probes'] if p.get('stage') == 'restart']
        if not restart_probes or any(int(p['budget']) != budget for p in restart_probes):
            problems.append('restart probe lines missing or with a budget other than the ledger')
    elif outcome == 'restart->budget-exhausted':
        if not (budget == 0 and counted == max_iter and int(r['iterations']) == max_iter and
                device == 0 and host == 0 and not result['probes']):
            problems.append(f'budget-exhausted ledger: counted {counted} budget {budget} '
                            f'iterations {r["iterations"]} device {device} host {host}')
    else:
        if not (host == 0 and int(r['iterations']) == counted + device and device > 0):
            problems.append(f'device ledger: iterations {r["iterations"]} != {counted} + {device}')
    if outcome != 'restart->budget-exhausted' and (
            int(r['iterations']) != result.get('iterations') or
            not 0 < result.get('iterations', 0) <= max_iter):
        problems.append(f"reported iterations {result.get('iterations')} != ledger "
                        f"{r['iterations']} or outside (0, {max_iter}]")
    if mode == 'off' and result['probes']:
        problems.append('switch 0 must not probe')
    if any(p.get('stage') != 'restart' for p in result['probes']) or result['handovers']:
        problems.append('a restart fit must not run the IT tail')
    return problems


def probe_problems(result):
    """Rebuild every probe decision from its own printed inputs."""
    import math
    problems = []
    for p in result['probes']:
        k, rhs, count = int(p['k']), int(p['rhs']), int(p['rhs_count'])
        budget, left = int(p['budget']), int(p['budget_left'])
        decision, selector, reason = p['decision'], p['selector'], p['reason']
        tag = f"rhs {rhs} k {k} stage {p.get('stage')}"
        if p.get('stage') not in ('tail', 'restart'):
            problems.append(f'{tag}: unknown stage')
        if left != budget - k:
            problems.append(f'{tag}: budget_left != budget - k')
        if float(p['threshold']) != THETA:
            problems.append(f'{tag}: threshold {p["threshold"]} != {THETA}')
        if int(p['evaluations']) > 1:
            problems.append(f'{tag}: the selector was evaluated {p["evaluations"]} times')
        if selector == 'forced':
            if reason != 'forced' or decision != ('handover' if left >= 1 else 'keep-budget'):
                problems.append(f'{tag}: forced decision {decision} inconsistent')
            continue
        norm_ar, half = float(p['norm_ar']), float(p['norm_ar_half'])
        test2, tol = float(p['test2']), float(p['tol'])
        q = (norm_ar / half) ** (2.0 / k)
        if not math.isclose(q, float(p['q']), rel_tol=1e-12, abs_tol=0.0):
            problems.append(f'{tag}: q {p["q"]} != {q!r}')
        m1 = float(p['m1'])
        if 0.0 < q < 1.0:
            gap = math.log(tol / test2)
            m0 = gap / math.log(q)
            m1_hat = (gap + 0.5 * math.log((k + m0) / k)) / math.log(q)
            if not math.isclose(m1_hat, m1, rel_tol=1e-9, abs_tol=1e-9):
                problems.append(f'{tag}: m1 {m1} != {m1_hat!r}')
            m_rem = max(1.0, math.ceil(m1))
            work = m_rem + (count - rhs - 1) * (k + m_rem)
        else:
            work = math.inf
        if float(p['work']) != work:
            problems.append(f'{tag}: work {p["work"]} != {work!r}')
        if work < THETA:
            expected = 'continue'
        elif left < 1:
            expected = 'keep-budget'
        else:
            expected = 'handover' if selector == 'auto-mlsmr' else 'keep-selector'
        if decision != expected:
            problems.append(f'{tag}: decision {decision} != {expected}')
        if expected in ('continue', 'keep-budget') and selector != 'not-asked':
            problems.append(f'{tag}: the selector was asked before a long prediction')
        if expected in ('handover', 'keep-selector'):
            asked = {'auto-mlsmr': ('voted', '1'), 'auto-sweep': ('voted', '1'),
                     'exception': ('exception', '0')}
            if selector in asked:
                ok = (reason, p['evaluated']) == asked[selector]
            else:
                ok = (selector == 'n/a' and p['evaluated'] == '0' and
                      reason in ('not-auto', 'no-selector', 'below-nmin'))
            if not ok:
                problems.append(f'{tag}: selector {selector} reason {reason} '
                                f'evaluated {p["evaluated"]} inconsistent')
    return problems


def decision_of(result):
    decided = [p for p in result['probes'] if p['decision'] != 'continue']
    return decided[0] if decided else None


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(*sys.argv[2:12])
        return 0
    import tempfile
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--module', type=Path, required=True)
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--output', type=Path)
    p.add_argument('--slow', action='store_true')
    args = p.parse_args()
    module = args.module.resolve()
    if not cuda_device_visible():
        report = dict(status='SKIP', reason='no visible CUDA device')
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        return 0
    sys.path.insert(0, str(HERE))
    from cuda_three_fe_covariance_contract import fixture, oracle, metrics
    y, x, fes = fixture()
    oracle_b, oracle_v, _ = oracle(y, x, fes)
    have_data = Path(args.data).is_file()
    failures, rows, skipped, not_exercised, variants = [], [], [], [], []
    device_residuals = []  # DEVICE_ROUTE_UNGRADED_RESIDUALS, recorded only

    def check(condition, label, message):
        if not condition:
            failures.append(f'{label}: {message}')

    def check_probes(r):
        for problem in probe_problems(r):
            failures.append(f"{r['label']}: probe {problem}")

    def check_long(r, selector, reason, decision):
        """One decided checkpoint: a long prediction and the expected selector."""
        d = decision_of(r)
        check(d is not None and float(d['work']) >= THETA and int(d['budget_left']) >= 1 and
              d['selector'] == selector and d['reason'] == reason and
              d['decision'] == decision, r['label'],
              f'expected a long prediction, selector {selector} ({reason}), {decision}; '
              f'got {d}')
        return d

    def run(*fit_args, **fit_kwargs):
        """fit() and record the row; a failed worker or an unexpected refusal is
        a failure of this case (the caller skips its other checks)."""
        r = fit(*fit_args, **fit_kwargs)
        rows.append(r)
        if r['worker_failed']:
            failures.append(f"{r['label']}: worker failed ({r.get('error')}, exit code "
                            f"{r.get('returncode')}); trace tail recorded in the row")
        elif r['refused']:
            failures.append(f"{r['label']}: refused: {r.get('error', '')[:300]}")
        return r

    def grade(r, reference, resid=True):
        """Floors (and residuals) against the CPU fit; NOT_EXECUTED without it."""
        if not usable(r):
            return
        if reference is None or not usable(reference):
            r['floors'] = 'NOT_EXECUTED'
            failures.append(f"{r['label']}: CPU reference missing; floors not executed")
        else:
            r['b_ratio'], r['V_ratio'] = floors(reference, r)
            r['floors'] = 'GRADED'
            check(r['b_ratio'] <= 1 and r['V_ratio'] <= 1, r['label'],
                  f"CPU floors b {r['b_ratio']:.3g} V {r['V_ratio']:.3g}")
        if resid:
            rel = r.get('residuals_rel', 'NOT_EXECUTED')
            if r.get('cuda_tail') == 'host-mlsmr':
                # Host routes reproduce the CPU route: hard check (harness bound).
                r['residuals_grade'] = 'GRADED'
                if not isinstance(rel, float):
                    failures.append(f"{r['label']}: residuals comparison {rel}")
                else:
                    check(rel <= RESID_FLOOR, r['label'], f'residuals relative L2 {rel:.3g}')
            else:
                # Device routes: recorded, no PASS/FAIL (missing coverage, P2/R4).
                r['residuals_grade'] = 'DEVICE_ROUTE_UNGRADED_RESIDUALS'
                device_residuals.append(dict(label=r['label'], cuda_tail=r.get('cuda_tail'),
                                             method=r.get('method'), route=r.get('route', ''),
                                             residuals_rel=rel))

    def oracle_floors(r):
        if not usable(r):
            return None
        m = metrics(np.asarray(r['b']), np.asarray(r['V']), oracle_b, oracle_v)
        r.update(m)
        return m

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Synthetic 240 000-row fixture.
        cpu = run(module, 'cpu', 'cov240k', 16, tmp, 'cpu_cov240k')
        if usable(cpu):
            check(cpu['cuda_tail'] == 'none' and cpu['cuda_tail_code'] == 0 and
                  not cpu['gpu_used'], cpu['label'], 'a CPU fit must report cuda_tail none')
        two = run(module, 'cuda', 'cov240k_2fe', 16, tmp, 'cov240k_2fe')
        if usable(two):
            check(two['cuda_tail'] == 'none' and two['cuda_tail_code'] == 0 and
                  not two['handovers'] and not two['probes'], two['label'],
                  'a two-FE fit runs no completion')
        fast_resid = tmp / 'cov240k_fast.resid.npy'
        fast = run(module, 'cuda', 'cov240k', 16, tmp, 'cov240k_fast', mode='xhdfe-fast',
                   resid_mode='save', resid_path=fast_resid)
        if usable(fast):
            check(fast['cuda_tail'] == 'none' and fast['cuda_tail_code'] == 0 and
                  not fast['handovers'] and not fast['probes'], fast['label'],
                  'xhdfe-fast runs no completion')
            check(fast['gpu_used'] and fast['converged'], fast['label'],
                  'no real GPU use or not converged')
            # P3-fix (brief section 3.1): the Fast route against the oracle, within
            # the 2.28.1 Fast envelope (b, V, standard errors and residuals).
            from cuda_fast_restart_contract import (ENVELOPE, envelope_metrics,
                                                    inside_envelope, oracle_with_residuals)
            fast_b, fast_v, fast_e, _ = oracle_with_residuals(y, x, fes)
            m = envelope_metrics(fast['b'], fast['V'], np.load(fast_resid),
                                 fast_b, fast_v, fast_e)
            fast['fast_envelope'] = m
            check(inside_envelope(m), fast['label'], 'outside the 2.28.1 Fast envelope: ' +
                  ', '.join(f'{k} {m[k]:.3g} (limit {v:.3g})' for k, v in ENVELOPE.items()))
        explicit = run(module, 'cuda', 'cov240k', 16, tmp, 'cov240k_explicit',
                       method='gauss-seidel')
        m = oracle_floors(explicit)
        if m is not None:
            check(explicit['cuda_tail'] in ('device-lsmr', 'none') and not explicit['handovers'],
                  explicit['label'], 'an explicit method must never hand over')
            check(explicit['gpu_used'] and explicit['converged'] and explicit['certified'] and
                  m['b_scaled'] <= 1e-9 and m['V_scaled'] <= 1e-8, explicit['label'],
                  f"explicit: oracle b {m['b_scaled']:.3g} V {m['V_scaled']:.3g}")
            check_probes(explicit)
        for threads in (1, 16):
            r = run(module, 'cuda', 'cov240k', threads, tmp, f'cov240k_default_t{threads}')
            m = oracle_floors(r)
            if m is not None:
                check(r['cuda_tail'] in ('device-lsmr', 'none') and not r['handovers'],
                      r['label'], 'n < 750 000 must never hand over without force')
                check(all(p['selector'] in ('not-asked', 'n/a') and
                          p['reason'] in ('-', 'below-nmin') for p in r['probes']),
                      r['label'], 'below the F2 guard the selector must not be asked')
                check_probes(r)
                check(r['gpu_used'] and r['gpu_status_code'] == 1 and r['converged'] and
                      r['certified'], r['label'], 'no real GPU use or not converged/certified')
                check(m['b_scaled'] <= 1e-9 and m['V_scaled'] <= 1e-8, r['label'],
                      f"oracle b {m['b_scaled']:.3g} V {m['V_scaled']:.3g}")
            f = run(module, 'cuda', 'cov240k', threads, tmp, f'cov240k_force_t{threads}',
                    dict(XHDFE_CUDA_TAIL_HANDOVER='force'))
            m = oracle_floors(f)
            if m is not None:
                check(f['cuda_tail'] == 'host-mlsmr' and f['cuda_tail_code'] == 2 and
                      f['method'] == METHOD_MLSMR, f['label'],
                      'force must report host-mlsmr, method 6')
                check(f['gpu_used'] and f['gpu_status_code'] == 1 and f['converged'] and
                      f['certified'], f['label'], 'no real GPU use or not converged/certified')
                check(accounting(f), f['label'], 'iterations != IT + probe + host')
                check_probes(f)
                check(m['b_scaled'] <= 1e-9 and m['V_scaled'] <= 1e-8, f['label'],
                      f"oracle b {m['b_scaled']:.3g} V {m['V_scaled']:.3g}")
        # Public 10 M difficult fixture.
        if not have_data:
            skipped.append('10 M difficult cases: fixture absent ' + args.data)
        else:
            resid8 = tmp / 'cpu_d10m3.resid.npy'
            ref = run(module, 'cpu', args.data, 16, tmp, 'cpu_d10m3',
                      resid_mode='save', resid_path=resid8)
            if usable(ref):
                check(ref['converged'] and ref['certified'] and not ref['gpu_used'] and
                      ref['cuda_tail'] == 'none', ref['label'],
                      'CPU reference not converged/certified')
            compare8 = dict(resid_mode='compare', resid_path=resid8)
            c = run(module, 'cuda', args.data, 16, tmp, 'cuda_d10m3', **compare8)
            if usable(c):
                check(c['cuda_tail'] == 'host-mlsmr' and c['cuda_tail_code'] == 2 and
                      c['method'] == METHOD_MLSMR, c['label'], 'expected host-mlsmr, method 6')
                check(c['gpu_used'] and c['gpu_status_code'] == 1 and c['converged'] and
                      c['certified'], c['label'], 'no real GPU use or not converged/certified')
                d = check_long(c, 'auto-mlsmr', 'voted', 'handover')
                check(d is not None and d['applicable'] == '1' and d['evaluated'] == '1' and
                      d['evaluations'] == '1', c['label'],
                      'the selector must be applicable and evaluated exactly once')
                check_probes(c)
                check(accounting(c), c['label'], 'iterations != IT + probe + host')
                grade(c, ref)
            if args.slow:
                # The 2.30.1 device tail (floors graded since round 2).
                o = run(module, 'cuda', args.data, 16, tmp, 'cuda_d10m3_off',
                        dict(XHDFE_CUDA_TAIL_HANDOVER='0'), **compare8)
                if usable(o):
                    check(o['cuda_tail'] == 'device-lsmr' and o['method'] == METHOD_LSMR and
                          not o['probes'] and not o['handovers'], o['label'],
                          'switch 0 must keep the 2.30.1 device tail without probe')
                    check(o['gpu_used'] and o['converged'] and o['certified'], o['label'],
                          'no real GPU use or not converged/certified')
                    grade(o, ref)
                s = run(module, 'cuda', args.data, 16, tmp, 'cuda_d10m3_sweepvote',
                        dict(XHDFE_AUTO_MLSMR_RHO_THRESHOLD='1'), **compare8)
                if usable(s):
                    d = check_long(s, 'auto-sweep', 'voted', 'keep-selector')
                    check(d is not None and d['evaluated'] == '1' and d['evaluations'] == '1' and
                          0.0 < float(d['rho']) < 1.0, s['label'],
                          'the sweep vote needs one evaluation and a finite rho < 1')
                    check(s['cuda_tail'] == 'device-lsmr' and s['method'] == METHOD_LSMR and
                          not s['handovers'], s['label'], 'a sweep vote must keep the device tail')
                    check(s['gpu_used'] and s['converged'] and s['certified'], s['label'],
                          'no real GPU use or not converged/certified')
                    check_probes(s)
                    grade(s, ref)
                e = run(module, 'cuda', args.data, 16, tmp, 'cuda_d10m3_explicit',
                        method='gauss-seidel', **compare8)
                if usable(e):
                    check_long(e, 'n/a', 'not-auto', 'keep-selector')
                    check(e['cuda_tail'] == 'device-lsmr' and not e['handovers'], e['label'],
                          'an explicit method must keep the device tail')
                    check(e['gpu_used'] and e['converged'] and e['certified'], e['label'],
                          'no real GPU use or not converged/certified')
                    check_probes(e)
                    grade(e, ref)
            # F2-B: the cold restart after a broken CG stage, tol 1e-10.
            resid10 = tmp / 'cpu_d10m3_1e10.resid.npy'
            ref10 = run(module, 'cpu', args.data, 16, tmp, 'cpu_d10m3_1e10', tol=TOL_RESTART,
                        resid_mode='save', resid_path=resid10)
            if usable(ref10):
                check(ref10['converged'] and ref10['certified'] and not ref10['gpu_used'] and
                      ref10['cuda_tail'] == 'none', ref10['label'],
                      'CPU reference at 1e-10 not converged/certified')
            compare10 = dict(resid_mode='compare', resid_path=resid10)

            def restart_variant(name, mode, outcome, method_used, tail, restart_checks,
                                extra=None, method='auto', want=1, max_iter=MAX_ITER):
                """Up to RESTART_ATTEMPTS fits; stop after `want` restarts. Every fit:
                converged, certified, GPU, floors, residuals, route recorded; the
                F2-B contract on the fits that restarted."""
                fits, restarted = [], []
                for attempt in range(1, RESTART_ATTEMPTS + 1):
                    r = run(module, 'cuda', args.data, 16, tmp, f'{name}_a{attempt}', extra,
                            method=method, tol=TOL_RESTART, max_iter=max_iter, **compare10)
                    r.update(variant=name, attempt=attempt)
                    fits.append(r)
                    if not usable(r):
                        r['route'] = 'failed'
                        continue
                    r['route'] = route_of(r)
                    check(r['gpu_used'] and r['gpu_status_code'] == 1 and r['converged'] and
                          r['certified'] and 0 < r['iterations'] <= max_iter, r['label'],
                          'no real GPU use, not converged/certified or iterations out of budget')
                    grade(r, ref10)
                    check_probes(r)
                    if r['route'] == 'restart':
                        restarted.append(r)
                        r['ledger'] = r['restarts'][0]
                        for problem in restart_problems(r, mode, outcome, max_iter):
                            failures.append(f"{r['label']}: restart {problem}")
                        check(r['cuda_tail'] == tail and r['method'] == method_used,
                              r['label'], f"expected cuda_tail {tail} and method "
                              f"{method_used}, got {r['cuda_tail']} and {r['method']}")
                        restart_checks(r)
                        if len(restarted) >= want:
                            break
                    elif r['route'] == 'cg_converged':
                        check(r['cuda_tail'] == 'none' and not r['probes'] and
                              not r['handovers'], r['label'],
                              'a CG-converged fit runs no completion and no restart')
                    else:
                        check(False, r['label'], f"unexpected route {r['route']}")
                summary = dict(variant=name, attempts=len(fits), restarts=len(restarted),
                               routes=[r['route'] for r in fits], want=want)
                if not restarted:
                    summary['label'] = 'NOT_EXERCISED_ROUTE'
                    not_exercised.append(name)
                variants.append(summary)

            def default_checks(r):
                d = check_long(r, 'auto-mlsmr', 'voted', 'handover')
                check(d is not None and d.get('stage') == 'restart' and
                      d['applicable'] == '1' and d['evaluated'] == '1' and
                      d['evaluations'] == '1', r['label'],
                      'the restart probe must ask the applicable selector exactly once')

            restart_variant('cuda_d10m3_1e10', 'adaptive', 'restart->host-mlsmr', METHOD_MLSMR,
                            'host-mlsmr', default_checks, want=RESTART_REPEATS)
            # Cumulative budget (round 2): the CG stage gets what the sweeps left.
            for label, extra in (('cuda_d10m3_1e10_budget_small', None),
                                 ('cuda_d10m3_1e10_budget_small_off',
                                  dict(XHDFE_CUDA_TAIL_HANDOVER='0'))):
                b = run(module, 'cuda', args.data, 16, tmp, label, extra, tol=TOL_RESTART,
                        max_iter=BUDGET_SMALL)
                if b['worker_failed']:
                    continue
                if b['refused']:
                    failures.remove(f"{label}: refused: {b.get('error', '')[:300]}")
                check(b['refused'] and not b.get('stale_estimates', True), label,
                      'expected a refusal with no estimates left')
                check(f'iterations={BUDGET_SMALL}/{BUDGET_SMALL}' in b.get('error', '') and
                      'did not converge' in b.get('error', ''), label,
                      f"the refusal must report iterations={BUDGET_SMALL}/{BUDGET_SMALL}: "
                      f"{b.get('error', '')[:200]}")
                mode = 'off' if extra else 'adaptive'
                outcome = 'restart->budget-exhausted'
                if mode == 'adaptive':
                    for problem in restart_problems(b, mode, outcome, BUDGET_SMALL):
                        failures.append(f'{label}: restart {problem}')
                    check(len(b['it_to_cg']) == 1 and len(b['cg_end']) == 1 and
                          int(b['it_to_cg'][0]['it_iterations']) +
                          int(b['cg_end'][0]['cg_iterations']) == BUDGET_SMALL, label,
                          'IT + CG must equal max_iter exactly')
                else:
                    # 2.30.1: the CG stage is not charged with the sweeps; it spends
                    # max_iter alone and the restart refuses before any work.
                    check(len(b['it_to_cg']) == 1 and len(b['cg_end']) == 1 and
                          int(b['it_to_cg'][0]['cg_budget']) == BUDGET_SMALL and
                          int(b['cg_end'][0]['cg_iterations']) == BUDGET_SMALL and
                          not b['restarts'] and not b['probes'], label,
                          'switch 0 must keep the 2.30.1 budget (CG stage up to max_iter)')

            def budget_checks(r):
                default_checks(r)
                led = r['restarts'][0]
                check(int(led['budget']) == BUDGET_ENOUGH - int(led['it_iterations']) -
                      int(led['cg_iterations']) and r['iterations'] <= BUDGET_ENOUGH, r['label'],
                      f'the restart must fit in what is left of {BUDGET_ENOUGH}')

            restart_variant('cuda_d10m3_1e10_budget_enough', 'adaptive', 'restart->host-mlsmr',
                            METHOD_MLSMR, 'host-mlsmr', budget_checks, max_iter=BUDGET_ENOUGH)
            if args.slow:
                restart_variant('cuda_d10m3_1e10_off', 'off', 'restart->device-lsmr',
                                METHOD_LSMR, 'none', lambda r: None,
                                extra=dict(XHDFE_CUDA_TAIL_HANDOVER='0'))

                def force_checks(r):
                    check(len(r['probes']) == 1 and r['probes'][0]['selector'] == 'forced' and
                          r['probes'][0]['k'] == '1' and r['probes'][0]['decision'] == 'handover',
                          r['label'], 'force must hand the restart over after one device iteration')

                restart_variant('cuda_d10m3_1e10_force', 'force', 'restart->host-mlsmr',
                                METHOD_MLSMR, 'host-mlsmr', force_checks,
                                extra=dict(XHDFE_CUDA_TAIL_HANDOVER='force'))

                def sweep_checks(r):
                    d = check_long(r, 'auto-sweep', 'voted', 'keep-selector')
                    check(d is not None and d.get('stage') == 'restart' and
                          d['evaluated'] == '1' and d['evaluations'] == '1' and
                          0.0 < float(d['rho']) < 1.0, r['label'],
                          'the sweep vote needs one evaluation and a finite rho < 1')

                restart_variant('cuda_d10m3_1e10_sweepvote', 'adaptive', 'restart->device-lsmr',
                                METHOD_LSMR, 'none', sweep_checks,
                                extra=dict(XHDFE_AUTO_MLSMR_RHO_THRESHOLD='1'))
                restart_variant('cuda_d10m3_1e10_explicit', 'adaptive', 'restart->device-lsmr',
                                METHOD_LSMR, 'none',
                                lambda r: check_long(r, 'n/a', 'not-auto', 'keep-selector'),
                                method='gauss-seidel')
    strip = lambda r: {k: v for k, v in r.items() if k not in ('b', 'V')}
    status = 'FAIL' if failures else ('NOT_EXERCISED_ROUTE' if not_exercised else 'PASS')
    report = dict(status=status, module=str(module), data=args.data,
                  tol=TOL, tol_restart=TOL_RESTART, slow=args.slow, failures=failures,
                  not_exercised=not_exercised, variants=variants, skipped=skipped,
                  device_route_ungraded_residuals=device_residuals,
                  rows=[strip(r) for r in rows])
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'failures', 'not_exercised',
                                             'variants', 'skipped',
                                             'device_route_ungraded_residuals')}))
    return {'PASS': 0, 'NOT_EXERCISED_ROUTE': 3}.get(status, 1)


if __name__ == '__main__':
    raise SystemExit(main())
