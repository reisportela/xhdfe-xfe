"""Run permanent Stata CUDA regressions; exit zero is insufficient evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
# F2-B: the public pyfixest "difficult" design (10 M rows), read in place.
D10M = Path('/srv/projetos/P008/dataset/pyfixest/benchmark_difficult_n10000000_k10.parquet')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('XHDFE_')}
    env.update(XHDFE_STATA_ADOPATH=str(ROOT/'stata'), OPENBLAS_NUM_THREADS='1')
    cases = [('cuda04_cpu', 'cuda_ordinary_weak_graph_contract.do', ['cpu'],
              'CUDA_ORDINARY_WEAK_GRAPH_PASS cpu'),
             ('cuda04_cuda', 'cuda_ordinary_weak_graph_contract.do', ['cuda'],
              'CUDA_ORDINARY_WEAK_GRAPH_PASS cuda'),
             ('cg03', 'cg_cold_restart_floor.do', [], 'CG_COLD_FLOOR_PASS')]
    from three_fe_fixture import prepare
    fixture, oracle = prepare(out)
    for backend in ('cpu', 'cuda'):
        cases.append(('three_fe_' + backend, 'cuda_three_fe_covariance.do',
                      [str(ROOT/'stata'), backend, str(fixture), str(oracle)],
                      'THREE_FE_COVARIANCE_STATA_PASS'))
        cases.append(('xfepout_' + backend, 'xfepout_three_fe_covariance.do',
                      [str(ROOT/'stata'), backend, str(fixture), str(oracle)],
                      'XFEPOUT_THREE_FE_CONTRACT_COMPLETE'))
    cases.append(('warm_budget', 'cuda_warm_budget.do',
                  [str(ROOT/'stata'), 'cuda', str(fixture), str(oracle)],
                  'CUDA_WARM_BUDGET_PASS'))
    for mode in ('default', 'force'):
        cases.append(('tail_' + mode, 'cuda_tail_handover.do',
                      [str(ROOT/'stata'), str(fixture), str(oracle), mode],
                      'CUDA_TAIL_HANDOVER_STATA_PASS ' + mode))
    # F2-B: the cold restart after a broken CG stage (10 M difficult design, tol
    # 1e-10) handed to the host MLSMR; skipped, and recorded, when the public file
    # is absent.
    skipped = []
    if D10M.is_file():
        cases.append(('tail_restart', 'cuda_tail_handover.do',
                      [str(ROOT/'stata'), str(D10M), '-', 'restart'],
                      'CUDA_TAIL_HANDOVER_STATA_PASS restart'))
    else:
        skipped.append('tail_restart: public fixture absent ' + str(D10M))
    # F2-A: only tail_force forces the hand-over of the three-FE completion. Both tail
    # cases trace the host Krylov solves (stderr, process.log) so the solver that
    # recomputes the stored residuals is recorded (R6 observation, no assertion).
    # tail_restart also traces the restart ledger (one cuda_tail_restart line).
    case_overrides = {'tail_default': {'XHDFE_MLSMR_TRACE': '1'},
                      'tail_force': {'XHDFE_CUDA_TAIL_HANDOVER': 'force',
                                     'XHDFE_MLSMR_TRACE': '1'},
                      'tail_restart': {'XHDFE_GI_LSMR_TRACE': '1',
                                       'XHDFE_MLSMR_TRACE': '1'}}
    tail_line = re.compile(r'^(?:THREE_FE_TAIL|XFEPOUT_TAIL|CUDA_WARM_BUDGET_TAIL|'
                           r'CUDA_TAIL_CPU|CUDA_TAIL_HANDOVER|CUDA_TAIL_RESIDUALS|'
                           r'CUDA_TAIL_RESTART_CPU|CUDA_TAIL_RESTART|'
                           r'CUDA_TAIL_RESTART_NOT_EXERCISED_ROUTE|'
                           r'DEVICE_ROUTE_UNGRADED_RESIDUALS) .*$', re.M)
    restart_line = re.compile(r'^cuda_tail_restart .*$', re.M)
    artifacts = [ROOT/'stata/xhdfe.ado', ROOT/'stata/xhdfe.plugin',
                 ROOT/'stata/xfepout.ado', ROOT/'stata/xfepout.plugin',
                 ROOT/'tests/core/xfepout_covariance.py']
    artifacts += [ROOT/'tests/stata'/name for _, name, _, _ in cases]
    artifacts += [fixture, oracle, ROOT/'tests/core/three_fe_fixture.py',
                  ROOT/'tests/analytic_precision/cuda_three_fe_covariance_contract.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    rows = []
    for name, script, arguments, marker in cases:
        folder = out/name
        folder.mkdir()
        case_env = dict(env, STATATMP=str(folder))
        case_env.update(case_overrides.get(name, {}))
        arguments_text = ' '.join('"' + str(value).replace('"', '""') + '"' for value in arguments)
        driver = (f'capture noisily do "{ROOT/"tests/stata"/script}" {arguments_text}\n'
                  'local driver_rc = _rc\ndisplay "DRIVER_RC " `driver_rc\'\nexit, clear\n')
        with (folder/'driver.do').open('x') as stream:
            stream.write(driver)
        # F2-B round 2: tail_restart runs up to five 10 M CUDA fits; a timeout is
        # recorded as the case's failure and the remaining cases still run.
        timeout = 1800 if name == 'tail_restart' else 420
        returncode = None
        with (folder/'process.log').open('x') as log:
            try:
                returncode = subprocess.run(['stata-mp', '-q', '-b', 'do',
                    'driver.do'], cwd=folder, env=case_env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                returncode = f'timeout after {timeout} s'
        log_path = folder/'driver.log'
        text = log_path.read_text() if log_path.is_file() else ''
        driver_ok = bool(re.search(r'^DRIVER_RC 0\s*$', text, re.M))
        ok = returncode == 0 and driver_ok and marker in text.splitlines()
        if name == 'warm_budget':
            ok = (ok and 'iterations=150/150' in text
                  and 'CUDA_WARM_BUDGET_EXPECTED_REFUSAL' in text.splitlines()
                  and 'CUDA_WARM_BUDGET_NUMERICAL_PASS' in text.splitlines())
        rows.append(dict(case=name, returncode=returncode,
                         driver_ok=driver_ok,
                         marker_found=marker in text.splitlines(), status='PASS' if ok else 'FAIL'))
        rows[-1]['tail_lines'] = tail_line.findall(text)
        if name.startswith('tail_'):
            stderr_text = (folder/'process.log').read_text(errors='replace')
            rows[-1].update(
                host_krylov_calls=stderr_text.count('xhdfe_krylov_trace atomic_scatter='),
                host_mlsmr_preconditioners=stderr_text.count(
                    'xhdfe_mlsmr_trace preconditioner_start'))
        if name == 'tail_restart':
            # Up to five CUDA fits stop at the first restart (route not deterministic):
            # PASS needs exactly one restart, handed to the host; five fits without a
            # restart are NOT_EXERCISED_ROUTE (missing coverage, never a pass).
            restarts = restart_line.findall(stderr_text)
            rows[-1]['restart_lines'] = restarts
            not_exercised = (returncode == 0 and driver_ok and not restarts and
                             'CUDA_TAIL_HANDOVER_STATA_NOT_EXERCISED restart' in text.splitlines())
            if not_exercised:
                rows[-1]['status'] = 'NOT_EXERCISED_ROUTE'
            elif not (len(restarts) == 1 and 'outcome=restart->host-mlsmr' in restarts[0]):
                rows[-1]['status'] = 'FAIL'
        if name == 'warm_budget':
            rows[-1].update(numerical_pass_cases=1 if ok else 0,
                            expected_refusals=1 if ok else 0)
        if name.startswith('xfepout_'):
            from xfepout_covariance import grade_log
            try:
                outcomes = grade_log(text, name.removeprefix('xfepout_'))
                rows[-1].update(
                    numerical_pass_cases=sum(r['status'] == 'PASS' for r in outcomes),
                    expected_refusals=sum(r['status'] == 'EXPECTED_REFUSAL' for r in outcomes),
                    refusal_is_not_numerical_success=True, outcomes=outcomes)
            except (AssertionError, ValueError):
                rows[-1]['status'] = 'FAIL'
    changed = [p for p, value in hashes.items()
               if hashlib.sha256(Path(p).read_bytes()).hexdigest() != value]
    passed = not changed and all(row['status'] == 'PASS' for row in rows)
    not_exercised = (not changed and not passed and
                     all(row['status'] in ('PASS', 'NOT_EXERCISED_ROUTE') for row in rows))
    status = 'PASS' if passed else ('NOT_EXERCISED_ROUTE' if not_exercised else 'FAIL')
    result = dict(status=status, rows=rows,
                  artifacts=hashes, changed_inputs=changed, skipped=skipped)
    with (out/'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
    return {'PASS': 0, 'NOT_EXERCISED_ROUTE': 3}.get(status, 1)


if __name__ == '__main__':
    raise SystemExit(main())
