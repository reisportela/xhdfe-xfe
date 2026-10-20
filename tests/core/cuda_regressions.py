"""Run permanent Stata CUDA regressions; exit zero is insufficient evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


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
        arguments_text = ' '.join('"' + str(value).replace('"', '""') + '"' for value in arguments)
        driver = (f'capture noisily do "{ROOT/"tests/stata"/script}" {arguments_text}\n'
                  'local driver_rc = _rc\ndisplay "DRIVER_RC " `driver_rc\'\nexit, clear\n')
        with (folder/'driver.do').open('x') as stream:
            stream.write(driver)
        with (folder/'process.log').open('x') as log:
            result = subprocess.run(['stata-mp', '-q', '-b', 'do',
                'driver.do'], cwd=folder, env=case_env,
                stdout=log, stderr=subprocess.STDOUT, timeout=420)
        text = (folder/'driver.log').read_text()
        driver_ok = bool(re.search(r'^DRIVER_RC 0\s*$', text, re.M))
        ok = result.returncode == 0 and driver_ok and marker in text.splitlines()
        if name == 'warm_budget':
            ok = (ok and 'iterations=150/150' in text
                  and 'CUDA_WARM_BUDGET_EXPECTED_REFUSAL' in text.splitlines()
                  and 'CUDA_WARM_BUDGET_NUMERICAL_PASS' in text.splitlines())
        rows.append(dict(case=name, returncode=result.returncode,
                         driver_ok=driver_ok,
                         marker_found=marker in text.splitlines(), status='PASS' if ok else 'FAIL'))
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
    result = dict(status='PASS' if passed else 'FAIL', rows=rows,
                  artifacts=hashes, changed_inputs=changed)
    with (out/'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
    return not passed


if __name__ == '__main__':
    raise SystemExit(main())
