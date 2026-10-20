"""Independent and negative controls for the core's adapters and GPU routing."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(output, package, dependencies):
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(dependencies, output / 'dependencies')
    shutil.copytree(HERE / 'stata', output / 'harness')
    rows = []
    with (output/'reporting.log').open('x') as stream:
        reporting_rc = subprocess.run([sys.executable, '-B', str(HERE/'test_reporting.py'),
            '--output', str(output/'reporting')], stdout=stream, stderr=subprocess.STDOUT).returncode
    rows.append(dict(control='reporting', status='PASS' if reporting_rc == 0 else 'FAIL',
                     negative_control=True, log=str(output/'reporting.log')))
    specs = [
        ('snapshot', 'cpu', output/'harness/snapshot_controls.do', '', 'CORE_SNAPSHOT_CONTROLS_PASS', False),
        ('snapshot_contract', 'cpu', output/'harness/snapshot_controls.do', '', 'CORE_SNAPSHOT_CONTROLS_PASS', False),
        ('snapshot_repeat_contract', 'cpu', output/'harness/snapshot_controls.do', '', 'CORE_SNAPSHOT_CONTROLS_PASS', False),
        ('routing', 'cuda', output/'harness/routing_controls.do', '', 'CORE_ROUTING_CONTROLS_PASS', False),
        ('masked_gpu', 'cuda', output/'harness/masked_gpu_control.do', '', 'CORE_MASKED_GPU_PASS', True),
        ('cold_restart_floor', 'cuda', ROOT/'tests/stata/cg_cold_restart_floor.do', '', 'CG_COLD_FLOOR_PASS', False),
    ]
    for backend in ('cpu', 'cuda'):
        specs += [(f'saturated_{backend}', backend, ROOT/'tests/stata/saturated_statistics_contract.do', backend,
                   f'SATURATED_STATISTICS_CONTRACT_PASS {backend}', False),
                  (f'dkraay_{backend}', backend, output/'harness/dkraay_capture.do', '', 'CORE_DK_CAPTURE_PASS', False)]
    for name, backend, script, argument, marker, masked in specs:
        folder = output / name
        folder.mkdir(); (folder / 'tmp').mkdir()
        env = {k:v for k,v in os.environ.items() if not k.startswith('XHDFE_')}
        env.update(XHDFE_UPSTREAM_ROOT=str(output/'harness'),
            XHDFE_UPSTREAM_TESTROOT=str(HERE/'upstream/reghdfe/test'),
            XHDFE_CORE_DEPENDENCIES=str(output/'dependencies'), XHDFE_CORE_PACKAGE=str(package),
            XHDFE_STATA_ADOPATH=str(package), XHDFE_UPSTREAM_BACKEND=backend,
            XHDFE_UPSTREAM_EQUIVALENT='0' if name.startswith('snapshot') else '1',
            XHDFE_UPSTREAM_FETOL='1e-8',
            XHDFE_UPSTREAM_CALLS=str(folder/'calls.tsv'),
            XHDFE_CORE_DK_CONTRACT='1' if name == 'snapshot_contract' else '0',
            XHDFE_CORE_REPEAT_CONTRACT='1' if name == 'snapshot_repeat_contract' else '0',
            STATATMP=str(folder/'tmp'), TMPDIR=str(folder/'tmp'),
            CUDA_CACHE_PATH=str(folder/'tmp/cuda'), XHDFE_ABSORPTION_CACHE_MODE='off')
        if masked:
            env['CUDA_VISIBLE_DEVICES'] = ''
        driver = folder/'entry.do'
        driver.write_text(f'include "{output}/harness/env.do"\n'
            'set seed 123456789\nset sortseed 123456789\n'
            f'do "{script}" {argument}\ndisplay "CORE_CONTROL_END"\n')
        with (folder/'process.log').open('x') as stream:
            try:
                proc = subprocess.run([shutil.which('stata-mp'), '-q', '-b', 'do', str(driver)],
                    cwd=folder, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=420)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = 124
        log = (folder/'entry.log').read_text(errors='replace') if (folder/'entry.log').is_file() else ''
        calls = (folder/'calls.tsv').read_text().splitlines() if (folder/'calls.tsv').is_file() else []
        passed = rc == 0 and '\n'+marker+'\n' in log and '\nCORE_CONTROL_END\n' in log
        passed &= not any('backend_ok=0' in call for call in calls)
        if name == 'routing':
            passed &= sum('gpu=1 ' in call for call in calls) == 4
            passed &= sum('cpu_only_noabsorb_control' in call for call in calls) == 2
        if name.startswith('dkraay_') and passed:
            tokens = iter((folder/'matrices.txt').read_text().split())
            matrices = {}
            for key in ('rb', 'rV', 'cb', 'cV'):
                assert next(tokens) == key
                nr, nc = int(next(tokens)), int(next(tokens))
                matrices[key] = [[float(next(tokens)) for _ in range(nc)] for _ in range(nr)]
            with (folder/'matrices.json').open('x') as stream:
                json.dump(matrices, stream, allow_nan=False)
            with (folder/'oracle.log').open('x') as stream:
                rc_oracle = subprocess.run([sys.executable, '-B', str(HERE/'dkraay_oracle.py'),
                    '--fixture', str(folder/'public_auto_fixture.dta'), '--actual', str(folder/'matrices.json'),
                    '--output', str(folder/'ORACLE.json')], stdout=stream, stderr=subprocess.STDOUT).returncode
            passed &= rc_oracle == 0
        rows.append(dict(control=name, status='PASS' if passed else 'FAIL', backend=backend,
            negative_control=masked or name.startswith('snapshot'), log=str(folder/'entry.log')))
        print(name, rows[-1]['status'], flush=True)
    result = dict(status='PASS_CONTROLS' if all(r['status']=='PASS' for r in rows) else 'FAIL', controls=rows)
    with (output/'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return int(result['status'] == 'FAIL')
