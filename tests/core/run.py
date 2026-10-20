"""One entry point for core24, upstream compatibility and independent oracles.

Prepare an immutable plan, run its stages serially, then report every stage.
Missing coverage, reference-only diagnostics and refusals never become PASS.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGES = ('controls', 'reference', 'literal', 'compatibility_cpu',
          'compatibility_cuda', 'analytic', 'performance')
DIAGNOSTIC = {'reference', 'literal'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def module(root, backend):
    paths = list((root / ('build_cuda' if backend == 'cuda' else 'build')).glob('py_hdfe_v11*.so'))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one {backend} module under {root}')
    return paths[0]


def prepare(args):
    if not args.dependencies:
        raise ValueError('Supply the isolated dependencies prepared by dependencies.py')
    candidate = (args.candidate or ROOT).resolve()
    # The analytic battery protects its own selected repository boundary.
    if candidate != ROOT:
        raise ValueError('Run this entry point from the selected candidate repository')
    dependencies = args.dependencies.resolve()
    assert (dependencies / 'ado/plus/r/reghdfe.ado').is_file()
    config = dict(candidate=str(candidate), dependencies=str(dependencies),
        baseline=str(args.baseline.resolve()) if args.baseline else None,
        registry=str(args.registry.resolve()) if args.registry else None,
        scratch=str(args.scratch.resolve()) if args.scratch else None)
    artifacts = [module(candidate, b) for b in ('cpu', 'cuda')]
    artifacts += [candidate / 'stata/xhdfe.ado', candidate / 'stata/xhdfe.plugin']
    artifacts += [candidate / 'tests/analytic_precision/cuda_ordinary_weak_graph_contract.py']
    artifacts += [candidate / 'tests/analytic_precision' / name for name in
                  ('fast_cg_mode_contract.py', 'cg_final_step_contract.py',
                   'fixtures/github_fast_aggregate_reference.json',
                   'fixtures/cg_seed1_2017.csv', 'fixtures/cg_seed1_2017_oracle.json')]
    artifacts += [candidate / 'tests/stata' / name for name in
                  ('cuda_ordinary_weak_graph_contract.do', 'cg_cold_restart_floor.do')]
    artifacts += [p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    artifacts += [p for p in dependencies.rglob('*') if p.is_file()]
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'PLAN.json', dict(schema=1, config=config,
        manifest=json.loads((HERE / 'manifest.json').read_text()), stages=list(STAGES),
        diagnostic_stages=sorted(DIAGNOSTIC), artifacts={str(p): sha(p) for p in artifacts},
        runtime_reference='2.29.1.20261014', precision_history='2.28.1.20260929',
        promotion=False, interpretation='A core report is not a multiplatform release certificate.'))


def execute(command, path, env=None):
    with path.open('x') as log:
        return subprocess.run([str(x) for x in command], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def controls(output, config):
    from controls import run
    return run(output, Path(config['candidate']) / 'stata', Path(config['dependencies']))


def run(args):
    plan = json.loads((args.output / 'PLAN.json').read_text())
    config = plan['config']
    selected = STAGES if args.stage == 'all' else STAGES[:-1] if args.stage == 'functional' else (args.stage,)
    failed = False
    for stage in selected:
        for path, expected in plan['artifacts'].items():
            assert Path(path).is_file() and sha(Path(path)) == expected, 'Changed input: ' + path
        receipt = args.output / (stage + '.json')
        if receipt.exists():
            raise RuntimeError(f'{stage} already has a receipt; retain it and prepare a new campaign')
        output = args.output / stage
        start = time.monotonic()
        commands = []
        code = None
        status = None
        if stage == 'controls':
            code = controls(output, config)
        elif stage in ('reference', 'literal', 'compatibility_cpu', 'compatibility_cuda'):
            command = [sys.executable, '-B', HERE / 'upstream.py', '--output', output,
                '--package', Path(config['candidate']) / 'stata',
                '--dependencies', config['dependencies'],
                '--engine', 'reghdfe' if stage == 'reference' else 'xhdfe',
                '--backend', 'cuda' if stage == 'compatibility_cuda' else 'cpu',
                '--mode', 'literal' if stage in DIAGNOSTIC else 'adapted']
            commands.append(command)
        elif stage == 'analytic':
            script = ROOT / 'tests/analytic_precision/run.py'
            common = [sys.executable, '-B', script]
            commands = [common + ['prepare', '--output', output,
                '--native-cpu-module', module(ROOT, 'cpu'),
                '--native-cuda-module', module(ROOT, 'cuda'),
                '--stata-package-root', ROOT],
                common + ['run', '--output', output],
                common + ['report', '--output', output]]
            commands += [[sys.executable, '-B',
                ROOT / 'tests/analytic_precision/cuda_ordinary_weak_graph_contract.py',
                '--module', module(ROOT, backend), '--backend', backend,
                '--output', output / ('cuda04_' + backend + '.json')]
                for backend in ('cpu', 'cuda')]
            commands += [[sys.executable, '-B', HERE / 'cuda_regressions.py',
                          '--output', output / 'cuda_stata_regressions']]
            github = None
            if config['registry']:
                registry = json.loads(Path(config['registry']).read_text())
                github = next(s['path'] for s in registry['specs'] if s['name'] == 'github')
            for backend in ('cpu', 'cuda'):
                command = [sys.executable, '-B', ROOT / 'tests/analytic_precision/fast_cg_mode_contract.py',
                    '--module', module(ROOT, backend), '--backend', backend,
                    '--output', output / ('fast_cg_' + backend + '.json')]
                if github:
                    command += ['--github-data', github]
                commands.append(command)
                commands.append([sys.executable, '-B', ROOT / 'tests/analytic_precision/cg_final_step_contract.py',
                    '--module', module(ROOT, backend), '--backend', backend,
                    '--output', output / ('cg02_' + backend + '.json')])
        elif stage == 'performance':
            if not all(config[k] for k in ('baseline', 'registry', 'scratch')):
                status = 'NOT_CONFIGURED'
            elif args.authorised_scratch != config['scratch']:
                raise ValueError('Provide --authorised-scratch with the explicitly authorised data temporary path')
            else:
                common = [sys.executable, '-B', HERE / 'performance/run_matrix.py']
                commands = [[sys.executable, '-B', HERE / 'performance/harness_smoke.py',
                    '--output', args.output / 'performance_smoke',
                    '--candidate', config['candidate'], '--baseline', config['baseline']],
                    common + ['prepare', '--output', output,
                    '--candidate', config['candidate'], '--baseline', config['baseline'],
                    '--registry', config['registry'], '--scratch', config['scratch']],
                    common + ['run', '--output', output, '--authorised-scratch', config['scratch']],
                    common + ['report', '--output', output]]
        codes = []
        for number, command in enumerate(commands):
            rc = execute(command, args.output / f'{stage}_{number}.log')
            codes.append(rc)
            # Preserve diagnostic failures; a failed prepare cannot be run.
            if rc and stage in ('analytic', 'performance') and not (output / 'MANIFEST.json').exists() and not (output / 'manifest.json').exists():
                break
        if codes:
            code = max(codes)
        if status is None:
            status = ('DIAGNOSTIC_COMPLETE' if stage in DIAGNOSTIC else
                      'COMPLETED_REQUIRES_REPORT' if code == 0 else 'FAIL')
        write(receipt, dict(stage=stage, status=status, returncode=code,
            seconds=time.monotonic()-start, commands=[[str(x) for x in cmd] for cmd in commands]))
        print(stage, status, flush=True)
        if stage not in DIAGNOSTIC and code:
            # Do not launch long timings on a known failing candidate.
            failed = True
            break
    return int(failed)


def report(args):
    plan = json.loads((args.output / 'PLAN.json').read_text())
    stages = []
    for stage in STAGES:
        receipt = args.output / (stage + '.json')
        row = json.loads(receipt.read_text()) if receipt.exists() else dict(stage=stage, status='NOT_RUN')
        result = args.output / stage / 'RESULT.json'
        if result.exists():
            data = json.loads(result.read_text())
            if 'outcomes' in data:
                row['counts'] = dict(Counter(r['status'] for r in data['outcomes']))
                row['cuda_calls'] = sum(r.get('cuda_calls', 0) for r in data['outcomes'])
                row['cpu_only_controls'] = sum(r.get('cpu_only_controls', 0) for r in data['outcomes'])
                row['cases_without_gpu_execution'] = [r['test'] for r in data['outcomes'] if not r.get('cuda_calls')]
                row['custody_unchanged'] = data['custody_unchanged']
                if stage not in DIAGNOSTIC:
                    row['status'] = 'PASS_COMPATIBILITY' if row.get('returncode') == 0 and data['custody_unchanged'] else 'FAIL'
            elif stage == 'controls':
                row['status'] = data['status']
                row['controls'] = data['controls']
        if stage in ('analytic', 'performance'):
            pattern = 'report_*.json' if stage == 'analytic' else 'REPORT_*.json'
            reports = sorted((args.output / stage).glob(pattern))
            if reports:
                data = json.loads(reports[-1].read_text())
                row['report'] = str(reports[-1])
                if stage == 'analytic':
                    row['status'] = ('PASS_SCOPED_ORACLES' if row.get('returncode') == 0
                        and data['xhdfe_scoped_certified'] and data['t01_scoped_pass'] else 'FAIL')
                    row['xhdfe_counts'] = data['xhdfe_counts']
                    row['cuda04'] = {backend: json.loads((args.output / stage /
                        ('cuda04_' + backend + '.json')).read_text())['status']
                        if (args.output / stage / ('cuda04_' + backend + '.json')).exists()
                        else 'NOT_RUN' for backend in ('cpu', 'cuda')}
                    if any(status != 'PASS' for status in row['cuda04'].values()):
                        row['status'] = 'FAIL'
                    stata_result = args.output / stage / 'cuda_stata_regressions/RESULT.json'
                    row['cuda_stata_regressions'] = (json.loads(stata_result.read_text())['status']
                                                    if stata_result.exists() else 'NOT_RUN')
                    if row['cuda_stata_regressions'] != 'PASS':
                        row['status'] = 'FAIL'
                    row['fast_cg_policy'] = {}
                    row['cg02_precision'] = {}
                    for backend in ('cpu', 'cuda'):
                        fast = args.output / stage / ('fast_cg_' + backend + '.json')
                        precise = args.output / stage / ('cg02_' + backend + '.json')
                        if fast.exists():
                            details = json.loads(fast.read_text())
                            row['fast_cg_policy'][backend] = dict(status=details['status'], github=details['github'])
                        else:
                            row['fast_cg_policy'][backend] = dict(status='NOT_RUN')
                        row['cg02_precision'][backend] = (json.loads(precise.read_text())['status']
                                                          if precise.exists() else 'NOT_RUN')
                        if (row['fast_cg_policy'][backend]['status'] != 'PASS_SCOPED_FAST_POLICY' or
                                row['cg02_precision'][backend] != 'PASS'):
                            row['status'] = 'FAIL'
                else:
                    row['status'] = data['status']
                    row['completed_receipts'] = data['completed_receipts']
        stages.append(row)
    changed = [p for p,h in plan['artifacts'].items() if not Path(p).is_file() or sha(Path(p)) != h]
    target = args.output / f'REPORT_{len(list(args.output.glob("REPORT_*.json")))+1:03d}.json'
    write(target, dict(stages=stages, changed_inputs=changed, core24_planned_fits=1152,
        status='STALE' if changed else 'REVIEW_REQUIRED',
        release_certified=False, reason='Read every stage; performance adjudication and platform release checks are separate.'))
    print(target)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'run', 'report'))
    p.add_argument('--output', type=Path, required=True)
    for name in ('candidate', 'baseline', 'registry', 'scratch', 'dependencies'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--stage', choices=(*STAGES, 'functional', 'all'), default='all')
    p.add_argument('--authorised-scratch')
    args = p.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT):
        p.error('Outputs must stay inside this repository')
    raise SystemExit({'prepare': prepare, 'run': run, 'report': report}[args.action](args) or 0)
