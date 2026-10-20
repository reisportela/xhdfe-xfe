"""Fresh-process, interleaved core24 validation; aggregate receipts only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = CANDIDATE = BASELINE = SCRATCH = None


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def cpu_counter():
    row = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(row), row[3] + row[4]


def quiet_admission():
    history = []
    deadline = time.monotonic() + 300
    while True:
        a = cpu_counter()
        time.sleep(2)
        b = cpu_counter()
        idle = (b[1] - a[1]) / max(1, b[0] - a[0])
        history.append(dict(epoch=time.time(), idle_fraction=idle, loadavg=os.getloadavg()))
        if idle >= .80:
            return history
        if time.monotonic() >= deadline:
            raise RuntimeError('Host remained busy; timing postponed, no other processes altered')
        time.sleep(5)


def gpu_state():
    out = {}
    for key, query in [('device', '--query-gpu=name,utilization.gpu,memory.used,power.draw,temperature.gpu'),
                       ('processes', '--query-compute-apps=pid,process_name,used_gpu_memory')]:
        result = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=15)
        result.check_returncode()
        out[key] = result.stdout.strip()
    return out


def paths(arm, backend):
    package = CANDIDATE if arm == 'candidate' else BASELINE
    module = (CANDIDATE / ('build_cuda' if backend == 'cuda' else 'build')
              if arm == 'candidate' else BASELINE / ('build_cuda' if backend == 'cuda' else 'build'))
    return package, module


def version(package):
    text = (package / 'stata/xhdfe.ado').read_text()
    found = re.search(r'^\*!\s+version\s+(\S+)\s+(\S+)', text, re.M)
    if not found:
        raise ValueError('Missing Stata version header')
    return ' '.join(found.groups())


def prepare(args):
    ROOT.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.registry.resolve(), ROOT / 'REGISTRY.json')
    registry = json.loads((ROOT / 'REGISTRY.json').read_text())
    core = json.loads((HERE.parent / 'manifest.json').read_text())['core24']
    assert len(registry['specs']) == 24
    assert {s['name'] for s in registry['specs']} == set(core['datasets'])
    assert version(BASELINE) == '2.29.1 14oct2026', 'Operational reference must be 2.29.1'
    artifacts = {}
    for arm in ['baseline', 'candidate']:
        for backend in ['cpu', 'cuda']:
            package, module = paths(arm, backend)
            binaries = list(module.glob('py_hdfe_v11*.so'))
            assert len(binaries) == 1
            for path in [binaries[0], package / 'stata/xhdfe.plugin',
                         package / 'stata/xhdfe.ado']:
                artifacts[str(path)] = digest(path)
    for path in [ROOT / 'REGISTRY.json', *sorted(HERE.glob('*.py')), HERE / 'stata_fit.do']:
        artifacts[str(path)] = digest(path)
    inputs = {}
    for spec in registry['specs']:
        path = Path(spec['path'])
        st = path.stat()
        inputs[spec['name']] = dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns,
                                   device=st.st_dev, inode=st.st_ino)
    write_new(ROOT / 'MANIFEST.json', dict(artifacts=artifacts, inputs=inputs,
        candidate=str(CANDIDATE), baseline=str(BASELINE),
        versions={'candidate': version(CANDIDATE), 'baseline': version(BASELINE)},
        design='24 specifications x 2 interfaces x 2 backends x 2 modes x 3 interleaved pairs',
        fits=1152, performance_reference='2.29.1.20261014', precision_contract_reference='2.28.1.20260929', threads=16, tolerance=1e-8, max_iter=100000, warmup=False,
        cache='off; fresh process', stata_workflow='ordinary; no residuals()',
        timing='fit only; data loading and receipt generation excluded',
        timeouts='900s normally; AKM2 CPU 3600s and CUDA 1800s based on historical duration',
        source_data='read-only in place; aggregates and digests only in evidence',
        stata_scratch=str(SCRATCH), scratch_authorisation='required before a real Stata run',
        performance='No promotion from a subset. Flag noisy or local losses for review.',
        precision='Compare complete b/V, sample, DoF and convergence; thresholds are diagnostic for fast mode.'))
    print('PREPARED 1152 planned fits; no source data read', flush=True)


def run(args):
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for name, expected in manifest['artifacts'].items():
        assert digest(Path(name)) == expected, 'artifact changed: ' + name
    names = list(manifest['inputs'])
    priority = ['main_95_21_ready', 'akm_v02_firstreg', 'simulated_panel', 'marta_group_individual_3m']
    names = [n for n in priority if n in names] + [n for n in names if n not in priority]
    if args.datasets:
        selected = args.datasets.split(',')
        assert set(selected) <= set(names)
        names = selected
    interfaces = args.interfaces.split(',')
    assert interfaces and set(interfaces) <= {'native', 'stata'}
    if 'stata' in interfaces:
        assert args.authorised_scratch == str(SCRATCH), 'Explicit scratch authorisation is required'
    env = {k: v for k, v in os.environ.items() if not k.startswith('XHDFE_') and k != 'LD_LIBRARY_PATH'}
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OMP_NUM_THREADS='16', XHDFE_ABSORPTION_CACHE_MODE='off',
               XHDFE_MOBILITY_MODE='off', XHDFE_FE_STRUCTURE_MODE='off')
    for name in names:
        info = manifest['inputs'][name]
        st = Path(info['path']).stat()
        assert (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev) == (
            info['size'], info['mtime_ns'], info['inode'], info['device'])
        for interface in interfaces:
            for backend in ['cpu', 'cuda']:
                for mode in ['fast', 'comparable']:
                    folder = ROOT / 'receipts' / name / f'{interface}_{backend}_{mode}'
                    folder.mkdir(parents=True, exist_ok=True)
                    for pair in range(1, 4):
                        if (ROOT / 'STOP_AFTER_PAIR').exists():
                            print('HELD between complete pairs; no fit running', flush=True)
                            return
                        arms = ['baseline', 'candidate'] if pair % 2 else ['candidate', 'baseline']
                        for arm in arms:
                            stem = f'p{pair}_{arm}'
                            receipt = folder / (stem + '.json')
                            context = folder / (stem + '.context.json')
                            if context.exists():
                                continue
                            assert not receipt.exists(), 'Incomplete attempt: review before resuming'
                            for path, expected in manifest['artifacts'].items():
                                assert digest(Path(path)) == expected, 'artifact changed: ' + path
                            admission = quiet_admission()
                            gpu_before = gpu_state()
                            if gpu_before['processes']:
                                raise RuntimeError('GPU has another compute process; run postponed')
                            package, module = paths(arm, backend)
                            timeout = (3600 if backend == 'cpu' else 1800) if name == 'akm_v02_secondreg' else 900
                            label = f'{name}/{interface}/{backend}/{mode}/p{pair}/{arm}'
                            common = ['--dataset', name, '--mode', mode, '--backend', backend,
                                      '--threads', '16', '--label', label,
                                      '--registry', str(ROOT / 'REGISTRY.json')]
                            if interface == 'native':
                                command = [sys.executable, '-B', str(HERE / 'py_persistent_worker.py'),
                                           '--module-dir', str(module), '--no-warmup', *common]
                                stdin = f'FIT {receipt}\nEXIT\n'
                            else:
                                expected_version = manifest['versions'][arm]
                                command = [sys.executable, '-B', str(HERE / 'stata_fit_worker.py'),
                                           '--package-dir', str(package), '--expected-version', expected_version,
                                           '--scratch', str(SCRATCH), '--out', str(receipt),
                                           '--timeout', str(timeout), *common]
                                stdin = None
                            before = cpu_counter()
                            start = time.monotonic()
                            with (folder / (stem + '.process.log')).open('x') as log:
                                try:
                                    proc = subprocess.run(command, input=stdin, text=True, env=env,
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout + 30)
                                    rc = proc.returncode
                                except subprocess.TimeoutExpired:
                                    rc = 124
                            after = cpu_counter()
                            write_new(context, dict(command=command, returncode=rc, admission=admission,
                                gpu_before=gpu_before, gpu_after=gpu_state(), wall=time.monotonic()-start,
                                aggregate_cpu_busy=1-(after[1]-before[1])/max(1,after[0]-before[0]),
                                receipt_exists=receipt.exists()))
                            status = 'NO_RECEIPT'
                            if receipt.exists():
                                r = json.loads(receipt.read_text())
                                status = ('PASS' if r.get('success') and r.get('converged') and
                                          (backend == 'cpu' or r.get('gpu_used') == 1) else 'FAIL')
                            print(label, status, flush=True)
    print('SELECTED MATRIX FINISHED; inspect report, including every failed or missing receipt', flush=True)


def report():
    import numpy as np
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    cells = []
    completed = 0
    for name in manifest['inputs']:
        for interface in ['native', 'stata']:
            for backend in ['cpu', 'cuda']:
                for mode in ['fast', 'comparable']:
                    folder = ROOT / 'receipts' / name / f'{interface}_{backend}_{mode}'
                    pairs = []
                    for pair in range(1, 4):
                        files = [folder / f'p{pair}_{arm}.json' for arm in ['baseline', 'candidate']]
                        completed += sum(p.is_file() for p in files)
                        if not all(p.is_file() for p in files):
                            pairs.append(dict(pair=pair, status='NOT_RUN'))
                            continue
                        a, b = [json.loads(p.read_text()) for p in files]
                        reasons = []
                        for arm, r in [('baseline', a), ('candidate', b)]:
                            context_path = folder / f'p{pair}_{arm}.context.json'
                            if not context_path.is_file() or json.loads(context_path.read_text()).get('returncode') != 0:
                                reasons.append(arm + ': missing or failed process receipt')
                            if not r.get('success') or not r.get('converged'):
                                reasons.append(arm + ': failed or unconverged')
                            if not r.get('precision_certified'):
                                reasons.append(arm + ': precision not certified')
                            seconds = r.get('elapsed_seconds')
                            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                                reasons.append(arm + ': invalid timing')
                            if backend == 'cuda' and (r.get('gpu_used') != 1 or r.get('gpu_status') != 'used'):
                                reasons.append(arm + ': CUDA not used')
                            if backend == 'cpu' and r.get('gpu_used') != 0:
                                reasons.append(arm + ': incorrect CPU backend')
                        metrics = {}
                        if not reasons:
                            for key in ['terms', 'nobs', 'nobs_full', 'num_singletons', 'df_a', 'df_m', 'df_resid']:
                                if key not in a or key not in b:
                                    reasons.append('missing ' + key)
                                if a.get(key) != b.get(key):
                                    reasons.append('changed ' + key)
                            sample_key = 'sample_sha256' if interface == 'native' else 'sample_signature'
                            if not a.get(sample_key) or a.get(sample_key) != b.get(sample_key):
                                reasons.append('changed or absent sample digest')
                            input_key = 'input_sha256' if interface == 'native' else 'input_signature'
                            if not a.get(input_key) or a.get(input_key) != b.get(input_key):
                                reasons.append('changed or absent input digest')
                            av, bv = np.array(a['covariance']), np.array(b['covariance'])
                            ab, bb = np.array(a['coef']), np.array(b['coef'])
                            if (ab.size == 0 or av.shape != bv.shape or ab.shape != bb.shape or
                                    av.shape != (ab.size, ab.size) or
                                    not all(np.isfinite(x).all() for x in (av, bv, ab, bb)) or
                                    np.any(np.diag(av) < 0)):
                                pairs.append(dict(pair=pair, status='REVIEW',
                                    reasons=['Non-finite, negative variance or incompatible b/V dimensions']))
                                continue
                            scale = np.sqrt(np.outer(np.diag(av), np.diag(av)))
                            delta = np.abs(av-bv)
                            scaled = np.divide(delta, scale, out=np.zeros_like(delta), where=scale>0)
                            scaled[(scale == 0) & (delta != 0)] = np.inf
                            verror = float(np.max(scaled))
                            berror = float(np.max(np.abs(ab-bb)/np.maximum(1, np.abs(ab))))
                            metrics = dict(b_scaled_error=berror, V_scaled_error=verror if np.isfinite(verror) else None,
                                seconds_baseline=a['elapsed_seconds'], seconds_candidate=b['elapsed_seconds'],
                                candidate_over_baseline=b['elapsed_seconds']/a['elapsed_seconds'],
                                iterations_baseline=a['iterations'], iterations_candidate=b['iterations'])
                            if mode == 'comparable' and (berror > 1e-9 or verror > 1e-8):
                                reasons.append('Comparable b/V review; independent diagnosis required, no automatic solver change')
                        pairs.append(dict(pair=pair, status='REVIEW' if reasons else 'PASS', reasons=reasons, **metrics))
                    ratios = [p['candidate_over_baseline'] for p in pairs if 'candidate_over_baseline' in p]
                    cells.append(dict(dataset=name, interface=interface, backend=backend, mode=mode, pairs=pairs,
                        median_ratio=statistics.median(ratios) if len(ratios)==3 else None))
    target = ROOT / f'REPORT_{len(list(ROOT.glob("REPORT_*.json")))+1:03d}.json'
    write_new(target, dict(planned_fits=1152, completed_receipts=completed, cells=cells,
        artifact_identity_unchanged=all(Path(p).is_file() and digest(Path(p)) == h
            for p,h in manifest['artifacts'].items()),
        status='INCOMPLETE' if completed !=1152 else 'REVIEW_REQUIRED',
        caution='Baseline parity is not an independent mathematical oracle. Fast diagnostic differences do not redefine its contract.'))
    print(target, completed, '/ 1152 receipts')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'report'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--registry', type=Path)
    p.add_argument('--scratch', type=Path)
    p.add_argument('--interfaces', default='native,stata')
    p.add_argument('--datasets')
    p.add_argument('--authorised-scratch')
    args = p.parse_args()
    ROOT = args.output.resolve()
    if not ROOT.is_relative_to(PROJECT):
        p.error('Outputs must stay inside this repository')
    if args.action == 'prepare':
        if not all((args.candidate, args.baseline, args.registry, args.scratch)):
            p.error('prepare requires candidate, baseline, registry and scratch paths')
        CANDIDATE, BASELINE, SCRATCH = (p.resolve() for p in
            (args.candidate, args.baseline, args.scratch))
        prepare(args)
    else:
        manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
        CANDIDATE = Path(manifest['candidate'])
        BASELINE = Path(manifest['baseline'])
        SCRATCH = Path(manifest['stata_scratch'])
        if args.action == 'run':
            run(args)
        else:
            report()
