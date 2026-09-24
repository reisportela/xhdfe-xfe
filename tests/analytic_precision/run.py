#!/usr/bin/env python3
"""Prepare, execute and report the permanent analytic precision certification."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from cases import NAMES, generate
from evaluate_fail_closed import assess, selftest
from evaluate_t01 import assess_t01, selftest_t01
from precision_contract import contract_manifest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ('auto', 'gauss-seidel', 'symmetric-gauss-seidel', 'jacobi', 'schwarz',
           'lsmr', 'mlsmr', 'auto-mlsmr')


def clean_json(value):
    if isinstance(value, dict):
        return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(clean_json(value), handle, indent=2, allow_nan=False)
        handle.write('\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def custody(extra_paths=()):
    paths = [*HERE.glob('*.py'), *HERE.glob('*.do'), *HERE.glob('*.R'),
             HERE/'supplemental_cases.json',
             *(p for p in (HERE/'supplemental').glob('*') if p.is_file()),
             ROOT/'src/fe_absorption.cpp',
             ROOT/'src/hdfe_regressor_v11.cpp', ROOT/'src/fe_absorption_cuda.cu',
             ]
    for name in ('reghdfe.ado', 'reghdfe.mata'):
        p = Path('/home/mangelo/ado/plus/r')/name
        if p.is_file():
            paths.append(p)
    paths.extend(Path(p) for p in extra_paths)
    return {str(p):digest(p) for p in dict.fromkeys(paths) if p.is_file()}


def catalogue(case, interfaces, backends):
    jobs = []
    def add(interface, engine, method='auto', precision='default', backend='cpu', **kwargs):
        job = dict(case=case.name, interface=interface, engine=engine, method=method,
                   precision=precision, backend=backend, **kwargs)
        if job.get('expected_rejection'):
            if engine=='reghdfe' and case.weight=='iw':
                job['reject_pattern'] = 'iweights not allowed'
            elif engine=='reghdfe':
                job['reject_pattern'] = 'FixedEffects::update_preconditioner'
            elif interface=='stata' and method=='schwarz':
                job['reject_pattern'] = 'unknown absorption_method: schwarz'
            elif precision=='invalid-strict-slope':
                job['reject_pattern'] = 'strict-residual.*not supported with heterogeneous slopes'
            elif job.get('savefe'):
                job['reject_pattern'] = '(savefe/savefes|retain_fes).*not supported.*(group|individual)'
            elif backend=='cuda' and method in ('lsmr','mlsmr'):
                job['reject_pattern'] = r'absorptionmethod\(lsmr/mlsmr\) is CPU-only'
            elif case.kind=='group_individual':
                job['reject_pattern'] = '(not supported with group/individual|CPU.only with group/individual)'
            elif case.slopes:
                job['reject_pattern'] = 'does not support heterogeneous slopes'
            elif (interface=='native' and backend=='cpu' and method=='schwarz'
                  and case.kind=='standard' and case.weight):
                job['reject_pattern'] = (r'absorptionmethod\(schwarz\).*CPU Schwarz requires no weights'
                                         r'.*use absorptionmethod\(auto\).*No estimates returned')
            else:
                job['reject_pattern'] = '(CPU.only|CPU only|not supported.*GPU|not supported.*CUDA|Requested GPU backend was unavailable during HDFE absorption)'
            if interface=='native' and backend=='cpu' and method=='schwarz' and case.weight:
                job['rejection_status'] = dict(converged=0, iterations=0, certified=0, method_used=4)
        job['id'] = '__'.join(str(job.get(k, '')) for k in
                             ('case','interface','engine','method','precision','backend','savefe','recovery','reference_options'))
        job['id'] = job['id'].replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        jobs.append(job)
    for interface in interfaces:
        if interface=='native' and case.weight=='iw':
            continue
        if interface == 'stata':
            add(interface, 'ols')
        methods = METHODS if len(case.fes)>=2 else ('auto',)
        if case.stress:
            methods = ('auto', 'lsmr')
        for method in methods:
            reject = case.kind == 'group_individual' and method in ('jacobi','schwarz','mlsmr','auto-mlsmr')
            reject |= bool(case.slopes) and method in ('lsmr','mlsmr')
            reject |= interface=='stata' and method=='schwarz'
            reject |= interface=='native' and method=='schwarz' and bool(case.weight)
            add(interface, 'xhdfe', method, expected_rejection=reject)
        add(interface, 'xhdfe', precision='strict')
        add(interface, 'xhdfe', precision='fast')
        if case.slopes:
            add(interface, 'xhdfe', precision='invalid-strict-slope', expected_rejection=True)
        if case.kind=='group_individual' and not case.stress:
            add(interface, 'xhdfe', savefe=True, recovery='hybrid', expected_rejection=True)
        if case.name in ('one_fe','two_fe','heterogeneous'):
            add(interface, 'xhdfe', savefe=True, recovery='hybrid')
            if interface == 'stata':
                add(interface, 'xhdfe', savefe=True, recovery='map')
        if interface == 'stata' and case.name != 'ols_no_constant':
            add(interface, 'reghdfe', method='default', expected_rejection=case.weight=='iw')
            if case.weight!='iw':
                add(interface, 'reghdfe', method='lsmr')
                add(interface, 'reghdfe', method='lsqr')
                add(interface, 'reghdfe', method='lsmr', precision='strict')
            if case.kind == 'group_individual':
                add(interface, 'reghdfe', method='map', expected_rejection=True)
            if case.name == 'two_fe':
                for acceleration in ('none','sd','aitken','cg'):
                    for transform in ('kaczmarz','symmetric_kaczmarz','cimmino'):
                        if acceleration == 'cg' and transform == 'kaczmarz':
                            continue
                        add(interface, 'reghdfe', method='map',
                            reference_options=f'acceleration({acceleration}) transform({transform})')
                for method in ('lsmr','lsqr'):
                    for preconditioner in ('none','diagonal','block_diagonal'):
                        add(interface, 'reghdfe', method=method,
                            reference_options=f'preconditioner({preconditioner})')
        if 'cuda' in backends and case.fes:
            gpu_methods = METHODS if not case.stress and len(case.fes)>=2 else ('auto',)
            for method in gpu_methods:
                reject = method in ('lsmr','mlsmr') or interface=='stata' and method=='schwarz'
                reject |= case.kind=='group_individual' and method in ('jacobi','schwarz','auto-mlsmr')
                add(interface, 'xhdfe', method, backend='cuda', expected_rejection=reject)
            add(interface, 'xhdfe', precision='strict', backend='cuda')
            if case.name in ('two_fe','heterogeneous'):
                add(interface, 'xhdfe', backend='cuda', savefe=True, recovery='hybrid')
    return jobs


def prepare(args):
    out = args.output.resolve()
    assert out.is_relative_to(ROOT), 'Keep certification outputs inside the repository'
    out.mkdir(parents=True, exist_ok=False)
    (out/'tmp').mkdir()
    (out/'attempts').mkdir()
    jobs, fixtures = [], {}
    names = args.cases.split(',') if args.cases else NAMES
    for name in names:
        case = generate(name)
        directory = out/'fixtures'/name
        meta, oracle = case.write(directory)
        selftest(oracle, meta, case.group_frame, directory)
        selftest_t01(oracle, meta, directory, assess)
        weights = f'[{case.weight}=weight]' if case.weight else ''
        groupopt = '' if case.kind == 'standard' else 'group(group)'
        if case.kind == 'group_individual':
            groupopt += f' individual(individual) aggregation({case.aggregation})'
        vce = f'vce(cluster {" ".join(case.clusters)})' if case.clusters else f'vce({case.vce})'
        options = dict(absorbs=case.absorb, weights=weights, groupopt=groupopt,
                       vceopt=vce, constantopt='' if case.intercept else 'noconstant',
                       ols_vceopt='vce(unadjusted)' if len(case.clusters)>1 else vce,
                       slope_recovery=' + '.join(f'__hdfe{i+1}__Slope1*{v}' for i,v,_ in case.slopes))
        (directory/'options.do').write_text(''.join(f'local {k} "{v}"\n' for k,v in options.items()))
        fixtures[name] = {p.name:digest(p) for p in directory.iterdir() if p.is_file()}
        jobs.extend(catalogue(case, args.interfaces.split(','), args.backends.split(',')))
        print('PREPARED', name, 'groups', oracle['groups'], 'rank_D', oracle['rank_D'], flush=True)
    def one_module(explicit):
        path = explicit.resolve()
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise RuntimeError(f"native module is absent: {path}")
        return path
    native_requested = 'native' in args.interfaces.split(',')
    cpu_module = one_module(args.native_cpu_module) if native_requested else None
    cuda_module = (one_module(args.native_cuda_module)
                   if native_requested and 'cuda' in args.backends.split(',') else None)
    stata_root = args.stata_package_root.resolve() if args.stata_package_root else None
    if stata_root is not None and not stata_root.is_relative_to(ROOT):
        raise RuntimeError("Stata package root must be inside the selective tree")
    stata_ado = stata_root/'stata/xhdfe.ado' if stata_root else None
    stata_plugin = stata_root/'stata/xhdfe.plugin' if stata_root else None
    if ('stata' in args.interfaces.split(',') and
            (not stata_ado.is_file() or not stata_plugin.is_file())):
        raise RuntimeError(f"Stata package is incomplete under {stata_root}")
    artifact_paths = [*( [cpu_module] if cpu_module else []),
                      *( [cuda_module] if cuda_module else []),
                      *( [stata_ado, stata_plugin] if 'stata' in args.interfaces.split(',') else [])]
    copy_manifest_path = ROOT/'provenance/COPY_MANIFEST.json'
    if copy_manifest_path.is_file():
        copy_manifest = json.loads(copy_manifest_path.read_text())
        source_identity = dict(kind='verified_selective_copy',
                               copy_manifest=str(copy_manifest_path),
                               copy_manifest_sha256=digest(copy_manifest_path),
                               private_head=copy_manifest.get('private_head'),
                               source_v48=copy_manifest.get('source_v48'))
    else:
        source_identity = dict(
            kind='git_checkout',
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    manifest = dict(schema=1, prepared_utc=datetime.now(timezone.utc).isoformat(),
                    source_identity=source_identity,
                    files=custody(artifact_paths), fixtures=fixtures, jobs=jobs,
                    artifacts={
                        'native_cpu': ({'path':str(cpu_module), 'sha256':digest(cpu_module)}
                                       if cpu_module else None),
                        'native_cuda': ({'path':str(cuda_module), 'sha256':digest(cuda_module)}
                                        if cuda_module else None),
                        'stata_package_root': str(stata_root) if stata_root else None,
                        'stata_ado_sha256': digest(stata_ado) if stata_ado and stata_ado.is_file() else None,
                        'stata_plugin_sha256': digest(stata_plugin) if stata_plugin and stata_plugin.is_file() else None,
                    },
                    certificate_mode=args.certificate_mode,
                    checker_negative_controls='PASS',
                    t01_contract=contract_manifest(),
                    t01_sources={name:digest(HERE/name) for name in
                                 ('precision_contract.py','evaluate_t01.py','t01_regrade.py')},
                    exclusions={'reghdfe_ols_no_constant': 'reghdfe noconstant hides the intercept display but still absorbs a constant; different model',
                                'native_iweights': 'Importance-weight Stata semantics are implemented in the ado frontend, not a separate native fit weight type',
                                'ols_multiway_vce': 'regress uses conventional VCE for the explicit OLS oracle; multiway SSC/PSD conventions are assessed by the separate matrix oracle'},
                    scope='Deterministic analytic fixtures; not universal software/platform/performance certification',
                    pending=['R/formula APIs','IV/2SLS','weighted-singleton cascades',
                             'explicit persistent-cache mutation','macOS/Windows runtime',
                             'full-data performance matrix'])
    write_json(out/'manifest.json', manifest)
    print('ANALYTIC_PREPARE_PASS', len(fixtures), 'fixtures', len(jobs), 'jobs', flush=True)


def stata_result(directory):
    table = pd.read_stata(directory/'fit.dta')
    assert len(table) == 1
    for column in table:
        assert table[column].dtype == np.float64, (column, str(table[column].dtype))
    row = table.iloc[0].to_dict()
    if row['rc']:
        return row
    residual = pd.read_stata(directory/'residuals.dta')
    residual = residual.loc[residual.fit_sample == 1]
    row.update(beta=[row[f'b{i}'] for i in range(1,4)],
               V=[[row[f'v{i}{j}'] for j in range(1,4)] for i in range(1,4)],
               groups=residual.group.astype(int).tolist(), residuals=residual.fit_residual.tolist(),
               recovered_fe=residual.recovered_fe.tolist(), gpu_used=row['gpu'],
               method_used=row['method'])
    return row


def run(args):
    out = args.output.resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    changed = [p for p,h in manifest['files'].items() if digest(Path(p)) != h]
    assert not changed, f'Custody mismatch; prepare a new run: {changed}'
    for name, files in manifest['fixtures'].items():
        assert all(digest(out/'fixtures'/name/p) == h for p,h in files.items()), name
    env = os.environ.copy()
    env.update(STATATMP=str(out/'tmp'), TMPDIR=str(out/'tmp'),
               CUDA_CACHE_PATH=str(out/'tmp'/'cuda-cache'),
               XHDFE_MOBILITY_MODE='off', XHDFE_ABSORPTION_CACHE_MODE='off',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    if manifest.get('certificate_mode') == 'audit':
        env['XHDFE_CERTIFY'] = '1'
    else:
        env.pop('XHDFE_CERTIFY', None)
    jobs = manifest['jobs']
    if args.filter:
        jobs = [j for j in jobs if args.filter in j['id']]
    if args.backend:
        jobs = [j for j in jobs if j['backend']==args.backend]
    if any(j['backend']=='cuda' for j in jobs):
        probe = subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=True)
        context = out/f'gpu_context_{len(list(out.glob("gpu_context_*.txt")))+1:03d}.txt'
        context.write_text(probe.stdout+probe.stderr)
    count = 0
    for job in jobs:
        destination = out/'attempts'/job['id']
        if (destination/'assessment.json').is_file():
            continue
        destination.mkdir(exist_ok=False)
        fixture = out/'fixtures'/job['case']
        actual_job = dict(job, fixture=str(fixture))
        if job['interface'] == 'native':
            artifact = manifest['artifacts']['native_'+job['backend']]
            actual_job.update(module=artifact['path'], module_sha256=artifact['sha256'])
        write_json(destination/'job.json', actual_job)
        start = time.monotonic()
        try:
            if job['interface'] == 'native':
                proc = subprocess.run([sys.executable,'-B',str(HERE/'native.py'),str(destination/'job.json')],
                                      cwd=ROOT, env=env, text=True, capture_output=True, timeout=args.timeout)
                raw = json.loads((destination/'raw.json').read_text()) if (destination/'raw.json').exists() else dict(harness_error=proc.stderr)
            else:
                extra = ''
                if job['engine'] == 'xhdfe':
                    extra = f'absorptionmethod({job["method"]}) gpubackend({job["backend"]}) numthreads(2)'
                    if job['precision'] == 'strict':
                        meta = json.loads((fixture/'case.json').read_text())
                        extra += (' tolerance(1e-12) tolerancemode(reghdfe-comparable) convergence(both)' if meta['slopes'] else
                                  ' tolerance(1e-12) tolerancemode(strict-residual)')
                    if job['precision'] == 'invalid-strict-slope':
                        extra += ' tolerancemode(strict-residual)'
                    if job['precision'] == 'fast':
                        extra += ' tolerancemode(xhdfe-fast)'
                    if job.get('savefe'):
                        extra += f' ferecoverymethod({job["recovery"]})'
                if job['engine'] == 'reghdfe':
                    if job['method'] != 'default':
                        extra = f'technique({job["method"]})'
                    extra += ' '+job.get('reference_options','')
                    if job['precision'] == 'strict':
                        extra += ' tolerance(1e-14) iterate(100000)'
                command = (f'do "{HERE / "audit.do"}" "{manifest["artifacts"]["stata_package_root"]}" "{fixture}" "{destination}" '
                           f'{job["engine"]} "{extra}" {"yes" if job.get("savefe") else "no"}\nexit, clear\n')
                (destination/'command.do').write_text(command)
                proc = subprocess.run([args.stata,'-q'],input=command,cwd=ROOT,env=env,
                                      text=True,capture_output=True,timeout=args.timeout)
                if '\nANALYTIC_ATTEMPT_RECORDED\n' not in proc.stdout:
                    raw = dict(harness_error=proc.stdout[-4000:])
                else:
                    raw = stata_result(destination)
                    if raw.get('rc'):
                        raw['diagnostic'] = proc.stdout[-12000:]
            (destination/'process.log').write_text(proc.stdout+proc.stderr)
        except subprocess.TimeoutExpired as error:
            raw = dict(timeout=True)
            data = error.stdout or b''
            (destination/'process.log').write_text(data.decode(errors='replace') if isinstance(data,bytes) else data)
        except Exception as error:
            raw = dict(harness_error=str(error))
        raw['wall_seconds'] = time.monotonic()-start
        if not (destination/'raw.json').exists():
            write_json(destination/'raw.json', raw)
        try:
            assessment = assess(job, raw, fixture)
        except Exception as error:
            assessment = dict(job, verdict='HARNESS_ERROR', result=raw, grading_error=str(error))
        try:
            assessment['t01'] = assess_t01(job, raw, fixture, assessment)
        except Exception as error:
            assessment['t01'] = dict(contract_id=manifest['t01_contract']['contract_id'],
                                     applicable=True, g1_verdict='HARNESS_FAILURE',
                                     passed=False, reason='T01 grading error',
                                     grading_error=str(error))
        # Raw vectors remain in raw.json and Stata DTA artifacts; report stays compact.
        assessment['result'] = {k:v for k,v in raw.items() if k not in ('residuals','groups','recovered_fe')}
        write_json(destination/'assessment.json', assessment)
        print(assessment['verdict'], job['id'], f'{raw["wall_seconds"]:.2f}s', flush=True)
        count += 1
        if args.limit and count >= args.limit:
            break
    print('RECORDED_ATTEMPTS', count, flush=True)


def report(args):
    out = args.output.resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    rows = []
    for job in manifest['jobs']:
        path = out/'attempts'/job['id']/'assessment.json'
        rows.append(json.loads(path.read_text()) if path.exists() else dict(job, verdict='NOT_RUN'))
    counts = Counter(r['verdict'] for r in rows)
    xhdfe = [r for r in rows if r['engine']=='xhdfe']
    failed = [r for r in xhdfe if r['verdict'] not in ('PASS','UNSUPPORTED_EXPECTED')]
    defaults = [r for r in xhdfe if r['method']=='auto' and r['precision']=='default'
                and not r.get('savefe') and not r.get('expected_rejection')]
    changed = [p for p,h in manifest['files'].items() if digest(Path(p)) != h]
    t01_rows = []
    for row in xhdfe:
        value = row.get('t01')
        if value is None and row.get('precision') in ('default','fast'):
            value = dict(applicable=True, g1_verdict='COVERAGE_MISSING',
                         passed=False, reason='T01 assessment absent')
        if value is not None and value.get('applicable'):
            t01_rows.append(value)
    t01_counts = Counter(r['g1_verdict'] for r in t01_rows)
    t01_failed = [r for r in t01_rows
                   if r['g1_verdict'] not in ('PASS','UNSUPPORTED_EXPECTED')]
    summary = dict(utc=datetime.now(timezone.utc).isoformat(), counts=dict(counts),
                   xhdfe_counts=dict(Counter(r['verdict'] for r in xhdfe)),
                   xhdfe_scoped_certified=not failed and not changed,
                   xhdfe_defaults_certified=bool(defaults) and all(r['verdict']=='PASS' for r in defaults) and not changed,
                   t01_counts=dict(t01_counts),
                   t01_scoped_pass=bool(t01_rows) and not t01_failed and not changed,
                   t01_joint_fit_metric_is_gate=False,
                   certificate_mode=manifest.get('certificate_mode','default'),
                   core24_x8_still_required=True,
                   default_rows=len(defaults),
                   custody_mismatches=changed, pending=manifest['pending'],
                   rows=rows)
    target = out/f'report_{len(list(out.glob("report_*.json")))+1:03d}.json'
    write_json(target, summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
    print('REPORT', target)
    return 0 if summary['xhdfe_scoped_certified'] and summary['t01_scoped_pass'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases')
    parser.add_argument('--interfaces',default='native,stata')
    parser.add_argument('--backends',default='cpu,cuda')
    parser.add_argument('--backend',choices=['cpu','cuda'])
    parser.add_argument('--filter')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--timeout',type=int,default=420)
    parser.add_argument('--stata',default=shutil.which('stata-mp') or '/usr/local/stata/stata-mp')
    parser.add_argument('--native-cpu-module',type=Path)
    parser.add_argument('--native-cuda-module',type=Path)
    parser.add_argument('--stata-package-root',type=Path)
    parser.add_argument('--certificate-mode',choices=('default','audit'),default='default')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT) or any(c in str(args.output) for c in ('"','\n','\r')):
        parser.error('output must be an unambiguous path inside this repository')
    if args.timeout <= 0 or args.limit is not None and args.limit <= 0:
        parser.error('timeout and limit must be positive')
    if args.action == 'prepare':
        for label, text, allowed in [('cases',args.cases or ','.join(NAMES),set(NAMES)),
                                     ('interfaces',args.interfaces,{'native','stata'}),
                                     ('backends',args.backends,{'cpu','cuda'})]:
            values = text.split(',')
            if not values or len(values)!=len(set(values)) or not set(values)<=allowed:
                parser.error(f'unknown or duplicate {label}: {text}')
        if 'cpu' not in args.backends.split(','):
            parser.error('prepare requires the CPU baseline; select CUDA alone at run time')
        interfaces = args.interfaces.split(',')
        backends = args.backends.split(',')
        if 'native' in interfaces and args.native_cpu_module is None:
            parser.error('prepare with native requires --native-cpu-module')
        if 'native' in interfaces and 'cuda' in backends and args.native_cuda_module is None:
            parser.error('prepare with native CUDA requires --native-cuda-module')
        if 'stata' in interfaces and args.stata_package_root is None:
            parser.error('prepare with Stata requires --stata-package-root')
    elif (args.native_cpu_module is not None or args.native_cuda_module is not None or
          args.stata_package_root is not None or args.certificate_mode != 'default'):
        parser.error('artifact and certificate-mode options are frozen by prepare')
    return {'prepare':prepare,'run':run,'report':report}[args.action](args) or 0


if __name__ == '__main__':
    sys.exit(main())
