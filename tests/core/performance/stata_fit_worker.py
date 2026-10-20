#!/usr/bin/env python3
"""One fresh Stata session per fit for the C2 matrix (06oct2026).

Derived from Verifications/perf_max_20260929/fecho_tiago_20260930/
stata_complete_worker.py. Changes: the do-file is scripts/stata_fit.do
(version and plugin path asserted, tol 1e-8, maxiter 100000,
absorptioncachemode(off)); STATATMP points to a private scratch directory, so
Stata's own temporary copies of the data never land in the repository; no
observation-level file is written (only estimates.dta with b and V, scalars,
and _datasignature digests). Exit code 0 is never taken as success: the
terminal sentinel must be present and the log must contain no r(#) error.
"""
import argparse
import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from py_persistent_worker import REGISTRY, as_list, load_spec, sha256  # noqa: E402

# The controller supplies the explicitly authorised private scratch path.
SCALARS = ('N', 'N_full', 'num_singletons', 'df_r', 'df_a', 'df_m', 'rank', 'iterations',
           'converged', 'precision_certified', 'gpu_used', 'threads_requested',
           'threads_effective', 'threads_used', 'parallel_workers_active', 'gpu_attempted',
           'gpu_absorption_iterations', 'absorption_method_used', 'abs_residual_rel', 'rss',
           'tss', 'tss_within', 'elapsed_seconds', 'data_read_seconds')
INTS = {'N', 'N_full', 'num_singletons', 'rank', 'iterations', 'converged',
        'precision_certified', 'gpu_used', 'threads_requested', 'threads_effective',
        'threads_used', 'parallel_workers_active', 'gpu_attempted',
        'gpu_absorption_iterations', 'absorption_method_used'}
ALIASES = {'N': 'nobs', 'N_full': 'nobs_full', 'df_r': 'df_resid',
           'absorption_method_used': 'method_code'}
METHODS = {0: 'auto', 1: 'gauss-seidel', 2: 'symmetric-gauss-seidel', 3: 'jacobi',
           4: 'schwarz', 5: 'lsmr', 6: 'mlsmr', 7: 'auto-mlsmr'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registry', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--package-dir', type=Path, required=True)
    ap.add_argument('--expected-version', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--mode', choices=('fast', 'comparable'), required=True)
    ap.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--extra-options', default='')
    ap.add_argument('--timeout', type=int, default=2400)
    ap.add_argument('--store-residuals', action='store_true')
    ap.add_argument('--profile', action='store_true')
    args = ap.parse_args()
    scratch = args.scratch.resolve()
    output = args.out.resolve()
    if output.exists():
        raise SystemExit('receipt exists; refusing to overwrite ' + str(output))
    package = args.package_dir.resolve()
    plugin = package / 'stata/xhdfe.plugin'
    ado = package / 'stata/xhdfe.ado'
    if not plugin.is_file() or not ado.is_file():
        raise FileNotFoundError(package)
    spec = load_spec(args.dataset, args.registry)
    data = Path(spec['path'])
    initial = data.stat()
    plugin_sha = sha256(plugin)
    ado_sha = sha256(ado)
    columns = list(dict.fromkeys([spec['y'], *spec['x'], *spec['fe'], *as_list(spec.get('cluster')),
        *[s['fe'] for s in spec.get('slopes', [])], *[s['var'] for s in spec.get('slopes', [])],
        *as_list(spec.get('group')), *as_list(spec.get('individual'))]))
    absorb = list(spec['fe'])
    for slope in spec.get('slopes', []):
        absorb.append(slope['fe'] + ('##c.' if slope.get('intercept') else '#c.') + slope['var'])
    options = ['residuals(__audit_residual)'] if args.store_residuals else []
    if spec.get('cluster'):
        options.append('vce(cluster ' + ' '.join(as_list(spec['cluster'])) + ')')
    for key in ('group', 'individual', 'aggregation'):
        if spec.get(key):
            options.append(key + '(' + spec[key] + ')')
    if not spec.get('singleton_drop', True):
        options.append('keepsingletons')
    if args.extra_options:
        options.append(args.extra_options)
    directory = output.parent / (output.stem + '.stata')
    directory.mkdir(parents=True)
    if not scratch.is_dir() or scratch.stat().st_uid != os.getuid() or scratch.stat().st_mode & 0o077:
        raise RuntimeError('The authorised private Stata temporary directory is unavailable')
    tmp = Path(tempfile.mkdtemp(dir=scratch, prefix=f'{args.dataset}__{args.backend}_{args.mode}__{output.stem}__'))
    mode = 'xhdfe-fast' if args.mode == 'fast' else 'reghdfe-comparable'
    stata_args = [str(package), str(data), spec['y'], ' '.join(spec['x']), ' '.join(absorb),
                  ' '.join(options), args.backend, mode, str(args.threads), str(directory),
                  ' '.join(columns), args.expected_version, str(int(args.store_residuals))]
    if any(any(char in value for char in ('"', '`', '$', '\n', '\r')) for value in stata_args):
        raise ValueError('unsupported quoting in a Stata argument')
    entry = directory / 'entry.do'
    with entry.open('x') as stream:
        stream.write('do "' + str(HERE / 'stata_fit.do') + '" ' +
                     ' '.join('"' + value + '"' for value in stata_args) + '\n')
    command = ['stata-mp', '-q', '-b', 'do', str(entry)]
    env = {k: v for k, v in os.environ.items()
           if k != 'LD_LIBRARY_PATH' and not k.startswith('XHDFE_')}
    env.update(STATATMP=str(tmp), TMPDIR=str(tmp), CUDA_CACHE_PATH=str(directory / 'cuda_cache'),
               XHDFE_GPU_BACKEND=args.backend)
    if args.profile:
        env['XHDFE_PROFILE_CPU'] = '1'
    load_before = os.getloadavg()
    start = time.monotonic()
    with (directory / 'process.log').open('x') as log:
        try:
            code = subprocess.run(command, cwd=directory, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            code = 124
    wall = time.monotonic() - start
    logfile = directory / 'receipt.log'
    text = logfile.read_text(errors='replace') if logfile.is_file() else ''
    batch_log = directory / 'entry.log'
    batch = batch_log.read_text(errors='replace') if batch_log.is_file() else ''
    errors = sorted(set(re.findall(r'^r\(\d+\);', text + '\n' + batch, flags=re.M)))
    sentinel = any(line.strip() == 'XHDFE_STATA_RECEIPT_PASS' for line in text.splitlines())
    version_line = re.search(r'^XHDFE_VERSION_CHECKED (.+)$', text, flags=re.M)
    success = (code == 0 and sentinel and not errors and (directory / 'estimates.dta').is_file()
               and version_line is not None)
    leftovers = sorted(p.name for p in tmp.iterdir()) if tmp.is_dir() else []
    receipt = dict(interface='stata', workflow='residuals()' if args.store_residuals else 'ordinary',
                   store_residuals=args.store_residuals, profile=args.profile, dataset=args.dataset, mode=args.mode,
                   backend=args.backend, threads=args.threads, tol=1e-8, max_iter=100000,
                   absorption_cache='off', label=args.label, retain_fes=False, weights='none',
                   method_requested='auto', module=str(plugin), module_sha256=plugin_sha,
                   module_sha256_after=sha256(plugin), ado=str(ado), ado_sha256=ado_sha,
                   ado_sha256_after=sha256(ado), expected_version=args.expected_version,
                   version_checked=version_line.group(1).strip() if version_line else None,
                   source_path=str(data), source_size=initial.st_size,
                   source_mtime_ns=initial.st_mtime_ns, success=int(success), returncode=code,
                   stata_errors=errors, command=command, extra_options=args.extra_options,
                   wall_seconds=wall,
                   process_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                   loadavg_before=load_before, loadavg_after=os.getloadavg(),
                   statatmp=str(tmp), statatmp_leftovers=leftovers,
                   error='' if success else 'Stata failed, sentinel/version absent or r(#) in log')
    if success:
        import numpy as np
        import pandas as pd
        frame = pd.read_stata(directory / 'estimates.dta', convert_categoricals=False)
        beta = frame['beta1'].to_numpy()
        size = len(beta)
        cov = frame[['cov' + str(j + 1) for j in range(size)]].to_numpy()
        stderr = np.sqrt(np.diag(cov))
        for name in SCALARS:
            value = float(frame[name].iloc[0])
            key = ALIASES.get(name, name)
            receipt[key] = None if not math.isfinite(value) else (int(value) if name in INTS else value)
        receipt.update(coef=beta.tolist(), coef_hex=[float(v).hex() for v in beta],
                       stderr=stderr.tolist(), stderr_hex=[float(v).hex() for v in stderr],
                       covariance=cov.tolist(),
                       covariance_hex=[[float(v).hex() for v in row] for row in cov],
                       terms=frame['term'].tolist(),
                       method=METHODS.get(receipt.get('method_code'), str(receipt.get('method_code'))),
                       gpu_status=frame['gpu_status'].iloc[0],
                       actual_backend=frame['actual_backend'].iloc[0],
                       input_signature=frame['input_signature'].iloc[0],
                       sample_signature=frame['sample_signature'].iloc[0],
                       residual_signature=frame['residual_signature'].iloc[0],
                       e_version=frame['e_version'].iloc[0])
        receipt['rss_hex'] = float(receipt['rss']).hex() if receipt.get('rss') is not None else None
    current = data.stat()
    if (receipt['module_sha256_after'] != plugin_sha or receipt['ado_sha256_after'] != ado_sha or
            current.st_size != initial.st_size or current.st_mtime_ns != initial.st_mtime_ns):
        receipt.update(success=0, error='source or loaded artifact custody changed')
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=1, allow_nan=False)
    print('PASS' if receipt['success'] else 'FAIL', args.label, args.dataset, args.backend, args.mode,
          receipt.get('elapsed_seconds'), receipt['error'], flush=True)
    return int(not receipt['success'])


if __name__ == '__main__':
    raise SystemExit(main())
