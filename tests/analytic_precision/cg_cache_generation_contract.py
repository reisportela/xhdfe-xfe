"""An old falsely certified CG transform must not survive cache migration."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from cg_projection_full_covariance_contract import fixture, reference


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(job_path):
    job = json.loads(job_path.read_text())
    os.environ.update(XHDFE_GPU_BACKEND='cpu', XHDFE_MOBILITY_MODE='off',
                      XHDFE_FE_STRUCTURE_MODE='off',
                      XHDFE_ABSORPTION_CACHE_MODE=job['cache_mode'],
                      XHDFE_ABSORPTION_CACHE=job['cache'])
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', job['module'])
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    final_step = job.get('fixture') == 'final-step'
    if final_step:
        import pandas as pd
        directory = Path(__file__).with_name('fixtures')
        oracle = json.loads((directory/'cg_seed1_2017_oracle.json').read_text())
        data_path = directory/'cg_seed1_2017.csv'
        assert digest(data_path) == oracle['csv_sha256']
        data = pd.read_csv(data_path, float_precision='round_trip')
        data = data.sort_values(['sid', 'year'], ascending=False)
        y, X = data.dagg.to_numpy(), data.x.to_numpy()[:, None]
        fes = [pd.factorize(data[k], sort=True)[0].astype(np.int32)
               for k in ('sid', 'year', 'entry_age')]
        clusters = pd.factorize(data.did, sort=True)[0].astype(np.int32)
        ref = oracle['references']['ols']
        expected = np.asarray(ref['b'], dtype=float)
        covariance = np.asarray(ref['V'], dtype=float)
        margin = np.asarray(ref['covariance_arithmetic_margin'])
        residual = None
    else:
        y, X, fes, weights, clusters = fixture('rare', False)
        expected, covariance, residual = reference(y, X, fes, weights, clusters)
        margin = np.zeros_like(covariance)
    reg = core.HdfeRegressor(se_type='cluster', num_threads=1 if final_step else 2,
                            fit_intercept=True, drop_singletons=False,
                            ssc_k_adj=False, ssc_g_adj=False,
                            **({'stats_style': 'legacy'} if final_step else {}))
    reg.fit(y, X, fes=fes, clusters=clusters[:, None])
    b = np.asarray(reg.coef_)
    V = np.asarray(reg.covariance_)
    result = dict(module=job['module'], module_sha256=digest(Path(job['module'])),
                  parameters=b.tolist(), covariance=V.tolist(),
                  residual_sha256=hashlib.sha256(np.asarray(reg.residuals_).tobytes()).hexdigest(),
                  b_error=float(np.max(np.abs(b-expected)/np.maximum(1, np.abs(expected)))),
                  V_error=float(np.max(np.abs(V-covariance)/
                                       np.sqrt(np.outer(covariance.diagonal(), covariance.diagonal())))),
                  V_ratio=float(np.max(np.abs(V-covariance)/(
                      1e-8*np.sqrt(np.outer(covariance.diagonal(), covariance.diagonal()))+margin))),
                  residual_error=(float(np.max(np.abs(np.asarray(reg.residuals_)-residual)))
                                  if residual is not None else None),
                  iterations=int(reg.num_iterations_), converged=bool(reg.converged_),
                  certified=bool(reg.precision_certified_), gpu_used=bool(reg.gpu_used_))
    with Path(job['result']).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-module', type=Path)
    p.add_argument('--module', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--fixture', choices=('projection', 'final-step'), default='projection')
    p.add_argument('--worker', type=Path)
    args = p.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    if args.baseline_module is None or args.module is None or args.output is None:
        p.error('baseline-module, module and output are required')
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output/'old_cg.cache'
    results = {}
    cache_digest = None
    for role, module, mode in [('old_write', args.baseline_module, 'write'),
                               ('old_read', args.baseline_module, 'read'),
                               ('candidate_read', args.module, 'read')]:
        result = args.output/f'{role}.json'
        job = args.output/f'{role}_job.json'
        job.write_text(json.dumps(dict(module=str(module.resolve()), cache=str(cache.resolve()),
                                       fixture=args.fixture,
                                       cache_mode=mode, result=str(result.resolve())))+'\n')
        run = subprocess.run([sys.executable, '-B', __file__, '--worker', str(job)],
                             capture_output=True, text=True, timeout=420)
        (args.output/f'{role}.stdout').write_text(run.stdout)
        (args.output/f'{role}.stderr').write_text(run.stderr)
        run.check_returncode()
        results[role] = json.loads(result.read_text())
        if role == 'old_write':
            assert cache.is_file(), 'old build did not write an absorption cache'
            cache_digest = digest(cache)
        else:
            assert digest(cache) == cache_digest, 'read-only cache use changed old bytes'
    old, reread, new = (results[k] for k in ('old_write', 'old_read', 'candidate_read'))
    assert old['converged'] and old['certified']
    assert old['b_error'] > 1e-9 or old['V_ratio'] > 1
    assert old['parameters'] == reread['parameters']
    assert old['covariance'] == reread['covariance']
    assert new['converged'] and new['certified'] and not new['gpu_used']
    assert new['b_error'] <= 1e-9 and new['V_ratio'] <= 1
    if new['residual_error'] is not None:
        assert new['residual_error'] <= 1e-7
    report = dict(status='PASS', fixture=args.fixture, cache_sha256=cache_digest,
                  old_b_error=old['b_error'], candidate_b_error=new['b_error'],
                  candidate_V_error=new['V_error'], old_cache_unchanged=True,
                  results=results)
    with (args.output/'RESULT.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
