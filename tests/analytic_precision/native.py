"""One isolated native C++/Python fit, with raw results for external grading."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

job_path = Path(sys.argv[1])
job = json.loads(job_path.read_text())
case_dir = Path(job['fixture'])
meta = json.loads((case_dir/'case.json').read_text())
frame = pd.read_stata(case_dir/'long.dta')
os.environ['XHDFE_GPU_BACKEND'] = job['backend']
module_path = Path(job['module']).resolve()
if hashlib.sha256(module_path.read_bytes()).hexdigest() != job['module_sha256']:
    raise RuntimeError('native module custody mismatch')
spec = importlib.util.spec_from_file_location('py_hdfe_v11', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
options = dict(num_threads=2, max_iter=100000, absorption_method=job['method'],
               se_type=meta['vce'], fit_intercept=meta['intercept'],
               retain_fes=job.get('savefe', False))
if job['precision'] == 'strict':
    options.update(tol=1e-12, tolerance_mode='reghdfe-comparable' if meta['slopes'] else 'strict-residual')
    if meta['slopes']:
        options['convergence'] = 'both'
if job['precision'] == 'invalid-strict-slope':
    options['tolerance_mode'] = 'strict-residual'
if job['precision'] == 'fast':
    options['tolerance_mode'] = 'xhdfe-fast'
model = module.HdfeRegressor(**options)
kwargs = dict(fes=[frame[c].to_numpy(dtype=np.int64) for c in meta['fes']])
if meta['weight']:
    kwargs.update(weights=frame.weight.to_numpy(), fweights=meta['weight'] == 'fw')
if meta['clusters']:
    kwargs['clusters'] = frame[meta['clusters']].to_numpy(dtype=np.int64)
if meta['slopes']:
    kwargs['slopes'] = [(i, frame[v].to_numpy(), intercept) for i,v,intercept in meta['slopes']]
if meta['kind'] != 'standard':
    kwargs['group'] = frame.group.to_numpy(dtype=np.int64)
if meta['kind'] == 'group_individual':
    kwargs.update(individual=frame.individual.to_numpy(dtype=np.int64), aggregation=meta['aggregation'])
start = time.monotonic()
result = dict(module=str(module_path))
try:
    model.fit(frame.y.to_numpy(), frame[['x1','x2','x3']].to_numpy(), **kwargs)
    selected = np.asarray(model.sample_index_, dtype=int)
    beta = np.asarray(model.coef_)
    result.update(rc=0, beta=beta[:3].tolist(), cons=float(beta[3]) if len(beta)>3 else 0.,
                  V=np.asarray(model.covariance_)[:3,:3].tolist(),
                  N=model.nobs_, rss=model.rss_, df_r=model.df_resid_, df_a=model.df_a_,
                  singletons=model.num_singletons_, iterations=model.num_iterations_,
                  converged=int(model.converged_), method_used=int(model.absorption_method_used),
                  certified=int(model.precision_certified_), gpu_used=int(model.gpu_used_),
                  gpu_status=int(model.gpu_status_code_), threads=model.threads_used_,
                  groups=frame.group.to_numpy()[selected].astype(int).tolist(),
                  residuals=np.asarray(model.residuals_).tolist())
    if job.get('savefe'):
        effects = np.asarray(model.fe_effects_)
        result['recovered_fe'] = effects.sum(axis=0).tolist()
        result['recovery_converged'] = bool(model.fe_recovery_converged_)
except Exception as error:
    result.update(rc=1, error=str(error), error_type=type(error).__name__,
                  converged=int(model.converged_),
                  iterations=int(model.num_iterations_),
                  certified=int(model.precision_certified_),
                  method_used=int(model.absorption_method_used),
                  gpu_used=int(model.gpu_used_),
                  gpu_status=int(model.gpu_status_code_),
                  threads=int(model.threads_used_),
                  lifecycle=model.lifecycle_state_,
                  generation=int(model.generation_),
                  beta=np.asarray(model.coef_).tolist(),
                  V=np.asarray(model.covariance_).tolist(),
                  residuals=np.asarray(model.residuals_).tolist())
result['seconds'] = time.monotonic()-start
with (job_path.parent/'raw.json').open('x') as handle:
    json.dump(result, handle, allow_nan=True)
    handle.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('residuals','groups','recovered_fe')}), flush=True)
