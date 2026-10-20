"""Exercise the matrix workers on newly generated synthetic data only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from run_matrix import HERE, PROJECT, version, write_new
from workload_contract import same_workload

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
assert out.is_relative_to(PROJECT)
def paths(arm, backend):
    root = (args.candidate if arm == 'candidate' else args.baseline).resolve()
    return root, root / ('build_cuda' if backend == 'cuda' else 'build')

out.mkdir()
scratch = out / 'scratch'
scratch.mkdir(mode=0o700)
row = np.arange(120000)
f1 = row % 200
f2 = (row // 200) % 100
x1 = np.sin(row*.007)
x2 = np.cos(row*.011)
y = .7*x1-.3*x2+f1*.01-f2*.003+np.sin(row*.019)
data = out / 'public_synthetic.parquet'
pd.DataFrame(dict(y=y, x1=x1, x2=x2, f1=f1, f2=f2)).to_parquet(data,index=False)
registry = out / 'REGISTRY.json'
write_new(registry, dict(specs=[dict(name='synthetic_smoke',path=str(data),y='y',
                                   x=['x1','x2'],fe=['f1','f2'],cluster='f1')]))
env = {k:v for k,v in os.environ.items() if not k.startswith('XHDFE_') and k!='LD_LIBRARY_PATH'}
env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
           OMP_NUM_THREADS='16',XHDFE_ABSORPTION_CACHE_MODE='off')
receipts = {}
for interface in ['native','stata']:
    for backend in ['cpu','cuda']:
        for mode in ['fast','comparable']:
            for arm in ['baseline','candidate']:
                label = f'{interface}_{backend}_{mode}_{arm}'
                target = out / (label+'.json')
                package,module=paths(arm,backend)
                common=['--registry',str(registry),'--dataset','synthetic_smoke','--mode',mode,
                        '--backend',backend,'--threads','16','--label',label]
                if interface=='native':
                    command=[sys.executable,'-B',str(HERE/'py_persistent_worker.py'),
                             '--module-dir',str(module),'--no-warmup',*common]
                    stdin=f'FIT {target}\nEXIT\n'
                else:
                    command=[sys.executable,'-B',str(HERE/'stata_fit_worker.py'),
                        '--package-dir',str(package),'--expected-version',
                        version(package),
                        '--scratch',str(scratch),'--out',str(target),*common]
                    stdin=None
                with (out/(label+'.log')).open('x') as log:
                    result=subprocess.run(command,input=stdin,text=True,stdout=log,
                        stderr=subprocess.STDOUT,env=env,cwd=out,timeout=420)
                assert result.returncode==0, label
                r=json.loads(target.read_text())
                assert r['success'] and r['converged'],label
                if backend=='cuda': assert r['gpu_used']==1 and r['gpu_status']=='used',label
                assert r['max_iter']==100000 and r['tol']==1e-8,label
                if interface=='native': assert r['warmup'] is None,label
                else: assert not r['store_residuals'] and r['workflow']=='ordinary',label
                receipts[label]=r
                print('PASS',label,flush=True)
for backend in ['cpu','cuda']:
    for mode in ['fast','comparable']:
        for arm in ['baseline','candidate']:
            n=receipts[f'native_{backend}_{mode}_{arm}']
            s=receipts[f'stata_{backend}_{mode}_{arm}']
            same_workload(n,s)
            try:
                same_workload(n, dict(s, store_residuals=True))
            except AssertionError as error:
                assert str(error) == 'Stata requests an extra residual absorption'
            else:
                raise AssertionError('Extra Stata recovery was accepted as the same workload')
            assert np.allclose(n['coef'],s['coef'],atol=1e-9,rtol=1e-9)
            assert np.allclose(n['covariance'],s['covariance'],atol=1e-12,rtol=1e-7)
write_new(out/'RESULT.json',dict(status='PASS',fits=len(receipts),cross_interface_workload_checks=8,
    extra_recovery_negative_controls=8,
    performance_evidence=False,data='newly generated synthetic trigonometric panel'))
print('MATRIX_HARNESS_SYNTHETIC_PASS',flush=True)
