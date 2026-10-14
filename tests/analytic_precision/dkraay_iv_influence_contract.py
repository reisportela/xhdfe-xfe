"""Known-score IV oracle: DK and its bandwidth-one cluster identity.

The strong direction is the sum of two weakly distinguished IV scores.
Equal regressor means make the recovered intercept depend on that direction.
MP90/MP120 references use Walsh scores, never candidate output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import mpmath as mp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/analytic_precision'))
from iv_intercept_covariance_contract import fixture


def oracle(case, bandwidth, digits):
    with mp.workdps(digits):
        n = len(case['y'])
        q = mp.matrix([[mp.mpf(int(case['score_numerator'][i])) / 5 + case['means'][0],
                        mp.mpf(int(case['score_numerator'][i])) / 5
                        + mp.mpf(case['delta']) * int(case['score_direction'][i]) + case['means'][1],
                        1] for i in range(n)])
        # These are exact projections on the represented, balanced Walsh inputs.
        weights = [mp.mpf(float(v)) for v in case['weights']]
        Wq = mp.matrix([[weights[i]*q[i,j] for j in range(3)] for i in range(n)])
        bread = (q.T * Wq) ** -1
        levels = sorted(set(map(int, case['cl1'])))
        h = mp.matrix([[mp.fsum(weights[i] * q[i,j] * int(case['residual'][i])
                               for i in range(n) if case['cl1'][i] == t)
                        for j in range(3)] for t in levels])
        meat = h.T * h
        for lag in range(1, bandwidth):
            cross = h[lag:,:].T * h[:-lag,:]
            meat += (1 - mp.mpf(lag) / bandwidth) * (cross + cross.T)
        effective = mp.fsum(weights) if case['weight_type'] == 'frequency' else mp.mpf(n)
        scale = (effective-1) / (effective-3) * len(levels) / (len(levels)-1)
        V = scale * bread * meat * bread.T
        return np.array(V.tolist(), dtype=float)


p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['cpu','cuda'], default='cpu')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--package', type=Path, default=ROOT/'stata')
p.add_argument('--extended', action='store_true', help='Add held-out weighted, omitted, no-constant/no-FE and permuted cases')
a = p.parse_args()
out = a.output.resolve()
package = a.package.resolve()
assert out.is_relative_to(ROOT), 'Keep synthetic outputs inside the repository'
out.mkdir(exist_ok=False)
specs = []
commands = ['version 16', 'clear all', 'set more off', 'set linesize 255',
            f'adopath ++ "{package}"', 'which xhdfe', 'which xhdfe_estat']
designs = [(e,m,'base') for e in (15,27) for m in ((0,0),(1,-1),(1,1))]
if a.extended:
    designs += [(23,(2,2),v) for v in ('analytic','frequency','noconstant','noabsorb','omitted','permuted')]
for exponent, means, variant in designs:
        case = fixture(2.0**-exponent, means, variant in ('analytic','frequency'), variant)
        for bandwidth in (1,3):
            ref = oracle(case, bandwidth, 90)
            ref120 = oracle(case, bandwidth, 120)
            scale = np.sqrt(np.outer(np.diag(ref120), np.diag(ref120)))
            assert np.max(np.abs(ref-ref120)/scale) < 1e-14
            for mode in ('xhdfe-fast','reghdfe-comparable'):
                for kind in ('dkraay','cluster') if bandwidth == 1 else ('dkraay',):
                    name = f'case_{len(specs):03d}'
                    df = pd.DataFrame(dict(y=case['y'], x1=case['x'][:,0], x2=case['x'][:,1],
                                           z1=case['z'][:,0], z2=case['z'][:,1],
                                           fe=case['fe'], time=case['cl1'], row=np.arange(1,513),
                                           weight=case['weights'], omitted=np.zeros(512)))
                    if variant == 'permuted':
                        df = df.sample(frac=1, random_state=1729).reset_index(drop=True)
                        # xtset must not undo the intended observation order.
                        df['row'] = np.arange(1, len(df)+1)
                    df.to_stata(out/(name+'.dta'), write_index=False, version=118)
                    vce = f'dkraay {bandwidth}' if kind == 'dkraay' else 'cluster time'
                    absorb = 'noabsorb' if variant == 'noabsorb' else 'absorb(fe)'
                    weight = '[aw=weight]' if variant == 'analytic' else '[fw=weight]' if variant == 'frequency' else ''
                    constant = 'noconstant' if variant == 'noconstant' else ''
                    extra = 'omitted' if variant == 'omitted' else ''
                    expected = ref120.copy()
                    beta = [.5,-.25,3.]
                    if variant == 'noconstant':
                        expected = expected[:2,:2]
                        beta = beta[:2]
                    if variant == 'omitted':
                        expected = np.zeros((4,4))
                        expected[np.ix_([0,1,3],[0,1,3])] = ref120
                        beta = [.5,-.25,0.,3.]
                    size = len(beta)
                    commands += [f'use "{out}/{name}.dta", clear', 'quietly xtset row time',
                        f'capture noisily xhdfe y x1 x2 {extra} {weight}, {absorb} {constant} endogenous(x1 x2) instruments(z1 z2) vce({vce}) tolerancemode({mode}) gpubackend({a.backend}) numthreads(2)',
                        'local fit_rc = _rc', f'file open receipt using "{out}/{name}.txt", write text',
                        'file write receipt "rc " (`fit_rc\') _n',
                        'capture confirm matrix e(b)', 'local has_estimates = !_rc',
                        'file write receipt "has_estimates " (`has_estimates\') _n',
                        'if !`fit_rc\' {', 'matrix b=e(b)', 'matrix V=e(V)',
                        'file write receipt "b"', f'forvalues j=1/{size} {{',
                        'file write receipt " " %24.17e (b[1,`j\'])', '}',
                        'file write receipt _n "V"', f'forvalues j=1/{size} {{', f'forvalues k=1/{size} {{',
                        'file write receipt " " %24.17e (V[`j\',`k\'])', '}', '}',
                        'file write receipt _n "gpu " (e(gpu_used)) _n "converged " (e(converged)) _n',
                        '}', 'file close receipt']
                    specs.append(dict(name=name, exponent=exponent, means=means, kind=kind,
                                      bandwidth=bandwidth, mode=mode, variant=variant,
                                      beta_reference=beta, reference=expected.tolist()))
commands += ['display "DK_WEAK_DIRECTION_PROBE_COMPLETE"','exit']
do = out/'probe.do'
do.write_text('\n'.join(commands)+'\n')
env = os.environ.copy()
env.update(CUDA_CACHE_PATH=str(out/'cuda'), STATATMP=str(out), TMPDIR=str(out))
proc = subprocess.run(['/usr/local/stata/stata-mp','-b','do',str(do)], cwd=out, env=env, timeout=420)
log = (out/'probe.log').read_text(errors='replace')
for row in specs:
    actual = {s.split()[0]: [float(v) if v != '.' else np.nan for v in s.split()[1:]]
              for s in (out/(row['name']+'.txt')).read_text().splitlines()}
    row['rc'] = actual['rc'][0]
    row['status'] = 'FAIL'
    if a.backend == 'cuda' and row['variant'] == 'noabsorb':
        if (row['rc'] == 198 and actual['has_estimates'][0] == 0 and
                'CUDA absorption requires at least one fixed-effect dimension' in log):
            row['status'] = 'EXPECTED_REFUSAL'
    if not row['rc']:
        ref = np.array(row['reference'])
        cov = np.array(actual['V']).reshape(ref.shape)
        row['V'] = cov.tolist()
        row['b'] = actual['b']
        row['b_error'] = float(np.max(np.abs(np.array(row['b'])-row['beta_reference'])))
        norms = np.sqrt(np.outer(np.diag(ref),np.diag(ref)))
        difference = np.abs(cov-ref)
        row['V_scaled_error'] = float(np.max(np.divide(difference, norms, out=np.zeros_like(difference), where=norms>0)))
        zero_rows_ok = bool(np.all(difference[norms==0] == 0))
        row['gpu'] = actual['gpu'][0]
        row['converged'] = actual['converged'][0]
        if (row['b_error'] <= 1e-9 and row['V_scaled_error'] <= 1e-8
                and zero_rows_ok and np.isfinite(cov).all()
                and row['converged'] == 1 and row['gpu'] == (a.backend == 'cuda')):
            row['status'] = 'PASS'
    print(row['name'],row['exponent'],row['means'],row['kind'],row['bandwidth'],row['mode'],
          row['status'],row.get('b_error'),row.get('V_scaled_error'),flush=True)
complete = bool(re.search(r'^DK_WEAK_DIRECTION_PROBE_COMPLETE$',log,re.M))
receipt = dict(backend=a.backend,complete=complete,rows=specs,
               reference='Walsh known scores; direct MP90/MP120 Bartlett sandwich; limits b 1e-9, scaled full V 1e-8',
               hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__),package/'xhdfe.ado',package/'xhdfe.plugin',
                        ROOT/'src/dkraay_covariance.hpp']})
(out/'results.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(not complete or any(r['status'] not in ('PASS','EXPECTED_REFUSAL') for r in specs))
