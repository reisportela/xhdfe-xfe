"""Grade artifacts against sealed fixture truth; no estimator is the oracle."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def assess(job, result, fixture):
    oracle = json.loads((fixture/'oracle.json').read_text())
    metadata = json.loads((fixture/'case.json').read_text())
    out = dict(job, result=result)
    if result.get('timeout'):
        return dict(out, verdict='TIMEOUT')
    if result.get('harness_error'):
        return dict(out, verdict='HARNESS_ERROR')
    if result.get('rc', 0) != 0 or result.get('error'):
        diagnostic = result.get('error', '')+' '+result.get('diagnostic', '')
        expected = job.get('expected_rejection') and re.search(job['reject_pattern'], diagnostic, re.I)
        verdict = 'UNSUPPORTED_EXPECTED' if expected else 'ESTIMATION_ERROR'
        return dict(out, verdict=verdict)
    if job.get('expected_rejection'):
        status = job.get('rejection_status')
        if status and all(result.get(k)==v for k,v in status.items()):
            return dict(out, verdict='UNSUPPORTED_EXPECTED')
        return dict(out, verdict='FAILED_TO_REJECT')
    C = np.asarray(oracle['contrast'])
    beta = np.asarray(result['beta'])
    V = np.asarray(result['V'])
    reference_V = np.asarray(oracle['V_ols'] if job['engine'] == 'ols' else oracle['V_contract'])
    delta = C@(beta-np.asarray(oracle['beta']))
    vb, vg = C@V@C.T, C@reference_V@C.T
    rows = pd.read_stata(fixture/'explicit.dta')
    wanted = rows.loc[rows.expected_sample == 1].set_index('group')
    ids = np.asarray(result['groups'], dtype=int)
    residuals = np.asarray(result['residuals'])
    sample_ok = len(ids) == len(set(ids)) == len(wanted) and set(ids) == set(wanted.index)
    numerical_ok = np.isfinite(beta).all() and np.isfinite(residuals).all()
    metrics = dict(beta_error=float(np.max(np.abs(delta))),
                   covariance_error=float(np.max(np.abs(vb-vg))),
                   covariance_relative_error=float(np.max(np.abs(vb-vg))/max(np.max(np.abs(vg)), 1e-30)),
                   rss_error=float(abs(result['rss']-oracle['rss'])),
                   sample_ok=bool(sample_ok), N_error=float(abs(result['N']-oracle['N'])))
    metrics['singleton_error'] = 0. if job['engine']=='ols' else float(abs(result.get('singletons', 0)-oracle['dropped']))
    if sample_ok:
        truth = wanted.loc[ids]
        e = truth.exact_residual.to_numpy()
        w = truth.weight.to_numpy() if metadata['weight'] else np.ones(len(ids))
        rss_scale = len(ids)/w.sum() if metadata['weight'] in ('aw', 'pw') else 1.
        X = truth[['x1','x2','x3']].to_numpy()
        D = np.load(fixture/'design.npz')['D'][ids]
        scores = np.c_[X, D].T@(w*residuals)
        col_norm = np.sqrt(np.sum(w[:, None]*np.c_[X, D]**2, axis=0))
        normalized_scores = np.abs(scores)/np.maximum(col_norm*np.sqrt(np.sum(w*residuals**2)), 1e-30)
        metrics.update(residual_error=float(np.max(np.abs(residuals-e))),
                       rss_direct_error=float(abs(result['rss']-rss_scale*np.dot(w, residuals**2))),
                       max_normalized_score=float(np.max(normalized_scores)))
        if job.get('savefe'):
            recovered = np.asarray(result.get('recovered_fe', []))
            expected = truth.y.to_numpy()-X@beta-residuals-result['cons']
            metrics['fe_error'] = float(np.max(np.abs(recovered-expected))) if recovered.shape == expected.shape else float('inf')
    strict = job.get('precision') == 'strict'
    beta_tol = 1e-10 if strict else 1e-8
    point_ok = numerical_ok and sample_ok and metrics['N_error'] == 0 and metrics['beta_error'] <= beta_tol
    point_ok &= metrics['singleton_error'] == 0
    point_ok &= metrics['rss_error'] <= 1e-8+1e-10*abs(oracle['rss'])
    point_ok &= metrics.get('residual_error', float('inf')) <= 1e-7
    point_ok &= metrics.get('rss_direct_error', float('inf')) <= 1e-8+1e-10*abs(oracle['rss'])
    covariance_ok = np.isfinite(V).all() and np.allclose(vb, vg, atol=1e-12, rtol=1e-7)
    backend_ok = job.get('backend') != 'cuda' or result.get('gpu_used') == 1
    recovery_ok = not job.get('savefe') or metrics.get('fe_error', float('inf')) <= 1e-6
    converged = result.get('converged') == 1
    expected_codes = {'gauss-seidel': [1], 'symmetric-gauss-seidel': [2], 'jacobi': [3],
                      'schwarz': [4], 'lsmr': [5], 'mlsmr': [6], 'auto-mlsmr': [1,2,6]}
    method_ok = True
    if job['engine'] == 'xhdfe' and len(metadata['fes']) >= 2 and job.get('method') in expected_codes:
        codes = [2] if metadata['slopes'] and job['method']=='jacobi' else expected_codes[job['method']]
        method_ok = result.get('method_used') in codes
    if not backend_ok:
        verdict = 'BACKEND_MISMATCH'
    elif not method_ok:
        verdict = 'METHOD_MISMATCH'
    elif not sample_ok:
        verdict = 'FAIL_SAMPLE_PROVENANCE'
    elif not point_ok:
        verdict = 'FALSE_CONVERGENCE' if converged else 'NOT_CONVERGED'
    elif not covariance_ok:
        verdict = 'FAIL_COVARIANCE'
    elif not recovery_ok:
        verdict = 'FAIL_FE_RECOVERY'
    elif not converged:
        verdict = 'NOT_CONVERGED'
    else:
        verdict = 'PASS'
    return dict(out, metrics=metrics, point_estimate_pass=bool(point_ok),
                covariance_pass=bool(covariance_ok), verdict=verdict)


def selftest(oracle, metadata, groups, fixture):
    """Mutations must not obtain a numerical PASS, including a forged certificate."""
    truth = pd.read_stata(fixture/'explicit.dta')
    truth = truth.loc[truth.expected_sample == 1]
    good = dict(rc=0, beta=oracle['beta'], V=oracle['V_contract'], N=oracle['N'],
                rss=oracle['rss'], groups=truth.group.tolist(),
                residuals=truth.exact_residual.tolist(), converged=1, gpu_used=0,
                singletons=oracle['dropped'])
    job = dict(engine='xhdfe', backend='cpu', case=metadata['name'], precision='default')
    assert assess(job, good, fixture)['verdict'] == 'PASS'
    bad = dict(good, beta=[oracle['beta'][0]+.01, *oracle['beta'][1:]], certified=1)
    assert assess(job, bad, fixture)['verdict'] == 'FALSE_CONVERGENCE'
    assert assess(job, dict(good, N=good['N']+1), fixture)['verdict'] != 'PASS'
    assert assess(job, dict(good, groups=[]), fixture)['verdict'] == 'FAIL_SAMPLE_PROVENANCE'
    assert assess(dict(job, backend='cuda'), good, fixture)['verdict'] == 'BACKEND_MISMATCH'
    assert assess(job, dict(good, V=(np.asarray(good['V'])*2).tolist()), fixture)['verdict'] == 'FAIL_COVARIANCE'
    assert assess(job, dict(good, residuals=[float('nan')]*len(truth)), fixture)['verdict'] != 'PASS'
    negative = dict(job, expected_rejection=True, reject_pattern='deliberate unsupported mode')
    assert assess(negative, dict(rc=199, error='command missing'), fixture)['verdict'] == 'ESTIMATION_ERROR'
    assert assess(negative, dict(rc=198, error='deliberate unsupported mode'), fixture)['verdict'] == 'UNSUPPORTED_EXPECTED'
