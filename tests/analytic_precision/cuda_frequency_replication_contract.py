"""Check equivalence across the CUDA Comparable row-count boundary."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda_weighted_cpu_route_contract import cov240k, grade, weight_vector, weighted_oracle


def run(module):
    for key in tuple(os.environ):
        if key.startswith('XHDFE_'):
            del os.environ[key]
    os.environ.update(XHDFE_GPU_BACKEND='cuda', XHDFE_ABSORPTION_CACHE_MODE='off',
                      XHDFE_MOBILITY_MODE='off', XHDFE_FE_STRUCTURE_MODE='off')
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', module.resolve())
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    y, x, fes = cov240k(4)
    repeat = 9
    rows = np.repeat(np.arange(len(y)), repeat)
    assert len(y) < 2000000 < len(rows)
    yy, xx = y[rows], np.asfortranarray(x[rows])
    expanded_fes = [effect[rows] for effect in fes]
    for kind in ('literal', 'nonuniform'):
        f = np.ones(len(y)) if kind == 'literal' else weight_vector('fw', len(y))
        reference = weighted_oracle(y, x, fes, repeat*f, True)
        sd = np.sqrt(np.diag(reference['V']))
        recordings = []
        for label, a, b, effects, weights in (
            ('compact', y, x, fes, repeat*f),
            ('expanded', yy, xx, expanded_fes, None if kind == 'literal' else f[rows]),
        ):
            reg = core.HdfeRegressor(se_type='cluster', num_threads=8, tol=1e-8,
                                     drop_singletons=True, max_iter=100000)
            reg.fit(a, b, fes=effects, weights=weights, fweights=weights is not None,
                    clusters=[effects[0]])
            assert reg.gpu_used_ and reg.gpu_status_code_ == 1 and reg.precision_certified_
            assert reg.converged_ and reg.num_iterations_ <= 100000
            assert reg.nobs_effective_ == repeat*f.sum() and reg.df_a_ == reference['df_a']
            result = {'b': np.asarray(reg.coef_), 'V': np.asarray(reg.covariance_)}
            error = grade(reference, result)
            scaled = float(np.max(np.abs(result['V']-reference['V']) / np.outer(sd, sd)))
            assert error['b_scaled'] <= 1e-8 and error['se_rel'] <= 1e-7 and scaled <= 1e-8, (kind,label,error,scaled)
            recordings.append((reg.df_resid_, np.asarray(reg.residuals_).copy(), result))
            print(kind, label, reg.nobs_, reg.nobs_effective_, reg.num_iterations_,
                  reg.absorption_method_used, scaled, flush=True)
        first, second = recordings
        assert first[0] == second[0]
        residual_difference = np.linalg.norm(first[1][rows]-second[1]) / np.linalg.norm(second[1])
        assert residual_difference <= 1e-8, (kind, residual_difference)
        print(kind, 'residual relative difference', residual_difference, flush=True)
    print('CUDA_FREQUENCY_REPLICATION_PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module', type=Path)
    run(parser.parse_args().module)
