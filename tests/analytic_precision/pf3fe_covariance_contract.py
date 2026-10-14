#!/usr/bin/env python3
"""PF3FE independent full-covariance regression and failed-refit contract.

Data are read in place; only coefficients, covariance, diagnostics and hashes
are persisted. Supply a pinned independently qualified aggregate oracle.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--oracle-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--max-iter', type=int, default=5000)
    parser.add_argument('--expect-refusal', action='store_true')
    parser.add_argument('--allow-refusal', action='store_true')
    parser.add_argument('--reverse-fes', action='store_true')
    parser.add_argument('--phase-proof', action='store_true')
    parser.add_argument('--no-cluster-dof-adjustment', action='store_true')
    parser.add_argument('--cluster', choices=['original', 'relabelled', 'non_nested'], default='original')
    args = parser.parse_args()
    if args.expect_refusal and args.allow_refusal:
        parser.error('Choose either a required refusal or an admissible bounded refusal')
    assert not args.output.exists()
    for key in list(os.environ):
        if key.startswith('XHDFE_'):
            del os.environ[key]
    os.environ.update(XHDFE_GPU_BACKEND=args.backend, CUDA_CACHE_DISABLE='1',
                      OPENBLAS_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
    if args.phase_proof:
        os.environ['XHDFE_MLSMR_TRACE'] = '1'
    import numpy as np
    import pandas as pd

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    module = args.module.resolve(strict=True)
    data = args.data.resolve(strict=True)
    oracle_path = args.oracle.resolve(strict=True)
    assert sha(oracle_path) == args.oracle_sha256
    oracle = json.loads(oracle_path.read_text())
    assert oracle['status'] in ['PASS_ORACLE', 'PASS_INDEPENDENT_ORACLE']
    assert bool(oracle.get('cluster_dof_adjustment_disabled', False)) == args.no_cluster_dof_adjustment
    module_sha = sha(module)
    source_stat = data.stat()
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    columns = ['y', *[f'x{j}' for j in range(1, 11)], 'indiv_id', 'firm_id', 'year']
    frame = pd.read_parquet(data, columns=columns)
    y = frame.y.to_numpy(np.float64)
    X = np.asfortranarray(frame[[f'x{j}' for j in range(1, 11)]].to_numpy(np.float64))
    fes = [pd.factorize(frame[name], sort=True)[0].astype(np.int32)
           for name in ['indiv_id', 'firm_id', 'year']]
    clusters = fes[0].copy()
    del frame

    def array_sha(value):
        value = np.asarray(value)
        digest = hashlib.sha256()
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(memoryview(np.ascontiguousarray(value).ravel(order='K')).cast('B'))
        return digest.hexdigest()

    original_input = hashlib.sha256('|'.join(array_sha(a) for a in [y, X, *fes, clusters]).encode()).hexdigest()
    assert original_input == oracle['input_sha256']
    if args.cluster == 'relabelled':
        clusters = (3*clusters+17).astype(np.int32)
    elif args.cluster == 'non_nested':
        assert oracle.get('cluster_variant') == 'row_mod_257'
        clusters = (np.arange(len(y), dtype=np.int64) % 257).astype(np.int32)
        assert array_sha(clusters) == oracle['actual_cluster_sha256']
    elif oracle.get('cluster_variant') not in [None, 'original']:
        raise AssertionError('Oracle clustering does not match the requested fixture')
    if args.reverse_fes:
        fes.reverse()

    model = core.HdfeRegressor(se_type='cluster', tol=1e-8, max_iter=args.max_iter,
        num_threads=args.threads, symmetric_sweep=False, absorption_method='auto',
        tolerance_mode='reghdfe-comparable', jacobi_relaxation=0.0,
        retain_fes=False, drop_singletons=True,
        dofadjustments='pairwise continuous' if args.no_cluster_dof_adjustment else None)
    i = np.arange(256)
    small_x = (2*(i % 2)-1).astype(float).reshape(-1, 1)
    small_fe = (i//8).astype(np.int32)
    small_y = 3+.5*small_x[:, 0]+small_fe % 4+.25*(2*((i//2) % 2)-1)
    model.fit(small_y, small_x, fes=[small_fe], clusters=small_fe)
    assert model.converged_ and np.isfinite(model.coef_).all()
    warm_beta = np.asarray(model.coef_).copy()
    warm_covariance = np.asarray(model.covariance_).copy()
    if args.backend == 'cuda':
        assert model.gpu_used_
    phase_log = args.output.with_suffix('.phases.log')

    @contextmanager
    def capture_phases():
        if not args.phase_proof:
            yield
            return
        with phase_log.open('x') as stream:
            saved = os.dup(2)
            try:
                os.dup2(stream.fileno(), 2)
                yield
            finally:
                os.dup2(saved, 2)
                os.close(saved)

    row = dict(backend=args.backend, cluster=args.cluster, max_iter=args.max_iter,
               reverse_fes=args.reverse_fes,
               cluster_dof_adjustment_disabled=args.no_cluster_dof_adjustment,
               prior_fit_succeeded=True, limits=dict(beta=1e-9, fullV_diagonal_scaled=1e-8),
               input_sha256=original_input, actual_cluster_sha256=array_sha(clusters))
    start = time.monotonic()
    try:
        with capture_phases():
            model.fit(y, X, fes=fes, clusters=clusters)
        if args.expect_refusal:
            raise AssertionError('Budget control unexpectedly returned estimates')
        b, v = np.asarray(model.coef_), np.asarray(model.covariance_)
        refb, refv = np.asarray(oracle['coef']), np.asarray(oracle['covariance'])
        assert b.shape == refb.shape and v.shape == refv.shape
        scale = np.sqrt(np.outer(refv.diagonal(), refv.diagonal()))
        beta_error = float(np.max(abs(b-refb)/np.maximum(1, abs(refb))))
        covariance_error = float(np.max(abs(v-refv)/scale))
        checks = dict(finite=bool(np.isfinite(b).all() and np.isfinite(v).all()),
            beta=beta_error <= 1e-9, fullV=covariance_error <= 1e-8,
            sample=array_sha(model.sample_index_) == oracle['sample_sha256'],
            nobs=int(model.nobs_) == oracle['nobs'],
            df_a=int(model.df_a_) == oracle['df_a_convention_independent'],
            convergence=bool(model.converged_ and model.precision_certified_),
            backend=bool(model.gpu_used_) == (args.backend == 'cuda'),
            threads=int(model.threads_used_) == args.threads,
            budget=int(model.num_iterations_) <= args.max_iter)
        if args.cluster == 'non_nested':
            means = X.mean(axis=0)
            forged = v.copy()
            forged[:-1, -1] = forged[-1, :-1] = -v[:-1, :-1]@means
            forged[-1, -1] = means@v[:-1, :-1]@means
            forged_error = float(np.max(abs(forged-refv)/scale))
            checks['false_nesting_rejected'] = forged_error > 1e-8
            row['false_nesting_fullV_error'] = forged_error
        row.update(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
            beta_scaled_error=beta_error, fullV_scaled_error=covariance_error,
            coef=b.tolist(), covariance=v.tolist(), iterations=int(model.num_iterations_),
            gpu_used=bool(model.gpu_used_), gpu_status_code=int(model.gpu_status_code_),
            threads_used=int(model.threads_used_), nobs=int(model.nobs_), df_a=float(model.df_a_))
    except RuntimeError as error:
        clean = model.lifecycle_state_ == 'failed' and not model.converged_ and not model.precision_certified_
        clean = clean and all(np.asarray(getattr(model, name)).size == 0
                              for name in ['coef_', 'covariance_', 'residuals_'])
        reason = str(error).lower()
        expected = any(text in reason for text in ['n1:', 'converg', 'iteration'])
        refusal_allowed = args.expect_refusal or args.allow_refusal
        row.update(status='EXPECTED_REFUSAL' if refusal_allowed and clean and expected else 'FAIL',
                   clean_failed_refit=bool(clean), error=str(error))
        if refusal_allowed and clean and expected:
            model.fit(small_y, small_x, fes=[small_fe], clusters=small_fe)
            recovered = bool(model.converged_ and
                np.array_equal(model.coef_, warm_beta) and
                np.array_equal(model.covariance_, warm_covariance))
            row['recovered_after_failure'] = recovered
            if not recovered:
                row['status'] = 'FAIL'
    row['fit_seconds'] = time.monotonic()-start
    if args.phase_proof:
        phases = {name:dict(units=int(units), team=int(team), workers=int(workers))
            for name,units,team,workers in re.findall(
                r'phase_workers phase=n1_graph_(\w+) units=(\d+) team=(\d+) active_workers=(\d+)',
                phase_log.read_text())}
        phase_pass = all(name in phases and
            phases[name]['team'] == args.threads and
            phases[name]['units'] >= args.threads and
            min(2, args.threads) <= phases[name]['workers'] <= args.threads
            for name in ['rhs', 'forward', 'backward', 'collect'])
        row.update(phase_workers=phases, parallel_phase_proof=phase_pass,
                   phase_log_sha256=sha(phase_log))
        if not phase_pass:
            row['status'] = 'FAIL'
    after = data.stat()
    custody = sha(module) == module_sha and sha(oracle_path) == args.oracle_sha256
    custody = custody and (source_stat.st_size, source_stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    if not custody:
        row['status'] = 'FAIL'
    row.update(custody_unchanged=custody, module=str(module), module_sha256=module_sha,
               oracle=str(oracle_path), oracle_sha256=args.oracle_sha256,
               test_sha256=sha(Path(__file__)), scope='PF3FE Comparable; real inputs read in place; independent full b/V oracle and explicit empty-state refusal controls; no performance verdict')
    with args.output.open('x') as stream:
        json.dump(row, stream, indent=2)
    print(json.dumps({k:row.get(k) for k in ['status', 'backend', 'cluster', 'max_iter',
        'beta_scaled_error', 'fullV_scaled_error', 'iterations', 'clean_failed_refit']}), flush=True)
    return 0 if row['status'] in ['PASS', 'EXPECTED_REFUSAL'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
