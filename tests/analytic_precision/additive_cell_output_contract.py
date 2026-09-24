"""Held-out analytic WLS/HC1 cases for the bounded additive-cell correction."""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path

import numpy as np

from n05_receipt import parse_n05_receipt
from precision_contract import (COMPARABLE_BETA_REL, COMPARABLE_V_REL,
    FE_RECOVERY_ABS, RESIDUAL_ABS, covariance_operation_count,
    oracle_covariance_arithmetic_margin)


def fixture(left, right, beta, exponent, representation):
    i = np.arange(64 * left * right)
    a, b = (i // 64) // right, (i // 64) % right
    A, B = 2 * (i % 2) - 1, 2 * ((i // 2) % 2) - 1
    x = A + a / 4 - b / 2
    noise = 2. ** -exponent * B
    y = beta * x + 8 * a - 16 * b + noise
    w = 2. ** ((a + 2 * b) % 5 - 2)
    cell_w = [Fraction(2) ** ((aa + 2 * bb) % 5 - 2)
              for aa in range(left) for bb in range(right)]
    gram = 64 * sum(cell_w)
    meat = 64 * sum(v * v for v in cell_w) * Fraction(2) ** (-2 * exponent)
    exact_v = meat / (gram * gram) * Fraction(len(i), len(i) - left - right)
    # The constructed within columns are A and beta*A + noise exactly.
    assert np.array_equal(y - (beta * (a / 4 - b / 2) + 8 * a - 16 * b), beta * A + noise)
    if representation == 'permuted':
        order = np.random.default_rng(20260912).permutation(len(i))
        y, x, w, a, b, noise = (v[order] for v in (y, x, w, a, b, noise))
    elif representation == 'recoded':
        # Bijections with gaps, always inside the declared eight-slot domain.
        a = np.linspace(0, 7, left, dtype=int)[::-1][a]
        b = np.roll(np.linspace(0, 7, right, dtype=int), 1)[b]
    return y, x, w, [a, b], noise, exact_v


def fingerprint(arrays):
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def run(args):
    module = args.module.resolve()
    module_sha = hashlib.sha256(module.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ('tmp', 'cuda_cache', 'xdg_cache'):
        (args.out / name).mkdir()
    os.environ.update(XHDFE_GPU_BACKEND=args.backend,
        XHDFE_ABSORPTION_CACHE_MODE='off', XHDFE_MOBILITY_MODE='off',
        XHDFE_FE_NORMALIZE='component', OMP_DYNAMIC='FALSE',
        TMPDIR=str(args.out / 'tmp'), CUDA_CACHE_PATH=str(args.out / 'cuda_cache'),
        XDG_CACHE_HOME=str(args.out / 'xdg_cache'))
    spec = importlib.util.spec_from_file_location('py_hdfe_v11', module)
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    native = ctypes.CDLL(str(module))
    counter = native.xhdfe_private_n05_work_count
    counter.argtypes, counter.restype = [ctypes.c_int], ctypes.c_ulonglong
    cases = list(itertools.product(((3, 5), (5, 3), (4, 4), (2, 7), (7, 2), (8, 8)),
        (.75, -1.25, 0.), (12, 20, 30), ('ordered', 'permuted', 'recoded'),
        (False, True), (False, True)))
    jobs = [(*case, 2) for case in cases]
    available = len(os.sched_getaffinity(0))
    for threads in (1, 8, 16, 24, 48):
        if threads <= available:
            for intercept, retain in itertools.product((False, True), repeat=2):
                jobs.append(((5, 3), -.25, 20, 'permuted', intercept, retain, threads))
    rows = []
    with (args.out / 'audit.stderr.log').open('xb+') as log:
        for grid, beta, exponent, representation, intercept, retain, threads in jobs:
            y, x, w, fes, noise, exact_v = fixture(*grid, beta, exponent, representation)
            arrays = [y, x, w, *fes]
            inputs_before = fingerprint(arrays)
            vref = float(exact_v)
            arithmetic, model_info = oracle_covariance_arithmetic_margin([[vref]],
                covariance_operation_count(len(y), 1, 'robust'), 1.)
            vlimit = COMPARABLE_V_REL * vref + float(arithmetic[0, 0])
            previous = None
            for audit in (False, True):
                os.environ['XHDFE_CERTIFY'] = '1' if audit else '0'
                counter(1)
                row = dict(grid=grid, beta=beta, exponent=exponent,
                    representation=representation, intercept=intercept,
                    retain=retain, threads=threads, audit=audit,
                    variance_reference=str(exact_v), variance_limit=vlimit,
                    arithmetic=model_info)
                model = cpp.HdfeRegressor(num_threads=threads, fit_intercept=intercept,
                    drop_singletons=False, max_iter=1000, tol=1e-8,
                    tolerance_mode='reghdfe-comparable', retain_fes=retain, se_type='robust')
                log.seek(0, 2)
                start = log.tell()
                stderr_fd = os.dup(2)
                try:
                    os.dup2(log.fileno(), 2)
                    model.fit(y, x[:, None], fes=fes, weights=w)
                except Exception as error:
                    row.update(verdict='FAIL_REFUSAL', error=str(error))
                finally:
                    os.dup2(stderr_fd, 2)
                    os.close(stderr_fd)
                end = log.seek(0, 2)
                log.seek(start)
                stderr = log.read(end - start).decode('utf-8', errors='replace')
                row.update(stderr_range=[start, end], audit_work=int(counter(0)))
                if 'verdict' not in row:
                    try:
                        receipt = parse_n05_receipt(stderr, required=audit,
                            expected_entrypoint='fit', expected_mode='reghdfe-comparable',
                            expected_backend=args.backend)
                        coef = np.array(model.coef_, copy=True)
                        covariance = np.array(model.covariance_, copy=True)
                        residuals = np.array(model.residuals_, copy=True)
                        effects = [np.array(v, copy=True) for v in model.fe_effects_] if retain else []
                        outputs = [coef, covariance, residuals, *effects]
                        row.update(beta_error=abs(float(coef[0]) - beta),
                            variance_relative_error=abs(float(covariance[0, 0]) / vref - 1),
                            residual_error=float(np.max(np.abs(residuals - noise))),
                            df_resid=float(model.df_resid_), iterations=int(model.num_iterations_),
                            gpu_used=bool(model.gpu_used_), threads_used=int(model.threads_used_),
                            input_unchanged=fingerprint(arrays) == inputs_before,
                            output_sha256=fingerprint(outputs), receipt=receipt)
                        if retain:
                            constant = float(coef[-1]) if intercept else 0.
                            row['reconstruction_error'] = float(np.max(np.abs(
                                y - coef[0] * x - constant - sum(effects) - noise)))
                        good = (row['beta_error'] <= COMPARABLE_BETA_REL * max(1, abs(beta))
                            and abs(float(covariance[0, 0]) - vref) <= vlimit
                            and row['residual_error'] <= RESIDUAL_ABS
                            and row['df_resid'] == len(y) - sum(grid)
                            and model.converged_ and model.precision_certified_
                            and row['gpu_used'] == (args.backend == 'cuda')
                            and row['threads_used'] == threads and row['input_unchanged']
                            and (not retain or (len(effects) == 2
                                and row['reconstruction_error'] <= FE_RECOVERY_ABS))
                            and ((audit and row['audit_work'] > 0) or (not audit and row['audit_work'] == 0)))
                        if audit:
                            row['pair_exact'] = all(np.array_equal(a, b) for a, b in zip(outputs, previous))
                            row['pair_within_arithmetic'] = all(np.allclose(a, b,
                                rtol=64 * np.finfo(float).eps, atol=64 * np.finfo(float).eps)
                                for a, b in zip(outputs, previous))
                            good = good and row['pair_within_arithmetic'] and receipt['identity_match']
                            good = good and all(v['status'] != 'ERROR_DEMONSTRATED'
                                                for v in receipt['families'].values())
                        previous = outputs
                        row['verdict'] = 'PASS' if good else 'FAIL'
                    except Exception as error:
                        row.update(verdict='FAIL_HARNESS', error=str(error))
                rows.append(row)
    assert hashlib.sha256(module.read_bytes()).hexdigest() == module_sha
    counts = dict(Counter(row['verdict'] for row in rows))
    report = dict(module=str(module), module_sha256=module_sha, backend=args.backend,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle='Analytic Walsh moments and exact Fraction HC1; no candidate-derived allowance',
        counts=counts, fits=len(rows), pairs=len(jobs), rows=rows)
    with (args.out / 'report.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(counts))
    return 0 if counts.get('PASS', 0) == len(rows) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
