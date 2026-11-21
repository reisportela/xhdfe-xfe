#!/usr/bin/env python3
"""Offline CPU covariance contract for two identified, nearly parallel firm slopes."""

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import svd


@dataclass(frozen=True)
class Panel:
    worker: np.ndarray
    firm: np.ndarray
    period: np.ndarray
    seniority: np.ndarray
    near_seniority: np.ndarray
    regressors: np.ndarray
    outcome: np.ndarray
    clusters: tuple[np.ndarray, np.ndarray]
    sample: np.ndarray


def make_panel():
    # Freeze the near_slope draw order and represented inputs of the independent audit.
    rng = np.random.default_rng(791823)
    workers, periods = 74, 18
    worker = np.repeat(np.arange(workers), periods)
    period = np.tile(np.arange(periods), workers)
    firm = 2 * ((worker // 2 + (period // 4) * 7 + rng.integers(0, 5, worker.size)) % 19)
    firm += worker % 2
    period = period % 9
    seniority = rng.integers(1, 12, worker.size).astype(float)
    seniority[firm % 5 == 2] = 4
    weak = rng.normal(size=worker.size)
    near_seniority = seniority + (2.0**-12) * weak
    regressors = rng.normal(size=(worker.size, 4))
    regressors[:, 0] += weak
    outcome = (regressors @ np.array([0.4, -0.1, 1.2, 0.25])
               + 0.02 * seniority * (firm % 7) - 0.03 * near_seniority * (firm % 3))
    outcome += 0.2 * (worker % 5) + 0.1 * period + rng.normal(size=worker.size)

    # A singleton worker and firm share an existing period. The permutation
    # makes the expected original-row mask different from a prefix.
    worker = np.r_[worker, workers]
    firm = np.r_[firm, 61]
    period = np.r_[period, 0]
    seniority = np.r_[seniority, 4]
    near_seniority = np.r_[near_seniority, 0]
    regressors = np.vstack([regressors, [1.0, 2.0, 3.0, 4.0]])
    outcome = np.r_[outcome, 6.0]
    perm = rng.permutation(worker.size)
    worker = worker[perm].astype(np.int32)
    firm = firm[perm].astype(np.int32)
    clusters = (5 * (workers + 1 - worker) + 401, 23 * (firm + 1) + 331)
    return Panel(
        worker, firm, period[perm].astype(np.int32), seniority[perm], near_seniority[perm],
        regressors[perm], outcome[perm], clusters, np.flatnonzero(worker != workers),
    )


def dummies(labels):
    _, codes = np.unique(labels, return_inverse=True)
    return np.eye(codes.max() + 1)[codes]


def adjusted_meat(scores, labels):
    _, codes = np.unique(labels, axis=0, return_inverse=True)
    count = codes.max() + 1
    sums = np.zeros((count, scores.shape[1]))
    np.add.at(sums, codes, scores)
    return (count / (count - 1)) * (sums.T @ sums)


def positive_part(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    if values[0] < 0:
        matrix = (vectors * np.maximum(values, 0)) @ vectors.T
    return matrix, values, vectors


def reference_covariance(panel):
    keep = panel.sample
    x, y = panel.regressors[keep], panel.outcome[keep]
    n, p = x.shape
    firm = dummies(panel.firm[keep])
    effects = np.column_stack([
        dummies(panel.worker[keep]), dummies(panel.period[keep]), firm,
        firm * panel.seniority[keep, None], firm * panel.near_seniority[keep, None],
    ])
    u, singular_values, _ = svd(effects, full_matrices=False, lapack_driver="gesdd")
    rank = int(np.count_nonzero(
        singular_values > np.finfo(float).eps * max(effects.shape) * singular_values[0]))
    if n != 1332 or p != 4 or rank != 186:
        raise AssertionError(f"fixture identity changed: N={n}, dimension={p}, FE rank={rank}")

    basis = u[:, :rank]
    original = np.column_stack([y, x])
    within = original - basis @ (basis.T @ original)
    xr, yr = within[:, 1:], within[:, 0]
    beta, _, x_rank, _ = np.linalg.lstsq(xr, yr, rcond=None)
    if x_rank != p:
        raise AssertionError("a structural coefficient is unidentified in the fixture")
    residual = yr - xr @ beta
    scores = np.column_stack([xr, np.ones(n)]) * residual[:, None]
    a, b = (cluster[keep, None] for cluster in panel.clusters)
    meat = adjusted_meat(scores, a) + adjusted_meat(scores, b)
    meat -= adjusted_meat(scores, np.column_stack([a, b]))

    bread = np.zeros((p + 1, p + 1))
    bread[:p, :p] = np.linalg.inv(xr.T @ xr)
    bread[p, p] = 1 / n
    transport = np.eye(p + 1)
    transport[p, :p] = -x.mean(axis=0)
    raw = transport @ bread @ meat @ bread @ transport.T
    scale = np.r_[np.maximum(x.std(axis=0, ddof=1), 1e-3), 1.0]
    standardised = raw * scale[:, None] * scale[None, :]

    # Apply the documented PSD policy in the standardised coordinates.
    full, full_eigenvalues, _ = positive_part(standardised)
    slope, eigenvalues, eigenvectors = positive_part(standardised[:p, :p])
    if full_eigenvalues[0] < 0 or eigenvalues[0] < 0:
        full[:p, :p] = slope
        positive = eigenvalues > 0
        active = eigenvectors[:, positive]
        coordinates = active.T @ full[:p, p]
        cross = active @ coordinates
        full[:p, p] = cross
        full[p, :p] = cross
        full[p, p] = max(full[p, p], np.sum(coordinates**2 / eigenvalues[positive]))
    covariance = full / scale[:, None] / scale[None, :]
    if not np.isfinite(covariance).all() or (np.diag(covariance) <= 0).any():
        raise AssertionError("the reference needs finite positive diagonal scales")
    return covariance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    module_path = args.module.resolve(strict=True)
    # Give the offline fit the same routing and cache settings in every process.
    for name in tuple(os.environ):
        if name.startswith("XHDFE_"):
            del os.environ[name]
    os.environ.update(
        XHDFE_GPU_BACKEND="cpu",
        XHDFE_EXPERIMENT_SLOPE_SCHUR="1",
        XHDFE_EXPERIMENT_NETWORK_SCORES="1",
    )
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    if Path(native.__file__).resolve() != module_path:
        raise AssertionError("loaded a different native module")

    panel = make_panel()
    expected = reference_covariance(panel)
    model = native.HdfeRegressor(
        se_type="cluster", num_threads=24, tol=1e-8, max_iter=100000,
        ssc_k_adj=False, ssc_g_adj=True, ssc_g_df="conventional",
        tolerance_mode="reghdfe-comparable", absorption_method="auto",
        reduced_pcg=True, reduced_pcg_cuda=False, retain_fes=False,
        drop_singletons=True,
    )
    model.fit(
        panel.outcome, np.asfortranarray(panel.regressors),
        fes=[panel.worker, panel.period, panel.firm, panel.firm],
        clusters=list(panel.clusters),
        slopes=[
            dict(fe_index=2, include_intercept=True, values=panel.seniority),
            dict(fe_index=3, include_intercept=False, values=panel.near_seniority),
        ],
    )
    if not model.converged_ or not model.precision_certified_ or model.gpu_used_:
        raise AssertionError("the CPU fit did not deliver a certified projection")
    sample_matches = int(model.nobs_) == panel.sample.size and np.array_equal(
        model.sample_index_, panel.sample)
    covariance = np.asarray(model.covariance_)
    if covariance.shape != expected.shape or not np.isfinite(covariance).all():
        raise AssertionError("the fit did not deliver a finite 5-by-5 covariance")

    reference_se = np.sqrt(np.diag(expected))
    error = np.abs(covariance - expected) / np.outer(reference_se, reference_se)
    structural_error = float(error[:4, :4].max())
    full_error = float(error.max())
    limit = 1e-8
    passed = sample_matches and structural_error <= limit and full_error <= limit
    result = dict(
        status="PASS" if passed else "FAIL",
        module=str(module_path), sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        nobs=int(model.nobs_), sample_matches=bool(sample_matches),
        structural_covariance_error=structural_error, full_covariance_error=full_error,
        covariance_limit=limit, iterations=int(model.num_iterations_),
        method=int(model.absorption_method_used),
    )
    if args.output:
        with args.output.open("x", encoding="utf-8") as destination:
            json.dump(result, destination, indent=2, allow_nan=False)
            destination.write("\n")
    print(json.dumps(result, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
