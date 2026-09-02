#!/usr/bin/env python3
"""Adversarial runtime gates for the 04aug2026 audit remediation."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import warnings
from pathlib import Path

import numpy as np


def load_module(module_dir: Path):
    sys.path.insert(0, str(module_dir.resolve()))
    return importlib.import_module("py_hdfe_v11")


def expect_error(label: str, call, *fragments: str) -> None:
    try:
        call()
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
        if not all(fragment in message for fragment in fragments):
            raise AssertionError(f"{label}: unexpected message: {message}") from exc
        return
    raise AssertionError(f"{label}: invalid input was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", required=True, type=Path)
    parser.add_argument("--grouped-cuda", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    mod = load_module(args.module_dir)

    rng = np.random.default_rng(20260804)
    n = 240
    row = np.arange(n)
    fe1 = (row % 17).astype(np.int32)
    fe2 = (row % 13).astype(np.int64)
    X = rng.normal(size=(n, 2))
    y = X @ np.array([0.4, -0.7]) + 0.1 * fe1 - 0.05 * fe2
    y += rng.normal(size=n)

    lifecycle = mod.HdfeRegressor(num_threads=1, drop_singletons=False)
    assert lifecycle.lifecycle_state_ == "empty"
    assert lifecycle.generation_ == 0
    assert not lifecycle.converged_ and not lifecycle.precision_certified_
    assert np.asarray(lifecycle.coef_).size == 0
    assert np.asarray(lifecycle.residuals_).size == 0
    assert lifecycle.threads_used_ == 0 and not lifecycle.gpu_used_
    assert lifecycle.absorption_method_used == mod.AbsorptionMethod.auto
    assert "No estimation result available" in lifecycle.summary()

    lifecycle.fit(y, X, [fe1, fe2])
    assert lifecycle.lifecycle_state_ == "standard_ready"
    assert lifecycle.generation_ == 1
    assert lifecycle.converged_ and np.asarray(lifecycle.coef_).size > 0
    assert "Converged: yes" in lifecycle.summary()

    def assert_failed_lifecycle(expected_generation: int) -> None:
        assert lifecycle.lifecycle_state_ == "failed"
        assert lifecycle.generation_ == expected_generation
        assert not lifecycle.converged_ and not lifecycle.precision_certified_
        assert np.asarray(lifecycle.coef_).size == 0
        assert np.asarray(lifecycle.residuals_).size == 0
        assert np.asarray(lifecycle.covariance_).size == 0
        assert lifecycle.absorption_method_used == mod.AbsorptionMethod.auto
        assert lifecycle.threads_used_ == 0 and lifecycle.threads_effective_ == 0
        assert not lifecycle.gpu_used_ and lifecycle.gpu_status_code_ == 0
        assert "No estimation result available" in lifecycle.summary()

    expect_error(
        "lifecycle-fweights-preflight",
        lambda: lifecycle.fit(y, X, [fe1, fe2], fweights=True),
        "fweights=True", "weights vector",
    )
    assert_failed_lifecycle(2)
    lifecycle.fit(y, X, [fe1, fe2])
    assert lifecycle.lifecycle_state_ == "standard_ready"
    assert lifecycle.generation_ == 3 and lifecycle.nobs_ == n

    expect_error(
        "lifecycle-y-shape-preflight",
        lambda: lifecycle.fit(y[:, None], X, [fe1, fe2]),
        "y must be a 1-D array",
    )
    assert_failed_lifecycle(4)
    lifecycle.fit(y, X, [fe1, fe2])
    assert lifecycle.generation_ == 5 and lifecycle.converged_

    expect_error(
        "lifecycle-weight-shape-preflight",
        lambda: lifecycle.fit(y, X, [fe1, fe2], weights=np.ones(n - 1),
                              fweights=True),
        "weights must be length n",
    )
    assert_failed_lifecycle(6)
    lifecycle.fit(y, X, [fe1, fe2])
    assert lifecycle.generation_ == 7 and lifecycle.nobs_ == n

    def fit(y_value=y, X_value=X, fes=(fe1, fe2), clusters=None,
            instruments=None, endogenous_idx=(), slopes=None):
        reg = mod.HdfeRegressor(num_threads=1, drop_singletons=False)
        reg.fit(y_value, X_value, list(fes), clusters=clusters,
                instruments=instruments, endogenous_idx=list(endogenous_idx),
                slopes=slopes)
        return reg

    y_nan = y.copy()
    y_nan[7] = np.nan
    expect_error("fit-y-nan", lambda: fit(y_value=y_nan), "y", "row 7")
    y_inf = y.copy()
    y_inf[11] = np.inf
    expect_error("fit-y-inf", lambda: fit(y_value=y_inf), "y", "row 11")
    X_nan = X.copy()
    X_nan[9, 1] = np.nan
    expect_error("fit-X-nan", lambda: fit(X_value=X_nan), "X", "row 9", "column 1")
    X_inf = X.copy()
    X_inf[12, 0] = -np.inf
    expect_error("fit-X-inf", lambda: fit(X_value=X_inf), "X", "row 12", "column 0")

    # Finite inputs can still overflow downstream moments. A completed fit
    # must reject that state and remain inspectably non-converged after the
    # binding propagates the exception.
    extreme_scale = np.linspace(1.0, 2.0, 20)
    extreme = mod.HdfeRegressor(
        fit_intercept=False, num_threads=1, drop_singletons=False,
    )
    expect_error(
        "fit-output-postcondition",
        lambda: extreme.fit(
            extreme_scale * 1e100,
            (extreme_scale * 1e-100)[:, None],
            [],
        ),
        "converged fit produced invalid", "standard-error", "rejected",
    )
    assert not extreme.converged_ and not extreme.precision_certified_

    gram_overflow = mod.HdfeRegressor(
        fit_intercept=False, num_threads=1, drop_singletons=False,
    )
    expect_error(
        "fit-gram-overflow",
        lambda: gram_overflow.fit(
            extreme_scale,
            (extreme_scale * 1e154)[:, None],
            [],
        ),
        "cross-product", "Gram", "rescale", "1e153",
    )
    assert not gram_overflow.converged_ and not gram_overflow.precision_certified_

    Z_bad = rng.normal(size=(n, 2))
    Z_bad[5, 1] = np.nan
    expect_error(
        "fit-instrument-nan",
        lambda: fit(instruments=Z_bad, endogenous_idx=(0,)),
        "instruments", "row 5", "column 1",
    )
    slope_bad = rng.normal(size=n)
    slope_bad[8] = np.inf
    expect_error(
        "fit-slope-inf",
        lambda: fit(slopes=[(0, slope_bad, True)]),
        "heterogeneous slope values", "row 8",
    )

    for dtype in (np.int32, np.int64):
        bad_fe = fe1.astype(dtype)
        bad_fe[3] = -1
        expect_error(
            f"negative-fe-{dtype.__name__}",
            lambda bad_fe=bad_fe: fit(fes=(bad_fe, fe2)),
            "nonnegative IDs", "factorize()", "-1",
        )
        bad_cluster = fe1.astype(dtype)
        bad_cluster[4] = -2
        expect_error(
            f"negative-cluster-{dtype.__name__}",
            lambda bad_cluster=bad_cluster: fit(clusters=[bad_cluster]),
            "nonnegative IDs", "factorize()", "-1",
        )

    # Exercise the int64 compact-remap branch: a negative code after an ID
    # above int32 must not be remapped into a valid positive category.
    remap_bad = fe1.astype(np.int64)
    remap_bad[0] = 2**40
    remap_bad[-1] = -1
    expect_error(
        "negative-after-int64-remap",
        lambda: fit(fes=(remap_bad, fe2)),
        "nonnegative IDs", "factorize()", "-1",
    )

    group = (row // 3).astype(np.int64)
    individual = (row % 71).astype(np.int64)
    group_y = rng.normal(size=int(group.max()) + 1)[group]
    group_X = rng.normal(size=(int(group.max()) + 1, 1))[group]
    bad_group = group.copy()
    bad_group[0] = -1
    expect_error(
        "negative-group-id",
        lambda: mod.HdfeRegressor(num_threads=1).fit(
            group_y, group_X, [individual], group=bad_group,
            individual=individual,
        ),
        "nonnegative IDs", "factorize()", "-1",
    )

    workers = np.repeat(np.arange(20, dtype=np.int64), 4)
    period = np.tile(np.arange(4, dtype=np.int64), 20)
    firms = (workers + period) % 11
    akm_y = rng.normal(size=workers.size)
    akm_y_bad = akm_y.copy()
    akm_y_bad[2] = np.nan
    expect_error(
        "akm-y-nan",
        lambda: mod.akm_kss(akm_y_bad, workers, firms, prune=False,
                            leverages="exact"),
        "y", "row 2",
    )
    akm_X_bad = rng.normal(size=(workers.size, 1))
    akm_X_bad[3, 0] = np.inf
    expect_error(
        "akm-X-inf",
        lambda: mod.akm_kss(akm_y, workers, firms, X=akm_X_bad,
                            prune=False, leverages="exact"),
        "X", "row 3", "column 0",
    )
    akm_Z_bad = rng.normal(size=(workers.size, 1))
    akm_Z_bad[4, 0] = np.nan
    expect_error(
        "akm-Z-nan",
        lambda: mod.akm_kss(akm_y, workers, firms, Z=akm_Z_bad,
                            prune=False, leverages="exact"),
        "Z", "row 4", "column 0",
    )
    workers_bad = workers.copy()
    workers_bad[1] = -1
    expect_error(
        "akm-negative-worker",
        lambda: mod.akm_leave_out_set(workers_bad, firms),
        "nonnegative IDs", "factorize()", "-1",
    )

    gn = 300
    gfe1 = (np.arange(gn) % 23).astype(np.int64)
    gfe2 = (np.arange(gn) % 11).astype(np.int64)
    X1 = rng.normal(size=(gn, 1))
    X2 = rng.normal(size=(gn, 2))
    gy = 0.5 * X1[:, 0] + X2 @ np.array([0.2, -0.3]) + rng.normal(size=gn)
    gy_bad = gy.copy()
    gy_bad[6] = np.nan
    expect_error(
        "gelbach-y-nan",
        lambda: mod.gelbach_decompose(gy_bad, X1, X2, [2], [gfe1, gfe2]),
        "gelbach: y", "row 6",
    )
    X1_bad = X1.copy()
    X1_bad[7, 0] = np.inf
    expect_error(
        "gelbach-X1-inf",
        lambda: mod.gelbach_decompose(gy, X1_bad, X2, [2], [gfe1, gfe2]),
        "gelbach: X1", "row 7", "column 0",
    )
    gfe_bad = gfe1.copy()
    gfe_bad[8] = -1
    expect_error(
        "gelbach-negative-fe",
        lambda: mod.gelbach_decompose(gy, X1, X2, [2], [gfe_bad, gfe2]),
        "nonnegative IDs", "factorize()", "-1",
    )
    cluster_bad = gfe1.copy()
    cluster_bad[9] = -1
    expect_error(
        "gelbach-negative-cluster",
        lambda: mod.gelbach_decompose(
            gy, X1, X2, [2], [gfe1, gfe2], cluster=cluster_bad,
            vce="cluster",
        ),
        "nonnegative IDs", "factorize()", "-1",
    )

    # regress and reghdfe keep the point estimates with one cluster but expose
    # inference as missing. xhdfe must do the same, not report tiny finite SEs.
    one_cluster = np.zeros(n, dtype=np.int32)
    g1 = mod.HdfeRegressor(
        se_type="cluster", num_threads=1, drop_singletons=False,
    )
    g1.fit(y, X, [fe1, fe2], clusters=[one_cluster])
    assert g1.num_clusters_ == 1
    assert np.isnan(np.asarray(g1.stderr_)).all()
    assert np.isnan(np.asarray(g1.tvalues_)).all()
    assert np.isnan(np.asarray(g1.pvalues_)).all()
    assert np.isnan(np.asarray(g1.conf_int_)).all()

    constant = mod.HdfeRegressor(num_threads=1, drop_singletons=False)
    constant.fit(np.ones(n), X, [fe1, fe2])
    assert constant.tss_ == 0.0 and constant.tss_within_ == 0.0
    assert np.isnan(constant.r2_) and np.isnan(constant.r2_within_)

    # A bounded four-FE Auto fit must stay certified when CUDA is requested.
    # On a CUDA build/device it must use the GPU; otherwise the documented
    # status-2 CPU fallback must remain scientifically valid.
    four_fes = [
        (row % divisor).astype(np.int32)
        for divisor in (17, 13, 11, 7)
    ]
    requested_backend = os.environ.get("XHDFE_GPU_BACKEND")
    os.environ["XHDFE_GPU_BACKEND"] = "cuda"
    try:
        four_auto = mod.HdfeRegressor(
            num_threads=1, drop_singletons=False, tol=1e-8,
            tolerance_mode="reghdfe-comparable",
        )
        four_auto.fit(y, X, four_fes)
    finally:
        if requested_backend is None:
            os.environ.pop("XHDFE_GPU_BACKEND", None)
        else:
            os.environ["XHDFE_GPU_BACKEND"] = requested_backend
    assert four_auto.converged_ and four_auto.precision_certified_
    if four_auto.gpu_used_:
        assert four_auto.gpu_status_code_ == 1
    else:
        assert four_auto.gpu_status_code_ == 2

    # An explicit but token-free DoF sequence has the same meaning as an
    # absent option. In particular, it must retain the default mobility-group
    # redundancy on a disconnected two-FE graph.
    disconnected_fe1 = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    disconnected_fe2 = np.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int32)
    disconnected_X = (
        np.arange(8, dtype=np.float64)
        + np.array([0.0, 0.3, -0.2, 0.4, 0.1, -0.1, 0.2, -0.3])
    )[:, None]
    disconnected_y = 1.5 * disconnected_X[:, 0] + np.array(
        [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4]
    )

    def disconnected_fit(dofadjustments):
        reg = mod.HdfeRegressor(
            num_threads=1, drop_singletons=False,
            dofadjustments=dofadjustments,
        )
        reg.fit(
            disconnected_y, disconnected_X,
            [disconnected_fe1, disconnected_fe2],
        )
        return reg

    default_dof = disconnected_fit(None)
    for token_free in ([], "", "   ", " , "):
        explicit_empty = disconnected_fit(token_free)
        assert explicit_empty.df_a_ == default_dof.df_a_
        assert explicit_empty.fe_base_redundant_ == default_dof.fe_base_redundant_

    # Main-fit frequency weights equal literal row expansion within FP64
    # summation noise for coefficients and all three supported VCE modes.
    fw = (1 + row % 3).astype(np.float64)
    expanded = np.repeat(row, fw.astype(np.int64))
    for se_type, tolerance in (("unadjusted", 1e-13),
                               ("robust", 1e-11),
                               ("cluster", 1e-11)):
        weighted = mod.HdfeRegressor(
            se_type=se_type, num_threads=1, drop_singletons=False, tol=1e-10,
        )
        expanded_fit = mod.HdfeRegressor(
            se_type=se_type, num_threads=1, drop_singletons=False, tol=1e-10,
        )
        cluster_arg = [fe1] if se_type == "cluster" else None
        expanded_cluster = [fe1[expanded]] if se_type == "cluster" else None
        weighted.fit(y, X, [fe1, fe2], weights=fw, clusters=cluster_arg,
                     fweights=True)
        expanded_fit.fit(y[expanded], X[expanded],
                         [fe1[expanded], fe2[expanded]],
                         clusters=expanded_cluster)
        np.testing.assert_allclose(weighted.coef_, expanded_fit.coef_,
                                   rtol=0.0, atol=tolerance)
        np.testing.assert_allclose(weighted.stderr_, expanded_fit.stderr_,
                                   rtol=0.0, atol=tolerance)
        assert weighted.df_resid_ == expanded_fit.df_resid_

    # Frequency weights must also equal literal row expansion when an
    # indefinite two-way-cluster covariance activates the CGM PSD repair.
    # The repair uses raw-regressor sample standard deviations as its metric,
    # so those moments must reflect the frequency-expanded sample.
    psd_rng = np.random.default_rng(20260817)
    psd_n = 720
    psd_row = np.arange(psd_n)
    psd_firm = np.repeat(np.arange(60, dtype=np.int64), 12)
    psd_year = np.tile(np.arange(12, dtype=np.int64), 60)
    psd_region = psd_firm % 9
    psd_cohort = psd_year % 4
    psd_x = psd_rng.normal(size=psd_n)
    psd_z1 = psd_rng.normal(size=psd_n)
    psd_z2 = psd_rng.normal(size=psd_n)
    psd_v = psd_rng.normal(size=psd_n)
    psd_d = (
        0.8 * psd_z1 - 0.45 * psd_z2 + 0.25 * psd_x + 0.7 * psd_v
        + psd_rng.normal(scale=0.6, size=psd_n)
    )
    psd_y = (
        1.4 * psd_d + 0.5 * psd_x + 0.8 * psd_v
        + psd_rng.normal(scale=0.7, size=psd_n)
    )
    psd_X = np.asfortranarray(np.column_stack((psd_x, psd_d)))
    psd_Z = np.asfortranarray(np.column_stack((psd_z1, psd_z2)))
    psd_fw = (1 + psd_row % 3).astype(np.float64)
    psd_expanded = np.repeat(psd_row, psd_fw.astype(np.int64))

    def psd_fweight_fit(expand_rows, use_iv):
        selected = psd_expanded if expand_rows else psd_row
        reg = mod.HdfeRegressor(
            se_type="cluster", num_threads=1, drop_singletons=False,
            tol=1e-12, max_iter=20_000,
        )
        fit_options = {}
        if use_iv:
            fit_options.update(
                instruments=psd_Z[selected], endogenous_idx=[1],
            )
        if not expand_rows:
            fit_options.update(weights=psd_fw, fweights=True)
        reg.fit(
            psd_y[selected], psd_X[selected],
            [psd_firm[selected], psd_year[selected]],
            clusters=[psd_region[selected], psd_cohort[selected]],
            **fit_options,
        )
        return reg

    for use_iv in (False, True):
        weighted = psd_fweight_fit(False, use_iv)
        expanded_fit = psd_fweight_fit(True, use_iv)
        np.testing.assert_allclose(
            weighted.coef_, expanded_fit.coef_, rtol=0.0, atol=2e-11,
        )
        np.testing.assert_allclose(
            weighted.stderr_, expanded_fit.stderr_, rtol=0.0, atol=2e-10,
        )
        np.testing.assert_allclose(
            weighted.covariance_, expanded_fit.covariance_,
            rtol=0.0, atol=2e-10,
        )
        assert weighted.df_resid_ == expanded_fit.df_resid_
        assert weighted.nobs_effective_ == expanded_fit.nobs_effective_

    expect_error(
        "fweights-without-vector",
        lambda: mod.HdfeRegressor().fit(y, X, [fe1, fe2], fweights=True),
        "requires a weights vector",
    )
    for label, invalid in (
        ("fractional", fw + 0.5),
        ("negative", np.where(row == 0, -1.0, fw)),
        ("nan", np.where(row == 0, np.nan, fw)),
    ):
        expect_error(
            f"fweights-{label}",
            lambda invalid=invalid: mod.HdfeRegressor().fit(
                y, X, [fe1, fe2], weights=invalid, fweights=True,
            ),
            "Frequency weights must be positive integers",
        )

    # The automatic method must repair a connected-chain solve until the
    # independent residual certificate passes. A deliberately forced sweep
    # remains observable through a typed, filterable warning if it stops on
    # the weaker update criterion.
    groups = 1000
    chain_fe1 = np.repeat(np.arange(groups, dtype=np.int32), 3)
    chain_fe2 = np.empty(3 * groups, dtype=np.int32)
    for i in range(groups):
        chain_fe2[3 * i:3 * i + 3] = (i, i, i + 1)
    chain_fe1 = np.append(chain_fe1, np.int32(groups - 1))
    chain_fe2 = np.append(chain_fe2, np.int32(groups))
    chain_X = rng.normal(size=(chain_fe1.size, 2))
    chain_y = chain_X @ np.array([0.7, -0.2]) + rng.normal(size=chain_fe1.size)
    chain_auto = mod.HdfeRegressor(
        num_threads=1, drop_singletons=False, tol=1e-8,
        tolerance_mode="reghdfe-comparable",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chain_auto.fit(chain_y, chain_X, [chain_fe1, chain_fe2])
    assert chain_auto.converged_ and chain_auto.precision_certified_
    assert not caught

    chain_forced = mod.HdfeRegressor(
        num_threads=1, drop_singletons=False, tol=1e-8,
        tolerance_mode="reghdfe-comparable",
        absorption_method="gauss-seidel",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chain_forced.fit(chain_y, chain_X, [chain_fe1, chain_fe2])
    assert chain_forced.converged_ and not chain_forced.precision_certified_
    assert len(caught) == 1
    assert caught[0].category is mod.PrecisionWarning
    assert issubclass(mod.PrecisionWarning, RuntimeWarning)
    with warnings.catch_warnings(record=True) as suppressed:
        warnings.simplefilter("ignore", mod.PrecisionWarning)
        chain_forced.fit(chain_y, chain_X, [chain_fe1, chain_fe2])
    assert not suppressed

    # Group/individual fits are stricter than the standard best-effort surface:
    # an exhausted CPU LSMR solve must raise before coefficients are exposed.
    group_count = 128
    grouped_id = np.repeat(np.arange(group_count, dtype=np.int32), 2)
    grouped_individual = np.column_stack((
        np.arange(group_count, dtype=np.int32),
        np.arange(1, group_count + 1, dtype=np.int32),
    )).reshape(-1)
    grouped_standard = np.repeat(
        np.arange(group_count, dtype=np.int32) % 7, 2,
    )
    grouped_x = np.repeat(
        np.sin(0.17 * np.arange(group_count, dtype=np.float64)), 2,
    )[:, None]
    grouped_y = (
        0.6 * grouped_x[:, 0]
        + np.repeat(np.cos(0.11 * np.arange(group_count)), 2)
    )
    grouped_fail = mod.HdfeRegressor(
        num_threads=1, drop_singletons=False, max_iter=1, tol=1e-14,
        tolerance_mode="reghdfe-comparable", absorption_method="lsmr",
    )
    expect_error(
        "grouped-cpu-fail-closed",
        lambda: grouped_fail.fit(
            grouped_y, grouped_x,
            [grouped_individual, grouped_standard],
            group=grouped_id, individual=grouped_individual,
            aggregation="mean",
        ),
        "Group/individual HDFE absorption", "no estimates were produced",
    )
    assert not grouped_fail.converged_ and not grouped_fail.precision_certified_

    # Extraction authority belongs only to the last successful, certified
    # group/individual fit. A standard refit or any failed grouped refit must
    # revoke it and clear stale coefficients/method state.
    previous_backend = os.environ.get("XHDFE_GPU_BACKEND")
    os.environ["XHDFE_GPU_BACKEND"] = "cpu"
    try:
        grouped_state = mod.HdfeRegressor(
            num_threads=1, drop_singletons=False, max_iter=2_000, tol=1e-10,
            tolerance_mode="reghdfe-comparable",
        )
        grouped_state.fit(
            grouped_y, grouped_x,
            [grouped_individual, grouped_standard],
        )
        expect_error(
            "grouped-extract-after-standard",
            lambda: grouped_state.extract_group_individual_fes(
                grouped_y, grouped_x,
                [grouped_individual, grouped_standard],
                grouped_id, grouped_individual,
                aggregation="mean",
            ),
            "successful", "precision-certified", "fit()",
        )

        grouped_state.fit(
            grouped_y, grouped_x,
            [grouped_individual, grouped_standard],
            group=grouped_id, individual=grouped_individual,
            aggregation="mean",
        )
        assert grouped_state.converged_ and grouped_state.precision_certified_
        extracted = grouped_state.extract_group_individual_fes(
            grouped_y, grouped_x,
            [grouped_individual, grouped_standard],
            grouped_id, grouped_individual,
            aggregation="mean",
        )
        assert extracted["converged"]
        assert grouped_state.lifecycle_state_ == "grouped_ready"

        grouped_no_x = np.empty((grouped_y.size, 0), dtype=np.float64)
        grouped_fe_only = mod.HdfeRegressor(
            num_threads=1, drop_singletons=False, fit_intercept=False,
            max_iter=2_000, tol=1e-10,
            tolerance_mode="reghdfe-comparable",
        )
        grouped_fe_only.fit(
            grouped_y, grouped_no_x,
            [grouped_individual, grouped_standard],
            group=grouped_id, individual=grouped_individual,
            aggregation="mean",
        )
        assert grouped_fe_only.lifecycle_state_ == "grouped_ready"
        assert grouped_fe_only.converged_ and grouped_fe_only.precision_certified_
        assert np.asarray(grouped_fe_only.coef_).size == 0
        fe_only_extracted = grouped_fe_only.extract_group_individual_fes(
            grouped_y, grouped_no_x,
            [grouped_individual, grouped_standard], grouped_id,
            grouped_individual, aggregation="mean",
        )
        assert fe_only_extracted["converged"]
        assert np.isfinite(fe_only_extracted["individual_effects"]).all()

        changed_fe_only_y = grouped_y.copy()
        changed_fe_only_y[0] += 1e-6
        expect_error(
            "grouped-fe-only-signature",
            lambda: grouped_fe_only.extract_group_individual_fes(
                changed_fe_only_y, grouped_no_x,
                [grouped_individual, grouped_standard], grouped_id,
                grouped_individual, aggregation="mean",
            ),
            "inputs", "semantics", "exactly match",
        )
        grouped_fe_only.fit(
            grouped_y, grouped_no_x,
            [grouped_individual, grouped_standard],
        )
        expect_error(
            "grouped-fe-only-stale-lifecycle",
            lambda: grouped_fe_only.extract_group_individual_fes(
                grouped_y, grouped_no_x,
                [grouped_individual, grouped_standard], grouped_id,
                grouped_individual, aggregation="mean",
            ),
            "successful", "precision-certified", "fit()",
        )

        def expect_signature_error(label, y_value=grouped_y,
                                   X_value=grouped_x, fes_value=None,
                                   group_value=grouped_id,
                                   individual_value=grouped_individual,
                                   aggregation_value="mean", weights_value=None):
            if fes_value is None:
                fes_value = [grouped_individual, grouped_standard]
            expect_error(
                label,
                lambda: grouped_state.extract_group_individual_fes(
                    y_value, X_value, fes_value, group_value,
                    individual_value, weights=weights_value,
                    aggregation=aggregation_value,
                ),
                "inputs", "semantics", "exactly match",
            )

        changed_y = grouped_y.copy()
        changed_y[0] += 1e-6
        expect_signature_error("grouped-signature-y", y_value=changed_y)
        changed_x = grouped_x.copy()
        changed_x[0, 0] += 1e-6
        expect_signature_error("grouped-signature-X", X_value=changed_x)
        changed_standard = grouped_standard.copy()
        changed_standard[0] += 1
        expect_signature_error(
            "grouped-signature-fe",
            fes_value=[grouped_individual, changed_standard],
        )
        changed_group = grouped_id.copy()
        changed_group[0] += 1
        expect_signature_error("grouped-signature-group", group_value=changed_group)
        changed_individual = grouped_individual.copy()
        changed_individual[0] = 10000
        expect_signature_error(
            "grouped-signature-individual",
            fes_value=[changed_individual, grouped_standard],
            individual_value=changed_individual,
        )
        expect_signature_error(
            "grouped-signature-weight-presence",
            weights_value=np.ones(grouped_y.size),
        )
        expect_signature_error(
            "grouped-signature-aggregation", aggregation_value="sum",
        )

        grouped_fw = np.ones(grouped_y.size)
        grouped_weighted = mod.HdfeRegressor(
            num_threads=1, drop_singletons=False, max_iter=2_000, tol=1e-10,
            tolerance_mode="reghdfe-comparable",
        )
        grouped_weighted.fit(
            grouped_y, grouped_x,
            [grouped_individual, grouped_standard], weights=grouped_fw,
            fweights=True, group=grouped_id,
            individual=grouped_individual, aggregation="mean",
        )
        weighted_extract = grouped_weighted.extract_group_individual_fes(
            grouped_y, grouped_x,
            [grouped_individual, grouped_standard], grouped_id,
            grouped_individual, weights=grouped_fw, aggregation="mean",
        )
        assert weighted_extract["converged"]
        assert grouped_weighted.lifecycle_state_ == "grouped_ready"

        os.environ["XHDFE_GPU_BACKEND"] = "metal"
        expect_error(
            "grouped-extract-inner-failure",
            lambda: grouped_state.extract_group_individual_fes(
                grouped_y, grouped_x,
                [grouped_individual, grouped_standard],
                grouped_id, grouped_individual,
                aggregation="mean",
            ),
            "extraction absorption", "did not converge", "certificate",
        )
        os.environ["XHDFE_GPU_BACKEND"] = "cpu"

        grouped_state.fit(
            grouped_y, grouped_x,
            [grouped_individual, grouped_standard],
        )
        expect_error(
            "grouped-extract-after-standard-refit",
            lambda: grouped_state.extract_group_individual_fes(
                grouped_y, grouped_x,
                [grouped_individual, grouped_standard],
                grouped_id, grouped_individual,
                aggregation="mean",
            ),
            "successful", "precision-certified", "fit()",
        )
        grouped_state.fit(
            grouped_y, grouped_x,
            [grouped_individual, grouped_standard],
            group=grouped_id, individual=grouped_individual,
            aggregation="mean",
        )
        grouped_y_bad = grouped_y.copy()
        grouped_y_bad[0] = np.nan
        expect_error(
            "grouped-success-then-failure",
            lambda: grouped_state.fit(
                grouped_y_bad, grouped_x,
                [grouped_individual, grouped_standard],
                group=grouped_id, individual=grouped_individual,
                aggregation="mean",
            ),
            "y", "row 0",
        )
        assert not grouped_state.converged_
        assert not grouped_state.precision_certified_
        assert np.asarray(grouped_state.coef_).size == 0
        assert grouped_state.absorption_method_used == mod.AbsorptionMethod.auto
        expect_error(
            "grouped-extract-after-failed-refit",
            lambda: grouped_state.extract_group_individual_fes(
                grouped_y, grouped_x,
                [grouped_individual, grouped_standard],
                grouped_id, grouped_individual,
                aggregation="mean",
            ),
            "successful", "precision-certified", "fit()",
        )
    finally:
        if previous_backend is None:
            os.environ.pop("XHDFE_GPU_BACKEND", None)
        else:
            os.environ["XHDFE_GPU_BACKEND"] = previous_backend

    if args.grouped_cuda:
        cuda_group_count = 48
        cuda_members = 4
        cuda_group = np.repeat(
            np.arange(cuda_group_count, dtype=np.int32), cuda_members,
        )
        cuda_member = np.tile(
            np.arange(cuda_members, dtype=np.int32), cuda_group_count,
        )
        cuda_individual = (
            cuda_group + 11 * cuda_member
        ) % cuda_group_count
        cuda_standard = np.repeat(
            np.arange(cuda_group_count, dtype=np.int32) % 7, cuda_members,
        )
        cuda_x_group = np.sin(
            0.19 * np.arange(cuda_group_count, dtype=np.float64)
        )
        cuda_y_group = (
            0.65 * cuda_x_group
            + np.cos(0.13 * np.arange(cuda_group_count, dtype=np.float64))
        )
        cuda_x = np.repeat(cuda_x_group, cuda_members)[:, None]
        cuda_y = np.repeat(cuda_y_group, cuda_members)
        cuda_fes = [cuda_individual, cuda_standard]

        previous_backend = os.environ.get("XHDFE_GPU_BACKEND")
        os.environ["XHDFE_GPU_BACKEND"] = "cuda"
        try:
            standard_before = mod.HdfeRegressor(
                num_threads=1, drop_singletons=False, tol=1e-10,
            )
            standard_before.fit(y, X, [fe1, fe2])
            if args.require_gpu:
                assert standard_before.gpu_used_

            cuda_auto = mod.HdfeRegressor(
                num_threads=1, drop_singletons=False, max_iter=5_000,
                tol=1e-10, tolerance_mode="reghdfe-comparable",
            )
            cuda_auto.fit(
                cuda_y, cuda_x, cuda_fes,
                group=cuda_group, individual=cuda_individual,
                aggregation="mean",
            )
            assert cuda_auto.converged_ and cuda_auto.precision_certified_
            assert cuda_auto.gpu_used_ and cuda_auto.gpu_status_code_ == 1
            assert cuda_auto.absorption_method_used == mod.AbsorptionMethod.lsmr
            cuda_extracted = cuda_auto.extract_group_individual_fes(
                cuda_y, cuda_x, cuda_fes, cuda_group, cuda_individual,
                aggregation="mean",
            )
            assert cuda_extracted["converged"]
            assert np.isfinite(cuda_extracted["individual_effects"]).all()

            cuda_no_x = np.empty((cuda_y.size, 0), dtype=np.float64)
            cuda_fe_only = mod.HdfeRegressor(
                num_threads=1, drop_singletons=False,
                fit_intercept=False, max_iter=5_000,
                tol=1e-10, tolerance_mode="reghdfe-comparable",
            )
            cuda_fe_only.fit(
                cuda_y, cuda_no_x, cuda_fes,
                group=cuda_group, individual=cuda_individual,
                aggregation="mean",
            )
            assert cuda_fe_only.lifecycle_state_ == "grouped_ready"
            assert cuda_fe_only.converged_ and cuda_fe_only.precision_certified_
            assert np.asarray(cuda_fe_only.coef_).size == 0
            assert cuda_fe_only.gpu_used_ and cuda_fe_only.gpu_status_code_ == 1
            assert cuda_fe_only.absorption_method_used == mod.AbsorptionMethod.lsmr
            cuda_fe_only_extracted = cuda_fe_only.extract_group_individual_fes(
                cuda_y, cuda_no_x, cuda_fes, cuda_group, cuda_individual,
                aggregation="mean",
            )
            assert cuda_fe_only_extracted["converged"]
            assert np.isfinite(
                cuda_fe_only_extracted["individual_effects"],
            ).all()

            changed_cuda_fe_y = cuda_y.copy()
            changed_cuda_fe_y[0] += 1e-6
            expect_error(
                "grouped-cuda-fe-only-signature",
                lambda: cuda_fe_only.extract_group_individual_fes(
                    changed_cuda_fe_y, cuda_no_x, cuda_fes,
                    cuda_group, cuda_individual, aggregation="mean",
                ),
                "inputs", "semantics", "exactly match",
            )
            cuda_fe_only.fit(cuda_y, cuda_no_x, cuda_fes)
            expect_error(
                "grouped-cuda-fe-only-stale-lifecycle",
                lambda: cuda_fe_only.extract_group_individual_fes(
                    cuda_y, cuda_no_x, cuda_fes,
                    cuda_group, cuda_individual, aggregation="mean",
                ),
                "successful", "precision-certified", "fit()",
            )

            for method, expected in (
                ("gauss-seidel", mod.AbsorptionMethod.gauss_seidel),
                ("symmetric-gauss-seidel",
                 mod.AbsorptionMethod.symmetric_gauss_seidel),
            ):
                explicit = mod.HdfeRegressor(
                    num_threads=1, drop_singletons=False, max_iter=5_000,
                    tol=1e-8, tolerance_mode="reghdfe-comparable",
                    absorption_method=method,
                )
                explicit.fit(
                    cuda_y, cuda_x, cuda_fes,
                    group=cuda_group, individual=cuda_individual,
                    aggregation="mean",
                )
                assert explicit.converged_ and explicit.precision_certified_
                assert explicit.gpu_used_ and explicit.gpu_status_code_ == 1
                assert explicit.absorption_method_used == expected

            explicit_lsmr = mod.HdfeRegressor(
                num_threads=1, drop_singletons=False,
                absorption_method="lsmr",
            )
            expect_error(
                "grouped-explicit-lsmr-cuda",
                lambda: explicit_lsmr.fit(
                    cuda_y, cuda_x, cuda_fes,
                    group=cuda_group, individual=cuda_individual,
                    aggregation="mean",
                ),
                "absorptionmethod(lsmr)", "CPU-only", "group/individual",
            )
            assert not explicit_lsmr.converged_
            assert not explicit_lsmr.precision_certified_
            assert np.asarray(explicit_lsmr.coef_).size == 0

            standard_after = mod.HdfeRegressor(
                num_threads=1, drop_singletons=False, tol=1e-10,
            )
            standard_after.fit(y, X, [fe1, fe2])
            assert standard_after.gpu_used_ and standard_after.gpu_status_code_ == 1
            np.testing.assert_allclose(
                standard_after.coef_, standard_before.coef_,
                rtol=0.0, atol=1e-12,
            )
            np.testing.assert_allclose(
                standard_after.residuals_, standard_before.residuals_,
                rtol=0.0, atol=1e-12,
            )
        finally:
            if previous_backend is None:
                os.environ.pop("XHDFE_GPU_BACKEND", None)
            else:
                os.environ["XHDFE_GPU_BACKEND"] = previous_backend

    print("PASS: audit 20260804 remediation contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
