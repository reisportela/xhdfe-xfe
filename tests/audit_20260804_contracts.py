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

    print("PASS: audit 20260804 remediation contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
