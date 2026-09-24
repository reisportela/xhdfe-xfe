"""group()/individual() with aggregation(sum): the constant is reported only
when it lies in the span of the absorbed effects.

Sum-aggregated membership columns span the ones vector for a uniform team
size, or through a data-dependent combination of individual columns; an
ordinary absorbed FE or mean aggregation always spans it. When it is not
spanned, the fit has no intercept and must be reported as such: the trailing
intercept slot is NaN, ``model_has_constant_`` is False, the total sum of
squares is uncentered, and the coefficients equal the explicit group-level OLS
without a constant (2.26.2 coefficients are unchanged; only the reporting of
an unidentified intercept was corrected in 2.27.0).
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    # pybind11 registers its types once per process: reuse the core another
    # test already imported as top-level ``py_hdfe_v11`` (the suite runs in
    # one process), otherwise load the default CPU build and register it.
    loaded = sys.modules.get("py_hdfe_v11")
    if loaded is not None:
        if not hasattr(loaded.HdfeRegressor, "model_has_constant_"):
            pytest.skip("the py_hdfe_v11 module already loaded predates model_has_constant_")
        return loaded
    candidates = [os.environ.get("XHDFE_PY_MODULE") or ""]
    candidates += sorted(glob.glob(str(ROOT / "build" / "py_hdfe_v11*.so")))
    for path in candidates:
        if path and Path(path).is_file():
            spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules["py_hdfe_v11"] = module
            return module
    pytest.skip("no built py_hdfe_v11 module (build/ or XHDFE_PY_MODULE)")


@pytest.fixture(scope="module")
def cpp():
    os.environ.setdefault("XHDFE_GPU_BACKEND", "cpu")
    return _load_module()


def _design(rng, n_ind, sizes, extra_individuals=None):
    """Long-format membership rows: one row per (group, individual)."""
    rows_g, rows_i = [], []
    for g, size in enumerate(sizes):
        members = rng.choice(n_ind, size=size, replace=False)
        if extra_individuals is not None:
            members = extra_individuals(g)
        for i in members:
            rows_g.append(g)
            rows_i.append(int(i))
    return np.asarray(rows_g, np.int64), np.asarray(rows_i, np.int64)


def _make(rng, n_ind, grp, ind, intercept=1.5):
    n_grp = int(grp.max()) + 1
    incidence = np.zeros((n_grp, n_ind))
    incidence[grp, ind] = 1.0
    alpha = rng.normal(size=n_ind)
    fe = incidence @ alpha
    x = rng.normal(size=(n_grp, 2)) + 0.3 * fe[:, None]
    y = intercept + x @ np.array([0.7, -0.4]) + fe + 0.1 * rng.normal(size=n_grp)
    return incidence, x, y


def _group_ols(incidence, x, y, with_constant):
    cols = [x, incidence] + ([np.ones(len(y))] if with_constant else [])
    Z = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return b[:2], y - Z @ b


def _fit(cpp, y, X, grp, ind, aggregation, fes=None, fit_intercept=True):
    model = cpp.HdfeRegressor(num_threads=2, se_type="unadjusted", fit_intercept=fit_intercept,
                              tolerance_mode="reghdfe-comparable")
    model.fit(y, X, fes=[ind] + (fes or []), group=grp, individual=ind, aggregation=aggregation)
    return model


def _ones_in_span(incidence):
    residual = np.linalg.lstsq(incidence, np.ones(incidence.shape[0]), rcond=None)[1]
    return bool(residual.size == 0 or residual[0] < 1e-18)


def test_sum_uneven_teams_has_no_constant(cpp):
    rng = np.random.default_rng(7)
    n_ind = 60
    sizes = rng.integers(1, 5, size=400)
    grp, ind = _design(rng, n_ind, sizes)
    incidence, x, y = _make(rng, n_ind, grp, ind)
    assert not _ones_in_span(incidence)
    model = _fit(cpp, y[grp], x[grp], grp, ind, "sum")
    b_ref, r_ref = _group_ols(incidence, x, y, with_constant=False)
    coef = np.asarray(model.coef_)
    assert model.model_has_constant_ is False
    assert coef.shape == (3,) and np.isnan(coef[2])
    assert np.allclose(coef[:2], b_ref, rtol=0, atol=1e-9)
    assert abs(float(model.tss_) - float(y @ y)) <= 1e-9 * float(y @ y)   # uncentered
    assert abs(float(model.rss_) - float(r_ref @ r_ref)) <= 1e-8 * float(r_ref @ r_ref)
    assert abs(float(model.r2_) - (1.0 - float(r_ref @ r_ref) / float(y @ y))) < 1e-9
    # Without a constant in the span the residuals do not sum to zero.
    assert abs(float(np.mean(model.residuals_))) > 1e-3
    # noconstant reports the same fit (the grouped design never carries an intercept column).
    model_nc = _fit(cpp, y[grp], x[grp], grp, ind, "sum", fit_intercept=False)
    assert model_nc.model_has_constant_ is False
    assert np.allclose(np.asarray(model_nc.coef_), b_ref, rtol=0, atol=1e-9)
    assert abs(float(model_nc.tss_) - float(model.tss_)) <= 1e-12 * float(model.tss_)


def test_sum_uniform_teams_spans_constant(cpp):
    rng = np.random.default_rng(11)
    n_ind = 60
    grp, ind = _design(rng, n_ind, np.full(300, 3))
    incidence, x, y = _make(rng, n_ind, grp, ind)
    assert _ones_in_span(incidence)
    model = _fit(cpp, y[grp], x[grp], grp, ind, "sum")
    b_ref, r_ref = _group_ols(incidence, x, y, with_constant=False)
    coef = np.asarray(model.coef_)
    assert model.model_has_constant_ is True
    assert np.isfinite(coef[2])
    assert np.allclose(coef[:2], b_ref, rtol=0, atol=1e-9)
    centered = float(np.sum((y - y.mean()) ** 2))
    assert abs(float(model.tss_) - centered) <= 1e-9 * centered
    assert abs(float(np.mean(model.residuals_))) < 1e-9


def test_sum_pattern_combination_spans_constant(cpp):
    """Uneven team sizes whose columns still combine to a constant: every
    group has exactly one member from a pool P2 and 0-3 members from a pool
    P1, so alpha = c on P2 and 0 on P1 reproduces the ones vector."""
    rng = np.random.default_rng(5)
    pool1, pool2 = np.arange(0, 30), np.arange(30, 60)
    sizes = rng.integers(0, 4, size=400)

    def members(g):
        return np.concatenate(([rng.choice(pool2)], rng.choice(pool1, size=sizes[g], replace=False)))

    grp, ind = _design(rng, 60, sizes + 1, extra_individuals=members)
    incidence, x, y = _make(rng, 60, grp, ind)
    assert len(set(np.diff(np.flatnonzero(np.r_[1, np.diff(grp), 1])))) > 1  # uneven sizes
    assert _ones_in_span(incidence)
    model = _fit(cpp, y[grp], x[grp], grp, ind, "sum")
    b_ref, _ = _group_ols(incidence, x, y, with_constant=False)
    assert model.model_has_constant_ is True
    coef = np.asarray(model.coef_)
    assert np.isfinite(coef[2])
    assert np.allclose(coef[:2], b_ref, rtol=0, atol=1e-9)
    centered = float(np.sum((y - y.mean()) ** 2))
    assert abs(float(model.tss_) - centered) <= 1e-9 * centered
    assert abs(float(np.mean(model.residuals_))) < 1e-9


def test_mean_and_ordinary_fe_always_span_constant(cpp):
    rng = np.random.default_rng(3)
    n_ind = 60
    sizes = rng.integers(1, 5, size=400)
    grp, ind = _design(rng, n_ind, sizes)
    incidence, x, y = _make(rng, n_ind, grp, ind)
    model_mean = _fit(cpp, y[grp], x[grp], grp, ind, "mean")
    assert model_mean.model_has_constant_ is True
    assert np.isfinite(np.asarray(model_mean.coef_)[2])
    assert abs(float(np.mean(model_mean.residuals_))) < 1e-9
    period = (grp % 4).astype(np.int64)
    model_fe = _fit(cpp, y[grp], x[grp], grp, ind, "sum", fes=[period])
    assert model_fe.model_has_constant_ is True
    assert np.isfinite(np.asarray(model_fe.coef_)[2])
    assert abs(float(np.mean(model_fe.residuals_))) < 1e-9
