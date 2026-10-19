"""Adversarial regressions for CG scale invariance and solved RHS lanes.

Run against a local CMake build, for example:

    XHDFE_TEST_BUILD_DIR=build_cuda python tests/test_cg_breakdown.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / os.environ.get("XHDFE_TEST_BUILD_DIR", "build_cuda")
sys.path.insert(0, str(BUILD_DIR))
import py_hdfe_v11 as cpp


def _fit(y, X, fes, *, convergence, weights=None):
    reg = cpp.HdfeRegressor(
        se_type="unadjusted",
        tol=1e-12,
        max_iter=100,
        fit_intercept=False,
        drop_singletons=False,
        num_threads=1,
        convergence=convergence,
        absorption_method="gauss-seidel",
        tolerance_mode="reghdfe-comparable",
    )
    kwargs = {"fes": fes}
    if weights is not None:
        kwargs["weights"] = weights
    reg.fit(y, X, **kwargs)
    assert reg.converged_
    assert reg.precision_certified_
    return reg


def _packed_fixture(scale):
    os.environ["XHDFE_PACKED"] = "1"
    i = np.arange(120)
    f1 = (i % 20).astype(np.int32)
    f2 = ((i * 7 + i // 20) % 20).astype(np.int32)
    y = 0.7 * np.sin(i / 3) + 0.2 * (f1 - 9.5) + 0.1 * (f2 - 9.5)
    X = np.cos(i / 5)[:, None]
    return _fit(
        scale * y,
        scale * X,
        [f1, f2],
        convergence="reghdfe",
    )


def _soa_fixture(scale):
    os.environ["XHDFE_PACKED"] = "0"
    i = np.arange(40)
    f1 = (i // 2).astype(np.int32)
    f2 = ((i + 1) // 2).astype(np.int32)
    y = np.sin(i / 3) + 0.2 * f1 - 0.1 * f2
    X = np.cos(i / 5)[:, None]
    return _fit(
        scale * y,
        scale * X,
        [f1, f2],
        convergence="auto",
        weights=np.ones(i.size),
    )


class CgBreakdownTest(unittest.TestCase):
    def test_near_redundant_fe_lanes_against_independent_projection(self):
        # Issue #16: most schools have age zero; only late entrants identify
        # the third FE. The rare outcome is solved before the random X lane.
        os.environ["XHDFE_PACKED"] = "1"
        sid = np.repeat(np.arange(1536), 5)
        year = np.tile(np.arange(2015, 2020), 1536)
        first = np.select(
            [sid < 1515, sid < 1522, sid < 1526, sid < 1532],
            [0, 2015, 2016, 2017], default=2018,
        )
        keep = year >= first
        sid, year, first = sid[keep], year[keep], first[keep]
        age = np.where(first == 0, 0, year - first + 1)
        fes = [v.astype(np.int32) for v in (sid, year - 2015, age)]
        rng = np.random.default_rng(16)
        X = rng.normal(size=(sid.size, 1))
        rare = ((sid < 2) | ((sid == 2) & (year != 2017)) |
                ((sid == 3) & (year == 2017))).astype(float)
        aggregate = 10 + rng.normal(size=1536)[sid] + .03*(year-2017)
        aggregate += .02*rng.normal(size=sid.size)

        def within(values):
            # Independent, unaccelerated alternating projections. No xhdfe
            # solver/certificate supplies the expected answer.
            out = values.copy()
            for _ in range(10000):
                before = out.copy()
                for fe in fes:
                    counts = np.bincount(fe)
                    for j in range(out.shape[1]):
                        means = np.bincount(fe, weights=out[:, j]) / counts
                        out[:, j] -= means[fe]
                if np.linalg.norm(out-before) < 1e-13:
                    return out
            self.fail("independent projection did not converge")

        for y in (rare, aggregate):
            projected = within(np.column_stack((y, X)))
            expected = np.linalg.lstsq(projected[:, 1:], projected[:, 0], rcond=None)[0]
            for scale in (1., 1e-10, 1e10):
                for order in (np.arange(sid.size), rng.permutation(sid.size)):
                    for convergence in ("auto", "reghdfe"):
                        with self.subTest(outcome="rare" if y is rare else "aggregate",
                                          scale=scale, convergence=convergence):
                            fit = _fit(scale*y[order], scale*X[order],
                                       [fe[order] for fe in fes],
                                       convergence=convergence)
                            np.testing.assert_allclose(fit.coef_, expected,
                                                       rtol=1e-8, atol=1e-12)

    def test_soa_early_solved_lane_against_explicit_dummies(self):
        i = np.arange(102)
        f1 = np.where(i < 3, 0, np.where(i < 99, 1+(i-3)//2, 49)).astype(np.int32)
        f2 = np.where(i < 2, 0, np.where(i == 2, 1,
                      np.where(i < 99, f1+(i-3)%2, np.where(i == 99, 49, 50)))).astype(np.int32)
        X = np.random.default_rng(91).normal(size=(i.size, 1))
        y = f1/7.0
        y[:2] += [1., -1.]
        dummy = np.column_stack([np.eye(fe.max()+1)[fe] for fe in (f1, f2)])
        expected = np.linalg.lstsq(np.column_stack((X, dummy)), y, rcond=None)[0][:1]
        # Unit weights explicitly select the SoA path rather than the packed
        # implementation, even when another test initialized the packed gate.
        for scale in (1., 1e-10, 1e10):
            fit = _fit(scale*y, scale*X, [f1, f2], convergence="auto",
                       weights=np.ones(i.size))
            np.testing.assert_allclose(fit.coef_, expected, rtol=1e-8, atol=1e-12)

    def test_cg_is_scale_invariant_and_precision_certified(self):
        for fixture in (_packed_fixture, _soa_fixture):
            with self.subTest(fixture=fixture.__name__):
                unit = fixture(1.0)
                tiny = fixture(1e-10)
                self.assertEqual(tiny.num_iterations_, unit.num_iterations_)
                np.testing.assert_allclose(
                    tiny.coef_, unit.coef_, rtol=1e-8, atol=1e-12
                )

    def test_cg_accepts_an_exactly_solved_zero_rhs_lane(self):
        os.environ["XHDFE_PACKED"] = "1"
        i = np.arange(120)
        f1 = (i % 20).astype(np.int32)
        f2 = ((i * 7 + i // 20) % 20).astype(np.int32)
        y = 0.7 * np.sin(i / 3) + 0.2 * (f1 - 9.5) + 0.1 * (f2 - 9.5)
        reg = _fit(
            y,
            np.zeros((i.size, 1)),
            [f1, f2],
            convergence="reghdfe",
        )
        np.testing.assert_array_equal(reg.coef_, np.zeros(1))


if __name__ == "__main__":
    unittest.main()
