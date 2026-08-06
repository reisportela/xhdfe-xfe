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
