"""Focused H100 regression for a rejected Candidate2 CUDA accuracy retry.

Run with an sm_90 CUDA build, for example:

    XHDFE_TEST_BUILD_DIR=build_candidate2b_failclosed_cuda_sm90_20260829 \
        python tests/test_cuda_accuracy_retry_fail_closed.py
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


CORE_COPIES = (
    ROOT / "src/hdfe_regressor_v11.cpp",
    ROOT / "stata/src/hdfe_regressor_v11.cpp",
    ROOT / "r/xhdfe/src/hdfe_regressor_v11.cpp",
    ROOT / "share/xhdfe_estimation_cpp/src/hdfe_regressor_v11.cpp",
    ROOT / "share/xhdfe_estimation_cpp/stata/src/hdfe_regressor_v11.cpp",
)


def difficult_fixture():
    rng = np.random.default_rng(20260829)
    n = 100_000
    years = 10
    individuals = n // years
    firms = round(individuals / 23)
    individual = np.repeat(
        np.arange(individuals, dtype=np.int32), years
    )
    year = np.tile(np.arange(years, dtype=np.int32), individuals)
    firm = np.tile(
        np.arange(firms, dtype=np.int32), n // firms + 1
    )[:n]
    X = rng.standard_normal((n, 2))
    y = (
        X @ np.array([0.7, -0.2])
        + rng.standard_normal(firms)[firm]
        + rng.standard_normal(individuals)[individual]
        + rng.standard_normal(years)[year]
        + rng.standard_normal(n)
    )
    return y, X, [individual, firm, year]


def make_regressor():
    return cpp.HdfeRegressor(
        se_type="unadjusted",
        tol=1e-8,
        max_iter=300,
        fit_intercept=False,
        num_threads=2,
        drop_singletons=False,
        absorption_method="auto",
        tolerance_mode="reghdfe-comparable",
    )


class CudaAccuracyRetryFailClosedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved_environment = {
            name: os.environ.get(name)
            for name in (
                "XHDFE_GPU_BACKEND",
                "XHDFE_CUDA_AUTO_COMPARABLE_ACCURACY_RETRY",
                "XHDFE_CUDA_AUTO_COMPARABLE_ACCURACY_LADDER",
                "XHDFE_MOBILITY_MODE",
                "XHDFE_ABSORPTION_CACHE_MODE",
                "XHDFE_FE_STRUCTURE_MODE",
            )
        }
        os.environ.update({
            "XHDFE_GPU_BACKEND": "cuda",
            "XHDFE_CUDA_AUTO_COMPARABLE_ACCURACY_LADDER": "0",
            "XHDFE_MOBILITY_MODE": "off",
            "XHDFE_ABSORPTION_CACHE_MODE": "off",
            "XHDFE_FE_STRUCTURE_MODE": "off",
        })

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_rejected_retry_maps_to_status3_and_revokes_results(self):
        canonical = CORE_COPIES[0].read_bytes()
        for mirror in CORE_COPIES[1:]:
            self.assertEqual(mirror.read_bytes(), canonical)
        source = canonical.decode()
        rejection = source.index("if (!retry_accepted)")
        return_site = source.index(
            "return finalize_backend_status(std::move(retried));", rejection
        )
        rejection_block = source[rejection:return_site]
        for required in (
            "retried.gpu_used = false;",
            "retried.gpu_status_code = 3;",
            "retried.gpu_absorption_converged = false;",
            "retried.converged = false;",
            "retried.precision_certified = false;",
        ):
            self.assertIn(required, rejection_block)

        y, X, fes = difficult_fixture()
        os.environ["XHDFE_CUDA_AUTO_COMPARABLE_ACCURACY_RETRY"] = "0"
        control = make_regressor()
        try:
            control.fit(y, X, fes=fes)
        except RuntimeError as exc:
            if "unavailable" in str(exc).lower():
                self.skipTest(f"CUDA unavailable: {exc}")
            raise
        self.assertTrue(control.converged_)
        self.assertTrue(control.precision_certified_)
        self.assertTrue(control.gpu_used_)
        self.assertEqual(control.gpu_status_code_, 1)
        self.assertGreater(control.abs_residual_rel_, 1e-10)

        os.environ["XHDFE_CUDA_AUTO_COMPARABLE_ACCURACY_RETRY"] = "1"
        rejected = make_regressor()
        with self.assertRaisesRegex(
            RuntimeError,
            "Requested GPU backend did not converge during HDFE absorption",
        ):
            rejected.fit(y, X, fes=fes)

        self.assertEqual(rejected.lifecycle_state_, "failed")
        self.assertFalse(rejected.converged_)
        self.assertFalse(rejected.precision_certified_)
        self.assertEqual(rejected.gpu_status_code_, 0)
        self.assertEqual(np.asarray(rejected.coef_).size, 0)
        self.assertEqual(np.asarray(rejected.stderr_).size, 0)
        self.assertEqual(np.asarray(rejected.residuals_).size, 0)
        self.assertEqual(np.asarray(rejected.covariance_).size, 0)
        self.assertEqual(
            rejected.summary(),
            "No estimation result available (state: failed)\n",
        )


if __name__ == "__main__":
    unittest.main()
