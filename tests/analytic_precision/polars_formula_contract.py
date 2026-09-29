"""Independent OLS/HC1 oracle for integer transforms in optional Polars input.

The reference constructs arithmetic terms in float64 before any multiplication;
it never reads the candidate's prepared matrix or covariance to set a target.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import xhdfe

try:
    import formulaic  # noqa: F401
    import pandas as pd
    import polars as pl
except ImportError:
    pd = pl = None


@unittest.skipUnless(pd is not None and pl is not None, "formula dependencies and Polars are required")
class PolarsIntegerPrecisionTest(unittest.TestCase):
    def _compare(self, robust):
        row = np.arange(96)
        x = np.sin(row * 0.7)
        g = row // 32
        for dtype, step in ((np.int16, 100), (np.int32, 10_000), (np.int64, 1_000_000_000)):
            z = np.tile(np.arange(1, 9) * step, 12).astype(dtype)
            scale = float((10 * step) ** 2)
            square = np.square(z.astype(np.float64)) / scale
            y = 0.4 * x + 0.7 * square + 0.2 * (g == 1) + 0.8 * (g == 2) + 0.01 * np.cos(row * 1.3)
            data = {"y": y, "x": x, "g": g, "z": z}
            formula = f"y ~ x + C(g) + I(z**2 / {scale})"
            for container in (pd.DataFrame, pl.DataFrame):
                with self.subTest(dtype=dtype, container=container, robust=robust):
                    options = {"num_threads": 1}
                    if robust:
                        options["se_type"] = "robust"
                    model = xhdfe.feols(formula, container(data), **options)
                    columns = {"x": x, "C(g)[T.1]": g == 1, "C(g)[T.2]": g == 2, "Intercept": np.ones(row.size)}
                    names = model.coef_names_
                    self.assertEqual(len(names), 5)
                    self.assertEqual(sum(name.startswith("I(") for name in names), 1)
                    design = np.column_stack([square if name.startswith("I(") else columns[name] for name in names])
                    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
                    self.assertEqual(rank, design.shape[1])
                    residual = y - design @ beta
                    bread = np.linalg.inv(design.T @ design)
                    n, k = design.shape
                    if robust:
                        scores = design * residual[:, None]
                        variance = (bread @ (scores.T @ scores) @ bread) * n / (n - k)
                    else:
                        variance = bread * (residual @ residual) / (n - k)
                    limit_b = 1e-9 * np.maximum(1.0, np.abs(beta))
                    self.assertTrue(np.all(np.abs(model.coef_ - beta) <= limit_b), (model.coef_, beta))
                    scale_v = np.sqrt(np.outer(np.diag(variance), np.diag(variance)))
                    self.assertLessEqual(float(np.max(np.abs(model.covariance_ - variance) / scale_v)), 1e-8)
                    np.testing.assert_allclose(model.residuals_, residual, rtol=0, atol=1e-11)
                    self.assertTrue(model.converged_)
                    self.assertEqual(model.nobs_, n)
                    self.assertEqual(model.df_resid_, n - k)
                    np.testing.assert_array_equal(model.sample_index_, row)

    def test_integer_squares_match_explicit_ols(self):
        self._compare(robust=False)

    def test_integer_squares_match_explicit_hc1(self):
        self._compare(robust=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
