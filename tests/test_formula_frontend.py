from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import xhdfe

# Some native-core tests import a CMake artifact as top-level ``py_hdfe_v11``.
# Reuse that already-registered pybind11 module under the package name when the
# complete unittest discovery suite runs in one process.
if "py_hdfe_v11" in sys.modules:
    sys.modules.setdefault("xhdfe.py_hdfe_v11", sys.modules["py_hdfe_v11"])

try:
    import formulaic  # noqa: F401
    import pandas as pd
except ImportError:
    pd = None


HAS_FORMULA_DEPENDENCIES = pd is not None
REGRESSOR_OPTIONS = {
    "num_threads": 1,
    "drop_singletons": False,
    "tol": 1e-10,
    "max_iter": 20_000,
}


def _fit_native(y, X, fes, **options):
    constructor_options = dict(REGRESSOR_OPTIONS)
    constructor_options.update(options)
    model = xhdfe.HdfeRegressor(**constructor_options)
    model.fit(
        np.ascontiguousarray(y, dtype=np.float64),
        np.asfortranarray(X, dtype=np.float64),
        fes=[np.ascontiguousarray(fe, dtype=np.int64) for fe in fes],
    )
    return model


def _assert_native_parity(testcase, actual, expected, *, rtol=2e-10, atol=2e-11):
    for name in ("coef_", "stderr_", "covariance_", "residuals_"):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=name,
        )
    for name in (
        "converged_",
        "num_iterations_",
        "nobs_",
        "nobs_full_",
        "num_singletons_",
        "df_a_",
        "df_m_",
        "df_resid_",
    ):
        testcase.assertEqual(getattr(actual, name), getattr(expected, name), name)
    np.testing.assert_array_equal(actual.sample_index_, expected.sample_index_)


@unittest.skipUnless(HAS_FORMULA_DEPENDENCIES, "formulaic extra is not installed")
class FormulaFrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20260814)
        n = 720
        firm = rng.integers(0, 45, size=n, dtype=np.int64)
        year = rng.integers(0, 9, size=n, dtype=np.int64)
        g = np.resize(np.array([0, 1, 2], dtype=np.int64), n)
        rng.shuffle(g)
        x = rng.normal(size=n)
        z = rng.normal(size=n)
        firm_effect = rng.normal(scale=0.6, size=45)
        year_effect = rng.normal(scale=0.3, size=9)
        y = (
            0.7
            + 1.2 * x
            - 0.35 * z
            + 0.5 * x * z
            + 0.25 * (g == 1)
            - 0.4 * (g == 2)
            + firm_effect[firm]
            + year_effect[year]
            + rng.normal(scale=0.4, size=n)
        )
        index = np.asarray([10_000 + (17 * i) % 311 for i in range(n)])
        cls.data = pd.DataFrame(
            {
                "y": y,
                "x": x,
                "z": z,
                "g": g,
                "firm": firm,
                "year": year,
                "firm_text": np.asarray([f"f{value}" for value in firm], dtype=object),
                "year_text": np.asarray([f"y{value}" for value in year], dtype=object),
                "cluster_text": np.asarray([f"c{value % 18}" for value in firm], dtype=object),
                "weight": rng.uniform(0.7, 1.8, size=n),
                "fw": rng.integers(1, 4, size=n),
                "Intercept": rng.normal(size=n),
            },
            index=index,
        )

    def test_numeric_main_effects_and_interaction_match_array_api(self):
        d = self.data
        X = np.column_stack((d["x"], d["z"], d["x"] * d["z"]))
        expected = _fit_native(d["y"], X, [d["firm"], d["year"]])
        actual = xhdfe.feols(
            "y ~ x*z | firm + year",
            d,
            **REGRESSOR_OPTIONS,
        )

        _assert_native_parity(self, actual, expected)
        self.assertIsInstance(actual, xhdfe.HdfeRegressor)
        self.assertTrue(actual.used_fast_path_)
        self.assertEqual(actual.coef_names_, ("x", "z", "x:z", "Intercept"))
        self.assertEqual(actual.fe_names_, ("firm", "year"))
        self.assertEqual(tuple(actual.tidy().index), actual.coef_names_)
        np.testing.assert_array_equal(actual.estimation_index_, d.index.to_numpy())

    def test_colon_is_product_only(self):
        d = self.data
        X = np.asarray(d["x"] * d["z"])[:, None]
        expected = _fit_native(d["y"], X, [d["firm"], d["year"]])
        actual = xhdfe.feols(
            "y ~ x:z | firm + year",
            d,
            **REGRESSOR_OPTIONS,
        )
        _assert_native_parity(self, actual, expected)
        self.assertEqual(actual.coef_names_, ("x:z", "Intercept"))
        self.assertTrue(actual.used_fast_path_)

    def test_numeric_fast_path_promotes_before_interaction_arithmetic(self):
        d = pd.DataFrame(
            {
                "y": np.arange(4, dtype=np.float64),
                "x": np.asarray(
                    [3_100_000_000, 3_200_000_000, 3_300_000_000, 3_400_000_000],
                    dtype=np.int64,
                ),
                "z": np.asarray(
                    [3_500_000_000, 3_600_000_000, 3_700_000_000, 3_800_000_000],
                    dtype=np.int64,
                ),
            }
        )
        prepared = xhdfe.prepare_formula("y ~ 0 + x:z", d)
        expected = (
            d["x"].to_numpy(dtype=np.float64)
            * d["z"].to_numpy(dtype=np.float64)
        )

        self.assertTrue(prepared.used_fast_path)
        np.testing.assert_array_equal(prepared.X[:, 0], expected)
        self.assertTrue(np.all(prepared.X[:, 0] > 0.0))

    def test_mixed_formulaic_path_promotes_lookup_interactions_to_float64(self):
        d = pd.DataFrame(
            {
                "y": np.arange(4, dtype=np.float64),
                "g": [0, 1, 0, 1],
                "x": np.asarray([2**62, 2**61, 2**60, 2**59], dtype=np.int64),
                "z": np.asarray([4, 3, 5, 7], dtype=np.int64),
            }
        )
        prepared = xhdfe.prepare_formula("y ~ C(g) + x:z", d)
        interaction = prepared.coef_names.index("x:z")
        expected = (
            d["x"].to_numpy(dtype=np.float64)
            * d["z"].to_numpy(dtype=np.float64)
        )

        self.assertFalse(prepared.used_fast_path)
        np.testing.assert_array_equal(prepared.X[:, interaction], expected)
        self.assertTrue(np.all(prepared.X[:, interaction] > 0.0))

        d["x"] = np.asarray([100_000.05, -100_000.06, 90_000.07, -90_000.08], dtype=np.float32)
        d["z"] = np.asarray([-99_999.85, 99_999.84, -80_000.03, 80_000.02], dtype=np.float32)
        prepared_float32 = xhdfe.prepare_formula("y ~ C(g) + x:z", d)
        interaction = prepared_float32.coef_names.index("x:z")
        expected_float32 = (
            d["x"].to_numpy(dtype=np.float64)
            * d["z"].to_numpy(dtype=np.float64)
        )
        np.testing.assert_array_equal(
            prepared_float32.X[:, interaction],
            expected_float32,
        )

    def test_square_requires_arithmetic_identity_wrapper(self):
        d = self.data
        expected = _fit_native(
            d["y"],
            np.square(np.asarray(d["x"], dtype=np.float64))[:, None],
            [d["firm"], d["year"]],
        )
        square = xhdfe.feols(
            "y ~ I(x**2) | firm + year",
            d,
            **REGRESSOR_OPTIONS,
        )
        _assert_native_parity(self, square, expected)
        self.assertEqual(square.coef_names_, ("I(x ** 2)", "Intercept"))

        for formula in ("y ~ x:x | firm + year", "y ~ x**2 | firm + year"):
            with self.subTest(formula=formula):
                prepared = xhdfe.prepare_formula(formula, d)
                np.testing.assert_array_equal(prepared.X[:, 0], d["x"])
                self.assertEqual(prepared.coef_names, ("x", "Intercept"))

        large = pd.DataFrame(
            {
                "y": np.arange(4, dtype=np.float64),
                "x": np.asarray(
                    [4_000_000_000, 4_100_000_000, 4_200_000_000, 4_300_000_000],
                    dtype=np.int64,
                ),
            }
        )
        large_square = xhdfe.prepare_formula("y ~ 0 + I(x**2)", large)
        np.testing.assert_array_equal(
            large_square.X[:, 0],
            np.square(large["x"].to_numpy(dtype=np.float64)),
        )
        self.assertTrue(np.all(large_square.X[:, 0] > 0.0))

        transformed_response = xhdfe.prepare_formula("I(x**2) ~ 0 + y", large)
        np.testing.assert_array_equal(
            transformed_response.y,
            np.square(large["x"].to_numpy(dtype=np.float64)),
        )

        quoted_square = xhdfe.prepare_formula('y ~ 0 + I(Q("x")**2)', large)
        np.testing.assert_array_equal(
            quoted_square.X[:, 0],
            np.square(large["x"].to_numpy(dtype=np.float64)),
        )
        quoted_response = xhdfe.prepare_formula('I(Q("x")**2) ~ 0 + y', large)
        np.testing.assert_array_equal(
            quoted_response.y,
            np.square(large["x"].to_numpy(dtype=np.float64)),
        )

    def test_fast_path_matches_formulaic_for_float64_supported_terms(self):
        from xhdfe import _formula

        for formula in (
            "y ~ x + z | firm + year",
            "y ~ z*x | firm + year",
            "y ~ x*z + x:z | firm + year",
            "y ~ 0 + x:z | firm + year",
        ):
            with self.subTest(formula=formula):
                fast = _formula._materialize_formula(
                    formula,
                    self.data,
                    weights=None,
                    clusters=None,
                    fweights=False,
                    na_action="raise",
                    context=None,
                )
                generic = _formula._materialize_formula(
                    formula,
                    self.data,
                    weights=None,
                    clusters=None,
                    fweights=False,
                    na_action="raise",
                    context=None,
                    force_formulaic=True,
                )
                self.assertTrue(fast.used_fast_path)
                np.testing.assert_array_equal(fast.y, generic.y)
                np.testing.assert_array_equal(fast.X, generic.X)
                self.assertEqual(fast.coef_names, generic.coef_names)
                self.assertEqual(fast.fit_intercept, generic.fit_intercept)

    def test_formula_without_fixed_effects_matches_array_api(self):
        d = self.data
        expected = _fit_native(d["y"], np.asarray(d["x"])[:, None], [])
        actual = xhdfe.feols("y ~ x", d, **REGRESSOR_OPTIONS)

        _assert_native_parity(self, actual, expected)
        self.assertEqual(actual.fe_names_, ())
        self.assertEqual(actual.coef_names_, ("x", "Intercept"))
        self.assertEqual(actual.intercept_index_, 1)

    def test_intercept_only_and_empty_rhs_are_supported(self):
        d = self.data
        for formula, expected_names in (
            ("y ~ 1 | firm + year", ("Intercept",)),
            ("y ~ 0 | firm + year", ()),
        ):
            with self.subTest(formula=formula):
                model = xhdfe.feols(formula, d, **REGRESSOR_OPTIONS)
                self.assertTrue(model.converged_)
                self.assertEqual(model.coef_names_, expected_names)
                self.assertEqual(len(model.coef_), len(expected_names))

    def test_mapping_supports_categories_and_transforms(self):
        d = self.data
        mapping = {name: d[name].to_numpy(copy=True) for name in d.columns}
        g1 = np.asarray(d["g"] == 1, dtype=np.float64)
        g2 = np.asarray(d["g"] == 2, dtype=np.float64)
        X = np.column_stack((g1, g2, np.square(np.asarray(d["x"]))))
        expected = _fit_native(d["y"], X, [d["firm"], d["year"]])
        actual = xhdfe.feols(
            "y ~ C(g) + I(x**2) | firm + year",
            mapping,
            **REGRESSOR_OPTIONS,
        )

        _assert_native_parity(self, actual, expected)
        self.assertEqual(
            actual.coef_names_,
            ("C(g)[T.1]", "C(g)[T.2]", "I(x ** 2)", "Intercept"),
        )
        np.testing.assert_array_equal(actual.data_index_, np.arange(len(d)))

    def test_explicit_context_transform_matches_manual_array(self):
        d = self.data

        def demean(values):
            values = np.asarray(values, dtype=np.float64)
            return values - values.mean()

        expected = _fit_native(
            d["y"],
            demean(d["x"])[:, None],
            [d["firm"], d["year"]],
        )
        actual = xhdfe.feols(
            "y ~ demean(x) | firm + year",
            d,
            context={"demean": demean},
            **REGRESSOR_OPTIONS,
        )
        _assert_native_parity(self, actual, expected)

    def test_treatment_categories_and_full_factorial_match_manual_design(self):
        d = self.data
        g1 = np.asarray(d["g"] == 1, dtype=np.float64)
        g2 = np.asarray(d["g"] == 2, dtype=np.float64)
        x = d["x"].to_numpy()
        X = np.column_stack((g1, g2, x, g1 * x, g2 * x))
        expected = _fit_native(d["y"], X, [d["firm"], d["year"]])
        actual = xhdfe.feols(
            "y ~ C(g)*x | firm + year",
            d,
            **REGRESSOR_OPTIONS,
        )

        _assert_native_parity(self, actual, expected)
        self.assertFalse(actual.used_fast_path_)
        self.assertEqual(
            actual.coef_names_,
            (
                "C(g)[T.1]",
                "C(g)[T.2]",
                "x",
                "C(g)[T.1]:x",
                "C(g)[T.2]:x",
                "Intercept",
            ),
        )
        self.assertIsNotNone(actual.model_spec_)

    def test_explicit_reference_category_matches_manual_design(self):
        d = self.data
        g0 = np.asarray(d["g"] == 0, dtype=np.float64)
        g1 = np.asarray(d["g"] == 1, dtype=np.float64)
        x = d["x"].to_numpy()
        X = np.column_stack((g0, g1, x, g0 * x, g1 * x))
        expected = _fit_native(d["y"], X, [d["firm"], d["year"]])
        actual = xhdfe.feols(
            "y ~ C(g, Treatment(reference=2))*x | firm + year",
            d,
            **REGRESSOR_OPTIONS,
        )

        _assert_native_parity(self, actual, expected)
        self.assertEqual(actual.coef_names_[-1], "Intercept")
        self.assertIn("[T.0]", actual.coef_names_[0])
        self.assertIn("[T.1]", actual.coef_names_[1])

    def test_declared_categorical_order_controls_default_reference(self):
        d = self.data.copy()
        labels = np.where(
            np.asarray(d["g"]) == 0,
            "zero",
            np.where(np.asarray(d["g"]) == 1, "one", "two"),
        )
        d["g_ordered"] = pd.Categorical(
            labels,
            categories=["two", "zero", "one"],
            ordered=True,
        )
        zero = np.asarray(labels == "zero", dtype=np.float64)
        one = np.asarray(labels == "one", dtype=np.float64)
        expected = _fit_native(
            d["y"],
            np.column_stack((zero, one)),
            [d["firm"], d["year"]],
        )
        actual = xhdfe.feols(
            "y ~ C(g_ordered) | firm + year",
            d,
            **REGRESSOR_OPTIONS,
        )

        _assert_native_parity(self, actual, expected)
        self.assertEqual(
            actual.coef_names_,
            ("C(g_ordered)[T.zero]", "C(g_ordered)[T.one]", "Intercept"),
        )

    def test_numeric_column_cannot_change_dtype_between_category_and_numeric_roles(self):
        large = np.int64(2**53)
        d = pd.DataFrame(
            {
                "y": np.arange(6, dtype=np.float64),
                "x": np.linspace(0.5, 1.0, 6),
                "g": np.asarray(
                    [large, large + 1, large, large + 1, large, large + 1],
                    dtype=np.int64,
                ),
            }
        )
        categorical = xhdfe.prepare_formula("y ~ C(g)", d)
        self.assertIn(str(int(large + 1)), categorical.coef_names[0])
        quoted_categorical = xhdfe.prepare_formula('y ~ C(Q("g"))', d)
        self.assertIn(str(int(large + 1)), quoted_categorical.coef_names[0])
        with self.assertRaisesRegex(ValueError, "both categorically"):
            xhdfe.prepare_formula("y ~ C(g) + g:x", d)
        with self.assertRaisesRegex(NotImplementedError, "bare column name"):
            xhdfe.prepare_formula("y ~ 0 + C(g**2)", d)

    def test_category_colon_uses_formulaic_group_slope_semantics(self):
        d = self.data
        x = d["x"].to_numpy()
        X = np.column_stack(
            tuple(np.asarray(d["g"] == level, dtype=np.float64) * x for level in range(3))
        )
        expected = _fit_native(d["y"], X, [d["firm"], d["year"]])
        actual = xhdfe.feols(
            "y ~ C(g):x | firm + year",
            d,
            **REGRESSOR_OPTIONS,
        )

        _assert_native_parity(self, actual, expected)
        self.assertEqual(
            actual.coef_names_,
            ("C(g)[0]:x", "C(g)[1]:x", "C(g)[2]:x", "Intercept"),
        )

    def test_no_intercept_keeps_all_category_columns(self):
        d = self.data
        X = np.column_stack(
            tuple(np.asarray(d["g"] == level, dtype=np.float64) for level in range(3))
        )
        expected = _fit_native(
            d["y"],
            X,
            [d["firm"], d["year"]],
            fit_intercept=False,
        )
        actual = xhdfe.feols(
            "y ~ 0 + C(g) | firm + year",
            d,
            **REGRESSOR_OPTIONS,
        )

        _assert_native_parity(self, actual, expected)
        self.assertEqual(actual.coef_names_, ("C(g)[0]", "C(g)[1]", "C(g)[2]"))
        self.assertIsNone(actual.intercept_index_)
        with self.assertRaisesRegex(TypeError, "controlled by the formula"):
            xhdfe.feols(
                "y ~ x | firm",
                d,
                fit_intercept=False,
                **REGRESSOR_OPTIONS,
            )

    def test_real_predictor_named_intercept_is_not_dropped(self):
        from xhdfe import _formula

        d = self.data
        expected = _fit_native(
            d["y"],
            np.asarray(d["Intercept"])[:, None],
            [d["firm"], d["year"]],
        )
        materialized = _formula._materialize_formula(
            "y ~ Intercept | firm + year",
            d,
            weights=None,
            clusters=None,
            fweights=False,
            na_action="raise",
            context=None,
            force_formulaic=True,
        )
        actual = _formula._fit_materialized(materialized, REGRESSOR_OPTIONS)

        _assert_native_parity(self, actual, expected)
        self.assertEqual(actual.coef_names_, ("Intercept", "Intercept [xhdfe]"))
        self.assertEqual(actual.intercept_index_, 1)
        self.assertEqual(len(set(actual.coef_names_)), len(actual.coef_names_))
        self.assertEqual(len(actual.coef_names_), len(actual.coef_))

    def test_string_fixed_effects_are_factorized_not_dummy_encoded(self):
        d = self.data
        firm_codes, firm_levels = pd.factorize(d["firm_text"], sort=False)
        year_codes, year_levels = pd.factorize(d["year_text"], sort=False)
        expected = _fit_native(d["y"], np.asarray(d["x"])[:, None], [firm_codes, year_codes])
        prepared = xhdfe.prepare_formula(
            "y ~ x | firm_text + year_text",
            d,
            **REGRESSOR_OPTIONS,
        )
        actual = prepared.fit()

        _assert_native_parity(self, actual, expected)
        self.assertEqual(prepared.X.shape, (len(d), 1))
        self.assertEqual(actual.fe_names_, ("firm_text", "year_text"))
        np.testing.assert_array_equal(actual.fe_levels_["firm_text"], firm_levels)
        np.testing.assert_array_equal(actual.fe_levels_["year_text"], year_levels)

    def test_large_int64_fixed_effect_ids_preserve_exact_identity(self):
        d = self.data.copy()
        offset = np.int64(2**53 + 101)
        d["firm_big"] = offset + d["firm"].to_numpy(dtype=np.int64)
        prepared = xhdfe.prepare_formula(
            "y ~ x | firm_big + year",
            d,
            **REGRESSOR_OPTIONS,
        )
        np.testing.assert_array_equal(prepared.fes[0], d["firm_big"])
        self.assertNotEqual(int(prepared.fes[0][0]), int(prepared.fes[0][0] + 1))

        expected = _fit_native(d["y"], np.asarray(d["x"])[:, None], [d["firm_big"], d["year"]])
        actual = prepared.fit()
        _assert_native_parity(self, actual, expected)

    def test_negative_integer_and_nonfinite_ids_fail_closed(self):
        cases = (
            (-1, "negative integer ID"),
            (-1.0, "negative numeric ID"),
            (np.inf, "non-finite ID"),
        )
        for bad_value, message in cases:
            with self.subTest(surface="fixed effect", bad_value=bad_value):
                d = self.data.copy()
                dtype = (
                    np.float64
                    if isinstance(bad_value, (float, np.floating))
                    else np.int64
                )
                d["bad_id"] = np.asarray(d["firm"], dtype=dtype)
                d.iloc[7, d.columns.get_loc("bad_id")] = bad_value
                with self.assertRaisesRegex(ValueError, message):
                    xhdfe.feols(
                        "y ~ x | bad_id + year",
                        d,
                        **REGRESSOR_OPTIONS,
                    )
            with self.subTest(surface="cluster", bad_value=bad_value):
                cluster = np.asarray(
                    self.data["firm"],
                    dtype=dtype,
                ).copy()
                cluster[7] = bad_value
                with self.assertRaisesRegex(ValueError, message):
                    xhdfe.feols(
                        "y ~ x | firm + year",
                        self.data,
                        clusters=cluster,
                        se_type="cluster",
                        **REGRESSOR_OPTIONS,
                    )

    def test_cluster_input_shapes_have_unambiguous_orientation(self):
        from xhdfe import _formula

        n = 6
        d = pd.DataFrame(
            {
                "y": np.arange(n, dtype=np.float64),
                "x": np.linspace(-1.0, 1.0, n),
            }
        )
        vector = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
        one = _formula._materialize_formula(
            "y ~ x",
            d,
            weights=None,
            clusters=vector,
            fweights=False,
            na_action="raise",
            context=None,
        )
        self.assertEqual(len(one.clusters), 1)
        string_list = _formula._materialize_formula(
            "y ~ x",
            d,
            weights=None,
            clusters=vector.tolist(),
            fweights=False,
            na_action="raise",
            context=None,
        )
        np.testing.assert_array_equal(string_list.clusters[0], one.clusters[0])

        matrix = np.column_stack(tuple((np.arange(n) + j) % (j + 2) for j in range(n)))
        row_matrix = _formula._materialize_formula(
            "y ~ x",
            d,
            weights=None,
            clusters=matrix,
            fweights=False,
            na_action="raise",
            context=None,
        )
        sequence = _formula._materialize_formula(
            "y ~ x",
            d,
            weights=None,
            clusters=[matrix[:, j].copy() for j in range(n)],
            fweights=False,
            na_action="raise",
            context=None,
        )
        self.assertEqual(len(row_matrix.clusters), n)
        self.assertEqual(len(sequence.clusters), n)
        for j in range(n):
            np.testing.assert_array_equal(row_matrix.clusters[j], matrix[:, j])
            np.testing.assert_array_equal(sequence.clusters[j], matrix[:, j])

    def test_weights_and_named_clusters_match_array_api(self):
        d = self.data
        cluster_codes, _ = pd.factorize(d["cluster_text"], sort=False)
        options = dict(REGRESSOR_OPTIONS, se_type="cluster")
        expected = xhdfe.HdfeRegressor(**options)
        expected.fit(
            np.asarray(d["y"], dtype=np.float64),
            np.asfortranarray(np.asarray(d["x"], dtype=np.float64)[:, None]),
            fes=[np.asarray(d["firm"], dtype=np.int64), np.asarray(d["year"], dtype=np.int64)],
            weights=np.asarray(d["weight"], dtype=np.float64),
            clusters=[np.asarray(cluster_codes, dtype=np.int64)],
        )
        actual = xhdfe.feols(
            "y ~ x | firm + year",
            d,
            weights="weight",
            clusters="cluster_text",
            **options,
        )

        _assert_native_parity(self, actual, expected)
        self.assertIn("cluster_text", actual.cluster_levels_)
        self.assertNotIn("cluster_text", actual.fe_levels_)

    def test_frequency_weights_when_supported_by_loaded_core(self):
        if "fweights" not in (xhdfe.HdfeRegressor.fit.__doc__ or ""):
            self.skipTest("the local compiled extension predates fweights")
        d = self.data
        expected = xhdfe.HdfeRegressor(**REGRESSOR_OPTIONS)
        expected.fit(
            np.asarray(d["y"], dtype=np.float64),
            np.asfortranarray(np.asarray(d["x"], dtype=np.float64)[:, None]),
            fes=[np.asarray(d["firm"], dtype=np.int64), np.asarray(d["year"], dtype=np.int64)],
            weights=np.asarray(d["fw"], dtype=np.float64),
            fweights=True,
        )
        actual = xhdfe.feols(
            "y ~ x | firm + year",
            d,
            weights="fw",
            fweights=True,
            **REGRESSOR_OPTIONS,
        )
        _assert_native_parity(self, actual, expected)

    def test_frequency_weights_require_a_weight_vector_before_native_fit(self):
        with self.assertRaisesRegex(ValueError, "requires weights"):
            xhdfe.feols(
                "y ~ x | firm + year",
                self.data,
                fweights=True,
                **REGRESSOR_OPTIONS,
            )

    def test_frequency_weight_preflight_preserves_integer_contract(self):
        d = pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0]})
        accepted = xhdfe.prepare_formula(
            "y ~ x",
            d,
            weights=np.asarray([2**53, 1], dtype=np.int64),
            fweights=True,
        )
        self.assertEqual(accepted.formula, "y ~ x")

        invalid = (
            (np.asarray([0, 1], dtype=np.int64), "positive integers"),
            (np.asarray([1.5, 1.0]), "positive integers"),
            (np.asarray([2**53 + 1, 1], dtype=np.int64), "exactly"),
            (np.asarray([2**62, 2**62], dtype=np.int64), "overflows int64"),
        )
        for values, message in invalid:
            with self.subTest(values=values, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    xhdfe.prepare_formula(
                        "y ~ x",
                        d,
                        weights=values,
                        fweights=True,
                    )
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            xhdfe.prepare_formula("y ~ x", d, weights=[1, 1], fweights=1)

    def test_nullable_string_category_has_actionable_error_and_explicit_C_works(self):
        d = self.data.copy()
        d["g_string"] = pd.Series(
            np.where(np.asarray(d["g"]) == 0, "a", np.where(np.asarray(d["g"]) == 1, "b", "c")),
            dtype="string",
            index=d.index,
        )
        with self.assertRaisesRegex(ValueError, r"C\(\.\.\.\)"):
            xhdfe.feols(
                "y ~ g_string | firm + year",
                d,
                **REGRESSOR_OPTIONS,
            )
        model = xhdfe.feols(
            "y ~ C(g_string) | firm + year",
            d,
            **REGRESSOR_OPTIONS,
        )
        self.assertEqual(
            model.coef_names_,
            ("C(g_string)[T.b]", "C(g_string)[T.c]", "Intercept"),
        )

    def test_missing_values_fail_closed_for_every_input_surface(self):
        cases = (
            ("y", {}, {}),
            ("x", {}, {}),
            ("firm", {}, {}),
            ("weight", {"weights": "weight"}, {}),
            ("cluster_text", {"clusters": "cluster_text"}, {}),
        )
        for column, call_options, _ in cases:
            with self.subTest(column=column):
                d = self.data.copy()
                d[column] = d[column].astype(object)
                d.iloc[5, d.columns.get_loc(column)] = None
                with self.assertRaises(ValueError):
                    xhdfe.feols(
                        "y ~ x | firm + year",
                        d,
                        **call_options,
                        **REGRESSOR_OPTIONS,
                    )
        with self.assertRaisesRegex(NotImplementedError, "only na_action='raise'"):
            xhdfe.feols(
                "y ~ x | firm + year",
                self.data,
                na_action="drop",
                **REGRESSOR_OPTIONS,
            )

    def test_prepared_formula_is_a_read_only_explicit_snapshot(self):
        d = self.data.copy(deep=True)
        prepared = xhdfe.prepare_formula(
            "y ~ x:z | firm + year",
            d,
            **REGRESSOR_OPTIONS,
        )
        y_before = prepared.y.copy()
        X_before = prepared.X.copy()
        fes_before = tuple(fe.copy() for fe in prepared.fes)

        d.loc[:, "y"] = 999.0
        d.loc[:, "x"] = -999.0
        d.loc[:, "firm"] = 0

        np.testing.assert_array_equal(prepared.y, y_before)
        np.testing.assert_array_equal(prepared.X, X_before)
        for actual, expected in zip(prepared.fes, fes_before):
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(prepared.y.flags.writeable)
        self.assertFalse(prepared.X.flags.writeable)
        self.assertTrue(prepared.X.flags.f_contiguous)
        with self.assertRaises(ValueError):
            prepared.X[0, 0] = 0.0

        first = prepared.fit()
        second = prepared.fit()
        _assert_native_parity(self, first, second, rtol=0.0, atol=0.0)

    def test_formula_metadata_is_cleared_on_manual_refit(self):
        d = self.data
        model = xhdfe.feols(
            "y ~ x | firm + year",
            d,
            **REGRESSOR_OPTIONS,
        )
        self.assertTrue(hasattr(model, "formula_"))
        model.fit(
            np.asarray(d["y"], dtype=np.float64),
            np.asfortranarray(np.asarray(d["z"], dtype=np.float64)[:, None]),
            fes=[np.asarray(d["firm"], dtype=np.int64)],
        )
        for name in (
            "formula_",
            "coef_names_",
            "intercept_index_",
            "fe_names_",
            "cluster_levels_",
            "model_spec_",
            "estimation_index_",
        ):
            self.assertFalse(hasattr(model, name), name)
        self.assertEqual(tuple(model.tidy().index), ("b0", "b1"))

    def test_singleton_sample_metadata_uses_positions_not_index_labels(self):
        d = pd.DataFrame(
            {
                "y": np.asarray([1.0, 1.4, 1.8, 2.2, 2.5, 2.9, 3.3, 3.8]),
                "x": np.asarray([-1.0, -0.5, 0.1, 0.4, 0.8, 1.2, 1.7, 2.1]),
                "firm": np.asarray([0, 0, 0, 1, 2, 2, 2, 2]),
            },
            index=[10, 10, 20, 99, 30, 30, 40, 40],
        )
        model = xhdfe.feols("y ~ x | firm", d, num_threads=1)
        expected_positions = np.asarray([0, 1, 2, 4, 5, 6, 7])
        np.testing.assert_array_equal(model.sample_index_, expected_positions)
        np.testing.assert_array_equal(
            model.estimation_index_,
            d.index.to_numpy()[expected_positions],
        )

    def test_backtick_name_containing_pipe_and_unsupported_fe_syntax(self):
        d = self.data.rename(columns={"firm_text": "firm|id"})
        model = xhdfe.feols(
            "y ~ x | `firm|id` + year",
            d,
            **REGRESSOR_OPTIONS,
        )
        self.assertEqual(model.fe_names_, ("firm|id", "year"))

        with self.assertRaisesRegex(NotImplementedError, "bare column names"):
            xhdfe.feols(
                "y ~ x | firm:year",
                self.data,
                **REGRESSOR_OPTIONS,
            )
        with self.assertRaisesRegex(NotImplementedError, "bare column names"):
            xhdfe.feols(
                "y ~ x | C(firm)",
                self.data,
                **REGRESSOR_OPTIONS,
            )
        with self.assertRaisesRegex(NotImplementedError, "at most one"):
            xhdfe.feols(
                "y ~ x | firm | year",
                self.data,
                **REGRESSOR_OPTIONS,
            )
        with self.assertRaisesRegex(NotImplementedError, "list the regression columns"):
            xhdfe.feols(
                "y ~ . | firm",
                self.data,
                **REGRESSOR_OPTIONS,
            )

    def test_invalid_formula_and_context_types_have_stable_errors(self):
        with self.assertRaisesRegex(TypeError, "non-empty string"):
            xhdfe.prepare_formula(["y ~ x"], self.data)
        with self.assertRaisesRegex(TypeError, "non-empty string"):
            xhdfe.prepare_formula("   ", self.data)
        with self.assertRaisesRegex(TypeError, "context must be a mapping"):
            xhdfe.prepare_formula("y ~ x", self.data, context=[])

    def test_formula_evaluation_does_not_mutate_dataframe(self):
        original = self.data.copy(deep=True)
        xhdfe.feols(
            "y ~ C(g)*x + I(z**2) | firm_text + year_text",
            self.data,
            **REGRESSOR_OPTIONS,
        )
        pd.testing.assert_frame_equal(self.data, original, check_exact=True)


class FormulaLazyImportTest(unittest.TestCase):
    def _run_fresh_python(self, code):
        env = os.environ.copy()
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_formula_exports_do_not_import_optional_stack(self):
        self._run_fresh_python(
            """
            import sys
            import xhdfe
            assert "xhdfe.py_hdfe_v11" not in sys.modules
            assert "py_hdfe_v11" not in sys.modules
            assert "formulaic" not in sys.modules
            assert "pandas" not in sys.modules
            assert "scipy" not in sys.modules
            from xhdfe import PreparedFormula, feols, prepare_formula
            assert callable(feols) and callable(prepare_formula)
            assert PreparedFormula.__name__ == "PreparedFormula"
            assert "xhdfe.py_hdfe_v11" not in sys.modules
            assert "py_hdfe_v11" not in sys.modules
            assert "formulaic" not in sys.modules
            assert "pandas" not in sys.modules
            assert "scipy" not in sys.modules
            """
        )

    def test_missing_extra_error_names_the_install_command(self):
        self._run_fresh_python(
            """
            import importlib.abc
            import sys

            class BlockFormulaic(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "formulaic" or fullname.startswith("formulaic."):
                        raise ModuleNotFoundError("blocked for test", name=fullname)
                    return None

            sys.meta_path.insert(0, BlockFormulaic())
            import xhdfe
            from xhdfe import feols
            try:
                feols("y ~ x", {"y": [1.0, 2.0], "x": [0.0, 1.0]})
            except ImportError as exc:
                assert ".[formula]" in str(exc)
                assert "formulaic>=1.2.1,<2" in str(exc)
                assert "pandas>=1.3" in str(exc)
            else:
                raise AssertionError("missing formula extra was not rejected")
            """
        )


if __name__ == "__main__":
    unittest.main()
