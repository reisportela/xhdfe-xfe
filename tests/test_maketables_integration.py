from __future__ import annotations

from pathlib import Path
import sys
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

from xhdfe import _maketables

try:
    import formulaic  # noqa: F401
    import pandas as pd
except ImportError:
    pd = None

try:
    import maketables as mt
except ImportError:
    mt = None


HAS_FORMULA_DEPENDENCIES = pd is not None
HAS_MAKETABLES = mt is not None

REGRESSOR_OPTIONS = {"num_threads": 1, "tol": 1e-10, "max_iter": 20_000}


def _panel(seed=17, n=2000, singletons=False):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 40, size=n)
    if singletons:
        firm[:15] = np.arange(500, 515)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "firm": firm,
            "year": rng.integers(0, 8, size=n),
        }
    )
    frame["y"] = (
        1.3 * frame.x1 - 0.6 * frame.x2 + 0.02 * frame.firm + rng.normal(size=n)
    )
    return frame


@unittest.skipUnless(HAS_FORMULA_DEPENDENCIES, "formulaic extra is not installed")
class MaketablesHookTest(unittest.TestCase):
    """The plug-in hooks themselves, which need no maketables install."""

    @classmethod
    def setUpClass(cls):
        cls.data = _panel()
        cls.model = xhdfe.feols(
            "y ~ x1 + x2 | firm + year",
            data=cls.data,
            se_type="cluster",
            clusters="firm",
            **REGRESSOR_OPTIONS,
        )

    def test_coefficient_table_uses_canonical_columns(self):
        table = self.model.__maketables_coef_table__
        for column in ("b", "se", "t", "p", "ci95l", "ci95u"):
            self.assertIn(column, table.columns)
        self.assertEqual(table.index.name, "Coefficient")
        self.assertEqual(list(table.index), list(self.model.coef_names_))

    def test_coefficient_table_agrees_with_tidy(self):
        table = self.model.__maketables_coef_table__
        tidy = self.model.tidy()
        np.testing.assert_array_equal(table["b"].to_numpy(), tidy["Estimate"].to_numpy())
        np.testing.assert_array_equal(
            table["se"].to_numpy(), tidy["Std. Error"].to_numpy()
        )
        np.testing.assert_array_equal(table["t"].to_numpy(), tidy["t value"].to_numpy())
        np.testing.assert_array_equal(
            table["p"].to_numpy(), tidy["Pr(>|t|)"].to_numpy()
        )
        np.testing.assert_array_equal(
            table["ci95l"].to_numpy(), tidy["CI Low"].to_numpy()
        )

    def test_depvar_and_fixef_string(self):
        self.assertEqual(self.model.__maketables_depvar__, "y")
        self.assertEqual(self.model.__maketables_fixef_string__, "firm+year")

    def test_canonical_statistics(self):
        stat = self.model.__maketables_stat__
        self.assertEqual(stat("N"), int(self.model.nobs_))
        self.assertEqual(stat("r2"), self.model.r2_)
        self.assertEqual(stat("r2_within"), self.model.r2_within_)
        self.assertEqual(stat("n_clusters"), int(self.model.num_clusters_))
        self.assertEqual(stat("se_type"), "by: firm")
        self.assertIsNone(stat("not_a_statistic"))

    def test_rmse_is_root_mean_squared_residual(self):
        # sqrt(RSS/N), the convention pyfixest reports. df_resid_ is the
        # inference denominator (G-1 when clustering), not a residual count.
        expected = float(np.sqrt(self.model.rss_ / self.model.nobs_))
        self.assertAlmostEqual(self.model.__maketables_stat__("rmse"), expected, places=12)

    def test_within_r2_is_withheld_without_fixed_effects(self):
        model = xhdfe.feols(
            "y ~ x1 + x2", data=self.data, se_type="robust", **REGRESSOR_OPTIONS
        )
        # The core reports the within R2 equal to the overall R2 when nothing is
        # absorbed; the hook must not put that in a table as if it meant something.
        self.assertEqual(model.r2_within_, model.r2_)
        self.assertIsNone(model.__maketables_stat__("r2_within"))
        self.assertIsNone(model.__maketables_fixef_string__)

    def test_unclustered_fit_reports_no_cluster_count(self):
        model = xhdfe.feols(
            "y ~ x1 | firm", data=self.data, se_type="robust", **REGRESSOR_OPTIONS
        )
        self.assertEqual(model.num_clusters_, 0)
        self.assertIsNone(model.__maketables_stat__("n_clusters"))
        self.assertEqual(model.__maketables_stat__("se_type"), "robust")

    def test_se_type_spellings_collapse(self):
        for spelling in ("unadjusted", "homoskedastic"):
            model = xhdfe.feols(
                "y ~ x1 | firm",
                data=self.data,
                se_type=spelling,
                **REGRESSOR_OPTIONS,
            )
            self.assertEqual(model.se_type_, "homoskedastic")
            self.assertEqual(model.__maketables_stat__("se_type"), "iid")
        default = xhdfe.feols("y ~ x1 | firm", data=self.data, **REGRESSOR_OPTIONS)
        self.assertEqual(default.se_type_, "homoskedastic")

    def test_multiway_clustering_names_every_dimension(self):
        model = xhdfe.feols(
            "y ~ x1 | firm",
            data=self.data,
            se_type="cluster",
            clusters=["firm", "year"],
            **REGRESSOR_OPTIONS,
        )
        self.assertEqual(model.cluster_names_, ("firm", "year"))
        self.assertEqual(model.__maketables_stat__("se_type"), "by: firm+year")
        self.assertEqual(
            model.__maketables_vcov_info__["cluster_var"], "firm+year"
        )

    def test_singleton_statistics(self):
        data = _panel(singletons=True)
        model = xhdfe.feols(
            "y ~ x1 | firm + year",
            data=data,
            se_type="robust",
            num_threads=1,
            tol=1e-10,
            max_iter=20_000,
            drop_singletons=True,
        )
        stat = model.__maketables_stat__
        self.assertEqual(stat("N_full"), int(model.nobs_full_))
        self.assertEqual(stat("n_singletons"), int(model.num_singletons_))
        self.assertEqual(stat("df_absorbed"), int(model.df_a_))
        self.assertGreater(stat("n_singletons"), 0)
        self.assertEqual(stat("N"), stat("N_full") - stat("n_singletons"))

    def test_variable_labels_travel_from_the_frame(self):
        data = _panel()
        data.attrs["variable_labels"] = {"y": "Log output", "x1": "Schooling"}
        model = xhdfe.feols(
            "y ~ x1 | firm", data=data, se_type="robust", **REGRESSOR_OPTIONS
        )
        self.assertEqual(
            model.__maketables_var_labels__, {"y": "Log output", "x1": "Schooling"}
        )

    def test_unlabelled_frame_yields_no_labels(self):
        self.assertIsNone(self.model.__maketables_var_labels__)

    def test_non_mapping_labels_are_ignored(self):
        data = _panel()
        data.attrs["variable_labels"] = ["not", "a", "mapping"]
        model = xhdfe.feols(
            "y ~ x1 | firm", data=data, se_type="robust", **REGRESSOR_OPTIONS
        )
        self.assertIsNone(model.__maketables_var_labels__)

    def test_mapping_data_without_attrs_still_fits(self):
        data = _panel()
        payload = {name: data[name].to_numpy() for name in ("y", "x1", "firm")}
        model = xhdfe.feols(
            "y ~ x1 | firm", data=payload, se_type="robust", **REGRESSOR_OPTIONS
        )
        self.assertIsNone(model.__maketables_var_labels__)
        self.assertEqual(model.__maketables_stat__("N"), int(model.nobs_))

    def test_default_and_supported_statistic_keys(self):
        self.assertEqual(
            self.model.__maketables_default_stat_keys__, ["N", "r2", "r2_within"]
        )
        supported = _maketables.supported_stats(self.model)
        self.assertLessEqual({"N", "r2", "r2_within", "rmse", "se_type"}, supported)

    def test_prepared_formula_carries_the_metadata(self):
        prepared = xhdfe.prepare_formula(
            "y ~ x1 | firm",
            data=self.data,
            se_type="cluster",
            clusters="firm",
            **REGRESSOR_OPTIONS,
        )
        model = prepared.fit()
        self.assertEqual(model.se_type_, "cluster")
        self.assertEqual(model.cluster_names_, ("firm",))
        self.assertEqual(model.__maketables_stat__("se_type"), "by: firm")

    def test_bare_refit_drops_stale_metadata_and_degrades(self):
        model = xhdfe.feols(
            "y ~ x1 | firm", data=self.data, se_type="robust", **REGRESSOR_OPTIONS
        )
        model.fit(
            np.ascontiguousarray(self.data.y.to_numpy(), dtype=np.float64),
            np.asfortranarray(self.data[["x1"]].to_numpy(), dtype=np.float64),
            fes=[np.ascontiguousarray(self.data.firm.to_numpy(), dtype=np.int64)],
        )
        self.assertFalse(hasattr(model, "formula_"))
        self.assertFalse(hasattr(model, "cluster_names_"))
        self.assertFalse(hasattr(model, "se_type_"))
        # The hooks stay usable, falling back to positional names.
        self.assertEqual(model.__maketables_depvar__, "y")
        self.assertEqual(
            list(model.__maketables_coef_table__.index), ["b0", "b1"]
        )
        self.assertIsNone(model.__maketables_stat__("se_type"))


class NativeRegressorHookTest(unittest.TestCase):
    """The array API carries no names, but must still tabulate."""

    def _fit(self, y, X, fes):
        model = xhdfe.HdfeRegressor(se_type="robust", **REGRESSOR_OPTIONS)
        model.fit(
            np.ascontiguousarray(y, dtype=np.float64),
            np.asfortranarray(X, dtype=np.float64),
            fes=(
                [np.ascontiguousarray(fe, dtype=np.int64) for fe in fes]
                if fes
                else None
            ),
        )
        return model

    def setUp(self):
        rng = np.random.default_rng(23)
        self.n = 1500
        self.X = rng.normal(size=(self.n, 2))
        self.firm = rng.integers(0, 30, size=self.n)
        self.y = self.X @ np.array([1.1, -0.4]) + 0.02 * self.firm + rng.normal(size=self.n)

    @unittest.skipUnless(HAS_FORMULA_DEPENDENCIES, "pandas is not installed")
    def test_hooks_are_present_with_positional_names(self):
        model = self._fit(self.y, self.X, [self.firm])
        self.assertTrue(hasattr(model, "__maketables_coef_table__"))
        table = model.__maketables_coef_table__
        self.assertEqual(list(table.index), ["b0", "b1", "b2"])
        self.assertEqual(model.__maketables_depvar__, "y")
        self.assertIsNone(model.__maketables_fixef_string__)

    @unittest.skipUnless(HAS_FORMULA_DEPENDENCIES, "pandas is not installed")
    def test_within_r2_tracks_whether_anything_was_absorbed(self):
        absorbed = self._fit(self.y, self.X, [self.firm])
        self.assertIsNotNone(absorbed.__maketables_stat__("r2_within"))
        plain = self._fit(self.y, self.X, None)
        self.assertEqual(len(plain.fe_num_levels_), 0)
        self.assertIsNone(plain.__maketables_stat__("r2_within"))

    def test_attaching_hooks_does_not_disturb_estimation(self):
        model = self._fit(self.y, self.X, [self.firm])
        # The hooks are read-only descriptors; touching them must not perturb
        # any stored result.
        before = (
            np.array(model.coef_, copy=True),
            np.array(model.stderr_, copy=True),
            float(model.r2_),
        )
        _maketables.supported_stats(model)
        np.testing.assert_array_equal(model.coef_, before[0])
        np.testing.assert_array_equal(model.stderr_, before[1])
        self.assertEqual(float(model.r2_), before[2])

    def test_attach_is_idempotent(self):
        cls = type(self._fit(self.y, self.X, [self.firm]))
        _maketables.attach(cls)
        _maketables.attach(cls)
        self.assertTrue(hasattr(cls, "__maketables_coef_table__"))


@unittest.skipUnless(HAS_MAKETABLES, "maketables is not installed")
@unittest.skipUnless(HAS_FORMULA_DEPENDENCIES, "formulaic extra is not installed")
class MaketablesRenderTest(unittest.TestCase):
    """End-to-end: maketables must discover and render an xhdfe result."""

    @classmethod
    def setUpClass(cls):
        cls.data = _panel()
        cls.model = xhdfe.feols(
            "y ~ x1 + x2 | firm + year",
            data=cls.data,
            se_type="cluster",
            clusters="firm",
            **REGRESSOR_OPTIONS,
        )

    def test_extractor_resolves_through_the_plugin_format(self):
        extractor = mt.get_extractor(self.model)
        self.assertEqual(type(extractor).__name__, "PluginExtractor")

    def test_etable_renders_latex_with_coefficients_and_fixed_effects(self):
        latex = str(mt.ETable([self.model], drop="Intercept").make(type="tex"))
        self.assertIn("x1", latex)
        self.assertIn("firm", latex)
        self.assertIn("year", latex)
        self.assertIn("Observations", latex)

    def test_etable_accepts_the_xhdfe_specific_statistics(self):
        latex = str(
            mt.ETable(
                [self.model],
                drop="Intercept",
                model_stats=["N", "n_singletons", "df_absorbed"],
            ).make(type="tex")
        )
        self.assertIn("Singletons dropped", latex)
        self.assertIn("Absorbed d.f.", latex)

    def test_variable_labels_reach_the_rendered_table(self):
        data = _panel()
        data.attrs["variable_labels"] = {"y": "Log output", "x1": "Schooling"}
        model = xhdfe.feols(
            "y ~ x1 | firm", data=data, se_type="robust", **REGRESSOR_OPTIONS
        )
        latex = str(mt.ETable([model], drop="Intercept").make(type="tex"))
        self.assertIn("Schooling", latex)
        self.assertIn("Log output", latex)

    def test_two_models_render_side_by_side(self):
        second = xhdfe.feols(
            "y ~ x1 | firm",
            data=self.data,
            se_type="cluster",
            clusters="firm",
            **REGRESSOR_OPTIONS,
        )
        latex = str(mt.ETable([second, self.model], drop="Intercept").make(type="tex"))
        self.assertIn("(1)", latex)
        self.assertIn("(2)", latex)


if __name__ == "__main__":
    unittest.main()
