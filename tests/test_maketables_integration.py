from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import xhdfe

# Full-suite discovery can register the same pybind11 module first under its
# historical top-level name. Reuse that object under the package name.
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


@unittest.skipUnless(HAS_FORMULA_DEPENDENCIES, "formula dependencies are not installed")
class MaketablesHookTest(unittest.TestCase):
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
        self.assertEqual(list(table.columns), ["b", "se", "t", "p"])
        self.assertEqual(table.index.name, "Coefficient")
        self.assertEqual(list(table.index), list(self.model.coef_names_))

    def test_coefficient_table_agrees_with_tidy(self):
        table = self.model.__maketables_coef_table__
        tidy = self.model.tidy()
        for actual, expected in (
            ("b", "Estimate"),
            ("se", "Std. Error"),
            ("t", "t value"),
            ("p", "Pr(>|t|)"),
        ):
            np.testing.assert_array_equal(
                table[actual].to_numpy(), tidy[expected].to_numpy()
            )

    def test_nondefault_confidence_level_is_not_mislabeled_as_95_percent(self):
        model = xhdfe.feols(
            "y ~ x1 | firm",
            data=self.data,
            se_type="robust",
            level=90,
            **REGRESSOR_OPTIONS,
        )
        self.assertNotIn("ci95l", model.__maketables_coef_table__.columns)
        self.assertNotIn("ci95u", model.__maketables_coef_table__.columns)

    def test_depvar_and_fixef_string(self):
        self.assertEqual(self.model.__maketables_depvar__, "y")
        self.assertEqual(self.model.__maketables_fixef_string__, "firm+year")

    def test_quoted_depvar_may_contain_formula_separator(self):
        data = self.data.rename(columns={"y": "out~come"})
        model = xhdfe.feols(
            "`out~come` ~ x1 | firm",
            data=data,
            se_type="robust",
            **REGRESSOR_OPTIONS,
        )
        self.assertEqual(model.__maketables_depvar__, "out~come")

    def test_canonical_statistics(self):
        stat = self.model.__maketables_stat__
        self.assertEqual(stat("N"), int(self.model.nobs_))
        self.assertEqual(stat("r2"), self.model.r2_)
        self.assertEqual(stat("r2_within"), self.model.r2_within_)
        self.assertEqual(stat("n_clusters"), int(self.model.num_clusters_))
        self.assertEqual(stat("se_type"), "by: firm")
        self.assertIsNone(stat("not_a_statistic"))

    def test_rmse_is_root_mean_squared_residual(self):
        expected = float(np.sqrt(self.model.rss_ / self.model.nobs_))
        self.assertAlmostEqual(self.model.__maketables_stat__("rmse"), expected, places=12)

    def test_frequency_weight_rmse_matches_literal_replication(self):
        data = self.data.iloc[:300].copy()
        data["fw"] = np.resize(np.array([1, 2, 3], dtype=np.int64), len(data))
        weighted = xhdfe.feols(
            "y ~ x1 + x2",
            data=data,
            weights="fw",
            fweights=True,
            se_type="robust",
            **REGRESSOR_OPTIONS,
        )
        repeated = data.loc[data.index.repeat(data.fw)].reset_index(drop=True)
        expanded = xhdfe.feols(
            "y ~ x1 + x2",
            data=repeated,
            se_type="robust",
            **REGRESSOR_OPTIONS,
        )
        self.assertEqual(weighted.__maketables_stat__("N"), len(repeated))
        self.assertAlmostEqual(
            weighted.__maketables_stat__("rmse"),
            expanded.__maketables_stat__("rmse"),
            places=12,
        )

    def test_within_r2_is_withheld_without_fixed_effects(self):
        model = xhdfe.feols(
            "y ~ x1 + x2", data=self.data, se_type="robust", **REGRESSOR_OPTIONS
        )
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

    def test_standard_error_aliases_are_canonical(self):
        aliases = {
            "unadjusted": "homoskedastic",
            "homoskedastic": "homoskedastic",
            "classical": "homoskedastic",
            "unadj": "homoskedastic",
            "ols": "homoskedastic",
            "robust": "robust",
            "hc1": "robust",
            "heteroskedastic": "robust",
        }
        for spelling, canonical in aliases.items():
            with self.subTest(spelling=spelling):
                model = xhdfe.feols(
                    "y ~ x1 | firm",
                    data=self.data,
                    se_type=spelling,
                    **REGRESSOR_OPTIONS,
                )
                self.assertEqual(model.se_type_, canonical)
                expected = "iid" if canonical == "homoskedastic" else "robust"
                self.assertEqual(model.__maketables_stat__("se_type"), expected)

    def test_default_standard_errors_are_homoskedastic(self):
        model = xhdfe.feols("y ~ x1 | firm", data=self.data, **REGRESSOR_OPTIONS)
        self.assertEqual(model.se_type_, "homoskedastic")
        self.assertEqual(model.__maketables_stat__("se_type"), "iid")

    def test_multiway_clustering_names_every_dimension(self):
        model = xhdfe.feols(
            "y ~ x1 | firm",
            data=self.data,
            se_type="clustered",
            clusters=["firm", "year"],
            **REGRESSOR_OPTIONS,
        )
        self.assertEqual(model.cluster_names_, ("firm", "year"))
        self.assertEqual(model.se_type_, "cluster")
        self.assertEqual(model.__maketables_stat__("se_type"), "by: firm+year")
        self.assertEqual(
            model.__maketables_vcov_info__,
            {"se_type": "by: firm+year", "cluster_var": "firm+year"},
        )

    def test_singleton_statistics(self):
        data = _panel(singletons=True)
        model = xhdfe.feols(
            "y ~ x1 | firm + year",
            data=data,
            se_type="robust",
            drop_singletons=True,
            **REGRESSOR_OPTIONS,
        )
        stat = model.__maketables_stat__
        self.assertEqual(stat("N_full"), int(model.nobs_full_))
        self.assertEqual(stat("n_singletons"), int(model.num_singletons_))
        self.assertEqual(stat("df_absorbed"), int(model.df_a_))
        self.assertGreater(stat("n_singletons"), 0)
        self.assertEqual(stat("N"), stat("N_full") - stat("n_singletons"))

    def test_variable_labels_are_snapshotted(self):
        data = _panel()
        labels = {"y": "Log output", "x1": "Schooling"}
        data.attrs["variable_labels"] = labels
        model = xhdfe.feols(
            "y ~ x1 | firm", data=data, se_type="robust", **REGRESSOR_OPTIONS
        )
        labels["x1"] = "Changed later"
        data.attrs["variable_labels"]["y"] = "Changed later"
        self.assertEqual(
            model.__maketables_var_labels__,
            {"y": "Log output", "x1": "Schooling"},
        )

    def test_invalid_or_missing_variable_labels_are_ignored(self):
        self.assertIsNone(self.model.__maketables_var_labels__)
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

    def test_prepared_formula_carries_metadata(self):
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

    def test_bare_refit_clears_named_metadata_and_degrades_explicitly(self):
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
        self.assertEqual(model.__maketables_depvar__, "y")
        self.assertEqual(list(model.__maketables_coef_table__.index), ["b0", "b1"])
        self.assertEqual(model.__maketables_fixef_string__, "fe1")
        self.assertIsNone(model.__maketables_stat__("se_type"))


class NativeRegressorHookTest(unittest.TestCase):
    def _fit(self, y, X, fes, **options):
        constructor = dict(REGRESSOR_OPTIONS, se_type="robust")
        constructor.update(options)
        model = xhdfe.HdfeRegressor(**constructor)
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
        self.year = rng.integers(0, 6, size=self.n)
        self.y = self.X @ np.array([1.1, -0.4]) + 0.02 * self.firm + rng.normal(size=self.n)

    @unittest.skipUnless(HAS_FORMULA_DEPENDENCIES, "pandas is not installed")
    def test_hooks_use_positional_and_generic_fe_names(self):
        model = self._fit(self.y, self.X, [self.firm, self.year])
        self.assertEqual(list(model.__maketables_coef_table__.index), ["b0", "b1", "b2"])
        self.assertEqual(model.__maketables_depvar__, "y")
        self.assertEqual(model.__maketables_fixef_string__, "fe1+fe2")

    @unittest.skipUnless(HAS_FORMULA_DEPENDENCIES, "pandas is not installed")
    def test_within_r2_tracks_actual_absorption(self):
        absorbed = self._fit(self.y, self.X, [self.firm])
        self.assertIsNotNone(absorbed.__maketables_stat__("r2_within"))
        plain = self._fit(self.y, self.X, None)
        self.assertIsNone(plain.__maketables_stat__("r2_within"))
        self.assertIsNone(plain.__maketables_fixef_string__)

    def test_touching_hooks_does_not_change_results(self):
        model = self._fit(self.y, self.X, [self.firm])
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


class AttachmentContractTest(unittest.TestCase):
    def test_failed_attach_rolls_back_partial_hooks(self):
        class RejectingMeta(type):
            def __setattr__(cls, name, value):
                if name == "__maketables_depvar__":
                    raise TypeError("simulated immutable extension type")
                super().__setattr__(name, value)

        class Candidate(metaclass=RejectingMeta):
            pass

        with self.assertRaises(TypeError):
            _maketables.attach(Candidate)
        self.assertNotIn("__maketables_coef_table__", Candidate.__dict__)
        self.assertNotIn(_maketables._ATTACHED_FLAG, Candidate.__dict__)

    def test_loading_native_core_does_not_import_pandas(self):
        code = (
            "import sys, xhdfe; "
            "assert 'pandas' not in sys.modules; "
            "assert xhdfe.HdfeRegressor is not None; "
            "assert 'pandas' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


@unittest.skipUnless(HAS_MAKETABLES, "maketables is not installed")
@unittest.skipUnless(HAS_FORMULA_DEPENDENCIES, "formula dependencies are not installed")
class MaketablesRenderTest(unittest.TestCase):
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

    def test_extractor_resolves_through_plugin_format(self):
        extractor = mt.get_extractor(self.model)
        self.assertEqual(type(extractor).__name__, "PluginExtractor")

    def test_etable_renders_coefficients_fixed_effects_and_stats(self):
        latex = str(mt.ETable([self.model], drop="Intercept").make(type="tex"))
        self.assertIn("x1", latex)
        self.assertIn("firm", latex)
        self.assertIn("year", latex)
        self.assertIn("Observations", latex)

    def test_etable_accepts_xhdfe_specific_statistics(self):
        latex = str(
            mt.ETable(
                [self.model],
                drop="Intercept",
                model_stats=["N", "n_singletons", "df_absorbed"],
            ).make(type="tex")
        )
        self.assertIn("Singletons dropped", latex)
        self.assertIn("Absorbed d.f.", latex)

    def test_variable_labels_reach_rendered_table(self):
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

    def test_native_array_model_renders_and_discloses_absorption(self):
        model = xhdfe.HdfeRegressor(se_type="robust", **REGRESSOR_OPTIONS)
        model.fit(
            np.ascontiguousarray(self.data.y.to_numpy(), dtype=np.float64),
            np.asfortranarray(self.data[["x1", "x2"]].to_numpy(), dtype=np.float64),
            fes=[np.ascontiguousarray(self.data.firm.to_numpy(), dtype=np.int64)],
        )
        latex = str(mt.ETable([model]).make(type="tex"))
        self.assertIn("b0", latex)
        self.assertIn("fe1", latex)


if __name__ == "__main__":
    unittest.main()
