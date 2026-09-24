#!/usr/bin/env python3
"""Exercise formula/native refit custody with explicit source and native module.

Loads the chosen formula source in memory as xhdfe._formula. No installation,
compilation, timing campaign, report file or production mutation is performed.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest

sys.dont_write_bytecode = True
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XHDFE_GPU_BACKEND"] = "cpu"
os.environ["XHDFE_ABSORPTION_CACHE_MODE"] = "off"

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OPTIONS = dict(num_threads=1, drop_singletons=False, tol=1e-10)


class FormulaRefitContract(unittest.TestCase):
    formula = None
    native = None

    def setUp(self):
        rng = np.random.default_rng(220922)
        self.x = rng.normal(size=192)
        self.z = rng.normal(size=192)
        self.y = 2 * self.x + 0.4 * self.z + rng.normal(size=192)
        self.X = np.asfortranarray(self.x[:, None])
        self.model = self.formula.feols("y ~ x", {"y": self.y, "x": self.x},
                                       **OPTIONS)

    def assert_conversion_failure_preserves_fit(self, call):
        model = self.model
        generation, state = model.generation_, model.lifecycle_state_
        beta, covariance = model.coef_.copy(), model.covariance_.copy()
        named_table = model.tidy().copy()
        metadata = {name: model.__dict__[name] for name in
                    self.formula._FORMULA_METADATA if name in model.__dict__}
        with self.assertRaises(TypeError):
            call()
        self.assertEqual(model.generation_, generation)
        self.assertEqual(model.lifecycle_state_, state)
        np.testing.assert_array_equal(model.coef_, beta)
        np.testing.assert_array_equal(model.covariance_, covariance)
        for name, value in metadata.items():
            self.assertIs(model.__dict__.get(name), value, name)
        self.assertEqual(tuple(model.tidy().index), ("x", "Intercept"))
        self.assertTrue(model.tidy().equals(named_table))

    def assert_failed_native_fit_clears_state(self, call):
        model = self.model
        generation = model.generation_
        with self.assertRaises(RuntimeError):
            call()
        self.assertGreater(model.generation_, generation)
        self.assertEqual(model.lifecycle_state_, "failed")
        self.assertFalse(model.converged_)
        for name in ("coef_", "covariance_", "residuals_"):
            self.assertEqual(np.asarray(getattr(model, name)).size, 0, name)
        for name in self.formula._FORMULA_METADATA:
            self.assertNotIn(name, model.__dict__)

    def test_none_response_preserves_fit_and_all_formula_metadata(self):
        self.assert_conversion_failure_preserves_fit(
            lambda: self.model.fit(None, self.X))

    def test_none_design_preserves_fit_and_all_formula_metadata(self):
        self.assert_conversion_failure_preserves_fit(
            lambda: self.model.fit(self.y, None))

    def test_invalid_keyword_preserves_fit_and_all_formula_metadata(self):
        self.assert_conversion_failure_preserves_fit(
            lambda: self.model.fit(self.y, self.X, invalid_keyword=True))

    def test_nan_response_clears_native_state_and_formula_metadata(self):
        bad_y = self.y.copy()
        bad_y[0] = np.nan
        self.assert_failed_native_fit_clears_state(
            lambda: self.model.fit(bad_y, self.X))

    def test_wrong_design_shape_clears_native_state_and_formula_metadata(self):
        self.assert_failed_native_fit_clears_state(
            lambda: self.model.fit(self.y, self.X[:-1]))

    def test_successful_raw_refit_has_generic_names_and_native_parity(self):
        model = self.model
        generation = model.generation_
        design = np.asfortranarray(self.z[:, None])
        expected = self.native.HdfeRegressor(**OPTIONS)
        native_return = expected.fit(self.y, design)
        actual_return = model.fit(self.y, design)
        self.assertEqual(type(actual_return), type(native_return))
        self.assertGreater(model.generation_, generation)
        self.assertEqual(model.lifecycle_state_, "standard_ready")
        for name in self.formula._FORMULA_METADATA:
            self.assertNotIn(name, model.__dict__)
        np.testing.assert_array_equal(model.coef_, expected.coef_)
        np.testing.assert_array_equal(model.covariance_, expected.covariance_)
        self.assertEqual(tuple(model.tidy().index), ("b0", "b1"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formula", required=True, type=Path)
    parser.add_argument("--module", required=True, type=Path)
    args = parser.parse_args()
    formula_path, module_path = args.formula.resolve(strict=True), args.module.resolve(strict=True)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    source = formula_path.read_bytes()
    formula_sha = hashlib.sha256(source).hexdigest()
    module_sha = digest(module_path)
    sys.path.insert(0, str(ROOT))
    import xhdfe
    spec = importlib.util.spec_from_file_location("xhdfe.py_hdfe_v11", module_path)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    formula = types.ModuleType("xhdfe._formula")
    formula.__file__ = str(formula_path)
    formula.__package__ = "xhdfe"
    sys.modules[formula.__name__] = formula
    exec(compile(source, str(formula_path), "exec"), formula.__dict__)
    FormulaRefitContract.formula = formula
    FormulaRefitContract.native = native
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FormulaRefitContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    stable = digest(formula_path) == formula_sha and digest(module_path) == module_sha
    report = dict(formula=str(formula_path), formula_sha256=formula_sha,
                  module=str(module_path), module_sha256=module_sha,
                  inputs_unchanged=stable, tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  status="PASS" if result.wasSuccessful() and stable else "FAIL",
                  performance_claim=False)
    print(json.dumps(report))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
