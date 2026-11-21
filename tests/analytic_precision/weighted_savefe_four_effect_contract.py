"""Regression test for E1's selection of the ordinary weighted CPU projection."""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

from cuda_three_fe_covariance_contract import fixture


def run(module):
    # Freeze the existing defaults and prevent unrelated cache/profile reuse.
    for key in tuple(os.environ):
        if key.startswith("XHDFE_"):
            del os.environ[key]
    os.environ.update(XHDFE_GPU_BACKEND="cpu", XHDFE_MOBILITY_MODE="off",
                      XHDFE_ABSORPTION_CACHE_MODE="off", XHDFE_FE_STRUCTURE_MODE="off")
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module.resolve())
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)

    y, x, fes = fixture()
    worker, year = fes[0], fes[2]
    rng = np.random.default_rng(20261117)
    workers = int(worker.max()) + 1
    base = rng.integers(0, 25, workers)[worker]
    threshold = rng.integers(5, 35, workers)[worker]
    fes = [*fes, ((base + (year >= threshold)) % 25).astype(np.int32)]
    weights = np.random.default_rng(20261111).uniform(0.5, 2.0, len(y))
    common = dict(se_type="cluster", tol=1e-8, max_iter=100000, num_threads=16,
                  drop_singletons=True, absorption_method="auto", tolerance_mode="reghdfe-comparable")
    plain = core.HdfeRegressor(**common)
    saved = core.HdfeRegressor(**common, retain_fes=True)
    for fit in (plain, saved):
        fit.fit(y, np.asfortranarray(x), fes=fes, clusters=[worker], weights=weights)
        assert fit.converged_ and fit.precision_certified_ and not fit.gpu_used_
        assert fit.nobs_ == len(y)
    assert plain.df_a_ == saved.df_a_ and plain.df_resid_ == saved.df_resid_
    np.testing.assert_array_equal(plain.sample_index_, saved.sample_index_)
    assert plain.absorption_method_used == saved.absorption_method_used == core.AbsorptionMethod.mlsmr
    assert saved.num_iterations_ > plain.num_iterations_
    for prop in ("coef_", "covariance_", "stderr_", "residuals_"):
        np.testing.assert_array_equal(getattr(plain, prop), getattr(saved, prop), err_msg=prop)
    effects = np.asarray(saved.fe_effects_)
    assert effects.shape == (4, len(y)) and np.isfinite(effects).all() and saved.fe_recovery_converged_
    mean_scale = 128 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(effects))))
    assert np.max(np.abs(np.average(effects, axis=1, weights=weights))) <= mean_scale
    print("weighted_savefe_four_effect_contract PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=Path)
    run(parser.parse_args().module)
