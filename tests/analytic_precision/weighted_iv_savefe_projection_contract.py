"""Retaining fixed effects preserves the ordinary weighted 2SLS projection."""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

from cuda_three_fe_covariance_contract import fixture


def run(module):
    for key in tuple(os.environ):
        if key.startswith("XHDFE_"):
            del os.environ[key]
    os.environ.update(XHDFE_GPU_BACKEND="cpu", XHDFE_MOBILITY_MODE="off",
                      XHDFE_ABSORPTION_CACHE_MODE="off", XHDFE_FE_STRUCTURE_MODE="off")
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module.resolve())
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    y, x, fes = fixture()
    weights = np.random.default_rng(20261111).uniform(0.5, 2.0, len(y))
    z = (x[:, 1] + 0.5 * np.random.default_rng(20261121).normal(size=len(y)))[:, None]
    common = dict(se_type="cluster", tol=1e-8, max_iter=100000, num_threads=16,
                  drop_singletons=True, absorption_method="auto", tolerance_mode="reghdfe-comparable")
    for columns, endogenous in (([0, 1, 2, 3], 1), ([1, 3, 0, 2], 0)):
        design = np.asfortranarray(x[:, columns])
        plain = core.HdfeRegressor(**common)
        saved = core.HdfeRegressor(**common, retain_fes=True)
        for fit in (plain, saved):
            fit.fit(y, design, fes=fes, clusters=[fes[0]], weights=weights,
                    instruments=z, endogenous_idx=[endogenous])
            assert fit.converged_ and fit.precision_certified_ and not fit.gpu_used_
            assert fit.nobs_ == len(y)
        np.testing.assert_array_equal(plain.sample_index_, saved.sample_index_)
        assert plain.df_a_ == saved.df_a_ and plain.df_resid_ == saved.df_resid_
        assert plain.absorption_method_used == saved.absorption_method_used == core.AbsorptionMethod.mlsmr
        assert saved.num_iterations_ > plain.num_iterations_
        for name in ("coef_", "covariance_", "stderr_", "residuals_"):
            np.testing.assert_array_equal(getattr(plain, name), getattr(saved, name), err_msg=name)
        effects = np.asarray(saved.fe_effects_)
        assert effects.shape == (3, len(y)) and np.isfinite(effects).all() and saved.fe_recovery_converged_
        gauge_limit = 128 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(effects))))
        assert np.max(np.abs(np.average(effects, axis=1, weights=weights))) <= gauge_limit
    print("weighted_iv_savefe_projection_contract PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=Path)
    run(parser.parse_args().module)
