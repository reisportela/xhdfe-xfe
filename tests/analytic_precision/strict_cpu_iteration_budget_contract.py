"""The Strict CPU repair chain honours the caller's iteration budget."""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

from strict_cuda_acceptance_contract import fixture


def run(module, cache):
    for key in tuple(os.environ):
        if key.startswith("XHDFE_"):
            del os.environ[key]
    os.environ.update(XHDFE_GPU_BACKEND="cpu", XHDFE_MOBILITY_MODE="off",
                      XHDFE_ABSORPTION_CACHE_MODE="off", XHDFE_FE_STRUCTURE_MODE="off")
    load = importlib.util.spec_from_file_location("py_hdfe_v11", module.resolve())
    core = importlib.util.module_from_spec(load)
    load.loader.exec_module(core)
    model = core.HdfeRegressor(absorption_method="auto", tolerance_mode="strict-residual",
                              tol=1e-8, max_iter=300, num_threads=4, drop_singletons=True)
    good = fixture("W0")
    bad = fixture("T3")

    def success():
        model.fit(good["y"], good["x"], fes=good["fes"])
        assert model.converged_ and model.precision_certified_ and not model.gpu_used_
        assert 0 < model.num_iterations_ <= 300 and np.isfinite(model.coef_).all()
        return np.asarray(model.coef_).copy(), np.asarray(model.covariance_).copy()

    expected = success()
    if cache is not None:
        assert cache.is_file()
        os.environ.update(XHDFE_ABSORPTION_CACHE_MODE="read", XHDFE_ABSORPTION_CACHE=str(cache.resolve()))
    try:
        model.fit(bad["y"], bad["x"], fes=bad["fes"])
    except RuntimeError as exc:
        assert "iterations=300/300" in str(exc), str(exc)
        assert np.asarray(model.coef_).size == 0 and not model.converged_
    else:
        raise AssertionError(f"T3 unexpectedly returned estimates after {model.num_iterations_} iterations")
    for before, after in zip(expected, success()):
        np.testing.assert_array_equal(before, after)
    print("strict_cpu_iteration_budget_contract PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=Path)
    parser.add_argument("--old-cache", type=Path)
    args = parser.parse_args()
    run(args.module, args.old_cache)
