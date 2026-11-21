"""Independent checks for the corrected CPU Gauss–Seidel and Strict projections."""

import argparse
import importlib.util
import os
from pathlib import Path
import struct
import sys

import numpy as np

from strict_cuda_acceptance_contract import fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda_weighted_cpu_route_contract import cov240k, grade, weight_vector, weighted_oracle


def cached_projection(path, n, p):
    data = path.read_bytes()
    assert data[:32].rstrip(b"\0") == b"xhdfe_absorption_cache_v5"
    assert struct.unpack_from("<3i", data, 64) == (n, p, p)
    offset = 64 + struct.calcsize("<6i4d3?3idi2d")
    for _ in range(2):
        size, = struct.unpack_from("<i", data, offset)
        offset += 4 + 4 * size
    size, = struct.unpack_from("<q", data, offset)
    assert size == n
    y = np.frombuffer(data, "<f8", n, offset + 8)
    offset += 8 + 8 * n
    size, = struct.unpack_from("<Q", data, offset)
    assert size == n * p and len(data) == offset + 8 + 8 * size + 32
    x = np.frombuffer(data, "<f8", size, offset + 8).reshape((n, p), order="F")
    return np.column_stack((y, x))


def max_group_mean(z, fes, weights):
    weights = np.asarray(weights, np.longdouble)
    maximum = np.longdouble(0)
    for fe in fes:
        mass = np.zeros(int(fe.max()) + 1, np.longdouble)
        moments = np.zeros((mass.size, z.shape[1]), np.longdouble)
        np.add.at(mass, fe, weights)
        np.add.at(moments, fe, weights[:, None] * z.astype(np.longdouble))
        keep = mass > 0
        maximum = max(maximum, np.max(np.abs(moments[keep] / mass[keep, None])))
    return maximum


def strict(core, output):
    for method in ("lsmr", "mlsmr"):
        for name, weighted in (("W0", False), ("T3", False), ("W0_2p26", False),
                               ("T3_2p26", False), ("W0_x1p24", True)):
            data = fixture(name)
            y, x, fes = (data[key] for key in ("y", "x", "fes"))
            w = weight_vector("aw", len(y)) if weighted else None
            cache = output / f"{method}_{name}.bin"
            os.environ.update(XHDFE_ABSORPTION_CACHE=str(cache),
                              XHDFE_ABSORPTION_CACHE_MODE="write")
            opts = dict(absorption_method=method, tolerance_mode="strict-residual",
                        tol=1e-8, max_iter=400, num_threads=4, drop_singletons=True)
            model = core.HdfeRegressor(**opts)
            try:
                model.fit(y, x, fes=fes, weights=w)
            except RuntimeError as exc:
                assert name != "W0" and (name != "T3" or method == "lsmr"), str(exc)
                assert "strict-residual max absolute group mean=" in str(exc), str(exc)
                assert np.asarray(model.coef_).size == 0 and not cache.exists()
                print(method, name, "safe refusal", str(exc))
                continue
            assert model.converged_ and model.precision_certified_ and not model.gpu_used_
            assert 0 < model.num_iterations_ <= opts["max_iter"]
            reading = max_group_mean(cached_projection(cache, len(y), x.shape[1]), fes,
                                     np.ones(len(y)) if w is None else w)
            assert reading <= opts["tol"], (method, name, reading, model.num_iterations_)
            os.environ["XHDFE_ABSORPTION_CACHE_MODE"] = "read"
            again = core.HdfeRegressor(**opts)
            again.fit(y, x, fes=fes, weights=w)
            np.testing.assert_array_equal(again.coef_, model.coef_)
            np.testing.assert_array_equal(again.covariance_, model.covariance_)
            assert again.num_iterations_ == model.num_iterations_
            print(method, name, "pass", model.num_iterations_, float(reading))


def gauss_seidel(core, output):
    y, x, fes = cov240k(3)
    w = weight_vector("aw", len(y))
    oracle = weighted_oracle(y, x, fes, w, False)
    cache = output / "gauss_seidel.bin"
    os.environ.update(XHDFE_ABSORPTION_CACHE=str(cache), XHDFE_ABSORPTION_CACHE_MODE="write")
    opts = dict(absorption_method="gauss-seidel", tol=1e-8,
                max_iter=100000, num_threads=8, drop_singletons=True, se_type="cluster")
    model = core.HdfeRegressor(**opts)
    model.fit(y, x, fes=fes, weights=w, clusters=[fes[0]])
    assert model.converged_ and model.precision_certified_ and not model.gpu_used_
    assert str(model.absorption_method_used).split(".")[-1] == "gauss_seidel"
    fit = dict(b=model.coef_, V=model.covariance_)
    error = grade(oracle, fit)
    reference = np.asarray(oracle["V"])
    sd = np.sqrt(np.diag(reference))
    covariance = float(np.max(np.abs(fit["V"] - reference) / np.outer(sd, sd)))
    assert model.df_a_ == oracle["df_a"] and model.nobs_ == len(y)
    assert error["b_scaled"] <= 1e-8 and error["se_rel"] <= 1e-7 and covariance <= 1e-8, error
    print("weighted gauss-seidel pass", model.num_iterations_, error, "scaled V", covariance)
    flags, = struct.unpack_from("<i", cache.read_bytes(), 64 + 5 * 4)
    assert flags == 5  # Converged, with the fit's refinement used.
    os.environ["XHDFE_ABSORPTION_CACHE_MODE"] = "read"
    warm = core.HdfeRegressor(**opts)
    warm.fit(y, x, fes=fes, weights=w, clusters=[fes[0]])
    assert warm.num_iterations_ == model.num_iterations_
    np.testing.assert_array_equal(warm.coef_, model.coef_)
    np.testing.assert_array_equal(warm.covariance_, model.covariance_)
    print("weighted gauss-seidel cache pass", flags)

    # Nine RHS cross the width of the prototype's batches. A single block
    # projection must preserve proportional columns and their identified fit.
    os.environ["XHDFE_ABSORPTION_CACHE_MODE"] = "off"
    wide = core.HdfeRegressor(**opts)
    wide.fit(y, np.asfortranarray(np.column_stack((x, x))), fes=fes, weights=w, clusters=[fes[0]])
    assert wide.converged_ and wide.precision_certified_
    assert int(np.sum(np.asarray(wide.omitted_)[:8])) == 4
    transport = np.zeros((5, 9))
    transport[:4, :4] = transport[:4, 4:8] = np.eye(4)
    transport[4, 8] = 1
    fit = dict(b=transport @ wide.coef_, V=transport @ wide.covariance_ @ transport.T)
    error = grade(oracle, fit)
    covariance = float(np.max(np.abs(fit["V"] - reference) / np.outer(sd, sd)))
    assert error["b_scaled"] <= 1e-8 and error["se_rel"] <= 1e-7 and covariance <= 1e-8, error
    print("weighted gauss-seidel aliased pass", wide.num_iterations_, error, "scaled V", covariance)


def run(module, output):
    for key in tuple(os.environ):
        if key.startswith("XHDFE_"):
            del os.environ[key]
    os.environ.update(XHDFE_GPU_BACKEND="cpu", XHDFE_MOBILITY_MODE="off",
                      XHDFE_FE_STRUCTURE_MODE="off", XHDFE_DEBUG_SOLVER="1")
    output.mkdir(mode=0o700)
    load = importlib.util.spec_from_file_location("py_hdfe_v11", module.resolve())
    core = importlib.util.module_from_spec(load)
    load.loader.exec_module(core)
    print("module", core.__file__)
    strict(core, output)
    gauss_seidel(core, output)
    print("cpu_projection_contract PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.module, args.output)
