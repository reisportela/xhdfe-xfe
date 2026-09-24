#!/usr/bin/env python3
"""Empty numeric labels have zero effect on an explicit Schwarz projection."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ.update(XHDFE_MOBILITY_MODE="off", XHDFE_FE_STRUCTURE_MODE="off",
                      XHDFE_ABSORPTION_CACHE_MODE="off", XHDFE_GPU_BACKEND="cpu")
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(9)
    n = 30000
    original = [np.repeat(np.arange(5000), 6).astype(np.int32),
                rng.integers(0, 500, n).astype(np.int32),
                rng.integers(0, 10, n).astype(np.int32)]
    x = rng.normal(size=(n, 2)) + rng.normal(size=5000)[original[0], None]
    y = x.sum(axis=1) + rng.normal(size=5000)[original[0]]
    y += rng.normal(size=500)[original[1]] + .1*original[2] + rng.normal(size=n)
    rows = []
    for dimension, missing in ((2, 5), (2, 0), (0, 5), (1, 5)):
        fes = [f.copy() for f in original]
        fes[dimension][fes[dimension] == missing] = 4 if missing == 5 else 1
        dense = [np.unique(f, return_inverse=True)[1].astype(np.int32) for f in fes]
        starts = np.cumsum([0]+[int(f.max())+1 for f in dense])
        columns = np.column_stack([f+starts[j] for j, f in enumerate(dense)]).ravel()
        d = csr_matrix((np.ones(3*n), (np.repeat(np.arange(n), 3), columns)),
                       shape=(n, int(starts[-1])))
        within = []
        for v in (y, x[:, 0], x[:, 1]):
            within.append(v-d @ lsmr(d, v, atol=1e-14, btol=1e-14, maxiter=5000)[0])
        b = np.linalg.lstsq(np.column_stack(within[1:]), within[0], rcond=None)[0]
        models = []
        for labels in (fes, dense):
            model = core.HdfeRegressor(num_threads=2, absorption_method="schwarz")
            model.fit(y, x, labels)
            error = float(np.max(abs(model.coef_[:2]-b)/np.maximum(1, abs(b))))
            assert model.converged_ and model.precision_certified_ and error <= 1e-9
            models.append(model)
        v = np.asarray(models[1].covariance_)
        covariance_error = np.max(abs(models[0].covariance_-v)/
                                  np.sqrt(np.outer(v.diagonal(), v.diagonal())))
        assert covariance_error <= 1e-8
        assert models[0].df_a_ == models[1].df_a_
        rows.append(dict(dimension=dimension, missing=missing,
                         b_error=error, iterations=models[0].num_iterations_, status="PASS"))
    args.output.write_text(json.dumps(rows, indent=2)+"\n")
    print("PASS: Schwarz empty levels, independent sparse oracle and compact-label parity")


if __name__ == "__main__":
    main()
