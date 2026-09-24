#!/usr/bin/env python3
"""AKM lincom against explicit dummy OLS and an 80-digit projection oracle."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import mpmath as mp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    mp.mp.dps = 80
    rng = np.random.default_rng(739)
    worker = np.repeat(np.arange(40), 6).astype(np.int32)
    firm = ((worker + 3*np.tile(np.arange(6), 40)) % 13).astype(np.int32)
    n = worker.size
    y = rng.normal(size=40)[worker] + rng.normal(size=13)[firm] + rng.normal(size=n)
    d = np.column_stack((np.eye(40)[worker], np.eye(13)[firm, :12]))
    q, r = np.linalg.qr(d, mode="reduced")
    beta = np.linalg.solve(r, q.T @ y)
    residual = y-d @ beta
    sigma = (y-y.mean())*residual/(1-(q*q).sum(axis=1))
    psi = np.r_[beta[40:], 0][firm]
    psi -= psi.mean()
    z = np.rint((rng.normal(size=n) + psi)*256)/256
    v = np.rint(rng.normal(size=n)*256)/256
    designs = {
        "ordinary": z[:, None], "offset": (z+2.**30)[:, None],
        "small_units": (z*2.**-30)[:, None], "large_units": (z*2.**30)[:, None],
        "two_columns": np.column_stack((z, v)),
        "weak_direction": np.column_stack((z, z+v*2.**-20)),
        "duplicate": np.column_stack((z, z)),
        "constant": np.column_stack((np.ones(n), z)),
    }
    rows = []
    previous = {}
    baseline = core.akm_kss(y, worker, firm, num_threads=2)
    np.testing.assert_allclose(baseline["psi"], psi, atol=1e-10, rtol=0)
    for block in (0, 8):
        os.environ["XHDFE_AKM_SE_BLOCK"] = str(block)
        for name, zmat in designs.items():
            result = core.akm_kss(y, worker, firm, Z=zmat, num_threads=2)
            assert result["converged"], result["notes"]
            for key in ("alpha", "psi", "pii", "sigma_i"):
                assert np.array_equal(result[key], baseline[key]), key
            lincom = result["lincom"]
            identified = np.flatnonzero(np.isfinite(lincom["coef"]))
            expected_rank = 1 if name in ("duplicate", "constant") else zmat.shape[1]
            assert identified.size == expected_rank
            if name == "constant":
                assert identified.tolist() == [1]
            for j in set(range(zmat.shape[1]))-set(identified):
                assert all(np.isnan(lincom[k][j]) for k in ("coef", "se_white", "se_kss", "t"))
                assert "omitted as collinear" in result["notes"]
            # High precision permits an independent normal-equation oracle even
            # when the represented design has a huge origin or a weak direction.
            a = mp.matrix(np.column_stack((np.ones(n), zmat[:, identified])).tolist())
            influence = (a.T*a)**-1*a.T
            coefficients = influence*mp.matrix(result["psi"].tolist())
            errors = []
            for position, j in enumerate(identified, 1):
                contrast = np.array([float(influence[position, i]) for i in range(n)])
                target = np.r_[np.zeros(40), np.bincount(firm, weights=contrast, minlength=13)[:12]]
                projected = q @ np.linalg.solve(r.T, target)
                white = np.sqrt(np.dot(residual**2, projected**2))
                variance_kss = np.dot(sigma, projected**2)
                kss = np.sqrt(variance_kss) if variance_kss > 0 else np.nan
                expected = {"coef": float(coefficients[position]), "se_white": white, "se_kss": kss}
                expected["t"] = expected["coef"]/kss
                for key, value in expected.items():
                    observed = lincom[key][j]
                    if np.isnan(value):
                        assert np.isnan(observed), (name, key)
                    else:
                        error = abs(observed-value)/max(abs(value), 1e-300)
                        assert error <= 1e-8, (name, block, key, error, observed, value)
                        errors.append(error)
            if block == 0:
                previous[name] = lincom
            else:
                for key in ("coef", "se_white", "se_kss", "t"):
                    np.testing.assert_allclose(lincom[key], previous[name][key], rtol=1e-10, atol=0, equal_nan=True)
            rows.append(dict(case=name, block=block, max_relative_error=max(errors), status="PASS"))
    args.output.write_text(json.dumps(rows, indent=2)+"\n")
    print(f"PASS: {len(rows)} AKM lincom cases with independent coefficient/White/KSS oracles")


if __name__ == "__main__":
    main()
