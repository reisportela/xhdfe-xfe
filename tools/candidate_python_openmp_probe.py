#!/usr/bin/env python3
"""Gate actual OpenMP work in an installed xhdfe wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


# Exact fixture and limits transcribed from tools/plugin_openmp_probe.cpp.
N = 64 * 64 * 32
BETA_ATOL = 1e-10
V_SCALED_ATOL = 1e-8
RSS_RTOL = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-prefix", type=Path, required=True)
    parser.add_argument("--forbid-prefix", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    expected_prefix = args.expected_prefix.resolve(strict=True)
    forbidden_prefix = args.forbid_prefix.resolve(strict=True)
    output = args.out.resolve()
    require(not output.exists(), f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        XHDFE_GPU_BACKEND="cpu",
        XHDFE_CERTIFY="0",
        XHDFE_ABSORPTION_CACHE_MODE="off",
        XHDFE_MOBILITY_MODE="off",
        OMP_DYNAMIC="FALSE",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )

    receipt: dict = {
        "schema": "xhdfe-candidate-python-openmp-v1",
        "status": "FAIL",
        "fixture_rows": N,
        "limits": {
            "beta_atol": BETA_ATOL,
            "covariance_diagonal_scaled": V_SCALED_ATOL,
            "rss_rtol": RSS_RTOL,
        },
    }
    try:
        import numpy as np
        import xhdfe
        from xhdfe import py_hdfe_v11 as core

        package = Path(xhdfe.__file__).resolve(strict=True).parent
        module = Path(core.__file__).resolve(strict=True)
        require(inside(package, expected_prefix), "xhdfe was not imported from the private venv")
        require(inside(module, expected_prefix), "native module was not imported from the private venv")
        require(not inside(package, forbidden_prefix), "xhdfe was imported from the checkout")
        require(not inside(module, forbidden_prefix), "native module was imported from the checkout")
        require(xhdfe.__version__ == args.expected_version, "installed wheel version differs")

        row = np.arange(N, dtype=np.int64)
        replicate = row % 32
        fe2 = ((row // 32) % 64).astype(np.int32)
        fe1 = (row // (32 * 64)).astype(np.int32)
        x1 = np.where((replicate & 1) != 0, 1.0, -1.0)
        x2 = np.where((replicate & 2) != 0, 1.0, -1.0)
        noise = np.where((replicate & 4) != 0, 0.125, -0.125)
        X = np.asfortranarray(np.column_stack((x1, x2)))
        y = 1.25 * x1 - 0.5 * x2 + 3.0 + 0.25 * fe1 - 0.125 * fe2 + noise
        expected_beta = np.array([1.25, -0.5])
        expected_rss = N * 0.015625

        rows = []
        thread_one = None
        for requested in (1, 2):
            model = xhdfe.HdfeRegressor(
                num_threads=requested,
                max_iter=1000,
                tol=1e-8,
                drop_singletons=False,
                fit_intercept=False,
                se_type="unadjusted",
                absorption_method="gauss-seidel",
                tolerance_mode="reghdfe-comparable",
            )
            model.fit(y, X, fes=[fe1, fe2])
            diagnostics = (
                int(model.threads_requested_),
                int(model.threads_effective_),
                int(model.threads_used_),
                int(model.parallel_workers_active_),
            )
            require(diagnostics == (requested,) * 4, f"thread diagnostics differ: {diagnostics}")
            require(bool(model.openmp_enabled_), "OpenMP is disabled")
            require(int(model.thread_capacity_) >= 2, "OpenMP capacity is below two")
            require(model.converged_ and model.precision_certified_, "fit is not converged/certified")
            require(not model.gpu_used_ and int(model.gpu_status_code_) == 0, "CPU probe used a GPU")
            require(int(model.nobs_) == N, "fit changed the analytic sample")
            require(1 <= float(model.df_resid_) <= N, "invalid residual degrees of freedom")
            require(int(model.absorption_method_used) == 1, "explicit Gauss-Seidel method changed")

            beta = np.asarray(model.coef_, dtype=np.float64)
            covariance = np.asarray(model.covariance_, dtype=np.float64)
            variance = 0.015625 / float(model.df_resid_)
            expected_v = np.diag([variance, variance])
            beta_error = float(np.max(np.abs(beta - expected_beta)))
            v_error = float(np.max(np.abs(covariance - expected_v)) / variance)
            rss_error = abs(float(model.rss_) - expected_rss) / expected_rss
            require(beta_error <= BETA_ATOL, f"analytic beta mismatch: {beta_error}")
            require(v_error <= V_SCALED_ATOL, f"analytic covariance mismatch: {v_error}")
            require(rss_error <= RSS_RTOL, f"analytic RSS mismatch: {rss_error}")

            if thread_one is not None:
                parity_b = float(np.max(np.abs(beta - thread_one[0])))
                parity_v = float(np.max(np.abs(covariance - thread_one[1])) / variance)
                parity_rss = abs(float(model.rss_) - thread_one[2]) / expected_rss
                require(parity_b <= BETA_ATOL, f"thread beta disagreement: {parity_b}")
                require(parity_v <= V_SCALED_ATOL, f"thread covariance disagreement: {parity_v}")
                require(parity_rss <= RSS_RTOL, f"thread RSS disagreement: {parity_rss}")
            else:
                parity_b = parity_v = parity_rss = 0.0
                thread_one = (beta.copy(), covariance.copy(), float(model.rss_))
            rows.append({
                "requested": requested,
                "threads_effective": diagnostics[1],
                "threads_used": diagnostics[2],
                "parallel_workers_active": diagnostics[3],
                "openmp_enabled": bool(model.openmp_enabled_),
                "beta_error": beta_error,
                "covariance_scaled_error": v_error,
                "rss_relative_error": rss_error,
                "thread_beta_difference": parity_b,
                "thread_covariance_scaled_difference": parity_v,
                "thread_rss_relative_difference": parity_rss,
            })
        receipt.update(
            status="PASS",
            package_path=str(package),
            package_version=xhdfe.__version__,
            module_path=str(module),
            module_sha256=sha256(module),
            rows=rows,
        )
    except Exception as error:
        receipt["reason"] = repr(error)

    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(output)}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
