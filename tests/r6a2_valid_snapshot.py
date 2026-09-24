#!/usr/bin/env python3
"""Bitwise valid-input snapshot for the R6A.2 zero-diff gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_module(module_dir: Path):
    sys.path.insert(0, str(module_dir.resolve()))
    return importlib.import_module("py_hdfe_v11")


def array_record(value) -> dict[str, object]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    record = {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256_c_bytes": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }
    if contiguous.size <= 16:
        record["bytes_hex"] = contiguous.tobytes().hex()
    return record


def float_record(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value.hex()


def fit_record(reg) -> dict[str, object]:
    return {
        "coef": array_record(reg.coef_),
        "stderr": array_record(reg.stderr_),
        "covariance": array_record(reg.covariance_),
        "residuals": array_record(reg.residuals_),
        "sample_index": array_record(reg.sample_index_),
        "iterations": int(reg.num_iterations_),
        "converged": bool(reg.converged_),
        "method": str(reg.absorption_method_used),
        "threads_used": int(reg.threads_used_),
        "gpu_used": bool(reg.gpu_used_),
        "gpu_status_code": int(reg.gpu_status_code_),
        "fe_recovery_iterations": int(reg.fe_recovery_iterations_),
        "fe_recovery_converged": bool(reg.fe_recovery_converged_),
        "fe_recovery_max_delta": float_record(reg.fe_recovery_max_delta_),
        "fe_effects": [array_record(effect) for effect in reg.fe_effects_],
    }


def make_reg(mod, **kwargs):
    options = {
        "se_type": "unadjusted",
        "tol": 1e-10,
        "max_iter": 100000,
        "check_interval": 1,
        "drop_singletons": False,
        "num_threads": 1,
        "tolerance_mode": "reghdfe-comparable",
    }
    options.update(kwargs)
    return mod.HdfeRegressor(**options)


def snapshots(mod) -> dict[str, object]:
    rng = np.random.default_rng(20260714)
    n = 1200
    row = np.arange(n)
    fe1 = (row % 47).astype(np.int32)
    fe2 = (row % 31).astype(np.int64)
    exog1 = rng.normal(size=n)
    exog2 = rng.normal(size=n)
    x = np.column_stack((exog1, exog2))
    y = 0.8 * exog1 - 0.35 * exog2 + 0.02 * fe1 - 0.01 * fe2
    y += rng.normal(scale=0.4, size=n)
    weights = 0.25 + rng.random(n) * 2.5
    out: dict[str, object] = {}

    reg = make_reg(mod)
    reg.fit(y, np.ascontiguousarray(x), fes=[fe1, fe2])
    out["plain_c_int32_int64"] = fit_record(reg)

    large_integral = (2**40 + (row % 53)).astype(np.float64)
    large_uint = (2**40 + (row % 29)).astype(np.uint64)
    reg = make_reg(mod)
    reg.fit(y, np.asfortranarray(x), fes=[large_integral, large_uint], weights=weights)
    out["weighted_f_integral_float_uint64"] = fit_record(reg)

    clusters = np.column_stack((fe1.astype(np.int64), fe2))
    reg = make_reg(mod, se_type="cluster")
    reg.fit(y, np.asfortranarray(x), fes=[fe1, fe2], clusters=clusters)
    out["multiway_cluster"] = fit_record(reg)

    slope = 0.5 + np.sin(row * 0.013)
    reg = make_reg(mod, retain_fes=True)
    reg.fit(y, np.asfortranarray(x[:, :1]), fes=[fe1, fe2],
            slopes=[(0, slope, True)])
    out["slope_savefe"] = fit_record(reg)

    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    endog = 0.7 * z1 - 0.25 * z2 + 0.2 * exog1 + rng.normal(scale=0.5, size=n)
    iv_x = np.asfortranarray(np.column_stack((exog1, endog)))
    iv_y = 0.6 * exog1 + 1.4 * endog + 0.02 * fe1 - 0.01 * fe2
    iv_y += rng.normal(scale=0.35, size=n)
    reg = make_reg(mod, se_type="robust")
    reg.fit(iv_y, iv_x, fes=[fe1, fe2], weights=weights,
            instruments=np.asfortranarray(np.column_stack((z1, z2))),
            endogenous_idx=[1])
    out["iv_weighted_fe"] = fit_record(reg)

    groups = (row // 3).astype(np.int64)
    individuals = (row % 173).astype(np.int64)
    grouped_fe = (groups % 23).astype(np.int64)
    group_y = rng.normal(size=int(groups.max()) + 1)[groups]
    group_x = rng.normal(size=int(groups.max()) + 1)[groups, None]
    group_weights = (0.5 + rng.random(int(groups.max()) + 1))[groups]
    reg = make_reg(mod)
    reg.fit(group_y, np.asfortranarray(group_x), fes=[grouped_fe, individuals],
            weights=group_weights, group=groups, individual=individuals,
            aggregation="mean")
    out["group_individual"] = fit_record(reg)

    workers = np.repeat(np.arange(80, dtype=np.int64), 5)
    period = np.tile(np.arange(5, dtype=np.int64), 80)
    firms = (workers + period) % 19
    akm_y = rng.normal(size=workers.size)
    fweights = (1 + (np.arange(workers.size) % 3)).astype(np.float64)
    akm = mod.akm_kss(
        akm_y, workers, firms, leave_out_level="match", leverages="exact",
        prune=False, num_threads=1, fweights=fweights,
    )
    out["akm_fweights"] = {
        "alpha": array_record(akm["alpha"]),
        "psi": array_record(akm["psi"]),
        "pii": array_record(akm["pii"]),
        "sigma_i": array_record(akm["sigma_i"]),
        "row_weight": array_record(akm["row_weight"]),
        "plugin": {key: float_record(akm["plugin"][key])
                   for key in ("var_alpha", "var_psi", "cov_alpha_psi")},
        "kss": {key: float_record(akm["kss"][key])
                for key in ("var_alpha", "var_psi", "cov_alpha_psi")},
        "converged": bool(akm["converged"]),
        "solver_iterations": int(akm["solver_iterations"]),
        "solver_direct": bool(akm["solver_direct"]),
        "gpu_used": bool(akm["gpu_used"]),
    }

    ng = 900
    gx1 = rng.normal(size=(ng, 2))
    gx2 = rng.normal(size=(ng, 3))
    gf1 = (np.arange(ng) % 41).astype(np.int64)
    gf2 = (np.arange(ng) % 17).astype(np.int64)
    gy = gx1 @ np.array([0.5, -0.2]) + gx2 @ np.array([0.3, 0.1, -0.4])
    gy += 0.01 * gf1 - 0.015 * gf2 + rng.normal(scale=0.35, size=ng)
    gelbach = mod.gelbach_decompose(
        gy, gx1, gx2, [2, 1], [gf1, gf2], tol=1e-9,
        num_threads=1, weights=0.5 + rng.random(ng),
    )
    out["gelbach_weighted"] = {
        key: array_record(gelbach[key])
        for key in ("b_base", "b_full", "delta", "cov", "total", "total_cov")
    } | {
        "identity_gap": float_record(gelbach["identity_gap"]),
        "converged": bool(gelbach["converged"]),
    }

    for label, record in out.items():
        if isinstance(record, dict) and "converged" in record and not record["converged"]:
            raise AssertionError(f"{label} did not converge")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", required=True, type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    module = load_module(args.module_dir)
    results = snapshots(module)
    comparison = None
    if args.compare:
        reference = json.loads(args.compare.read_text(encoding="utf-8"))["results"]
        comparison = {
            "bit_identical": results == reference,
            "mismatched_cases": sorted(
                (set(results) | set(reference))
                - {name for name in set(results) & set(reference)
                   if results[name] == reference[name]}
            ),
        }
        if not comparison["bit_identical"]:
            raise AssertionError(
                "valid-input snapshots differ: " + ", ".join(comparison["mismatched_cases"])
            )
    payload = {
        "schema": "xhdfe-r6a2-valid-snapshot-v1",
        "module": str(Path(module.__file__).resolve()),
        "numpy": np.__version__,
        "results": results,
        "comparison": comparison,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
