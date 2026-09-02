#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, expected: str):
    if sha256(path) != expected:
        raise AssertionError("module hash differs")
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    group_rows = []
    individual_rows = []
    firm_rows = []
    year_rows = []
    y_rows = []
    x_rows = []
    weight_rows = []
    for group in range(96):
        members = (group % 31, (group + 1) % 31, (group + 2) % 31)
        firm = group % 13
        year = group % 8
        x1 = np.sin(group * 0.17)
        x2 = np.cos(group * 0.11)
        outcome = 1.25 * x1 - 0.4 * x2 + firm * 0.03 - year * 0.02
        weight = 1.0 + (group % 4)
        for individual in members:
            group_rows.append(group)
            individual_rows.append(individual)
            firm_rows.append(firm)
            year_rows.append(year)
            y_rows.append(outcome)
            x_rows.append((x1, x2))
            weight_rows.append(weight)
    return (
        np.asarray(y_rows, dtype=np.float64),
        np.asfortranarray(np.asarray(x_rows, dtype=np.float64)),
        [np.asarray(individual_rows, dtype=np.int32),
         np.asarray(firm_rows, dtype=np.int32),
         np.asarray(year_rows, dtype=np.int32)],
        np.asarray(group_rows, dtype=np.int32),
        np.asarray(individual_rows, dtype=np.int32),
        np.asarray(weight_rows, dtype=np.float64),
    )


def fit(module, method: str, weights):
    y, X, fes, group, individual, fixture_weights = fixture()
    reg = module.HdfeRegressor(
        se_type="unadjusted", tol=1.0e-8, max_iter=100000,
        fit_intercept=True, num_threads=4, drop_singletons=True,
        retain_fes=False, symmetric_sweep=False,
        absorption_method=method, tolerance_mode="reghdfe-comparable",
    )
    reg.fit(
        y, X, fes=fes, weights=fixture_weights if weights else None,
        clusters=None, group=group, individual=individual,
        aggregation="sum", slopes=[],
    )
    return reg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--module-sha256", required=True)
    args = parser.parse_args()
    module = load_module(args.module, args.module_sha256)

    records = []
    for weighted in (False, True):
        auto = fit(module, "auto", weighted)
        explicit = fit(module, "lsmr", weighted)
        for reg in (auto, explicit):
            assert reg.converged_ and reg.precision_certified_
            assert int(reg.absorption_method_used) == 5
            assert not reg.auto_routing_retry_policy_enabled_
            assert not reg.auto_routing_retry_eligible_
            assert not reg.auto_routing_retry_fired_
            assert int(reg.auto_routing_retry_status_) == 0
            assert np.all(np.isfinite(np.asarray(reg.coef_)))
            assert np.all(np.isfinite(np.asarray(reg.covariance_)))
        assert np.array_equal(np.asarray(auto.sample_index_),
                              np.asarray(explicit.sample_index_))
        assert np.array_equal(np.asarray(auto.coef_), np.asarray(explicit.coef_))
        assert np.array_equal(np.asarray(auto.covariance_),
                              np.asarray(explicit.covariance_))
        assert np.array_equal(np.asarray(auto.residuals_),
                              np.asarray(explicit.residuals_))
        records.append((weighted, int(auto.num_iterations_),
                        float(auto.abs_residual_rel_)))

    print(
        "GROUP_INDIVIDUAL_TOLERANCE_PASS "
        + " ".join(
            f"weighted={int(weighted)}:iterations={iterations}:rel={rel:.3e}"
            for weighted, iterations, rel in records
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
