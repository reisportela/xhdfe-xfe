"""Exact constant-span and varying-slope controls for pure heterogeneous FEs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from exact_wls_reference import fit


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--module", type=Path, required=True)
parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

os.environ.update(
    XHDFE_GPU_BACKEND=args.backend,
    XHDFE_ABSORPTION_CACHE_MODE="off",
    XHDFE_MOBILITY_MODE="off",
    XHDFE_FE_NORMALIZE="component",
)
spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
cpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp)

n = 192
i = np.arange(n)
worker = i // 48
a = 2 * (i % 2) - 1
b = 2 * ((i // 2) % 2) - 1
noise = 0.125 * (2 * ((i // 4) % 2) - 1)
x = a + 0.5 * worker
weights = 1.0 + worker
z_group = np.array([1.0, 2.0, 4.0, 8.0])[worker]
alpha = np.array([0.5, -0.25, 0.375, -0.125])[worker]

cases = []
for name, z, explicit_constant in (
    ("constant_by_group", z_group, False),
    ("varying_within_group", z_group + 0.25 * b, True),
):
    pure_columns = np.column_stack([(worker == g) * z for g in range(4)])
    design = np.column_stack((pure_columns, np.ones(n), x)) if explicit_constant else np.column_stack((pure_columns, x))
    y = 3.0 + 0.75 * x + alpha * z + noise
    expected_rank = 6 if explicit_constant else 5
    assert np.linalg.matrix_rank(design) == expected_rank
    reference = fit(design, y, weights)
    reference_residuals = np.array([float(value) for value in reference["residuals"]])
    assert np.max(np.abs(reference_residuals - noise)) == 0.0
    weighted_mean = float(np.average(y, weights=weights))
    expected_tss = float(np.sum(weights * (y - weighted_mean) ** 2) * n / np.sum(weights))
    expected_intercept = 3.0 if explicit_constant else float(np.average(y - 0.75 * x, weights=weights))
    cases.append(dict(
        name=name,
        z=z,
        y=y,
        intercept_identified=explicit_constant,
        expected_rank=expected_rank,
        expected_df=n - expected_rank,
        expected_variance=float(reference["variance"]),
        expected_residuals=reference_residuals,
        expected_tss=expected_tss,
        expected_intercept=expected_intercept,
    ))

rows = []
for case in cases:
    for retain in (False, True):
        model = cpp.HdfeRegressor(
            num_threads=2,
            max_iter=1000,
            tol=1e-8,
            drop_singletons=False,
            fit_intercept=True,
            retain_fes=retain,
            se_type="unadjusted",
        )
        row = dict(case=case["name"], retain_fes=retain, backend=args.backend)
        try:
            model.fit(
                case["y"],
                x[:, None],
                fes=[worker],
                slopes=[(0, case["z"], False)],
                weights=weights,
            )
        except RuntimeError as error:
            row.update(status="FAIL", error=str(error))
        else:
            coefficients = np.asarray(model.coef_)
            residuals = np.asarray(model.residuals_)
            row.update(
                beta_error=abs(float(coefficients[0]) - 0.75),
                intercept_error=abs(float(coefficients[-1]) - case["expected_intercept"]),
                variance_relative_error=abs(float(model.covariance_[0, 0]) / case["expected_variance"] - 1.0),
                residual_error=float(np.max(np.abs(residuals - case["expected_residuals"]))),
                rss_error=abs(float(model.rss_) - float(np.sum(weights * noise ** 2) * n / np.sum(weights))),
                tss_error=abs(float(model.tss_) - case["expected_tss"]),
                df_resid=float(model.df_resid_),
                df_a=float(model.df_a_),
                df_m=float(model.df_m_),
                iterations=int(model.num_iterations_),
            )
            if retain:
                fe_sum = sum((np.asarray(value) for value in model.fe_effects_), np.zeros(n))
                fitted = coefficients[0] * x + coefficients[-1] + fe_sum
                row["reconstruction_error"] = float(
                    np.max(np.abs(case["y"] - fitted - case["expected_residuals"]))
                )
            passed = (
                coefficients.size == 2
                and np.all(np.isfinite(coefficients))
                and row["beta_error"] <= 1e-10
                and (not case["intercept_identified"] or row["intercept_error"] <= 1e-10)
                and row["variance_relative_error"] <= 1e-10
                and row["residual_error"] <= 1e-10
                and row["rss_error"] <= 1e-10
                and row["tss_error"] <= 1e-9
                and row.get("reconstruction_error", 0.0) <= 1e-10
                and model.nobs_ == n
                and model.df_resid_ == case["expected_df"]
                and model.df_m_ == 1
                and model.converged_
                and model.precision_certified_
                and (not retain or model.fe_recovery_converged_)
                and bool(model.gpu_used_) == (args.backend == "cuda")
                and model.num_iterations_ <= 1000
            )
            row["status"] = "PASS" if passed else "FAIL"
        rows.append(row)

passed = all(row["status"] == "PASS" for row in rows)
payload = dict(
    status="PASS" if passed else "FAIL",
    backend=args.backend,
    module=str(args.module.resolve()),
    module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
    positive_weight_range=[float(weights.min()), float(weights.max())],
    rows=rows,
)
with args.out.open("x") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
print(json.dumps(dict(status=payload["status"], rows=len(rows), backend=args.backend)))
raise SystemExit(0 if passed else 1)
