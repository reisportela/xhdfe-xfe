"""The warm completion must share the original iteration budget and clear failure."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("--module", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
from cuda_three_fe_covariance_contract import fixture, oracle, metrics
import numpy as np
for key in list(os.environ):
    if key.startswith("XHDFE_"):
        del os.environ[key]
os.environ.update(XHDFE_GPU_BACKEND="cuda", XHDFE_ABSORPTION_CACHE_MODE="off",
                  XHDFE_MOBILITY_MODE="off", XHDFE_FE_STRUCTURE_MODE="off")
spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
y, x, fes = fixture()
reference_b, reference_v, _ = oracle(y, x, fes)
rows = []
# Frozen fixture: the initial IT stage finishes at143 iterations; the warm
# Krylov solve needs more than the seven remaining at maximum150.
for maximum in [150, 1000]:
    model = core.HdfeRegressor(
        se_type="cluster", tol=1e-8, max_iter=maximum, num_threads=1,
        drop_singletons=True, retain_fes=False, symmetric_sweep=False,
        absorption_method="auto", tolerance_mode="reghdfe-comparable",
        jacobi_relaxation=0.)
    row = dict(max_iter=maximum)
    try:
        model.fit(y, np.asfortranarray(x), fes=fes, clusters=fes[0])
        difference = metrics(np.asarray(model.coef_), np.asarray(model.covariance_),
                             reference_b, reference_v)
        valid = (maximum == 1000 and model.converged_ and model.precision_certified_
                 and model.gpu_used_ and model.gpu_status_code_ == 1
                 and 0 <= model.num_iterations_ <= maximum
                 and difference["b_scaled"] <= 1e-9 and difference["V_scaled"] <= 1e-8)
        row.update(status="PASS" if valid else "FAIL_UNEXPECTED_RETURN",
                   returned=True, iterations=int(model.num_iterations_),
                   converged=bool(model.converged_), gpu_used=bool(model.gpu_used_),
                   **difference)
    except RuntimeError as error:
        message = str(error)
        match = re.search(r"iterations=(\d+)/(\d+)", message)
        stale = any(np.asarray(getattr(model, name)).size
                    for name in ["coef_", "covariance_", "residuals_"])
        valid = (maximum == 150 and match is not None and int(match[2]) == maximum
                 and 0 <= int(match[1]) <= maximum and not stale
                 and not model.converged_ and not model.precision_certified_)
        row.update(status="EXPECTED_REFUSAL" if valid else "FAIL", returned=False,
                   error=message, stale_estimates=bool(stale))
    rows.append(row)
    print(json.dumps(row), flush=True)
result = dict(status="PASS_BUDGET_CONTRACT" if all(
    row["status"] in ["PASS", "EXPECTED_REFUSAL"] for row in rows) else "FAIL",
    rows=rows, module=str(args.module.resolve()),
    module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
    numerical_pass_cases=sum(row["status"] == "PASS" for row in rows),
    expected_refusals=sum(row["status"] == "EXPECTED_REFUSAL" for row in rows),
    refusal_is_not_numerical_success=True)
with args.output.open("x") as out:
    json.dump(result, out, indent=2)
    out.write("\n")
raise SystemExit(result["status"] == "FAIL")
