#!/usr/bin/env python3
"""Permanent CPU IV contract for the 200,000-row weak, identified fixture.

Supply either --module for a fresh fit or --results for an existing
r6a2_iv_preflight.py receipt, and a new --out JSON. The reference uses the
represented binary64 inputs at 90 decimal digits, checked again at 120.
No rank truncation, precision relaxation or performance verdict is applied.
Requires NumPy and mpmath. Historical receipts do not seal fixture arrays;
receipt-only adjudication is explicitly weaker than a fresh module run.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import mpmath as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tests/r6a2_iv_preflight.py"
CASE = "large_sample_weak_full_rank"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(case, digits):
    if case.get("fes") or case.get("weights") is not None or case.get("fit_intercept", False):
        raise ValueError("This contract requires unweighted inputs without FE or an intercept")
    x = np.asarray(case["x"], dtype=np.float64)
    y = np.asarray(case["y"], dtype=np.float64)[:, None]
    exogenous = [j for j in range(x.shape[1]) if j not in case["endogenous"]]
    z = np.column_stack((x[:, exogenous], case["z"]))
    if not all(np.isfinite(a).all() for a in (x, y, z)):
        raise ValueError("Non-finite reference input")
    with mp.workdps(digits):
        def columns(array):
            return [[mp.mpf(float(v)) for v in array[:, j]]
                    for j in range(array.shape[1])]
        def cross(a, b):
            return mp.matrix([[mp.fsum(u * v for u, v in zip(c, d))
                               for d in b] for c in a])
        X, Z, Y = columns(x), columns(z), columns(y)
        zx, zy, zz = cross(Z, X), cross(Z, Y), cross(Z, Z)
        xx, xy, yy = cross(X, X), cross(X, Y), cross(Y, Y)
        zz_inverse = mp.inverse(zz)
        gram = zx.T * zz_inverse * zx
        beta = mp.lu_solve(gram, zx.T * zz_inverse * zy)
        rss = (yy - 2 * beta.T * xy + beta.T * xx * beta)[0]
        df = x.shape[0] - x.shape[1]
        return beta, (rss / df) * mp.inverse(gram), rss, df


def grade(coef, covariance, ref):
    beta, variance, _, _ = ref
    p = len(beta)
    b = np.asarray(coef, dtype=np.float64)
    v = np.asarray(covariance, dtype=np.float64)
    if b.shape != (p,) or v.shape != (p, p) or not np.isfinite(b).all() or not np.isfinite(v).all():
        return dict(status="FAIL", reason="Non-finite or incorrectly shaped coefficient/covariance")
    with mp.workdps(120):
        if any(variance[i, i] <= 0 for i in range(p)):
            raise ValueError("This fixture requires strictly positive reference variances")
        db = max(abs(mp.mpf(float(b[i])) - beta[i]) / max(1, abs(beta[i]))
                 for i in range(p))
        dv = max(abs(mp.mpf(float(v[i, j])) - variance[i, j]) /
                 mp.sqrt(variance[i, i] * variance[j, j])
                 for i in range(p) for j in range(p))
        return dict(status="PASS" if db <= mp.mpf("1e-9") and dv <= mp.mpf("1e-8") else "FAIL",
                    coefficient_scaled_error=float(db), full_covariance_scaled_error=float(dv))


def reference_agreement(low, high):
    with mp.workdps(120):
        beta, variance, rss, _ = high
        errors = [abs(low[0][i] - beta[i]) / max(1, abs(beta[i]))
                  for i in range(len(beta))]
        errors.extend(abs(low[1][i, j] - variance[i, j]) /
                      mp.sqrt(variance[i, i] * variance[j, j])
                      for i in range(len(beta)) for j in range(len(beta)))
        errors.append(abs(low[2] - rss) / max(1, abs(rss)))
        maximum = max(errors)
        return dict(max_scaled_difference=float(maximum), agrees=bool(maximum <= mp.mpf("1e-60")))


def walsh_self_check():
    i = np.arange(128)
    w, t, u, e = (2.0 * ((i // (2 ** k)) % 2) - 1.0 for k in range(4))
    x = np.column_stack((w, t + u))
    case = dict(y=x @ np.array([2.0, 3.0]) + e, x=x,
                z=(w + 2.0 ** -27 * t)[:, None], endogenous=[1])
    beta, variance, rss, df = reference(case, 90)
    with mp.workdps(120):
        error = max(abs(beta[0] - 2), abs(beta[1] - 3), abs(rss - 128),
                    *(abs(variance[i, j] - (mp.mpf(1) / 126 if i == j else 0))
                      for i in range(2) for j in range(2)))
        return dict(status="PASS" if error <= mp.mpf("1e-60") and df == 126 else "FAIL",
                    exact_beta=[2, 3], exact_covariance="I / 126", max_absolute_error=float(error))


def fit(module_path, case, threads):
    for key in tuple(os.environ):
        if key.startswith("XHDFE_"):
            del os.environ[key]
    os.environ["XHDFE_GPU_BACKEND"] = "cpu"
    os.environ["XHDFE_ABSORPTION_CACHE_MODE"] = "off"
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    model = core.HdfeRegressor(se_type="unadjusted", fit_intercept=False,
                               drop_singletons=False, tol=1e-11, max_iter=100000,
                               num_threads=threads, tolerance_mode="reghdfe-comparable")
    try:
        model.fit(case["y"], np.asfortranarray(case["x"]),
                  instruments=np.asfortranarray(case["z"]), endogenous_idx=case["endogenous"])
        return dict(coef=np.asarray(model.coef_).tolist(), covariance=np.asarray(model.covariance_).tolist(),
                    converged=bool(model.converged_), certified=bool(model.precision_certified_),
                    gpu_used=bool(model.gpu_used_), observations=int(model.nobs_),
                    df_resid=float(model.df_resid_), threads_used=int(model.threads_used_))
    except Exception as error:
        return dict(error_type=type(error).__name__, error=str(error), converged=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--module", type=Path)
    source.add_argument("--results", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    input_path = (args.module or args.results).resolve(strict=True)
    sources = (Path(__file__).resolve(), GENERATOR, input_path)
    hashes = {str(path): sha256(path) for path in sources}
    case = runpy.run_path(str(GENERATOR), run_name="large_weak_fixture")["make_cases"]()[CASE]
    ref = reference(case, 90)
    agreement = reference_agreement(ref, reference(case, 120))
    self_check = walsh_self_check()
    if args.module:
        observed = fit(input_path, case, args.threads)
        receipt_ok = (observed.get("converged") and observed.get("certified") and
                      observed.get("gpu_used") is False and observed.get("observations") == len(case["y"]) and
                      observed.get("df_resid") == ref[3])
    else:
        receipt = json.loads(input_path.read_text())
        result = receipt["results"][CASE]
        observed = result.get("value", {})
        receipt_ok = result.get("status") == "ok" and observed.get("converged", False)
    assessed = grade(observed.get("coef"), observed.get("covariance"), ref)
    if not receipt_ok:
        assessed = dict(status="FAIL", reason="Missing successful fit or invalid lifecycle/backend/sample diagnostics")
    legacy = observed.get("oracle")
    legacy_grade = None if legacy is None else grade(legacy.get("coef"), legacy.get("covariance"), ref)
    custody = all(sha256(path) == hashes[str(path)] for path in sources)
    beta, variance, rss, df = ref
    report = dict(schema="xhdfe-large-weak-reference-v1", case=CASE, source_hashes=hashes,
                  custody_unchanged=custody, input_mode="fresh_module" if args.module else "historical_receipt",
                  scope="CPU comparable IV, conventional full covariance, no FE; no performance claim",
                  provenance_note=None if args.module else "Receipt does not seal historical fixture arrays or binary bytes; current generator/input hashes are recorded",
                  fixture={key: dict(shape=list(np.asarray(case[key]).shape),
                                     sha256=hashlib.sha256(np.ascontiguousarray(case[key], dtype=np.float64).tobytes()).hexdigest())
                           for key in ("x", "y", "z")},
                  method="MP90 on exact represented float64 inputs, checked at MP120; empirical precision check, not a formal enclosure",
                  limits=dict(coefficient_scaled=1e-9, diagonal_scaled_full_covariance=1e-8, arithmetic_allowance=0),
                  reference_mp90=dict(coef=[mp.nstr(v, 85) for v in beta],
                                      covariance=[[mp.nstr(variance[i, j], 85) for j in range(len(beta))] for i in range(len(beta))],
                                      rss=mp.nstr(rss, 85), df_resid=df),
                  precision_90_vs_120=agreement, walsh_reference_self_check=self_check,
                  observed=observed, estimator=assessed, historical_numpy_reference=legacy_grade,
                  status="PASS" if assessed["status"] == "PASS" and agreement["agrees"] and
                  self_check["status"] == "PASS" and custody else "FAIL")
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "case", "estimator", "precision_90_vs_120")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
