#!/usr/bin/env python3
"""Independent Walsh IV contracts; writes one new JSON report exclusively.

The represented instrument w + delta*t spans w,t exactly for the dyadic
deltas below. Thus x=t+v projects to t, b=(2,3), and the structural residual
is e. Oracle scores never use a solve involving the ill-conditioned Z.
CPU: 128 rows, no FE. CUDA: 3072 rows and eight balanced level FEs, with
explicit real-GPU-use checks; no-FE CUDA is an unsupported API combination.

Limits: the published coefficient/full-V contract, the requested residual
limit 1e-7, and the 1e-8 absolute RSS limit in n1_iv_cache_contract.py.
No timing or performance conclusions are produced.
The pre-remediation 22sep2026 IV implementation failed this contract.
The isolated universal-QR candidate was rejected for incomplete
precision and runtime regressions; do not relax these oracle limits.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


MODES = ("xhdfe-fast", "reghdfe-comparable")
DELTAS = (1.0, 2.0**-10, 2.0**-20, 2.0**-25, 2.0**-27)


def fixture(delta, weighted, backend, scale=1.0, permuted=False, multiple=False,
            zero_weight=False):
    n = 128 if backend == "cpu" else 3072
    i = np.arange(n)
    w, t, v, e, a, b = (2.0 * ((i // (2**k)) % 2) - 1.0 for k in range(6))
    score = np.column_stack((w, t))
    actual = np.column_stack((w, t + v))
    beta = np.array([2.0, 3.0])
    instrument = (scale * (w + delta * t))[:, None]
    endogenous = [1]
    if multiple:
        score = np.column_stack((score, a))
        actual = np.column_stack((actual, a + b))
        beta = np.array([2.0, 3.0, -0.5])
        instrument = np.column_stack((instrument, a))
        endogenous = [1, 2]
    weights = 1.0 + i // 16 if weighted else np.ones(n)
    if zero_weight:
        weights[:16] = 0.0
    fes = [] if backend == "cpu" else [i // 384]
    y = actual @ beta + e
    if fes:
        y += 0.5 * fes[0]
    order = np.random.default_rng(23).permutation(n) if permuted else i
    return dict(y=y[order], X=actual[order], Z=instrument[order],
                score=score[order], beta=beta, residual=e[order],
                weights=weights[order] if weighted or zero_weight else None,
                fes=[f[order] for f in fes], clusters=(i % 7)[order],
                endogenous_idx=endogenous, fe_rank=0 if not fes else 8)


def oracle(data, vce):
    n, p = data["X"].shape
    raw_weights = data["weights"]
    weights = np.ones(n) if raw_weights is None else raw_weights * (n / raw_weights.sum())
    score, actual, residual = data["score"], data["X"], data["residual"]
    gram = score.T @ (weights[:, None] * actual)
    # Small, well-conditioned score design, independent of the instrument QR.
    bread = np.linalg.inv(gram)
    df = n - p - data["fe_rank"]
    rss = float(weights @ (residual * residual))
    if vce == "unadjusted":
        covariance = (rss / df) * bread
        inference_df = df
    else:
        scores = score * (weights * residual)[:, None]
        if vce == "robust":
            meat = scores.T @ scores
            correction = n / df
            inference_df = df
        else:
            ids = np.unique(data["clusters"])
            totals = np.stack([scores[data["clusters"] == g].sum(axis=0) for g in ids])
            meat = totals.T @ totals
            correction = len(ids) / (len(ids) - 1) * (n - 1) / df
            inference_df = len(ids) - 1
        covariance = correction * bread @ meat @ bread.T
    return covariance, rss, inference_df


def fit_arguments(data, vce):
    kwargs = dict(fes=data["fes"], weights=data["weights"],
                  instruments=np.asfortranarray(data["Z"]),
                  endogenous_idx=data["endogenous_idx"])
    if vce == "cluster":
        kwargs["clusters"] = data["clusters"]
    return kwargs


def evaluate(cpp, data, backend, mode, vce, label):
    row = dict(label, mode=mode, vce=vce)
    try:
        retained = np.arange(data["y"].size)
        reference = data
        if data["weights"] is not None and np.any(data["weights"] == 0):
            retained = np.flatnonzero(data["weights"] > 0)
            reference = dict(data)
            for key in ("y", "X", "Z", "score", "residual", "weights", "clusters"):
                reference[key] = data[key][retained]
            reference["fes"] = [fe[retained] for fe in data["fes"]]
        reference_v, reference_rss, reference_df = oracle(reference, vce)
        model = cpp.HdfeRegressor(num_threads=2, fit_intercept=False,
                                  drop_singletons=False, se_type=vce,
                                  tolerance_mode=mode)
        model.fit(data["y"], np.asfortranarray(data["X"]), **fit_arguments(data, vce))
        if not np.array_equal(model.sample_index_, retained):
            raise AssertionError("estimation sample differs from positive-weight rows")
        beta = np.asarray(model.coef_)
        covariance = np.asarray(model.covariance_)
        residual = np.asarray(model.residuals_)
        stderr = np.asarray(model.stderr_)
        arrays = (beta, covariance, residual, stderr,
                  np.array([model.rss_, model.df_resid_]))
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise AssertionError("non-finite fitted output")
        if beta.shape != data["beta"].shape or covariance.shape != reference_v.shape:
            raise AssertionError("coefficient/covariance shape differs from oracle")
        vscale = np.sqrt(np.outer(np.diag(reference_v), np.diag(reference_v)))
        row.update(coef=beta.tolist(),
                   scaled_beta_error=float(np.max(np.abs(beta - data["beta"]) /
                                                   np.maximum(1.0, np.abs(data["beta"])))),
                   scaled_v_error=float(np.max(np.abs(covariance - reference_v) / vscale)),
                   relative_se_error=float(np.max(np.abs(stderr - np.sqrt(np.diag(reference_v))) /
                                                   np.sqrt(np.diag(reference_v)))),
                   residual_error=float(np.max(np.abs(residual - reference["residual"]))),
                   rss_error=abs(float(model.rss_) - reference_rss),
                   df=float(model.df_resid_), reference_df=float(reference_df),
                   converged=bool(model.converged_),
                   precision_certified=bool(model.precision_certified_),
                   gpu_used=bool(model.gpu_used_),
                   gpu_status_code=int(model.gpu_status_code_),
                   iterations=int(model.num_iterations_), threads_used=int(model.threads_used_))
        checks = dict(coef=row["scaled_beta_error"] <= 1e-9,
                      covariance=row["scaled_v_error"] <= 1e-8,
                      stderr=row["relative_se_error"] <= 1e-8,
                      residual=row["residual_error"] <= 1e-7,
                      rss=row["rss_error"] <= 1e-8,
                      df=row["df"] == reference_df,
                      convergence=row["converged"] and row["precision_certified"],
                      backend=row["gpu_used"] == (backend == "cuda"))
        row["failed_checks"] = [key for key, passed in checks.items() if not passed]
        row["status"] = "PASS" if all(checks.values()) else "FAIL"
    except Exception as error:
        row.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    return row


def failed_refit(cpp, backend, mode):
    row = dict(kind="failed_refit_clears_state", mode=mode)
    data = fixture(1.0, False, backend)
    try:
        model = cpp.HdfeRegressor(num_threads=2, fit_intercept=False,
                                  drop_singletons=False, tolerance_mode=mode)
        args = fit_arguments(data, "unadjusted")
        model.fit(data["y"], data["X"], **args)
        if not model.converged_:
            raise AssertionError("initial valid fit did not converge")
        # Redundancy alone is valid. A span contained in the included exogenous
        # column genuinely fails to identify the endogenous coefficient.
        args["instruments"] = np.column_stack((data["X"][:, 0], data["X"][:, 0]))
        try:
            model.fit(data["y"], data["X"], **args)
        except RuntimeError as error:
            row["expected_error"] = str(error)
            if "rank deficient" not in str(error):
                raise AssertionError("unexpected failure reason") from error
        else:
            raise AssertionError("underidentified first stage was accepted")
        if (np.asarray(model.coef_).size or np.asarray(model.covariance_).size or
                np.asarray(model.residuals_).size or model.converged_ or
                model.precision_certified_ or model.lifecycle_state_ != "failed"):
            raise AssertionError("failed refit retained consumable estimation state")
        args = fit_arguments(data, "unadjusted")
        model.fit(data["y"], data["X"], **args)
        if np.max(np.abs(np.asarray(model.coef_) - data["beta"]) /
                  np.maximum(1.0, np.abs(data["beta"]))) > 1e-9:
            raise AssertionError("valid refit after failure differs from oracle")
        row["status"] = "PASS"
    except Exception as error:
        row.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    module_path = args.module.resolve(strict=True)
    if args.out.exists():
        raise FileExistsError(f"report already exists: {args.out}")
    # A fresh process must not inherit experiment switches or cache writers.
    for key in tuple(os.environ):
        if key.startswith("XHDFE_"):
            del os.environ[key]
    os.environ["XHDFE_GPU_BACKEND"] = args.backend
    module_hash = hashlib.sha256(module_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path)
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    rows = []
    if args.backend == "cpu":
        # 120 cells; every mode/weight/delta/VCE crosses the original order and
        # a fixed permutation with alternating dyadic instrument rescaling.
        for d, delta in enumerate(DELTAS):
            for weighted in (False, True):
                for mode in MODES:
                    for vce in ("unadjusted", "robust", "cluster"):
                        for permuted in (False, True):
                            scale = (2.0**(-10 if d % 2 == 0 else 10)) if permuted else 1.0
                            data = fixture(delta, weighted, "cpu", scale, permuted)
                            label = dict(kind="dyadic", delta=delta, weighted=weighted,
                                         scale=scale, permuted=permuted)
                            rows.append(evaluate(cpp, data, "cpu", mode, vce, label))
        for mode in MODES:
            for vce in ("unadjusted", "robust"):
                data = fixture(2.0**-25, False, "cpu", multiple=True)
                rows.append(evaluate(cpp, data, "cpu", mode, vce,
                                     dict(kind="multiple_endogenous", delta=2.0**-25)))
            data = fixture(2.0**-25, False, "cpu", zero_weight=True)
            rows.append(evaluate(cpp, data, "cpu", mode, "unadjusted",
                                 dict(kind="zero_weight_block", delta=2.0**-25)))
    else:
        # Separate CUDA fixture: FE rank eight and df=N-2-8 are known exactly.
        for delta in (1.0, 2.0**-25, 2.0**-27):
            for weighted in (False, True):
                for mode in MODES:
                    for vce in ("unadjusted", "robust", "cluster"):
                        data = fixture(delta, weighted, "cuda")
                        rows.append(evaluate(cpp, data, "cuda", mode, vce,
                                             dict(kind="balanced_fe_cuda", delta=delta,
                                                  weighted=weighted, fe_rank=8)))
    for mode in MODES:
        rows.append(failed_refit(cpp, args.backend, mode))
    unchanged = hashlib.sha256(module_path.read_bytes()).hexdigest() == module_hash
    failures = sum(row["status"] != "PASS" for row in rows)
    report = dict(module=str(module_path), module_sha256=module_hash,
                  worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  module_unchanged=unchanged, backend=args.backend,
                  limits=dict(coef=1e-9, diagonal_scaled_full_v=1e-8,
                              relative_se=1e-8, residual_absolute=1e-7, rss_absolute=1e-8),
                  unsupported=["CUDA without an absorbed FE; not substituted by CPU"],
                  oracle="exact Walsh score space; weighted structural-residual sandwich",
                  cases=len(rows), passed=len(rows)-failures, failed=failures,
                  status="PASS" if not failures and unchanged else "FAIL", results=rows)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "cases", "passed", "failed")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
