#!/usr/bin/env python3
"""Known-score IV, absorbed constant and an omitted collinear exogenous regressor.

Reuses the dyadic design of iv_intercept_covariance_contract.fixture and adds
an exogenous Walsh regressor c together with its exact multiple 2c, so one of
the two is omitted. The instrument span becomes {3a+c, b, c, 1} = {a, b, c, 1},
which removes v and d exactly, so the identified scores are
[2a+m0, 2a+db+m1, c, 1] and the MP90 reference is explicit on the represented
inputs. Grading maps the kept duplicate to the c slot with its scale; the
omitted row and column of the reported covariance must be exactly zero and
its coefficient zero. The retained wide IV covariance (OlsWideCovariance) is
taken for every IV fit with a recovered intercept whose wide matrix spans the
kept slopes plus the constant; the branch itself is not observable from
Python, so this contract verifies its preconditions and its output.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpmath as mp
import numpy as np
from iv_intercept_covariance_contract import fixture  # noqa: E402


def omitted_case(delta, means=(0, 0), weighted=False, weight_type="analytic",
                 partition="equivalent", noise=1.0, order="c_first"):
    case = fixture(delta, means, weighted, weight_type, partition, noise)
    i = np.arange(len(case["y"]))
    if partition == "distinct":
        # The frozen fixture's second partition (seed 314, 23 groups) gives an
        # indefinite CGM reference once the c score is added; this partition
        # keeps the reference positive definite for every SSC variant, so the
        # case grades the omitted column, not the PSD repair.
        case["cl2"] = np.random.default_rng(300).integers(0, 23, len(case["y"]))
    c = 2.0 * ((i // 4) % 2) - 1
    scales = [1.0, 2.0] if order == "c_first" else [2.0, 1.0]
    case["x"] = np.column_stack((case["x"], scales[0] * c, scales[1] * c))
    case["extra_scales"] = scales
    case["c"] = c
    case["order"] = order
    return case


def reference(case, vce, g_df="min", g_adj=True, digits=90):
    n = len(case["y"])
    i = np.arange(n)
    a = 2.0 * (i % 2) - 1
    b = 2.0 * ((i // 2) % 2) - 1
    frequency = case["weighted"] and case["weight_type"] == "frequency"
    with mp.workdps(digits):
        weights = [mp.mpf(float(value)) for value in case["weights"]]
        residual = [mp.mpf(float(value)) for value in case["residual"]]
        scores = [[2 * int(a[k]) + case["means"][0],
                   2 * int(a[k]) + mp.mpf(case["delta"]) * int(b[k]) + case["means"][1],
                   mp.mpf(int(case["c"][k])), mp.mpf(1)] for k in range(n)]
        q = 4
        gram = mp.matrix([[mp.fsum(weights[k] * scores[k][j] * scores[k][l] for k in range(n))
                           for l in range(q)] for j in range(q)])
        bread = mp.inverse(gram)
        effective = mp.fsum(weights) if frequency else mp.mpf(n)
        df = effective - q
        rss = mp.fsum(weights[k] * residual[k] ** 2 for k in range(n))
        groups = []
        if vce == "unadjusted":
            covariance = (rss / df) * bread
            inference_df = df
        elif vce == "robust":
            factors = [(weights[k] if frequency else weights[k] ** 2) * residual[k] ** 2 for k in range(n)]
            meat = mp.matrix([[mp.fsum(factors[k] * scores[k][j] * scores[k][l] for k in range(n))
                               for l in range(q)] for j in range(q)])
            covariance = (effective / df) * bread * meat * bread.T
            inference_df = df
        else:
            partitions = [(1, [(int(g),) for g in case["cl1"]])]
            if vce == "multiway":
                partitions.extend([(1, [(int(g),) for g in case["cl2"]]),
                                   (-1, list(zip(map(int, case["cl1"]), map(int, case["cl2"]))))])
            meat = mp.zeros(q)
            for sign, ids in partitions:
                totals = {}
                for k, group in enumerate(ids):
                    row = totals.setdefault(group, [mp.mpf(0)] * q)
                    for j in range(q):
                        row[j] += weights[k] * residual[k] * scores[k][j]
                count = len(totals)
                groups.append(count)
                factor = mp.mpf(sign)
                if g_adj and g_df == "conventional":
                    factor *= mp.mpf(count) / (count - 1)
                meat += factor * mp.matrix([[mp.fsum(row[j] * row[l] for row in totals.values())
                                             for l in range(q)] for j in range(q)])
            minimum = min(groups[:2]) if vce == "multiway" else groups[0]
            factor = (effective - 1) / df
            if g_adj and g_df == "min":
                factor *= mp.mpf(minimum) / (minimum - 1)
            covariance = factor * bread * meat * bread.T
            inference_df = min(df, minimum - 1)
        reported_rss = rss if frequency or not case["weighted"] else rss * n / mp.fsum(weights)
        if case["noise"]:
            mp.cholesky(covariance)
        return dict(b=[mp.mpf("0.5"), mp.mpf("-0.25"), mp.mpf(0), mp.mpf(3)],
                    V=covariance, rss=reported_rss, df=inference_df, effective_n=effective, groups=groups)


def evaluate(cpp, case, vce, mode, g_df="min", g_adj=True):
    row = {key: case[key] for key in ("delta", "means", "weighted", "weight_type", "partition", "noise", "order")}
    row.update(vce=vce, mode=mode, g_df=g_df, g_adj=g_adj)
    try:
        ref = reference(case, vce, g_df, g_adj)
        model = cpp.HdfeRegressor(num_threads=2, fit_intercept=True,
                                  se_type="cluster" if vce == "multiway" else vce,
                                  tolerance_mode=mode, ssc_g_df=g_df, ssc_g_adj=g_adj)
        kwargs = dict(fes=[case["fe"]], instruments=case["z"], endogenous_idx=[0, 1])
        if case["weighted"]:
            kwargs["weights"] = case["weights"]
            kwargs["fweights"] = case["weight_type"] == "frequency"
        if vce in ("cluster", "multiway"):
            kwargs["clusters"] = case["cl1"] if vce == "cluster" else np.column_stack((case["cl1"], case["cl2"]))
        model.fit(case["y"], case["x"], **kwargs)
        beta, covariance = np.asarray(model.coef_), np.asarray(model.covariance_)
        omitted = [bool(flag) for flag in list(model.omitted_)]
        row.update(omitted=omitted, coef=beta.tolist())
        if not (np.isfinite(beta).all() and np.isfinite(covariance).all()):
            raise AssertionError("non-finite output")
        if beta.shape != (5,) or covariance.shape != (5, 5):
            raise AssertionError("incorrect full coefficient/covariance dimensions")
        extras = omitted[2:4]
        if sum(extras) != 1 or any(omitted[:2]) or (len(omitted) > 4 and any(omitted[4:])):
            raise AssertionError("exactly one of the duplicate regressors must be omitted")
        dropped = 2 + extras.index(True)
        kept = [0, 1, 2 + extras.index(False), 4]
        scale = {0: 1.0, 1: 1.0, kept[2]: case["extra_scales"][extras.index(False)], 4: 1.0}
        if beta[dropped] != 0.0 or np.any(covariance[dropped, :] != 0.0) or np.any(covariance[:, dropped] != 0.0):
            raise AssertionError("omitted regressor must report a zero coefficient and zero covariance")
        with mp.workdps(120):
            db = max(abs(mp.mpf(float(beta[kept[j]])) * scale[kept[j]] - ref["b"][j]) /
                     max(1, abs(ref["b"][j])) for j in range(4))
            dv = max(abs(mp.mpf(float(covariance[kept[j], kept[l]])) * scale[kept[j]] * scale[kept[l]] - ref["V"][j, l]) /
                     mp.sqrt(ref["V"][j, j] * ref["V"][l, l]) for j in range(4) for l in range(4)) if case["noise"] else None
        row.update(dropped_column=dropped, beta_scaled_error=float(db),
                   full_v_scaled_error=None if dv is None else float(dv),
                   residual_error=float(np.max(np.abs(np.asarray(model.residuals_) - case["residual"]))),
                   rss_error=abs(float(model.rss_) - float(ref["rss"])),
                   df=float(model.df_resid_), nobs=float(model.nobs_),
                   converged=bool(model.converged_), precision_certified=bool(model.precision_certified_))
        checks = dict(beta=db <= mp.mpf("1e-9"), full_v=dv is None or dv <= mp.mpf("1e-8"),
                      residual=row["residual_error"] <= 1e-7, rss=row["rss_error"] <= 1e-8,
                      df=row["df"] == float(ref["df"]), nobs=row["nobs"] == float(ref["effective_n"]),
                      convergence=row["converged"] and row["precision_certified"])
        row["failed_checks"] = [key for key, passed in checks.items() if not passed]
        row["status"] = "PASS" if all(checks.values()) else "FAIL"
    except Exception as error:
        row.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("xhdfe-fast", "reghdfe-comparable"), default="reghdfe-comparable")
    args = parser.parse_args()
    module_path = args.module.resolve(strict=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    for key in tuple(os.environ):
        if key.startswith("XHDFE_"):
            del os.environ[key]
    os.environ["XHDFE_GPU_BACKEND"] = "cpu"
    paths = (module_path, Path(__file__).resolve())
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path)
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    rows = []
    for order in ("c_first", "twice_first"):
        for delta in (2.0 ** -15, 2.0 ** -27):
            for means in ((0, 0), (1, -1)):
                for weighted in (False, True):
                    for vce in ("robust", "cluster", "multiway"):
                        rows.append(evaluate(cpp, omitted_case(delta, means, weighted, order=order), vce, args.mode))
                for vce in ("robust", "cluster", "multiway"):
                    rows.append(evaluate(cpp, omitted_case(delta, means, True, "frequency", order=order), vce, args.mode))
        for weight_type in ("analytic", "frequency"):
            for g_df in ("min", "conventional"):
                for g_adj in (False, True):
                    case = omitted_case(2.0 ** -15, (1, -1), True, weight_type, "distinct", order=order)
                    rows.append(evaluate(cpp, case, "multiway", args.mode, g_df, g_adj))
            rows.append(evaluate(cpp, omitted_case(2.0 ** -27, (1, -1), True, weight_type, noise=0.0, order=order),
                                 "unadjusted", args.mode))
    custody = all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)] for path in paths)
    failures = sum(row["status"] != "PASS" for row in rows)
    report = dict(classification="missing coverage: omitted collinear regressor on the retained wide IV covariance path",
                  module=str(module_path), source_hashes=hashes, custody_unchanged=custody,
                  limits=dict(beta=1e-9, full_v_diagonal_scaled=1e-8, residual_absolute=1e-7, rss_absolute=1e-8),
                  cases=len(rows), passed=len(rows) - failures, failed=failures,
                  status="PASS" if custody and not failures else "FAIL", results=rows)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "cases", "passed", "failed")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
