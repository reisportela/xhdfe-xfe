#!/usr/bin/env python3
"""Known-score IV + absorbed constant: full covariance including the intercept.

The original 24 fixtures exposed six errors after an accurate within-IV solve.
MP90 references use rational Walsh projections and the represented inputs.
Frequency weights and distinct cluster partitions additionally exercise SSC.
No estimator output enters the reference; every JSON destination is exclusive.
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
import mpmath as mp
import numpy as np


def fixture(delta, means=(0, 0), weighted=False, weight_type="analytic",
            partition="equivalent", noise=1.0):
    n = 512
    i = np.arange(n)
    a, b, c, v, d, e = (2.0 * ((i // 2**j) % 2) - 1 for j in range(6))
    x = np.column_stack((2*a + v + means[0], 2*a + delta*b + d + means[1]))
    z = np.column_stack((3*a + c, b))
    residual = noise * e
    y = x @ np.array([0.5, -0.25]) + 3 + residual
    cl1 = np.random.default_rng(91).integers(0, 17, n)
    cl2 = 3*cl1 + 2 if partition == "equivalent" else np.random.default_rng(314).integers(0, 23, n)
    weights = 1.0 + i // 64 if weighted else np.ones(n)
    return dict(x=x, y=y, z=z, residual=residual, weights=weights,
                fe=np.zeros(n, dtype=np.int64), cl1=cl1, cl2=cl2,
                score_numerator=9*a+3*c, score_direction=b,
                coef=np.array([0.5, -0.25, 3.0]), delta=float(delta),
                means=list(means), weighted=weighted, weight_type=weight_type,
                partition=partition, noise=float(noise))


def reference(case, vce, g_df="min", g_adj=True, k_adj=True, digits=90):
    n = len(case["y"])
    frequency = case["weighted"] and case["weight_type"] == "frequency"
    with mp.workdps(digits):
        weights = [mp.mpf(float(value)) for value in case["weights"]]
        residual = [mp.mpf(float(value)) for value in case["residual"]]
        scores = [[mp.mpf(float(case["score_numerator"][i]))/5 + case["means"][0],
                   mp.mpf(float(case["score_numerator"][i]))/5 +
                   mp.mpf(case["delta"])*int(case["score_direction"][i]) + case["means"][1],
                   mp.mpf(1)] for i in range(n)]
        gram = mp.matrix([[mp.fsum(weights[i]*scores[i][j]*scores[i][k] for i in range(n))
                           for k in range(3)] for j in range(3)])
        bread = mp.inverse(gram)
        effective = mp.fsum(weights) if frequency else mp.mpf(n)
        df = effective - 3
        rss = mp.fsum(weights[i]*residual[i]**2 for i in range(n))
        groups = []
        if vce == "unadjusted":
            covariance = (rss/df)*bread
            inference_df = df
        elif vce == "robust":
            factors = [(weights[i] if frequency else weights[i]**2)*residual[i]**2 for i in range(n)]
            meat = mp.matrix([[mp.fsum(factors[i]*scores[i][j]*scores[i][k] for i in range(n))
                               for k in range(3)] for j in range(3)])
            covariance = (effective/df if k_adj else 1)*bread*meat*bread.T
            inference_df = df
        else:
            partitions = [(1, [(int(g),) for g in case["cl1"]])]
            if vce == "multiway":
                partitions.extend([(1, [(int(g),) for g in case["cl2"]]),
                                   (-1, list(zip(map(int, case["cl1"]), map(int, case["cl2"]))))])
            meat = mp.zeros(3)
            for index, (sign, ids) in enumerate(partitions):
                totals = {}
                for i, group in enumerate(ids):
                    row = totals.setdefault(group, [mp.mpf(0), mp.mpf(0), mp.mpf(0)])
                    for j in range(3):
                        row[j] += weights[i]*residual[i]*scores[i][j]
                count = len(totals)
                groups.append(count)
                factor = mp.mpf(sign)
                if g_adj and g_df == "conventional":
                    factor *= mp.mpf(count)/(count-1)
                meat += factor*mp.matrix([[mp.fsum(row[j]*row[k] for row in totals.values())
                                            for k in range(3)] for j in range(3)])
            minimum = min(groups[:2]) if vce == "multiway" else groups[0]
            factor = (effective-1)/df if k_adj else mp.mpf(1)
            if g_adj and g_df == "min":
                factor *= mp.mpf(minimum)/(minimum-1)
            covariance = factor*bread*meat*bread.T
            inference_df = min(df, minimum-1)
        reported_rss = rss if frequency or not case["weighted"] else rss*n/mp.fsum(weights)
        # Distinct partitions are predeclared. Verify positive definiteness
        # before grading, without deriving a PSD adjustment from the estimator.
        if case["noise"]:
            mp.cholesky(covariance)
        return dict(b=[mp.mpf("0.5"), mp.mpf("-0.25"), mp.mpf(3)],
                    V=covariance, rss=reported_rss, df=inference_df,
                    model_df=df, effective_n=effective, groups=groups)


def serialized_reference(ref):
    return dict(b=[mp.nstr(value, 85) for value in ref["b"]],
                V=[[mp.nstr(ref["V"][j, k], 85) for k in range(3)] for j in range(3)],
                rss=mp.nstr(ref["rss"], 85), df=mp.nstr(ref["df"], 85),
                model_df=mp.nstr(ref["model_df"], 85), groups=ref["groups"])


def evaluate(cpp, case, vce, mode, g_df="min", g_adj=True):
    row = {key: case[key] for key in ("delta", "means", "weighted", "weight_type", "partition", "noise")}
    row.update(vce=vce, mode=mode, g_df=g_df, g_adj=g_adj)
    try:
        ref = reference(case, vce, g_df, g_adj)
        ref120 = reference(case, vce, g_df, g_adj, digits=120)
        with mp.workdps(120):
            if case["noise"]:
                agreement = max(abs(ref["V"][j,k]-ref120["V"][j,k]) /
                                mp.sqrt(ref120["V"][j,j]*ref120["V"][k,k])
                                for j in range(3) for k in range(3))
                if agreement > mp.mpf("1e-60"):
                    raise AssertionError("MP90 and MP120 references disagree")
                row["reference_stability"] = float(agreement)
        model = cpp.HdfeRegressor(num_threads=2, fit_intercept=True,
                                  se_type="cluster" if vce == "multiway" else vce,
                                  tolerance_mode=mode, ssc_g_df=g_df, ssc_g_adj=g_adj)
        args = dict(fes=[case["fe"]], instruments=case["z"], endogenous_idx=[0, 1])
        if case["weighted"]:
            args["weights"] = case["weights"]
            args["fweights"] = case["weight_type"] == "frequency"
        if vce in ("cluster", "multiway"):
            args["clusters"] = case["cl1"] if vce == "cluster" else np.column_stack((case["cl1"],case["cl2"]))
        model.fit(case["y"], case["x"], **args)
        beta, covariance = np.asarray(model.coef_), np.asarray(model.covariance_)
        if not (np.isfinite(beta).all() and np.isfinite(covariance).all() and
                np.isfinite(model.rss_) and np.isfinite(model.residuals_).all()):
            raise AssertionError("non-finite output")
        if beta.shape != (3,) or covariance.shape != (3,3):
            raise AssertionError("incorrect full coefficient/covariance dimensions")
        with mp.workdps(120):
            db = max(abs(mp.mpf(float(beta[j]))-ref["b"][j])/max(1,abs(ref["b"][j])) for j in range(3))
            dv = max(abs(mp.mpf(float(covariance[j,k]))-ref["V"][j,k]) /
                     mp.sqrt(ref["V"][j,j]*ref["V"][k,k])
                     for j in range(3) for k in range(3)) if case["noise"] else None
        row.update(coef=beta.tolist(), covariance=covariance.tolist(), reference=serialized_reference(ref),
                   beta_scaled_error=float(db), full_v_scaled_error=None if dv is None else float(dv),
                   residual_error=float(np.max(np.abs(np.asarray(model.residuals_)-case["residual"]))),
                   rss_error=abs(float(model.rss_)-float(ref["rss"])),
                   df=float(model.df_resid_), nobs=float(model.nobs_),
                   converged=bool(model.converged_), precision_certified=bool(model.precision_certified_))
        checks = dict(beta=db<=mp.mpf("1e-9"), full_v=dv is None or dv<=mp.mpf("1e-8"),
                      residual=row["residual_error"]<=1e-7, rss=row["rss_error"]<=1e-8,
                      df=row["df"]==float(ref["df"]), nobs=row["nobs"]==float(ref["effective_n"]),
                      convergence=row["converged"] and row["precision_certified"])
        row["grade_scope"] = "full positive covariance" if dv is not None else "perfect-fit finite-output guard; zero-V relative metric undefined"
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
    for delta in (2.0**-15, 2.0**-27):
        for means in ((0,0), (1,-1)):
            for weighted in (False,True):
                for vce in ("robust", "cluster", "multiway"):
                    rows.append(evaluate(cpp, fixture(delta,means,weighted), vce, args.mode))
            for vce in ("robust", "cluster", "multiway"):
                rows.append(evaluate(cpp, fixture(delta,means,True,"frequency"), vce, args.mode))
    for weight_type in ("analytic", "frequency"):
        for g_df in ("min", "conventional"):
            for g_adj in (False,True):
                case = fixture(2.0**-15,(1,-1),True,weight_type,"distinct")
                rows.append(evaluate(cpp,case,"multiway",args.mode,g_df,g_adj))
        case = fixture(2.0**-27,(1,-1),True,weight_type,noise=0.0)
        rows.append(evaluate(cpp,case,"unadjusted",args.mode))
    custody = all(hashlib.sha256(path.read_bytes()).hexdigest()==hashes[str(path)] for path in paths)
    failures = sum(row["status"]!="PASS" for row in rows)
    report = dict(classification="demonstrated estimator error: precision lost in post-FE intercept covariance",
                  module=str(module_path), source_hashes=hashes, custody_unchanged=custody,
                  limits=dict(beta=1e-9, full_v_diagonal_scaled=1e-8, residual_absolute=1e-7, rss_absolute=1e-8),
                  cases=len(rows), passed=len(rows)-failures, failed=failures,
                  status="PASS" if custody and not failures else "FAIL", results=rows)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:report[key] for key in ("status","cases","passed","failed")}))
    return 0 if report["status"]=="PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
