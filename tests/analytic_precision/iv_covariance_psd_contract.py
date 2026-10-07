#!/usr/bin/env python3
"""Independent MP90/MP120 oracle for the IV multiway covariance PSD rule.

Positive controls must preserve the sandwich; distinct, fixed partitions with
genuinely negative eigenvalues must apply the existing standardized-metric
projection, including its separate slope-block correction. No estimator result
enters the reference. The original intercept covariance contract is unchanged.
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
from iv_intercept_covariance_contract import fixture


def cases():
    for delta in (2.0**-15, 2.0**-27):
        for means in ((0, 0), (1, -1)):
            for weight in ("none", "analytic", "frequency"):
                for seed in (None, 2, 11):
                    case = fixture(delta, means, weight != "none", weight)
                    if seed is not None:
                        case["cl1"] = np.random.default_rng(seed).integers(0, 4, 512)
                        case["cl2"] = np.random.default_rng(seed+100).integers(0, 5, 512)
                    case.update(seed=seed, g_df="min", g_adj=True, k_adj=True)
                    yield case
    for weight in ("analytic", "frequency"):
        for g_adj in (False, True):
            for k_adj in (False, True):
                case = fixture(2.0**-27, (1, -1), True, weight)
                case["cl1"] = np.random.default_rng(2).integers(0, 4, 512)
                case["cl2"] = np.random.default_rng(102).integers(0, 5, 512)
                case.update(seed=2, g_df="conventional", g_adj=g_adj, k_adj=k_adj)
                yield case


def reference(case, digits=90):
    """Known Walsh IV scores, sandwich, and the documented eigenvalue rule."""
    with mp.workdps(digits):
        n = len(case["y"])
        w = [mp.mpf(float(v)) for v in case["weights"]]
        u = [mp.mpf(float(v)) for v in case["residual"]]
        q = [[mp.mpf(float(case["score_numerator"][i]))/5 + case["means"][0],
              mp.mpf(float(case["score_numerator"][i]))/5 +
              mp.mpf(case["delta"])*int(case["score_direction"][i]) + case["means"][1],
              mp.mpf(1)] for i in range(n)]
        bread = mp.inverse(mp.matrix([[mp.fsum(w[i]*q[i][j]*q[i][k] for i in range(n))
                                       for k in range(3)] for j in range(3)]))
        meat, counts = mp.zeros(3), []
        for sign, labels in ((1, [(int(g),) for g in case["cl1"]]),
                             (1, [(int(g),) for g in case["cl2"]]),
                             (-1, list(zip(map(int, case["cl1"]), map(int, case["cl2"]))))):
            groups = {}
            for i, label in enumerate(labels):
                row = groups.setdefault(label, [mp.mpf(0) for _ in range(3)])
                for j in range(3):
                    row[j] += w[i]*u[i]*q[i][j]
            count = len(groups)
            counts.append(count)
            factor = mp.mpf(sign)
            if case["g_adj"] and case["g_df"] == "conventional":
                factor *= mp.mpf(count)/(count-1)
            meat += factor*mp.matrix([[mp.fsum(row[j]*row[k] for row in groups.values())
                                       for k in range(3)] for j in range(3)])
        frequency = case["weighted"] and case["weight_type"] == "frequency"
        effective = mp.fsum(w) if frequency else mp.mpf(n)
        factor = (effective-1)/(effective-3) if case["k_adj"] else mp.mpf(1)
        if case["g_adj"] and case["g_df"] == "min":
            g = min(counts[:2])
            factor *= mp.mpf(g)/(g-1)
        raw = factor*bread*meat*bread.T
        scales = []
        sw = w if frequency else [mp.mpf(1)]*n
        mass = mp.fsum(sw)
        for j in range(2):
            x = [mp.mpf(float(v)) for v in case["x"][:, j]]
            mean = mp.fsum(sw[i]*x[i] for i in range(n))/mass
            scales.append(max(mp.sqrt(mp.fsum(sw[i]*(x[i]-mean)**2 for i in range(n))/(mass-1)), mp.mpf("0.001")))
        scales.append(mp.mpf(1))

        def project(v, d):
            scale = mp.diag(d)
            standardized = scale*((v+v.T)/2)*scale
            eigenvalues, vectors = mp.eigsy(standardized)
            fixed = any(value < 0 for value in eigenvalues)
            if not fixed:
                return v.copy(), eigenvalues, False
            positive = mp.diag([max(value, mp.mpf(0)) for value in eigenvalues])
            inverse = mp.diag([1/value for value in d])
            return inverse*vectors*positive*vectors.T*inverse, eigenvalues, True

        corrected, eig, full_fixed = project(raw, scales)
        slopes, slope_eig, slopes_fixed = project(raw[:2, :2], scales[:2])
        corrected[:2, :2] = slopes
        return dict(raw=raw, V=corrected, eigenvalues=eig, slope_eigenvalues=slope_eig,
                    fixed=full_fixed or slopes_fixed, full_fixed=full_fixed,
                    slopes_fixed=slopes_fixed, df=min(effective-3, min(counts[:2])-1))


def matrix_strings(matrix):
    return [[mp.nstr(matrix[i,j], 85) for j in range(matrix.cols)] for i in range(matrix.rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--mode", choices=("xhdfe-fast", "reghdfe-comparable"), required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    for name in tuple(os.environ):
        if name.startswith("XHDFE_"):
            del os.environ[name]
    os.environ["XHDFE_GPU_BACKEND"] = args.backend
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    module_path = args.module.resolve(strict=True)
    paths = (module_path, Path(__file__).resolve(), Path(__file__).with_name("iv_intercept_covariance_contract.py").resolve())
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path)
    cpp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpp)
    rows = []
    for case in cases():
        row = {key: case[key] for key in ("delta", "means", "weight_type", "seed", "g_df", "g_adj", "k_adj")}
        try:
            ref, check = reference(case), reference(case, 120)
            with mp.workdps(120):
                scale = [mp.sqrt(max(abs(check["raw"][j,j]), abs(check["V"][j,j]))) for j in range(3)]
                stability = max(abs(ref["V"][j,k]-check["V"][j,k])/(scale[j]*scale[k])
                                for j in range(3) for k in range(3))
                assert stability < mp.mpf("1e-60"), "MP90/MP120 disagreement"
                assert ref["fixed"] == (case["seed"] is not None), "control eigenvalue signs differ from specification"
                correction = max(abs(ref["V"][j,k]-ref["raw"][j,k])/(scale[j]*scale[k])
                                 for j in range(3) for k in range(3))
            model = cpp.HdfeRegressor(num_threads=2, fit_intercept=True, se_type="cluster",
                                     tolerance_mode=args.mode, ssc_g_df=case["g_df"],
                                     ssc_g_adj=case["g_adj"], ssc_k_adj=case["k_adj"])
            fitargs = dict(fes=[case["fe"]], instruments=case["z"], endogenous_idx=[0,1],
                           clusters=np.column_stack((case["cl1"], case["cl2"])))
            if case["weighted"]:
                fitargs.update(weights=case["weights"], fweights=case["weight_type"] == "frequency")
            model.fit(case["y"], case["x"], **fitargs)
            b, v = np.asarray(model.coef_), np.asarray(model.covariance_)
            assert np.isfinite(b).all() and np.isfinite(v).all(), "nonfinite result"
            with mp.workdps(120):
                error = max(abs(mp.mpf(float(v[j,k]))-ref["V"][j,k])/(scale[j]*scale[k])
                            for j in range(3) for k in range(3))
            db = float(np.max(np.abs(b-case["coef"])))
            gpu_used = bool(model.gpu_used_)
            checks = dict(covariance=error<=mp.mpf("1e-8"), coefficients=db<=1e-9,
                          convergence=bool(model.converged_ and model.precision_certified_),
                          df=float(model.df_resid_)==float(ref["df"]),
                          backend=gpu_used if args.backend == "cuda" else not gpu_used)
            row.update(status="PASS" if all(checks.values()) else "FAIL",
                       failed_checks=[k for k,v in checks.items() if not v],
                       covariance=v.tolist(), coef=b.tolist(), beta_error=db,
                       covariance_scaled_error=float(error), reference_stability=float(stability),
                       correction_size=float(correction), reference_fixed=ref["fixed"],
                       detects_disabled_projection=bool(correction > mp.mpf("1e-8")),
                       gpu_used=gpu_used, gpu_status_code=int(model.gpu_status_code_),
                       raw_reference=matrix_strings(ref["raw"]), reference=matrix_strings(ref["V"]),
                       raw_standardized_eigenvalues=[mp.nstr(x,85) for x in ref["eigenvalues"]])
        except Exception as error:
            row.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        rows.append(row)
    custody = all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in paths)
    failed = sum(row["status"] != "PASS" for row in rows)
    # Large regressor offsets can make a real correction small in this norm.
    # Require the battery, rather than every offset case, to detect a disabled
    # projection. The numerical acceptance limit remains 1e-8 in every row.
    detectable_controls = sum(row.get("detects_disabled_projection", False) for row in rows)
    assert detectable_controls >= 12, "insufficient controls against disabled PSD correction"
    report = dict(status="PASS" if custody and not failed else "FAIL", cases=len(rows),
                  passed=len(rows)-failed, failed=failed, backend=args.backend, mode=args.mode,
                  limits=dict(beta_absolute=1e-9, covariance_diagonal_scaled=1e-8),
                  source_hashes=hashes, custody_unchanged=custody,
                  detectable_negative_controls=detectable_controls, results=rows)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:report[k] for k in ("status", "cases", "passed", "failed")}))
    return 1 if failed or not custody else 0


if __name__ == "__main__":
    raise SystemExit(main())
