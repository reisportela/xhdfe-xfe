#!/usr/bin/env python3
"""Stata IV covariance against the permanent MP90/MP120 Walsh oracle.

Uses synthetic data only. Every run has a new output directory and pins the
loaded ado/plugin, backend and full covariance. No package is installed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import mpmath as mp
import numpy as np
import pandas as pd

import iv_intercept_covariance_contract as oracle


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(package, out, stata, expected_version, backend, mode, weight,
             partition="equivalent"):
    case = oracle.fixture(2.0**-27, (0, 0), weight != "none",
                          "frequency" if weight == "fw" else "analytic", partition)
    ref = oracle.reference(case, "multiway")
    ref120 = oracle.reference(case, "multiway", digits=120)
    with mp.workdps(120):
        stability = max(abs(ref["V"][j, k] - ref120["V"][j, k]) /
                        mp.sqrt(ref120["V"][j, j]*ref120["V"][k, k])
                        for j in range(3) for k in range(3))
        assert stability <= mp.mpf("1e-60")
    out.mkdir()
    (out / "tmp").mkdir()
    pd.DataFrame(dict(y=case["y"], x1=case["x"][:, 0], x2=case["x"][:, 1],
                      z1=case["z"][:, 0], z2=case["z"][:, 1],
                      fe=case["fe"], cl1=case["cl1"], cl2=case["cl2"],
                      weight=case["weights"])).to_stata(
                          out / "synthetic.dta", write_index=False, version=118)
    names = ["b1", "b2", "cons"] + [f"v{j}{k}" for j in range(1, 4) for k in range(1, 4)]
    names += ["nobs", "df", "rss", "iterations", "gpu_used"]
    values = [f"(B[1,{j}])" for j in range(1, 4)]
    values += [f"(V[{j},{k}])" for j in range(1, 4) for k in range(1, 4)]
    values += ["(e(N))", "(e(df_r))", "(e(rss))", "(e(iterations))", "(e(gpu_used))"]
    weights = "" if weight == "none" else f"[{weight}=weight]"
    cuda_check = "" if backend == "cpu" else '''assert "`e(gpu_backend)'"=="cuda"
assert "`e(gpu_status)'"=="used"
'''
    source = f'''version 16
clear all
set more off
set processors 2
adopath ++ "{package}"
discard
findfile xhdfe.ado
assert `"`r(fn)'"' == "{package}/xhdfe.ado"
findfile xhdfe.plugin
assert `"`r(fn)'"' == "{package}/xhdfe.plugin"
use "synthetic.dta", clear
quietly xhdfe y x1 x2 {weights}, absorb(fe) endogenous(x1 x2) instruments(z1 z2) ///
    vce(cluster cl1 cl2) tolerance(1e-8) tolerancemode({mode}) numthreads(2) gpubackend({backend})
assert "`e(version)'"=="{expected_version}"
assert e(converged)==1 & e(gpu_used)=={int(backend == 'cuda')}
{cuda_check}matrix B=e(b)
matrix V=e(V)
local names : colnames B
assert "`names'"=="x1 x2 _cons"
assert colsof(B)==3 & rowsof(V)==3 & colsof(V)==3
postfile P double({' '.join(names)}) using "estimates.dta"
post P {' '.join(values)}
postclose P
display "STATA_IV_ORACLE_PASS"
'''
    (out / "fit.do").write_text(source)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("XHDFE_") and k != "LD_LIBRARY_PATH"}
    env.update(STATATMP=str(out / "tmp"), TMPDIR=str(out / "tmp"),
               CUDA_CACHE_PATH=str(out / "cuda_cache"), OMP_DYNAMIC="FALSE",
               PYTHONDONTWRITEBYTECODE="1")
    row = dict(backend=backend, mode=mode, weight=weight, partition=partition,
               reference=oracle.serialized_reference(ref), reference_stability=float(stability),
               do_sha256=sha(out / "fit.do"), data_sha256=sha(out / "synthetic.dta"))
    try:
        with (out / "process.log").open("x") as log:
            result = subprocess.run([str(stata), "-q", "-b", "do", "fit.do"],
                cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=420)
        log = (out / "fit.log").read_text(errors="replace")
        if (result.returncode != 0 or "\nSTATA_IV_ORACLE_PASS" not in log or
                re.search(r"^r\(\d+\);", log, re.M)):
            raise AssertionError("Stata failed; inspect the preserved fit.log")
        records = pd.read_stata(out / "estimates.dta")
        assert len(records) == 1
        record = records.iloc[0]
        b = record[["b1", "b2", "cons"]].to_numpy(dtype=float)
        v = record[[f"v{j}{k}" for j in range(1, 4) for k in range(1, 4)]].to_numpy(dtype=float).reshape(3, 3)
        assert np.isfinite(b).all() and np.isfinite(v).all()
        with mp.workdps(120):
            db = max(abs(mp.mpf(float(b[j])) - ref["b"][j]) / max(1, abs(ref["b"][j])) for j in range(3))
            dv = max(abs(mp.mpf(float(v[j, k])) - ref["V"][j, k]) /
                     mp.sqrt(ref["V"][j, j]*ref["V"][k, k]) for j in range(3) for k in range(3))
        checks = dict(beta=bool(db <= mp.mpf("1e-9")), full_v=bool(dv <= mp.mpf("1e-8")),
                      nobs=bool(record.nobs == float(ref["effective_n"])),
                      df=bool(record.df == float(ref["df"])),
                      rss=bool(abs(record.rss - float(ref["rss"])) <= 1e-8))
        row.update(coef=b.tolist(), covariance=v.tolist(), beta_scaled_error=float(db),
                   full_v_scaled_error=float(dv), nobs=float(record.nobs), df=float(record.df),
                   rss=float(record.rss), iterations=float(record.iterations),
                   gpu_used=int(record.gpu_used), checks=checks,
                   status="PASS" if all(checks.values()) else "FAIL")
    except Exception as error:
        row.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stata", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--partition", choices=("equivalent", "distinct"), default="equivalent")
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    out = args.output.resolve()
    paths = [package / "xhdfe.ado", package / "xhdfe.plugin", Path(__file__).resolve(), Path(oracle.__file__).resolve()]
    hashes = {str(path): sha(path) for path in paths}
    out.mkdir(exist_ok=False)
    rows = []
    for backend in ("cpu", "cuda"):
        for mode in ("reghdfe-comparable", "xhdfe-fast"):
            for weight in (("none", "aw", "fw") if args.partition == "equivalent" else ("aw", "fw")):
                row = run_case(package, out / f"{backend}_{mode}_{weight}",
                               args.stata.resolve(strict=True), args.expected_version, backend, mode, weight,
                               args.partition)
                rows.append(row)
                print(backend, mode, weight, row["status"], flush=True)
    custody = all(sha(path) == hashes[str(path)] for path in paths)
    report = dict(status="PASS" if custody and all(r["status"] == "PASS" for r in rows) else "FAIL",
                  cases=len(rows), passed=sum(r["status"] == "PASS" for r in rows),
                  source_hashes=hashes, custody_unchanged=custody,
                  scope=f"IV {args.partition} cluster partitions, weak direction 2^-27, full covariance including constant",
                  limits=dict(beta=1e-9, full_v_diagonal_scaled=1e-8), results=rows)
    with (out / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "cases", "passed")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
