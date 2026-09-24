#!/usr/bin/env python3
"""Run the CPU contract's independent IV/intercept oracles with real CUDA use."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelFactory:
    def __init__(self, cpp):
        self.cpp = cpp
        self.last = None

    def HdfeRegressor(self, **kwargs):
        self.last = self.cpp.HdfeRegressor(**kwargs)
        return self.last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("xhdfe-fast", "reghdfe-comparable"), default="reghdfe-comparable")
    args = parser.parse_args()
    module_path = args.module.resolve(strict=True)
    helper_path = Path(__file__).resolve().with_name("iv_intercept_covariance_contract.py")
    if args.out.exists():
        raise FileExistsError(args.out)
    for key in tuple(os.environ):
        if key.startswith("XHDFE_"):
            del os.environ[key]
    os.environ["XHDFE_GPU_BACKEND"] = "cuda"
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    paths = (module_path, helper_path, Path(__file__).resolve())
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    contract = load(helper_path, "iv_intercept_covariance_contract")
    cpp = load(module_path, "py_hdfe_v11")
    cases = []
    for delta in (2.0**-15, 2.0**-27):
        for means in ((0,0), (1,-1)):
            for weighted in (False,True):
                for vce in ("robust", "cluster", "multiway"):
                    cases.append((contract.fixture(delta,means,weighted),vce,"min",True))
            for vce in ("robust", "cluster", "multiway"):
                cases.append((contract.fixture(delta,means,True,"frequency"),vce,"min",True))
    for weight_type in ("analytic", "frequency"):
        for g_df in ("min", "conventional"):
            for g_adj in (False,True):
                cases.append((contract.fixture(2.0**-15,(1,-1),True,weight_type,"distinct"),
                              "multiway",g_df,g_adj))
        cases.append((contract.fixture(2.0**-27,(1,-1),True,weight_type,noise=0.0),
                      "unadjusted","min",True))
    rows = []
    for case, vce, g_df, g_adj in cases:
        factory = ModelFactory(cpp)
        row = contract.evaluate(factory,case,vce,args.mode,g_df,g_adj)
        model = factory.last
        row["gpu_used"] = bool(model is not None and model.gpu_used_)
        row["gpu_status_code"] = None if model is None else int(model.gpu_status_code_)
        row["gpu_attempted"] = bool(model is not None and model.gpu_attempted_)
        if not row["gpu_used"]:
            row["status"] = "FAIL"
            row.setdefault("failed_checks",[]).append("real_cuda_use")
        rows.append(row)
    custody = all(hashlib.sha256(path.read_bytes()).hexdigest()==hashes[str(path)] for path in paths)
    failed = sum(row["status"]!="PASS" for row in rows)
    report = dict(module=str(module_path), source_hashes=hashes, custody_unchanged=custody,
                  backend="cuda", oracle="unchanged iv_intercept_covariance_contract.py MP90/MP120",
                  limits=dict(beta=1e-9, full_v_diagonal_scaled=1e-8, residual_absolute=1e-7, rss_absolute=1e-8),
                  cases=len(rows), passed=len(rows)-failed, failed=failed,
                  status="PASS" if custody and not failed else "FAIL", results=rows)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:report[key] for key in ("status","cases","passed","failed")}))
    return 0 if report["status"]=="PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
