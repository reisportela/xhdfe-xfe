#!/usr/bin/env python3
"""Weak IV with distinct cluster partitions: independent full-covariance gate.

The zero-mean 2^-27 cases expose an inherited constant-variance error also
present in the frozen 9 October composition. Shifted means are controls.
The fixture, MP90/MP120 oracle and numerical limits are reused unchanged.
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
import iv_intercept_covariance_contract as oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    module = args.module.resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    for key in tuple(os.environ):
        if key.startswith("XHDFE_"):
            del os.environ[key]
    os.environ.update(XHDFE_GPU_BACKEND=args.backend, CUDA_CACHE_DISABLE="1")
    paths = [module, Path(__file__).resolve(), Path(oracle.__file__).resolve()]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)

    class Factory:
        def HdfeRegressor(self, **kwargs):
            self.last = core.HdfeRegressor(**kwargs)
            return self.last

    rows = []
    for means in ((0, 0), (1, -1)):
        for weight in ("analytic", "frequency"):
            case = oracle.fixture(2.0**-27, means, True, weight, "distinct")
            for mode in ("reghdfe-comparable", "xhdfe-fast"):
                factory = Factory()
                row = oracle.evaluate(factory, case, "multiway", mode)
                if hasattr(factory, "last"):
                    model = factory.last
                    row["gpu_used"] = bool(model.gpu_used_)
                    row["cluster_counts"] = list(model.cluster_counts_)
                    row["intersection_counts"] = list(model.cluster_combo_counts_)
                    backend_ok = row["gpu_used"] == (args.backend == "cuda")
                    partitions_ok = row["cluster_counts"] == [17, 23]
                    if not backend_ok or not partitions_ok:
                        row["status"] = "FAIL"
                        row.setdefault("failed_checks", []).extend(
                            key for key, ok in (("backend", backend_ok), ("partitions", partitions_ok)) if not ok)
                rows.append(row)
    custody = all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)] for path in paths)
    failures = sum(row["status"] != "PASS" for row in rows)
    report = dict(status="PASS" if custody and not failures else "FAIL",
                  classification="demonstrated estimator error inherited before the closure tranche",
                  backend=args.backend, cases=len(rows), passed=len(rows)-failures, failed=failures,
                  source_hashes=hashes, custody_unchanged=custody,
                  limits=dict(beta=1e-9, full_v_diagonal_scaled=1e-8), results=rows,
                  scope="Synthetic distinct17/23 partitions; zero means expose constant variance, shifted means control the transport")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "cases", "passed", "failed")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
