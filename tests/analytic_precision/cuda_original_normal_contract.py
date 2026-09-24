#!/usr/bin/env python3
"""Original-model normal equations: analytic positive cases and K6 containment."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--known-cases", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists()
    os.environ["XHDFE_GPU_BACKEND"] = "cuda"
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    i = np.arange(4096)
    a, b = (i // 8) % 16, (i // 128) % 8
    u, v, noise = (2*((i // 2**k) % 2)-1 for k in range(3))
    base_variance = .25**2/(len(i)-(16+8-1)-2)
    cases = []
    for near in (False, True):
        delta = 2.**-12
        X = np.column_stack((u, u+delta*v if near else v)).astype(float)
        bread = np.array([[1+1/delta**2, -1/delta**2],
                          [-1/delta**2, 1/delta**2]]) if near else np.eye(2)
        vb = np.zeros((3, 3))
        vb[:2, :2], vb[2, 2] = bread*base_variance, base_variance
        for zero in (False, True):
            beta = np.zeros(2) if zero else np.array([1.25, -.5])
            y = X @ beta + (a-7.5)/8 + (b-3.5)/4 + 3 + .25*noise
            for xs in (1., 2.**-10):
                for ys in (1., 2.**-26):
                    for offset in (0., 2.**20):
                        transform = np.diag([ys/xs, ys/xs, ys])
                        transform[2, 0] = -offset*ys/xs
                        bref, vref = transform @ np.r_[beta, 3.], transform @ vb @ transform.T
                        model = core.HdfeRegressor(num_threads=2, se_type="unadjusted")
                        row = dict(kind="positive", near=near, zero=zero, xs=xs, ys=ys, offset=offset)
                        try:
                            model.fit(y*ys, X*xs+np.array([offset, 0.]), [a, b])
                            be = float(np.max(abs(model.coef_-bref)/np.maximum(1, abs(bref))))
                            ve = float(np.max(abs(model.covariance_-vref)/
                                              np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
                            good = be <= 1e-9 and ve <= 1e-8 and model.gpu_used_ and model.precision_certified_
                            row.update(b_error=be, V_error=ve, status="PASS" if good else "FAIL")
                        except RuntimeError as error:
                            row.update(status="REFUSAL", error=str(error))
                        cases.append(row)
    if args.known_cases:
        import pandas as pd
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root/"benchmarks/encompassing"))
        from enc_worker_py import load_registry, select
        references = json.loads((root/"Verifications/core24_matrix_20260921/qualified_references.json").read_text())
        for name in ("directors", "pf_difficult_10m_3fe"):
            data = select(load_registry(root/"benchmarks/encompassing/registry.json"), "core24", name)[0]
            clusters = data.get("cluster", [])
            if isinstance(clusters, str):
                clusters = [clusters]
            columns = list(dict.fromkeys([data["y"], *data["x"], *data["fe"], *clusters]))
            df = pd.read_parquet(data["path"], columns=columns)
            codes = lambda column: pd.factorize(df[column], sort=True)[0].astype("int32")
            model = core.HdfeRegressor(num_threads=16, se_type="cluster" if clusters else "unadjusted")
            row = dict(kind="K6", dataset=name)
            try:
                model.fit(df[data["y"]].to_numpy(float),
                          np.asfortranarray(df[data["x"]].to_numpy(float)),
                          [codes(c) for c in data["fe"]], clusters=[codes(c) for c in clusters] or None)
                reference = np.asarray(references[name]["beta"])
                error = float(np.max(abs(np.asarray(model.coef_)[:len(reference)]-reference)/
                                     np.maximum(1, abs(reference))))
                row.update(b_error=error, status="PASS" if error <= 1e-9 and model.gpu_used_ else "FAIL",
                           outcome="returned_estimates_checked_against_independent_b")
            except RuntimeError as error:
                clean = model.lifecycle_state_ == "failed" and not model.converged_
                clean = clean and np.asarray(model.coef_).size == 0 and np.asarray(model.covariance_).size == 0
                expected = "original-design normal equation" in str(error) and "single N1 refinement" in str(error)
                row.update(status="PASS" if clean and expected else "FAIL",
                           outcome="informative_refusal_no_estimates", error=str(error))
            cases.append(row)
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=cases)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" for row in cases), report
    print(f"PASS: {len(cases)} positive/containment cases; K6 refusal is not GPU convergence certification")


if __name__ == "__main__":
    main()
