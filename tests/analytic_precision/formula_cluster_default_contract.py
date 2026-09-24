#!/usr/bin/env python3
"""Formula clusters select clustered inference unless the VCE is explicit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--formula", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import xhdfe
    spec = importlib.util.spec_from_file_location("xhdfe.py_hdfe_v11", args.module)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    formula = types.ModuleType("xhdfe._formula")
    formula.__file__, formula.__package__ = str(args.formula), "xhdfe"
    sys.modules[formula.__name__] = formula
    exec(compile(args.formula.read_bytes(), str(args.formula), "exec"), formula.__dict__)
    rng = np.random.default_rng(17)
    n = 2400
    cluster = np.repeat(np.arange(40), 60)
    x = rng.normal(size=n) + rng.normal(size=40)[cluster]
    y = .7 * x + rng.normal(size=40)[cluster] + rng.normal(size=n)
    w = 1. + np.arange(n) % 3
    data = dict(y=y, x=x, c=cluster, w=w)
    rows = []
    for weighted in (False, True):
        weights = w if weighted else np.ones(n)
        a = np.column_stack((x, np.ones(n)))
        sw = np.sqrt(weights)
        q, r = np.linalg.qr(a * sw[:, None])
        beta = np.linalg.solve(r, q.T @ (sw * y))
        invr = np.linalg.solve(r, np.eye(2))
        bread = invr @ invr.T
        scores = a * (weights * (y - a @ beta))[:, None]
        totals = np.zeros((40, 2))
        np.add.at(totals, cluster, scores)
        vref = bread @ (totals.T @ totals) @ bread * 40 / 39 * (n - 1) / (n - 2)
        kw = dict(weights="w", pweights=True) if weighted else {}
        model = formula.feols("y ~ x", data, clusters="c", num_threads=2, **kw)
        error = float(np.max(np.abs(np.asarray(model.covariance_) - vref) /
                            np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
        rows.append(dict(weighted=weighted, V_error=error, se_type=model.se_type_,
                         status="PASS" if error <= 1e-8 and model.se_type_ == "cluster" else "FAIL"))
        explicit = formula.feols("y ~ x", data, clusters="c", se_type="robust", num_threads=2, **kw)
        control = formula.feols("y ~ x", data, se_type="robust", num_threads=2, **kw)
        same = np.array_equal(explicit.covariance_, control.covariance_)
        rows.append(dict(case="explicit_robust", weighted=weighted, status="PASS" if same else "FAIL"))
    report = dict(module=str(args.module.resolve()), formula=str(args.formula.resolve()),
        module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
        formula_sha256=hashlib.sha256(args.formula.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" for row in rows), report
    print("PASS: clustered default with/without pweights and explicit robust precedence")


if __name__ == "__main__":
    main()
