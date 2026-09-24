#!/usr/bin/env python3
"""Bootstrap on retained rows: literal trimmed-sample parity and OLS oracles."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings

sys.dont_write_bytecode = True
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from xhdfe.gelbach import bootstrap

    rng = np.random.default_rng(86)
    retained_n, singles = 400, 80
    fe = np.r_[np.repeat(np.arange(20), 20), np.arange(20, 20 + singles)]
    x, z = rng.normal(size=(2, fe.size))
    y = .7 * x + .3 * z + rng.normal(size=100)[fe] + rng.normal(size=fe.size)
    y[retained_n:] += 20 * x[retained_n:] + 100
    cases = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for method in ("pairs", "cluster_pairs"):
            kw = dict(method=method, reps=12, min_valid_reps=10,
                      seed=381, num_threads=2, sample_info=True)
            if method == "cluster_pairs":
                kw["bootstrap_cluster"] = fe
            full = bootstrap(y, x, {"z": z}, {"fe": fe}, **kw)
            if method == "cluster_pairs":
                kw["bootstrap_cluster"] = fe[:retained_n]
            trimmed = bootstrap(y[:retained_n], x[:retained_n],
                                {"z": z[:retained_n]}, {"fe": fe[:retained_n]}, **kw)
            assert np.array_equal(full["sample_index"], np.arange(retained_n))
            assert full["bootstrap"]["n_rows_population"] == retained_n
            errors = {}
            for key, draw in full["bootstrap"]["draws"].items():
                reference = trimmed["bootstrap"]["draws"][key]
                assert np.array_equal(np.isnan(draw), np.isnan(reference))
                good = np.isfinite(reference)
                errors[key] = float(np.max(np.abs(draw[good] - reference[good]))) if good.any() else 0.
                assert errors[key] <= 1e-12, (method, key, errors[key])
            base = np.column_stack((x[:retained_n], np.ones(retained_n)))
            full_design = np.column_stack((x[:retained_n], z[:retained_n],
                                           np.eye(20)[fe[:retained_n]]))
            b_base = np.linalg.lstsq(base, y[:retained_n], rcond=None)[0][0]
            b_full = np.linalg.lstsq(full_design, y[:retained_n], rcond=None)[0][0]
            assert abs(full["b_base"][0] - b_base) <= 1e-10
            assert abs(full["b_full"][0] - b_full) <= 1e-10
            cases.append(dict(method=method, errors=errors, status="PASS"))
        # Population with only one retained cluster cannot use cluster pairs.
        clusters = np.r_[np.zeros(retained_n), np.arange(1, singles + 1)]
        try:
            bootstrap(y, x, {"z": z}, {"fe": fe}, method="cluster_pairs",
                      bootstrap_cluster=clusters, reps=2, num_threads=2)
        except ValueError as exc:
            assert "retained clusters" in str(exc)
        else:
            raise AssertionError("accepted one retained resampling cluster")
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=cases)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("PASS: pairs/cluster pairs equal retained-sample bootstrap; OLS point oracles")


if __name__ == "__main__":
    main()
