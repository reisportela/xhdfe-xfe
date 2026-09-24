#!/usr/bin/env python3
"""AKM unavailable outputs and fweight counts, checked by literal replication."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rows = []
    for n in (1, 2):
        result = core.akm_kss(np.arange(1., n+1), np.ones(n, dtype="int32"),
                              np.full(n, 7, dtype="int32"), num_threads=1)
        valid = not result["converged"] and result["alpha"].size == 0 and result["psi"].size == 0
        valid = valid and np.isnan(result["var_y"]) and np.isnan(result["sigma2_ho"])
        for name in ("plugin", "agsu", "kss"):
            valid = valid and all(np.isnan(result[name][key]) for key in
                                 ("var_alpha", "var_psi", "cov_alpha_psi"))
        rows.append(dict(case="unavailable", n=n, status="PASS" if valid else "FAIL"))
    y = np.array([1., 2., 3.])
    worker, firm = np.array([1, 1, 2]), np.array([7, 7, 8])
    frequency = np.array([5., 5., 1.])
    index = np.repeat(np.arange(3), frequency.astype(int))
    compact = core.akm_kss(y, worker, firm, fweights=frequency, num_threads=1)
    expanded = core.akm_kss(y[index], worker[index], firm[index], num_threads=1)
    # Independent exact values for ten retained observations, five 1s/five 2s:
    # one worker and one firm imply zero component variation; RSS/(10-1)=2.5/9.
    valid = compact["converged"] and expanded["converged"]
    valid = valid and compact["sample"]["n_obs_input"] == 11 and compact["sample"]["n_obs"] == 10
    valid = valid and abs(compact["var_y"] - 2.5/9) <= 1e-14
    valid = valid and abs(compact["sigma2_ho"] - 2.5/9) <= 1e-14
    valid = valid and np.all(compact["alpha"] == 1.5) and np.all(compact["psi"] == 0)
    for name in ("plugin", "agsu", "kss"):
        for key in ("var_alpha", "var_psi", "cov_alpha_psi"):
            valid = valid and compact[name][key] == expanded[name][key] == 0
    rows.append(dict(case="frequency_replication", status="PASS" if valid else "FAIL"))
    report = dict(module=str(args.module.resolve()),
        sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert all(row["status"] == "PASS" for row in rows), report
    print("PASS: unavailable samples and exact frequency replication")


if __name__ == "__main__":
    main()
