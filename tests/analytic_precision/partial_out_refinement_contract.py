#!/usr/bin/env python3
"""Build/run the credit2 partial-out contract against an explicit core archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=False)
    data = pd.read_parquet(args.data, columns=["y", "x1", "x2", "id1", "id2"]).dropna()
    raw = data[["y", "x1", "x2"]].to_numpy(dtype="float64")
    ids = np.column_stack([pd.factorize(data[name], sort=True)[0] for name in ("id1", "id2")]).astype("int32")
    keep = np.ones(len(raw), dtype=bool)
    while True:
        positions = np.flatnonzero(keep)
        good = np.ones(positions.size, dtype=bool)
        for j in range(2):
            _, codes, counts = np.unique(ids[positions, j], return_inverse=True, return_counts=True)
            good &= counts[codes] > 1
        if good.all():
            break
        keep[positions[~good]] = False
    raw, ids = np.ascontiguousarray(raw[keep]), np.ascontiguousarray(ids[keep])
    n = len(raw)
    blocks = []
    for j in range(2):
        _, code = np.unique(ids[:, j], return_inverse=True)
        blocks.append(sp.csr_matrix((np.ones(n), (np.arange(n), code)), shape=(n, code.max()+1)))
    dummy = sp.hstack(blocks, format="csr")
    dummy = dummy @ sp.diags(1/np.sqrt(np.asarray(dummy.power(2).sum(axis=0)).ravel()))
    reference = raw.copy()
    agreements = []
    for j in range(3):
        first = sl.lsmr(dummy, raw[:, j], atol=1e-15, btol=1e-15, maxiter=50000)
        second = sl.lsqr(dummy, raw[:, j], atol=1e-15, btol=1e-15, iter_lim=50000)
        reference[:, j] -= dummy @ first[0]
        other = raw[:, j] - dummy @ second[0]
        agreement = float(np.linalg.norm(reference[:, j]-other)/np.linalg.norm(reference[:, j]))
        assert agreement < 1e-11
        agreements.append(agreement)
    raw.tofile(args.scratch / "matrix.bin")
    ids.tofile(args.scratch / "fes.bin")
    reference.tofile(args.scratch / "reference.bin")
    source = Path(__file__).with_suffix(".cpp")
    binary = args.scratch / "contract"
    command = ["g++", "-std=c++17", "-O2", "-DNDEBUG", "-march=native", "-mtune=native",
               "-DHDFE_USE_OPENMP", "-fopenmp", "-I"+str(args.source/"include"),
               "-I"+str(args.source/"r/xhdfe/src/eigen"), str(source), str(args.library), "-o", str(binary)]
    subprocess.run(command, check=True, timeout=420)
    results = []
    environment = {key: value for key, value in os.environ.items() if not key.startswith("XHDFE_")}
    environment.update(XHDFE_GPU_BACKEND="cpu", XHDFE_MOBILITY_MODE="off",
                       XHDFE_ABSORPTION_CACHE_MODE="off", XHDFE_FE_STRUCTURE_MODE="off")
    for budget, success in ((15, False), (16, True), (100000, True)):
        run = subprocess.run([str(binary.resolve()), str(args.scratch.resolve()), str(n),
                              str(budget), str(int(success))], capture_output=True, text=True,
                             env=environment, timeout=420)
        results.append(dict(budget=budget, expect_success=success, returncode=run.returncode,
                            stdout=run.stdout, stderr=run.stderr))
    report = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  library=str(args.library.resolve()), library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  data=str(args.data.resolve()), data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  oracle_agreement=agreements, compile=command, cases=results)
    (args.scratch/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    assert all(case["returncode"] == 0 for case in results), results
    print("PASS: same-target partial-out continuation, independent oracle and hard iteration budget")


if __name__ == "__main__":
    main()
