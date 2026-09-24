#!/usr/bin/env python3
"""FE cache must preserve the input design despite corruption or stale files."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct

import numpy as np
from scipy.linalg import qr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", required=True, type=Path)
    p.add_argument("--scratch", required=True, type=Path)
    a = p.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", a.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(11)
    n = 3000
    f1 = rng.integers(0, 60, n).astype(np.int32)
    f2 = rng.integers(0, 45, n).astype(np.int32)
    f3 = rng.integers(0, 45, n).astype(np.int32)
    x = rng.normal(size=(n, 1)) + .7*rng.normal(size=45)[f2, None]
    y = x[:, 0] + rng.normal(size=60)[f1] + 3*rng.normal(size=45)[f2] + rng.normal(size=n)
    f1[-1] = f2[-1] = f3[-1] = 80  # one independently known singleton
    design = np.column_stack((np.eye(60)[f1[:-1]], np.eye(45)[f2[:-1]]))
    q, r, _ = qr(design, mode="economic", pivoting=True)
    rank = int(np.sum(abs(np.diag(r)) > 1e-10))
    q = q[:, :rank]
    xt = x[:-1, 0] - q @ (q.T @ x[:-1, 0])
    yt = y[:-1] - q @ (q.T @ y[:-1])
    b = xt @ yt / (xt @ xt)
    u = yt - xt*b
    v = (u @ u)/(n-1-rank-1)/(xt @ xt)
    os.environ.update(XHDFE_ABSORPTION_CACHE_MODE="off", XHDFE_MOBILITY_MODE="off")
    rows = []

    def fit(name, mode, fe=f2, check=True, threads=2):
        path = a.scratch/name
        os.environ.update(XHDFE_FE_STRUCTURE_CACHE=str(path), XHDFE_FE_STRUCTURE_MODE=mode)
        previous = path.read_bytes() if path.exists() else None
        m = core.HdfeRegressor(num_threads=threads)
        m.fit(y, x, [f1, fe])
        if check:
            assert abs(m.coef_[0]-b) <= 1e-9
            assert abs(m.covariance_[0, 0]-v) <= 1e-8*v
            assert np.array_equal(m.sample_index_, np.arange(n-1))
        if mode == "read":
            assert path.read_bytes() == previous
        rows.append(dict(case=name, mode=mode, b=float(m.coef_[0]), status="PASS"))
        return path.read_bytes()

    original = fit("A.bin", "write")
    assert original.startswith(b"xhdfe_fe_structure_cache_v2\0")
    fit("A.bin", "read", threads=1)
    other = fit("B.bin", "write", f3, False)
    # magic(32), signature(32), dimensions(16), keep count(4), kept indices.
    first_dim = 84 + 4*(n-1)
    second_dim = first_dim + 8 + 4*(n-1)
    damaged = {
        "torn.bin": original[:second_dim] + other[second_dim:],
        "truncated.bin": original[:-40],
        "trailing.bin": original+b"unexpected",
        "legacy.bin": original.replace(b"cache_v2", b"cache_v1", 1),
        "wrong_input.bin": other,
    }
    for name, position, value in [
        ("nobs.bin", 68, n+1), ("keep_count.bin", 80, 2**31-1),
        ("kept_negative.bin", 84, -1), ("kept_range.bin", 84, n),
        ("kept_order.bin", 88, 0), ("groups.bin", first_dim, n+10),
        ("id_negative.bin", first_dim+8, -1), ("id_range.bin", first_dim+8, 2**31-1),
        ("id_changed.bin", first_dim+8, 1),
    ]:
        payload = bytearray(original)
        struct.pack_into("<i", payload, position, value)
        damaged[name] = bytes(payload)
    for name, payload in damaged.items():
        (a.scratch/name).write_bytes(payload)
        fit(name, "read")
    (a.scratch/"results.json").write_text(json.dumps(rows, indent=2)+"\n")
    print("PASS: FE cache identity, payload, range, legacy and sample checks; independent QR")


if __name__ == "__main__":
    main()
