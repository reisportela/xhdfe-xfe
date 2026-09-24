#!/usr/bin/env python3
"""Cache hits, changed inputs, corruption and concurrent publication in scratch."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np


def fixture(variant):
    rng = np.random.default_rng(7)
    n = 2000
    a = rng.integers(0, 50, n).astype("int32")
    b = rng.integers(0, 40, n).astype("int32")
    x = rng.normal(size=(n, 2))
    y = x @ np.array([1.5, -.7]) + rng.normal(size=50)[a]
    y += rng.normal(size=40)[b] + rng.normal(size=n)
    if variant == "negative_y":
        y = -y
    elif variant == "two_y_signs":
        y[[3, 1500]] *= -1
    elif variant == "negative_x":
        x[:, 0] *= -1
    elif variant == "changed_fe":
        a[0] = (a[0] + 1) % 50
    return y, x, [a, b]


def oracle(y, x, fes, robust):
    d = np.column_stack((np.eye(50)[fes[0]], np.eye(40)[fes[1], 1:]))
    a = np.column_stack((x, d))
    q, r = np.linalg.qr(a, mode="reduced")
    assert np.abs(np.diag(r)).min() > 1e-10
    beta = np.linalg.solve(r, q.T @ y)
    residual = y - a @ beta
    t = np.zeros((3, a.shape[1]))
    t[:2, :2] = np.eye(2)
    t[2, 2:] = d.mean(axis=0)
    influence = np.linalg.solve(r.T, t.T).T @ q.T
    if robust:
        score = influence * residual[None, :]
        v = score @ score.T * y.size / (y.size - a.shape[1])
    else:
        v = (influence @ influence.T) * (residual @ residual) / (y.size - a.shape[1])
    return t @ beta, v


def worker(args):
    for name in tuple(os.environ):
        if name.startswith("XHDFE_"):
            del os.environ[name]
    os.environ.update(XHDFE_GPU_BACKEND="cpu", XHDFE_MOBILITY_MODE="off",
        XHDFE_FE_STRUCTURE_MODE="off", XHDFE_ABSORPTION_CACHE=str(args.cache.resolve()),
        XHDFE_ABSORPTION_CACHE_MODE=args.operation, XHDFE_DEBUG_SOLVER="1")
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    y, x, fes = fixture(args.variant)
    robust = args.variant == "robust"
    model = core.HdfeRegressor(num_threads=2, tolerance_mode=args.mode,
        max_iter=100001 if args.variant == "changed_options" else 100000,
        se_type="robust" if robust else "unadjusted")
    model.fit(y, x, fes, weights=np.ones(y.size) if args.variant == "unit_weights" else None)
    bref, vref = oracle(y, x, fes, robust)
    berror = float(np.max(np.abs(np.asarray(model.coef_) - bref) / np.maximum(1, np.abs(bref))))
    verror = float(np.max(np.abs(np.asarray(model.covariance_) - vref) /
        np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
    print(json.dumps(dict(b_error=berror, V_error=verror,
        certified=bool(model.precision_certified_),
        status="PASS" if berror <= 1e-9 and verror <= 1e-8 else "FAIL")))


def y_offset(data):
    assert data[:32].startswith(b"xhdfe_absorption_cache_v5")
    # Fixed fields are emitted individually; no C++ struct padding is stored.
    position = 32 + 32 + struct.calcsize("<6i4d3?3idi2d")
    for _ in range(2):
        count, = struct.unpack_from("<i", data, position)
        position += 4 + 4 * count
    n, = struct.unpack_from("<q", data, position)
    assert n == 2000
    return position + 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--operation", default="read")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--mode", default="reghdfe-comparable")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.scratch.mkdir(parents=True, exist_ok=False)
    rows = []
    def run(cache, operation, variant, mode, expected_hit):
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
            "--module", str(args.module.resolve()), "--scratch", str(args.scratch.resolve()),
            "--cache", str(cache.resolve()), "--operation", operation,
            "--variant", variant, "--mode", mode]
        result = subprocess.run(cmd, cwd=args.scratch, capture_output=True, text=True, timeout=420)
        row = dict(mode=mode, operation=operation, variant=variant, expected_hit=expected_hit,
                   cache=cache.name, returncode=result.returncode)
        if result.returncode:
            row.update(status="FAIL", stdout=result.stdout, stderr=result.stderr)
        else:
            row.update(json.loads(result.stdout.strip().splitlines()[-1]))
            hits = re.findall(r"cache_hit=([01])", result.stderr)
            row["cache_hit"] = bool(int(hits[-1])) if hits else None
            if expected_hit is not None and row["cache_hit"] != expected_hit:
                row["status"] = "FAIL"
        return row
    for mode in ("xhdfe-fast", "reghdfe-comparable"):
        cache = args.scratch / (mode + ".bin")
        rows.append(run(cache, "write", "base", mode, False))
        pristine = cache.read_bytes()
        original_sha = hashlib.sha256(pristine).hexdigest()
        for variant in ("base", "negative_y", "two_y_signs", "negative_x", "changed_fe",
                        "changed_options", "unit_weights", "robust"):
            rows.append(run(cache, "read", variant, mode, variant in ("base", "robust")))
            assert hashlib.sha256(cache.read_bytes()).hexdigest() == original_sha
        altered = bytearray(pristine)
        offset = y_offset(altered)
        altered[offset + 8 * 3 + 7] ^= 0x80
        altered[offset + 8 * 1500 + 7] ^= 0x80
        variants = dict(two_payload_signs=bytes(altered), truncated=pristine[:-11],
                        trailing=pristine + b"x",
                        legacy_magic=b"xhdfe_absorption_cache_v4".ljust(32, b"\0") + pristine[32:])
        for name, data in variants.items():
            corrupted = args.scratch / (mode + "_" + name + ".bin")
            corrupted.write_bytes(data)
            rows.append(run(corrupted, "read", "base", mode, False))
            assert corrupted.read_bytes() == data
        # Different writers share one destination; every returned fit must be
        # correct and every subsequently opened file must be a complete entry.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run, cache, "write", variant, mode, False)
                       for variant in ("base", "negative_y", "negative_x")]
            rows.extend(f.result() for f in futures)
        for variant in ("base", "negative_y", "negative_x"):
            rows.append(run(cache, "read", variant, mode, None))
    assert not list(args.scratch.glob(".xhdfe-cache-*.tmp")), "temporary cache publication was not completed"
    report = dict(module=str(args.module.resolve()),
        sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), cases=rows)
    (args.scratch / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    assert all(row["status"] == "PASS" and row["certified"] for row in rows), report
    print("PASS: 38 cache identity, covariance, corruption and concurrent-publication cases")


if __name__ == "__main__":
    main()
