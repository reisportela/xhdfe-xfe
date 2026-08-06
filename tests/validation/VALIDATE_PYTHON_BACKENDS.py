#!/usr/bin/env python3
"""Validate packaged py_hdfe_v11 CPU and CUDA artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def find_module(directory: Path, prefix: str = "") -> Path:
    candidates = sorted(directory.glob(f"{prefix}py_hdfe_v11*.so"))
    if not candidates and prefix:
        candidates = sorted(directory.glob("py_hdfe_v11*.so"))
    if not candidates:
        raise FileNotFoundError(f"py_hdfe_v11 module not found in {directory}")
    return candidates[0]


def run_module(module_file: Path, backend: str, require_gpu: bool) -> dict:
    suffix = "".join(module_file.suffixes)
    with tempfile.TemporaryDirectory(prefix="xhdfe_py_validate_") as tmp:
        tmpdir = Path(tmp)
        target = tmpdir / ("py_hdfe_v11" + suffix)
        shutil.copy2(module_file, target)
        code = r'''
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.environ["XHDFE_VALIDATE_MODULE_DIR"])
import py_hdfe_v11

rng = np.random.default_rng(20260605)
n = 50000
fe1 = np.arange(n, dtype=np.int64) % 1000
fe2 = np.arange(n, dtype=np.int64) % 97
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
X = np.column_stack([x1, x2])
y = 1.25 * x1 - 0.70 * x2 + 0.01 * fe1 - 0.02 * fe2 + rng.normal(scale=0.01, size=n)

reg = py_hdfe_v11.HdfeRegressor()
reg.fit(y, X, fes=[fe1, fe2])

payload = {
    "backend": os.environ.get("XHDFE_GPU_BACKEND", ""),
    "converged": bool(reg.converged_),
    "gpu_used": bool(reg.gpu_used_),
    "iterations": int(reg.num_iterations_),
    "coef": [float(v) for v in np.asarray(reg.coef_).ravel()],
    "stderr": [float(v) for v in np.asarray(reg.stderr_).ravel()],
}
print(json.dumps(payload, sort_keys=True))
'''
        env = os.environ.copy()
        env["XHDFE_VALIDATE_MODULE_DIR"] = str(tmpdir)
        env["XHDFE_GPU_BACKEND"] = backend
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout)
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        if not payload["converged"]:
            raise RuntimeError(f"{backend} validation did not converge: {payload}")
        if require_gpu and not payload["gpu_used"]:
            raise RuntimeError(f"{backend} validation did not use GPU: {payload}")
        return payload


def main() -> int:
    default_cpu_dir = ROOT / "build_release_cpu_20260605_dist_265"
    default_cuda_dir = ROOT / "build_release_cuda_sm90_20260605_dist_265"
    packaged_python_dir = ROOT / "prebuilt" / "linux_x86_64" / "python"
    if not default_cpu_dir.exists() and packaged_python_dir.exists():
        default_cpu_dir = packaged_python_dir
    if not default_cuda_dir.exists() and packaged_python_dir.exists():
        default_cuda_dir = packaged_python_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-dir", type=Path, default=default_cpu_dir)
    parser.add_argument("--cuda-dir", type=Path, default=default_cuda_dir)
    args = parser.parse_args()

    cpu_module = find_module(args.cpu_dir, "cpu_")
    try:
        cuda_module = find_module(args.cuda_dir, "cuda_sm90_")
    except FileNotFoundError:
        cuda_module = find_module(args.cuda_dir, "cuda_sm75_80_86_89_90_")

    cpu = run_module(cpu_module, "cpu", require_gpu=False)
    cuda = run_module(cuda_module, "cuda", require_gpu=True)

    max_coef_diff = max(abs(a - b) for a, b in zip(cpu["coef"], cuda["coef"]))
    max_se_diff = max(abs(a - b) for a, b in zip(cpu["stderr"], cuda["stderr"]))
    if max_coef_diff > 1e-8 or max_se_diff > 1e-8:
        raise RuntimeError(
            f"CPU/CUDA mismatch too large: coef={max_coef_diff:.3g}, se={max_se_diff:.3g}"
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "cpu": cpu,
                "cuda": cuda,
                "max_coef_diff": max_coef_diff,
                "max_se_diff": max_se_diff,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
