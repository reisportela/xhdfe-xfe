#!/usr/bin/env python3
"""A real v4 cache is rejected safely; implicit mobility-cache hits remain valid."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
from cache_identity_contract import fixture, oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--legacy-module", type=Path)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--implicit-worker", choices=("write", "read", "negative"))
    args = parser.parse_args()
    if args.implicit_worker:
        for name in tuple(os.environ):
            if name.startswith("XHDFE_"):
                del os.environ[name]
        os.environ.update(XHDFE_GPU_BACKEND="cpu", XHDFE_DEBUG_SOLVER="1")
        if args.implicit_worker == "write":
            os.environ.update(XHDFE_MOBILITY_PROFILE="xhdfe_mobility_profile.txt",
                              XHDFE_MOBILITY_MODE="write")
        spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        y, x, fes = fixture("negative_y" if args.implicit_worker == "negative" else "base")
        model = core.HdfeRegressor(num_threads=2)
        model.fit(y, x, fes)
        bref, vref = oracle(y, x, fes, False)
        import numpy as np
        berror = float(np.max(np.abs(model.coef_ - bref)/np.maximum(1, np.abs(bref))))
        verror = float(np.max(np.abs(model.covariance_ - vref)/
                             np.sqrt(np.outer(np.diag(vref), np.diag(vref)))))
        assert berror <= 1e-9 and verror <= 1e-8
        print(json.dumps(dict(b_error=berror, V_error=verror)))
        return
    assert args.legacy_module is not None
    args.scratch.mkdir(parents=True, exist_ok=False)
    helper = Path(__file__).with_name("cache_identity_contract.py")
    cache = args.scratch / "real_v4.bin"
    rows = []
    for role, module, operation in (("legacy_write", args.legacy_module, "write"),
                                    ("candidate_read", args.module, "read")):
        cmd = [sys.executable, "-B", str(helper), "--worker", "--module", str(module.resolve()),
               "--scratch", str(args.scratch.resolve()), "--cache", str(cache.resolve()),
               "--operation", operation]
        run = subprocess.run(cmd, cwd=args.scratch, capture_output=True, text=True, timeout=420)
        (args.scratch / (role + ".log")).write_text(run.stdout + "\n" + run.stderr)
        run.check_returncode()
        result = json.loads(run.stdout.strip().splitlines()[-1])
        assert result["status"] == "PASS"
        if role == "legacy_write":
            assert cache.read_bytes().startswith(b"xhdfe_absorption_cache_v4")
            legacy_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
        else:
            assert re.findall(r"cache_hit=([01])", run.stderr)[-1] == "0"
            assert hashlib.sha256(cache.read_bytes()).hexdigest() == legacy_hash
        rows.append(dict(case=role, **result))
    implicit = args.scratch / "implicit"
    implicit.mkdir()
    for operation, hit in (("write", "0"), ("read", "1"), ("negative", "0")):
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--module", str(args.module.resolve()),
               "--scratch", str(implicit.resolve()), "--implicit-worker", operation]
        run = subprocess.run(cmd, cwd=implicit, capture_output=True, text=True, timeout=420)
        (implicit / (operation + ".log")).write_text(run.stdout + "\n" + run.stderr)
        run.check_returncode()
        assert re.findall(r"cache_hit=([01])", run.stderr)[-1] == hit
        rows.append(dict(case="implicit_" + operation, cache_hit=int(hit),
                         **json.loads(run.stdout.strip().splitlines()[-1])))
    report = dict(module=str(args.module.resolve()),
                  module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
                  legacy_module=str(args.legacy_module.resolve()),
                  legacy_module_sha256=hashlib.sha256(args.legacy_module.read_bytes()).hexdigest(),
                  legacy_cache_sha256=legacy_hash, cases=rows)
    (args.scratch / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: authentic v4 rejection, implicit exact hit and changed-input miss")


if __name__ == "__main__":
    main()
