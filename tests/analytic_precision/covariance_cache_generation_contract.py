"""Old solve-generation caches must miss; new exact caches must remain usable."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-module", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old, new = args.old_module.resolve(), args.module.resolve()
    helper = (args.contracts / "cache_identity_contract.py").resolve()
    assert old.is_file() and new.is_file() and helper.is_file()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = {str(path): digest(path) for path in (old, new, helper)}
    write_new(output / "PLAN.json", dict(
        status="FROZEN", inputs=frozen, fits=10,
        purpose="Solve-generation separation, read-only custody and exact-hit preservation",
        fixtures="Existing independent synthetic cache identity oracle",
        backend="CPU, with either CPU-only or CUDA-capable native module",
        negative_control="An unchanged-generation candidate must fail old-cache rejection"))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("XHDFE_") and k != "LD_LIBRARY_PATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", CUDA_CACHE_DISABLE="1")
    rows = []
    for mode in ("xhdfe-fast", "reghdfe-comparable"):
        old_cache = output / (mode + "_old.bin")
        new_cache = output / (mode + "_new.bin")
        sequence = [
            ("old_write", old, "write", old_cache, False),
            ("old_read", old, "read", old_cache, True),
            ("new_read_old", new, "read", old_cache, False),
            ("new_write", new, "write", new_cache, False),
            ("new_read", new, "read", new_cache, True),
        ]
        for role, module, operation, cache, expected in sequence:
            for name, expected_hash in frozen.items():
                assert digest(Path(name)) == expected_hash, name
            before = digest(cache) if cache.exists() else None
            command = [
                sys.executable, "-B", str(helper), "--worker",
                "--module", str(module), "--scratch", str(output),
                "--cache", str(cache), "--operation", operation, "--mode", mode,
            ]
            run = subprocess.run(command, env=env, cwd=output, text=True,
                                 capture_output=True, timeout=420)
            label = mode + "_" + role
            with (output / (label + ".log")).open("x") as stream:
                stream.write(run.stdout + "\n" + run.stderr)
            errors = []
            result = {}
            hits = re.findall(r"cache_hit=([01])", run.stderr)
            observed = bool(int(hits[-1])) if hits else None
            if run.returncode:
                errors.append("Worker failed")
            else:
                try:
                    result = json.loads(run.stdout.strip().splitlines()[-1])
                except (ValueError, IndexError):
                    errors.append("Missing numerical receipt")
            if result.get("status") != "PASS" or not result.get("certified"):
                errors.append("Independent numerical contract did not pass")
            if observed != expected:
                errors.append("Unexpected cache hit/miss")
            after = digest(cache) if cache.exists() else None
            if after is None:
                errors.append("Missing cache file")
            if operation == "read" and before != after:
                errors.append("Read changed cache bytes")
            row = dict(mode=mode, role=role, status="FAIL" if errors else "PASS",
                       returncode=run.returncode, expected_hit=expected,
                       observed_hit=observed, numerical=result, errors=errors,
                       cache_sha256_before=before, cache_sha256_after=after)
            rows.append(row)
            print(mode, role, row["status"], flush=True)
    for name, expected_hash in frozen.items():
        assert digest(Path(name)) == expected_hash, name
    passed = len(rows) == 10 and all(row["status"] == "PASS" for row in rows)
    write_new(output / "RESULT.json", dict(
        status="PASS" if passed else "FAIL", inputs=frozen, fits=len(rows), rows=rows,
        limitation="This contract tests cache generation and CPU cache hits; CUDA execution is validated separately."))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
