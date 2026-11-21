#!/usr/bin/env python3
"""The rational operator oracle of 15 November against the integrated CUDA operator of the reduced tier
(package 2, Amendment 2, C6; 17 November 2026).

The fixtures are the oracle's own (synthetic layouts and operands, exact Fraction results and the
operation-count bounds fixed before any output was seen), read unchanged from the folder given:
Verifications/reduced_cuda_comparable_20261115_01/audit/operator_oracle_02 in the main checkout, made by
prepare_operator_oracle_02.py there. The runner is the build target xhdfe_reduced_cuda_operator_oracle
(tests/reduced_cuda_operator_oracle.cpp, linked with the library's operator). Builds nothing.

Grading, as the oracle's: every live lane within its predeclared absolute bound; a deliberate perturbation
(32 bounds away) must be rejected by the grader; two calls bit-identical. Added for the integrated operator:
the padding lanes of q are +0.0 bit for bit; and the two negative controls of the operator, a one-byte budget
(a Memory decline) and an illegal width of lanes + 1 (Fatal).

    python3 -B -I tests/reduced_cuda_operator_oracle.py --runner <build>/xhdfe_reduced_cuda_operator_oracle \\
        --fixtures <.../operator_oracle_02> --output <new directory>
"""
import argparse
import hashlib
import json
import os
import resource
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

CONFIGS = 7  # P, B_tiles, B_radix, T and the three *_items_of_2 (see the runner)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runner", required=True)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    exe, fix, out = Path(args.runner).resolve(), Path(args.fixtures).resolve(), Path(args.output).resolve()
    out.mkdir(parents=False, exist_ok=False)  # a new directory: nothing is overwritten
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    paths = sorted(fix.glob("*.txt"))
    if not paths:
        print(f"no fixtures in {fix}")
        return 2
    identities = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [exe, Path(__file__).resolve()]}
    host = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True)
    (out / "host.json").write_text(json.dumps(dict(processes=host.stdout, load=os.getloadavg()), indent=2))
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    env["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] = "0"
    result = subprocess.run([str(exe), *(str(p) for p in paths)], capture_output=True, text=True, timeout=1800, env=env)
    (out / "stdout.jsonl").write_text(result.stdout)
    (out / "stderr.txt").write_text(result.stderr)
    rows = []
    for line in result.stdout.splitlines():
        row = json.loads(line)
        case = json.loads(Path(row["path"]).with_suffix(".json").read_text())
        expected = np.asarray(case["expected"])
        bound = np.asarray(case["absolute_forward_error_bound"])
        actual = np.asarray(row["q"], dtype=np.float64).reshape(expected.shape)
        width = row["width"]
        live_a, live_e, live_b = actual[:, :width], expected[:, :width], bound[:, :width]
        error = np.abs(live_a - live_e)
        passed = bool(np.all(np.isfinite(live_a)) and np.all(error <= live_b))
        padding_ok = all(struct.pack("<d", float(v)) == b"\x00" * 8 for v in actual[:, width:].ravel())
        ratio = float(np.max(np.divide(error, live_b, out=np.zeros_like(error), where=live_b > 0))) if error.size else 0.0
        perturb = live_e + 32 * live_b + np.spacing(np.abs(live_e) + 1.0)
        negative = bool(np.any(np.abs(perturb - live_e) > live_b))
        ok = (passed and padding_ok and negative and row["repeat_bit_identical"] and row["budget_declined"]
              and row["illegal_width_rejected"])
        rows.append(dict(name=case["name"], config=row["config"], classes=row["classes"], width=width,
                         status="PASS" if ok else "FAIL", max_error_bound_ratio=ratio,
                         max_abs_error=float(error.max()) if error.size else 0.0, padding_plus_zero=padding_ok,
                         repeat_bit_identical=row["repeat_bit_identical"], budget_declined=row["budget_declined"],
                         illegal_width_rejected=row["illegal_width_rejected"], negative_control_rejected=negative))
    planned = sum(CONFIGS * (5 if json.loads(p.with_suffix(".json").read_text())["layout"]["lanes"] == 8 else 1) for p in paths)
    status = ("PASS" if result.returncode == 0 and len(rows) == planned and all(r["status"] == "PASS" for r in rows)
              else "FAIL")
    summary = dict(status=status, returncode=result.returncode, fixtures=len(paths), planned=planned, executed=len(rows),
                   identities=identities, rows=rows,
                   scope="Independent rational operator action on the integrated operator; predeclared "
                         "operation-derived bounds of 15 November.")
    (out / "receipt.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["status", "returncode", "fixtures", "planned", "executed"]}))
    print("MAX_ERROR_BOUND_RATIO", max((r["max_error_bound_ratio"] for r in rows), default=None))
    by = {}
    for r in rows:
        by.setdefault(r["config"], [0, 0])[0 if r["status"] == "PASS" else 1] += 1
    print("BY_CONFIG", json.dumps(by))
    if result.returncode != 0:
        print("RUNNER_STDERR", result.stderr[-2000:])
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
