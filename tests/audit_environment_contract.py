#!/usr/bin/env python3
"""Check candidate audit switches with a fresh native module per case."""

import os
import subprocess
import sys


WORKER = r"""
import importlib.util
import numpy as np
import sys
spec = importlib.util.spec_from_file_location('py_hdfe_v11', sys.argv[1])
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
x = np.arange(96, dtype=float)
y = 0.5 * x + np.sin(x)
m = core.HdfeRegressor(num_threads=1, absorption_method='gauss_seidel',
                       tolerance_mode='reghdfe-comparable', max_iter=1000)
m.fit(y, np.asfortranarray(x[:, None]), fes=[np.arange(96, dtype=np.int32) % 8])
assert m.converged_
"""

BOOL_FLAGS = (
    "XHDFE_CERTIFICATE_TEAM", "XHDFE_GI_FUSE_OPERATOR_REGIONS",
    "XHDFE_HUGE_PAGES", "XHDFE_MLSMR_AC_ARENA",
    "XHDFE_MLSMR_FACTOR_PRODUCTS", "XHDFE_MLSMR_LOCKSTEP",
    "XHDFE_MLSMR_LOCKSTEP_CHECK", "XHDFE_MLSMR_LOCKSTEP_SCAN",
    "XHDFE_MLSMR_STATE_TRACE", "XHDFE_PACKED_FUSED",
    "XHDFE_MLSMR_PARALLEL_APPLY", "XHDFE_MLSMR_DISJOINT_DIRECT_APPLY",
    "XHDFE_MLSMR_CSR_DOMAIN",
)


def main():
    module = os.path.abspath(sys.argv[1])
    cases = 0
    for name in BOOL_FLAGS + ("XHDFE_MLSMR_BUILD", "XHDFE_MLSMR_EDGE_ORDER",
                              "XHDFE_PACKED_BLOCK_BYTES"):
        if name == "XHDFE_MLSMR_BUILD":
            values = (("assembled", True), (" reference ", True),
                      ("legacy", True), ("0", False), ("1", False))
        elif name == "XHDFE_MLSMR_EDGE_ORDER":
            values = (("sorted", True), (" hash ", True),
                      ("0", False), ("1", False))
        elif name == "XHDFE_PACKED_BLOCK_BYTES":
            values = (("0", True), ("1", True), (" 512 ", True))
        else:
            values = (("0", True), ("1", True), (" 0 ", True),
                      (" yes ", True))
        values += (("-1", False), ("18446744073709551616", False),
                   ("junk", False), (" ", False), ("", False))
        for value, valid in values:
            env = os.environ.copy()
            env.pop("LD_LIBRARY_PATH", None)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env[name] = value
            completed = subprocess.run(
                [sys.executable, "-B", "-c", WORKER, module], env=env,
                capture_output=True, text=True, timeout=30)
            if valid and completed.returncode != 0:
                raise AssertionError((name, value, completed.stderr[-300:]))
            if not valid and (completed.returncode == 0 or name not in completed.stderr):
                raise AssertionError((name, value, completed.returncode,
                                      completed.stderr[-300:]))
            cases += 1
    print(f"AUDIT_ENVIRONMENT_CONTRACT PASS {cases}/{cases}")


if __name__ == "__main__":
    main()
