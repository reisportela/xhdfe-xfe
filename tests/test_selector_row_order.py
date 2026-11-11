#!/usr/bin/env python3
"""The automatic selector must not depend on the row order.

The selector estimates the contraction of the sweep on four contiguous
blocks of 50 000 rows, which represent a panel sorted by its largest
identifier. On the same data in random row order the sample is far sparser
than the data (almost every level of the largest identifier appears once),
the contraction looks worse than it is and MLSMR was chosen (30.6 s against
0.8 s on this design, routing/row_order_reproducer). Since the
representation check, a block sample that chose MLSMR on at least 1 000 000
rows with fewer than a quarter of the full data's observations per level of
the largest identifier is replaced by the same probe on all rows.

Synthetic data, generated in memory: 2 000 000 rows, 10 regressors, three
well-mixed fixed effects of 200 000, 20 000 and 20 levels. Asserted: the same
route and the same iterations in both row orders, both certified; the check
applied in random order and not in sorted order; the full-data contraction
bit-identical at 1, 4 and 16 threads. Time is not asserted.
--with-kill-switch adds the fit with XHDFE_AUTO_MLSMR_FULL_PROBE=0 in random
order (today's selector: MLSMR, about 30 s).

usage: test_selector_row_order.py --module <py_hdfe_v11 .so> --module-sha256 <hex>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SITE_PATHS = (
    "/home/mangelo/.local/lib/python3.12/site-packages",
    "/home/mangelo/miniconda3/lib/python3.12/site-packages",
)
for path in SITE_PATHS:
    if path not in sys.path:
        sys.path.append(path)

import numpy as np

ROWS = 2_000_000
REGRESSORS = 10
GAUSS_SEIDEL = 1
MLSMR = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def splitmix64(values, seed):
    with np.errstate(over="ignore"):
        z = values.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def design(order: str):
    """The reproducer's data; 'sorted' reorders the rows by the largest identifier."""
    i = np.arange(ROWS, dtype=np.int64)
    fe1 = (splitmix64(i, 1) % np.uint64(ROWS // 10)).astype(np.int32)
    fe2 = (splitmix64(i, 2) % np.uint64(ROWS // 100)).astype(np.int32)
    fe3 = (splitmix64(i, 3) % np.uint64(20)).astype(np.int32)
    X = np.asfortranarray(np.column_stack([
        np.sin(0.001 * (j + 1) * i) + ((i * (17 + 2 * j)) % (101 + j)) / (101 + j)
        for j in range(REGRESSORS)]))
    y = (X @ np.linspace(0.5, -0.5, REGRESSORS) + 0.5 * np.sin(fe1) + 0.3 * np.cos(fe2)
         + 0.01 * fe3 + np.sin(0.0021 * i))
    if order == "sorted":
        perm = np.argsort(fe1, kind="stable")
        fe1, fe2, fe3, y = fe1[perm], fe2[perm], fe3[perm], y[perm]
        X = np.asfortranarray(X[perm])
    return y, X, [fe1, fe2, fe3]


def child(args) -> int:
    module_path = Path(args.module)
    if sha256(module_path) != args.module_sha256:
        raise AssertionError("module hash differs")
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import module")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    y, X, fes = design(args.order)
    reg = core.HdfeRegressor(
        se_type="robust", tol=1e-8, drop_singletons=False, num_threads=args.threads,
        absorption_method="auto", tolerance_mode="reghdfe-comparable")
    reg.fit(y, X, fes=fes)
    result = {
        "method": int(reg.absorption_method_used),
        "iterations": int(reg.num_iterations_),
        "certified": bool(reg.precision_certified_),
        "converged": bool(reg.converged_),
        "coef": [float(value) for value in np.asarray(reg.coef_)],
    }
    Path(args.output).write_text(json.dumps(result, sort_keys=True) + "\n")
    return 0


def run_child(script, root, label, module, module_sha, order, threads, environment=None):
    output = root / f"{label}.json"
    command = [
        sys.executable, "-B", "-I", "-S", str(script), "--child",
        "--module", str(module), "--module-sha256", module_sha,
        "--order", order, "--threads", str(threads), "--output", str(output),
    ]
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("XHDFE_"):
            del env[name]
    env.update({
        "XHDFE_GPU_BACKEND": "cpu", "XHDFE_AUTO_MLSMR_TRACE": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if environment:
        env.update(environment)
    completed = subprocess.run(command, check=True, env=env, timeout=600,
                               capture_output=True, text=True)
    lines = [line for line in completed.stderr.splitlines()
             if line.startswith("auto_mlsmr ") and " sample_rho=" in line]
    if len(lines) != 1:
        raise AssertionError(f"{label}: expected one selector line, got {len(lines)}")
    fields = dict(token.split("=", 1) for token in lines[0].split()[1:] if "=" in token)
    return json.loads(output.read_text()), fields


def parent(args) -> int:
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="xhdfe-selector-order-") as raw:
        root = Path(raw)
        shuffled, shuffled_trace = run_child(
            script, root, "random", args.module, args.module_sha256, "random", 16)
        ordered, ordered_trace = run_child(
            script, root, "sorted", args.module, args.module_sha256, "sorted", 16)
        one, one_trace = run_child(
            script, root, "random_t1", args.module, args.module_sha256, "random", 1)
        four, four_trace = run_child(
            script, root, "random_t4", args.module, args.module_sha256, "random", 4)

        # The same route and the same iterations in both row orders.
        for fit in (shuffled, ordered, one, four):
            assert fit["certified"] and fit["converged"], fit
        assert shuffled["method"] == ordered["method"] == GAUSS_SEIDEL, (shuffled, ordered)
        assert shuffled["iterations"] == ordered["iterations"], (shuffled, ordered)
        # Random order: the block sample under-represents the data and chose
        # MLSMR; the check applies and the full-data probe decides.
        assert shuffled_trace["check"] == "applied", shuffled_trace
        assert shuffled_trace["full_probe"] == "1", shuffled_trace
        assert float(shuffled_trace["representation"]) < 0.25, shuffled_trace
        assert shuffled_trace["selected"] != "mlsmr", shuffled_trace
        # Sorted: the block sample represents the data and chose the sweep;
        # nothing else runs.
        assert ordered_trace["full_probe"] == "0", ordered_trace
        assert ordered_trace["selected"] != "mlsmr", ordered_trace
        assert float(ordered_trace["representation"]) > 0.9, ordered_trace
        # The full-data contraction does not depend on the thread count.
        assert (one_trace["full_rho_hex"] == four_trace["full_rho_hex"]
                == shuffled_trace["full_rho_hex"]), (one_trace, four_trace, shuffled_trace)
        assert one["method"] == four["method"] == shuffled["method"]
        assert one["iterations"] == four["iterations"] == shuffled["iterations"]

        kill_switch = "skipped"
        if args.with_kill_switch:
            today, today_trace = run_child(
                script, root, "random_today", args.module, args.module_sha256, "random",
                16, environment={"XHDFE_AUTO_MLSMR_FULL_PROBE": "0"})
            assert today["method"] == MLSMR and today_trace["full_probe"] == "0", today_trace
            kill_switch = "1"

    print(
        "SELECTOR_ROW_ORDER_PASS same_route=1 same_iterations=1 "
        f"iterations={shuffled['iterations']} random_check_applied=1 sorted_unchanged=1 "
        f"representation_random={shuffled_trace['representation']} "
        f"representation_sorted={ordered_trace['representation']} "
        f"sample_rho_random={shuffled_trace['sample_rho']} "
        f"full_rho_random={shuffled_trace['full_rho']} "
        f"check_s_random={shuffled_trace['check_s']} probe_s_random={shuffled_trace['probe_s']} "
        f"probe_s_t1={one_trace['probe_s']} "
        f"full_rho_thread_invariant=1 kill_switch={kill_switch}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--order", choices=("random", "sorted"), default="random")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-kill-switch", action="store_true")
    args = parser.parse_args()
    if args.child:
        return child(args)
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
