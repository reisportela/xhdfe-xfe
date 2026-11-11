#!/usr/bin/env python3
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
import pandas as pd

TWO = ("x1", "x2")
DATASETS = {
    "credit2": (
        "/home/mangelo/Documents/BigData/Sergio/all-dta/credit2.parquet",
        "6f5fc748e7dcaf751d671331150f772d42c19893ba1ce350ddd26eae7eaa4d52",
        ("id1", "id2"), TWO,
    ),
    "credit": (
        "/home/mangelo/Documents/BigData/Sergio/all-dta/credit.parquet",
        "bbf5ddee09e67fc320ce8607a9ab0dc10d9969639363e1afd5ab6adabd50dd93",
        ("id1", "id2"), TWO,
    ),
    # Early routing (kappa read at each checkpoint, M = 30). In the wider D1
    # trace (16 threads), pf_bench_difficult_n100k_3fe predicted 21.9e-11 at
    # checkpoint 8 and 77.0e-11 at 16, so it leaves its accelerated sweep at
    # 16 (predictive rule); github, whose conjugate-gradient stage starts at
    # the automatic gate, predicted 25.5e-11 at stage iteration 32 and
    # 32.6e-11 at 64, so it leaves at 64. patents, with the accelerator
    # capped at 10 iterations, reaches the hand-off to a conjugate-gradient
    # stage (structural rule). synthetic-uniform-hard reaches checkpoints with a
    # prediction below 1e-11, so it stays.
    "pf_bench_difficult_n100k_3fe": (
        "/srv/projetos/P008/dataset/pyfixest/benchmark_difficult_n100000_k10.parquet",
        "9b38df80176c3e89b7812ddc3022c3fdf08aa2fe3d0054a9be31a5e75cdf7489",
        ("indiv_id", "firm_id", "year"), tuple(f"x{j}" for j in range(1, 11)),
    ),
    "patents": (
        "/home/mangelo/Documents/BigData/Sergio/all-dta/patents.parquet",
        "e2eb0c3f2fca6fa9d610bc4ee12ee02a90aebcf15edab1f3a4fb37607d75d5d8",
        ("id1", "id2"), TWO,
    ),
    "github": (
        "/home/mangelo/Documents/BigData/Sergio/all-dta/github.parquet",
        "958d890ccffc999f669fb1a4cc85e6d68f2e9e14a1d71c48481dcc8dc53ef63b",
        ("id1", "id2", "id3"), TWO,
    ),
    "synthetic-uniform-hard": (
        "/home/mangelo/Documents/BigData/Sergio/all-dta/synthetic-uniform-hard.parquet",
        "4d696a3fe5b29a3b313a6e0db8377b664ee62df32d3e3c27d8a2589dba7a75ff",
        ("id1", "id2"), TWO,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def factorize(series) -> np.ndarray:
    codes, _ = pd.factorize(series, sort=True)
    return codes.astype(np.int32, copy=False)


def child(args) -> int:
    module_path = Path(args.module)
    if sha256(module_path) != args.module_sha256:
        raise AssertionError("module hash differs")
    if args.dataset == "synthetic_huge4":
        n = 2_000_001
        row = np.arange(n, dtype=np.int64)
        x1 = np.sin(row * 0.001)
        x2 = np.cos(row * 0.0007)
        X = np.asfortranarray(np.column_stack((x1, x2)))
        fes = [(row % modulus).astype(np.int32) for modulus in (17, 19, 23, 29)]
        y = 0.7 * x1 - 0.2 * x2 + (row % 17) * 0.01 + (row % 29) * 0.002
        clusters = None
    else:
        raw_path, digest, fe_columns, x_columns = DATASETS[args.dataset]
        dataset_path = Path(raw_path)
        if sha256(dataset_path) != digest:
            raise AssertionError("dataset hash differs")
        frame = pd.read_parquet(
            dataset_path, columns=["y", *x_columns, *fe_columns])
        y = frame["y"].to_numpy(np.float64, copy=False)
        X = np.asfortranarray(
            frame[list(x_columns)].to_numpy(np.float64, copy=False))
        fes = [factorize(frame[name]) for name in fe_columns]
        clusters = fes[0]
    while len(fes) < args.fe_count:
        fes.append(fes[(len(fes) - 2) % 2].copy())
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reg = module.HdfeRegressor(
        se_type="cluster" if clusters is not None else "unadjusted",
        tol=args.tolerance, max_iter=100000,
        fit_intercept=True, num_threads=args.threads, drop_singletons=True,
        retain_fes=args.retain_fes, symmetric_sweep=False,
        absorption_method=args.method, tolerance_mode=args.tolerance_mode,
    )
    weights = np.ones(len(y), dtype=np.float64) if args.weights else None
    instruments = None
    endogenous = []
    if args.iv:
        instruments = np.asfortranarray(
            (X[:, 1] + 0.5 * X[:, 0]).reshape(-1, 1))
        endogenous = [1]
    # Since 2.27.0 an absorption that does not pass the convergence and
    # precision checks returns no estimates (fail-closed, uniform across
    # Stata, Python and R); record that outcome instead of crashing so the
    # parent can assert on it.
    returned = True
    error = ""
    try:
        reg.fit(
            y, X, fes=fes, clusters=clusters, weights=weights,
            instruments=instruments, endogenous_idx=endogenous, slopes=[])
    except RuntimeError as exc:
        returned = False
        error = str(exc)
        if "No estimates returned" not in error:
            raise
    result = {
        "returned": returned,
        "error": error,
        "method": int(reg.absorption_method_used),
        "iterations": int(reg.num_iterations_),
        "abs_residual_rel": float(reg.abs_residual_rel_),
        "converged": bool(reg.converged_),
        "precision": bool(reg.precision_certified_),
        "threads_used": int(reg.threads_used_),
        "gpu_used": bool(reg.gpu_used_),
        "gpu_attempted": bool(reg.gpu_attempted_),
        "gpu_status_code": int(reg.gpu_status_code_),
        "gpu_absorption_converged": bool(reg.gpu_absorption_converged_),
        "gpu_absorption_iterations": int(reg.gpu_absorption_iterations_),
        "policy": bool(getattr(reg, "auto_routing_retry_policy_enabled_", False)),
        "eligible": bool(getattr(reg, "auto_routing_retry_eligible_", False)),
        "fired": bool(getattr(reg, "auto_routing_retry_fired_", False)),
        "status": int(getattr(reg, "auto_routing_retry_status_", 0)),
        "primary_method": int(getattr(reg, "auto_routing_retry_primary_method_", -1)),
        "primary_iterations": int(getattr(reg, "auto_routing_retry_primary_iterations_", 0)),
        "primary_rel": float(getattr(reg, "auto_routing_retry_primary_abs_residual_rel_", 0.0)),
        "retry_iterations": int(getattr(reg, "auto_routing_retry_iterations_", 0)),
        "retry_rel": float(getattr(reg, "auto_routing_retry_abs_residual_rel_", 0.0)),
        "retry_seconds": float(getattr(reg, "auto_routing_retry_elapsed_seconds_", 0.0)),
    }
    Path(args.output).write_text(json.dumps(result, sort_keys=True) + "\n")
    np.savez(
        args.arrays, b=np.asarray(reg.coef_), V=np.asarray(reg.covariance_),
        residual=np.asarray(reg.residuals_), sample=np.asarray(reg.sample_index_),
    )
    return 0


def run_child(script, root, label, module, module_sha, *, dataset="credit2",
              method="auto", tolerance=1e-8,
              tolerance_mode="reghdfe-comparable", weights=False,
              retain_fes=False, threads=12, fe_count=2, iv=False,
              environment=None):
    output = root / f"{label}.json"
    arrays = root / f"{label}.npz"
    command = [
        sys.executable, "-B", "-I", "-S", str(script), "--child",
        "--module", str(module), "--module-sha256", module_sha,
        "--dataset", dataset, "--method", method,
        "--tolerance", repr(tolerance), "--tolerance-mode", tolerance_mode,
        "--threads", str(threads), "--fe-count", str(fe_count),
        "--output", str(output), "--arrays", str(arrays),
    ]
    if weights:
        command.append("--weights")
    if retain_fes:
        command.append("--retain-fes")
    if iv:
        command.append("--iv")
    env = os.environ.copy()
    env.update({
        "XHDFE_GPU_BACKEND": "cpu", "XHDFE_ABSORPTION_CACHE_MODE": "off",
        "XHDFE_FE_STRUCTURE_MODE": "off", "XHDFE_MOBILITY_MODE": "off",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
    })
    if environment:
        env.update(environment)
    subprocess.run(command, check=True, env=env, timeout=120)
    return json.loads(output.read_text()), np.load(arrays, allow_pickle=False)


def same_arrays(left, right) -> bool:
    return all(np.array_equal(left[name], right[name]) for name in left.files)


def parent(args) -> int:
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="xhdfe-auto-retry-") as raw:
        root = Path(raw)
        fired, fired_arrays = run_child(
            script, root, "fired", args.module, args.module_sha256)
        explicit, explicit_arrays = run_child(
            script, root, "explicit", args.module, args.module_sha256,
            method="mlsmr")
        disabled, disabled_arrays = run_child(
            script, root, "disabled", args.module, args.module_sha256,
            environment={"XHDFE_AUTO_ROUTING_RETRY": "0"})
        old, old_arrays = run_child(
            script, root, "old", args.old_module, args.old_module_sha256)
        # The kill-switch identity is checked against the old module with the
        # retry disabled as well: an old module that already carries the
        # automatic retry (2.26.0 or later) fires it in its default run.
        old_disabled, old_disabled_arrays = run_child(
            script, root, "old_disabled", args.old_module, args.old_module_sha256,
            environment={"XHDFE_AUTO_ROUTING_RETRY": "0"})
        failed, failed_arrays = run_child(
            script, root, "failed", args.module, args.module_sha256,
            environment={"XHDFE_TEST_AUTO_ROUTING_RETRY_FAIL": "1"})
        below, below_arrays = run_child(
            script, root, "below", args.module, args.module_sha256,
            dataset="credit")
        below_off, below_off_arrays = run_child(
            script, root, "below_off", args.module, args.module_sha256,
            dataset="credit", environment={"XHDFE_AUTO_EARLY_ROUTING": "0"})
        # Early routing (status 4): the switch taken and its identities.
        no_early = {"XHDFE_AUTO_EARLY_ROUTING": "0"}
        forced_failure = {"XHDFE_TEST_AUTO_ROUTING_RETRY_FAIL": "1"}
        handoff_cap = {"XHDFE_CG_ITER_CAP": "10"}
        taken = "pf_bench_difficult_n100k_3fe"
        early, early_arrays = run_child(
            script, root, "early", args.module, args.module_sha256, dataset=taken)
        early_off, early_off_arrays = run_child(
            script, root, "early_off", args.module, args.module_sha256,
            dataset=taken, environment=no_early)
        early_failed, early_failed_arrays = run_child(
            script, root, "early_failed", args.module, args.module_sha256,
            dataset=taken, environment=forced_failure)
        early_off_failed, early_off_failed_arrays = run_child(
            script, root, "early_off_failed", args.module, args.module_sha256,
            dataset=taken, environment={**forced_failure, **no_early})
        stays, stays_arrays = run_child(
            script, root, "stays", args.module, args.module_sha256,
            dataset="synthetic-uniform-hard")
        stays_off, stays_off_arrays = run_child(
            script, root, "stays_off", args.module, args.module_sha256,
            dataset="synthetic-uniform-hard", environment=no_early)
        handoff, handoff_arrays = run_child(
            script, root, "handoff", args.module, args.module_sha256,
            dataset="patents", environment=handoff_cap)
        handoff_off, handoff_off_arrays = run_child(
            script, root, "handoff_off", args.module, args.module_sha256,
            dataset="patents", environment={**handoff_cap, **no_early})
        gate, gate_arrays = run_child(
            script, root, "gate", args.module, args.module_sha256, dataset="github")
        gate_off, gate_off_arrays = run_child(
            script, root, "gate_off", args.module, args.module_sha256,
            dataset="github", environment=no_early)
        early_cache_path = root / "early.cache"
        early_cache_write, early_cache_write_arrays = run_child(
            script, root, "early_cache_write", args.module, args.module_sha256,
            dataset=taken,
            environment={
                "XHDFE_ABSORPTION_CACHE": str(early_cache_path),
                "XHDFE_ABSORPTION_CACHE_MODE": "write",
            })
        early_cache_read, early_cache_read_arrays = run_child(
            script, root, "early_cache_read", args.module, args.module_sha256,
            dataset=taken,
            environment={
                "XHDFE_ABSORPTION_CACHE": str(early_cache_path),
                "XHDFE_ABSORPTION_CACHE_MODE": "read",
            })
        custom, _ = run_child(
            script, root, "custom", args.module, args.module_sha256,
            tolerance=1e-10)
        fast, _ = run_child(
            script, root, "fast", args.module, args.module_sha256,
            tolerance_mode="xhdfe-fast")
        weighted, _ = run_child(
            script, root, "weighted", args.module, args.module_sha256,
            weights=True)
        retained, _ = run_child(
            script, root, "retained", args.module, args.module_sha256,
            retain_fes=True)
        actual_mlsmr, _ = run_child(
            script, root, "actual_mlsmr", args.module, args.module_sha256,
            environment={
                "XHDFE_AUTO_MLSMR_MIN_ROWS": "1",
                "XHDFE_AUTO_MLSMR_MIN_LEVELS": "1",
                "XHDFE_AUTO_MLSMR_RHO_THRESHOLD": "0",
                "XHDFE_AUTO_MLSMR_PARITY_SMALL_N_RHO_THRESHOLD": "0",
            })
        five_fe, _ = run_child(
            script, root, "five_fe", args.module, args.module_sha256,
            fe_count=5)
        huge_four_fe, _ = run_child(
            script, root, "huge_four_fe", args.module, args.module_sha256,
            dataset="synthetic_huge4", fe_count=4)
        iv_fit, _ = run_child(
            script, root, "iv", args.module, args.module_sha256, iv=True)
        one_thread, _ = run_child(
            script, root, "one_thread", args.module, args.module_sha256,
            threads=1)
        one_thread_explicit, one_thread_explicit_arrays = run_child(
            script, root, "one_thread_explicit", args.module,
            args.module_sha256, method="mlsmr", threads=1)
        cache_path = root / "retry.cache"
        cache_write, cache_write_arrays = run_child(
            script, root, "cache_write", args.module, args.module_sha256,
            environment={
                "XHDFE_ABSORPTION_CACHE": str(cache_path),
                "XHDFE_ABSORPTION_CACHE_MODE": "write",
            })
        cache_read, cache_read_arrays = run_child(
            script, root, "cache_read", args.module, args.module_sha256,
            environment={
                "XHDFE_ABSORPTION_CACHE": str(cache_path),
                "XHDFE_ABSORPTION_CACHE_MODE": "read",
            })
        old_cache_path = root / "old.cache"
        run_child(
            script, root, "old_cache_write", args.old_module,
            args.old_module_sha256,
            environment={
                "XHDFE_ABSORPTION_CACHE": str(old_cache_path),
                "XHDFE_ABSORPTION_CACHE_MODE": "write",
            })
        generation_miss, generation_miss_arrays = run_child(
            script, root, "generation_miss", args.module,
            args.module_sha256,
            environment={
                "XHDFE_ABSORPTION_CACHE": str(old_cache_path),
                "XHDFE_ABSORPTION_CACHE_MODE": "read",
            })

        assert fired["policy"] and fired["eligible"] and fired["fired"]
        assert fired["status"] == 2 and fired["method"] == 6
        assert fired["primary_method"] == 1 and fired["primary_rel"] > 1e-11
        assert fired["retry_rel"] <= 1e-11 and fired["iterations"] == fired["retry_iterations"]
        assert fired["threads_used"] == 12 and same_arrays(fired_arrays, explicit_arrays)
        assert explicit["status"] == 0 and not explicit["fired"]
        assert not disabled["policy"] and disabled["status"] == 0
        assert same_arrays(disabled_arrays,
                           old_arrays if not old.get("policy") else old_disabled_arrays)
        assert failed["status"] == 3 and failed["fired"] and failed["method"] == 1
        assert same_arrays(failed_arrays, disabled_arrays)
        assert below["policy"] and below["eligible"] and not below["fired"]
        assert below["status"] == 1 and below["abs_residual_rel"] <= 1e-11
        for excluded in (custom, fast, weighted):
            assert not excluded["eligible"] and not excluded["fired"]
        # retain_fes keeps the alpha-capturing Gauss-Seidel path, which stops
        # on credit2 at a relative residual above the comparable tolerance
        # (4.97e-8 > 1e-8, 17 iterations); 2.26.2 returned those uncertified
        # estimates with converged=0. Since 2.27.0 the projection is then
        # recomputed without alpha capture by the ordinary route (here the
        # retry to MLSMR fires, as without savefe), certified, and the fixed
        # effects are recovered from the partial: the result must be certified
        # and carry the retry diagnostics of that projection.
        assert retained["returned"] and retained["converged"] and retained["precision"]
        assert retained["eligible"] and retained["fired"]
        for scenario in (fired, explicit, disabled, old, failed, below, custom, fast,
                         weighted, actual_mlsmr, five_fe, huge_four_fe, iv_fit,
                         one_thread, one_thread_explicit, cache_write, cache_read,
                         generation_miss):
            assert scenario.get("returned", True), scenario.get("error", "")
        assert actual_mlsmr["method"] == 6
        assert actual_mlsmr["primary_method"] == 6
        assert not actual_mlsmr["eligible"] and not actual_mlsmr["fired"]
        for excluded in (five_fe, huge_four_fe, iv_fit):
            assert excluded["policy"]
            assert not excluded["eligible"] and not excluded["fired"]
        assert one_thread["precision"] and one_thread["eligible"]
        assert one_thread["fired"] and one_thread["status"] == 2
        assert same_arrays(np.load(root / "one_thread.npz"),
                           one_thread_explicit_arrays)
        assert cache_write["status"] == 2 and cache_read == cache_write
        assert same_arrays(cache_write_arrays, cache_read_arrays)
        assert generation_miss["status"] == 2
        assert same_arrays(generation_miss_arrays, fired_arrays)

        # Early routing, predictive rule: pf_bench_difficult_n100k_3fe leaves
        # its accelerated sweep; the retry's MLSMR runs at once and returns
        # exactly what the retry returns with early routing off (status 2).
        for scenario in (early, early_off, early_failed, early_off_failed, stays,
                         stays_off, handoff, handoff_off, gate, gate_off,
                         early_cache_write, early_cache_read, below_off):
            assert scenario.get("returned", True), scenario.get("error", "")
        assert early["policy"] and early["eligible"] and early["fired"]
        assert early["status"] == 4 and early["method"] == 6
        # Checkpoint 16: where the D1 trace put this fit above M = 30.
        assert early["primary_method"] == 1 and early["primary_iterations"] == 16
        assert early["primary_rel"] > 1e-11 and early["retry_rel"] <= 1e-11
        assert early["iterations"] == early["retry_iterations"]
        assert early_off["status"] == 2 and early_off["method"] == 6
        assert early["primary_iterations"] < early_off["primary_iterations"]
        assert early["iterations"] == early_off["iterations"]
        assert same_arrays(early_arrays, early_off_arrays)
        # Fail-safe: when MLSMR fails after the leave, the primary runs again
        # with early routing off and today's route returns its result.
        compared = ("status", "method", "iterations", "primary_method",
                    "primary_iterations", "primary_rel", "retry_iterations",
                    "retry_rel", "converged", "precision")
        assert early_failed["status"] == 3 and early_failed["method"] == 1
        assert all(early_failed[key] == early_off_failed[key] for key in compared)
        assert same_arrays(early_failed_arrays, early_off_failed_arrays)
        # Not taken: below the trigger (credit) and a prediction below the
        # margin at every checkpoint (synthetic-uniform-hard).
        assert below_off["status"] == 1 and same_arrays(below_arrays, below_off_arrays)
        assert stays["status"] == 1 and stays_off["status"] == 1
        assert stays["iterations"] == stays_off["iterations"] >= 8
        assert same_arrays(stays_arrays, stays_off_arrays)
        # Structural rule: patents, its accelerator capped at 10 iterations,
        # does not hand its iterate to a conjugate-gradient stage.
        assert handoff["status"] == 4 and handoff["method"] == 6
        assert handoff["primary_iterations"] == 10 and handoff["primary_rel"] == 0.0
        assert handoff["iterations"] == handoff["retry_iterations"]
        if handoff_off["method"] == 6:
            # Today's route also ends in MLSMR (the retry, or the restart
            # after a breakdown): the same result.
            assert same_arrays(handoff_arrays, handoff_off_arrays)
        else:
            # Today's hand-off stage converged below the trigger and was
            # kept: the accepted case, two certified results.
            assert handoff_off["status"] == 1 and handoff_off["precision"]
            assert np.array_equal(handoff_arrays["sample"], handoff_off_arrays["sample"])
            assert np.allclose(handoff_arrays["b"], handoff_off_arrays["b"],
                               rtol=1e-6, atol=1e-10)
        # A stage entered at a direct gate (github): the predictive rule
        # decides on the stage's own iterations and leaves at 64. The result
        # is today's (MLSMR after the breakdown restart), with MLSMR
        # iterations only.
        # With F1 the CG may converge above the 1e-11 retry trigger and the same MLSMR runs afterwards (status 2): the route differs, the result does not.
        assert gate_off["status"] in (0, 2) and gate_off["method"] == 6
        assert gate["status"] == 4 and gate["method"] == 6
        assert gate["primary_iterations"] == 64 and gate["primary_rel"] > 1e-11
        assert gate["iterations"] == gate["retry_iterations"] <= gate_off["iterations"]
        assert same_arrays(gate_arrays, gate_off_arrays)
        # Cache round trip with status 4.
        assert early_cache_write["status"] == 4 and early_cache_read == early_cache_write
        assert same_arrays(early_cache_write_arrays, early_cache_read_arrays)
        assert same_arrays(early_cache_write_arrays, early_arrays)

    print(
        "AUTO_ROUTING_RETRY_PASS fired_bitidentical=1 kill_switch_old_identity=1 "
        "failure_primary_identity=1 below_trigger=1 exclusions=4 "
        "actual_mlsmr_no_retry=1 promotion_band_exclusions=2 iv_excluded=1 "
        "cache_roundtrip=1 old_generation_miss=1 one_thread_bitidentical=1 "
        "early_taken_bitidentical=1 early_failsafe_identity=1 early_not_taken=2 "
        f"early_handoff=1 early_handoff_today_mlsmr={int(handoff_off['method'] == 6)} "
        f"early_gate_status={gate['status']} early_kill_switch=1 early_cache_roundtrip=1"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--old-module", type=Path)
    parser.add_argument("--old-module-sha256")
    parser.add_argument(
        "--dataset", choices=(*DATASETS, "synthetic_huge4"), default="credit2")
    parser.add_argument("--method", default="auto")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--tolerance-mode", default="reghdfe-comparable")
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--retain-fes", action="store_true")
    parser.add_argument("--fe-count", type=int, default=2)
    parser.add_argument("--iv", action="store_true")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arrays", type=Path)
    args = parser.parse_args()
    if args.child:
        return child(args)
    if args.old_module is None or not args.old_module_sha256:
        raise SystemExit("parent requires old module identity")
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
