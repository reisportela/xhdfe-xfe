#!/usr/bin/env python3
"""Paired Gelbach performance triage for a persistently busy shared host.

This harness has a deliberately narrow claim:

* the post-audit candidate is compared with the preserved pre-change 2.19.0
  Release artifact in interleaved ABBA/BAAB blocks;
* the absorbed-target path is timed on the candidate as contextual evidence,
  while the certification decision is based on paired standard-path ratios;
* the 30% margin is a catastrophic-regression screen, not an acceptable
  performance loss and not a substitute for a quiet-host certification run.

Each binary is loaded in a fresh worker process.  Fixture construction and
warm-up are outside the timed region.  The runner records short CPU samples,
load averages, and continuous vmstat/NVIDIA telemetry alongside every block.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


CELLS = tuple(
    (vce, weighted)
    for vce in ("unadjusted", "cluster")
    for weighted in (False, True)
)


def cell_name(vce: str, weighted: bool) -> str:
    return f"{vce}/{'w' if weighted else 'unw'}"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fixture(n: int) -> dict[str, Any]:
    """Reproduce the audit's deterministic standard/absorbed fixtures."""
    rng = np.random.default_rng(20260720)
    worker = (np.arange(n) % 5000) + 1
    firm = ((np.arange(n) // 5 + 7 * worker) % 1000) + 1
    exper = (np.arange(n) % 10) + rng.normal(0, 0.2, n)
    focal_standard = rng.integers(0, 2, n).astype(float)
    focal_absorbed = (worker % 2).astype(float)

    def make(focal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        job = 0.3 * focal + 0.2 * exper + rng.normal(0, 1, n)
        y = (
            0.25 * focal
            + 0.08 * exper
            + 0.5 * job
            + 0.03 * worker
            - 0.02 * firm
            + rng.normal(0, 1, n)
        )
        return np.column_stack([focal, exper]), job.reshape(-1, 1), y

    x1_standard, x2_standard, y_standard = make(focal_standard)
    x1_absorbed, x2_absorbed, y_absorbed = make(focal_absorbed)
    return {
        "standard": (x1_standard, x2_standard, y_standard),
        "absorbed": (x1_absorbed, x2_absorbed, y_absorbed),
        "fes": [worker.astype(np.int64), firm.astype(np.int64)],
        "cluster": worker.astype(np.int64),
        "weights": 1.0 + (np.arange(n) % 3),
    }


def load_extension(module_path: pathlib.Path):
    module_path = module_path.resolve()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct extension loader for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = pathlib.Path(module.__file__).resolve()
    if loaded != module_path:
        raise RuntimeError(f"requested {module_path}, loaded {loaded}")
    return module


def worker(args: argparse.Namespace) -> int:
    module_path = pathlib.Path(args.module).resolve()
    os.environ["XHDFE_GPU_BACKEND"] = args.backend
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    module = load_extension(module_path)
    data = fixture(args.n)
    x1, x2, y = data[args.mode]

    def call(vce: str, weighted: bool):
        kwargs: dict[str, Any] = {
            "x2_group_sizes": [1],
            "fes": data["fes"],
            "vce": vce,
            "gamma0": False,
            "cov0": False,
            "tol": 1e-8,
            "num_threads": args.threads,
        }
        if vce == "cluster":
            kwargs["cluster"] = data["cluster"]
        if weighted:
            kwargs["weights"] = data["weights"]
        if args.mode == "absorbed":
            kwargs["absorbed_x1"] = [0]
        return module.gelbach_decompose(y, x1, x2, **kwargs)

    # Untimed warm-up makes first-load, allocation, and CUDA-context effects
    # explicit rather than charging them to one side of the A/B pair.
    for vce, weighted in CELLS:
        result = call(vce, weighted)
        if not bool(result["converged"]):
            raise RuntimeError(f"warm-up did not converge: {cell_name(vce, weighted)}")

    samples: dict[str, list[dict[str, Any]]] = {
        cell_name(vce, weighted): [] for vce, weighted in CELLS
    }
    rng = random.Random(20260720 + 1009 * args.round)
    for inner in range(args.inner_reps):
        order = list(CELLS)
        rng.shuffle(order)
        for vce, weighted in order:
            started = time.perf_counter()
            result = call(vce, weighted)
            elapsed = time.perf_counter() - started
            if not bool(result["converged"]):
                raise RuntimeError(f"timed call did not converge: {cell_name(vce, weighted)}")
            if (
                args.backend == "cuda"
                and "gpu_used" in result
                and not bool(result["gpu_used"])
            ):
                raise RuntimeError(
                    f"CUDA request silently fell back: {cell_name(vce, weighted)} "
                    f"status={result.get('gpu_status')} backend={result.get('gpu_backend')}"
                )
            samples[cell_name(vce, weighted)].append(
                {
                    "inner": inner,
                    "seconds": elapsed,
                    "identity_gap": float(result["identity_gap"]),
                    "notes": str(result.get("notes", "")),
                    "gpu_requested": result.get("gpu_requested"),
                    "gpu_used": result.get("gpu_used"),
                    "gpu_backend": result.get("gpu_backend"),
                    "gpu_status": result.get("gpu_status"),
                }
            )

    payload = {
        "schema": 1,
        "module": str(module_path),
        "module_sha256": sha256(module_path),
        "backend_requested": args.backend,
        "mode": args.mode,
        "round": args.round,
        "n": args.n,
        "threads": args.threads,
        "inner_reps": args.inner_reps,
        "load_at_end": list(os.getloadavg()),
        "samples": samples,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def read_cpu() -> tuple[int, int]:
    fields = pathlib.Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def cpu_busy_sample(seconds: float = 0.5) -> float:
    total0, idle0 = read_cpu()
    time.sleep(seconds)
    total1, idle1 = read_cpu()
    delta = total1 - total0
    return 100.0 * (1.0 - (idle1 - idle0) / delta) if delta > 0 else float("nan")


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=10)
        return {"ok": True, "raw": result.stdout.strip()}
    except Exception as exc:  # telemetry failure must be visible, not fatal to CPU A/B
        return {"ok": False, "error": repr(exc)}


def telemetry_point() -> dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "load": list(os.getloadavg()),
        "cpu_busy_pct_0p5s": cpu_busy_sample(),
        "gpu": gpu_snapshot(),
    }


def start_monitor(command: list[str], output: pathlib.Path):
    handle = output.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    except Exception:
        handle.close()
        raise
    return process, handle


def stop_monitor(item) -> None:
    process, handle = item
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    handle.close()


def run_one(
    script: pathlib.Path,
    output_dir: pathlib.Path,
    module: pathlib.Path,
    label: str,
    backend: str,
    mode: str,
    round_number: int,
    args: argparse.Namespace,
) -> pathlib.Path:
    destination = output_dir / backend / mode / f"pair_{round_number + 1:02d}_{label}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = telemetry_point()
    command = [
        sys.executable,
        str(script),
        "worker",
        "--module",
        str(module),
        "--backend",
        backend,
        "--mode",
        mode,
        "--round",
        str(round_number),
        "--n",
        str(args.n),
        "--threads",
        str(args.threads),
        "--inner-reps",
        str(args.inner_reps),
        "--output",
        str(destination),
    ]
    started = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, timeout=420)
    wall = time.perf_counter() - started
    after = telemetry_point()
    telemetry = {
        "label": label,
        "backend": backend,
        "mode": mode,
        "round": round_number,
        "command": command,
        "returncode": result.returncode,
        "wall_seconds": wall,
        "before": before,
        "after": after,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    telemetry_path = destination.with_suffix(".telemetry.json")
    telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"worker failed ({backend}/{mode}/{label}): {result.stderr}")
    print(
        f"{backend:4s} pair={round_number + 1:02d} {mode:8s} {label:9s} "
        f"wall={wall:6.2f}s load={after['load'][0]:5.1f} "
        f"background_busy={after['cpu_busy_pct_0p5s']:5.1f}%",
        flush=True,
    )
    return destination


def values(path: pathlib.Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        cell: [float(row["seconds"]) for row in rows]
        for cell, rows in payload["samples"].items()
    }


def bootstrap_median_ci(observations: list[float], draws: int = 20000) -> list[float]:
    if not observations:
        return [float("nan"), float("nan")]
    if len(observations) == 1:
        return [observations[0], observations[0]]
    rng = np.random.default_rng(20260720)
    source = np.asarray(observations, dtype=float)
    sampled = source[rng.integers(0, source.size, size=(draws, source.size))]
    medians = np.median(sampled, axis=1)
    return [float(x) for x in np.percentile(medians, [2.5, 97.5])]


def one_sided_sign_p(observations: list[float], margin: float, direction: str) -> float:
    """Exact sign-test tail under a null median at the supplied margin."""
    if direction == "below":
        successes = sum(value < margin for value in observations)
    elif direction == "above":
        successes = sum(value > margin for value in observations)
    else:
        raise ValueError(f"unknown sign-test direction: {direction}")
    n = len(observations)
    if n == 0:
        return float("nan")
    return sum(math.comb(n, k) for k in range(successes, n + 1)) / (2.0**n)


def summarize(
    output_dir: pathlib.Path, pairs: int, backends: tuple[str, ...] = ("cpu", "cuda")
) -> dict[str, Any]:
    summary: dict[str, Any] = {"schema": 1, "pairs": pairs, "backends": {}}
    for backend in backends:
        backend_summary: dict[str, Any] = {"standard_ab": {}, "absorbed_cost": {}}
        for cell in (cell_name(vce, weighted) for vce, weighted in CELLS):
            ratios: list[float] = []
            absorbed_ratios: list[float] = []
            baseline_seconds: list[float] = []
            candidate_seconds: list[float] = []
            absorbed_seconds: list[float] = []
            for round_number in range(pairs):
                stem = output_dir / backend / "standard" / f"pair_{round_number + 1:02d}"
                base = values(stem.with_name(stem.name + "_baseline.json"))[cell]
                candidate = values(stem.with_name(stem.name + "_candidate.json"))[cell]
                absorbed_path = (
                    output_dir
                    / backend
                    / "absorbed"
                    / f"pair_{round_number + 1:02d}_candidate.json"
                )
                absorbed = values(absorbed_path)[cell]
                base_median = statistics.median(base)
                candidate_median = statistics.median(candidate)
                absorbed_median = statistics.median(absorbed)
                baseline_seconds.append(base_median)
                candidate_seconds.append(candidate_median)
                absorbed_seconds.append(absorbed_median)
                ratios.append(candidate_median / base_median)
                absorbed_ratios.append(absorbed_median / candidate_median)

            ci = bootstrap_median_ci(ratios)
            p_below = one_sided_sign_p(ratios, 1.30, "below")
            p_above = one_sided_sign_p(ratios, 1.30, "above")
            if p_below < 0.05:
                decision = "rules_out_regression_ge_30pct"
            elif p_above < 0.05:
                decision = "confirms_regression_gt_30pct"
            else:
                decision = "inconclusive_at_30pct_margin"
            backend_summary["standard_ab"][cell] = {
                "pair_ratios_candidate_over_baseline": ratios,
                "median_ratio": statistics.median(ratios),
                "bootstrap_95pct_ci": ci,
                "one_sided_sign_p_median_below_1p30": p_below,
                "one_sided_sign_p_median_above_1p30": p_above,
                "decision": decision,
                "baseline_pair_medians_seconds": baseline_seconds,
                "candidate_pair_medians_seconds": candidate_seconds,
            }
            backend_summary["absorbed_cost"][cell] = {
                "pair_ratios_absorbed_over_candidate_standard": absorbed_ratios,
                "median_ratio": statistics.median(absorbed_ratios),
                "bootstrap_95pct_ci": bootstrap_median_ci(absorbed_ratios),
                "one_sided_sign_p_median_below_1p30": one_sided_sign_p(
                    absorbed_ratios, 1.30, "below"
                ),
                "absorbed_pair_medians_seconds": absorbed_seconds,
            }
        summary["backends"][backend] = backend_summary

    telemetry_files = sorted(output_dir.glob("**/*.telemetry.json"))
    busy: list[float] = []
    load1: list[float] = []
    for path in telemetry_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        for side in ("before", "after"):
            busy.append(float(record[side]["cpu_busy_pct_0p5s"]))
            load1.append(float(record[side]["load"][0]))
    summary["environment"] = {
        "telemetry_points": len(busy),
        "cpu_busy_pct_median": statistics.median(busy) if busy else None,
        "cpu_busy_pct_min": min(busy) if busy else None,
        "cpu_busy_pct_max": max(busy) if busy else None,
        "load1_median": statistics.median(load1) if load1 else None,
        "load1_min": min(load1) if load1 else None,
        "load1_max": max(load1) if load1 else None,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def available_pair_numbers(output_dir: pathlib.Path, backend: str) -> list[int]:
    standard = output_dir / backend / "standard"
    baseline = {
        int(path.name.split("_")[1])
        for path in standard.glob("pair_*_baseline.json")
        if not path.name.endswith(".telemetry.json")
    }
    candidate = {
        int(path.name.split("_")[1])
        for path in standard.glob("pair_*_candidate.json")
        if not path.name.endswith(".telemetry.json")
    }
    return sorted(baseline & candidate)


def pair_environment(output_dir: pathlib.Path, backend: str, pair_number: int) -> dict[str, Any]:
    paths = [
        output_dir / backend / "standard" / f"pair_{pair_number:02d}_baseline.telemetry.json",
        output_dir / backend / "standard" / f"pair_{pair_number:02d}_candidate.telemetry.json",
        output_dir / backend / "absorbed" / f"pair_{pair_number:02d}_candidate.telemetry.json",
    ]
    busy: list[float] = []
    loads: list[float] = []
    found: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        found.append(str(path))
        record = json.loads(path.read_text(encoding="utf-8"))
        for side in ("before", "after"):
            busy.append(float(record[side]["cpu_busy_pct_0p5s"]))
            loads.append(float(record[side]["load"][0]))
    return {
        "telemetry_files": found,
        "max_background_cpu_busy_pct": max(busy) if busy else None,
        "max_load1": max(loads) if loads else None,
    }


def analyze_partial(args: argparse.Namespace) -> int:
    """Analyze completed blocks after a host-regime interruption.

    Validity thresholds are environmental, not outcome-based.  They exclude a
    block if the short background sample reached near-total saturation or if
    load1 entered the distinct shock regime observed during the attempt.
    """
    output_dir = pathlib.Path(args.output_dir).resolve()
    summary: dict[str, Any] = {
        "schema": 1,
        "analysis": "partial_after_environmental_interruption",
        "validity_rule": {
            "max_background_cpu_busy_pct_strictly_below": args.max_background_busy,
            "max_load1_strictly_below": args.max_load1,
        },
        "backends": {},
    }
    for backend in args.backends:
        complete = available_pair_numbers(output_dir, backend)
        environments = {
            pair: pair_environment(output_dir, backend, pair) for pair in complete
        }
        valid = [
            pair
            for pair in complete
            if environments[pair]["max_background_cpu_busy_pct"] is not None
            and environments[pair]["max_background_cpu_busy_pct"]
            < args.max_background_busy
            and environments[pair]["max_load1"] < args.max_load1
        ]
        invalid = [pair for pair in complete if pair not in valid]
        backend_summary: dict[str, Any] = {
            "complete_pairs": complete,
            "valid_pairs": valid,
            "invalid_pairs": invalid,
            "pair_environment": environments,
            "standard_ab": {},
            "absorbed_cost": {},
        }
        for cell in (cell_name(vce, weighted) for vce, weighted in CELLS):
            ratios: list[float] = []
            absorbed_ratios: list[float] = []
            baseline_seconds: list[float] = []
            candidate_seconds: list[float] = []
            absorbed_seconds: list[float] = []
            absorbed_pairs: list[int] = []
            for pair in valid:
                stem = output_dir / backend / "standard" / f"pair_{pair:02d}"
                base = values(stem.with_name(stem.name + "_baseline.json"))[cell]
                candidate = values(stem.with_name(stem.name + "_candidate.json"))[cell]
                base_median = statistics.median(base)
                candidate_median = statistics.median(candidate)
                baseline_seconds.append(base_median)
                candidate_seconds.append(candidate_median)
                ratios.append(candidate_median / base_median)

                absorbed_path = (
                    output_dir / backend / "absorbed" / f"pair_{pair:02d}_candidate.json"
                )
                if absorbed_path.exists():
                    absorbed = values(absorbed_path)[cell]
                    absorbed_median = statistics.median(absorbed)
                    absorbed_pairs.append(pair)
                    absorbed_seconds.append(absorbed_median)
                    absorbed_ratios.append(absorbed_median / candidate_median)

            ci = bootstrap_median_ci(ratios)
            p_below = one_sided_sign_p(ratios, 1.30, "below")
            p_above = one_sided_sign_p(ratios, 1.30, "above")
            if ratios and p_below < 0.05:
                decision = "rules_out_regression_ge_30pct"
            elif ratios and p_above < 0.05:
                decision = "confirms_regression_gt_30pct"
            else:
                decision = "inconclusive_at_30pct_margin"
            backend_summary["standard_ab"][cell] = {
                "pair_ratios_candidate_over_baseline": ratios,
                "median_ratio": statistics.median(ratios) if ratios else None,
                "min_ratio": min(ratios) if ratios else None,
                "max_ratio": max(ratios) if ratios else None,
                "bootstrap_95pct_ci": ci,
                "one_sided_sign_p_median_below_1p30": p_below,
                "one_sided_sign_p_median_above_1p30": p_above,
                "decision": decision,
                "baseline_pair_medians_seconds": baseline_seconds,
                "candidate_pair_medians_seconds": candidate_seconds,
            }
            backend_summary["absorbed_cost"][cell] = {
                "pairs": absorbed_pairs,
                "pair_ratios_absorbed_over_candidate_standard": absorbed_ratios,
                "median_ratio": statistics.median(absorbed_ratios) if absorbed_ratios else None,
                "min_ratio": min(absorbed_ratios) if absorbed_ratios else None,
                "max_ratio": max(absorbed_ratios) if absorbed_ratios else None,
                "bootstrap_95pct_ci": bootstrap_median_ci(absorbed_ratios),
                "one_sided_sign_p_median_below_1p30": one_sided_sign_p(
                    absorbed_ratios, 1.30, "below"
                ),
                "absorbed_pair_medians_seconds": absorbed_seconds,
            }
        summary["backends"][backend] = backend_summary

    destination = output_dir / "partial_summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def pool_complete_runs(args: argparse.Namespace) -> int:
    """Pool every pair from independent, fully completed runs.

    This intentionally accepts only completed ``summary.json`` files and never
    filters individual pairs.  It is used when a first complete run is noisy
    enough to leave one cell inconclusive and a second, prospectively sized
    complete run is collected.
    """
    inputs = [pathlib.Path(path).resolve() for path in args.inputs]
    summaries = [
        json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for path in inputs
    ]
    pooled: dict[str, Any] = {
        "schema": 1,
        "analysis": "pooled_complete_runs_no_pair_exclusions",
        "margin_candidate_over_baseline": 1.30,
        "input_directories": [str(path) for path in inputs],
        "backends": {},
    }
    for backend in args.backends:
        contents: dict[str, Any] = {"standard_ab": {}}
        for cell in (cell_name(vce, weighted) for vce, weighted in CELLS):
            ratios: list[float] = []
            run_pair_counts: list[int] = []
            for summary in summaries:
                row = summary["backends"][backend]["standard_ab"][cell]
                run_ratios = [
                    float(value)
                    for value in row["pair_ratios_candidate_over_baseline"]
                ]
                ratios.extend(run_ratios)
                run_pair_counts.append(len(run_ratios))
            p_below = one_sided_sign_p(ratios, 1.30, "below")
            p_above = one_sided_sign_p(ratios, 1.30, "above")
            if p_below < 0.05:
                decision = "rules_out_regression_ge_30pct"
            elif p_above < 0.05:
                decision = "confirms_regression_gt_30pct"
            else:
                decision = "inconclusive_at_30pct_margin"
            contents["standard_ab"][cell] = {
                "run_pair_counts": run_pair_counts,
                "pair_ratios_candidate_over_baseline": ratios,
                "n_pairs": len(ratios),
                "n_below_1p30": sum(value < 1.30 for value in ratios),
                "median_ratio": statistics.median(ratios),
                "bootstrap_95pct_ci": bootstrap_median_ci(ratios),
                "one_sided_sign_p_median_below_1p30": p_below,
                "one_sided_sign_p_median_above_1p30": p_above,
                "decision": decision,
            }
        pooled["backends"][backend] = contents
    destination = pathlib.Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(pooled, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(pooled, indent=2))
    return 0


def runner(args: argparse.Namespace) -> int:
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script = pathlib.Path(__file__).resolve()
    modules = {
        "cpu": {
            "baseline": pathlib.Path(args.baseline_cpu).resolve(),
            "candidate": pathlib.Path(args.candidate_cpu).resolve(),
        },
        "cuda": {
            "baseline": pathlib.Path(args.baseline_cuda).resolve(),
            "candidate": pathlib.Path(args.candidate_cuda).resolve(),
        },
    }
    manifest = {
        backend: {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in paths.items()
        }
        for backend, paths in modules.items()
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    monitors = []
    try:
        monitors.append(
            start_monitor(
                ["vmstat", "-w", "1"],
                output_dir / "vmstat_1s.log",
            )
        )
        monitors.append(
            start_monitor(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
                    "--format=csv,noheader",
                    "--loop-ms=500",
                ],
                output_dir / "nvidia_gpu_500ms.csv",
            )
        )
        monitors.append(
            start_monitor(
                [
                    "nvidia-smi",
                    "--query-compute-apps=timestamp,pid,process_name,used_memory",
                    "--format=csv,noheader",
                    "--loop-ms=500",
                ],
                output_dir / "nvidia_apps_500ms.csv",
            )
        )

        for backend in args.backends:
            for round_number in range(args.pairs):
                # ABBA/BAAB start order controls monotone host drift.
                standard_order = (
                    ("baseline", "candidate")
                    if round_number % 4 in (0, 3)
                    else ("candidate", "baseline")
                )
                if round_number % 2:
                    run_one(
                        script,
                        output_dir,
                        modules[backend]["candidate"],
                        "candidate",
                        backend,
                        "absorbed",
                        round_number,
                        args,
                    )
                for label in standard_order:
                    run_one(
                        script,
                        output_dir,
                        modules[backend][label],
                        label,
                        backend,
                        "standard",
                        round_number,
                        args,
                    )
                if not round_number % 2:
                    run_one(
                        script,
                        output_dir,
                        modules[backend]["candidate"],
                        "candidate",
                        backend,
                        "absorbed",
                        round_number,
                        args,
                    )
    finally:
        for monitor in reversed(monitors):
            stop_monitor(monitor)

    summary = summarize(output_dir, args.pairs, tuple(args.backends))
    print(json.dumps(summary["environment"], indent=2))
    for backend, contents in summary["backends"].items():
        for cell, row in contents["standard_ab"].items():
            print(
                f"RESULT {backend:4s} {cell:16s} ratio={row['median_ratio']:.4f} "
                f"CI95=[{row['bootstrap_95pct_ci'][0]:.4f},"
                f"{row['bootstrap_95pct_ci'][1]:.4f}] {row['decision']}"
            )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)

    work = sub.add_parser("worker")
    work.add_argument("--module", required=True)
    work.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    work.add_argument("--mode", choices=("standard", "absorbed"), required=True)
    work.add_argument("--round", type=int, required=True)
    work.add_argument("--n", type=int, default=500_000)
    work.add_argument("--threads", type=int, default=16)
    work.add_argument("--inner-reps", type=int, default=3)
    work.add_argument("--output", required=True)
    work.set_defaults(function=worker)

    run = sub.add_parser("run")
    run.add_argument("--baseline-cpu", required=True)
    run.add_argument("--candidate-cpu", required=True)
    run.add_argument("--baseline-cuda", required=True)
    run.add_argument("--candidate-cuda", required=True)
    run.add_argument("--pairs", type=int, default=8)
    run.add_argument("--n", type=int, default=500_000)
    run.add_argument("--threads", type=int, default=16)
    run.add_argument("--inner-reps", type=int, default=3)
    run.add_argument("--backends", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    run.add_argument("--output-dir", required=True)
    run.set_defaults(function=runner)

    analyze = sub.add_parser("analyze-partial")
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--backends", nargs="+", choices=("cpu", "cuda"), default=["cpu"])
    analyze.add_argument("--max-background-busy", type=float, default=95.0)
    analyze.add_argument("--max-load1", type=float, default=55.0)
    analyze.set_defaults(function=analyze_partial)

    pool = sub.add_parser("pool-complete-runs")
    pool.add_argument("--inputs", nargs="+", required=True)
    pool.add_argument("--backends", nargs="+", choices=("cpu", "cuda"), default=["cpu"])
    pool.add_argument("--output", required=True)
    pool.set_defaults(function=pool_complete_runs)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
