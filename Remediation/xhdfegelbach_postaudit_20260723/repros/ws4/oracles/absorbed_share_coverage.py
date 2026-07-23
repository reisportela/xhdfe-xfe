#!/usr/bin/env python3
"""Focused implementation MC for an absorbed-target joint base share.

The DGP and population truth are the same as the audit's MC2, but this script
does not run a dense-LSDV oracle on every replication.  Oracle equality is a
separate deterministic gate; omitting that repeated dense solve makes it
practical to distinguish a 100-replication fluctuation from actual coverage.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = Path("/home/mangelo/Documents/GitHub/xhdfe")
sys.path.insert(0, str(REPO))


def _load_requested_extension() -> str | None:
    path = os.environ.get("XHDFE_GELBACH_EXTENSION")
    if not path:
        return None
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["py_hdfe_v11"] = module
    spec.loader.exec_module(module)
    return path


EXTENSION = _load_requested_extension()
import xhdfe.gelbach as gb  # noqa: E402


Z95 = 1.959963984540054
TRUTH = 1.0 / 3.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=150)
    parser.add_argument("--periods", type=int, default=5)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument(
        "--output",
        default=str(HERE / "absorbed_share_coverage.json"),
    )
    args = parser.parse_args()
    if args.reps <= 0 or args.workers < 30 or args.periods < 2:
        parser.error("positive reps, at least 30 workers, and at least 2 periods")

    estimates = np.empty(args.reps)
    ses = np.empty(args.reps)
    total_identity_error = np.empty(args.reps)
    for rep in range(args.reps):
        rng = np.random.default_rng(args.seed + rep)
        worker = np.repeat(np.arange(args.workers), args.periods)
        n = worker.size
        female = rng.integers(0, 2, args.workers).astype(float)[worker]
        exper = (
            np.tile(np.arange(args.periods), args.workers)
            + rng.normal(0.0, 0.3, n)
        )
        job = 0.4 * female + 0.15 * exper + rng.normal(size=n)
        eta = rng.normal(0.0, 0.6, args.workers)[worker]
        y = (
            0.15 * female
            + 0.05 * exper
            + 0.5 * job
            + 0.25 * female
            + eta
            + rng.normal(0.0, 0.7, n)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = gb.decompose(
                y,
                np.column_stack([female, exper]),
                {"job": job},
                {"worker": worker},
                vce="cluster",
                cluster=worker,
                absorbed_targets=[0],
                x1_names=["female", "exper"],
                focal="female",
            )
            rows = gb.tidy(
                result,
                share="base",
                include_total=False,
                include_full=False,
            )
        row = next(
            item
            for item in rows
            if item["coefficient"] == "female" and item["component"] == "job"
        )
        estimates[rep] = row["share"]
        ses[rep] = row["share_std_error"]
        total_identity_error[rep] = abs(
            result["total"]["se"][0]
            - np.sqrt(result["base_cov"][0, 0])
        )

    covered = np.abs(estimates - TRUTH) <= Z95 * ses
    coverage = float(covered.mean())
    empirical_sd = float(estimates.std(ddof=1))
    out = {
        "extension": EXTENSION or "repository package artifact",
        "reps": args.reps,
        "seed0": args.seed,
        "workers_or_clusters": args.workers,
        "periods": args.periods,
        "population_share_job_over_base": TRUTH,
        "mean_estimate": float(estimates.mean()),
        "bias": float(estimates.mean() - TRUTH),
        "empirical_sd": empirical_sd,
        "mean_reported_se": float(ses.mean()),
        "se_to_sd_ratio": float(ses.mean() / empirical_sd),
        "coverage": coverage,
        "coverage_mc_se": float(
            np.sqrt(coverage * (1.0 - coverage) / args.reps)
        ),
        "max_total_se_base_se_identity_error": float(
            total_identity_error.max()
        ),
    }
    output = Path(args.output)
    output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
