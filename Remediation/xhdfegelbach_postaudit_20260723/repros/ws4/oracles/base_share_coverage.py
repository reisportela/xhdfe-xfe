#!/usr/bin/env python3
"""Focused Monte Carlo for the joint ``share=base`` confidence interval.

This is deliberately narrower and faster than ``mc_coverage.py``.  It tests
the implementation returned by ``xhdfe.gelbach`` (not a reimplementation of
the estimator) under a DGP whose population linear projections are available
in closed form.

The clustered experiment accepts the number of independent clusters as an
argument.  This matters: a normal-reference CRVE interval can undercover with
only a few dozen clusters even when its covariance algebra is correct.  The
script therefore reports coverage, Monte-Carlo standard error, coefficient
bias, empirical sampling SD, and mean reported SE instead of selecting a
single favourable seed.

To test a newly built extension without overwriting production artifacts:

  XHDFE_GELBACH_EXTENSION=/path/to/py_hdfe_v11...so \
    python base_share_coverage.py --reps 5000 --clusters 40 120
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
TRUE = {
    "A": 0.48 / 1.48,
    "B": -0.20 / 1.48,
}


def _summarize(estimates: np.ndarray, ses: np.ndarray, truth: float) -> dict:
    covered = np.abs(estimates - truth) <= Z95 * ses
    reps = estimates.size
    coverage = float(covered.mean())
    return {
        "truth": float(truth),
        "mean_estimate": float(estimates.mean()),
        "bias": float(estimates.mean() - truth),
        "empirical_sd": float(estimates.std(ddof=1)),
        "mean_reported_se": float(ses.mean()),
        "se_to_sd_ratio": float(ses.mean() / estimates.std(ddof=1)),
        "coverage": coverage,
        "coverage_mc_se": float(np.sqrt(coverage * (1.0 - coverage) / reps)),
    }


def run_design(
    *,
    reps: int,
    seed0: int,
    vce: str,
    n_clusters: int,
    rows_per_cluster: int,
) -> dict:
    clustered = vce == "cluster"
    n = n_clusters * rows_per_cluster
    values = {name: [] for name in TRUE}
    ses = {name: [] for name in TRUE}
    for rep in range(reps):
        rng = np.random.default_rng(seed0 + rep)
        cluster = np.repeat(np.arange(n_clusters), rows_per_cluster)
        if clustered:
            z = rng.normal(0.0, np.sqrt(0.30), n_clusters)[cluster]
            x = z + rng.normal(0.0, np.sqrt(0.70), n)
        else:
            x = rng.normal(size=n)
        a = 0.6 * x + rng.normal(size=n)
        b = -0.4 * x + rng.normal(size=n)
        cluster_error = (
            rng.normal(0.0, np.sqrt(0.50), n_clusters)[cluster]
            if clustered
            else 0.0
        )
        y = 1.2 * x + 0.8 * a + 0.5 * b + cluster_error + rng.normal(size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = gb.decompose(
                y,
                x[:, None],
                {"A": a, "B": b},
                vce=vce,
                cluster=cluster if clustered else None,
            )
            tidy = gb.tidy(
                result,
                share="base",
                include_total=False,
                include_full=False,
            )
        rows = {
            row["component"]: row
            for row in tidy
            if row["coefficient"] == "x1_1"
        }
        for name in TRUE:
            values[name].append(rows[name]["share"])
            ses[name].append(rows[name]["share_std_error"])
    return {
        "reps": reps,
        "seed0": seed0,
        "vce": vce,
        "n": n,
        "n_clusters": n_clusters if clustered else None,
        "rows_per_cluster": rows_per_cluster if clustered else None,
        "normal_critical_value": Z95,
        "components": {
            name: _summarize(
                np.asarray(values[name]), np.asarray(ses[name]), truth
            )
            for name, truth in TRUE.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--clusters", type=int, nargs="+", default=[40, 120])
    parser.add_argument("--rows-per-cluster", type=int, default=20)
    parser.add_argument(
        "--output",
        default=str(HERE / "base_share_coverage.json"),
    )
    args = parser.parse_args()
    if args.reps <= 0 or min(args.clusters) < 2 or args.rows_per_cluster <= 0:
        parser.error("reps/rows must be positive and clusters must be >=2")

    # Keep the iid N equal to the largest clustered design.  Both the classical
    # and heteroskedasticity-robust covariance paths are exercised.
    iid_groups = max(args.clusters)
    out = {
        "extension": EXTENSION or "repository package artifact",
        "dgp": (
            "y=1.2*x+0.8*A+0.5*B+c_g+e; "
            "A=0.6*x+e_A; B=-0.4*x+e_B"
        ),
        "population_base_coefficient": 1.48,
        "population_component_changes": {"A": 0.48, "B": -0.20},
        "designs": {},
    }
    out["designs"]["iid_unadjusted"] = run_design(
        reps=args.reps,
        seed0=110_000,
        vce="unadjusted",
        n_clusters=iid_groups,
        rows_per_cluster=args.rows_per_cluster,
    )
    # The audit's original 100-replication MC1 sequence began at seed 50000.
    # Retain that exact sequence as a sensitivity check, rather than dismissing
    # its low clustered realization after changing the seed.
    out["designs"]["iid_unadjusted_audit_seed"] = run_design(
        reps=args.reps,
        seed0=50_000,
        vce="unadjusted",
        n_clusters=40,
        rows_per_cluster=20,
    )
    out["designs"]["iid_robust"] = run_design(
        reps=args.reps,
        seed0=210_000,
        vce="robust",
        n_clusters=iid_groups,
        rows_per_cluster=args.rows_per_cluster,
    )
    for n_clusters in args.clusters:
        out["designs"][f"cluster_{n_clusters}"] = run_design(
            reps=args.reps,
            seed0=310_000 + n_clusters * 10_000,
            vce="cluster",
            n_clusters=n_clusters,
            rows_per_cluster=args.rows_per_cluster,
        )
    if 40 in args.clusters:
        out["designs"]["cluster_40_audit_seed"] = run_design(
            reps=args.reps,
            seed0=50_000,
            vce="cluster",
            n_clusters=40,
            rows_per_cluster=20,
        )

    output = Path(args.output)
    output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for name, design in out["designs"].items():
        print(f"== {name}: reps={design['reps']} n={design['n']}")
        for component, stats in design["components"].items():
            print(
                f"  {component}: coverage={stats['coverage']:.4f} "
                f"(MCSE={stats['coverage_mc_se']:.4f}), "
                f"SE/SD={stats['se_to_sd_ratio']:.4f}, "
                f"bias={stats['bias']:+.3e}"
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
