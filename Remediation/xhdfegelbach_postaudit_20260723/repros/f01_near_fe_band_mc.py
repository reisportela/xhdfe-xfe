#!/usr/bin/env python3
"""Monte Carlo rationale and contract test for the F-01 warning band.

The exercise changes diagnostics only.  For each requested post-absorption
ratio it constructs an X1 target whose within-worker squared-norm share is
that ratio exactly, then records the dispersion of the reported movement,
the reported conditional-FE SE, and the public warning/mask contract.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from xhdfe import gelbach


# Use 9e-5 for the upper-band MC cell: constructing an exact 1e-4 target can
# round a few ulps to either side of the inclusive public boundary.
RATIOS = (4e-9, 1e-6, 9e-5, 1e-2)
REPS = int(os.environ.get("XHDFE_F01_MC_REPS", "40"))


def make_target(base, worker, rng, ratio):
    noise = rng.normal(size=base.size)
    counts = np.bincount(worker)
    sums = np.bincount(worker, weights=noise)
    within = noise - (sums / counts)[worker]
    base_ss = float(base @ base)
    within_ss = float(within @ within)
    eps = np.sqrt(ratio * base_ss / ((1.0 - ratio) * within_ss))
    target = base + eps * within
    achieved = float((eps * eps * within_ss) / (target @ target))
    return target, achieved


def one(seed, ratio):
    rng = np.random.default_rng(seed)
    workers, periods = 200, 5
    worker = np.repeat(np.arange(workers), periods)
    n = worker.size
    invariant = rng.integers(0, 2, workers).astype(float)[worker]
    exper = np.tile(np.arange(periods), workers) + rng.normal(0, 0.2, n)
    job = 0.4 * invariant + 0.15 * exper + rng.normal(size=n)
    alpha = rng.normal(0, 0.6, workers)[worker]
    y = (
        0.2 * invariant + 0.05 * exper + 0.5 * job + alpha
        + rng.normal(0, 0.5, n)
    )
    target, achieved = make_target(invariant, worker, rng, ratio)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gelbach.decompose(
            y,
            np.column_stack([target, exper]),
            {"job": job},
            {"worker": worker},
            vce="cluster",
            cluster=worker,
            tol=1e-10,
        )
    return {
        "achieved": achieved,
        "movement": float(result["total"]["coef"][0]),
        "reported_se": float(result["total"]["se"][0]),
        "ratio_field": float(result["x1_fe_collinear_ratio"][0]),
        "mask": bool(result["x1_near_collinear_mask"][0]),
        "warned": any(
            "near-FE-collinear focal" in str(item.message) for item in caught
        ),
    }


def main():
    output = {"reps": REPS, "warning_band": [1e-9, 1e-4], "cells": {}}
    for ratio in RATIOS:
        cells = [one(120000 + rep, ratio) for rep in range(REPS)]
        movements = np.asarray([cell["movement"] for cell in cells])
        reported = np.asarray([cell["reported_se"] for cell in cells])
        expect_warning = 1e-9 < ratio <= 1e-4
        summary = {
            "target_ratio": ratio,
            "max_ratio_field_error": float(
                max(abs(cell["ratio_field"] - cell["achieved"]) for cell in cells)
            ),
            "movement_empirical_sd": float(movements.std(ddof=1)),
            "reported_se_mean": float(reported.mean()),
            "sd_over_reported_se": float(
                movements.std(ddof=1) / reported.mean()
            ),
            "all_masks_match_contract": all(
                cell["mask"] == expect_warning for cell in cells
            ),
            "all_warnings_match_contract": all(
                cell["warned"] == expect_warning for cell in cells
            ),
        }
        output["cells"][f"{ratio:.0e}"] = summary
        print(ratio, summary)
    path = os.path.join(
        os.path.dirname(__file__), "f01_near_fe_band_mc.json"
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
