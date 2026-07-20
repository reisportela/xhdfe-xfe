#!/usr/bin/env python3
"""Absorbed-target allocation: a worker-invariant descriptive group gap."""

import numpy as np

from xhdfe.gelbach import decompose

rng = np.random.default_rng(20260719)
n_workers, periods = 200, 5
worker = np.repeat(np.arange(n_workers), periods)
group = rng.integers(0, 2, n_workers)[worker].astype(float)
experience = np.tile(np.arange(periods), n_workers) + rng.normal(scale=0.2,
                                                                 size=worker.size)
job = 0.3 * group + 0.2 * experience + rng.normal(size=worker.size)
y = (0.25 * group + 0.08 * experience + 0.5 * job +
     rng.normal(size=n_workers)[worker] + rng.normal(size=worker.size))

# `group` is invariant within worker and therefore not identified together
# with worker FE. The opt-in mode imposes b_full[group] = 0 and allocates the
# base coefficient under that explicit constraint. It does not estimate a
# within-worker group effect and it is not a causal mediation analysis.
result = decompose(
    y,
    np.column_stack([group, experience]),
    x2_groups={"job_covariate": job},
    fes={"worker_fe": worker},
    vce="cluster",
    cluster=worker,
    absorbed_targets=[0],
)

print(result["estimand"])
print("b_full status:", result["b_full_status"])
print("group base / imposed full:", result["b_base"][0], result["b_full"][0])
print("group allocation:", {name: block["coef"][0]
                            for name, block in result["delta"].items()})
print("total SE type:", result["total_se_type"])
