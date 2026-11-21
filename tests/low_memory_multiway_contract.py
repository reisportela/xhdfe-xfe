#!/usr/bin/env python3
"""Low-memory plan, package L3.1-C: multi-way clustering without its n x (p+1) intermediates
(Verifications/low_memory_20261116/briefs/L31C_MULTIWAY_CLUSTER_INTERMEDIATES.md).

C1: without weights, the OLS stage and the covariance rebuild read y, X and the scores where they
are instead of copies (XHDFE_MULTIWAY_INPUT_COPY=1 restores the copies). C2: the rebuild with the
recovered constant takes the column of ones implicitly instead of forming [X | 1]
(XHDFE_MULTIWAY_SCORES_COPY=1 restores the matrix). Both are exact: every published number must
keep its bits.

Builds nothing; never run during a timing campaign. Synthetic data only, made in each child
process from fixed seeds, column by column (no BLAS product, so the inputs do not depend on the
BLAS thread count).

    python3 -I tests/low_memory_multiway_contract.py --module <module or its folder> \
        [--frozen-module <module of the previous revision>] [--only NAME] [--harness-proof]

Arms, every scenario in a fresh process per arm (the switches are read once per process), CPU
backend, caches and the team registry off:
  new      --module, both switches unset
  inputs   --module, XHDFE_MULTIWAY_INPUT_COPY=1    (C2 alone: the copies of before)
  scores   --module, XHDFE_MULTIWAY_SCORES_COPY=1   (C1 alone: the [X | 1] matrix of before)
  both     --module, the two switches restoring     (today's behaviour)
  frozen   --frozen-module, when given
These arms keep the former publication of saved effects (XHDFE_SAVEFE_ACCURATE_PROJECTION=0, see
ARMS). One more arm, "projection", runs the scenario with saved effects with that correction on, and
the same data without saving the effects: the fit with saved effects must either keep the bits of the
former publication or have the coefficients, covariance, standard errors and residuals of the fit
without them, bit for bit.
new is compared with every other arm, bit for bit, on every published property of the fitted
regressor (coefficients, the complete covariance with the intercept row and column, standard
errors, residuals, cluster counts per dimension and per intersection, degrees of freedom, N,
singletons, iterations, method, flags, certificate, threads), and on the error if any.

The scenarios reach what the package changed: two- and three-way clustering with the recovered
constant (the rebuild), with no weights (the implicit column and the views) and with analytic,
frequency and probability weights (the scaled copies that stay), widths 1, 7 and 8 around the
run-fast boundary of the meats (p <= 8 counts the column of ones), intersections of about n
cells and of few cells, nested dimensions, sparse raw identifiers (hash-map meats), a collinear
regressor with two dimensions (the omitted-column rebuild), IV, no absorbed effects (OLS stage
with an explicit constant and an origin shift), saved effects, group()/individual() (the second
rebuild site), Fast and Comparable, explicit and automatic teams, residuals scaled by 2^-460
(range checks of the meats), and one-way and robust controls.

--harness-proof runs the new arm twice, the second time with the first value of y moved by one
part in 2^30, and passes only if every scenario is then reported different.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

WORKERS = 5000

SCENARIOS = {
    # name: (mode, columns, threads, clusters, options)
    "comparable_p4_two_way": ("comparable", 4, 4, ("worker", "firm"), {}),
    "fast_p4_two_way": ("fast", 4, 4, ("worker", "firm"), {}),
    "comparable_p9_two_way_aweights": ("comparable", 9, 4, ("worker", "firm"), {"weights": "aw"}),
    "fast_p9_two_way_fweights": ("fast", 9, 4, ("worker", "firm"), {"weights": "fw"}),
    "comparable_p4_two_way_pweights": ("comparable", 4, 4, ("worker", "firm"), {"weights": "pw"}),
    "comparable_p25_n_cells": ("comparable", 25, 4, ("worker", "year"), {}),
    "fast_p1_n_cells": ("fast", 1, 1, ("worker", "year"), {}),
    "comparable_p8_nested": ("comparable", 8, 4, ("firm", "region"), {}),
    "comparable_p7_few_cells_aweights": ("comparable", 7, 4, ("sector", "year"), {"weights": "aw"}),
    "comparable_p4_sparse_ids": ("comparable", 4, 4, ("worker_sparse", "firm_sparse"), {}),
    "fast_p4_three_way": ("fast", 4, 4, ("worker", "firm", "year"), {}),
    "comparable_p4_three_way_aweights": ("comparable", 4, 0, ("worker", "firm", "year"), {"weights": "aw"}),
    "comparable_omitted_two_way": ("comparable", 2, 4, ("worker", "firm_sparse"), {"omitted": True}),
    "fast_omitted_two_way_aweights": ("fast", 4, 4, ("firm", "region"), {"omitted": True, "weights": "aw"}),
    "comparable_iv_two_way": ("comparable", 4, 4, ("worker", "firm"), {"iv": True}),
    "comparable_noabsorb_two_way": ("comparable", 4, 4, ("worker", "firm"), {"nofe": True}),
    "comparable_retain_two_way": ("comparable", 4, 4, ("worker", "firm"), {"retain_fes": True}),
    "comparable_group_individual_two_way": ("comparable", 9, 4, ("c1", "c2"), {"gi": True}),
    "comparable_range_two_way": ("comparable", 4, 4, ("worker", "firm"), {"yscale": -460}),
    "comparable_p4_one_way_control": ("comparable", 4, 4, ("worker",), {}),
    "comparable_p4_robust_control": ("comparable", 4, 4, (), {"vce": "robust"}),
}

# Correction E1 (16nov2026, tests/savefe_comparable_projection_contract.py): a reghdfe-comparable fit with
# saved effects whose capturing sweep stops above the floor of the mode now publishes the projection of the
# mode, which changes its numbers on purpose. The arms below check the multi-way covariance against the
# previous revision, so they keep the former publication of saved effects (the frozen module does not know
# the switch); the correction itself is checked through this covariance by the arm "projection" in main().
FORMER_SAVEFE = {"XHDFE_SAVEFE_ACCURATE_PROJECTION": "0"}
ARMS = {
    "new": dict(FORMER_SAVEFE),
    "inputs": dict(FORMER_SAVEFE, XHDFE_MULTIWAY_INPUT_COPY="1"),
    "scores": dict(FORMER_SAVEFE, XHDFE_MULTIWAY_SCORES_COPY="1"),
    "both": dict(FORMER_SAVEFE, XHDFE_MULTIWAY_INPUT_COPY="1", XHDFE_MULTIWAY_SCORES_COPY="1"),
}
# The scenario with saved effects; LOW_MEMORY_CONTRACT_PLAIN_TWIN=1 makes the child fit the same data
# without saving them (each scenario has its own seed, so the twin must be the same scenario).
SAVED_EFFECTS_SCENARIO = "comparable_retain_two_way"
SKIP = re.compile(r"(seconds|elapsed|_time|time_)")


def module_path(path):
    """The module file itself. Never imported by the parent: two builds cannot share a process."""
    path = Path(path).resolve()
    if path.is_dir():
        found = sorted(path.glob("py_hdfe_v11*.so"))
        if len(found) != 1:
            raise SystemExit(f"{path}: expected exactly one py_hdfe_v11 module, found {len(found)}")
        path = found[0]
    if not path.is_file():
        raise SystemExit(f"{path}: no such module")
    return path


def load_module(path):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    import numpy as np
    a = np.ascontiguousarray(np.asarray(value))
    return f"{a.dtype.str}{a.shape}{hashlib.sha256(a.tobytes()).hexdigest()}"


def describe(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return digest(value)
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    if isinstance(value, float):
        return value.hex()
    return str(value)


def panel(seed, pmax):
    """Workers with 3 to 8 rows (grouped), a home firm, years, regions and sectors; sparse raw ids."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = rng.integers(3, 9, size=WORKERS)
    worker = np.repeat(np.arange(WORKERS), rows)
    n = int(worker.size)
    t = np.concatenate([np.arange(r) for r in rows])
    year = (rng.integers(0, 8, size=WORKERS)[worker] + t) % 8
    firms = 400
    home = rng.integers(0, firms, size=WORKERS)
    firm = np.where(rng.random(n) < 0.15, rng.integers(0, firms, size=n), home[worker])
    X = np.empty((n, pmax), dtype=np.float64, order="F")
    for j in range(pmax):
        X[:, j] = rng.standard_normal(n)
    y = 0.3 * rng.standard_normal(n)
    for j in range(pmax):
        y = y + (0.5 - 0.02 * j) * X[:, j]
    for f in (worker, firm, year):
        y = y + rng.standard_normal(int(f.max()) + 1)[f]
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    xend = 0.8 * z1 + 0.5 * z2 + rng.standard_normal(n)

    def sparse(codes):
        levels = int(codes.max()) + 1
        values = np.sort(rng.choice(int(8.7 * n), size=levels, replace=False))
        return values[rng.permutation(levels)][codes].astype(np.int32)

    return dict(n=n, y=y, X=X, worker=worker.astype(np.int32), firm=firm.astype(np.int32),
                year=year.astype(np.int32), region=(firm // 30).astype(np.int32),
                sector=((firm * 7) % 11).astype(np.int32), worker_sparse=sparse(worker),
                firm_sparse=sparse(firm), aw=rng.uniform(0.5, 2.0, n),
                fw=rng.integers(1, 4, n).astype(np.float64), z1=z1, z2=z2, xend=xend)


def group_panel(seed, p):
    """group()/individual(): group-level y and X on each member row, clusters constant in group."""
    import numpy as np
    rng = np.random.default_rng(seed)
    groups, individuals, members = 1500, 250, 3
    mem = np.array([rng.choice(individuals, members, replace=False) for _ in range(groups)])
    group = np.repeat(np.arange(groups), members)
    X = np.empty((groups * members, p), dtype=np.float64, order="F")
    for j in range(p):
        X[:, j] = rng.standard_normal(groups)[group]
    y = (0.8 * X[:, 0] - 0.3 * X[:, 1] + rng.standard_normal(groups)[group]).copy()
    return dict(y=y, X=X, ind=mem.ravel().astype(np.int32), gfe=(group % 40).astype(np.int32),
                group=group.astype(np.int64), c1=(group // 10).astype(np.int32),
                c2=(group % 13).astype(np.int32))


def child(module_file, names):
    import numpy as np
    core = load_module(module_file)
    out = {}
    for name in names:
        mode, p, threads, clu, opt = SCENARIOS[name]
        seed = 20261116 + 101 * sorted(SCENARIOS).index(name)
        reg_kw = dict(se_type=opt.get("vce", "cluster"), tol=1e-8, max_iter=100000, num_threads=threads,
                      absorption_method="auto",
                      retain_fes=bool(opt.get("retain_fes", False)) and
                      os.environ.get("LOW_MEMORY_CONTRACT_PLAIN_TWIN") != "1",
                      tolerance_mode="xhdfe-fast" if mode == "fast" else "reghdfe-comparable",
                      drop_singletons=not opt.get("gi", False))
        if opt.get("gi"):
            d = group_panel(seed, p)
            y, X = d["y"], d["X"]
            kw = dict(fes=[d["ind"], d["gfe"]], group=d["group"], individual=d["ind"], aggregation="mean",
                      clusters=[d[c] for c in clu])
        else:
            d = panel(seed, 25)
            y = d["y"]
            if opt.get("omitted"):
                X = np.empty((d["n"], p + 1), dtype=np.float64, order="F")
                for j in range(p):
                    X[:, j] = d["X"][:, j]
                X[:, p] = d["X"][:, 0] + d["X"][:, 1]
            elif opt.get("iv"):
                X = np.asfortranarray(d["X"][:, :p].copy())
                X[:, 0] = d["xend"]
                y = y + 0.7 * d["xend"]
            else:
                X = np.asfortranarray(d["X"][:, :p].copy())
            if opt.get("nofe"):
                X[:, 0] = X[:, 0] + 1000.0  # an origin shift for the OLS stage's explicit constant
            if "yscale" in opt:
                y = y * 2.0 ** opt["yscale"]
            kw = dict(fes=None if opt.get("nofe") else [d["worker"], d["firm"], d["year"]])
            if clu:
                kw["clusters"] = d[clu[0]] if len(clu) == 1 else [d[c] for c in clu]
            if opt.get("weights") == "aw":
                kw["weights"] = d["aw"]
            elif opt.get("weights") == "fw":
                kw.update(weights=d["fw"], fweights=True)
            elif opt.get("weights") == "pw":
                kw.update(weights=d["aw"], pweights=True)
            if opt.get("iv"):
                Z = np.empty((d["n"], 2), dtype=np.float64, order="F")
                Z[:, 0], Z[:, 1] = d["z1"], d["z2"]
                kw.update(instruments=Z, endogenous_idx=[0])
        if os.environ.get("LOW_MEMORY_CONTRACT_PERTURB") == "1":
            y = y.copy()
            y[0] += abs(y[0]) * 2.0 ** -30 + 2.0 ** -40 * (2.0 ** opt.get("yscale", 0))
        row = {} if os.environ.get("LOW_MEMORY_CONTRACT_PERTURB") == "1" else {"inputs": digest(y) + digest(X)}
        try:
            reg = core.HdfeRegressor(**reg_kw)
            reg.fit(y, X, **kw)
        except Exception as exc:  # a refusal is a result: the arms must refuse alike
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            out[name] = row
            continue
        for prop in sorted(n for n in dir(type(reg)) if isinstance(getattr(type(reg), n, None), property)):
            if SKIP.search(prop):
                continue
            try:
                row[prop] = describe(getattr(reg, prop))
            except Exception as exc:  # not available for this fit
                row[prop] = f"!{type(exc).__name__}"
        out[name] = row
    print("RESULT " + json.dumps(out))


def run_arm(module_path, names, switches):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("XHDFE_") and k not in ("LD_LIBRARY_PATH",)}
    env.update(XHDFE_GPU_BACKEND="cpu", XHDFE_MOBILITY_MODE="off", XHDFE_ABSORPTION_CACHE_MODE="off",
               XHDFE_FE_STRUCTURE_MODE="off", XHDFE_AUTO_TEAM_COORDINATION="0")
    env.update(switches)
    done = subprocess.run([sys.executable, "-I", __file__, "--child", str(module_path), "--only", ",".join(names)],
                          env=env, capture_output=True, text=True)
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    if done.returncode != 0 or len(lines) != 1:
        raise SystemExit(f"arm failed (rc={done.returncode}):\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    return json.loads(lines[0][7:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module")
    ap.add_argument("--frozen-module")
    ap.add_argument("--only", default="")
    ap.add_argument("--child")
    ap.add_argument("--harness-proof", action="store_true")
    a = ap.parse_args()
    names = [n for n in a.only.split(",") if n] or sorted(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenarios: {unknown}")
    if a.child:
        child(a.child, names)
        return
    if not a.module:
        raise SystemExit("--module is required")
    candidate = module_path(a.module)
    if a.harness_proof:
        plain = run_arm(candidate, names, {})
        moved = run_arm(candidate, names, {"LOW_MEMORY_CONTRACT_PERTURB": "1"})
        blind = [n for n in names
                 if all(plain[n].get(k) == moved[n].get(k) for k in set(plain[n]) - {"inputs"})]
        if blind:
            print(f"low_memory_multiway_contract: harness proof FAILED, no difference seen in {blind}")
            raise SystemExit(1)
        print(f"low_memory_multiway_contract: harness proof OK, {len(names)} scenarios see y[0] moved by 2^-30")
        return
    results = {arm: run_arm(candidate, names, switches) for arm, switches in ARMS.items()}
    if a.frozen_module:
        # The frozen module runs with the same switch: one from before correction E1 ignores it, a later one
        # then keeps the former publication of saved effects, as the arms above do.
        results["frozen"] = run_arm(module_path(a.frozen_module), names, dict(FORMER_SAVEFE))
    failures = 0
    for name in names:
        new = results["new"][name]
        if "error" in new and "range" not in name:
            print(f"FAIL {name}: the new arm refused: {new['error']}")
            failures += 1
            continue
        for arm, rows in results.items():
            if arm == "new":
                continue
            other = rows[name]
            moved = sorted(k for k in set(new) | set(other) if new.get(k) != other.get(k))
            if moved:
                print(f"FAIL {name}: new differs from {arm} in {moved[:12]}")
                failures += 1
    if SAVED_EFFECTS_SCENARIO in names:
        fields = ("coef_", "covariance_", "stderr_", "residuals_")
        saved = run_arm(candidate, [SAVED_EFFECTS_SCENARIO], {})[SAVED_EFFECTS_SCENARIO]
        plain = run_arm(candidate, [SAVED_EFFECTS_SCENARIO],
                        {"LOW_MEMORY_CONTRACT_PLAIN_TWIN": "1"})[SAVED_EFFECTS_SCENARIO]
        results["projection"] = {SAVED_EFFECTS_SCENARIO: saved}
        if saved.get("inputs") != plain.get("inputs") or "fe_effects_" not in saved or \
                saved.get("fe_effects_") == plain.get("fe_effects_"):
            print(f"FAIL {SAVED_EFFECTS_SCENARIO}: harness error, the twin is not the same fit without saved effects")
            failures += 1
        elif saved != results["new"][SAVED_EFFECTS_SCENARIO]:  # the projection was published
            moved = [k for k in fields if k not in saved or saved.get(k) != plain.get(k)]
            if "error" in saved or "error" in plain or moved:
                print(f"FAIL {SAVED_EFFECTS_SCENARIO}: with the projection of saved effects on, the fit differs "
                      f"from the same fit without saved effects in {moved or 'an error'}")
                failures += 1
            else:
                print(f"{SAVED_EFFECTS_SCENARIO}: projection published, equal to the fit without saved effects")
        else:
            print(f"{SAVED_EFFECTS_SCENARIO}: sweep published as before (at or below the floor of the mode)")
    arms = ", ".join(results)
    if failures:
        print(f"low_memory_multiway_contract: {failures} failure(s); arms {arms}")
        raise SystemExit(1)
    print(f"low_memory_multiway_contract: OK, {len(names)} scenarios, arms {arms}, every property bit for bit")


if __name__ == "__main__":
    main()
