#!/usr/bin/env python3
"""Low-memory plan, package L3.1-A: the sweep absorber reads the caller's memory instead
of a temporary copy of y and X, and from sixteen columns on the threaded level sums go by
blocks of at most eight columns (Verifications/low_memory_20261116/briefs/
L31A_GS_INPUT_COPY_AND_THREAD_SUMS.md and L31A_R1_BLOCK_THRESHOLD.md, and the audit in
l31a_audit/). Both are exact: every published number must keep its bits.

Builds nothing; never run during a timing campaign. Synthetic data only, made in each
child process from fixed seeds, column by column (no BLAS product, so the inputs do not
depend on the BLAS thread count).

    python3 -I tests/low_memory_sweep_inputs_contract.py --module <module or its folder> \
        [--frozen-module <module of the previous revision>] [--only NAME] [--harness-proof]

Arms, every scenario in a fresh process per arm (the switches are read once per process),
CPU backend, caches and the team registry off:
  new     --module, both switches unset
  copy    --module, XHDFE_ABSORB_INPUT_COPY=1   (the temporary copies of before)
  wide    --module, XHDFE_TLS_COLUMN_BLOCKS=0   (the full-width per-thread sums of before)
  both    --module, the two switches restoring
  frozen  --frozen-module, when given
new is compared with every other arm, bit for bit, on the coefficients, the complete
covariance, the standard errors, the residuals, the saved effects where asked, and the
published scalars (iterations, N, singletons, flags, threads used, degrees of freedom, RSS,
TSS, within R2, the certificate), on the method, and on the error if any.

The scenarios reach what the package changed: sixteen columns or more with a team (the
block routine; exactly sixteen is two full blocks), with and without weights (the weighted
sum is a product and a sum, which the compiler may fuse: the two loops must be compiled
alike), both bookkeepings of the per-thread sums (dense under 64 MiB, sparse above), inputs
at three alignments modulo 64 bytes, saved effects under reghdfe-comparable (early routing
reads the raw inputs), explicit Schwarz (it copies its inputs), heterogeneous slopes, one
and four effects, and controls with 8, 9 and 15 columns, which keep the full-width sums
(unweighted fits with 9 to 16 columns take the packed sweep, which has no such sums).

--harness-proof runs the new arm twice, the second time with the first value of y moved by
one part in 2^30, and passes only if every scenario is then reported different: the
comparison is shown not to be vacuous. (A move of one unit in the last place is not a
valid proof: on 16nov2026 it left every compared output unchanged in 5 of the 16
scenarios.)
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

N = 200_000

SCENARIOS = {
    # name: (mode, columns, threads, options)
    "fast_p8_control": ("fast", 8, 4, {}),
    "fast_p9_aweights_control": ("fast", 9, 4, {"weights": "aweights"}),
    "fast_p15_aweights_control": ("fast", 15, 4, {"weights": "aweights"}),
    "fast_p16_aweights": ("fast", 16, 4, {"weights": "aweights"}),
    "fast_p17": ("fast", 17, 4, {}),
    "fast_p17_fweights": ("fast", 17, 3, {"weights": "fweights"}),
    "fast_p33_sparse_sums": ("fast", 33, 16, {}),
    "fast_p33_aweights_sparse_sums": ("fast", 33, 16, {"weights": "aweights"}),
    "fast_p17_offset24": ("fast", 17, 4, {"xoff": 24, "yoff": 40}),
    "fast_p17_offset56": ("fast", 17, 4, {"xoff": 56, "yoff": 8}),
    "fast_p17_one_effect": ("fast", 17, 4, {"nfe": 1}),
    "fast_p17_four_effects": ("fast", 17, 4, {"nfe": 4}),
    "fast_p17_slopes": ("fast", 17, 4, {"slopes": True}),
    "fast_p17_retain": ("fast", 17, 4, {"retain_fes": True}),
    "comparable_p9_aweights_control": ("comparable", 9, 4, {"weights": "aweights"}),
    "comparable_p17_aweights": ("comparable", 17, 4, {"weights": "aweights"}),
    "comparable_p9_retain_offset16": ("comparable", 9, 4, {"retain_fes": True, "xoff": 16, "yoff": 48}),
    "comparable_p9_schwarz": ("comparable", 9, 4, {"method": "schwarz"}),
    "comparable_p17_symmetric_aweights": ("comparable", 17, 4,
                                          {"method": "symmetric_gauss_seidel", "weights": "aweights"}),
}

ARMS = {
    "new": {},
    "copy": {"XHDFE_ABSORB_INPUT_COPY": "1"},
    "wide": {"XHDFE_TLS_COLUMN_BLOCKS": "0"},
    "both": {"XHDFE_ABSORB_INPUT_COPY": "1", "XHDFE_TLS_COLUMN_BLOCKS": "0"},
}

ARRAYS = ("coef_", "stderr_", "covariance_", "residuals_", "omitted_")
SCALARS = ("num_iterations_", "nobs_", "num_singletons_", "converged_", "precision_certified_",
           "threads_used_", "df_resid_", "rss_", "tss_", "r2_within_", "abs_residual_rel_")


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


def aligned_copy(a, off_bytes):
    """A copy of a (1-D, or 2-D in Fortran order) whose data pointer is off_bytes modulo 64."""
    import numpy as np
    flat = np.empty(a.size + 8, dtype=np.float64)
    start = ((off_bytes - flat.ctypes.data) % 64) // 8
    view = flat[start:start + a.size]
    out = view.reshape(a.shape, order="F") if a.ndim == 2 else view
    out[...] = a
    assert out.ctypes.data % 64 == off_bytes
    return out


def design(p, seed):
    """Ten rows per worker, firms and a fourth effect at random, rows shuffled (no effect is contiguous)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    workers = N // 10
    worker = np.repeat(np.arange(workers, dtype=np.int32), 10)
    year = np.tile(np.arange(10, dtype=np.int32), workers)
    firm = rng.integers(0, N // 200, N, dtype=np.int32)
    other = rng.integers(0, 50, N, dtype=np.int32)
    X = np.empty((N, p), dtype=np.float64, order="F")
    for j in range(p):
        X[:, j] = rng.standard_normal(N)
    a_w, a_f = rng.standard_normal(workers), rng.standard_normal(N // 200)
    y = rng.standard_normal(N) + a_w[worker] + a_f[firm] + year / 10.0
    for j in range(min(p, 3)):
        X[:, j] += 0.3 * a_w[worker] + 0.3 * a_f[firm]
        y += (0.5 - 0.25 * j) * X[:, j]
    aw = 0.5 + rng.random(N)
    fw = rng.integers(1, 4, N).astype(np.float64)
    z = rng.standard_normal(N)
    perm = rng.permutation(N)
    return dict(y=y[perm].copy(), X=np.asfortranarray(X[perm]), worker=worker[perm].copy(),
                firm=firm[perm].copy(), year=year[perm].copy(), other=other[perm].copy(),
                aw=aw[perm].copy(), fw=fw[perm].copy(), z=z[perm].copy())


def child(module_file, names):
    import numpy as np
    core = load_module(module_file)
    out = {}
    for name in names:
        mode, p, threads, opt = SCENARIOS[name]
        d = design(p, 20261116 + 101 * sorted(SCENARIOS).index(name))
        y, X = d["y"], d["X"]
        if os.environ.get("LOW_MEMORY_CONTRACT_PERTURB") == "1":
            y[0] += abs(y[0]) * 2.0 ** -30 + 2.0 ** -40
        if "xoff" in opt:
            X, y = aligned_copy(X, opt["xoff"]), aligned_copy(y, opt["yoff"])
        fes = [d["worker"], d["firm"], d["year"], d["other"]][:opt.get("nfe", 3)]
        kw = dict(fes=fes, clusters=d["worker"])
        if opt.get("weights") == "aweights":
            kw["weights"] = d["aw"]
        elif opt.get("weights") == "fweights":
            kw["weights"] = d["fw"]
            kw["fweights"] = True
        if opt.get("slopes"):
            kw["slopes"] = [dict(fe_index=1, values=d["z"], include_intercept=True)]
        reg = core.HdfeRegressor(
            se_type="cluster", tol=1e-8, max_iter=100000, num_threads=threads, symmetric_sweep=False,
            absorption_method=opt.get("method", "auto"),
            tolerance_mode="xhdfe-fast" if mode == "fast" else "reghdfe-comparable",
            jacobi_relaxation=0.0, retain_fes=bool(opt.get("retain_fes", False)), drop_singletons=True)
        row = {} if os.environ.get("LOW_MEMORY_CONTRACT_PERTURB") == "1" else {"inputs": digest(y) + digest(X)}
        try:
            reg.fit(y, X, **kw)
        except Exception as exc:  # a refusal is a result: the arms must refuse alike
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            out[name] = row
            continue
        for prop in ARRAYS:
            row[prop] = digest(getattr(reg, prop))
        if opt.get("retain_fes"):
            row["fe_effects_"] = [digest(v) for v in reg.fe_effects_]
        for prop in SCALARS:
            value = getattr(reg, prop)
            row[prop] = float(value).hex() if isinstance(value, float) else str(value)
        row["method"] = str(reg.absorption_method_used)
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
            print(f"low_memory_sweep_inputs_contract: harness proof FAILED, no difference seen in {blind}")
            raise SystemExit(1)
        print(f"low_memory_sweep_inputs_contract: harness proof OK, {len(names)} scenarios see y[0] moved by 2^-30")
        return
    results = {arm: run_arm(candidate, names, switches) for arm, switches in ARMS.items()}
    if a.frozen_module:
        results["frozen"] = run_arm(module_path(a.frozen_module), names, {})
    failures = 0
    for name in names:
        new = results["new"][name]
        if "error" in new:
            print(f"FAIL {name}: the new arm refused: {new['error']}")
            failures += 1
            continue
        if SCENARIOS[name][1] > 8 and new["method"].endswith("reduced_pcg"):
            print(f"FAIL {name}: took the reduced tier, so it does not reach the sweep absorber")
            failures += 1
        for arm, rows in results.items():
            if arm == "new":
                continue
            other = rows[name]
            moved = sorted(k for k in set(new) | set(other) if new.get(k) != other.get(k))
            if moved:
                print(f"FAIL {name}: new differs from {arm} in {moved}")
                failures += 1
    arms = ", ".join(results)
    if failures:
        print(f"low_memory_sweep_inputs_contract: {failures} failure(s); arms {arms}")
        raise SystemExit(1)
    print(f"low_memory_sweep_inputs_contract: OK, {len(names)} scenarios, arms {arms}, every property bit for bit")


if __name__ == "__main__":
    main()
