#!/usr/bin/env python3
"""Module-level contract of the reduced-system tier, 2.32 candidate (on by default;
XHDFE_REDUCED_PCG=0 turns it off).

Builds a Python module first; never run during a timing campaign. Synthetic data
generated in each child process; nothing is read from disk except the modules.
Every fit runs in a fresh process.

    python3 -I tests/reduced_pcg_contract.py \
        --module build_c232_cpu/py_hdfe_v11....so \
        --base-module <daba197c build>/py_hdfe_v11....so \
        [--cuda-module build_c232_cuda/py_hdfe_v11....so] \
        [--r1-module build_r1_cpu/py_hdfe_v11....so]

Checks (each failure prints one FAIL line; the last line is PASS or the count):
  1. switch off == base module, bit for bit, on every public output the base
     module has; off reports status 0, reason "off", 0 discarded iterations;
  2. default (on): accepted, reported as reduced-pcg (code 8) with the tier's
     iterations, status 1, reason "none", 0 discarded; b within
     1e-9 max(1,|b|) and V within 1e-8 sqrt(V_jj V_kk) of off; one final line;
  3. forced declines (geometry, iterate, residuals, certificate) and an injected
     std::bad_alloc in the adapter's setup: == off bit for bit (the current
     route's result), status 2, reason "forced"/"memory", discarded iterations
     equal to the declined attempt's (its final line);
  4. ineligible fits (weights, savefe, xhdfe-fast, IV, explicit method): == off
     bit for bit, status 0 with the exact reason, 0 discarded (savefe: its
     effect-capturing sweep, run with XHDFE_SAVEFE_ACCURATE_PROJECTION=0; by
     default, correction E1: either that sweep as before, or the projection
     that follows it, equal to the plain fit bit for bit with its diagnostics);
  5. reduced-pcg requested (both spellings): rejected with an informative error;
  6. GPU requested (with --cuda-module): status 0, reason "gpu", GPU used;
  7. threads 1, 2, 8, 16: on gives the same bits wherever off does;
  8. absorption cache, fresh processes: an accepted fit written then read back
     gives every public output and diagnostic bit-identical, reduced-pcg in
     both, and the read marks the hit (final line status=cache); with
     --r1-module, an entry written by the R1 tier is never read;
  9. C++-internal regressions (AKM controls, Gelbach): with the tier on and the
     trace on, at least one final line, none accepted or declined, AKM's FWL
     fit ineligible by "method" and a Gelbach fit by "parity_floor", and the
     outputs bit-identical to the switch off.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

PUBLIC_FIELDS = (
    "absorption_method_used", "num_iterations_", "abs_residual_", "abs_residual_rel_",
    "converged_", "precision_certified_", "threads_used_",
    "auto_routing_retry_policy_enabled_", "auto_routing_retry_eligible_",
    "auto_routing_retry_fired_", "auto_routing_retry_status_",
    "auto_routing_retry_primary_method_", "auto_routing_retry_primary_iterations_",
    "auto_routing_retry_iterations_",
)
DIAGNOSTICS = ("reduced_pcg_status_", "reduced_pcg_reason_", "reduced_pcg_discarded_iterations_",
               "gpu_used_")
ARRAYS = ("coef_", "covariance_", "residuals_", "sample_index_")
REDUCED_PCG = 8
SEED = 20261113


def load(path: str):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def design(kind: str, seed: int):
    """Synthetic worker-firm-year panels (public by construction)."""
    rng = np.random.default_rng(seed)
    if kind == "akm3":
        workers, firms, years = 40_000, 3_000, 12
        rows = rng.integers(3, 9, size=workers)
        worker = np.repeat(np.arange(workers), rows)
        n = worker.size
        year = (rng.integers(0, years - 8, size=workers)[worker] +
                np.concatenate([np.arange(r) for r in rows])) % years
        home = rng.integers(0, firms, size=workers)
        move = rng.random(n) < 0.15
        firm = np.where(move, rng.integers(0, firms, size=n), home[worker])
        fes = [worker, firm, year]
    elif kind == "four":
        n = 300_000
        unit = np.repeat(np.arange(n // 60), 60)
        fes = [unit, rng.integers(0, 40, n), rng.integers(0, 15, n), rng.integers(0, 6, n)]
    else:
        raise ValueError(kind)
    n = fes[0].size
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n) + 0.1 * fes[1] % 7,
                         1.0e3 + rng.normal(size=n)])
    y = X @ np.array([0.5, -0.25, 0.01]) + 0.3 * rng.normal(size=n)
    for f in fes:
        y = y + rng.normal(size=int(f.max()) + 1)[f]
    return y, np.asfortranarray(X), [np.ascontiguousarray(f, dtype=np.int32) for f in fes]


def value(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, float):
        return float(v)
    return int(v)


def child_fit(args) -> int:
    module = load(args.module)
    y, X, fes = design(args.design, args.seed)
    weights = np.ones(y.size) if args.case == "weights" else None
    instruments = None
    endogenous: list[int] = []
    if args.case == "iv":
        instruments = np.asfortranarray((X[:, 1] + 0.5 * X[:, 0]).reshape(-1, 1))
        endogenous = [1]
    reg = module.HdfeRegressor(
        se_type="cluster", tol=1e-8, max_iter=100000, fit_intercept=True,
        num_threads=args.threads, drop_singletons=True,
        retain_fes=args.case == "savefe", symmetric_sweep=False,
        absorption_method="mlsmr" if args.case == "method" else "auto",
        tolerance_mode="xhdfe-fast" if args.case == "fast" else "reghdfe-comparable")
    reg.fit(y, X, fes=fes, clusters=fes[0], weights=weights,
            instruments=instruments, endogenous_idx=endogenous, slopes=[])
    public = {name: value(getattr(reg, name)) for name in PUBLIC_FIELDS if hasattr(reg, name)}
    diagnostics = {name: value(getattr(reg, name)) for name in DIAGNOSTICS if hasattr(reg, name)}
    Path(args.output).write_text(json.dumps({"public": public, "diagnostics": diagnostics}, sort_keys=True))
    np.savez(args.arrays, **{name: np.asarray(getattr(reg, name)) for name in ARRAYS})
    return 0


def child_request(args) -> int:
    module = load(args.module)
    outcome = {}
    for name in ("reduced-pcg", "reduced_pcg", "REDUCED-PCG"):
        try:
            module.HdfeRegressor(absorption_method=name)
            outcome[name] = "accepted"
        except Exception as exc:  # the binding raises RuntimeError
            outcome[name] = f"{type(exc).__name__}: {exc}"
    outcome["enum"] = int(module.AbsorptionMethod.reduced_pcg)
    Path(args.output).write_text(json.dumps(outcome, sort_keys=True))
    np.savez(args.arrays, empty=np.zeros(0))
    return 0


def child_internal(args) -> int:
    """AKM with controls (FWL fit) and a Gelbach decomposition with two FEs."""
    module = load(args.module)
    rng = np.random.default_rng(args.seed)
    workers = np.repeat(np.arange(400, dtype=np.int64), 5)
    period = np.tile(np.arange(5, dtype=np.int64), 400)
    firms = (workers + 3 * period) % 37
    controls = rng.normal(size=(workers.size, 2))
    akm_y = controls @ np.array([0.4, -0.2]) + 0.05 * firms + rng.normal(size=workers.size)
    akm = module.akm_kss(akm_y, workers, firms, X=controls, leave_out_level="match",
                         leverages="exact", prune=False, num_threads=1)
    ng = 3000
    gx1 = rng.normal(size=(ng, 2))
    gx2 = rng.normal(size=(ng, 3))
    gf1 = (np.arange(ng) % 41).astype(np.int64)
    gf2 = (np.arange(ng) % 17).astype(np.int64)
    gy = gx1 @ np.array([0.5, -0.2]) + gx2 @ np.array([0.3, 0.1, -0.4])
    gy += 0.01 * gf1 - 0.015 * gf2 + rng.normal(scale=0.35, size=ng)
    gel = module.gelbach_decompose(gy, gx1, gx2, [2, 1], [gf1, gf2], tol=1e-9, num_threads=1)
    arrays = {f"akm_{k}": np.asarray(akm[k]) for k in ("alpha", "psi")}
    arrays.update({f"gelbach_{k}": np.asarray(gel[k])
                   for k in ("b_base", "b_full", "delta", "cov", "total", "total_cov")})
    Path(args.output).write_text(json.dumps(
        {"akm_converged": bool(akm["converged"]), "gelbach_converged": bool(gel["converged"])}))
    np.savez(args.arrays, **arrays)
    return 0


def run(script, module, root, label, *, design_name="akm3", case="plain", threads=8, kind="fit",
        switch=None, trace=False, stage=None, extra=None):
    """switch: None (default: the tier on), "0" (off) or "1" (R1's explicit on)."""
    output, arrays = root / f"{label}.json", root / f"{label}.npz"
    env = {k: v for k, v in os.environ.items() if not k.startswith("XHDFE_")}
    env.update({"XHDFE_GPU_BACKEND": "cpu", "XHDFE_ABSORPTION_CACHE_MODE": "off",
                "XHDFE_FE_STRUCTURE_MODE": "off", "XHDFE_MOBILITY_MODE": "off",
                "OMP_WAIT_POLICY": "passive"})
    # D10's switch passes through from the caller, so the contract runs with D10
    # on (default) and off (XHDFE_REDUCED_PCG_CHAT_STOP=0).
    if "XHDFE_REDUCED_PCG_CHAT_STOP" in os.environ:
        env["XHDFE_REDUCED_PCG_CHAT_STOP"] = os.environ["XHDFE_REDUCED_PCG_CHAT_STOP"]
    if switch is not None:
        env["XHDFE_REDUCED_PCG"] = switch
    if trace:
        env["XHDFE_REDUCED_PCG_TRACE"] = "1"
    if stage:
        env["XHDFE_TEST_REDUCED_PCG_DECLINE"] = stage
    if extra:
        env.update(extra)
    command = [sys.executable, "-B", "-I", str(script), "--child", kind, "--module", str(module),
               "--design", design_name, "--case", case, "--threads", str(threads),
               "--seed", str(SEED), "--output", str(output), "--arrays", str(arrays)]
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{label}: child failed ({done.returncode}): {done.stderr[-2000:]}")
    finals = [line for line in done.stderr.splitlines() if "reduced_pcg event=final" in line]
    record = json.loads(output.read_text())
    with np.load(arrays) as data:
        record_arrays = {k: data[k].copy() for k in data.files}
    return record, record_arrays, finals


def field(line: str, name: str):
    match = re.search(rf"\b{name}=(\S+)", line)
    return match.group(1) if match else None


def same_public(a, b) -> bool:
    """Every public output both records have, and every array, bit for bit."""
    (ra, xa, _), (rb, xb, _) = a, b
    pa, pb = ra["public"], rb["public"]
    keys = set(pa) & set(pb)
    return all(pa[k] == pb[k] for k in keys) and all(np.array_equal(xa[k], xb[k]) for k in xa)


def within_contract(on, off) -> bool:
    b_on, b_off = on[1]["coef_"], off[1]["coef_"]
    V_on, V_off = on[1]["covariance_"], off[1]["covariance_"]
    scale = np.sqrt(np.outer(np.diag(V_off), np.diag(V_off)))
    return bool(np.all(np.abs(b_on - b_off) <= 1e-9 * np.maximum(1.0, np.abs(b_off))) and
                np.all(np.abs(V_on - V_off) <= 1e-8 * scale))


def diag(record):
    return record[0]["diagnostics"]


def parent(args) -> int:
    script = Path(__file__).resolve()
    failures = []

    def fail(message):
        failures.append(message)
        print("FAIL", message, flush=True)

    with tempfile.TemporaryDirectory(prefix="c232_contract_") as tmp:
        root = Path(tmp)
        for design_name in ("akm3", "four"):
            base = run(script, args.base_module, root, f"{design_name}_base", design_name=design_name)
            off = run(script, args.module, root, f"{design_name}_off", design_name=design_name,
                      switch="0", trace=True)
            if not same_public(base, off):
                fail(f"{design_name}: switch off differs from the base module")
            d = diag(off)
            if (d.get("reduced_pcg_status_"), d.get("reduced_pcg_reason_"),
                    d.get("reduced_pcg_discarded_iterations_")) != (0, "off", 0) or off[2]:
                fail(f"{design_name}: switch off diagnostics {d} lines {off[2]}")
            if off[0]["public"]["absorption_method_used"] == REDUCED_PCG:
                fail(f"{design_name}: switch off reports reduced-pcg")

            on = run(script, args.module, root, f"{design_name}_on", design_name=design_name, trace=True)
            line = on[2][0] if len(on[2]) == 1 else ""
            print(f"{design_name} on: {line or on[2]}", flush=True)
            if "status=accepted" not in line:
                # A tier that always declined would otherwise pass by identity.
                fail(f"{design_name}: default not accepted: {on[2]}")
            else:
                if on[0]["public"]["absorption_method_used"] != REDUCED_PCG:
                    fail(f"{design_name}: accepted fit reports method {on[0]['public']['absorption_method_used']}")
                if on[0]["public"]["num_iterations_"] != int(field(line, "iterations")):
                    fail(f"{design_name}: accepted fit reports {on[0]['public']['num_iterations_']} "
                         f"iterations, the tier {field(line, 'iterations')}")
                d = diag(on)
                if (d.get("reduced_pcg_status_"), d.get("reduced_pcg_reason_"),
                        d.get("reduced_pcg_discarded_iterations_")) != (1, "none", 0):
                    fail(f"{design_name}: accepted diagnostics {d}")
            if not within_contract(on, off):
                fail(f"{design_name}: default outside the b/V contract of off")

            for stage in ("geometry", "iterate", "residuals", "certificate", "bad_alloc"):
                forced = run(script, args.module, root, f"{design_name}_{stage}",
                             design_name=design_name, trace=True, stage=stage)
                line = forced[2][0] if len(forced[2]) == 1 else ""
                reason = "memory" if stage == "bad_alloc" else "forced"
                d = diag(forced)
                print(f"{design_name} {stage}: {line or forced[2]}", flush=True)
                if not same_public(forced, off):
                    fail(f"{design_name}: forced decline at {stage} differs from off")
                if f"status=declined reason={reason}" not in line:
                    fail(f"{design_name}: forced decline at {stage}: final lines {forced[2]}")
                elif (d.get("reduced_pcg_status_"), d.get("reduced_pcg_reason_"),
                      d.get("reduced_pcg_discarded_iterations_")) != (2, reason, int(field(line, "iterations"))):
                    fail(f"{design_name}: forced decline at {stage}: diagnostics {d}")
                if stage == "residuals" and d.get("reduced_pcg_discarded_iterations_", 0) <= 0:
                    fail(f"{design_name}: decline after a solved attempt reports no discarded iterations")

            for case, reason in (("weights", "weights"), ("savefe", "savefe"), ("fast", "tolerance_mode"),
                                 ("iv", "iv"), ("method", "method")):
                # E1 (Verifications/low_memory_20261116/briefs/E1_SAVEFE_COMPARABLE_PROJECTION_FIRST.md):
                # the effect-capturing sweep never enters the tier; with XHDFE_SAVEFE_ACCURATE_PROJECTION=0
                # it is the whole fit, as before E1 (the projection that may follow it is checked below).
                e1_former = {"XHDFE_SAVEFE_ACCURATE_PROJECTION": "0"} if case == "savefe" else None
                c_off = run(script, args.module, root, f"{design_name}_{case}_off", design_name=design_name,
                            case=case, switch="0", extra=e1_former)
                c_on = run(script, args.module, root, f"{design_name}_{case}_on", design_name=design_name,
                           case=case, trace=True, extra=e1_former)
                d = diag(c_on)
                if not same_public(c_on, c_off):
                    fail(f"{design_name}: ineligible case {case} differs from off")
                if len(c_on[2]) != 1 or f"status=ineligible reason={reason}" not in c_on[2][0]:
                    fail(f"{design_name}: ineligible case {case}: final lines {c_on[2]}")
                if (d.get("reduced_pcg_status_"), d.get("reduced_pcg_reason_"),
                        d.get("reduced_pcg_discarded_iterations_")) != (0, reason, 0):
                    fail(f"{design_name}: ineligible case {case}: diagnostics {d}")

            # E1, default: a certified capturing sweep that stopped above the floor of the mode is followed by
            # the projection, eligible for the tier like the plain fit; the fit with saved effects is then the
            # plain fit (coefficients, covariance, residuals, sample bit for bit; the tier's status and reason).
            # A sweep at or below the floor is published as before (equal to the switch at 0).
            e1 = run(script, args.module, root, f"{design_name}_savefe_e1", design_name=design_name, case="savefe")
            e1_old = run(script, args.module, root, f"{design_name}_savefe_e1_former", design_name=design_name,
                         case="savefe", extra={"XHDFE_SAVEFE_ACCURATE_PROJECTION": "0"})
            projected = not same_public(e1, e1_old)
            print(f"{design_name} savefe (E1): {'projection published' if projected else 'sweep as before'}",
                  flush=True)
            if projected:
                if not all(np.array_equal(e1[1][k], on[1][k]) for k in ARRAYS):
                    fail(f"{design_name}: E1 projection published, but the fit differs from the plain fit")
                if (diag(e1).get("reduced_pcg_status_"), diag(e1).get("reduced_pcg_reason_")) != \
                        (diag(on).get("reduced_pcg_status_"), diag(on).get("reduced_pcg_reason_")):
                    fail(f"{design_name}: E1 projection diagnostics {diag(e1)}, plain fit {diag(on)}")

            offs = {t: run(script, args.module, root, f"{design_name}_off_t{t}", design_name=design_name,
                           threads=t, switch="0") for t in (1, 2, 8, 16)}
            ons = {t: run(script, args.module, root, f"{design_name}_on_t{t}", design_name=design_name,
                          threads=t) for t in (1, 2, 8, 16)}
            for t in (2, 8, 16):
                off_same = all(np.array_equal(offs[1][1][k], offs[t][1][k]) for k in ("coef_", "covariance_"))
                on_same = all(np.array_equal(ons[1][1][k], ons[t][1][k]) for k in ("coef_", "covariance_"))
                if off_same and not on_same:
                    fail(f"{design_name}: threads {t} changes the bits with the tier on only")

        # 5. The enumerator is output only.
        request = run(script, args.module, root, "request", kind="request")[0]
        print("request:", json.dumps(request, sort_keys=True), flush=True)
        if request.get("enum") != REDUCED_PCG:
            fail(f"request: enum value {request.get('enum')}")
        for name in ("reduced-pcg", "reduced_pcg", "REDUCED-PCG"):
            if "cannot be requested" not in request.get(name, ""):
                fail(f"request: {name} -> {request.get(name)}")

        # 8. Cache: write then read in fresh processes.
        cache = root / "c232.cache"
        cache_env = {"XHDFE_ABSORPTION_CACHE": str(cache)}
        written = run(script, args.module, root, "cache_write", trace=True,
                      extra={**cache_env, "XHDFE_ABSORPTION_CACHE_MODE": "write"})
        read = run(script, args.module, root, "cache_read", trace=True,
                   extra={**cache_env, "XHDFE_ABSORPTION_CACHE_MODE": "read"})
        print(f"cache: write {written[2]} read {read[2]}", flush=True)
        if not cache.exists():
            fail("cache: nothing written")
        if written[0] != read[0] or not all(np.array_equal(written[1][k], read[1][k]) for k in written[1]):
            fail(f"cache: the hit differs from the fresh fit: {written[0]} vs {read[0]}")
        if written[0]["public"]["absorption_method_used"] != REDUCED_PCG:
            fail("cache: the written fit is not reduced-pcg")
        if len(read[2]) != 1 or "status=cache method=reduced-pcg" not in read[2][0]:
            fail(f"cache: the read leg did not hit: {read[2]}")
        if args.r1_module:
            r1_cache = root / "r1.cache"
            r1_env = {"XHDFE_ABSORPTION_CACHE": str(r1_cache)}
            run(script, args.r1_module, root, "r1_cache_write", switch="1",
                extra={**r1_env, "XHDFE_ABSORPTION_CACHE_MODE": "write"})
            r1_read = run(script, args.module, root, "r1_cache_read", trace=True,
                          extra={**r1_env, "XHDFE_ABSORPTION_CACHE_MODE": "read"})
            print(f"cache: R1 entry read by the candidate: {r1_read[2]}", flush=True)
            if not r1_cache.exists():
                fail("cache: the R1 module wrote nothing")
            if len(r1_read[2]) != 1 or "status=cache" in r1_read[2][0] or "status=accepted" not in r1_read[2][0]:
                fail(f"cache: an R1 entry was read: {r1_read[2]}")
        else:
            print("NOT_RUN cache: R1 entry rejection (no --r1-module)", flush=True)

        # 6. GPU requested.
        if args.cuda_module:
            gpu = run(script, args.cuda_module, root, "gpu_on", trace=True, extra={"XHDFE_GPU_BACKEND": "cuda"})
            d = diag(gpu)
            print(f"gpu: {gpu[2]} {d}", flush=True)
            if (d.get("reduced_pcg_status_"), d.get("reduced_pcg_reason_"), d.get("gpu_used_")) != (0, "gpu", 1):
                fail(f"gpu: diagnostics {d}")
            if len(gpu[2]) != 1 or "status=ineligible reason=gpu" not in gpu[2][0]:
                fail(f"gpu: final lines {gpu[2]}")
        else:
            print("NOT_RUN gpu: no --cuda-module", flush=True)

        # 9. C++-internal regressions never enter the tier.
        internal_on = run(script, args.module, root, "internal_on", kind="internal", trace=True)
        internal_off = run(script, args.module, root, "internal_off", kind="internal", switch="0")
        reasons = [field(line, "reason") for line in internal_on[2]]
        print(f"internal: {len(internal_on[2])} final lines, reasons {sorted(set(reasons))}", flush=True)
        if not internal_on[2]:
            fail("internal: no final line (trace dead?)")
        if any("status=ineligible" not in line for line in internal_on[2]):
            fail(f"internal: an internal fit was attempted: {internal_on[2]}")
        for reason in ("method", "parity_floor"):
            if reason not in reasons:
                fail(f"internal: no fit ineligible by {reason}: {reasons}")
        if internal_on[0] != internal_off[0] or not all(
                np.array_equal(internal_on[1][k], internal_off[1][k]) for k in internal_on[1]):
            fail("internal: outputs differ between the tier on and off")
    print("PASS" if not failures else f"{len(failures)} failures", flush=True)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("fit", "request", "internal"))
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--base-module", type=Path)
    parser.add_argument("--cuda-module", type=Path)
    parser.add_argument("--r1-module", type=Path)
    parser.add_argument("--design", default="akm3")
    parser.add_argument("--case", default="plain")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arrays", type=Path)
    args = parser.parse_args()
    if args.child == "fit":
        return child_fit(args)
    if args.child == "request":
        return child_request(args)
    if args.child == "internal":
        return child_internal(args)
    if args.base_module is None:
        parser.error("--base-module is required")
    for path in (args.module, args.base_module, args.cuda_module, args.r1_module):
        if path is not None:
            print(f"module {path}: {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}", flush=True)
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
