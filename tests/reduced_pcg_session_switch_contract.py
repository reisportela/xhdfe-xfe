#!/usr/bin/env python3
"""Session switch of the reduced-system tier, 2.32 candidate: the request of one
fit (HdfeOptions::reduced_pcg) through the constructor argument
reduced_pcg=None|True|False, with the process default (XHDFE_REDUCED_PCG) behind it.

Builds nothing; never run during a timing campaign. Synthetic data generated in
each child process; nothing is read from disk except the modules.

    python3 -I tests/reduced_pcg_session_switch_contract.py \
        --module build_c232b_cpu/py_hdfe_v11....so \
        --base-module <2.31.1 build>/py_hdfe_v11....so \
        --candidate-module build_c232_cpu/py_hdfe_v11....so \
        [--interop-ref-module <2.31.1 module> --interop-candidate-module <committed candidate module>]

Every result is compared on the REQUIRED set of properties (a missing property is a
failure) by dtype, shape and the raw bytes of its value, so +0.0 and -0.0 differ;
timings and scheduling-dependent counters are left out (EXCLUDED).

Checks (each failure prints one FAIL line; the last line is PASS or the count):
  1. ONE process, four regressors on the same data, in the order default, off,
     on, off: off == the base module (2.31.1) bit for bit, both offs equal,
     status 0, reason "off", 0 discarded, no final trace line; on == the
     committed candidate (tier on by default) bit for bit, outputs and
     diagnostics, status 1, reason "none", method 8; default == on;
  2. ONE process, the reverse order (on, off, default): the same bits as in 1;
  3. a process with XHDFE_REDUCED_PCG=0: reduced_pcg=True still enters the tier
     and equals the on result bit for bit; None and False equal the off result;
  4. absorption cache: an entry written by an on-fit is never read by an off-fit
     and the reverse, in the same process and in a fresh process; each entry is
     read by a fit of its own kind (positive control; XHDFE_DEBUG_SOLVER prints
     cache_hit=1);
  5. reduced_pcg given as a string or an integer raises TypeError;
  6. XHDFE_REDUCED_PCG_TRACE set to the empty string: no trace line and no fault
     (an empty value is trace level 0), the on fit unchanged;
  7. (optional, with both --interop modules) absorption-cache interoperability
     across versions for a fit, one cache file per stage (a file holds one
     record): an entry written by 2.31.1 is read by the new module with
     reduced_pcg=False, and the reverse; an entry written by the committed
     candidate (tier on) is read by the new module with the tier on, and the
     reverse; each read is a hit with the writer's bits. Crossed negatives: an
     off entry is a miss for an on fit and an on entry a miss for an off fit,
     across versions, and the reader's result is its own fresh fit.
     partial_out computes its key with the same hash_absorption_signature and
     the same effective switch, and is not separately exercised across versions.

D10 (XHDFE_REDUCED_PCG_CHAT_STOP, passed through from the caller; on unless "0")
changes the tier's stop rule, and the committed candidate predates it. With D10
off every check above runs as written. With D10 on, "on == the committed
candidate" in 1 is not checked, and in 7 the tier entries of the candidate and of
the new module are crossed negatives instead of hits (D10's own cache tag): each
read is a miss and returns the reader's fresh fit.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

# Every result property of the binding (2.31.1 and 2.32 alike) except EXCLUDED.
REQUIRED = (
    "absorption_method_used", "abs_residual_", "abs_residual_rel_", "any_omitted_",
    "auto_routing_retry_abs_residual_rel_", "auto_routing_retry_eligible_",
    "auto_routing_retry_fired_", "auto_routing_retry_iterations_",
    "auto_routing_retry_policy_enabled_", "auto_routing_retry_primary_abs_residual_rel_",
    "auto_routing_retry_primary_iterations_", "auto_routing_retry_primary_method_",
    "auto_routing_retry_status_", "cluster_combo_counts_", "cluster_counts_", "cluster_scale_",
    "coef_", "conf_int_", "converged_", "covariance_", "cuda_tail_", "cuda_tail_code_",
    "df_a_", "df_a_exact_", "df_a_levels_", "df_a_nested_", "df_m_", "df_resid_",
    "df_resid_unadj_", "fe_base_levels_", "fe_base_redundant_", "fe_inexact_", "fe_num_coefs_",
    "fe_num_levels_", "fe_recovery_converged_", "fe_recovery_iterations_",
    "fe_recovery_max_delta_", "fe_redundant_", "gpu_absorption_converged_",
    "gpu_absorption_iterations_", "gpu_attempted_", "gpu_status_code_", "gpu_used_",
    "groupvar_", "krylov_internal_tolerance_", "krylov_max_condition_",
    "krylov_max_condition_times_backward_error_", "krylov_max_final_backward_error_",
    "lifecycle_state_", "model_has_constant_", "nobs_", "nobs_effective_", "nobs_full_",
    "nobs_full_effective_", "nobs_full_rows_", "nobs_rows_", "num_clusters_",
    "num_iterations_", "num_singletons_", "num_singletons_effective_", "num_singletons_rows_",
    "omitted_", "omitted_reason_", "openmp_enabled_", "precision_certified_", "pvalues_",
    "r2_", "r2_within_", "residuals_", "rss_", "sample_index_", "saturated_",
    "slope_accuracy_retry_iterations_", "slope_accuracy_retry_stages_",
    "slope_block_checked_", "slope_block_frobenius_rel_", "slope_block_max_rel_",
    "slope_block_residual_rel_", "slope_block_rms_rel_", "slope_block_skipped_max_rel_",
    "slope_certificate_worst_fe_", "slope_certificate_worst_moment_",
    "slope_internal_tolerance_", "stderr_", "thread_limit_code_", "thread_limit_reason_",
    "threads_effective_", "threads_requested_", "threads_used_", "tss_", "tss_within_",
    "tvalues_",
)
# Timings, scheduling-dependent or host-dependent counters, object state, and
# outputs this fit does not produce.
EXCLUDED = ("auto_routing_retry_elapsed_seconds_", "parallel_workers_active_",
            "thread_capacity_", "generation_", "fe_effects_")
DIAGNOSTICS = ("reduced_pcg_status_", "reduced_pcg_reason_", "reduced_pcg_discarded_iterations_")
REDUCED_PCG = 8
SEED = 20261114
REQUEST = {"default": None, "on": True, "off": False}


def load(path: str):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def design(seed: int):
    """A synthetic worker-firm-year panel (public by construction), the akm3
    design of tests/reduced_pcg_contract.py."""
    rng = np.random.default_rng(seed)
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
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n) + 0.1 * fes[1] % 7,
                         1.0e3 + rng.normal(size=n)])
    y = X @ np.array([0.5, -0.25, 0.01]) + 0.3 * rng.normal(size=n)
    for f in fes:
        y = y + rng.normal(size=int(f.max()) + 1)[f]
    return y, np.asfortranarray(X), [np.ascontiguousarray(f, dtype=np.int32) for f in fes]


def describe(value):
    """dtype, shape and the SHA-256 of the raw bytes (so -0.0 != +0.0 and NaN
    payloads count); strings as themselves."""
    if isinstance(value, str):
        return {"str": value}
    if not isinstance(value, (bool, int, float, list, tuple, np.ndarray, np.generic)) and \
            hasattr(value, "__int__"):
        value = int(value)  # a pybind11 enum such as AbsorptionMethod
    array = np.asarray(value)
    if array.dtype == object:
        return {"repr": repr(value)}
    raw = np.ascontiguousarray(array).tobytes()
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(raw).hexdigest()}


def collect(obj, names):
    out, missing = {}, []
    for name in names:
        try:
            value = getattr(obj, name)
        except AttributeError:
            missing.append(name)
            continue
        except Exception as exc:  # a deterministic refusal is part of the result
            out[name] = {"raises": type(exc).__name__}
            continue
        out[name] = describe(value)
    return out, missing


@contextlib.contextmanager
def captured_stderr(path: Path):
    """Route file descriptor 2 (where the C++ core writes) to `path`."""
    sys.stderr.flush()
    saved = os.dup(2)
    with open(path, "w") as sink:
        os.dup2(sink.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)


def one_fit(module, request, data, out: Path, label: str, *, pass_request: bool):
    y, X, fes = data
    kwargs = dict(se_type="cluster", tol=1e-8, max_iter=100000, fit_intercept=True,
                  num_threads=8, drop_singletons=True, retain_fes=False, symmetric_sweep=False,
                  absorption_method="auto", tolerance_mode="reghdfe-comparable")
    if pass_request:
        kwargs["reduced_pcg"] = REQUEST[request]
    reg = module.HdfeRegressor(**kwargs)
    log = out / f"{label}.stderr"
    with captured_stderr(log):
        reg.fit(y, X, fes=fes, clusters=fes[0], weights=None, instruments=None,
                endogenous_idx=[], slopes=[])
    lines = log.read_text().splitlines()
    public, missing = collect(reg, REQUIRED)
    diagnostics = {name: getattr(reg, name) for name in DIAGNOSTICS if hasattr(reg, name)}
    (out / f"{label}.json").write_text(json.dumps({
        "public": public, "missing": missing, "diagnostics": diagnostics,
        "method": int(reg.absorption_method_used),
        "finals": [l for l in lines if "reduced_pcg event=final" in l],
        "cache_hit": any("xhdfe solver:" in l and "cache_hit=1" in l for l in lines)},
        sort_keys=True))


def child(args) -> int:
    module = load(args.module)
    data = design(SEED)
    out = Path(args.out)
    if args.kind == "sequence":
        for i, request in enumerate(args.requests.split(",")):
            one_fit(module, request, data, out, f"{args.label}_{i}_{request}", pass_request=True)
    elif args.kind == "plain":  # a module without the argument (2.31.1, the committed candidate)
        one_fit(module, "default", data, out, args.label, pass_request=False)
    elif args.kind == "cache":
        # steps: request:mode:file, the cache configured per fit through the environment
        for i, step in enumerate(args.requests.split(",")):
            request, mode, name = step.split(":")
            os.environ["XHDFE_ABSORPTION_CACHE"] = str(out / name)
            os.environ["XHDFE_ABSORPTION_CACHE_MODE"] = mode
            one_fit(module, request, data, out, f"{args.label}_{i}_{request}_{mode}_{name}",
                    pass_request=True)
        os.environ["XHDFE_ABSORPTION_CACHE_MODE"] = "off"
    elif args.kind == "interop":
        # requests: "<request or plain>:<mode>:<file>", one fit on its own cache
        # file (a cache file holds a single record).
        request, mode, name = args.requests.split(":")
        os.environ["XHDFE_ABSORPTION_CACHE"] = str(out / name)
        os.environ["XHDFE_ABSORPTION_CACHE_MODE"] = mode
        one_fit(module, "default" if request == "plain" else request, data, out,
                f"{args.label}_fit", pass_request=request != "plain")
    elif args.kind == "badarg":
        outcome = {}
        for bad in ("on", 1, 0, "False"):
            try:
                module.HdfeRegressor(reduced_pcg=bad)
                outcome[repr(bad)] = "accepted"
            except Exception as exc:
                outcome[repr(bad)] = type(exc).__name__
        (out / f"{args.label}.json").write_text(json.dumps(outcome, sort_keys=True))
    return 0


def launch(script, module, out, label, kind, requests="", env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("XHDFE_")}
    env.update({"XHDFE_GPU_BACKEND": "cpu", "XHDFE_ABSORPTION_CACHE_MODE": "off",
                "XHDFE_FE_STRUCTURE_MODE": "off", "XHDFE_MOBILITY_MODE": "off",
                "XHDFE_REDUCED_PCG_TRACE": "1", "XHDFE_DEBUG_SOLVER": "1",
                "OMP_WAIT_POLICY": "passive"})
    # D10's switch passes through from the caller, so the contract runs with D10
    # on (default) and off (XHDFE_REDUCED_PCG_CHAT_STOP=0).
    if "XHDFE_REDUCED_PCG_CHAT_STOP" in os.environ:
        env["XHDFE_REDUCED_PCG_CHAT_STOP"] = os.environ["XHDFE_REDUCED_PCG_CHAT_STOP"]
    if env_extra:
        env.update(env_extra)
    command = [sys.executable, "-B", "-I", str(script), "--child", "--module", str(module),
               "--kind", kind, "--requests", requests, "--label", label, "--out", str(out)]
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{label}: child failed ({done.returncode}): {done.stderr[-2000:]}")


def record(out: Path, label: str):
    return json.loads((out / f"{label}.json").read_text())


def parent(args) -> int:
    script = Path(__file__).resolve()
    failures = []

    def fail(message):
        failures.append(message)
        print("FAIL", message, flush=True)

    def complete(r, what):
        if r["missing"]:
            fail(f"{what}: missing properties {r['missing']}")

    def same(a, b, what, *, diagnostics: bool):
        complete(a, what)
        complete(b, what)
        diff = sorted(k for k in REQUIRED if a["public"].get(k) != b["public"].get(k))
        if diff:
            fail(f"{what}: differs in {diff}")
        if diagnostics and a["diagnostics"] != b["diagnostics"]:
            fail(f"{what}: diagnostics {a['diagnostics']} != {b['diagnostics']}")

    def is_off(r, what):
        d = r["diagnostics"]
        if (d.get("reduced_pcg_status_"), d.get("reduced_pcg_reason_"),
                d.get("reduced_pcg_discarded_iterations_")) != (0, "off", 0) or r["finals"]:
            fail(f"{what}: off diagnostics {d}, final lines {r['finals']}")
        if r["method"] == REDUCED_PCG:
            fail(f"{what}: off reports reduced-pcg")

    def is_on(r, what, *, read_back=False):
        d = r["diagnostics"]
        if (d.get("reduced_pcg_status_"), d.get("reduced_pcg_reason_"),
                d.get("reduced_pcg_discarded_iterations_")) != (1, "none", 0):
            fail(f"{what}: on diagnostics {d}")
        if r["method"] != REDUCED_PCG:
            fail(f"{what}: on does not report reduced-pcg")
        status = "status=cache" if read_back else "status=accepted"
        if len(r["finals"]) != 1 or status not in r["finals"][0]:
            fail(f"{what}: on final lines {r['finals']}")

    with tempfile.TemporaryDirectory(prefix="c232b_switch_") as tmp:
        out = Path(tmp)
        launch(script, args.base_module, out, "base", "plain")
        launch(script, args.candidate_module, out, "candidate", "plain")
        base, candidate = record(out, "base"), record(out, "candidate")

        # 1 and 2: one process each, both orders.
        launch(script, args.module, out, "fwd", "sequence", "default,off,on,off")
        launch(script, args.module, out, "rev", "sequence", "on,off,default")
        fwd = [record(out, f"fwd_{i}_{r}") for i, r in enumerate(("default", "off", "on", "off"))]
        rev = [record(out, f"rev_{i}_{r}") for i, r in enumerate(("on", "off", "default"))]
        default, off, on, off2 = fwd
        is_off(off, "1 off")
        is_off(off2, "1 off again")
        is_on(on, "1 on")
        is_on(default, "1 default")
        same(off, base, "1: off against the base module", diagnostics=False)
        same(off2, off, "1: off after on against off before on", diagnostics=True)
        d10_on = os.environ.get("XHDFE_REDUCED_PCG_CHAT_STOP") != "0"
        if not d10_on:
            same(on, candidate, "1: on against the committed candidate", diagnostics=True)
        else:
            print("NOTE 1: D10 on: on is not compared with the committed candidate (pre-D10)", flush=True)
        same(default, on, "1: default against on", diagnostics=True)
        for r, ref, what in ((rev[0], on, "on"), (rev[1], off, "off"), (rev[2], default, "default")):
            same(r, ref, f"2: reverse order, {what}", diagnostics=True)

        # 3: the process default off; the request still decides.
        launch(script, args.module, out, "env0", "sequence", "on,default,off",
               env_extra={"XHDFE_REDUCED_PCG": "0"})
        env_on, env_default, env_off = (record(out, f"env0_{i}_{r}")
                                        for i, r in enumerate(("on", "default", "off")))
        is_on(env_on, "3 on under XHDFE_REDUCED_PCG=0")
        is_off(env_default, "3 default under XHDFE_REDUCED_PCG=0")
        is_off(env_off, "3 off under XHDFE_REDUCED_PCG=0")
        same(env_on, on, "3: True under XHDFE_REDUCED_PCG=0 against on", diagnostics=True)
        same(env_default, off, "3: default under XHDFE_REDUCED_PCG=0 against off", diagnostics=True)
        same(env_off, off, "3: False under XHDFE_REDUCED_PCG=0 against off", diagnostics=True)

        # 4: cache entries never cross, in one process and across processes.
        steps = ("on:write:on.cache,off:read:on.cache,on:read:on.cache,"
                 "off:write:off.cache,on:read:off.cache,off:read:off.cache")
        launch(script, args.module, out, "cache", "cache", steps)
        names = [s.replace(":", "_") for s in steps.split(",")]
        cache = [record(out, f"cache_{i}_{n}") for i, n in enumerate(names)]
        launch(script, args.module, out, "xcache", "cache", "off:read:on.cache,on:read:off.cache")
        xnames = ["off_read_on.cache", "on_read_off.cache"]
        xcache = [record(out, f"xcache_{i}_{n}") for i, n in enumerate(xnames)]
        for r, what in ((cache[1], "off read the on entry"), (cache[4], "on read the off entry"),
                        (xcache[0], "fresh process: off read the on entry"),
                        (xcache[1], "fresh process: on read the off entry")):
            if r["cache_hit"]:
                fail(f"4: {what}")
        for r, what in ((cache[1], "same process"), (xcache[0], "fresh process")):
            is_off(r, f"4 off after the on entry, {what}")
            same(r, off, f"4: off reading the on entry ({what}) against a fresh off", diagnostics=True)
        for r, what in ((cache[4], "same process"), (xcache[1], "fresh process")):
            same(r, on, f"4: on reading the off entry ({what}) against a fresh on", diagnostics=True)
        if not cache[2]["cache_hit"]:
            fail("4: positive control, the on entry was not read by an on-fit")
        is_on(cache[2], "4 on reading the on entry", read_back=True)
        same(cache[2], on, "4: on read back against a fresh on", diagnostics=True)
        if not cache[5]["cache_hit"]:
            fail("4: positive control, the off entry was not read by an off-fit")
        same(cache[5], off, "4: off read back against a fresh off", diagnostics=True)

        # 5: the argument takes None, True or False only.
        launch(script, args.module, out, "badarg", "badarg")
        bad = json.loads((out / "badarg.json").read_text())
        if any(v != "TypeError" for v in bad.values()):
            fail(f"5: invalid reduced_pcg values {bad}")

        # 6: an empty trace variable is level 0 (no line, no fault).
        launch(script, args.module, out, "emptytrace", "sequence", "on",
               env_extra={"XHDFE_REDUCED_PCG_TRACE": ""})
        empty = record(out, "emptytrace_0_on")
        lines = (out / "emptytrace_0_on.stderr").read_text().splitlines()
        if any("reduced_pcg" in l for l in lines):
            fail(f"6: trace lines with an empty XHDFE_REDUCED_PCG_TRACE: {lines[:3]}")
        same(empty, on, "6: on with an empty trace variable against on", diagnostics=True)

        # 7: cache interoperability across versions (optional), fit level, one
        # cache file per stage. partial_out computes its key with the same
        # hash_absorption_signature and the same effective switch, and is not
        # separately exercised across versions.
        if args.interop_ref_module and args.interop_candidate_module:
            ref, cand, new = args.interop_ref_module, args.interop_candidate_module, args.module

            def write(module, request, name):
                launch(script, module, out, f"w_{name}", "interop", f"{request}:write:{name}")
                return record(out, f"w_{name}_fit")

            def read(module, request, name, label):
                launch(script, module, out, f"r_{label}", "interop", f"{request}:read:{name}")
                return record(out, f"r_{label}_fit")

            w1 = write(ref, "plain", "i1.cache")     # 2.31.1: the base key
            w2 = write(new, "off", "i2.cache")       # new module, tier off
            w3 = write(cand, "plain", "i3.cache")    # committed candidate, tier on
            w4 = write(new, "on", "i4.cache")        # new module, tier on
            same(w1, base, "7: the 2.31.1 writer against the base module", diagnostics=False)
            same(w2, off, "7: the new off writer against a fresh off", diagnostics=True)
            same(w3, candidate, "7: the candidate writer against the candidate", diagnostics=True)
            same(w4, on, "7: the new on writer against a fresh on", diagnostics=True)
            hits = [
                (read(new, "off", "i1.cache", "i1_new_off"), w1, "2.31.1 writes, new off reads", False),
                (read(ref, "plain", "i2.cache", "i2_ref"), w2, "new off writes, 2.31.1 reads", False),
            ]
            tier_cross = [
                (read(new, "on", "i3.cache", "i3_new_on"), w3, on, "candidate writes, new on reads", True),
                (read(cand, "plain", "i4.cache", "i4_cand"), w4, candidate, "new on writes, candidate reads", True),
            ]
            if not d10_on:
                hits += [(r, writer, what, diagnostics) for r, writer, _, what, diagnostics in tier_cross]
            for r, writer, what, diagnostics in hits:
                if not r["cache_hit"]:
                    fail(f"7: {what}: the entry was not read")
                same(r, writer, f"7: {what}: read back against the writer", diagnostics=diagnostics)
            same(hits[0][0], off, "7: new off reading the 2.31.1 entry against a fresh off", diagnostics=True)
            if not d10_on:
                is_on(hits[2][0], "7 new on reading the candidate's entry", read_back=True)
            else:  # D10 on: its own cache tag, so the tier entries do not cross versions
                for r, _, fresh, what, diagnostics in tier_cross:
                    if r["cache_hit"]:
                        fail(f"7: D10 on: {what}: the entry was read")
                    same(r, fresh, f"7: D10 on: {what}: against the reader's fresh fit", diagnostics=diagnostics)
                is_on(tier_cross[0][0], "7 D10 on: new on after the candidate's entry")
            misses = (
                (read(cand, "plain", "i1.cache", "i1_cand"), candidate, "the 2.31.1 off entry, candidate on reads", True),
                (read(new, "on", "i1.cache", "i1_new_on"), on, "the 2.31.1 off entry, new on reads", True),
                (read(ref, "plain", "i3.cache", "i3_ref"), base, "the candidate on entry, 2.31.1 reads", False),
                (read(new, "off", "i3.cache", "i3_new_off"), off, "the candidate on entry, new off reads", True),
            )
            for r, fresh, what, diagnostics in misses:
                if r["cache_hit"]:
                    fail(f"7: {what}: the entry was read")
                same(r, fresh, f"7: {what}: against the reader's fresh fit", diagnostics=diagnostics)
        else:
            print("SKIP 7: --interop-ref-module and --interop-candidate-module not given", flush=True)

        print(f"on: method {on['method']} iterations {on['public']['num_iterations_']}; "
              f"off: method {off['method']} iterations {off['public']['num_iterations_']}", flush=True)
    print("PASS" if not failures else f"{len(failures)} FAILURES", flush=True)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--module", required=True)
    parser.add_argument("--base-module")
    parser.add_argument("--candidate-module")
    parser.add_argument("--interop-ref-module")
    parser.add_argument("--interop-candidate-module")
    parser.add_argument("--kind", default="")
    parser.add_argument("--requests", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.child:
        return child(args)
    if not args.base_module or not args.candidate_module:
        parser.error("--base-module and --candidate-module are required")
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
