#!/usr/bin/env python3
"""Per-fit request of the experimental CUDA reduced tier (package 1, 16 November 2026, with its
Amendment 1): HdfeOptions::reduced_pcg_cuda through the constructor argument
reduced_pcg_cuda=None|True|False, with the process default behind it (the environment variable
XHDFE_REDUCED_PCG_CUDA, read once per process; only "1" requests the tier).

Builds nothing; never run during a timing campaign. Synthetic data generated in each child process
(the design of tests/reduced_cuda_runtime_contract.py, and for the saved-effects cases a
low-mobility worker-firm-year design, see design()); nothing is read from disk except the
modules. The Python regressor has NO backend argument: the child sets os.environ["XHDFE_GPU_BACKEND"]
before each fit (the core reads it at every fit), which is how one process, and one regressor
object, alternate CPU and CUDA fits.

    python3 -B -I tests/reduced_cuda_request_contract.py \
        --module build_p1_cuda/py_hdfe_v11....so --output <new directory> \
        [--cpu-module build_p1_cpu/py_hdfe_v11....so] \
        [--frozen-cuda <frozen CUDA module>] [--frozen-cpu <frozen CPU module>]

"Within the runtime limits" means the limits of tests/reduced_cuda_runtime_contract.py: the same
sample and omissions, b within 1e-8*max(1,|b|), V within 1e-8*sqrt(Vjj*Vkk), standard errors within
1e-7 relative and residuals within 1e-7 absolute. "Bit for bit" compares the required result
properties by dtype, shape and the raw bytes of their values (as
tests/reduced_pcg_session_switch_contract.py does). CUDA fits are never required to repeat bits
across processes; the comparison reports when they do.

Checks (each failure prints one FAIL line; the last line is PASS or the count):
  1. ONE process, in this order: CPU with the CPU tier (default); CUDA with reduced_pcg_cuda=True;
     CUDA with no request; CUDA with reduced_pcg_cuda=False; CPU with reduced_pcg=False; CUDA with
     the request again. Each fit: method, tier status and reason, gpu_used_, gpu_status_code_ and its
     trace's final line. The two CUDA tier fits agree within the runtime limits, and so do the two
     CUDA fits without the tier. ONE process, the reverse order: the same diagnostics, the CPU fits
     bit for bit those of the forward order, the CUDA fits within the runtime limits.
  2. The two CPU fits of 1 equal, bit for bit, the same fits run first, each in a fresh process.
  3. Fresh processes: the request with nothing in the environment enters. With
     XHDFE_REDUCED_PCG_CUDA=0: the request on enters, off and absent do not (reason gpu). With
     XHDFE_REDUCED_PCG_CUDA=1: absent, None and on enter, off does not (reason gpu). The first fit of
     each of these two processes has the trace lines of the request with nothing in the environment
     (timings left out) and agrees with it within the runtime limits. The general switch decides
     first: reduced_pcg=False with the request does not enter (reason off, no final line); in a
     process with XHDFE_REDUCED_PCG=0 the request alone does not enter (reason off) and
     reduced_pcg=True with the request does. In xhdfe-fast the request does not enter (reason
     tolerance_mode).
  4. A profile or a cache keeps the request out and says which (Amendment 1, A1): with a mobility
     profile, an absorption cache or an FE-structure cache active (mode write, a file of its own per
     fit) the request reports status 0 and reason mobility_profile, absorption_cache or
     fe_structure_cache, and runs the CUDA route of the same fit without the request, which reports
     reason gpu: both use the GPU and neither reports method 8; with either cache they agree within
     the runtime limits (a profile writer keeps the fastest of several methods, so its two legs are
     compared on diagnostics only). The implicit profile: a file xhdfe_mobility_profile.txt in the
     working directory gives reason mobility_profile, and the same fit enters once the process works
     in a directory without one (os.chdir; nothing is deleted).
  5. reduced_pcg_cuda given as a string or an integer raises TypeError; None, True and False are
     accepted; the results of a regressor fitted before the invalid construction are untouched.
  6. ONE regressor object with the request, reused across CPU, CUDA, CPU, CUDA fits: CPU tier, CUDA
     tier, the same CPU bits (and those of the fresh CPU fit of 2), the CUDA fits within the runtime
     limits.
  7. Saved effects (A2), on the low-mobility design whose CUDA capturing sweep stops certified and
     above the floor of the mode, so that the projection of the correction E1 runs: the projection
     is published
     (XHDFE_AUTO_ROUTING_RETRY_DIAG), it reaches the CUDA tier (an operator line with products), the
     coefficients, covariance and residuals agree with the twin without saved effects within the
     runtime limits, the saved effects are present and finite, and the diagnostics are printed as a
     user sees them. In ONE regressor with saved effects and the request: a fit, the same fit with
     XHDFE_TEST_REDUCED_PCG_CUDA_FATAL=1 (an error naming the injected fatal error, no estimates,
     lifecycle failed), the same fit with the injection off again (the bytes of the first fit).
  8. Other regressors are not contaminated (A5): an AKM decomposition and a Gelbach decomposition
     (CPU) give the same bytes before and after a fit that carried the request, in one process.
  9. (with --cpu-module) the CPU module in one process: default, reduced_pcg=False, and
     reduced_pcg_cuda=True, which equals the default bit for bit (the request is inert on CPU); with
     --frozen-cpu, the default and reduced_pcg=False fits equal those of the frozen CPU module bit
     for bit.
 10. (with --frozen-cuda) the frozen CUDA module of 90f8aeda8, fresh processes: its two CPU fits
     equal those of 2 bit for bit; its CUDA fit without XHDFE_REDUCED_PCG_CUDA has the diagnostics of
     the new no-request fit and agrees with it within the runtime limits; its CUDA fit with
     XHDFE_REDUCED_PCG_CUDA=1 has the diagnostics of the new request with nothing in the environment,
     and agrees with it within the runtime limits. Since package 2 the two binaries carry different
     CUDA operators, so their traces are no longer compared line for line (that comparison is check 3,
     inside the same new binary): both must have entered the CUDA tier (method 8, tier status 1, an
     operator line with products), and the iterations, tau, chat and the final residual of both are
     printed and recorded. For the record (a NOTE, not a check): the frozen module's saved-effects fit
     with XHDFE_REDUCED_PCG_CUDA=1 and the injected fatal error.
 11. (package 2, C4) Failures injected into the CUDA operator, ONE process, one fit each:
     XHDFE_TEST_REDUCED_PCG_CUDA_FAIL=allocation (a device allocation in the middle of the set-up),
     staging (a page-locked upload chunk) and register (the registration of p and q). Each attempt
     declines, with reason memory, pinned and pinned respectively (tier status 2, a declined final
     line on CUDA), and the fit is the CUDA route's without the request, within the runtime limits
     of it; then, with the variable cleared, the same process enters the tier again and agrees with
     the fresh request within the runtime limits. (The failure in the middle of the build of the
     level arrays is tests/reduced_cuda_operator_smoke.cpp's: these designs build no such array.)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys

import numpy as np

# Every result property of the binding except EXCLUDED (the list of
# tests/reduced_pcg_session_switch_contract.py).
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
# Timings, scheduling-dependent or host-dependent counters, object state, and outputs
# this fit does not produce.
EXCLUDED = ("auto_routing_retry_elapsed_seconds_", "parallel_workers_active_",
            "thread_capacity_", "generation_", "fe_effects_")
DIAGNOSTICS = ("reduced_pcg_status_", "reduced_pcg_reason_", "reduced_pcg_discarded_iterations_")
ARRAYS = ("coef_", "covariance_", "stderr_", "residuals_", "sample_index_", "omitted_reason_")
REDUCED_PCG = 8
SEED = 2026111502
THREADS = 4
VALUES = {"none": None, "on": True, "off": False}
MODES = {"comparable": "reghdfe-comparable", "fast": "xhdfe-fast"}
# Trace fields that measure time; every other field of a reduced_pcg line is compared.
TIMING = re.compile(r"(seconds|_ms)$|^timed$")
CACHE_ENV = {
    "absorption_cache": ("XHDFE_ABSORPTION_CACHE_MODE", "XHDFE_ABSORPTION_CACHE"),
    "fe_structure_cache": ("XHDFE_FE_STRUCTURE_MODE", "XHDFE_FE_STRUCTURE_CACHE"),
    "mobility_profile": ("XHDFE_MOBILITY_MODE", "XHDFE_MOBILITY_PROFILE"),
}
# The reason a requested fit reports when that profile or cache keeps it out (A1).
CACHE_REASON = {"absorption_cache": "absorption_cache", "fe_structure_cache": "fe_structure_cache",
                "mobility_profile": "mobility_profile"}
# The implicit mobility profile (kMobilityProfileDefaultPath of the regressor).
PROFILE_NAME = "xhdfe_mobility_profile.txt"
FATAL = "XHDFE_TEST_REDUCED_PCG_CUDA_FATAL"
FAIL = "XHDFE_TEST_REDUCED_PCG_CUDA_FAIL"
# The failures injected into the CUDA operator (check 11) and the decline reason of each.
INJECTED = (("allocation", "memory"), ("staging", "pinned"), ("register", "pinned"))


def load(path: str):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == Path(path).resolve()
    return module


def design(kind: str):
    """runtime: the worker-firm-year design of tests/reduced_cuda_runtime_contract.py (it enters
    the CUDA tier there; its capturing sweep on CUDA ends below the floor of the mode, about
    3e-18, so the projection does not run). lowmob: 8000 workers over 10 years in 400 firms, each
    worker leaving its firm with probability 0.005 per year: the CUDA capturing sweep stops
    certified and above the floor (about 2.2e-9) and the CUDA tier publishes the projection (the
    path-like design of tests/savefe_comparable_projection_contract.py does not converge on the
    CUDA route within max_iter). Public by construction. Returns y, X, fes, the cluster
    identifier."""
    if kind == "lowmob":
        rng = np.random.default_rng(20261117)
        workers, firms, years = 8_000, 400, 10
        worker = np.repeat(np.arange(workers), years)
        year = np.tile(np.arange(years), workers)
        home = rng.integers(0, firms, size=workers)
        moved = np.cumsum(rng.random(worker.size).reshape(workers, years) < 0.005, axis=1).ravel()
        firm = (home[worker] + 7 * moved) % firms
        fes = [worker.astype(np.int32), firm.astype(np.int32), year.astype(np.int32)]
        X = np.asfortranarray(rng.normal(size=(worker.size, 2)))
        y = X @ np.array([0.5, -0.25]) + rng.normal(size=worker.size)
        for f in fes:
            y += rng.normal(size=int(f.max()) + 1)[f]
        return y, X, fes, fes[0]
    rng = np.random.default_rng(SEED)
    n = 49152
    fes = [np.repeat(np.arange(n // 24, dtype=np.int32), 24),
           rng.integers(0, 83, n, dtype=np.int32), rng.integers(0, 13, n, dtype=np.int32)]
    X = np.asfortranarray(rng.normal(size=(n, 3)))
    y = X @ np.array([.5, -.25, .01]) + rng.normal(size=n)
    for f in fes:
        y += rng.normal(size=int(f.max()) + 1)[f]
    return y, X, fes, fes[0]


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


def attribute(obj, name, convert):
    try:
        return convert(getattr(obj, name))
    except Exception as exc:  # recorded, so that a failed fit still yields a record
        return f"raises {type(exc).__name__}"


def saved_effects(reg):
    """Shape and finiteness of the saved effects, or what reading them raised."""
    try:
        columns = [np.asarray(v, dtype=np.float64) for v in reg.fe_effects_]
    except Exception as exc:
        return {"raises": type(exc).__name__}
    if not columns:
        return {"shape": [0, 0], "finite": False}
    stacked = np.column_stack(columns)
    return {"shape": list(stacked.shape), "finite": bool(np.all(np.isfinite(stacked)))}


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


def parse_fit(step: str):
    """backend/reduced_pcg/reduced_pcg_cuda/mode[/savefe]; '-' leaves an argument out."""
    parts = step.split("/")
    return parts[0], parts[1], parts[2], parts[3], len(parts) > 4 and parts[4] == "savefe"


def regressor(module, request, cuda_request, mode, savefe):
    kwargs = dict(se_type="cluster", tol=1e-8, max_iter=10000, num_threads=THREADS,
                  tolerance_mode=MODES[mode], retain_fes=savefe)
    if request != "-":
        kwargs["reduced_pcg"] = VALUES[request]
    if cuda_request != "-":
        kwargs["reduced_pcg_cuda"] = VALUES[cuda_request]
    return module.HdfeRegressor(**kwargs)


def one_fit(reg, backend, step, data, out: Path, label: str):
    os.environ["XHDFE_GPU_BACKEND"] = backend
    y, X, fes, cluster = data
    log = out / f"{label}.stderr"
    error = None
    with captured_stderr(log):
        try:
            reg.fit(y, X, fes=fes, clusters=[cluster])
        except RuntimeError as exc:
            error = str(exc)
    lines = log.read_text().splitlines()
    public, missing = collect(reg, REQUIRED)
    arrays = {}
    for name in ARRAYS:
        try:
            arrays[name] = np.asarray(getattr(reg, name))
        except Exception:
            arrays[name] = np.asarray([])
    np.savez(out / f"{label}.npz", **arrays)
    (out / f"{label}.json").write_text(json.dumps({
        "step": step, "cwd": os.getcwd(), "error": error, "public": public, "missing": missing,
        "diagnostics": {name: attribute(reg, name, lambda v: v) for name in DIAGNOSTICS},
        "method": attribute(reg, "absorption_method_used", int),
        "gpu_used": attribute(reg, "gpu_used_", bool),
        "gpu_status_code": attribute(reg, "gpu_status_code_", int),
        "converged": attribute(reg, "converged_", bool),
        "precision_certified": attribute(reg, "precision_certified_", bool),
        "lifecycle": attribute(reg, "lifecycle_state_", str),
        "iterations": attribute(reg, "num_iterations_", int),
        "abs_residual_rel": attribute(reg, "abs_residual_rel_", float),
        "saved_effects": saved_effects(reg),
        "trace": [l for l in lines if "xhdfe reduced_pcg" in l],
        "finals": [l for l in lines if "reduced_pcg event=final" in l],
        "projection": [l for l in lines if "savefe_accurate_projection" in l],
        "npz": str(out / f"{label}.npz")}, sort_keys=True, default=str))


def akm_gelbach(module, out: Path, label: str):
    """The AKM (with controls, so the FWL fit runs) and Gelbach cases of
    tests/reduced_pcg_contract.py, on the CPU."""
    os.environ["XHDFE_GPU_BACKEND"] = "cpu"
    rng = np.random.default_rng(20261116)
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
    np.savez(out / f"{label}.npz", **arrays)


def child(args) -> int:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    module = load(args.module)
    out = Path(args.out)
    if args.kind == "badarg":
        # A regressor fitted first: an invalid construction must leave its results untouched.
        data = design("runtime")
        reg = regressor(module, "-", "-", "comparable", False)
        one_fit(reg, "cpu", "cpu/-/-/comparable", data, out, f"{args.label}_before")
        outcome = {}
        for value in ("on", 1, 0, "False", None, True, False):
            try:
                module.HdfeRegressor(reduced_pcg_cuda=value)
                outcome[repr(value)] = "accepted"
            except Exception as exc:
                outcome[repr(value)] = type(exc).__name__
        after, _ = collect(reg, REQUIRED)
        before = json.loads((out / f"{args.label}_before.json").read_text())["public"]
        outcome["untouched"] = sorted(k for k in REQUIRED if before.get(k) != after.get(k))
        (out / f"{args.label}.json").write_text(json.dumps(outcome, sort_keys=True))
        return 0
    data = design(args.design)
    if args.kind == "internal":
        akm_gelbach(module, out, f"{args.label}_before")
        reg = regressor(module, "-", "on", "comparable", False)
        one_fit(reg, "cuda", "cuda/-/on/comparable", data, out, f"{args.label}_0")
        akm_gelbach(module, out, f"{args.label}_after")
        return 0
    reused, fits = None, 0
    for step in args.steps.split(","):
        if step.startswith("chdir:"):
            os.chdir(step[len("chdir:"):])
            continue
        if step.startswith("env:"):
            name, value = step[len("env:"):].split("=", 1)
            os.environ[name] = value
            continue
        backend, request, cuda_request, mode, savefe = parse_fit(step)
        if args.kind == "reuse":
            if reused is None:
                reused = (regressor(module, request, cuda_request, mode, savefe),
                          (request, cuda_request, mode, savefe))
            # One object: only the backend may change between its fits.
            assert reused[1] == (request, cuda_request, mode, savefe), step
            reg = reused[0]
        else:
            reg = regressor(module, request, cuda_request, mode, savefe)
        one_fit(reg, backend, step, data, out, f"{args.label}_{fits}")
        fits += 1
    return 0


def trace_signature(record):
    """The fit's reduced_pcg trace lines without their timing fields."""
    out = []
    for line in record["trace"]:
        kept = [token for token in line.split()
                if "=" not in token or not TIMING.search(token.split("=", 1)[0])]
        out.append(" ".join(kept))
    return out


def parent(args) -> int:
    script = Path(__file__).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=False, exist_ok=False)  # a new directory: nothing is overwritten
    failures, rows = [], []

    def fail(message):
        failures.append(message)
        print("FAIL", message, flush=True)

    if shutil.which("nvidia-smi"):
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True)
        (out / "gpu_apps_before.txt").write_text(apps.stdout + apps.stderr)

    def launch(module, label, steps="", env_extra=None, kind="sequence", design_kind="runtime",
               cwd=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("XHDFE_")}
        env.update({"XHDFE_ABSORPTION_CACHE_MODE": "off", "XHDFE_FE_STRUCTURE_MODE": "off",
                    "XHDFE_MOBILITY_MODE": "off", "XHDFE_REDUCED_PCG_TRACE": "1",
                    "OMP_WAIT_POLICY": "passive", "OPENBLAS_NUM_THREADS": "1",
                    "CUDA_CACHE_PATH": str(out / "cuda_cache"),
                    "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "0"})
        # D10's switch passes through from the caller, as in the session-switch contract.
        if "XHDFE_REDUCED_PCG_CHAT_STOP" in os.environ:
            env["XHDFE_REDUCED_PCG_CHAT_STOP"] = os.environ["XHDFE_REDUCED_PCG_CHAT_STOP"]
        if env_extra:
            env.update(env_extra)
        # A variable set to None is left out (the implicit profile needs no XHDFE_MOBILITY_MODE).
        env = {k: v for k, v in env.items() if v is not None}
        command = [sys.executable, "-B", "-I", str(script), "--child",
                   "--module", str(Path(module).resolve()),
                   "--kind", kind, "--steps", steps, "--label", label, "--out", str(out),
                   "--design", design_kind]
        done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1800,
                              cwd=cwd)
        (out / f"{label}.child_stdout.txt").write_text(done.stdout)
        (out / f"{label}.child_stderr.txt").write_text(done.stderr)
        if done.returncode != 0:
            raise RuntimeError(f"{label}: child failed ({done.returncode}): {done.stderr[-2000:]}")
        if kind == "badarg":
            return json.loads((out / f"{label}.json").read_text())
        if kind == "internal":
            return json.loads((out / f"{label}_0.json").read_text())
        count = sum(1 for s in steps.split(",") if not s.startswith(("chdir:", "env:")))
        records = [json.loads((out / f"{label}_{i}.json").read_text()) for i in range(count)]
        for i, r in enumerate(records):
            r["label"] = f"{label}_{i}"
        return records

    def summary(r):
        d = r["diagnostics"]
        return (f"method {r['method']} status {d.get('reduced_pcg_status_')} "
                f"reason {d.get('reduced_pcg_reason_')} gpu_used {r['gpu_used']} "
                f"gpu_status {r['gpu_status_code']} iterations {r['iterations']} "
                f"abs_residual_rel {r['abs_residual_rel']} error {r['error']}")

    def expect(r, what, *, tier: bool, reason: str, gpu: bool):
        d = r["diagnostics"]
        got = (r["error"], r["method"] == REDUCED_PCG, d.get("reduced_pcg_status_"),
               d.get("reduced_pcg_reason_"), r["gpu_used"], r["gpu_status_code"])
        want = (None, tier, 1 if tier else 0, "none" if tier else reason, gpu, 1 if gpu else 0)
        if got != want:
            fail(f"{what}: {summary(r)}; expected method 8 {tier}, status {want[2]}, "
                 f"reason {want[3]}, gpu_used {gpu}, gpu_status {want[5]}")
        finals = r["finals"]
        if tier:
            backend = " backend=cuda " in finals[0] + " " if finals else False
            if len(finals) != 1 or "status=accepted" not in finals[0] or backend != gpu:
                fail(f"{what}: final lines {finals}")
            calls = [l for l in r["trace"] if "reduced_pcg_cuda event=operator calls=" in l]
            if gpu and (len(calls) != 1 or " calls=0 " in calls[0] + " "):
                fail(f"{what}: CUDA operator lines {calls}")
            if not gpu and calls:
                fail(f"{what}: a CPU fit printed CUDA operator lines {calls}")
        elif reason == "off":
            if finals:
                fail(f"{what}: the tier is off, yet final lines {finals}")
        elif len(finals) != 1 or f"status=ineligible reason={reason}" not in finals[0]:
            fail(f"{what}: final lines {finals}")
        rows.append({"label": r["label"], "what": what, "result": summary(r)})

    def same_bits(a, b, what):
        for r in (a, b):
            if r["missing"]:
                fail(f"{what}: missing properties {r['missing']}")
        diff = sorted(k for k in REQUIRED if a["public"].get(k) != b["public"].get(k))
        if diff:
            fail(f"{what}: differs in {diff}")
        if a["diagnostics"] != b["diagnostics"]:
            fail(f"{what}: diagnostics {a['diagnostics']} != {b['diagnostics']}")
        rows.append({"what": what, "comparison": "bit for bit", "differs": diff})

    def within(a, b, what):
        if a["error"] or b["error"]:
            fail(f"{what}: errors {a['error']!r} {b['error']!r}")
            return
        x, y = np.load(a["npz"]), np.load(b["npz"])
        exact = all(a["public"].get(k) == b["public"].get(k) for k in ARRAYS)
        for key in ("sample_index_", "omitted_reason_"):
            if not np.array_equal(x[key], y[key]):
                fail(f"{what}: {key} differs")
                return
        use = y["omitted_reason_"] == 0
        db = float(np.max(np.abs(x["coef_"][use] - y["coef_"][use]) /
                          np.maximum(1, np.abs(y["coef_"][use]))))
        V = y["covariance_"][np.ix_(use, use)]
        sd = np.sqrt(np.maximum(0, np.diag(V)))
        scale = np.outer(sd, sd)
        delta = np.abs(x["covariance_"][np.ix_(use, use)] - V)
        dv = float(np.max(np.divide(delta, scale, out=np.where(delta == 0, 0.0, np.inf),
                                    where=scale > 0)))
        ds = float(np.max(np.abs(x["stderr_"][use] / y["stderr_"][use] - 1)))
        dr = float(np.max(np.abs(x["residuals_"] - y["residuals_"])))
        rows.append({"what": what, "comparison": "exact bytes" if exact else "runtime limits",
                     "b": db, "V": dv, "se": ds, "residual": dr})
        if not (db <= 1e-8 and dv <= 1e-8 and ds <= 1e-7 and dr <= 1e-7):
            fail(f"{what}: b {db:.3e} V {dv:.3e} se {ds:.3e} residual {dr:.3e} "
                 f"beyond the runtime limits")
        print(f"{what}: {'exact bytes' if exact else 'within the runtime limits'} "
              f"(b {db:.3e} V {dv:.3e} se {ds:.3e} residual {dr:.3e})", flush=True)

    def same_trace(a, b, what):
        sa, sb = trace_signature(a), trace_signature(b)
        if sa != sb:
            first = next((i for i, (u, v) in enumerate(zip(sa, sb)) if u != v), min(len(sa), len(sb)))
            fail(f"{what}: trace lines differ at line {first}: "
                 f"{sa[first] if first < len(sa) else '<none>'} | {sb[first] if first < len(sb) else '<none>'}")
        rows.append({"what": what, "comparison": "trace lines without timings", "lines": len(sa)})

    def entered_cuda_tier(r):
        return any("reduced_pcg_cuda event=operator calls=" in l and " calls=0 " not in l + " "
                   for l in r["trace"])

    def expect_declined(r, what, reason):
        """An attempt of the CUDA tier that declined: tier status 2 with the reason, the CUDA
        route's fit (gpu_used, not method 8), one declined final line on CUDA."""
        d = r["diagnostics"]
        got = (r["error"], r["method"] == REDUCED_PCG, d.get("reduced_pcg_status_"), d.get("reduced_pcg_reason_"),
               r["gpu_used"], r["gpu_status_code"])
        if got != (None, False, 2, reason, True, 1):
            fail(f"{what}: {summary(r)}; expected a declined attempt (status 2, reason {reason}) and the CUDA route")
        finals = r["finals"]
        if (len(finals) != 1 or f"status=declined reason={reason} " not in finals[0] + " "
                or " backend=cuda " not in finals[0] + " "):
            fail(f"{what}: final lines {finals}")
        rows.append({"label": r["label"], "what": what, "result": summary(r), "finals": finals})

    def tier_record(r):
        """Iterations, tau, chat of every column and the final residual of a tier fit."""
        columns = [" ".join(t for t in l.split() if t.split("=")[0] in ("col", "iterations", "tau", "chat"))
                   for l in r["trace"] if "reduced_pcg event=column" in l]
        final = next((l for l in r["finals"]), "")
        rel = next((t.split("=", 1)[1] for t in final.split() if t.startswith("rel=")), None)
        return {"columns": columns, "rel": rel, "iterations": r["iterations"]}

    # 1: one process, CPU and CUDA fits alternating, then the reverse order.
    sequence = ("cpu/-/-/comparable", "cuda/-/on/comparable", "cuda/-/-/comparable",
                "cuda/-/off/comparable", "cpu/off/-/comparable", "cuda/-/on/comparable")
    cpu_tier, cuda_tier, cuda_plain, cuda_off, cpu_off, cuda_tier2 = \
        launch(args.module, "seq", ",".join(sequence))
    expect(cpu_tier, "1 CPU, tier by default", tier=True, reason="none", gpu=False)
    expect(cuda_tier, "1 CUDA with the request", tier=True, reason="none", gpu=True)
    expect(cuda_plain, "1 CUDA without the request", tier=False, reason="gpu", gpu=True)
    expect(cuda_off, "1 CUDA with reduced_pcg_cuda=False", tier=False, reason="gpu", gpu=True)
    expect(cpu_off, "1 CPU with reduced_pcg=False", tier=False, reason="off", gpu=False)
    expect(cuda_tier2, "1 CUDA with the request again", tier=True, reason="none", gpu=True)
    within(cuda_tier2, cuda_tier, "1: the two CUDA tier fits")
    within(cuda_off, cuda_plain, "1: the two CUDA fits without the tier")
    rev = launch(args.module, "rev", ",".join(reversed(sequence)))
    for r, ref, what in zip(rev, (cuda_tier2, cpu_off, cuda_off, cuda_plain, cuda_tier, cpu_tier),
                            ("CUDA request", "CPU reduced_pcg=False", "CUDA False", "CUDA absent",
                             "CUDA request", "CPU default")):
        d = ref["diagnostics"]
        expect(r, f"1 reverse order, {what}", tier=ref["method"] == REDUCED_PCG,
               reason=d.get("reduced_pcg_reason_"), gpu=bool(ref["gpu_used"]))
        if ref["gpu_used"]:
            within(r, ref, f"1: reverse order, {what}, against the forward order")
        else:
            same_bits(r, ref, f"1: reverse order, {what}, against the forward order")

    # 2: the CPU fits of 1 against the same fits first in a fresh process.
    (fresh_cpu,) = launch(args.module, "fresh_cpu", "cpu/-/-/comparable")
    (fresh_cpu_off,) = launch(args.module, "fresh_cpu_off", "cpu/off/-/comparable")
    same_bits(cpu_tier, fresh_cpu, "2: CPU tier fit after CUDA fits against a fresh process")
    same_bits(cpu_off, fresh_cpu_off, "2: CPU fit with reduced_pcg=False against a fresh process")

    # 3: precedence and the general switch, fresh processes.
    (request,) = launch(args.module, "request", "cuda/-/on/comparable")
    expect(request, "3 request, nothing in the environment", tier=True, reason="none", gpu=True)
    within(request, cuda_tier, "3: fresh request against the request of 1")
    env0 = launch(args.module, "env0", "cuda/-/on/comparable,cuda/-/off/comparable,cuda/-/-/comparable",
                  env_extra={"XHDFE_REDUCED_PCG_CUDA": "0"})
    expect(env0[0], "3 XHDFE_REDUCED_PCG_CUDA=0, request on", tier=True, reason="none", gpu=True)
    expect(env0[1], "3 XHDFE_REDUCED_PCG_CUDA=0, request off", tier=False, reason="gpu", gpu=True)
    expect(env0[2], "3 XHDFE_REDUCED_PCG_CUDA=0, no request", tier=False, reason="gpu", gpu=True)
    env1 = launch(args.module, "env1",
                  "cuda/-/-/comparable,cuda/-/none/comparable,cuda/-/on/comparable,cuda/-/off/comparable",
                  env_extra={"XHDFE_REDUCED_PCG_CUDA": "1"})
    expect(env1[0], "3 XHDFE_REDUCED_PCG_CUDA=1, no argument", tier=True, reason="none", gpu=True)
    expect(env1[1], "3 XHDFE_REDUCED_PCG_CUDA=1, None", tier=True, reason="none", gpu=True)
    expect(env1[2], "3 XHDFE_REDUCED_PCG_CUDA=1, request on", tier=True, reason="none", gpu=True)
    expect(env1[3], "3 XHDFE_REDUCED_PCG_CUDA=1, request off", tier=False, reason="gpu", gpu=True)
    for r, what in ((env0[0], "XHDFE_REDUCED_PCG_CUDA=0 with the request"),
                    (env1[0], "XHDFE_REDUCED_PCG_CUDA=1 without the argument")):
        same_trace(r, request, f"3: {what} against the request alone")
        within(r, request, f"3: {what} against the request alone")
    (veto,) = launch(args.module, "veto", "cuda/off/on/comparable")
    expect(veto, "3 request with reduced_pcg=False", tier=False, reason="off", gpu=True)
    general0 = launch(args.module, "general0", "cuda/-/on/comparable,cuda/on/on/comparable",
                      env_extra={"XHDFE_REDUCED_PCG": "0"})
    expect(general0[0], "3 XHDFE_REDUCED_PCG=0, request alone", tier=False, reason="off", gpu=True)
    expect(general0[1], "3 XHDFE_REDUCED_PCG=0, reduced_pcg=True and the request", tier=True,
           reason="none", gpu=True)
    (fast,) = launch(args.module, "fast", "cuda/-/on/fast")
    expect(fast, "3 request in xhdfe-fast", tier=False, reason="tolerance_mode", gpu=True)

    # 4: a profile or a cache keeps the request out and says which (A1).
    for cache, (mode_var, path_var) in CACHE_ENV.items():
        legs = {}
        for leg, cuda_request in (("request", "on"), ("plain", "-")):
            label = f"{cache}_{leg}"
            (legs[leg],) = launch(args.module, label, f"cuda/-/{cuda_request}/comparable",
                                  env_extra={mode_var: "write", path_var: str(out / f"{label}.file")})
        expect(legs["request"], f"4 request with {cache}", tier=False,
               reason=CACHE_REASON[cache], gpu=True)
        expect(legs["plain"], f"4 no request with {cache}", tier=False, reason="gpu", gpu=True)
        if cache == "mobility_profile":
            # Writing a profile benchmarks the absorption methods and keeps the fastest
            # (elapsed time), so two processes may legitimately publish different routes.
            print("NOTE 4: mobility_profile legs compared on their diagnostics only", flush=True)
        else:
            within(legs["request"], legs["plain"], f"4: {cache}, request against no request")
    # The implicit profile: the file in the working directory, and, as in a session that sets
    # nothing, no profile or cache variable in the environment.
    profile_dir, clean_dir = out / "implicit_profile", out / "no_profile"
    profile_dir.mkdir()
    clean_dir.mkdir()
    (profile_dir / PROFILE_NAME).write_text(
        "# synthetic placeholder: an implicit mobility profile (tests/reduced_cuda_request_contract.py)\n")
    implicit = launch(args.module, "implicit",
                      f"cuda/-/on/comparable,cuda/-/-/comparable,chdir:{clean_dir},cuda/-/on/comparable",
                      env_extra={"XHDFE_MOBILITY_MODE": None, "XHDFE_ABSORPTION_CACHE_MODE": None,
                                 "XHDFE_FE_STRUCTURE_MODE": None}, cwd=str(profile_dir))
    expect(implicit[0], "4 request with the implicit profile", tier=False, reason="mobility_profile",
           gpu=True)
    expect(implicit[1], "4 no request with the implicit profile", tier=False, reason="gpu", gpu=True)
    expect(implicit[2], "4 request, working directory without a profile", tier=True, reason="none",
           gpu=True)
    within(implicit[2], request, "4: the request without the profile against the request alone")
    written = sorted(p.name for p in list(profile_dir.iterdir()) + list(clean_dir.iterdir()))
    if written != [PROFILE_NAME]:
        fail(f"4: the implicit-profile fits wrote files {written}")

    # 5: the argument takes None, True or False only; earlier results untouched.
    bad = launch(args.module, "badarg", kind="badarg")
    expected_bad = {"'on'": "TypeError", "1": "TypeError", "0": "TypeError", "'False'": "TypeError",
                    "None": "accepted", "True": "accepted", "False": "accepted", "untouched": []}
    if bad != expected_bad:
        fail(f"5: reduced_pcg_cuda values {bad}")

    # 6: ONE regressor object with the request, CPU and CUDA alternating (os.environ).
    reuse = launch(args.module, "reuse", "cpu/-/on/comparable,cuda/-/on/comparable,"
                   "cpu/-/on/comparable,cuda/-/on/comparable", kind="reuse")
    for r, gpu, what in zip(reuse, (False, True, False, True), ("CPU", "CUDA", "CPU again", "CUDA again")):
        expect(r, f"6 one regressor, {what}", tier=True, reason="none", gpu=gpu)
    same_bits(reuse[2], reuse[0], "6: one regressor, the two CPU fits")
    same_bits(reuse[0], fresh_cpu, "6: one regressor, its CPU fit against a fresh CPU fit")
    within(reuse[3], reuse[1], "6: one regressor, the two CUDA fits")
    within(reuse[1], request, "6: one regressor, its CUDA fit against the request alone")

    # 7: saved effects, the projection of E1 on CUDA with the request (A2).
    diag = {"XHDFE_AUTO_ROUTING_RETRY_DIAG": "1"}
    twin, savefe = launch(args.module, "savefe", "cuda/-/on/comparable,cuda/-/on/comparable/savefe",
                          env_extra=diag, design_kind="lowmob")
    print(f"7 twin without saved effects: {summary(twin)}; final {twin['finals']}", flush=True)
    print(f"7 with saved effects: {summary(savefe)}; final {savefe['finals']}; projection "
          f"{savefe['projection']}; saved effects {savefe['saved_effects']}", flush=True)
    rows.append({"what": "7 saved effects, what a user sees", "result": summary(savefe),
                 "finals": savefe["finals"], "projection": savefe["projection"],
                 "saved_effects": savefe["saved_effects"]})
    if savefe["error"] or not (savefe["converged"] and savefe["precision_certified"]):
        fail(f"7: the fit with saved effects: {summary(savefe)}")
    if not any("published=1" in l for l in savefe["projection"]):
        fail(f"7: the projection was not published: {savefe['projection']}")
    if not entered_cuda_tier(savefe):
        fail(f"7: the projection did not reach the CUDA tier: {savefe['trace']}")
    se = savefe["saved_effects"]
    if se.get("shape", [0])[0] != len(np.load(savefe["npz"])["residuals_"]) or not se.get("finite"):
        fail(f"7: saved effects {se}")
    within(savefe, twin, "7: saved effects against the twin without them")
    fatal = launch(args.module, "savefe_fatal",
                   f"cuda/-/on/comparable/savefe,env:{FATAL}=1,cuda/-/on/comparable/savefe,"
                   f"env:{FATAL}=0,cuda/-/on/comparable/savefe", kind="reuse", design_kind="lowmob")
    first, failed, again = fatal
    print(f"7 fatal injection: {summary(failed)}; lifecycle {failed['lifecycle']}", flush=True)
    if first["error"] or not entered_cuda_tier(first):
        fail(f"7: the first fit of the fatal case: {summary(first)}")
    if not (failed["error"] and "injected fatal error" in failed["error"]):
        fail(f"7: the injected fatal error did not leave the fit: {summary(failed)}")
    if failed["converged"] or failed["precision_certified"] or failed["lifecycle"] != "failed":
        fail(f"7: after the fatal error: converged {failed['converged']} certified "
             f"{failed['precision_certified']} lifecycle {failed['lifecycle']}")
    failed_arrays = np.load(failed["npz"])
    if any(failed_arrays[k].size for k in ("coef_", "covariance_", "residuals_")):
        fail("7: estimates left after the fatal error")
    if any(first["public"].get(k) != again["public"].get(k) for k in ARRAYS):
        fail("7: the fit after the fatal error does not repeat the bytes of the first fit")
    rows.append({"what": "7 fatal injection", "failed": summary(failed), "again": summary(again)})

    # 8: AKM and Gelbach after a fit that carried the request (A5).
    internal = launch(args.module, "internal", kind="internal")
    expect(dict(internal, label="internal_0"), "8 the fit with the request between them",
           tier=True, reason="none", gpu=True)
    before, after = np.load(out / "internal_before.npz"), np.load(out / "internal_after.npz")
    changed = sorted(k for k in before.files
                     if before[k].tobytes() != after[k].tobytes() or before[k].shape != after[k].shape)
    if changed:
        fail(f"8: AKM or Gelbach changed after the request: {changed}")
    rows.append({"what": "8 AKM and Gelbach before and after", "changed": changed})

    # 9: the CPU module, and the frozen CPU module.
    if args.cpu_module:
        c_default, c_off, c_request = launch(
            args.cpu_module, "cpumod", "cpu/-/-/comparable,cpu/off/-/comparable,cpu/-/on/comparable")
        expect(c_default, "9 CPU module, default", tier=True, reason="none", gpu=False)
        expect(c_off, "9 CPU module, reduced_pcg=False", tier=False, reason="off", gpu=False)
        expect(c_request, "9 CPU module, reduced_pcg_cuda=True", tier=True, reason="none", gpu=False)
        same_bits(c_request, c_default, "9: CPU module, the request is inert")
        if args.frozen_cpu:
            f_default, f_off = launch(args.frozen_cpu, "frozen_cpu",
                                      "cpu/-/-/comparable,cpu/off/-/comparable")
            same_bits(c_default, f_default, "9: CPU module default against the frozen CPU module")
            same_bits(c_off, f_off, "9: CPU module reduced_pcg=False against the frozen CPU module")
    else:
        print("SKIP 9: no --cpu-module", flush=True)

    # 10: the frozen CUDA module of 90f8aeda8.
    if args.frozen_cuda:
        f_cpu, f_cpu_off = launch(args.frozen_cuda, "frozen_cuda_cpu",
                                  "cpu/-/-/comparable,cpu/off/-/comparable")
        same_bits(fresh_cpu, f_cpu, "10: CPU tier fit against the frozen CUDA module")
        same_bits(fresh_cpu_off, f_cpu_off, "10: CPU fit with reduced_pcg=False against the frozen CUDA module")
        (f_plain,) = launch(args.frozen_cuda, "frozen_cuda_plain", "cuda/-/-/comparable")
        expect(f_plain, "10 frozen CUDA module, no environment", tier=False, reason="gpu", gpu=True)
        within(cuda_plain, f_plain, "10: CUDA without the request against the frozen CUDA module")
        (f_env1,) = launch(args.frozen_cuda, "frozen_cuda_env1", "cuda/-/-/comparable",
                           env_extra={"XHDFE_REDUCED_PCG_CUDA": "1"})
        expect(f_env1, "10 frozen CUDA module, XHDFE_REDUCED_PCG_CUDA=1", tier=True, reason="none", gpu=True)
        # Different operators since package 2: no line-for-line trace; both entered the tier
        # (expect() above and in 3 asserted method 8, status 1 and one operator line with
        # products), the iterations, tau, chat and final residual of both are reported.
        for r, who in ((request, "new"), (f_env1, "frozen")):
            if not entered_cuda_tier(r):
                fail(f"10: the {who} module's request did not reach the CUDA operator: {r['trace']}")
        new_rec, frozen_rec = tier_record(request), tier_record(f_env1)
        print(f"10 new module, the request: iterations {new_rec['iterations']} rel {new_rec['rel']} "
              f"columns {new_rec['columns']}", flush=True)
        print(f"10 frozen module, XHDFE_REDUCED_PCG_CUDA=1: iterations {frozen_rec['iterations']} "
              f"rel {frozen_rec['rel']} columns {frozen_rec['columns']}", flush=True)
        rows.append({"what": "10 the request, new against frozen (reported, not compared line for line)",
                     "new": new_rec, "frozen": frozen_rec})
        within(request, f_env1, "10: the request against the frozen module with XHDFE_REDUCED_PCG_CUDA=1")
        f_fatal = launch(args.frozen_cuda, "frozen_savefe_fatal",
                         f"env:{FATAL}=1,cuda/-/-/comparable/savefe", kind="reuse",
                         design_kind="lowmob", env_extra={"XHDFE_REDUCED_PCG_CUDA": "1"})
        print(f"NOTE 10: the frozen module, saved effects with XHDFE_REDUCED_PCG_CUDA=1 and the "
              f"injected fatal error: {summary(f_fatal[0])}", flush=True)
        rows.append({"what": "10 NOTE frozen module, saved effects, injected fatal error",
                     "result": summary(f_fatal[0])})
    else:
        print("SKIP 10: no --frozen-cuda", flush=True)

    # 11: failures injected into the CUDA operator, then the same process again (C4).
    steps = []
    for point, _ in INJECTED:
        steps += [f"env:{FAIL}={point}", "cuda/-/on/comparable"]
    steps += [f"env:{FAIL}=", "cuda/-/on/comparable"]
    injected = launch(args.module, "inject", ",".join(steps))
    for r, (point, reason) in zip(injected, INJECTED):
        expect_declined(r, f"11 injected {point}", reason)
        within(r, cuda_plain, f"11: injected {point}, the fit against the CUDA route without the request")
    expect(injected[-1], "11 the same process after the injections", tier=True, reason="none", gpu=True)
    within(injected[-1], request, "11: the same process after the injections against the request alone")

    print(f"request: {summary(cuda_tier)}; final {cuda_tier['finals']}", flush=True)
    print(f"no request: {summary(cuda_plain)}; final {cuda_plain['finals']}", flush=True)
    status = "PASS" if not failures else f"{len(failures)} FAILURES"
    (out / "receipt.json").write_text(json.dumps(
        {"status": status, "failures": failures, "rows": rows,
         "module": str(Path(args.module).resolve()),
         "module_sha256": hashlib.sha256(Path(args.module).read_bytes()).hexdigest()},
        indent=2, default=str) + "\n")
    print(status, flush=True)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--module", required=True)
    parser.add_argument("--output")
    parser.add_argument("--cpu-module")
    parser.add_argument("--frozen-cuda")
    parser.add_argument("--frozen-cpu")
    parser.add_argument("--kind", default="sequence")
    parser.add_argument("--steps", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--design", default="runtime", choices=("runtime", "lowmob"))
    args = parser.parse_args()
    if args.child:
        return child(args)
    if not args.output:
        parser.error("--output (a new directory) is required")
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
