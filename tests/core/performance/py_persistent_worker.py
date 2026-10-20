#!/usr/bin/env python3
"""C++/Python worker for the PR15/PR17 integration matrix (19oct2026).

This campaign always uses --no-warmup and one FIT per fresh process.
The max_iter budget is 100000, as in the canonical core24 worker.
Historical derivation follows:

Derived from ``Verifications/perf_max_20260929/perf_worker.py`` (same data
loading, same estimator options, same receipt fields), with three changes
required by the C2 protocol:

* the process stays alive: it loads the data once, runs one warm-up fit that
  is discarded, and then runs one timed fit per ``FIT <receipt>`` command read
  from stdin, so two workers (candidate and reference) can be interleaved;
* ``tol=1e-8`` and ``max_iter=100000`` (C2 protocol);
* memory is reported from read-only process counters; VmHWM is the process
  high-water mark, not a reset per-fit peak.

Nothing individual-level is written: only estimates, scalars and SHA-256
digests of the sample and residual vectors.  The data are read in place.

Protocol (one line each):  stdout ``READY <json>`` after the warm-up;
stdin ``FIT <path>`` -> stdout ``DONE <path>``;  stdin ``EXIT``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import resource
import sys
import time
from pathlib import Path

REGISTRY = None  # The caller supplies its private, read-only dataset registry.
GPU_STATUS = {0: "not_requested", 1: "used", 2: "backend_unavailable",
              3: "gpu_absorption_not_converged", 4: "gpu_backend_failed",
              5: "cpu_cache_or_profile_result"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_spec(name: str, registry_path=REGISTRY) -> dict:
    registry = json.loads(Path(registry_path).read_text())
    for spec in registry['specs']:
        if spec['name'] == name:
            return dict(spec)
    raise SystemExit(f"dataset {name!r} is not in {REGISTRY}")


def as_list(value):
    if value is None or value == "":
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def get(reg, name, default=None):
    try:
        value = getattr(reg, name)
    except Exception:
        return default
    return default if value is None else value


def proc_status(*keys):
    out = {}
    with open("/proc/self/status") as handle:
        for line in handle:
            name = line.split(":", 1)[0]
            if name in keys:
                out[name] = int(line.split()[1])  # kB
    return out


def openmp_runtimes() -> list[str]:
    found = set()
    with open("/proc/self/maps") as handle:
        for line in handle:
            path = line.rsplit(" ", 1)[-1].strip()
            base = os.path.basename(path)
            if base.startswith(("libgomp", "libnvomp", "libiomp", "libomp")):
                found.add(path)
    return sorted(found)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--module-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", choices=["fast", "comparable"], required=True)
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--retain-fes", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--method", default="auto")
    ap.add_argument("--weights", choices=["none", "aw", "fw"], default="none")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    for env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[env] = str(args.threads)
    os.environ["XHDFE_GPU_BACKEND"] = "cuda" if args.backend == "cuda" else "cpu"

    import numpy as np
    import pandas as pd

    spec = load_spec(args.dataset, args.registry)
    module_dir = Path(args.module_dir).resolve()
    sys.path.insert(0, str(module_dir))
    py = importlib.import_module("py_hdfe_v11")
    module_path = Path(py.__file__).resolve()
    if module_path.parent != module_dir:
        raise SystemExit(f"module custody failure: loaded {module_path}")
    module_sha = sha256(module_path)

    def fac(series):
        codes, _ = pd.factorize(series, sort=True)
        return codes.astype(np.int32, copy=False)

    cluster_vars = as_list(spec.get("cluster"))
    slopes = spec.get("slopes", []) or []
    group_var = spec.get("group")
    individual_var = spec.get("individual")
    columns = list(dict.fromkeys(
        [spec["y"], *spec["x"], *cluster_vars, *spec["fe"],
         *[s["fe"] for s in slopes], *[s["var"] for s in slopes],
         *([group_var] if group_var else []),
         *([individual_var] if individual_var else [])]))

    data_path = Path(spec["path"])
    data_stat = data_path.stat()
    t_load = time.perf_counter()
    df = pd.read_parquet(spec["path"], columns=columns)
    parquet_read_seconds = time.perf_counter() - t_load
    y = df[spec["y"]].to_numpy(np.float64)
    X = np.asfortranarray(df[spec["x"]].to_numpy(np.float64))
    fes = [fac(df[c]) for c in spec["fe"]]
    slope_arg = []
    for s in slopes:
        fes.append(fac(df[s["fe"]]))
        slope_arg.append(dict(fe_index=len(fes) - 1,
                              values=df[s["var"]].to_numpy(np.float64),
                              include_intercept=bool(s.get("intercept", False))))
    if len(cluster_vars) == 1:
        clusters = fac(df[cluster_vars[0]])
    elif cluster_vars:
        clusters = [fac(df[c]) for c in cluster_vars]
    else:
        clusters = None
    group_arg = fac(df[group_var]) if group_var else None
    individual_arg = fac(df[individual_var]) if individual_var else None
    weights = None
    if args.weights != "none":
        rng = np.random.default_rng(20260929)
        weight_count = int(group_arg.max()) + 1 if group_var else len(df)
        weights = (rng.integers(1, 4, size=weight_count).astype(np.float64)
                   if args.weights == "fw"
                   else rng.uniform(0.5, 1.5, size=weight_count))
        if group_var:
            weights = weights[group_arg]
    del df
    load_seconds = time.perf_counter() - t_load

    def array_sha(value):
        arr = np.asarray(value)
        digest = hashlib.sha256()
        digest.update(str((arr.dtype.str, arr.shape)).encode())
        raw = memoryview(np.ascontiguousarray(arr).ravel(order="K")).cast("B")
        for start in range(0, len(raw), 1 << 20):
            digest.update(raw[start:start + (1 << 20)])
        return digest.hexdigest()

    input_parts = [array_sha(y), array_sha(X), *[array_sha(f) for f in fes]]
    for value in ([clusters] if isinstance(clusters, np.ndarray) else clusters or []):
        input_parts.append(array_sha(value))
    for value in (weights, group_arg, individual_arg):
        if value is not None:
            input_parts.append(array_sha(value))
    input_parts.extend(array_sha(s["values"]) for s in slope_arg)
    input_sha256 = hashlib.sha256("|".join(input_parts).encode()).hexdigest()

    tolerance_mode = "xhdfe-fast" if args.mode == "fast" else "reghdfe-comparable"
    print("LOADED " + json.dumps(dict(dataset=args.dataset, load_seconds=load_seconds,
        module=str(module_path), module_sha256=module_sha, input_sha256=input_sha256)), flush=True)

    def one_fit():
        reg = py.HdfeRegressor(
            se_type="cluster" if cluster_vars else "unadjusted",
            tol=1e-8, max_iter=100000, num_threads=args.threads,
            symmetric_sweep=False, absorption_method=args.method,
            tolerance_mode=tolerance_mode, jacobi_relaxation=0.0,
            retain_fes=bool(args.retain_fes),
            drop_singletons=bool(spec.get("singleton_drop", True)))
        fit_args = dict(fes=fes, clusters=clusters)
        if weights is not None:
            fit_args.update(weights=weights, fweights=args.weights == "fw")
        if group_var:
            fit_args.update(group=group_arg, individual=individual_arg,
                            aggregation=spec.get("aggregation", "mean"))
        if slope_arg:
            fit_args["slopes"] = slope_arg
        before = proc_status("VmRSS")
        load_before = os.getloadavg()
        error = ""
        t0 = time.perf_counter()
        try:
            reg.fit(y, X, **fit_args)
        except Exception as exc:  # the refusal text is the evidence
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0
        after = proc_status("VmHWM", "VmRSS")
        return reg, error, elapsed, dict(rss_before_kib=before.get("VmRSS"),
                                         process_hwm_after_kib=after.get("VmHWM"),
                                         rss_after_kib=after.get("VmRSS"),
                                         loadavg_before=load_before,
                                         loadavg_after=os.getloadavg())

    warmup = None
    if not args.no_warmup:
        warm_reg, werr, welapsed, wmem = one_fit()
        warmup = dict(elapsed_seconds=welapsed, error=werr, **wmem)
        del warm_reg
    print("READY " + json.dumps(dict(label=args.label, dataset=args.dataset,
                                     load_seconds=load_seconds, warmup=warmup,
                                     module=str(module_path), module_sha256=module_sha)),
          flush=True)

    for line in sys.stdin:
        command = line.strip().split(" ", 1)
        if not command or command[0] == "EXIT":
            break
        if command[0] != "FIT" or len(command) != 2:
            print("ERROR bad command", flush=True)
            continue
        out_path = Path(command[1])
        reg, error, elapsed, memory = one_fit()
        receipt = {
            "interface": "native",
            "label": args.label,
            "dataset": args.dataset,
            "mode": args.mode,
            "backend": args.backend,
            "threads": args.threads,
            "tol": 1e-8,
            "max_iter": 100000,
            "retain_fes": bool(args.retain_fes),
            "weights": args.weights,
            "method_requested": args.method,
            "module": str(module_path),
            "module_sha256": module_sha,
            "load_seconds": load_seconds,
            "parquet_read_seconds": parquet_read_seconds,
            "preparation_seconds": load_seconds - parquet_read_seconds,
            "warmup": warmup,
            "env_xhdfe": {k: v for k, v in os.environ.items() if k.startswith("XHDFE_")},
            "elapsed_seconds": elapsed,
            "input_sha256": input_sha256,
            "source_path": spec["path"],
            "source_size": data_stat.st_size,
            "source_mtime_ns": data_stat.st_mtime_ns,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "openmp_runtimes": openmp_runtimes(),
            "error": error,
            "success": int(not error),
            **memory,
        }
        if sha256(module_path) != module_sha:
            receipt.update(error="module custody changed during execution", success=0)
        if receipt["success"]:
            coef = np.asarray(reg.coef_, np.float64)
            se = np.asarray(reg.stderr_, np.float64)
            cov = np.asarray(reg.covariance_, np.float64)
            status = int(get(reg, "gpu_status_code_", 0))
            sample = get(reg, "sample_index_")
            residuals = get(reg, "residuals_")
            receipt.update({
                "sample_sha256": array_sha(sample) if sample is not None else None,
                "residual_sha256": array_sha(residuals) if residuals is not None else None,
                "parallel_workers_active": int(get(reg, "parallel_workers_active_", -1)),
                "terms": list(spec["x"]) + (["_cons"] if len(coef)==len(spec["x"])+1 else []),
                "coef": coef.tolist(),
                "coef_hex": [float(v).hex() for v in coef],
                "stderr": se.tolist(),
                "stderr_hex": [float(v).hex() for v in se],
                "covariance": cov.tolist(),
                "covariance_hex": [[float(v).hex() for v in row] for row in cov],
                "nobs": int(reg.nobs_),
                "nobs_full": int(get(reg, "nobs_full_", reg.nobs_)),
                "num_singletons": int(get(reg, "num_singletons_", 0)),
                "rss": float(get(reg, "rss_", float("nan"))),
                "rss_hex": float(get(reg, "rss_", float("nan"))).hex(),
                "df_resid": float(get(reg, "df_resid_", float("nan"))),
                "df_a": float(get(reg, "df_a_", float("nan"))),
                "df_m": float(get(reg, "df_m_", float("nan"))),
                "converged": int(bool(get(reg, "converged_", False))),
                "iterations": int(get(reg, "num_iterations_", -1)),
                "method": str(get(reg, "absorption_method_used", "")).rsplit(".", 1)[-1],
                "precision_certified": int(bool(get(reg, "precision_certified_", False))),
                "abs_residual_rel": float(get(reg, "abs_residual_rel_", float("nan"))),
                "threads_requested": int(get(reg, "threads_requested_", -1)),
                "threads_effective": int(get(reg, "threads_effective_", -1)),
                "threads_used": int(get(reg, "threads_used_", -1)),
                "gpu_attempted": int(bool(get(reg, "gpu_attempted_", False))),
                "gpu_used": int(bool(get(reg, "gpu_used_", False))),
                "gpu_status": GPU_STATUS.get(status, f"unknown_{status}"),
                "gpu_absorption_iterations": int(get(reg, "gpu_absorption_iterations_", 0)),
            })
            if args.retain_fes:
                fe_digest = hashlib.sha256()
                fe_stats = []
                for attr in ("fe_effects_", "fixed_effects_", "fes_"):
                    effects = get(reg, attr)
                    if effects is None:
                        continue
                    for block in effects:
                        arr = np.ascontiguousarray(np.asarray(block, np.float64))
                        fe_digest.update(arr.tobytes())
                        fe_stats.append({"size": int(arr.size)})
                    receipt["fe_attribute"] = attr
                    break
                receipt["fe_sha256"] = fe_digest.hexdigest()
                receipt["fe_blocks"] = fe_stats
                receipt["fe_recovery_converged"] = int(bool(get(reg, "fe_recovery_converged_", False)))
                if not fe_stats or not receipt["fe_recovery_converged"]:
                    receipt.update(error="retained FE evidence absent or unconverged", success=0)
        del reg
        state = data_path.stat()
        if state.st_size != data_stat.st_size or state.st_mtime_ns != data_stat.st_mtime_ns:
            receipt.update(error="source file changed during the cell", success=0)
        with out_path.open("x") as handle:
            json.dump(receipt, handle, indent=1)
        print(f"DONE {out_path} {elapsed:.3f} {receipt['success']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
