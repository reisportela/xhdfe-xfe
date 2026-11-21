#!/usr/bin/env python3
"""Phase 1 of the Stata-path plan: sparse identifiers ranked on the team
(src/sparse_rank.hpp) give every output of today's serial branches.

Builds nothing; never run during a timing campaign. Synthetic data only, made in
each child process from fixed seeds; nothing is read from disk but the modules.

    python3 -I tests/sparse_rank_contract.py --module <this branch's module> \
        --frozen-module <dea5fa00 module, packages/cand_c232b_py_cpu> [--only NAME] [--large]

Every scenario runs its fits in four fresh processes:
  new    the branch's module, XHDFE_SPARSE_RANK unset (ranking on)
  off    the branch's module, XHDFE_SPARSE_RANK=0 (today's serial branches)
  base   the frozen module of dea5fa00
  verify the branch's module with XHDFE_SPARSE_RANK_VERIFY=1: every site that took
         the index recomputes today's serial result and compares it field by field
         (a "label=sparse_rank_verify ... ok=0" line, or an error, is a failure;
         some "ok=1" line is required wherever the index ran)
all with XHDFE_PROFILE_CPU=1 (it only adds lines on the CPU routes used here),
XHDFE_GPU_BACKEND=cpu and the caches off unless a scenario sets them. new is
compared with base and with off on the REQUIRED properties (a missing one is a
failure) by dtype, shape and raw bytes, on the tier's diagnostics, on the method,
on the error raised if any, and on the Gelbach dictionaries. Old and new agree by
construction, so every scenario also asserts the positive control: the
"cpu_profile label=sparse_rank" lines that prove the new code ran (path=bitmap in
new, path=serial reason=off in off), and for the marking scenario the component
counter's own line (site=components path=marking with the lookup modes).

Scenarios (design note section 5 and review round 17):
  firm        a firstreg-like panel with a raw sparse firm id (range 8.7 n):
              comparable with the tier on and off, fast, cluster on a sparse
              grouping of the firm (nested test, both outcomes), fweights,
              savefe with the component normalisation, groupvar with
              firstpair and all, heterogeneous slopes, group()/individual()
              with sum and mean, and a Gelbach decomposition (partial_out)
  precoded    the firm panel with the firm id pre-coded 0..G-1 against the raw
              id, in new: b, V, residuals, df, levels, iterations and method
              by bytes, groupvar as a partition
  counter     the reviewer's counterexample: 8 rows (today's branch, the guard),
              32 rows with an explicit team of 2 and 2^20 x 8 rows with spread
              values (both through the new code), firstpair and all
  marking     2^20 + 7 rows: a Rows lookup paired with Dense, DenseMap, Rows and
              a Sparse fallback (range above the guard), labels and counts, and
              weighted first-pair statistics through Gelbach
  decline     the tier attempted then declined at the certificate stage
              (XHDFE_TEST_REDUCED_PCG_DECLINE=certificate) with a sparse id
  dof         singletons whose removal changes the first appearances, a
              non-leading effect removed by coarsening, both outcomes of the
              refinement test with the conflict on the last row, and a nested
              test whose conflict is on the last row; firstpair, firstpair
              clusters and all (the nested test runs only with the clusters
              adjustment, so plain firstpair must leave it out in both variants)
  negative    raw int32 ids with -1 and with INT_MIN: the same error as base
  cache       absorption and FE-structure caches across versions: each module
              writes, and each module reads each file; the two modules reading
              the same file agree byte for byte, as does one module reading
              either file; an absorption-cache read hits and gives the writer's
              bytes; an FE-structure read hands run_fit the stored
              first-appearance codes instead of the raw ids, so it may differ
              from the fresh fit in rounding in both modules alike (reported)
  large       (--large) a compact id with 60 M levels (above the 50 M cap)
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

REQUIRED = (
    "absorption_method_used", "abs_residual_", "abs_residual_rel_", "any_omitted_",
    "auto_routing_retry_abs_residual_rel_", "auto_routing_retry_eligible_",
    "auto_routing_retry_fired_", "auto_routing_retry_iterations_",
    "auto_routing_retry_policy_enabled_", "auto_routing_retry_primary_abs_residual_rel_",
    "auto_routing_retry_primary_iterations_", "auto_routing_retry_primary_method_",
    "auto_routing_retry_status_", "cluster_combo_counts_", "cluster_counts_", "cluster_scale_",
    "coef_", "conf_int_", "converged_", "covariance_", "cuda_tail_", "cuda_tail_code_",
    "df_a_", "df_a_exact_", "df_a_levels_", "df_a_nested_", "df_m_", "df_resid_",
    "df_resid_unadj_", "fe_base_levels_", "fe_base_redundant_", "fe_effects_", "fe_inexact_",
    "fe_num_coefs_", "fe_num_levels_", "fe_recovery_converged_", "fe_recovery_iterations_",
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
DIAGNOSTICS = ("reduced_pcg_status_", "reduced_pcg_reason_", "reduced_pcg_discarded_iterations_")
PRECODED_SAME = ("coef_", "covariance_", "residuals_", "rss_", "df_a_", "df_a_exact_",
                 "df_resid_", "fe_num_levels_", "fe_redundant_", "num_iterations_",
                 "absorption_method_used", "sample_index_", "nobs_", "num_singletons_")
INT32_MIN = -(2 ** 31)


def load(path: str):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(value):
    """dtype, shape and the SHA-256 of the raw bytes; strings as themselves; lists
    of arrays element by element."""
    if isinstance(value, str):
        return {"str": value}
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], np.ndarray):
        return {"list": [describe(v) for v in value]}
    if not isinstance(value, (bool, int, float, list, tuple, np.ndarray, np.generic)) and \
            hasattr(value, "__int__"):
        value = int(value)  # a pybind11 enum such as AbsorptionMethod
    array = np.asarray(value)
    if array.dtype == object:
        return {"repr": repr(value)}
    raw = np.ascontiguousarray(array).tobytes()
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(raw).hexdigest()}


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


# --- Synthetic data -------------------------------------------------------------

def distinct_values(count, span, rng):
    """`count` distinct sorted integers in [0, span), both ends included, without
    materialising the population."""
    if count <= 2:
        return np.array([0, span - 1][:count], dtype=np.int64)
    values = np.unique(np.concatenate([[0, span - 1], rng.integers(0, span, size=count + 16)]))
    while values.size < count:
        values = np.unique(np.concatenate([values, rng.integers(0, span, size=count)]))
    keep = np.sort(rng.choice(np.arange(1, values.size - 1), size=count - 2, replace=False))
    return np.concatenate([[values[0]], values[keep], [values[-1]]])


def spread_codes(codes, ratio, rng, low=0):
    """Distinct raw values spread over about ratio * n, assigned to the codes in a
    random order (so first appearance and value order differ)."""
    codes = np.asarray(codes)
    levels = int(codes.max()) + 1
    span = max(levels, int(ratio * codes.size))
    values = distinct_values(levels, span, rng) + low
    return values[rng.permutation(levels)][codes].astype(np.int32)


def firm_panel(seed=20261114, workers=30_000, firms=2_500, years=10):
    rng = np.random.default_rng(seed)
    rows = rng.integers(3, 9, size=workers)
    worker = np.repeat(np.arange(workers), rows)
    n = worker.size
    year = (rng.integers(0, years - 8, size=workers)[worker] +
            np.concatenate([np.arange(r) for r in rows])) % years
    home = rng.integers(0, firms, size=workers)
    move = rng.random(n) < 0.15
    firm = np.where(move, rng.integers(0, firms, size=n), home[worker])
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n) + 0.1 * (firm % 7)])
    y = X @ np.array([0.5, -0.25]) + 0.3 * rng.normal(size=n)
    for f in (worker, firm, year):
        y = y + rng.normal(size=int(f.max()) + 1)[f]
    firm_raw = spread_codes(firm, 8.7, rng)
    return {
        "y": y, "X": np.asfortranarray(X), "n": n,
        "worker": worker.astype(np.int32), "firm_code": firm.astype(np.int32),
        "firm_raw": firm_raw, "year": year.astype(np.int32),
        # A sparse grouping of the firm: the firm is nested in it.
        "firm_group_raw": spread_codes(firm // 5, 9.0, rng),
        "fw": rng.integers(1, 4, size=n).astype(np.float64),
        "z": rng.normal(size=n),
    }


def ctor(module, **kw):
    base = dict(se_type="cluster", tol=1e-8, max_iter=100000, fit_intercept=True,
                num_threads=4, drop_singletons=True, symmetric_sweep=False,
                absorption_method="auto", tolerance_mode="reghdfe-comparable")
    base.update(kw)
    return module.HdfeRegressor(**base)


def run_fit(module, out: Path, label: str, ctor_kw, fit_args, fit_kw):
    log = out / f"{label}.stderr"
    record = {"error": None}
    try:
        reg = ctor(module, **ctor_kw)
        with captured_stderr(log):
            reg.fit(*fit_args, **fit_kw)
        public, missing = {}, []
        for name in REQUIRED:
            try:
                public[name] = describe(getattr(reg, name))
            except AttributeError:
                missing.append(name)
            except Exception as exc:  # a deterministic refusal is part of the result
                public[name] = {"raises": type(exc).__name__}
        record.update(public=public, missing=missing,
                      diagnostics={k: getattr(reg, k) for k in DIAGNOSTICS if hasattr(reg, k)},
                      method=int(reg.absorption_method_used))
        try:
            record["groupvar"] = np.asarray(reg.groupvar_).astype(np.int64).tolist()
        except Exception:
            record["groupvar"] = None
        # A few small outputs as values, for the fixture's own assertions and the
        # cache scenario's report.
        record["values"] = {}
        for name in ("df_a_", "df_a_nested_", "fe_redundant_", "fe_num_levels_",
                     "num_singletons_", "coef_", "stderr_"):
            try:
                record["values"][name] = np.asarray(getattr(reg, name)).tolist()
            except Exception:
                record["values"][name] = None
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["lines"] = [l for l in (log.read_text().splitlines() if log.exists() else [])
                       if "label=sparse_rank" in l or "reduced_pcg event=final" in l
                       or "cache_hit=" in l]
    (out / f"{label}.json").write_text(json.dumps(record, sort_keys=True))


def run_gelbach(module, out: Path, label: str, y, x1, x2, fes, weights):
    log = out / f"{label}.stderr"
    record = {"error": None}
    try:
        with captured_stderr(log):
            res = module.gelbach_decompose(y, x1, x2, [x2.shape[1]], fes, tol=1e-9,
                                           num_threads=4, weights=weights,
                                           connectivity_fe_pair=[0, 1])
        record["gelbach"] = {k: describe(v) for k, v in sorted(res.items())}
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["lines"] = [l for l in (log.read_text().splitlines() if log.exists() else [])
                       if "label=sparse_rank" in l]
    (out / f"{label}.json").write_text(json.dumps(record, sort_keys=True))


# --- Scenarios (children) ---------------------------------------------------------

def scenario_firm(module, out):
    d = firm_panel()
    y, X = d["y"], d["X"]
    fes = [d["worker"], d["firm_raw"], d["year"]]
    common = (y, X)
    run_fit(module, out, "cmp_on", dict(reduced_pcg=True), common,
            dict(fes=fes, clusters=d["worker"]))
    run_fit(module, out, "cmp_off", dict(reduced_pcg=False), common,
            dict(fes=fes, clusters=d["worker"]))
    run_fit(module, out, "fast", dict(tolerance_mode="xhdfe-fast"), common,
            dict(fes=fes, clusters=d["worker"]))
    run_fit(module, out, "cluster_group", dict(), common,
            dict(fes=fes, clusters=d["firm_group_raw"]))
    run_fit(module, out, "fweights", dict(se_type="robust"), common,
            dict(fes=fes, weights=d["fw"], fweights=True))
    # savefe: the recovery and the component normalisation (the default style).
    run_fit(module, out, "savefe", dict(retain_fes=True, se_type="robust"), common, dict(fes=fes))
    run_fit(module, out, "groupvar_first", dict(groupvar=True, dofadjustments="firstpair",
                                                se_type="robust"),
            common, dict(fes=[d["firm_raw"], d["worker"], d["year"]]))
    run_fit(module, out, "groupvar_all", dict(groupvar=True, dofadjustments="all",
                                              se_type="robust"),
            common, dict(fes=[d["firm_raw"], d["worker"], d["year"]]))
    run_fit(module, out, "slopes", dict(se_type="robust"), common,
            dict(fes=fes, slopes=[(1, d["z"], True)]))
    # Rows sorted by the raw firm id: the indexers' contiguity flag and group_ptr
    # (build_indexer_sparse_rank) are only built for sorted identifiers.
    order = np.argsort(d["firm_raw"], kind="stable")
    sorted_fes = [np.ascontiguousarray(f[order]) for f in (d["worker"], d["firm_raw"], d["year"])]
    run_fit(module, out, "sorted_by_firm", dict(reduced_pcg=True),
            (y[order], np.asfortranarray(X[order])),
            dict(fes=[sorted_fes[1], sorted_fes[0], sorted_fes[2]], clusters=sorted_fes[0]))
    run_fit(module, out, "sorted_by_firm_off", dict(reduced_pcg=False),
            (y[order], np.asfortranarray(X[order])),
            dict(fes=[sorted_fes[1], sorted_fes[0], sorted_fes[2]], clusters=sorted_fes[0]))
    # group()/individual(): one row per (group, individual) edge, two to four
    # distinct individuals per group, a group-level effect and the individuals
    # both raw and sparse; y and X constant within the group.
    grng = np.random.default_rng(7)
    n_groups, pool = 40_000, 20_000
    sizes = grng.integers(2, 5, size=n_groups)
    group = np.repeat(np.arange(n_groups), sizes).astype(np.int64)
    individual = np.concatenate([grng.choice(pool, size=s, replace=False) for s in sizes])
    individual_raw = spread_codes(individual, 9.0, grng).astype(np.int64)
    group_fe_raw = spread_codes((np.arange(n_groups) % 700)[group], 9.0, grng)
    gy = grng.normal(size=n_groups)[group]
    gx = np.asfortranarray(grng.normal(size=(n_groups, 1))[group])
    for agg in ("mean", "sum"):
        run_fit(module, out, f"group_{agg}", dict(se_type="robust"), (gy, gx),
                dict(fes=[group_fe_raw, individual_raw.astype(np.int32)], group=group,
                     individual=individual_raw, aggregation=agg))
    rng = np.random.default_rng(9)
    run_gelbach(module, out, "gelbach", y, X[:, :1].copy(), np.asfortranarray(X[:, 1:]),
                [d["firm_raw"], d["worker"]], 0.5 + rng.random(d["n"]))


def scenario_precoded(module, out):
    d = firm_panel()
    for name, firm in (("raw", d["firm_raw"]), ("coded", None)):
        if firm is None:  # the ascending codes of the raw ids
            firm = np.unique(d["firm_raw"], return_inverse=True)[1].astype(np.int32)
        run_fit(module, out, f"precoded_{name}", dict(groupvar=True, dofadjustments="all",
                                                      se_type="cluster"),
                (d["y"], d["X"]), dict(fes=[firm, d["worker"], d["year"]],
                                       clusters=d["worker"]))


def counter_data(copies, spread):
    fe1 = np.array([1000] * 4 + [10] * 4, dtype=np.int64)
    fe2 = np.array([500] * 4 + [5] * 4, dtype=np.int64)
    fe1, fe2 = np.tile(fe1, copies), np.tile(fe2, copies)
    if spread:  # values apart, so that the range exceeds 2n and the index fits
        fe1 = np.where(fe1 == 1000, 10 * fe1.size, 10)
        fe2 = np.where(fe2 == 500, 9 * fe2.size, 5)
    rng = np.random.default_rng(11)
    x = rng.normal(size=(fe1.size, 1))
    y = 0.5 * x[:, 0] + rng.normal(size=fe1.size)
    return y, np.asfortranarray(x), [fe1.astype(np.int32), fe2.astype(np.int32)]


def scenario_counter(module, out):
    cases = (("c8", 1, False, 1), ("c32", 4, False, 2), ("cbig", 2 ** 20, True, 0))
    for name, copies, spread, threads in cases:
        y, X, fes = counter_data(copies, spread)
        for dof in ("firstpair", "all"):
            run_fit(module, out, f"{name}_{dof}",
                    dict(groupvar=True, dofadjustments=dof, se_type="robust", num_threads=threads),
                    (y, X), dict(fes=fes))


def marking_data(kind, seed=5):
    """2^20 + 7 rows (the marking needs 2^20), a base effect with 40 000 levels
    (raw, sparse: a Rows lookup) in ten blocks, and a partner whose levels stay
    within the block but are not a function of the base (noise), so that there
    are ten components, no coarsening, and the cells stay within 2^28."""
    rng = np.random.default_rng(seed)
    n = 2 ** 20 + 7
    a = rng.integers(0, 40_000, size=n)
    block = a // 4_000
    noise = rng.integers(0, 3, size=n)
    base = spread_codes(a, 9.0, rng)
    if kind == "dense":  # compact, every level present: Dense
        other = (block * 100 + (a + noise) % 100).astype(np.int32)
    elif kind == "densemap":  # compact range with holes: DenseMap
        other = (2 * (block * 100 + (a + noise) % 100)).astype(np.int32)
    elif kind == "rows":  # sparse, through the index: Rows (40 000 x 5 000 cells)
        other = spread_codes(block * 500 + (a // 8 + noise) % 500, 9.0, rng)
    else:  # sparse above the 8n guard: the Sparse fallback. 3 000 levels: with
        # 30 000 the absorption inside the Gelbach decomposition ran for more than
        # 25 minutes (harness fix, 14 Nov).
        other = spread_codes(block * 300 + (a + noise) % 300, 900.0, rng)
    x = rng.normal(size=(n, 2))
    y = x @ np.array([0.3, -0.2]) + rng.normal(size=n)
    w = 0.5 + rng.random(n)
    return y, np.asfortranarray(x), [base, other], w


def scenario_marking(module, out):
    for kind in ("dense", "densemap", "rows", "sparse"):
        y, X, fes, w = marking_data(kind)
        run_fit(module, out, f"mark_{kind}", dict(groupvar=True, dofadjustments="firstpair",
                                                  se_type="robust", num_threads=4),
                (y, X), dict(fes=fes))
        run_gelbach(module, out, f"mark_{kind}_gelbach", y, X[:, :1].copy(),
                    np.asfortranarray(X[:, 1:]), fes, w)


def scenario_decline(module, out):
    d = firm_panel(seed=4)
    run_fit(module, out, "decline", dict(reduced_pcg=True), (d["y"], d["X"]),
            dict(fes=[d["worker"], d["firm_raw"], d["year"]], clusters=d["worker"]))


def dof_data(seed=6):
    """A (leading, sparse), B (refined, sparse) and C (a function of B, so
    coarsened away, non-leading). Row 0 carries a singleton of A, whose removal
    moves the first appearance of B's and C's levels. A is a function of B except
    on the last row (the refinement test fails there); the cluster nests A
    except on the last row (the nested test fails there)."""
    rng = np.random.default_rng(seed)
    n = 6000
    b = rng.integers(0, 900, size=n)
    a = b // 3
    c = b // 9
    a[0] = 10_000  # singleton of A: dropped
    a[-1] = (a[-2] + 1) % 300 if b[-1] == b[-2] else (b[-1] // 3 + 1) % 300
    b_raw, a_raw, c_raw = spread_codes(b, 5.0, rng), spread_codes(a, 5.0, rng), \
        spread_codes(c, 5.0, rng)
    # Two cluster variables: B // 3 (B is nested in it, the test returns true
    # after a full pass) and the same with the last row moved (B and A conflict
    # on the last row).
    cluster_true = b // 3
    cluster_last = cluster_true.copy()
    cluster_last[-1] = 999_999
    x = rng.normal(size=(n, 1))
    y = 0.4 * x[:, 0] + rng.normal(size=n)
    return (y, np.asfortranarray(x), [a_raw, b_raw, c_raw],
            spread_codes(cluster_true, 5.0, rng), spread_codes(cluster_last, 5.0, rng))


# Label, dofadjustments string. Only the clusters adjustment runs the nested test.
DOF_VARIANTS = (("firstpair", "firstpair"), ("firstpairc", "firstpair clusters"), ("all", "all"))


def scenario_dof(module, out):
    y, X, fes, cluster_true, cluster_last = dof_data()
    for tag, dof in DOF_VARIANTS:
        for name, cluster in (("nested_true", cluster_true), ("conflict_last", cluster_last)):
            run_fit(module, out, f"dof_{tag}_{name}", dict(groupvar=True, dofadjustments=dof,
                                                           se_type="cluster", num_threads=4),
                    (y, X), dict(fes=fes, clusters=cluster))


def scenario_negative(module, out):
    d = firm_panel(seed=3, workers=2000, firms=300)
    for name, bad in (("minus_one", -1), ("int_min", INT32_MIN)):
        firm = d["firm_raw"].copy()
        firm[17] = bad
        run_fit(module, out, f"neg_{name}", dict(), (d["y"], d["X"]),
                dict(fes=[d["worker"], firm, d["year"]], clusters=d["worker"]))


def scenario_cache(module, out, root: Path, step: str):
    # step: "<abs|fe>:<write|read>:<file>"; one fit on its own cache file, kept
    # in the run's root folder so that another process can read it.
    kind, mode, name = step.split(":")
    d = firm_panel(seed=8)
    if kind == "abs":
        os.environ["XHDFE_ABSORPTION_CACHE"] = str(root / name)
        os.environ["XHDFE_ABSORPTION_CACHE_MODE"] = mode
    else:
        os.environ["XHDFE_FE_STRUCTURE_CACHE"] = str(root / name)
        os.environ["XHDFE_FE_STRUCTURE_MODE"] = mode
    run_fit(module, out, "cache", dict(), (d["y"], d["X"]),
            dict(fes=[d["worker"], d["firm_raw"], d["year"]], clusters=d["worker"]))


def scenario_large(module, out):
    rng = np.random.default_rng(10)
    n, levels = 64_000_000, 60_000_000
    big = np.concatenate([np.arange(levels), rng.integers(0, levels, size=n - levels)])
    big = big[rng.permutation(n)].astype(np.int32)
    other = (np.arange(n) % 1000).astype(np.int32)
    x = rng.normal(size=(n, 1))
    y = 0.3 * x[:, 0] + rng.normal(size=n)
    run_fit(module, out, "large", dict(se_type="robust", num_threads=16),
            (y, np.asfortranarray(x)), dict(fes=[big, other]))


SCENARIOS = {
    "firm": scenario_firm, "precoded": scenario_precoded, "counter": scenario_counter,
    "marking": scenario_marking, "decline": scenario_decline, "dof": scenario_dof,
    "negative": scenario_negative,
}


def child(args) -> int:
    module = load(args.module)
    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    if args.scenario == "cache":
        scenario_cache(module, out, Path(args.out), args.step)
    elif args.scenario == "large":
        scenario_large(module, out)
    else:
        SCENARIOS[args.scenario](module, out)
    return 0


# --- Parent -------------------------------------------------------------------------

def launch(script, module, out, label, scenario, *, sparse_rank=True, step="", extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("XHDFE_")}
    env.update({"XHDFE_GPU_BACKEND": "cpu", "XHDFE_ABSORPTION_CACHE_MODE": "off",
                "XHDFE_FE_STRUCTURE_MODE": "off", "XHDFE_MOBILITY_MODE": "off",
                "XHDFE_PROFILE_CPU": "1", "XHDFE_REDUCED_PCG_TRACE": "1",
                "XHDFE_DEBUG_SOLVER": "1", "OMP_WAIT_POLICY": "passive",
                # The frozen module predates D10: compared bit for bit with D10 off
                # (D10 has its own contract, tests/reduced_pcg_chat_stop_contract.py).
                "XHDFE_REDUCED_PCG_CHAT_STOP": "0"})
    if not sparse_rank:
        env["XHDFE_SPARSE_RANK"] = "0"
    if extra:
        env.update(extra)
    command = [sys.executable, "-B", "-I", str(script), "--child", "--module", str(module),
               "--scenario", scenario, "--label", label, "--out", str(out), "--step", step]
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{label}: child failed ({done.returncode}): {done.stderr[-2000:]}")
    return out / label


def records(folder: Path):
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


def partition(labels):
    """Labels renumbered by first appearance: equal partitions give equal lists."""
    seen = {}
    return [seen.setdefault(v, len(seen)) for v in labels]


def parent(args) -> int:
    script = Path(__file__).resolve()
    failures = []

    def fail(message):
        failures.append(message)
        print("FAIL", message, flush=True)

    def same(a, b, what):
        if a.get("error") or b.get("error"):
            if a.get("error") != b.get("error"):
                fail(f"{what}: errors differ: {a.get('error')} != {b.get('error')}")
            return
        if "gelbach" in a or "gelbach" in b:
            if a.get("gelbach") != b.get("gelbach"):
                keys = sorted(k for k in set(a.get("gelbach", {})) | set(b.get("gelbach", {}))
                              if a.get("gelbach", {}).get(k) != b.get("gelbach", {}).get(k))
                fail(f"{what}: Gelbach differs in {keys}")
            return
        for r in (a, b):
            if r.get("missing"):
                fail(f"{what}: missing properties {r['missing']}")
        diff = sorted(k for k in REQUIRED if a["public"].get(k) != b["public"].get(k))
        if diff:
            fail(f"{what}: differs in {diff}")
        if a.get("diagnostics") != b.get("diagnostics") or a.get("method") != b.get("method"):
            fail(f"{what}: diagnostics {a.get('diagnostics')} / {b.get('diagnostics')}, "
                 f"method {a.get('method')} / {b.get('method')}")

    def lines_of(r, site):
        return [l for l in r.get("lines", []) if f"site={site} " in l]

    def control(r, site, path, what, reason=None):
        hits = [l for l in lines_of(r, site) if f"path={path} " in l and
                (reason is None or f"reason={reason} " in l)]
        if not hits:
            fail(f"{what}: positive control, no site={site} path={path}"
                 f"{' reason=' + reason if reason else ''} line")

    def three(scenario, *, step="", extra=None):
        new = records(launch(script, args.module, out, f"{scenario}_new{step}", scenario,
                             step=step, extra=extra))
        off = records(launch(script, args.module, out, f"{scenario}_off{step}", scenario,
                             sparse_rank=False, step=step, extra=extra))
        base = records(launch(script, args.frozen_module, out, f"{scenario}_base{step}", scenario,
                              step=step, extra=extra))
        # The in-situ verifier: every site that took the index recomputes today's
        # serial result and compares it (XHDFE_SPARSE_RANK_VERIFY=1).
        ver = records(launch(script, args.module, out, f"{scenario}_verify{step}", scenario,
                             step=step,
                             extra={**(extra or {}), "XHDFE_SPARSE_RANK_VERIFY": "1"}))
        if not new:
            fail(f"{scenario}: no records (vacuous)")
        for label in sorted(new):
            if label not in base or label not in off or label not in ver:
                fail(f"{scenario}/{label}: missing in base, off or verify")
                continue
            same(new[label], base[label], f"{scenario}/{label}: new against the frozen module")
            same(new[label], off[label], f"{scenario}/{label}: new against the switch off")
            same(new[label], ver[label], f"{scenario}/{label}: new against the verify arm")
            for l in off[label].get("lines", []):
                if "label=sparse_rank" in l and "site=components" not in l and "path=bitmap" in l:
                    fail(f"{scenario}/{label}: the switch off ran the index: {l}")
            for l in ver[label].get("lines", []):
                if "label=sparse_rank_verify" in l and "ok=0" in l:
                    fail(f"{scenario}/{label}: the verifier found a difference: {l}")
        took_index = any("path=bitmap" in l for r in new.values() for l in r.get("lines", []))
        verified = any("label=sparse_rank_verify" in l and "ok=1" in l
                       for r in ver.values() for l in r.get("lines", []))
        if took_index and not verified:
            fail(f"{scenario}: the index ran but the verify arm verified nothing")
        return new, off, base

    only = set(args.only.split(",")) if args.only else None
    wanted = lambda name: only is None or name in only  # noqa: E731

    with tempfile.TemporaryDirectory(prefix="sparse_rank_contract_") as tmp:
        out = Path(tmp)

        if wanted("firm"):
            # Correction E1 (Verifications/low_memory_20261116/briefs/E1_SAVEFE_COMPARABLE_PROJECTION_FIRST.md):
            # a fit with saved effects whose capturing sweep stops above the floor of the mode is followed by
            # the projection of the mode, which the tier may publish. The savefe fit of this scenario checks the
            # former rule against the frozen module (the capturing sweep, never in the tier);
            # XHDFE_SAVEFE_ACCURATE_PROJECTION=0 keeps that path. No other fit here saves effects, so none
            # reads the switch, and the frozen module does not know it.
            new, off, _ = three("firm", extra={"XHDFE_SAVEFE_ACCURATE_PROJECTION": "0"})
            control(new["cmp_on"], "tier_codes", "bitmap", "firm/cmp_on: tier codes")
            # Since phase 2 the tier's certificate takes the tier's codes over
            # (tests/shared_fe_coding_contract.py); the indexer builder runs in the
            # fits without the tier.
            control(new["cmp_on"], "cert_shared", "codes", "firm/cmp_on: certificate takes the codes")
            control(new["cmp_on"], "singleton_lite", "bitmap", "firm/cmp_on: singleton scan")
            control(new["cmp_on"], "lookup", "bitmap", "firm/cmp_on: DoF lookups")
            control(new["cmp_off"], "indexer", "bitmap", "firm/cmp_off: absorber indexers")
            control(new["fweights"], "indexer_lite", "bitmap", "firm/fweights: weighted singletons")
            # The nested test keeps today's hash (bench decision of 14 Nov): no control.
            control(new["sorted_by_firm_off"], "indexer", "bitmap", "firm/sorted_by_firm_off: indexers")
            control(new["sorted_by_firm"], "cert_shared", "codes", "firm/sorted_by_firm: codes taken")
            control(off["cmp_on"], "tier_codes", "serial", "firm/cmp_on off: tier codes", "off")
            control(off["cmp_off"], "indexer", "serial", "firm/cmp_off off: indexers", "off")
            print(f"INFO firm/cmp_on: method {new['cmp_on'].get('method')} "
                  f"(8 = the tier published)", flush=True)

        if wanted("precoded"):
            new, _, _ = three("precoded")
            raw, coded = new["precoded_raw"], new["precoded_coded"]
            diff = sorted(k for k in PRECODED_SAME if raw["public"].get(k) != coded["public"].get(k))
            if diff:
                fail(f"precoded: raw against pre-coded differs in {diff}")
            if raw.get("groupvar") is None or \
                    partition(raw["groupvar"]) != partition(coded["groupvar"]):
                fail("precoded: groupvar partitions differ")

        if wanted("counter"):
            new, off, _ = three("counter")
            for label in ("c8_firstpair", "c8_all", "c32_firstpair", "c32_all"):
                g = new[label].get("groupvar")
                expect = ([1] * 4 + [2] * 4) * (len(g) // 8 if g else 0)
                if g != expect:
                    fail(f"counter/{label}: labels {g[:8] if g else g} (want [1 x4, 2 x4] per copy)")
            for label in ("cbig_firstpair", "cbig_all"):
                g = new[label].get("groupvar") or []
                if g[:8] != [1] * 4 + [2] * 4:
                    fail(f"counter/{label}: labels {g[:8]}")
            for label in ("c32_firstpair", "cbig_firstpair"):
                control(new[label], "lookup", "bitmap", f"counter/{label}: DoF lookups")
            if lines_of(new["c8_firstpair"], "lookup") and \
                    not all("path=serial" in l for l in lines_of(new["c8_firstpair"], "lookup")):
                fail("counter/c8: the 8-row case reached the index (it must not: the guard)")

        if wanted("marking"):
            new, off, _ = three("marking")
            for kind, other in (("dense", "dense"), ("densemap", "densemap"), ("rows", "rows")):
                for label in (f"mark_{kind}", f"mark_{kind}_gelbach"):
                    hits = [l for l in lines_of(new[label], "components")
                            if "path=marking" in l and "base=rows" in l and f"other={other} " in l]
                    if not hits:
                        fail(f"marking/{label}: no components line path=marking base=rows other={other}")
            for label in ("mark_sparse", "mark_sparse_gelbach"):
                if not [l for l in lines_of(new[label], "components")
                        if "base=rows" in l and "other=sparse" in l]:
                    fail(f"marking/{label}: no components line base=rows other=sparse")
            for label in sorted(off):
                if [l for l in lines_of(off[label], "components") if "base=rows" in l]:
                    fail(f"marking/{label}: Rows lookup with the switch off")

        if wanted("decline"):
            new, off, _ = three("decline", extra={"XHDFE_TEST_REDUCED_PCG_DECLINE": "certificate"})
            r = new["decline"]
            finals = [l for l in r.get("lines", []) if "reduced_pcg event=final" in l]
            if not any("status=declined" in l and "reason=forced" in l for l in finals):
                fail(f"decline: the tier did not decline at the forced stage: {finals}")
            control(r, "tier_codes", "bitmap", "decline: tier codes before the decline")
            control(r, "indexer", "bitmap", "decline: certificate or legacy indexers")
            if r.get("method") == 8:
                fail("decline: method 8 after a forced decline")

        if wanted("dof"):
            new, off, _ = three("dof")
            for tag, dof in DOF_VARIANTS:
                for name in ("nested_true", "conflict_last"):
                    label = f"dof_{tag}_{name}"
                    control(new[label], "lookup", "bitmap", f"dof/{label}: DoF lookups")
                    control(new[label], "singleton_lite", "bitmap", f"dof/{label}: singleton scan")
                # The fixture itself: a singleton dropped, C coarsened away (fully
                # redundant), and, with the clusters adjustment, the nested test true
                # for B in one variant and false (conflict on the last row) in the other.
                a, b = new[f"dof_{tag}_nested_true"], new[f"dof_{tag}_conflict_last"]
                if a.get("error") or b.get("error"):
                    fail(f"dof/{tag}: fit failed: {a.get('error') or b.get('error')}")
                    continue
                va, vb = a["values"], b["values"]
                if not va.get("num_singletons_"):
                    fail(f"dof/{tag}: no singleton dropped ({va.get('num_singletons_')})")
                if dof == "all" and (va.get("fe_redundant_") or [0, 0, -1])[2] != \
                        (va.get("fe_num_levels_") or [0, 0, -2])[2]:
                    fail(f"dof/{tag}: C not coarsened away: redundant {va.get('fe_redundant_')}, "
                         f"levels {va.get('fe_num_levels_')}")
                if dof == "all" or "clusters" in dof.split():
                    if va.get("df_a_nested_") == vb.get("df_a_nested_"):
                        fail(f"dof/{tag}: the nested test gave the same outcome in both variants "
                             f"(df_a_nested_ {va.get('df_a_nested_')})")
                elif va.get("df_a_nested_") or vb.get("df_a_nested_"):
                    fail(f"dof/{tag}: the nested test ran without the clusters adjustment "
                         f"(df_a_nested_ {va.get('df_a_nested_')}, {vb.get('df_a_nested_')})")

        if wanted("negative"):
            new, _, base = three("negative")
            for label in ("neg_minus_one", "neg_int_min"):
                if not new[label].get("error"):
                    fail(f"negative/{label}: accepted (today it is refused)")

        if wanted("cache"):
            def cache_leg(module, who, step):
                folder = launch(script, module, out, f"cache_{who}_{step.replace(':', '_')}",
                                "cache", step=step)
                return records(folder)["cache"]
            def gap(r, w, name):
                a = np.asarray(r.get("values", {}).get(name) or [], dtype=float)
                b = np.asarray(w.get("values", {}).get(name) or [], dtype=float)
                if a.shape != b.shape or not a.size:
                    return float("nan")
                return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))

            for kind in ("abs", "fe"):
                wb = cache_leg(args.frozen_module, "base", f"{kind}:write:{kind}_b.cache")
                wn = cache_leg(args.module, "new", f"{kind}:write:{kind}_n.cache")
                same(wn, wb, f"cache/{kind}: the two writers")
                writer = {"b": wb, "n": wn}
                reads = {}
                for who, reader in (("new", args.module), ("base", args.frozen_module)):
                    for src in ("b", "n"):
                        name = f"{kind}_{src}.cache"
                        r = reads[who, src] = cache_leg(reader, who, f"{kind}:read:{name}")
                        if kind == "abs":
                            same(r, writer[src], f"cache/abs: {who} reads {name}")
                            if not any("cache_hit=1" in l for l in r.get("lines", [])):
                                fail(f"cache/abs: {who} did not hit {name}")
                        elif lines_of(wn, "singleton_lite") and lines_of(r, "singleton_lite"):
                            # A hit skips the singleton scan (hr:15070).
                            fail(f"cache/fe: {who} scanned for singletons reading {name} (no hit)")
                # The read path itself across versions: both modules reading one file,
                # and one module reading either (byte-identical) file.
                for src in ("b", "n"):
                    same(reads["new", src], reads["base", src],
                         f"cache/{kind}: new and base read {kind}_{src}.cache")
                for who in ("new", "base"):
                    same(reads[who, "b"], reads[who, "n"], f"cache/{kind}: {who} reads either file")
                if kind == "fe":
                    # An FE-structure read hands run_fit the stored first-appearance codes
                    # instead of the raw ids (hr:15077), so it may differ from the fresh
                    # fit in rounding; reported, in both modules alike (checked above).
                    if not lines_of(wn, "singleton_lite"):
                        fail("cache/fe: the new writer printed no singleton_lite line, "
                             "so a hit is not observable")
                    for who, src in (("base", "b"), ("new", "n")):
                        print(f"INFO cache/fe: {who} reading its own file against its fresh fit, "
                              f"max relative gap coef {gap(reads[who, src], writer[src], 'coef_'):.3g}, "
                              f"stderr {gap(reads[who, src], writer[src], 'stderr_'):.3g}",
                              flush=True)
                fb, fn = out / f"{kind}_b.cache", out / f"{kind}_n.cache"
                if not (fb.exists() and fn.exists()):
                    fail(f"cache/{kind}: a writer left no cache file")
                else:
                    print(f"INFO cache/{kind}: written files "
                          f"{'identical' if fb.read_bytes() == fn.read_bytes() else 'differ'}",
                          flush=True)

        if args.large:
            three("large")

    print("PASS" if not failures else f"{len(failures)} FAILURES", flush=True)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--module", required=True)
    parser.add_argument("--frozen-module")
    parser.add_argument("--scenario", default="")
    parser.add_argument("--step", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--only", default="")
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    if args.child:
        return child(args)
    if not args.frozen_module:
        parser.error("--frozen-module is required")
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
