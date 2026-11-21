#!/usr/bin/env python3
"""Phase 2 of the Stata-path plan: the reduced-system tier's certificate takes the
tier's codes over (certify_tier_absorption, src/fe_absorption.cpp) and every output
stays today's (design note Verifications/perf_stata_path_20261114/fase2/
DESENHO_FASE2.md, section 6).

Builds nothing; never run during a timing campaign. Synthetic data only, made in
each child process from fixed seeds; nothing is read from disk but the modules.

    python3 -I tests/shared_fe_coding_contract.py --module <this branch's module> \
        --frozen-module <module of the core at 9d7ba996f, packages/cand_sp1_py_cpu> \
        [--only NAME] [--large]

Every scenario runs its fits in four fresh processes:
  new    the branch's module, XHDFE_SHARED_FE_CODING unset (the codes are taken over)
  off    the branch's module, XHDFE_SHARED_FE_CODING=0 (the certificate numbers again)
  base   the frozen module
  verify the branch's module with XHDFE_SPARSE_RANK_VERIFY=1: the certificate compares
         every indexer it takes over with build_indexer_team() field by field (an
         "ok=0" line, or an error, is a failure; an "ok=1" line for site=cert_shared is
         required wherever new took codes over)
all with XHDFE_PROFILE_CPU=1, XHDFE_REDUCED_PCG_TRACE=1, XHDFE_GPU_BACKEND=cpu and the
caches off. new is compared with base, off and verify on every REQUIRED property by
dtype, shape and raw bytes (threads_used_ and parallel_workers_active_ among them:
condition 3), on the tier's diagnostics, on the method and on the error raised if any.
Old and new agree by construction, so the positive controls carry the evidence: one
"label=sparse_rank site=cert_shared" line per effect of the tier's certificate,
path=codes where the effect's indexer numbers by rank and path=build reason=holes
where it keeps holes, identified by range and levels; none with the switch off, none
where the tier does not run its certificate.

Scenarios:
  occupancy  four effects: sparse (range 8.7 n), 89 % of a range of 1000 (compacts),
             90 % of a range of 1000 with the holes next to both ends (keeps them: not
             shared) and a full range of 500; the tier on with teams 1, 2, 8, 16, 24
             and 48, rows sorted by the sparse effect (contiguous groups), the tier off,
             fast mode and frequency weights (no tier: no line)
  boundary   ranges of exactly 2n and 2n + 1 rows, 90 of 101 (compacts) and 91 of 101
             (keeps its holes), at 16 threads
  declines   the occupancy design at 16 threads with the tier forced to decline at
             each stage (geometry, iterate, residuals, certificate, bad_alloc): every
             arm gives the legacy chain's result; only the certificate stage, which
             runs after the certificate, prints the lines
  large      (--large) an effect of 6.25 million levels whose range of 50 000 001
             passes the 50 million cap of the builder's dense branch
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
    "omitted_", "omitted_reason_", "openmp_enabled_", "parallel_workers_active_",
    "precision_certified_", "pvalues_", "r2_", "r2_within_", "residuals_", "rss_",
    "sample_index_", "saturated_", "slope_accuracy_retry_iterations_",
    "slope_accuracy_retry_stages_", "slope_block_checked_", "slope_block_frobenius_rel_",
    "slope_block_max_rel_", "slope_block_residual_rel_", "slope_block_rms_rel_",
    "slope_block_skipped_max_rel_", "slope_certificate_worst_fe_",
    "slope_certificate_worst_moment_", "slope_internal_tolerance_", "stderr_",
    "thread_limit_code_", "thread_limit_reason_", "threads_effective_", "threads_requested_",
    "threads_used_", "tss_", "tss_within_", "tvalues_",
)
DIAGNOSTICS = ("reduced_pcg_status_", "reduced_pcg_reason_", "reduced_pcg_discarded_iterations_")
TEAMS = (1, 2, 8, 16, 24, 48)
STAGES = ("geometry", "iterate", "residuals", "certificate", "bad_alloc")


def load(path: str):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(value):
    """dtype, shape and the SHA-256 of the raw bytes; strings as themselves."""
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

def present_values(low, span, present, rng, holes_at_ends=False):
    """`present` distinct values of [low, low + span), both ends included; the missing
    ones inside at random, or next to the two ends first."""
    missing = span - present
    if holes_at_ends:
        drop = [1 + k // 2 if k % 2 == 0 else span - 2 - k // 2 for k in range(missing)]
    else:
        drop = rng.choice(np.arange(1, span - 1), size=missing, replace=False)
    keep = np.ones(span, dtype=bool)
    keep[np.asarray(drop, dtype=np.int64)] = False
    return (low + np.flatnonzero(keep)).astype(np.int64)


def assign(values, n, rng):
    """Every value on at least two rows, the rest at random, in a random order."""
    values = np.asarray(values)
    base = np.repeat(values, 2)
    rest = values[rng.integers(0, values.size, size=n - base.size)]
    return rng.permutation(np.concatenate([base, rest])).astype(np.int32)


def outcome(fes, rng, n):
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    y = X @ np.array([0.5, -0.25]) + 0.3 * rng.normal(size=n)
    for f in fes:
        _, codes = np.unique(f, return_inverse=True)
        y = y + rng.normal(size=int(codes.max()) + 1)[codes]
    return y, np.asfortranarray(X)


def occupancy_design(seed=20261114, n=400_000):
    rng = np.random.default_rng(seed)
    sparse_levels = 40_000
    top = int(8.7 * n)
    inner = np.sort(rng.choice(top - 1, size=sparse_levels - 2, replace=False) + 1)  # in [1, top)
    sparse_values = np.concatenate([[0], inner, [top]])  # range 8.7 n + 1
    fes = [assign(sparse_values, n, rng),
           assign(present_values(1000, 1000, 890, rng), n, rng),        # 89 %: compacts
           assign(present_values(5000, 1000, 900, rng, True), n, rng),  # 90 %, holes at the ends
           assign(np.arange(3000, 3500), n, rng)]                       # full range
    y, X = outcome(fes, rng, n)
    expect = [(int(8.7 * n) + 1, sparse_levels, "codes"), (1000, 890, "codes"),
              (1000, 900, "build"), (500, 500, "codes")]
    return y, X, fes, expect


def boundary_design(seed=7, n=200_000):
    rng = np.random.default_rng(seed)
    two_n = present_values(0, 2 * n, 40_000, rng)          # range exactly 2n
    two_n_plus = present_values(0, 2 * n + 1, 40_000, rng)  # range 2n + 1
    fes = [assign(two_n, n, rng), assign(two_n_plus, n, rng),
           assign(present_values(11, 101, 90, rng), n, rng),   # 900 < 909: compacts
           assign(present_values(500, 101, 91, rng), n, rng)]  # 910 >= 909: keeps its holes
    y, X = outcome(fes, rng, n)
    expect = [(2 * n, 40_000, "codes"), (2 * n + 1, 40_000, "codes"), (101, 90, "codes"),
              (101, 91, "build")]
    return y, X, fes, expect


def ctor(module, **kw):
    base = dict(se_type="cluster", tol=1e-8, max_iter=100000, fit_intercept=True,
                num_threads=16, drop_singletons=True, symmetric_sweep=False,
                absorption_method="auto", tolerance_mode="reghdfe-comparable")
    base.update(kw)
    return module.HdfeRegressor(**base)


def run_fit(module, out: Path, label: str, ctor_kw, fit_args, fit_kw, expect=None):
    log = out / f"{label}.stderr"
    record = {"error": None, "expect": expect}
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
                      method=int(reg.absorption_method_used),
                      threads=[int(reg.threads_used_), int(reg.parallel_workers_active_)])
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["lines"] = [l for l in (log.read_text().splitlines() if log.exists() else [])
                       if "label=sparse_rank" in l or "reduced_pcg event=final" in l]
    (out / f"{label}.json").write_text(json.dumps(record, sort_keys=True))


# --- Scenarios (children) ---------------------------------------------------------

def scenario_occupancy(module, out):
    y, X, fes, expect = occupancy_design()
    clusters = fes[0]
    for team in TEAMS:
        run_fit(module, out, f"tier_t{team}", dict(reduced_pcg=True, num_threads=team), (y, X),
                dict(fes=fes, clusters=clusters), expect)
    order = np.argsort(fes[0], kind="stable")
    sorted_fes = [np.ascontiguousarray(f[order]) for f in fes]
    run_fit(module, out, "tier_sorted", dict(reduced_pcg=True), (y[order], np.asfortranarray(X[order])),
            dict(fes=sorted_fes, clusters=sorted_fes[0]), expect)
    run_fit(module, out, "tier_off", dict(reduced_pcg=False), (y, X), dict(fes=fes, clusters=clusters))
    run_fit(module, out, "fast", dict(tolerance_mode="xhdfe-fast"), (y, X),
            dict(fes=fes, clusters=clusters))
    weights = np.random.default_rng(3).integers(1, 4, size=y.size).astype(np.float64)
    run_fit(module, out, "fweights", dict(se_type="robust"), (y, X),
            dict(fes=fes, weights=weights, fweights=True))


def scenario_boundary(module, out):
    y, X, fes, expect = boundary_design()
    run_fit(module, out, "boundary", dict(reduced_pcg=True, se_type="robust"), (y, X),
            dict(fes=fes), expect)


def scenario_declines(module, out):
    y, X, fes, expect = occupancy_design(seed=5, n=200_000)
    run_fit(module, out, "declined", dict(reduced_pcg=True), (y, X),
            dict(fes=fes, clusters=fes[0]), expect)


def scenario_large(module, out):
    # 6 250 001 levels of four rows (12.5 million levels of two rows exceed the
    # tier's admission budget, which then declines before its certificate).
    rng = np.random.default_rng(10)
    levels = 6_250_001
    big = np.repeat(8 * np.arange(levels, dtype=np.int64), 4)      # range 50 000 001 <= 2n
    big = big[rng.permutation(big.size)].astype(np.int32)
    n = big.size
    other = assign(np.arange(1000), n, rng)
    y, X = outcome([other], rng, n)
    run_fit(module, out, "large", dict(reduced_pcg=True, se_type="robust"), (y, X),
            dict(fes=[big, other]), [(50_000_001, levels, "codes"), (1000, 1000, "codes")])


SCENARIOS = {"occupancy": scenario_occupancy, "boundary": scenario_boundary,
             "declines": scenario_declines, "large": scenario_large}


def child(args) -> int:
    module = load(args.module)
    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    SCENARIOS[args.scenario](module, out)
    return 0


# --- Parent -------------------------------------------------------------------------

def launch(script, module, out, label, scenario, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("XHDFE_")}
    env.update({"XHDFE_GPU_BACKEND": "cpu", "XHDFE_ABSORPTION_CACHE_MODE": "off",
                "XHDFE_FE_STRUCTURE_MODE": "off", "XHDFE_MOBILITY_MODE": "off",
                "XHDFE_PROFILE_CPU": "1", "XHDFE_REDUCED_PCG_TRACE": "1",
                "XHDFE_DEBUG_SOLVER": "1", "OMP_WAIT_POLICY": "passive",
                # The frozen module predates D10: compared bit for bit with D10 off
                # (D10 has its own contract, tests/reduced_pcg_chat_stop_contract.py).
                "XHDFE_REDUCED_PCG_CHAT_STOP": "0"})
    if extra:
        env.update(extra)
    command = [sys.executable, "-B", "-I", str(script), "--child", "--module", str(module),
               "--scenario", scenario, "--label", label, "--out", str(out)]
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{label}: child failed ({done.returncode}): {done.stderr[-2000:]}")
    return out / label


def records(folder: Path):
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


LINE = re.compile(r"site=cert_shared path=(\w+) reason=(\w+) n=(\d+) range=(\d+) levels=(\d+)")


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
        for r in (a, b):
            if r.get("missing"):
                fail(f"{what}: missing properties {r['missing']}")
        diff = sorted(k for k in REQUIRED if a["public"].get(k) != b["public"].get(k))
        if diff:
            fail(f"{what}: differs in {diff}")
        if a.get("diagnostics") != b.get("diagnostics") or a.get("method") != b.get("method"):
            fail(f"{what}: diagnostics {a.get('diagnostics')} / {b.get('diagnostics')}, "
                 f"method {a.get('method')} / {b.get('method')}")

    def shared_lines(r):
        return [LINE.search(l).groups() for l in r.get("lines", []) if LINE.search(l)]

    def expect_lines(r, what):
        """One line per effect, path as expected, matched by range and levels."""
        got = sorted((int(rng), int(lev), path, reason) for path, reason, _, rng, lev in shared_lines(r))
        want = sorted((rng, lev, path, "rank" if path == "codes" else "holes")
                      for rng, lev, path in r.get("expect") or [])
        if got != want:
            fail(f"{what}: cert_shared lines {got}, expected {want}")

    def no_lines(r, what):
        if shared_lines(r):
            fail(f"{what}: cert_shared lines where none may be: {shared_lines(r)}")

    def four(scenario, extra=None, tag=""):
        arms = {}
        for arm, env in (("new", {}), ("off", {"XHDFE_SHARED_FE_CODING": "0"}),
                         ("base", None), ("verify", {"XHDFE_SPARSE_RANK_VERIFY": "1"})):
            module = args.frozen_module if env is None else args.module
            arms[arm] = records(launch(script, module, out, f"{scenario}{tag}_{arm}", scenario,
                                       extra={**(extra or {}), **(env or {})}))
        new, off, base, ver = (arms[a] for a in ("new", "off", "base", "verify"))
        if not new:
            fail(f"{scenario}{tag}: no records (vacuous)")
        for label in sorted(new):
            if any(label not in arms[a] for a in arms):
                fail(f"{scenario}{tag}/{label}: missing in an arm")
                continue
            for arm in ("base", "off", "verify"):
                same(new[label], arms[arm][label], f"{scenario}{tag}/{label}: new against {arm}")
            if new[label].get("threads") != base[label].get("threads"):
                fail(f"{scenario}{tag}/{label}: threads_used, active workers "
                     f"{new[label].get('threads')} against base {base[label].get('threads')}")
            no_lines(off[label], f"{scenario}{tag}/{label} off")
            no_lines(base[label], f"{scenario}{tag}/{label} base")
            for l in ver[label].get("lines", []):
                if "label=sparse_rank_verify" in l and "ok=0" in l:
                    fail(f"{scenario}{tag}/{label}: the verifier found a difference: {l}")
            took = any(p == "codes" for p, *_ in shared_lines(new[label]))
            verified = any("label=sparse_rank_verify site=cert_shared ok=1" in l
                           for l in ver[label].get("lines", []))
            if took and not verified:
                fail(f"{scenario}{tag}/{label}: codes taken over but no cert_shared ok=1 in verify")
        return new

    only = set(args.only.split(",")) if args.only else None
    wanted = lambda name: only is None or name in only  # noqa: E731

    with tempfile.TemporaryDirectory(prefix="shared_fe_coding_contract_") as tmp:
        out = Path(tmp)

        if wanted("occupancy"):
            new = four("occupancy")
            for team in TEAMS:
                expect_lines(new[f"tier_t{team}"], f"occupancy/tier_t{team}")
                if new[f"tier_t{team}"].get("method") != 8:
                    print(f"INFO occupancy/tier_t{team}: method {new[f'tier_t{team}'].get('method')} "
                          f"(8 = the tier published)", flush=True)
            expect_lines(new["tier_sorted"], "occupancy/tier_sorted")
            for label in ("tier_off", "fast", "fweights"):
                no_lines(new[label], f"occupancy/{label}")
            print("INFO occupancy: threads_used, active workers by team "
                  + ", ".join(f"{t}:{new[f'tier_t{t}'].get('threads')}" for t in TEAMS), flush=True)

        if wanted("boundary"):
            new = four("boundary")
            expect_lines(new["boundary"], "boundary")

        if wanted("declines"):
            for stage in STAGES:
                new = four("declines", extra={"XHDFE_TEST_REDUCED_PCG_DECLINE": stage}, tag=f"_{stage}")
                r = new["declined"]
                finals = [l for l in r.get("lines", []) if "reduced_pcg event=final" in l]
                if not any("status=declined" in l for l in finals):
                    fail(f"declines/{stage}: the tier did not decline: {finals}")
                if r.get("method") == 8:
                    fail(f"declines/{stage}: method 8 after a forced decline")
                if stage == "certificate":
                    expect_lines(r, f"declines/{stage}")
                else:
                    no_lines(r, f"declines/{stage}")

        if args.large:
            new = four("large")
            expect_lines(new["large"], "large")

    print("PASS" if not failures else f"{len(failures)} FAILURES", flush=True)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--module", required=True)
    parser.add_argument("--frozen-module")
    parser.add_argument("--scenario", default="")
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
