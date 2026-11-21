#!/usr/bin/env python3
"""D10: the reduced-system tier stops its columns on the confirmed chat at 1e-12
(pre-registration and design note Verifications/perf_stata_path_20261114/d10/,
reading B, section 11), against the switch off and the frozen phase chain.

Builds nothing; never run during a timing campaign. Synthetic data only, made in
each child process from fixed seeds.

    python3 -I tests/reduced_pcg_chat_stop_contract.py --module <this branch's module> \
        --frozen-module <phase chain, packages/cand_sp2_py_cpu> [--only NAME]

Arms, every scenario in fresh processes, all with XHDFE_REDUCED_PCG_TRACE=2 (the
tier's per-column chat and certificate) and XHDFE_DEBUG_SOLVER=1:
  on    the branch's module (D10 on by default)
  off   the branch's module with XHDFE_REDUCED_PCG_CHAT_STOP=0
  base  the frozen module
Checks:
  - off == base bit for bit on every REQUIRED property, the diagnostics and the method;
  - on == off bit for bit wherever D10 cannot act: routes outside the tier (fast,
    weights, saved effects, slopes, IV, an explicit method, the tier off),
    group()/individual(), a public tolerance whose certificate is stricter than the
    trigger (1e-14 and 1.2e-12: scope), and the result of every forced decline;
  - where the tier publishes under D10: b within 1e-9 max(1,|b|) and V within
    1e-8 sqrt(V_jj V_kk) of the switch off and of reducedpcg(off); every column
    stopped by chat has a finite chat <= 1e-12; the published rel is listed, and a
    value above 1e-12 must be explained by a column stopped by span, test1 or zero,
    or by the certificate's rounding against the confirmed chat (reported); every
    accepted tier result is certified with rel <= 1e-11;
  - scope: the plan line carries chat_stop=1e-12 at tol 1.3e-12 and 1e-8, and no
    chat_stop at 1.2e-12 and 1e-14;
  - cache: an entry written with D10 is never read with the switch off and the
    reverse; each state reads its own entry back, bit for bit. The same at tol
    1.3e-12 (in scope); at tol 1.2e-12 (out of scope, today's key) each state reads
    the other's entry, bit for bit.
Scenarios: panel (a worker-firm-year panel, every route), dominant (regressors
around the dominant-origin boundary 1/1024 of the certificate's scale, a nearly
constant one, a column nearly in the span of the effects, a zero column), declines
(each forced stage), cache.
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
    "absorption_method_used", "abs_residual_", "abs_residual_rel_", "coef_", "conf_int_",
    "converged_", "covariance_", "df_a_", "df_m_", "df_resid_", "fe_num_levels_", "nobs_",
    "num_iterations_", "num_singletons_", "omitted_", "precision_certified_", "pvalues_", "r2_",
    "r2_within_", "residuals_", "rss_", "sample_index_", "stderr_", "threads_used_",
    "parallel_workers_active_", "tss_", "tss_within_", "tvalues_", "auto_routing_retry_fired_",
    "auto_routing_retry_status_", "gpu_used_",
)
DIAGNOSTICS = ("reduced_pcg_status_", "reduced_pcg_reason_", "reduced_pcg_discarded_iterations_")
STAGES = ("geometry", "iterate", "residuals", "certificate", "bad_alloc")


def load(path: str):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(value):
    if isinstance(value, str):
        return {"str": value}
    if not isinstance(value, (bool, int, float, list, tuple, np.ndarray, np.generic)) and hasattr(value, "__int__"):
        value = int(value)
    array = np.asarray(value)
    if array.dtype == object:
        return {"repr": repr(value)}
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


@contextlib.contextmanager
def captured_stderr(path: Path):
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


def panel(seed=20261115, workers=30_000, firms=2_500, years=12):
    rng = np.random.default_rng(seed)
    rows = rng.integers(3, 9, size=workers)
    worker = np.repeat(np.arange(workers), rows)
    n = worker.size
    year = (rng.integers(0, years - 8, size=workers)[worker] + np.concatenate([np.arange(r) for r in rows])) % years
    home = rng.integers(0, firms, size=workers)
    firm = np.where(rng.random(n) < 0.15, rng.integers(0, firms, size=n), home[worker])
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n) + 0.1 * (firm % 7), 1.0e3 + rng.normal(size=n)])
    y = X @ np.array([0.5, -0.25, 0.01]) + 0.3 * rng.normal(size=n)
    fes = [worker, firm, year]
    for f in fes:
        y = y + rng.normal(size=int(f.max()) + 1)[f]
    return y, np.asfortranarray(X), [np.ascontiguousarray(f, dtype=np.int32) for f in fes], rng


def dominant_columns(fes, rng, n):
    """Regressors around the certificate's dominant-origin boundary: x = m + z with m
    solving sum*(sum/n) = (1 - 1/1024) raw exactly for the drawn z, then m(1 +- 10^-k);
    a nearly constant column; a column of the span plus 1e-10 noise; a zero column."""
    cols = []
    for k in (3, 9, 13):
        z = rng.normal(size=n)
        z1, z2 = z.sum(), (z * z).sum()
        c = 1.0 - 1.0 / 1024.0
        # (n m + z1)^2 / n = c (n m^2 + 2 m z1 + z2): a m^2 + b m + d = 0
        a, b, d = n - c * n, 2 * z1 - 2 * c * z1, z1 * z1 / n - c * z2
        m = (-b + np.sqrt(b * b - 4 * a * d)) / (2 * a)
        for sign in (-1.0, 1.0):
            cols.append(m * (1.0 + sign * 10.0 ** (-k)) + z)
    cols.append(1.0e6 + rng.normal(size=n))
    span = np.zeros(n)
    for f in fes:
        span += rng.normal(size=int(f.max()) + 1)[f]
    cols.append(span + 1.0e-10 * rng.normal(size=n))
    cols.append(np.zeros(n))
    return np.asfortranarray(np.column_stack(cols))


def ctor(module, **kw):
    base = dict(se_type="cluster", tol=1e-8, max_iter=100000, fit_intercept=True, num_threads=16,
                drop_singletons=True, symmetric_sweep=False, absorption_method="auto",
                tolerance_mode="reghdfe-comparable")
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
            except Exception as exc:
                public[name] = {"raises": type(exc).__name__}
        record.update(public=public, missing=missing,
                      diagnostics={k: getattr(reg, k) for k in DIAGNOSTICS if hasattr(reg, k)},
                      method=int(reg.absorption_method_used), rel=float(reg.abs_residual_rel_),
                      certified=bool(reg.precision_certified_),
                      coef=np.asarray(reg.coef_).tolist(), cov=np.asarray(reg.covariance_).tolist())
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["lines"] = [l for l in (log.read_text().splitlines() if log.exists() else [])
                       if "reduced_pcg event=" in l or "cache_hit=" in l]
    (out / f"{label}.json").write_text(json.dumps(record, sort_keys=True))


def scenario_panel(module, out):
    y, X, fes, rng = panel()
    common = (y, X)
    kw = dict(fes=fes, clusters=fes[0])
    run_fit(module, out, "tier", dict(), common, kw)
    run_fit(module, out, "tier_pcgoff", dict(reduced_pcg=False), common, kw)
    for tol in (1.3e-12, 1.2e-12, 1e-14):
        run_fit(module, out, f"tol_{tol:.1e}", dict(tol=tol), common, kw)
    run_fit(module, out, "fast", dict(tolerance_mode="xhdfe-fast"), common, kw)
    run_fit(module, out, "weights", dict(), common, dict(kw, weights=np.ones(y.size)))
    run_fit(module, out, "savefe", dict(retain_fes=True, se_type="robust"), common, dict(fes=fes))
    run_fit(module, out, "slopes", dict(se_type="robust"), common,
            dict(fes=fes, slopes=[(1, rng.normal(size=y.size), True)]))
    run_fit(module, out, "iv", dict(), common,
            dict(kw, instruments=np.asfortranarray((X[:, 1] + 0.5 * X[:, 0]).reshape(-1, 1)), endogenous_idx=[1]))
    run_fit(module, out, "method", dict(absorption_method="mlsmr"), common, kw)
    g = np.repeat(np.arange(20_000), 3)
    ind = rng.integers(0, 9_000, size=g.size).astype(np.int64)
    gy = rng.normal(size=20_000)[g]
    gx = np.asfortranarray(rng.normal(size=(20_000, 1))[g])
    gfe = ((np.arange(20_000) % 300)[g]).astype(np.int32)
    run_fit(module, out, "group", dict(se_type="robust"), (gy, gx),
            dict(fes=[gfe, ind.astype(np.int32)], group=g.astype(np.int64), individual=ind, aggregation="mean"))


def scenario_dominant(module, out):
    y, X, fes, rng = panel(seed=7, workers=20_000, firms=1_500)
    D = dominant_columns(fes, rng, y.size)
    Xd = np.asfortranarray(np.column_stack([X[:, :2], D]))
    run_fit(module, out, "dominant", dict(se_type="robust"), (y, Xd), dict(fes=fes))
    run_fit(module, out, "dominant_pcgoff", dict(se_type="robust", reduced_pcg=False), (y, Xd), dict(fes=fes))


def scenario_declines(module, out):
    y, X, fes, _ = panel(seed=5, workers=15_000, firms=1_200)
    run_fit(module, out, "declined", dict(), (y, X), dict(fes=fes, clusters=fes[0]))


def scenario_cache(module, out, root: Path, step: str):
    mode, name, *tol = step.split(":")  # an optional third field: the public tolerance
    os.environ["XHDFE_ABSORPTION_CACHE"] = str(root / name)
    os.environ["XHDFE_ABSORPTION_CACHE_MODE"] = mode
    y, X, fes, _ = panel(seed=9, workers=12_000, firms=1_000)
    run_fit(module, out, "cache", dict(tol=float(tol[0])) if tol else dict(), (y, X), dict(fes=fes, clusters=fes[0]))


SCENARIOS = {"panel": scenario_panel, "dominant": scenario_dominant, "declines": scenario_declines}


def child(args) -> int:
    module = load(args.module)
    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    if args.scenario == "cache":
        scenario_cache(module, out, Path(args.out), args.step)
    else:
        SCENARIOS[args.scenario](module, out)
    return 0


def launch(script, module, out, label, scenario, step="", extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("XHDFE_")}
    env.update({"XHDFE_GPU_BACKEND": "cpu", "XHDFE_ABSORPTION_CACHE_MODE": "off", "XHDFE_FE_STRUCTURE_MODE": "off",
                "XHDFE_MOBILITY_MODE": "off", "XHDFE_REDUCED_PCG_TRACE": "2", "XHDFE_DEBUG_SOLVER": "1",
                "OMP_WAIT_POLICY": "passive"})
    if extra:
        env.update(extra)
    command = [sys.executable, "-B", "-I", str(script), "--child", "--module", str(module), "--scenario", scenario,
               "--label", label, "--out", str(out), "--step", step]
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{label}: child failed ({done.returncode}): {done.stderr[-2000:]}")
    return out / label


def records(folder: Path):
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


COLUMN = re.compile(r"event=column col=(\d+) iterations=(\d+) first_tau_pass=-?\d+ reason=(\w+) tau=(\S+) "
                    r"chat=(\S+) .*? rel=(\S+)")


def columns(r):
    return [COLUMN.search(l).groups() for l in r.get("lines", []) if COLUMN.search(l)]


def plan_chat_stop(r):
    for l in r.get("lines", []):
        if "event=plan" in l:
            m = re.search(r"chat_stop=(\S+)", l)
            return float(m.group(1)) if m else 0.0
    return None


def parent(args) -> int:
    script = Path(__file__).resolve()
    failures = []

    def fail(message):
        failures.append(message)
        print("FAIL", message, flush=True)

    def same(a, b, what, diagnostics=True):
        if a.get("error") or b.get("error"):
            if a.get("error") != b.get("error"):
                fail(f"{what}: errors differ: {a.get('error')} != {b.get('error')}")
            return
        diff = sorted(k for k in REQUIRED if a["public"].get(k) != b["public"].get(k))
        if diff:
            fail(f"{what}: differs in {diff}")
        if a.get("method") != b.get("method") or (diagnostics and a.get("diagnostics") != b.get("diagnostics")):
            fail(f"{what}: method {a.get('method')}/{b.get('method')}, diagnostics {a.get('diagnostics')} / "
                 f"{b.get('diagnostics')}")

    def within(a, b, what):
        ba, bb = np.asarray(a["coef"]), np.asarray(b["coef"])
        Va, Vb = np.asarray(a["cov"]), np.asarray(b["cov"])
        scale = np.sqrt(np.outer(np.diag(Vb), np.diag(Vb)))
        db = float(np.max(np.abs(ba - bb) / np.maximum(1.0, np.abs(bb)))) if ba.size else 0.0
        dV = float(np.max(np.abs(Va - Vb) / np.where(scale > 0, scale, 1.0))) if Va.size else 0.0
        if not (db <= 1e-9 and dV <= 1e-8):
            fail(f"{what}: |db|/max(1,|b|) {db:.3e}, |dV|/sqrt(VjjVkk) {dV:.3e}")
        return db, dV

    def published(r, what):
        """Every Chat stop at or below 1e-12; a published rel above 1e-12 explained."""
        cols = columns(r)
        for c, it, why, tau, chat, rel in cols:
            if why == "chat" and not (float(chat) <= 1e-12):
                fail(f"{what}: col {c} stopped on chat with chat {chat}")
            if why == "test2":
                fail(f"{what}: col {c} stopped on test 2 with D10 on")
        rel = r.get("rel", float("nan"))
        note = ""
        if rel > 1e-12:
            worst = max(cols, key=lambda t: float(t[5])) if cols else None
            if worst and worst[2] in ("span", "test1", "zero"):
                note = f"explained: col {worst[0]} stopped by {worst[2]}"
            elif worst and worst[2] == "chat":
                note = f"explained: certificate rounding, col {worst[0]} rel {worst[5]} against chat {worst[4]}"
            else:
                fail(f"{what}: published rel {rel:.3e} above 1e-12 without explanation")
        ratios = [float(rl) / float(ch) for _, _, why, _, ch, rl in cols if why == "chat" and float(ch) > 0]
        print(f"INFO {what}: rel {rel:.4e} method {r.get('method')} certified {r.get('certified')} stops "
              f"{sorted({w for _, _, w, _, _, _ in cols})} max rel/chat at chat stops "
              f"{max(ratios) if ratios else float('nan'):.5f} {note}", flush=True)
        if r.get("method") == 8 and not (r.get("certified") and rel <= 1e-11):
            fail(f"{what}: a tier result not certified or above the trigger ({rel:.3e})")

    def three(scenario, extra=None, tag=""):
        arms = {}
        for arm, env in (("on", {}), ("off", {"XHDFE_REDUCED_PCG_CHAT_STOP": "0"}), ("base", None)):
            module = args.frozen_module if env is None else args.module
            arms[arm] = records(launch(script, module, out, f"{scenario}{tag}_{arm}", scenario,
                                       extra={**(extra or {}), **(env or {})}))
        on, off, base = arms["on"], arms["off"], arms["base"]
        if not on:
            fail(f"{scenario}{tag}: no records")
        for label in sorted(on):
            same(off[label], base[label], f"{scenario}{tag}/{label}: off against the frozen phase chain")
        return on, off, base

    only = set(args.only.split(",")) if args.only else None
    wanted = lambda name: only is None or name in only  # noqa: E731
    with tempfile.TemporaryDirectory(prefix="chat_stop_contract_") as tmp:
        out = Path(tmp)
        if wanted("panel"):
            # Correction E1 (Verifications/low_memory_20261116/briefs/E1_SAVEFE_COMPARABLE_PROJECTION_FIRST.md):
            # a fit with saved effects whose capturing sweep stops above the floor of the mode is followed by
            # the projection of the mode, which the tier may publish. The savefe fit of this scenario checks the
            # former rule (the capturing sweep never enters the tier: D10 cannot act, off equals the frozen
            # chain); XHDFE_SAVEFE_ACCURATE_PROJECTION=0 keeps that path. No other fit here saves effects, so
            # none reads the switch, and the frozen module does not know it.
            on, off, _ = three("panel", extra={"XHDFE_SAVEFE_ACCURATE_PROJECTION": "0"})
            for label in ("tier_pcgoff", "tol_1.2e-12", "tol_1.0e-14", "fast", "weights", "savefe", "slopes", "iv",
                          "method", "group"):
                same(on[label], off[label], f"panel/{label}: on against off (D10 cannot act)")
            for label in ("tier", "tol_1.3e-12"):
                published(on[label], f"panel/{label}")
                within(on[label], off[label], f"panel/{label}: on against off")
                within(on[label], on["tier_pcgoff"], f"panel/{label}: on against reducedpcg(off)")
            for label, want in (("tier", 1e-12), ("tol_1.3e-12", 1e-12), ("tol_1.2e-12", 0.0), ("tol_1.0e-14", 0.0)):
                got = plan_chat_stop(on[label])
                if got is None or abs(got - want) > 1e-24:
                    fail(f"panel/{label}: plan chat_stop {got}, want {want}")
        if wanted("dominant"):
            on, off, _ = three("dominant")
            published(on["dominant"], "dominant")
            within(on["dominant"], off["dominant"], "dominant: on against off")
            within(on["dominant"], on["dominant_pcgoff"], "dominant: on against reducedpcg(off)")
            same(on["dominant_pcgoff"], off["dominant_pcgoff"], "dominant_pcgoff: on against off")
            for c, it, why, tau, chat, rel in columns(on["dominant"]):
                ratio = float(rel) / float(chat) if float(chat) > 0 else float("nan")
                print(f"INFO dominant col {c}: {why} at {it}, chat {float(chat):.4e}, rel {float(rel):.4e}, "
                      f"rel/chat {ratio:.5f}", flush=True)
        if wanted("declines"):
            for stage in STAGES:
                on, off, _ = three("declines", extra={"XHDFE_TEST_REDUCED_PCG_DECLINE": stage}, tag=f"_{stage}")
                same(on["declined"], off["declined"], f"declines/{stage}: result on against off", diagnostics=False)
                print(f"INFO declines/{stage}: diagnostics on {on['declined'].get('diagnostics')} off "
                      f"{off['declined'].get('diagnostics')}", flush=True)
        if wanted("cache"):
            def leg(arm_env, who, step):
                folder = launch(script, args.module, out, f"cache_{who}_{step.replace(':', '_')}", "cache", step=step,
                                extra=arm_env)
                return records(folder)["cache"]
            on_env, off_env = {}, {"XHDFE_REDUCED_PCG_CHAT_STOP": "0"}
            w_on = leg(on_env, "on", "write:on.cache")
            w_off = leg(off_env, "off", "write:off.cache")
            hit = lambda r: any("cache_hit=1" in l for l in r.get("lines", []))  # noqa: E731
            r_on_on = leg(on_env, "on", "read:on.cache")
            r_off_off = leg(off_env, "off", "read:off.cache")
            r_off_on = leg(off_env, "off", "read:on.cache")
            r_on_off = leg(on_env, "on", "read:off.cache")
            if not (hit(r_on_on) and hit(r_off_off)):
                fail(f"cache: a state did not read its own entry (on {hit(r_on_on)}, off {hit(r_off_off)})")
            same(r_on_on, w_on, "cache: on reads its own entry")
            same(r_off_off, w_off, "cache: off reads its own entry")
            if hit(r_off_on) or hit(r_on_off):
                fail(f"cache: crossed read hit (off reads on {hit(r_off_on)}, on reads off {hit(r_on_off)})")
            same(r_off_on, w_off, "cache: off with the on entry present computes as off", diagnostics=False)
            same(r_on_off, w_on, "cache: on with the off entry present computes as on", diagnostics=False)
            # The scope reaches the key exactly as it reaches the stop: at tol 1.3e-12 (certificate limit
            # 1.04e-11 >= the 1e-11 trigger) the entries of the two states never cross; at 1.2e-12 (limit
            # 9.6e-12) D10 does not apply, the key is today's and each state reads the other's entry bit for bit.
            for tol, crosses in (("1.3e-12", False), ("1.2e-12", True)):
                w_on_t = leg(on_env, "on", f"write:on_{tol}.cache:{tol}")
                w_off_t = leg(off_env, "off", f"write:off_{tol}.cache:{tol}")
                r_off_on_t = leg(off_env, "off", f"read:on_{tol}.cache:{tol}")
                r_on_off_t = leg(on_env, "on", f"read:off_{tol}.cache:{tol}")
                if hit(r_off_on_t) != crosses or hit(r_on_off_t) != crosses:
                    fail(f"cache tol {tol}: crossed reads hit (off reads on {hit(r_off_on_t)}, on reads off "
                         f"{hit(r_on_off_t)}), expected {crosses}")
                if crosses:
                    same(w_on_t, w_off_t, f"cache tol {tol}: the two states' fresh fits", diagnostics=False)
                    same(r_off_on_t, w_on_t, f"cache tol {tol}: off reads the on entry", diagnostics=False)
                    same(r_on_off_t, w_off_t, f"cache tol {tol}: on reads the off entry", diagnostics=False)
                else:
                    same(r_off_on_t, w_off_t, f"cache tol {tol}: off with the on entry computes as off",
                         diagnostics=False)
                    same(r_on_off_t, w_on_t, f"cache tol {tol}: on with the off entry computes as on",
                         diagnostics=False)
                print(f"INFO cache tol {tol}: crossed reads hit {hit(r_off_on_t)}/{hit(r_on_off_t)} "
                      f"(expected {crosses})", flush=True)
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
    args = parser.parse_args()
    if args.child:
        return child(args)
    if not args.frozen_module:
        parser.error("--frozen-module is required")
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
