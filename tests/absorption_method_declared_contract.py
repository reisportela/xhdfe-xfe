#!/usr/bin/env python3
"""Point 14 (17nov2026): the method a fit declares when the Krylov-PCG route is forced.

AbsorptionMethod gains KrylovPcg (code 9, public name krylov-pcg, Python AbsorptionMethod.krylov_pcg),
reported only, never a request. A fit declares it when, and only when, its route was forced to Krylov
(the regressor's use_krylov in force: XHDFE_USE_KRYLOV=1 here, or a mobility profile that suggests it)
and the Krylov-PCG absorber (absorb_fixed_effects_krylov) produced the transform the fit publishes
(AbsorptionResult::krylov_pcg_used). A result that another route served or replaced declares that
route's method, as before; the automatic Jacobi-PCG gate, a default route, declares what it declared;
the absorption cache stores the route's method as before (a hit on a Krylov-produced transform
declares the stored sweep code with reason "cache").

Builds nothing; never run during a timing campaign. Synthetic data only, made in each child from fixed
seeds; one fresh process per arm, CPU backend, profiles, caches and the team registry off unless an arm
sets them, 4 explicit threads.

    python3 -I tests/absorption_method_declared_contract.py --module <module> --frozen-module <module> \\
        [--only NAME[,NAME]]

Checked in every arm, against the frozen module (the build before code 9): every published number
(b, V, standard errors, residuals, rss, iterations, abs_residual_rel, saved effects) bit for bit, the
reduced tier's status and reason, and the declared method: 9 (with summary() printing krylov-pcg) in
the forced arms whose transform the Krylov-PCG absorber produced, the frozen module's code everywhere
else. Also: the enum values 0..8 unchanged and krylov_pcg = 9; krylov-pcg and krylov_pcg as a request
rejected like reduced-pcg; the absorption-cache file written under the forced route byte-identical
to the frozen module's, and its hit reporting the frozen module's values.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

KRYLOV = {"XHDFE_USE_KRYLOV": "1"}
# The automatic Schwarz/Jacobi-PCG gate (fe_absorption.cpp, absorb_fixed_effects_v6) on small data:
# the default auto-MLSMR selector, which otherwise consumes from_auto for up to three effects, off;
# the gate's size and difficulty thresholds at their minimum; its decision printed.
GATE = {"XHDFE_AUTO_MLSMR_DEFAULT_MAX_FES": "1", "XHDFE_SCHWARZ_NMIN": "1", "XHDFE_SCHWARZ_MSWITCH": "0",
        "XHDFE_SCHWARZ_DIAG": "1"}
# name: (design, tolerance mode, saved effects, weights, environment, expectation)
# expectation: "krylov" (declares 9), "frozen" (declares the frozen module's code)
ARMS = {
    "forced_fast_w": ("w", "xhdfe-fast", False, None, KRYLOV, "krylov"),
    "forced_comparable_w": ("w", "reghdfe-comparable", False, None, KRYLOV, "krylov"),
    "forced_fast_lowmob": ("lowmob", "xhdfe-fast", False, None, KRYLOV, "krylov"),
    "forced_comparable_lowmob": ("lowmob", "reghdfe-comparable", False, None, KRYLOV, "krylov"),
    "forced_weighted_w": ("w", "reghdfe-comparable", False, "aweights", KRYLOV, "krylov"),
    "plain_fast_w": ("w", "xhdfe-fast", False, None, {}, "frozen"),
    "plain_comparable_w": ("w", "reghdfe-comparable", False, None, {}, "frozen"),
    "plain_fast_lowmob": ("lowmob", "xhdfe-fast", False, None, {}, "frozen"),
    "plain_comparable_lowmob": ("lowmob", "reghdfe-comparable", False, None, {}, "frozen"),
    "unforced_variable_zero": ("w", "xhdfe-fast", False, None, {"XHDFE_USE_KRYLOV": "0"}, "frozen"),
    # Served by another route although forced: the exact projection of a weighted 2x2 design.
    "forced_served_2x2": ("w2x2", "reghdfe-comparable", False, "aweights", KRYLOV, "frozen"),
    # Replaced: the automatic gate's Krylov probe (one iteration) bails and MLSMR publishes.
    "forced_replaced_gate_mlsmr": ("lowmob", "xhdfe-fast", False, None,
                                   {**KRYLOV, **GATE, "XHDFE_KRYLOV_PROBE": "1"}, "frozen"),
    # The automatic Jacobi-PCG gate, not forced: today's code.
    "gate_jacobi_pcg": ("w", "xhdfe-fast", False, None, GATE, "frozen"),
    # A1 and A2 together: saved effects in reghdfe-comparable under the forced route.
    "forced_savefe_comparable": ("lowmob", "reghdfe-comparable", True, None, KRYLOV, "krylov"),
    "plain_savefe_comparable": ("lowmob", "reghdfe-comparable", True, None, {}, "frozen"),
}


def load(path):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def design(kind):
    import numpy as np
    if kind == "w2x2":
        rng = np.random.default_rng(141)
        n = 2000
        a = (np.arange(n) % 2).astype(np.int32)
        b = ((np.arange(n) // 2) % 2).astype(np.int32)
        X = np.asfortranarray(rng.normal(size=(n, 2)))
        y = X @ np.array([0.5, -0.3]) + 0.7 * a - 0.4 * b + rng.normal(size=n)
        return dict(y=y, X=X, fes=[a, b])
    if kind == "lowmob":
        rng = np.random.default_rng(20261117)
        workers, years, firms = 20_000, 8, 2_000
        n = workers * years
        worker = np.repeat(np.arange(workers, dtype=np.int32), years)
        home = rng.integers(0, firms, workers, dtype=np.int32)
        firm = np.repeat(home, years)
        move = rng.random(n) < 0.03
        firm = np.where(move, rng.integers(0, firms, n, dtype=np.int32), firm).astype(np.int32)
        year = np.tile(np.arange(years, dtype=np.int32), workers)
        X = np.empty((n, 3), dtype=np.float64, order="F")
        for j in range(3):
            X[:, j] = rng.standard_normal(n)
        a_w, a_f = rng.standard_normal(workers), rng.standard_normal(firms)
        X[:, 0] += 0.4 * a_w[worker] + 0.4 * a_f[firm]
        y = 0.5 * X[:, 0] - 0.25 * X[:, 1] + a_w[worker] + a_f[firm] + year / 8.0 + rng.standard_normal(n)
        return dict(y=y, X=X, fes=[worker, firm, year])
    rng = np.random.default_rng(20261116)  # w: ten rows per worker, firms at random, three effects
    n, workers = 60_000, 6_000
    worker = np.repeat(np.arange(workers, dtype=np.int32), 10)
    year = np.tile(np.arange(10, dtype=np.int32), workers)
    firm = rng.integers(0, n // 200, n, dtype=np.int32)
    X = np.empty((n, 4), dtype=np.float64, order="F")
    for j in range(4):
        X[:, j] = rng.standard_normal(n)
    a_w, a_f = rng.standard_normal(workers), rng.standard_normal(n // 200)
    y = rng.standard_normal(n) + a_w[worker] + a_f[firm] + year / 10.0
    for j in range(3):
        X[:, j] += 0.3 * a_w[worker] + 0.3 * a_f[firm]
        y += (0.5 - 0.25 * j) * X[:, j]
    return dict(y=y, X=X, fes=[worker, firm, year])


def digest(reg, savefe):
    import numpy as np
    h = hashlib.sha256()
    for name in ("coef_", "covariance_", "stderr_", "residuals_", "rss_", "num_iterations_", "abs_residual_rel_"):
        a = np.ascontiguousarray(np.asarray(getattr(reg, name)))
        h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    if savefe:
        for e in reg.fe_effects_:
            h.update(np.ascontiguousarray(np.asarray(e)).tobytes())
    return h.hexdigest()[:24]


def child(args):
    import numpy as np
    core = load(args.child)
    out = {}
    if args.arm == "requests":
        # The enum, and krylov-pcg as a request.
        names = ["auto", "gauss_seidel", "symmetric_gauss_seidel", "jacobi", "schwarz", "lsmr", "mlsmr",
                 "auto_mlsmr", "reduced_pcg", "krylov_pcg"]
        out["enum"] = {n: int(getattr(core.AbsorptionMethod, n)) for n in names if hasattr(core.AbsorptionMethod, n)}
        out["rejections"] = {}
        for name in ("reduced-pcg", "krylov-pcg", "krylov_pcg"):
            try:
                core.HdfeRegressor(absorption_method=name)
                out["rejections"][name] = None
            except Exception as exc:  # noqa: BLE001
                out["rejections"][name] = [type(exc).__name__, str(exc)]
        print("RESULT " + json.dumps(out), flush=True)
        return
    kind, mode, savefe, weights, _, _ = ARMS.get(args.arm, ARMS.get(args.arm.split(":")[0], (None,) * 6))
    if args.arm.startswith("cache_"):
        kind, mode, savefe, weights = "w", "reghdfe-comparable", False, None
    d = design(kind)
    reg = core.HdfeRegressor(se_type="unadjusted", tol=1e-8, max_iter=100000, num_threads=4,
                             symmetric_sweep=False, absorption_method="auto", tolerance_mode=mode,
                             retain_fes=savefe, drop_singletons=True)
    fit_kw = {}
    if weights:
        fit_kw["weights"] = np.random.default_rng(31).uniform(0.5, 2.0, size=d["y"].size)
    reg.fit(d["y"], d["X"], fes=d["fes"], **fit_kw)
    out.update(method=int(reg.absorption_method_used), method_name=str(reg.absorption_method_used),
               status=int(reg.reduced_pcg_status_), reason=str(reg.reduced_pcg_reason_),
               iterations=int(reg.num_iterations_), digest=digest(reg, savefe),
               summary_krylov="Absorption: krylov-pcg" in str(reg.summary()))
    print("RESULT " + json.dumps(out), flush=True)


def run(script, module, arm, extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("XHDFE_") and k != "LD_LIBRARY_PATH"}
    env.update(XHDFE_GPU_BACKEND="cpu", XHDFE_MOBILITY_MODE="off", XHDFE_ABSORPTION_CACHE_MODE="off",
               XHDFE_FE_STRUCTURE_MODE="off", XHDFE_AUTO_TEAM_COORDINATION="0", XHDFE_DEBUG_SOLVER="1",
               OMP_WAIT_POLICY="passive")
    env.update(extra)
    done = subprocess.run([sys.executable, "-B", "-I", str(script), "--child", str(module), "--arm", arm],
                          env=env, capture_output=True, text=True)
    lines = [l for l in done.stdout.splitlines() if l.startswith("RESULT ")]
    result = json.loads(lines[0][7:]) if lines else {"error": f"rc={done.returncode} {done.stderr[-800:]}"}
    result["stderr"] = [l for l in done.stderr.splitlines()
                        if l.startswith("xhdfe solver:") or l.startswith("[schwarz-gate]")]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child")
    ap.add_argument("--arm")
    ap.add_argument("--module")
    ap.add_argument("--frozen-module")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.child:
        child(a)
        return 0
    if not a.module or not a.frozen_module:
        ap.error("--module and --frozen-module are required")
    script = Path(__file__).resolve()
    only = set(filter(None, a.only.split(",")))
    failures = []

    def fail(msg):
        failures.append(msg)
        print("FAIL", msg, flush=True)

    for name, (kind, mode, savefe, weights, extra, expect) in ARMS.items():
        if only and name not in only:
            continue
        new, old = run(script, a.module, name, extra), run(script, a.frozen_module, name, extra)
        if "error" in new or "error" in old:
            fail(f"{name}: error new={new.get('error')} frozen={old.get('error')}")
            continue
        same = all(new[k] == old[k] for k in ("digest", "status", "reason", "iterations"))
        if not same:
            fail(f"{name}: published numbers or tier report differ from frozen: {new} / {old}")
        if expect == "krylov":
            if new["method"] != 9 or new["method_name"] != "AbsorptionMethod.krylov_pcg" or not new["summary_krylov"]:
                fail(f"{name}: declared {new['method']} {new['method_name']} (summary krylov-pcg "
                     f"{new['summary_krylov']}), want 9")
            if not any("use_krylov=1" in l for l in new["stderr"]):
                fail(f"{name}: the Krylov route was not forced ({new['stderr']})")
        elif new["method"] != old["method"] or new["summary_krylov"]:
            fail(f"{name}: declared {new['method']}, frozen {old['method']}")
        if name == "forced_replaced_gate_mlsmr" and not any("MLSMR (PCG-bail" in l for l in new["stderr"]):
            fail(f"{name}: the gate's Krylov probe did not bail to MLSMR ({new['stderr']})")
        if name == "gate_jacobi_pcg" and not any("JACOBI-PCG" in l for l in new["stderr"]):
            fail(f"{name}: the automatic Jacobi-PCG gate was not taken ({new['stderr']})")
        print(f"INFO {name}: declared {new['method']} ({new['method_name']}) frozen {old['method']}; "
              f"report {new['status']}/{new['reason']} iterations {new['iterations']}; numbers "
              f"{'bit-identical' if same else 'DIFFER'}; {' | '.join(new['stderr'])[:160]}", flush=True)

    if not only or "requests" in only:
        new, old = run(script, a.module, "requests", {}), run(script, a.frozen_module, "requests", {})
        want = {"auto": 0, "gauss_seidel": 1, "symmetric_gauss_seidel": 2, "jacobi": 3, "schwarz": 4, "lsmr": 5,
                "mlsmr": 6, "auto_mlsmr": 7, "reduced_pcg": 8, "krylov_pcg": 9}
        if new.get("enum") != want:
            fail(f"requests: enum {new.get('enum')} != {want}")
        if {k: v for k, v in old.get("enum", {}).items()} != {k: v for k, v in want.items() if k != "krylov_pcg"}:
            fail(f"requests: the frozen enum moved? {old.get('enum')}")
        rej = new.get("rejections", {})
        ref = rej.get("reduced-pcg")
        for name in ("krylov-pcg", "krylov_pcg"):
            got = rej.get(name)
            if not got or not ref or got[0] != ref[0] or "krylov-pcg is reported" not in got[1] or \
                    "cannot be requested" not in got[1]:
                fail(f"requests: {name} not rejected like reduced-pcg: {got} / {ref}")
        print(f"INFO requests: enum {new.get('enum')}; rejections {rej}; frozen rejections {old.get('rejections')}",
              flush=True)

    if not only or "cache" in only:
        with tempfile.TemporaryDirectory(prefix="absorption_method_declared_") as tmp:
            files = {}
            reports = {}
            for arm, module in (("new", a.module), ("frozen", a.frozen_module)):
                path = Path(tmp) / f"{arm}.cache"
                for step in ("write", "read"):
                    extra = {**KRYLOV, "XHDFE_ABSORPTION_CACHE": str(path), "XHDFE_ABSORPTION_CACHE_MODE": step}
                    reports[(arm, step)] = run(script, module, f"cache_{step}", extra)
                files[arm] = path.read_bytes() if path.exists() else b""
            if not files["new"] or files["new"] != files["frozen"]:
                fail(f"cache: the files differ ({len(files['new'])} / {len(files['frozen'])} bytes)")
            for step in ("write", "read"):
                n, o = reports[("new", step)], reports[("frozen", step)]
                if "error" in n or "error" in o:
                    fail(f"cache {step}: error {n.get('error')} / {o.get('error')}")
                    continue
                if any(n[k] != o[k] for k in ("digest", "status", "reason", "iterations")):
                    fail(f"cache {step}: differs from frozen: {n} / {o}")
            nw, nr, orr = reports[("new", "write")], reports[("new", "read")], reports[("frozen", "read")]
            if nw.get("method") != 9:
                fail(f"cache write: the forced fit declared {nw.get('method')}, want 9")
            if nr.get("method") != orr.get("method") or nr.get("reason") != "cache":
                fail(f"cache read: declared {nr.get('method')} reason {nr.get('reason')}, frozen "
                     f"{orr.get('method')} {orr.get('reason')}")
            if not any("cache_hit=1" in l for l in nr.get("stderr", [])):
                fail(f"cache read: no hit ({nr.get('stderr')})")
            print(f"INFO cache: file {len(files['new'])} bytes, sha256 "
                  f"{hashlib.sha256(files['new']).hexdigest()[:16]} new = frozen {files['new'] == files['frozen']}; "
                  f"write declared {nw.get('method')} (frozen {reports[('frozen', 'write')].get('method')}); "
                  f"hit declared {nr.get('method')} reason {nr.get('reason')} (frozen {orr.get('method')} "
                  f"{orr.get('reason')})", flush=True)
    print("METHOD_DECLARED_CONTRACT_PASS" if not failures else f"METHOD_DECLARED_CONTRACT_FAIL {len(failures)} failures",
          flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
