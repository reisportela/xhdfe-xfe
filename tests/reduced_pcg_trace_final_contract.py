#!/usr/bin/env python3
"""Point 13 (17nov2026): the final line of the reduced tier's trace in a fit with saved effects.

Correction E1 (publish_savefe_projection, src/hdfe_regressor_v11.cpp) recomputes the projection of a
reghdfe-comparable fit with saved effects whose capturing sweep stopped above the floor of the mode.
Before this correction the fit's only "xhdfe reduced_pcg event=final" line was the capturing sweep's
"status=ineligible reason=savefe", whatever the fit then reported. Now, with XHDFE_REDUCED_PCG_TRACE
on, every public fit still prints exactly one final line, and:
  published   when E1 publishes the projection, the line is the projection's attempt, the one the same
              fit without saved effects prints (accepted, declined with its reason, ineligible with its
              reason), and its status and reason are those the fit reports;
  untouched   when E1 does not attempt the projection (capturing sweep at or below the floor, E1 off,
              a reason before "savefe" in the eligibility chain such as weights) and in every fit
              without saved effects, the tier's trace lines are those of the frozen module, byte for
              byte once the seconds= values are masked;
  refused     a fit that ends without estimates prints exactly one final line, the frozen module's.
In every fit the line agrees with the fit's report (reduced_pcg_status_, reduced_pcg_reason_), and
every published number (b, V, standard errors, residuals, rss, iterations, abs_residual_rel, saved
effects, the declared method) is bit for bit the frozen module's (CPU).

Builds nothing; never run during a timing campaign. Synthetic data only, made in each child from
fixed seeds; one fresh process per scenario (the trace is read once per process), CPU backend, caches,
profiles and the team registry off, 4 explicit threads.

    python3 -I tests/reduced_pcg_trace_final_contract.py --module <module> --frozen-module <module> \\
        [--cuda-module <module> --frozen-cuda-module <module>] [--only NAME[,NAME]]

With --cuda-module, two CUDA scenarios run as well (the CUDA tier requested per fit,
reduced_pcg_cuda=True): the projection published by the CUDA tier (line against report), and the
fatal error of the CUDA tier inside E1's projection (XHDFE_TEST_REDUCED_PCG_CUDA_FATAL=1: the fit
ends without estimates; one final line, the frozen module's). CUDA does not repeat bits: there only
the lines and the reports are compared.
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

FINAL = "xhdfe reduced_pcg event=final "
TRACE = "xhdfe reduced_pcg "
SECONDS = re.compile(r"seconds=\S+")
MARK = "TRACE_FINAL_CONTRACT_FIT "

# name: (fits [(design, saved effects, weights)], extra environment, kind of each fit, backend)
# kinds: "published" (E1 publishes the projection), "untouched" (byte equality with frozen),
# "refused" (no estimates; the frozen module's single line), "plain" (no saved effects: untouched).
SCENARIOS = {
    "lowmob_savefe": ([("lowmob", True, None)], {}, ["published"], "cpu"),
    "lowmob_plain": ([("lowmob", False, None)], {}, ["plain"], "cpu"),
    "pathlike_savefe": ([("pathlike", True, None)], {}, ["published"], "cpu"),
    "pathlike_plain": ([("pathlike", False, None)], {}, ["plain"], "cpu"),
    "weighted_savefe": ([("lowmob", True, "aweights")], {}, ["untouched"], "cpu"),
    "weighted_plain": ([("lowmob", False, "aweights")], {}, ["plain"], "cpu"),
    "zero_weights_savefe": ([("lowmob", True, "zeros")], {}, ["untouched"], "cpu"),
    "below_floor_savefe": ([("w", True, None)], {}, ["untouched"], "cpu"),
    "below_floor_plain": ([("w", False, None)], {}, ["plain"], "cpu"),
    "e1_off_savefe": ([("lowmob", True, None)], {"XHDFE_SAVEFE_ACCURATE_PROJECTION": "0"}, ["untouched"], "cpu"),
    "e1_off_plain": ([("lowmob", False, None)], {"XHDFE_SAVEFE_ACCURATE_PROJECTION": "0"}, ["plain"], "cpu"),
    "krylov_savefe": ([("lowmob", True, None)], {"XHDFE_USE_KRYLOV": "1"}, ["published"], "cpu"),
    "krylov_plain": ([("lowmob", False, None)], {"XHDFE_USE_KRYLOV": "1"}, ["plain"], "cpu"),
    "recovery_refused": ([("lowmob_refused", True, None)], {}, ["refused"], "cpu"),
    "sequence": ([("lowmob", True, None), ("lowmob", False, None), ("pathlike", True, None),
                  ("w", True, None), ("lowmob", True, "aweights"), ("lowmob", True, None)], {},
                 ["published", "plain", "published", "untouched", "untouched", "published"], "cpu"),
    "cuda_tier_savefe": ([("cuda_panel", True, None)], {}, ["cuda_published"], "cuda"),
    "cuda_tier_fatal": ([("cuda_panel", True, None)], {"XHDFE_TEST_REDUCED_PCG_CUDA_FATAL": "1"},
                        ["refused"], "cuda"),
}
# The fit without saved effects whose final line a published fit must print.
PLAIN_OF = {"lowmob_savefe": "lowmob_plain", "pathlike_savefe": "pathlike_plain",
            "krylov_savefe": "krylov_plain"}


def load(path):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def design(kind):
    """The designs of Verifications/corrigir_erros_20261117/levantamento/p13_p14_repro.py."""
    import numpy as np
    if kind == "pathlike":
        n = 12000
        m = (n - 2) // 2
        k = np.arange(n - 2)
        fe_a = np.r_[k // 2, [0, m // 3]].astype(np.int32)
        fe_b = np.r_[((k + 1) // 2) % m, [m // 2, (2 * m) // 3]].astype(np.int32)
        rng = np.random.default_rng(20261109)
        alpha, beta = rng.normal(size=m), rng.normal(size=m)
        x = rng.normal(size=n) + 0.3 * alpha[fe_a] - 0.2 * beta[fe_b]
        y = 0.5 * x + alpha[fe_a] + beta[fe_b] + 0.1 * rng.normal(size=n)
        return dict(y=y, X=np.asfortranarray(x.reshape(-1, 1)), fes=[fe_a, fe_b])
    if kind in ("lowmob", "lowmob_refused", "cuda_panel"):
        rng = np.random.default_rng(20261117)
        workers, years, firms = 20_000, 8, 2_000
        n = workers * years
        worker = np.repeat(np.arange(workers, dtype=np.int32), years)
        home = rng.integers(0, firms, workers, dtype=np.int32)
        firm = np.repeat(home, years)
        # cuda_panel: on the GPU the capturing sweep still stops above the floor (E1 attempts the
        # projection) while staying certified (Verifications/corrigir_erros_20261117/a, probe).
        move = rng.random(n) < (0.03 if kind != "cuda_panel" else 0.08)
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


def weights_for(kind, n):
    import numpy as np
    w = np.random.default_rng(31).uniform(0.5, 2.0, size=n)
    if kind == "zeros":  # one row in a hundred at weight zero: a nested fit on the positive rows
        w[::100] = 0.0
    return w


def digest(reg, savefe):
    """Raw bytes of every published number, and the declared method."""
    import numpy as np
    h = hashlib.sha256()
    names = ["coef_", "covariance_", "stderr_", "residuals_", "rss_", "num_iterations_", "abs_residual_rel_"]
    for name in names:
        a = np.ascontiguousarray(np.asarray(getattr(reg, name)))
        h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    if savefe:
        for e in reg.fe_effects_:
            a = np.ascontiguousarray(np.asarray(e))
            h.update(a.tobytes())
    return h.hexdigest()[:24]


def child(args):
    core = load(args.child)
    fits, _, _, backend = SCENARIOS[args.scenario]
    for index, (kind, savefe, weights) in enumerate(fits):
        d = design(kind)
        refused = kind == "lowmob_refused"
        kw = dict(se_type="unadjusted", tol=1e-8, max_iter=200 if refused else 100000, num_threads=4,
                  symmetric_sweep=False, absorption_method="auto", tolerance_mode="reghdfe-comparable",
                  retain_fes=savefe, drop_singletons=True)
        if refused:
            kw["fe_tolerance"] = 1e-300  # the recovery cannot meet it in 200 iterations
        if backend == "cuda":
            kw["reduced_pcg_cuda"] = True
        reg = core.HdfeRegressor(**kw)
        fit_kw = {}
        if weights:
            fit_kw["weights"] = weights_for(weights, d["y"].size)
        sys.stderr.write(MARK + str(index) + "\n")
        sys.stderr.flush()
        out = {"index": index}
        try:
            reg.fit(d["y"], d["X"], fes=d["fes"], **fit_kw)
            out.update(status=int(reg.reduced_pcg_status_), reason=str(reg.reduced_pcg_reason_),
                       method=int(reg.absorption_method_used), iterations=int(reg.num_iterations_),
                       gpu_used=bool(getattr(reg, "gpu_used_", False)), digest=digest(reg, savefe),
                       has_code_9=hasattr(core.AbsorptionMethod, "krylov_pcg"))
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        sys.stderr.flush()
        print("RESULT " + json.dumps(out), flush=True)
    sys.stderr.write(MARK + "end\n")
    sys.stderr.flush()


def run(script, module, scenario):
    fits, extra, _, backend = SCENARIOS[scenario]
    env = {k: v for k, v in os.environ.items() if not k.startswith("XHDFE_") and k != "LD_LIBRARY_PATH"}
    env.update(XHDFE_GPU_BACKEND=backend, XHDFE_MOBILITY_MODE="off", XHDFE_ABSORPTION_CACHE_MODE="off",
               XHDFE_FE_STRUCTURE_MODE="off", XHDFE_AUTO_TEAM_COORDINATION="0", XHDFE_REDUCED_PCG_TRACE="1",
               XHDFE_AUTO_ROUTING_RETRY_DIAG="1", OMP_WAIT_POLICY="passive")
    env.update(extra)
    done = subprocess.run([sys.executable, "-B", "-I", str(script), "--child", str(module), "--scenario", scenario],
                          env=env, capture_output=True, text=True)
    results = [json.loads(l[7:]) for l in done.stdout.splitlines() if l.startswith("RESULT ")]
    # Split stderr at the markers: the lines of each fit.
    per_fit, current = {}, None
    for line in done.stderr.splitlines():
        if line.startswith(MARK):
            current = line[len(MARK):]
            per_fit.setdefault(current, [])
        elif current is not None:
            per_fit[current].append(line)
    return {"rc": done.returncode, "results": results, "stderr": per_fit,
            "tail": done.stderr[-1500:] if len(results) != len(fits) else ""}


def masked(lines):
    return [SECONDS.sub("seconds=*", l) for l in lines if l.startswith(TRACE)]


def finals(lines):
    return [l[len(FINAL):] for l in lines if l.startswith(FINAL)]


def agrees(line, result):
    """The final line against the fit's own report."""
    fields = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
    status = fields.get("status")
    if status == "accepted":
        return result["status"] == 1 and result["reason"] == "none" and fields.get("reason") == "none"
    if status == "declined":
        return result["status"] == 2 and result["reason"] == fields.get("reason")
    if status == "ineligible":
        return result["status"] == 0 and result["reason"] == fields.get("reason")
    return False


def projection_decision(lines):
    for l in lines:
        if l.startswith("xhdfe savefe_accurate_projection "):
            return dict(kv.split("=", 1) for kv in l.split()[2:] if "=" in kv)
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child")
    ap.add_argument("--scenario")
    ap.add_argument("--module")
    ap.add_argument("--frozen-module")
    ap.add_argument("--cuda-module")
    ap.add_argument("--frozen-cuda-module")
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

    runs = {}
    for name, (fits, extra, kinds, backend) in SCENARIOS.items():
        if only and name not in only:
            continue
        if backend == "cuda" and not a.cuda_module:
            print(f"SKIP {name}: no --cuda-module", flush=True)
            continue
        new_mod, old_mod = ((a.cuda_module, a.frozen_cuda_module or a.cuda_module) if backend == "cuda"
                            else (a.module, a.frozen_module))
        new, old = run(script, new_mod, name), run(script, old_mod, name)
        runs[name] = new
        for arm, r in (("new", new), ("frozen", old)):
            if len(r["results"]) != len(fits):
                fail(f"{name} [{arm}]: {len(r['results'])} results for {len(fits)} fits (rc {r['rc']}) {r['tail']}")
        if failures and len(new["results"]) != len(fits):
            continue
        for i, kind in enumerate(kinds):
            res, ref = new["results"][i], old["results"][i] if i < len(old["results"]) else {}
            lines, ref_lines = new["stderr"].get(str(i), []), old["stderr"].get(str(i), [])
            got = finals(lines)
            what = f"{name}/{i}"
            if len(got) != 1:
                fail(f"{what}: {len(got)} final lines {got}")
                continue
            decision = projection_decision(lines)
            if "error" in res:
                if kind != "refused":
                    fail(f"{what}: unexpected error {res['error']}")
                if res.get("error") != ref.get("error"):
                    fail(f"{what}: error differs from frozen ({ref.get('error')})")
                if got != finals(ref_lines) or got[0] != "status=ineligible reason=savefe":
                    fail(f"{what}: refused fit prints {got}, frozen {finals(ref_lines)}")
                print(f"INFO {what}: refused ({res['error'][:90]}); final line {got[0]} "
                      f"(frozen: {finals(ref_lines)}); E1 {decision.get('published', '-')}", flush=True)
                continue
            if kind == "refused":
                fail(f"{what}: expected a fit without estimates, got a report")
                continue
            if not agrees(got[0], res):
                fail(f"{what}: line '{got[0]}' disagrees with the report {res['status']}/{res['reason']}")
            if backend == "cpu" and (res.get("digest") != ref.get("digest") or res.get("status") != ref.get("status")
                                     or res.get("reason") != ref.get("reason")
                                     or res.get("iterations") != ref.get("iterations")):
                fail(f"{what}: published numbers or report differ from frozen ({res} / {ref})")
            # The declared method: the frozen module's, except under the forced Krylov route in a
            # module that has code 9 (point 14, a separate patch).
            forced = extra.get("XHDFE_USE_KRYLOV") == "1" and res.get("has_code_9")
            want_method = 9 if forced else ref.get("method")
            if backend == "cpu" and res.get("method") != want_method:
                fail(f"{what}: declared method {res.get('method')}, want {want_method} (frozen {ref.get('method')})")
            if kind == "published":
                if decision.get("published") != "1":
                    fail(f"{what}: E1 did not publish the projection ({decision})")
            elif kind in ("untouched", "plain"):
                if masked(lines) != masked(ref_lines):
                    fail(f"{what}: trace lines differ from frozen: {masked(lines)} != {masked(ref_lines)}")
            elif kind == "cuda_published":
                if decision.get("published") != "1" or not res.get("gpu_used") or "backend=cuda" not in got[0]:
                    fail(f"{what}: the CUDA tier did not publish the projection ({decision}, {got[0]})")
            print(f"INFO {what}: {kind} report {res['status']}/{res['reason']} method {res['method']} "
                  f"final '{SECONDS.sub('seconds=*', got[0])}' (frozen '{SECONDS.sub('seconds=*', finals(ref_lines)[0]) if finals(ref_lines) else None}') "
                  f"E1 published={decision.get('published', '-')} skipped={decision.get('skipped', '-')}", flush=True)
    # A published projection prints the line of the same fit without saved effects, and the
    # tier's other lines of its attempt as that fit does.
    for name, plain in PLAIN_OF.items():
        if name in runs and plain in runs:
            a_lines, b_lines = runs[name]["stderr"].get("0", []), runs[plain]["stderr"].get("0", [])
            if masked(a_lines) != masked(b_lines):
                fail(f"{name}: trace lines differ from {plain}: {masked(a_lines)} != {masked(b_lines)}")
            else:
                print(f"INFO {name}: the trace lines equal those of {plain} ({len(masked(a_lines))} lines)",
                      flush=True)
    if "sequence" in runs:
        seq = runs["sequence"]
        all_finals = [f for i in range(len(SCENARIOS["sequence"][0])) for f in finals(seq["stderr"].get(str(i), []))]
        print(f"INFO sequence: {len(all_finals)} final lines for {len(SCENARIOS['sequence'][0])} fits, in order",
              flush=True)
    print("TRACE_FINAL_CONTRACT_PASS" if not failures else f"TRACE_FINAL_CONTRACT_FAIL {len(failures)} failures",
          flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
