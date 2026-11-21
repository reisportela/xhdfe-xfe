#!/usr/bin/env python3
"""Correction E1: with saved effects in reghdfe-comparable, the coefficients, the covariance and the
residuals come from the projection of the mode (Verifications/low_memory_20261116/briefs/
E1_SAVEFE_COMPARABLE_PROJECTION_FIRST.md and e1/RESULTADO_E1_20261116.md).

A certified effect-capturing sweep used to be published as it was, although its exit is looser than
the route of the mode (public path-like fixture: b 8.7e-8 and standard error 9.1e-5 away from the
same fit without saved effects). When the sweep's certificate (abs_residual_rel) stopped above the
floor of the mode (the Krylov parity floor, detail::mode_parity_floor, 1e-11 at tol 1e-8), the
projection is now computed again without capture by the route of the mode, eligible for the reduced
tier; it supplies y~, X~, the certificate and the flags, and the recovery evaluates the captured
effects at its coefficients. At or below the floor the sweep is published exactly as before.
XHDFE_SAVEFE_ACCURATE_PROJECTION=0 restores the former publication everywhere.

Builds nothing; never run during a timing campaign. Synthetic data only, made in each child process
from fixed seeds, column by column (no BLAS product).

    python3 -I tests/savefe_comparable_projection_contract.py --module <module or its folder> \
        [--frozen-module <module of the previous revision>] [--only NAME] [--harness-proof]

Arms, every scenario in a fresh process per arm (the switch is read once per process), CPU backend,
caches and the team registry off, explicit threads:
  new     --module, the switch unset
  off     --module, XHDFE_SAVEFE_ACCURATE_PROJECTION=0
  frozen  --frozen-module, when given, also with the switch at 0: a module from before the correction
          ignores it, a later one then publishes the capturing sweep as the off arm does
Each scenario fits the same model twice, without and with saved effects (reghdfe-comparable, method
and criterion Auto), and declares on which side of the floor its capturing sweep stops. Checked:
  1. above the floor (the projection is published: new differs from off): the fit with saved effects
     equals the plain fit, bit for bit in the coefficients (constant included), the complete
     covariance, the standard errors and the residuals when the two report the same method; otherwise
     within 1e-11 max(1,|b|), 1e-10 sqrt(Vjj Vkk) and 1e-10 of the residuals' largest magnitude;
     below the floor: new equals off bit for bit in every published number, and the fit with saved
     effects is within the comparable contract of the plain fit (1e-9 max(1,|b|), 1e-8 sqrt(Vjj Vkk));
  2. new against off: the sum of the saved effects within the recovery check's tolerance
     (fe_tolerance 1e-6 times 2^e, src/fe_recovery_scale.hpp); and the reconstruction identity
     y - Xb - constant - effects - residual no larger than off's distance between its residuals and the
     plain fit's (the error of the captured effects, which off keeps in its residuals) plus that
     tolerance;
  3. off against frozen: every published number bit for bit (the switch restores the former bits);
  4. the correction is not vacuous: on the path-like fixture off is outside 1e-9 max(1,|b|) of the
     plain fit, new is not.
--harness-proof runs the new arm twice, the second time with the first value of y moved by one part in
2^30, and passes only if every scenario is then reported different.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

SCENARIOS = {
    # name: (design, columns, weights, covariance, threads, side of the floor of the mode)
    "pathlike_unadjusted": ("pathlike", 1, None, "unadjusted", 4, "above"),
    "w_p4_cluster": ("w", 4, None, "cluster", 4, "below"),
    "w_p9_aweights_cluster": ("w", 9, "aweights", "cluster", 4, "below"),
}
SWITCH = "XHDFE_SAVEFE_ACCURATE_PROJECTION"
ARRAYS = ("coef_", "stderr_", "covariance_", "residuals_")
SCALARS = ("num_iterations_", "nobs_", "num_singletons_", "converged_", "precision_certified_",
           "threads_used_", "df_resid_", "rss_", "tss_", "r2_within_", "abs_residual_rel_",
           "fe_recovery_iterations_", "fe_recovery_max_delta_", "fe_recovery_converged_",
           "reduced_pcg_status_")


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


def pack(value):
    import numpy as np
    a = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return dict(shape=list(a.shape), sha=hashlib.sha256(a.tobytes()).hexdigest(),
                b64=base64.b64encode(a.tobytes()).decode("ascii"))


def unpack(entry):
    import numpy as np
    return np.frombuffer(base64.b64decode(entry["b64"]), dtype=np.float64).reshape(entry["shape"])


def design(kind, p, seed):
    """pathlike: tests/analytic_precision/cuda_strict_restart_contract.py:124 (an even cycle plus two
    chords; the exact sparse route serves both fits). w: ten rows per worker, firms at random."""
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
        return dict(y=y, X=np.asfortranarray(x.reshape(-1, 1)), fes=[fe_a, fe_b], cluster=fe_a,
                    aw=0.5 + np.random.default_rng(20261110).random(n))
    rng = np.random.default_rng(seed)
    n, workers = 60_000, 6_000
    worker = np.repeat(np.arange(workers, dtype=np.int32), 10)
    year = np.tile(np.arange(10, dtype=np.int32), workers)
    firm = rng.integers(0, n // 200, n, dtype=np.int32)
    X = np.empty((n, p), dtype=np.float64, order="F")
    for j in range(p):
        X[:, j] = rng.standard_normal(n)
    a_w, a_f = rng.standard_normal(workers), rng.standard_normal(n // 200)
    y = rng.standard_normal(n) + a_w[worker] + a_f[firm] + year / 10.0
    for j in range(min(p, 3)):
        X[:, j] += 0.3 * a_w[worker] + 0.3 * a_f[firm]
        y += (0.5 - 0.25 * j) * X[:, j]
    return dict(y=y, X=X, fes=[worker, firm, year], cluster=worker, aw=0.5 + rng.random(n))


def child(module_file, names):
    import numpy as np
    core = load_module(module_file)
    out = {}
    for name in names:
        kind, p, weights, vce, threads, _ = SCENARIOS[name]
        d = design(kind, p, 20261116 + 101 * sorted(SCENARIOS).index(name))
        y = d["y"]
        if os.environ.get("SAVEFE_CONTRACT_PERTURB") == "1":
            y[0] += abs(y[0]) * 2.0 ** -30 + 2.0 ** -40
        kw = dict(fes=d["fes"])
        if vce == "cluster":
            kw["clusters"] = d["cluster"]
        if weights == "aweights":
            kw["weights"] = d["aw"]
        row = {}
        for fit in ("plain", "savefe"):
            reg = core.HdfeRegressor(
                se_type=vce, tol=1e-8, max_iter=100000, num_threads=threads, symmetric_sweep=False,
                absorption_method="auto", tolerance_mode="reghdfe-comparable", jacobi_relaxation=0.0,
                retain_fes=(fit == "savefe"), drop_singletons=True)
            try:
                reg.fit(y, d["X"], **kw)
            except Exception as exc:  # a refusal is a result: the arms must refuse alike
                row[fit] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                continue
            r = {prop: pack(getattr(reg, prop)) for prop in ARRAYS}
            for prop in SCALARS:
                value = getattr(reg, prop)
                r[prop] = float(value).hex() if isinstance(value, float) else str(value)
            r["method"] = str(reg.absorption_method_used)
            if fit == "savefe":
                fe = np.column_stack([np.asarray(v, dtype=np.float64) for v in reg.fe_effects_])
                r["fe_effects_"] = pack(fe)
                idx = np.asarray(reg.sample_index_, dtype=np.int64)
                b = np.asarray(reg.coef_, dtype=np.float64)
                Xs = d["X"][idx]
                fitted = np.zeros(idx.size)
                for j in range(p):
                    fitted += Xs[:, j] * b[j]
                partial = y[idx] - fitted
                r["recon_max"] = float(np.abs(partial - b[p] - fe.sum(axis=1) -
                                              np.asarray(reg.residuals_)).max())
                peak, root_n = float(np.abs(partial).max()), float(np.sqrt(partial.size))
                expo = 0
                if 0.0 < peak < root_n:
                    rms = float(np.linalg.norm(partial) / root_n)
                    if 0.0 < rms < 1.0:
                        expo = int(np.frexp(rms)[1]) - 1
                r["fe_tol"] = float(np.ldexp(1e-6, expo))
            row[fit] = r
        out[name] = row
    print("RESULT " + json.dumps(out))


def run_arm(module, names, switches):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("XHDFE_") and k not in ("LD_LIBRARY_PATH",)}
    env.update(XHDFE_GPU_BACKEND="cpu", XHDFE_MOBILITY_MODE="off", XHDFE_ABSORPTION_CACHE_MODE="off",
               XHDFE_FE_STRUCTURE_MODE="off", XHDFE_AUTO_TEAM_COORDINATION="0")
    env.update(switches)
    done = subprocess.run([sys.executable, "-I", __file__, "--child", str(module), "--only", ",".join(names)],
                          env=env, capture_output=True, text=True)
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    if done.returncode != 0 or len(lines) != 1:
        raise SystemExit(f"arm failed (rc={done.returncode}):\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    return json.loads(lines[0][7:])


def published(fit):
    """Everything an arm publishes, without the transport payloads (hashes stand for the arrays)."""
    return {k: (v["sha"] if isinstance(v, dict) and "sha" in v else v) for k, v in fit.items()}


def within_contract(name, plain, savefe):
    """Check 1 below the floor: the published sweep within the comparable contract of the plain fit."""
    import numpy as np
    b0, b1 = unpack(plain["coef_"]), unpack(savefe["coef_"])
    V0, V1 = unpack(plain["covariance_"]), unpack(savefe["covariance_"])
    k = b0.size - 1
    db = float(np.max(np.abs(b1 - b0) / np.maximum(1.0, np.abs(b0))))
    dV = float(np.max(np.abs(V1[:k, :k] - V0[:k, :k]) / np.sqrt(np.outer(np.diag(V0)[:k], np.diag(V0)[:k]))))
    if not (db <= 1e-9 and dV <= 1e-8):
        return [f"{name}: below the floor, outside the comparable contract of the plain fit: b {db:.2e}, "
                f"V {dV:.2e} (limits 1e-9, 1e-8)"]
    return []


def invariant(name, plain, savefe):
    """Check 1 above the floor on one scenario of the new arm; returns the failures."""
    import numpy as np
    failures = []
    same_route = plain["method"] == savefe["method"]
    for prop in ARRAYS:
        if same_route and plain[prop]["sha"] != savefe[prop]["sha"]:
            failures.append(f"{name}: {prop} not bit-identical to the plain fit (same method {plain['method']})")
    if not same_route:
        b0, b1 = unpack(plain["coef_"]), unpack(savefe["coef_"])
        V0, V1 = unpack(plain["covariance_"]), unpack(savefe["covariance_"])
        e0, e1 = unpack(plain["residuals_"]), unpack(savefe["residuals_"])
        k = b0.size - 1  # slopes; the variance of the constant is not published
        db = float(np.max(np.abs(b1 - b0) / np.maximum(1.0, np.abs(b0))))
        dV = float(np.max(np.abs(V1[:k, :k] - V0[:k, :k]) /
                          np.sqrt(np.outer(np.diag(V0)[:k], np.diag(V0)[:k]))))
        de = float(np.max(np.abs(e1 - e0)) / max(np.max(np.abs(e0)), 1e-300))
        if not (db <= 1e-11 and dV <= 1e-10 and de <= 1e-10):
            failures.append(f"{name}: routes {plain['method']} / {savefe['method']}: b {db:.2e}, V {dV:.2e}, "
                            f"residuals {de:.2e} (limits 1e-11, 1e-10, 1e-10)")
    return failures


def main():
    import numpy as np
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
        moved = run_arm(candidate, names, {"SAVEFE_CONTRACT_PERTURB": "1"})
        blind = [n for n in names if all(published(plain[n][f]) == published(moved[n][f]) for f in plain[n])]
        if blind:
            print(f"savefe_comparable_projection_contract: harness proof FAILED, no difference seen in {blind}")
            raise SystemExit(1)
        print(f"savefe_comparable_projection_contract: harness proof OK, {len(names)} scenarios see y[0] "
              f"moved by 2^-30")
        return
    new = run_arm(candidate, names, {})
    off = run_arm(candidate, names, {SWITCH: "0"})
    frozen = (run_arm(module_path(a.frozen_module), names, {"XHDFE_SAVEFE_ACCURATE_PROJECTION": "0"})
              if a.frozen_module else None)
    failures = []
    for name in names:
        n_row, o_row = new[name], off[name]
        errors = {arm: [f for f in row if "error" in row[f]] for arm, row in (("new", n_row), ("off", o_row))}
        if errors["new"] or errors["off"]:
            failures.append(f"{name}: refused: new {[n_row[f]['error'] for f in errors['new']]}, "
                            f"off {[o_row[f]['error'] for f in errors['off']]}")
            continue
        side = SCENARIOS[name][5]
        projected = published(n_row["savefe"]) != published(o_row["savefe"])
        if side == "above":
            if not projected:
                failures.append(f"{name}: declared above the floor, but new equals off (no projection)")
            failures += invariant(name, n_row["plain"], n_row["savefe"])
        else:
            if projected:
                failures.append(f"{name}: declared below the floor, but new differs from off")
            failures += within_contract(name, n_row["plain"], n_row["savefe"])
        if published(n_row["plain"]) != published(o_row["plain"]):
            failures.append(f"{name}: the plain fit depends on the switch")
        fe_new, fe_off = unpack(n_row["savefe"]["fe_effects_"]), unpack(o_row["savefe"]["fe_effects_"])
        tol = n_row["savefe"]["fe_tol"]
        d_sum = float(np.max(np.abs(fe_new.sum(axis=1) - fe_off.sum(axis=1))))
        if not d_sum <= tol:
            failures.append(f"{name}: saved effects (their sum) moved by {d_sum:.2e} from the off arm "
                            f"(recovery tolerance {tol:.0e})")
        today_error = float(np.max(np.abs(unpack(o_row["savefe"]["residuals_"]) -
                                          unpack(o_row["plain"]["residuals_"]))))
        if not n_row["savefe"]["recon_max"] <= today_error + tol:
            failures.append(f"{name}: reconstruction identity {n_row['savefe']['recon_max']:.2e} above the "
                            f"captured effects' error {today_error:.2e} + {tol:.0e}")
        if frozen is not None:
            for fit in ("plain", "savefe"):
                moved = sorted(k for k in set(published(o_row[fit])) | set(published(frozen[name][fit]))
                               if published(o_row[fit]).get(k) != published(frozen[name][fit]).get(k))
                if moved:
                    failures.append(f"{name}: off differs from frozen in the {fit} fit: {moved}")
        if name.startswith("pathlike"):
            b_plain = unpack(o_row["plain"]["coef_"])
            gap_off = float(np.max(np.abs(unpack(o_row["savefe"]["coef_"]) - b_plain) /
                                   np.maximum(1.0, np.abs(b_plain))))
            gap_new = float(np.max(np.abs(unpack(n_row["savefe"]["coef_"]) - b_plain) /
                                   np.maximum(1.0, np.abs(b_plain))))
            if not (gap_off > 1e-9 and gap_new <= 1e-11):
                failures.append(f"{name}: not the fixture of the error: off {gap_off:.2e}, new {gap_new:.2e}")
        sweep_rel = float.fromhex(o_row["savefe"]["abs_residual_rel_"])
        print(f"{name}: {side} the floor (sweep certificate {sweep_rel:.2e}), projected={projected}; methods "
              f"{n_row['plain']['method']} / "
              f"{n_row['savefe']['method']}; effects' sum vs off {d_sum:.2e} (tol {tol:.0e}); reconstruction "
              f"{n_row['savefe']['recon_max']:.2e} (captured effects' error {today_error:.2e})")
    arms = "new, off" + (", frozen" if frozen is not None else "")
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print(f"savefe_comparable_projection_contract: {len(failures)} failure(s); arms {arms}")
        raise SystemExit(1)
    print(f"savefe_comparable_projection_contract: OK, {len(names)} scenarios, arms {arms}")


if __name__ == "__main__":
    main()
