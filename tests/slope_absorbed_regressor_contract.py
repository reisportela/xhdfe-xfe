#!/usr/bin/env python3
"""Correction E2: with heterogeneous slopes in reghdfe-comparable, a regressor that the absorbed effects
annihilate no longer drives the slope-block certificate (Verifications/slopes_absorbed_regressor_20261116/
briefs/E2_SLOPES_ABSORBED_REGRESSOR.md, its amendment 1, and e2/RESULTADO_E2_20261116.md).

The certificate used to divide such a column's block moments by its absorbed norm, which is rounding noise
or a tiny within component: the fit was refused, or the continuation polished a column that the regressor
then omits. A column of X that the FE-collinearity rule omits (detail::kFeCollinearTol, centred variant when
the model has a constant) is now judged in its input scale; y and every retained column keep today's test.
XHDFE_SLOPE_ABSORBED_INPUT_SCALE=0 restores the former certificate everywhere.

Builds nothing; never run during a timing campaign. Synthetic data only, made in each child process from
fixed seeds (an unbalanced panel with an experience-like slope variable z and year effects).

    python3 -I tests/slope_absorbed_regressor_contract.py --module <module or its folder> \
        [--frozen-module <module of the previous revision>] [--only NAME] [--harness-proof]

Arms, every scenario in a fresh process per arm (the switch is read once per process), CPU backend,
caches and the team registry off, two explicit threads:
  new     --module, the switch unset
  off     --module, XHDFE_SLOPE_ABSORBED_INPUT_SCALE=0
  frozen  --frozen-module, when given, also with the switch at 0: a module from before the correction
          ignores it, a later one then uses the former certificate as the off arm does
Each scenario fits the model with idcode##c.z and year effects twice: with the regressor x ("full") and
without it on the same rows ("twin"). "Today" below is the frozen arm when given, the off arm otherwise.
  refused_today  x = g, constant within the individual (annihilated to rounding): today refuses the full fit;
                 new accepts it, reports x omitted (reason 1) and equals the twin bit for bit (b and se of
                 the retained columns, their covariance, the residuals, iterations, slope diagnostics);
  stage_changed  x = g + c (year * z), c set so that ||M_D x||^2 = 1e-12 of the centred sum of squares of
                 x (omitted by the rule): today accepts the full fit with more iterations than the twin (the
                 omitted column drove the continuation); new accepts it with the twin's iterations, equals the
                 twin bit for bit, and lies within the comparable contract of today's fit (1e-9 max(1,|b|),
                 1e-8 sqrt(Vjj Vkk)) - the class "stage changed" of amendment 1;
  near_above     x = g + c u, ratio 2e-9 (above the constant: the regressor keeps x): every arm gives today's
                 outcome bit for bit.
In every scenario the off arm equals the frozen arm bit for bit, and the twin is the same in every arm.
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

SCENARIOS = {"refused_today": ("pure", 0.0), "stage_changed": ("year_x_z", 1e-12),
             "near_above": ("random", 2e-9)}
SWITCH = "XHDFE_SLOPE_ABSORBED_INPUT_SCALE"
ARRAYS = ("coef_", "stderr_", "covariance_", "residuals_")
SCALARS = ("num_iterations_", "nobs_", "num_singletons_", "converged_", "precision_certified_", "df_resid_",
           "rss_", "tss_within_", "r2_within_", "slope_block_residual_rel_", "slope_block_frobenius_rel_",
           "slope_block_rms_rel_", "slope_block_max_rel_", "slope_block_skipped_max_rel_",
           "slope_certificate_worst_fe_", "slope_certificate_worst_moment_", "slope_accuracy_retry_stages_",
           "slope_internal_tolerance_")


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


def pack(value):
    import numpy as np
    a = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return dict(shape=list(a.shape), sha=hashlib.sha256(a.tobytes()).hexdigest(),
                b64=base64.b64encode(a.tobytes()).decode("ascii"))


def unpack(entry):
    import numpy as np
    return np.frombuffer(base64.b64decode(entry["b64"]), dtype=np.float64).reshape(entry["shape"])


def panel(seed=20261116, ni=2000, years=12):
    """Unbalanced spells of 3 to 10 years; z is cumulative experience within the individual."""
    import numpy as np
    rng = np.random.default_rng(seed)
    ind, yr = [], []
    for i in range(ni):
        length = int(rng.integers(3, 11))
        start = int(rng.integers(0, years - length + 1))
        ind += [i] * length
        yr += list(range(start, start + length))
    ind, yr = np.array(ind, dtype=np.int32), np.array(yr, dtype=np.int32)
    n = ind.size
    work = rng.uniform(0.2, 1.0, n)
    z = np.empty(n)
    for k in range(n):
        z[k] = (z[k - 1] if k and ind[k] == ind[k - 1] else 0.0) + work[k]
    a, c = rng.normal(size=ni), 0.05 * rng.normal(size=ni)
    g = rng.integers(8, 19, ni).astype(np.float64)[ind]
    x1 = rng.normal(size=n) + 0.3 * a[ind]
    y = 0.5 * x1 + a[ind] + c[ind] * z + 0.05 * yr + 0.2 * rng.normal(size=n)
    u = np.random.default_rng(seed + 1).normal(size=n)
    return dict(ind=ind, yr=yr, z=z, g=g, x1=x1, y=y, u=u)


def within(v, ind, z, yr):
    """M_D v for D = (per-individual [1, z], year dummies): Frisch-Waugh in two exact steps."""
    import numpy as np
    n_i = np.bincount(ind).astype(np.float64)
    zc = z - (np.bincount(ind, z) / n_i)[ind]
    szz = np.bincount(ind, zc * zc)

    def resid(w):
        wbar = (np.bincount(ind, w) / n_i)[ind]
        slope = np.where(szz > 0, np.bincount(ind, zc * (w - wbar)) / np.where(szz > 0, szz, 1.0), 0.0)
        return w - wbar - slope[ind] * zc
    P = np.column_stack([resid((yr == t).astype(np.float64)) for t in np.unique(yr)[1:]])
    r = resid(v)
    return r - P @ np.linalg.lstsq(P, r, rcond=None)[0]


def regressor(d, direction, ratio):
    import numpy as np
    if direction == "pure":
        return d["g"]
    w = d["yr"] * d["z"] if direction == "year_x_z" else d["u"]
    mw = within(w, d["ind"], d["z"], d["yr"])
    css = float(np.sum((d["g"] - d["g"].mean()) ** 2))
    return d["g"] + float(np.sqrt(ratio * css / float(mw @ mw))) * w


def child(module_file, names):
    import numpy as np
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path(module_file))
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    d = panel()
    y = d["y"].copy()
    if os.environ.get("SLOPE_CONTRACT_PERTURB") == "1":
        y[0] += abs(y[0]) * 2.0 ** -30 + 2.0 ** -40
    out = {}
    for name in names:
        x = regressor(d, *SCENARIOS[name])
        row = {}
        for fit, X in (("full", np.column_stack([x, d["x1"]])), ("twin", d["x1"].reshape(-1, 1))):
            reg = core.HdfeRegressor(se_type="cluster", tol=1e-8, max_iter=100000, num_threads=2,
                                     symmetric_sweep=False, absorption_method="auto",
                                     tolerance_mode="reghdfe-comparable", jacobi_relaxation=0.0, drop_singletons=True)
            try:
                reg.fit(y, np.asfortranarray(X), fes=[d["ind"], d["yr"]], clusters=d["ind"],
                        slopes=[(0, d["z"], True)])
            except Exception as exc:  # a refusal is a result: the arms must refuse alike
                row[fit] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
                continue
            r = {prop: pack(getattr(reg, prop)) for prop in ARRAYS}
            for prop in SCALARS:
                value = getattr(reg, prop)
                r[prop] = float(value).hex() if isinstance(value, float) else str(value)
            r["omitted_reason_"] = [int(v) for v in reg.omitted_reason_]
            row[fit] = r
        out[name] = row
    print("RESULT " + json.dumps(out))


def run_arm(module, names, switches):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("XHDFE_") and k not in ("LD_LIBRARY_PATH", "OMP_NUM_THREADS")}
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


def equals_twin(name, full, twin):
    """The full fit against its twin: the retained columns (x is column 0; the constant is last)."""
    import numpy as np
    failures = []
    if full.get("omitted_reason_", [None])[0] != 1:
        failures.append(f"{name}: x not reported omitted as collinear with the effects: {full.get('omitted_reason_')}")
    b, bt = unpack(full["coef_"]), unpack(twin["coef_"])
    keep = list(range(1, b.size))
    pairs = {"coef_": (b[keep], bt), "stderr_": (unpack(full["stderr_"])[keep], unpack(twin["stderr_"])),
             "covariance_": (unpack(full["covariance_"])[np.ix_(keep, keep)], unpack(twin["covariance_"])),
             "residuals_": (unpack(full["residuals_"]), unpack(twin["residuals_"]))}
    for prop, (left, right) in pairs.items():
        if left.shape != right.shape or not np.array_equal(left.view(np.uint64), right.view(np.uint64)):
            failures.append(f"{name}: {prop} of the retained columns not bit-identical to the twin")
    moved = [s for s in SCALARS if full.get(s) != twin.get(s)]
    if moved:
        failures.append(f"{name}: full fit differs from the twin in {moved}")
    return failures


def within_contract(name, today, new):
    import numpy as np
    b0, b1 = unpack(today["coef_"]), unpack(new["coef_"])
    V0, V1 = unpack(today["covariance_"]), unpack(new["covariance_"])
    keep = list(range(1, b0.size - 1))  # retained slopes (x omitted in both, the constant not published in V)
    db = float(np.max(np.abs(b1[keep] - b0[keep]) / np.maximum(1.0, np.abs(b0[keep]))))
    dg = np.sqrt(np.outer(np.diag(V0)[keep], np.diag(V0)[keep]))
    dV = float(np.max(np.abs(V1[np.ix_(keep, keep)] - V0[np.ix_(keep, keep)]) / dg))
    if not (db <= 1e-9 and dV <= 1e-8):
        return [f"{name}: outside the comparable contract of today's fit: b {db:.2e}, V {dV:.2e}"], db, dV
    return [], db, dV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module")
    ap.add_argument("--frozen-module")
    ap.add_argument("--only", default="")
    ap.add_argument("--child")
    ap.add_argument("--harness-proof", action="store_true")
    a = ap.parse_args()
    names = [n for n in a.only.split(",") if n] or list(SCENARIOS)
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
        moved = run_arm(candidate, names, {"SLOPE_CONTRACT_PERTURB": "1"})
        blind = [n for n in names if all(published(plain[n][f]) == published(moved[n][f]) for f in plain[n])]
        if blind:
            print(f"slope_absorbed_regressor_contract: harness proof FAILED, no difference seen in {blind}")
            raise SystemExit(1)
        print(f"slope_absorbed_regressor_contract: harness proof OK, {len(names)} scenarios see y[0] moved by 2^-30")
        return
    new = run_arm(candidate, names, {})
    off = run_arm(candidate, names, {SWITCH: "0"})
    frozen = run_arm(module_path(a.frozen_module), names, {SWITCH: "0"}) if a.frozen_module else None
    today = frozen if frozen is not None else off
    failures = []
    for name in names:
        n_row, t_row = new[name], today[name]
        for arm, row in (("new", n_row), ("today", t_row)):
            if "error" in row["twin"]:
                failures.append(f"{name}: the twin is refused in the {arm} arm: {row['twin']['error'][:200]}")
        if any("error" in row["twin"] for row in (n_row, t_row)):
            continue
        if published(n_row["twin"]) != published(t_row["twin"]) or published(off[name]["twin"]) != published(t_row["twin"]):
            failures.append(f"{name}: the twin depends on the arm")
        if frozen is not None:
            for fit in ("full", "twin"):
                if published(off[name][fit]) != published(frozen[name][fit]):
                    failures.append(f"{name}: off differs from frozen in the {fit} fit")
        detail = ""
        if name == "refused_today":
            if "error" not in t_row["full"]:
                failures.append(f"{name}: today accepts the full fit (not the fixture of the error)")
            if "error" in n_row["full"]:
                failures.append(f"{name}: new refuses the full fit: {n_row['full']['error'][:200]}")
            else:
                failures += equals_twin(name, n_row["full"], n_row["twin"])
            detail = f"today: {t_row['full'].get('error', 'accepted')[:90]}"
        elif name == "stage_changed":
            if "error" in t_row["full"] or "error" in n_row["full"]:
                failures.append(f"{name}: a full fit is refused: today {t_row['full'].get('error', 'ok')[:120]}, "
                                f"new {n_row['full'].get('error', 'ok')[:120]}")
                continue
            it_today, it_new = int(t_row["full"]["num_iterations_"]), int(n_row["full"]["num_iterations_"])
            it_twin = int(n_row["twin"]["num_iterations_"])
            if not it_today > it_twin:
                failures.append(f"{name}: today's full fit does not iterate beyond the twin ({it_today} vs {it_twin}): "
                                f"not the fixture of the class")
            if not it_new < it_today:
                failures.append(f"{name}: new does not take fewer iterations than today ({it_new} vs {it_today})")
            failures += equals_twin(name, n_row["full"], n_row["twin"])
            more, db, dV = within_contract(name, t_row["full"], n_row["full"])
            failures += more
            detail = f"iterations today {it_today}, new {it_new}, twin {it_twin}; new vs today b {db:.2e}, V {dV:.2e}"
        else:
            if published(n_row["full"]) != published(t_row["full"]):
                failures.append(f"{name}: above the constant, new differs from today in the full fit")
            detail = f"today: {'accepted' if 'error' not in t_row['full'] else 'refused'}, x omitted " \
                     f"{t_row['full'].get('omitted_reason_', ['-'])[0] == 1}"
        print(f"{name}: {detail}")
    arms = "new, off" + (", frozen" if frozen is not None else "")
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print(f"slope_absorbed_regressor_contract: {len(failures)} failure(s); arms {arms}")
        raise SystemExit(1)
    print(f"slope_absorbed_regressor_contract: OK, {len(names)} scenarios, arms {arms}")


if __name__ == "__main__":
    main()
