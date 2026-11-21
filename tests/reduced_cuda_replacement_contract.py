#!/usr/bin/env python3
"""Package 3-lite (17 November 2026): the residual replacement S x on the device. On by default within the
requested CUDA tier since 17 November 2026 (the owner's decision); XHDFE_REDUCED_PCG_CUDA_REPLACEMENT=0 keeps the
CPU product (the kill switch, here "off").

Synthetic designs only (generated here); one fresh process per fit, because the switch is read once per process.
Not every output of a fit repeats bit for bit from one process to the next: a route outside the tier (the capturing
sweep of saved effects, a fallback after a declined attempt) runs on the GPU with reductions whose order varies.
The reproducible set is therefore MEASURED: build_p2b runs twice (p2b_off, p2b_again), and the arrays and blanked
trace lines equal in both are the ones held bit for bit below; the others are held within the limits.
Modes per design:
  p2b_off, p2b_again  the frozen build_p2b module, no variable, twice: the reference of the OFF proof;
  off          this module, the variable at 0: the reproducible arrays bit for bit with p2b_off (the others within the
               limits), the same outcome; every reproducible trace line of build_p2b present, as a multiset, once times
               and free-memory readings are blanked; one event=replacement line more per attempt (item 5 of the brief);
               page_locked_iteration_bytes unchanged;
  off_compare  the qualification mode with the switch off: outputs as off (same rule); comparison lines;
  on           this module, no variable (the default): the tier on the device, every replacement on the device, four
               registered slices; within the runtime contract's limits of off (|db| <= 1e-8 max(1,|b|), V 1e-8, SE
               1e-7), except two degenerate designs, classified in the code (formed: b and SE only; span: recorded);
  on_compare   the qualification mode with the switch on: outputs as on (same rule); comparison lines;
  on_nod10     the switch on with XHDFE_REDUCED_PCG_CHAT_STOP=0 (D10 out of scope): not in force, replacements on the
               CPU, the tier's own trace lines equal to off_nod10's, the outputs as off_nod10 (same rule);
  on_explicit  (groups) the variable at 1: the bits and the outcome of on;
  on_other     (groups) the variable at "off" (any value but exactly 0 is on): the bits of on, in force;
  on_late0     (groups) one process fits, sets the variable to 0 and fits again (the variable is read once per
               process): the second fit has the bits of on and every attempt is in force;
  on_tight     (groups) tol 1e-12, below D10's scope (1.25e-12): not in force, no device replacement, the two
               registered slices of off.
Designs: groups (11 columns: two groups of widths 6 and 5), formed (y in the span of the effects up to 1e-9: the formed
regime), hook (Config's test hook through XHDFE_TEST_REDUCED_PCG_CUDA_RECURRENCE_CHAT: the recurrence's chat read as
1e-20 at iteration 3, a candidate the replacement rejects; build_p2b has no such hook, so this design has no OFF proof
against it), savefe (retained effects on a low-mobility panel: the projection runs through the tier), span (y and a
regressor in the span of the effects up to 1e-13: their columns stop on test 1 under D10). On groups, a
fatal error injected at the first device replacement must raise with nothing published, and the same process must
then fit again with the bits of on; and with the comparison made strict (a zero bound: any difference is a
divergence), the attempt must decline with reason replacement_divergence and the fit be published by the next route
within the limits.
usage: reduced_cuda_replacement_contract.py --module <CUDA module> --p2b-module <build_p2b CUDA module> --output <new dir>
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DESIGNS = ("groups", "formed", "hook", "savefe", "span")


def design(name):
    import numpy as np
    rng = np.random.default_rng(2026111703 + DESIGNS.index(name))
    if name == "savefe":
        # The low-mobility panel of the Stata request contract (8000 workers over 10 years in 400 firms, a move
        # with probability 0.005 a year): its CUDA capturing sweep stops above the floor of the mode, so that the
        # saved effects' projection (correction E1) runs through the tier.
        workers, years = 8000, 10
        home = rng.integers(0, 400, workers)
        moved = np.cumsum(rng.random((workers, years)) < 0.005, axis=1)
        fes = [np.repeat(np.arange(workers, dtype=np.int32), years),
               ((home[:, None] + 7 * moved) % 400).ravel().astype(np.int32),
               np.tile(np.arange(years, dtype=np.int32), workers)]
        X = np.asfortranarray(rng.normal(size=(workers * years, 2)))
        y = .5 * X[:, 0] - .25 * X[:, 1] + np.sin(fes[0]) + np.cos(fes[1]) + fes[2] / 10 + rng.normal(size=workers * years)
        return y, X, fes
    n = 49152
    fes = [np.repeat(np.arange(n // 24, dtype=np.int32), 24), rng.integers(0, 83, n, dtype=np.int32),
           rng.integers(0, 13, n, dtype=np.int32)]
    p = 10 if name == "groups" else 3
    X = np.asfortranarray(rng.normal(size=(n, p)))
    effects = [rng.normal(size=int(f.max()) + 1)[f] for f in fes]
    if name == "formed":
        y = effects[0] + effects[1] + effects[2] + 1e-9 * rng.normal(size=n)
    elif name == "span":
        # y and the last regressor in the span of the effects up to 1e-13: their columns stop on test 1 under D10.
        y = effects[0] + effects[1] + effects[2] + 1e-13 * rng.normal(size=n)
        X[:, p - 1] = effects[1] + effects[2] + 1e-13 * rng.normal(size=n)
    else:
        y = X @ np.linspace(-0.5, 0.5, p) + rng.normal(size=n) + effects[0] + effects[1] + effects[2]
    return y, X, fes


def fit(mod, name, out):
    import numpy as np
    y, X, fes = design(name)
    os.environ["XHDFE_GPU_BACKEND"] = "cuda"
    reg = mod.HdfeRegressor(se_type="cluster", tol=float(os.environ.get("REPLACEMENT_CONTRACT_TOL", "1e-8")),
                            max_iter=10000, num_threads=4,
                            tolerance_mode="reghdfe-comparable", reduced_pcg_cuda=True, retain_fes=(name == "savefe"))
    reg.fit(y, X, fes=fes, clusters=[fes[0]])
    arrays = dict(coef=np.asarray(reg.coef_), cov=np.asarray(reg.covariance_), se=np.asarray(reg.stderr_),
                  res=np.asarray(reg.residuals_))
    if name == "savefe":
        arrays["fe"] = np.column_stack([np.asarray(v, dtype=np.float64) for v in reg.fe_effects_])
    np.savez(out + ".npz", **arrays)
    Path(out + ".json").write_text(json.dumps(dict(
        method=int(reg.absorption_method_used), status=int(reg.reduced_pcg_status_), reason=str(reg.reduced_pcg_reason_),
        gpu_used=bool(reg.gpu_used_), iterations=int(reg.num_iterations_), rel=float(reg.abs_residual_rel_))))


def child(module_path, name, out, special):
    fatal = special == "fatal"
    import importlib.util
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if fatal:
        os.environ["XHDFE_TEST_REDUCED_PCG_CUDA_FAIL"] = "replacement"
        try:
            fit(mod, name, out + "_fatal")
            raised = ""
        except Exception as exc:  # the injected fatal error must leave the fit
            raised = f"{type(exc).__name__}: {exc}"[:300]
        del os.environ["XHDFE_TEST_REDUCED_PCG_CUDA_FAIL"]
        Path(out + "_fatal.txt").write_text(raised)
    if special == "late0":
        fit(mod, name, out + "_first")
        os.environ["XHDFE_REDUCED_PCG_CUDA_REPLACEMENT"] = "0"  # after the process's first read: no effect
    fit(mod, name, out)


def run(module, name, mode, out_dir, extra=None, special="plain"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("XHDFE_") and k != "LD_LIBRARY_PATH"}
    env.update(XHDFE_REDUCED_PCG_TRACE="1", XHDFE_ABSORPTION_CACHE_MODE="off", XHDFE_FE_STRUCTURE_MODE="off",
               XHDFE_MOBILITY_MODE="off", OMP_WAIT_POLICY="passive", PYTHONDONTWRITEBYTECODE="1")
    env.update(extra or {})
    label = f"{name}_{mode}"
    done = subprocess.run([sys.executable, "-B", "-I", __file__, "--child", module, name, str(out_dir / label),
                           special], env=env, capture_output=True, text=True, timeout=1800)
    (out_dir / f"{label}.stderr").write_text(done.stderr)
    if done.returncode:
        raise RuntimeError(f"{label} failed: {done.stderr[-2000:]}")
    return label


def load(out_dir, label):
    import numpy as np
    arrays = dict(np.load(out_dir / f"{label}.npz"))
    info = json.loads((out_dir / f"{label}.json").read_text())
    trace = [l for l in (out_dir / f"{label}.stderr").read_text().splitlines() if l.startswith("xhdfe ")]
    return arrays, info, trace


TIMING = re.compile(r"(^|_)(seconds|ms)$")


def blank(line):
    """A trace line with its times, its readings of free device memory and the residual of the saved effects'
    capturing sweep (primary_rel, and rel on the auto_routing_retry line: a GPU route outside the tier whose bits
    vary from process to process, measured below) blanked."""
    out = []
    retry = line.startswith("xhdfe auto_routing_retry ")
    for token in line.split():
        key, sep, value = token.partition("=")
        if sep and (TIMING.search(key) or "free" in key or "taken" in key or key == "primary_rel" or (retry and key == "rel")):
            token = key + "=*"
        out.append(token)
    return " ".join(out)


# Outputs computed outside the tier, by a GPU route whose reductions vary from process to process: held within the
# limits, never bit for bit (their spread between two runs of build_p2b is reported).
NOT_REPRODUCIBLE = {"savefe": {"fe"}}


def field(lines, event, key):
    for l in lines:
        if f"event={event} " in l:
            for token in l.split():
                if token.startswith(key + "="):
                    return token.split("=", 1)[1]
    return None


def replacements(lines):
    """The event=replacement lines of a fit (one per attempt of the tier): totals, and in force on every attempt."""
    rows = [dict(t.split("=", 1) for t in l.split() if "=" in t) for l in lines if "event=replacement " in l]
    return dict(attempts=len(rows), in_force=sorted({r["in_force"] for r in rows}),
                device=sum(int(r["replacement_device"]) for r in rows), cpu=sum(int(r["replacement_cpu"]) for r in rows),
                registered=sorted({int(r["registered_doubles"]) for r in rows if int(r["registered_doubles"]) > 0}))


def same_arrays(a, b):
    import numpy as np
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k], equal_nan=True) and a[k].tobytes() == b[k].tobytes()
                                        for k in a)


def within(a, b):
    import numpy as np
    db = float(np.max(np.abs(a["coef"] - b["coef"]) / np.maximum(1, np.abs(b["coef"]))))
    sd = np.sqrt(np.maximum(0, np.diag(b["cov"])))
    scale = np.outer(sd, sd)
    dv = float(np.max(np.abs(a["cov"] - b["cov"]) / np.where(scale > 0, scale, np.inf)))
    ds = float(np.max(np.abs(a["se"] / np.where(b["se"] > 0, b["se"], np.inf) - 1)))
    return db, dv, ds


def reproducible(x, y):
    """The arrays equal bit for bit in two runs of the same module and setting."""
    return {k for k in x if k in y and x[k].tobytes() == y[k].tobytes()}


def agree(a, b, keys):
    """Bit for bit on `keys`; the other arrays within the limits (b 1e-8, SE 1e-7, V 1e-8, residuals and effects
    1e-6 of their scale). Returns (ok, the largest differences of the arrays held within the limits)."""
    import numpy as np
    if a.keys() != b.keys():
        return False, {}
    ok = all(a[k].tobytes() == b[k].tobytes() for k in keys)
    rest = {}
    for k in a:
        if k in keys:
            continue
        d = float(np.max(np.abs(a[k] - b[k]))) if a[k].size else 0.0
        rest[k] = d
        if k == "coef":
            ok = ok and within(a, b)[0] <= 1e-8
        elif k == "se":
            ok = ok and within(a, b)[2] <= 1e-7
        elif k == "cov":
            ok = ok and within(a, b)[1] <= 1e-8
        else:
            ok = ok and d <= 1e-6 * max(1.0, float(np.max(np.abs(b[k]))) if b[k].size else 1.0)
    return ok, rest


def main():
    if sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--p2b-module", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=False, exist_ok=False)
    off = {"XHDFE_REDUCED_PCG_CUDA_REPLACEMENT": "0"}  # the kill switch; no variable is the default, on
    compare = {"XHDFE_TEST_REDUCED_PCG_CUDA_REPLACEMENT_COMPARE": "1"}
    nod10 = {"XHDFE_REDUCED_PCG_CHAT_STOP": "0"}
    failures, rows = [], []
    for name in DESIGNS:
        hook = {"XHDFE_TEST_REDUCED_PCG_CUDA_RECURRENCE_CHAT": "3:1e-20"} if name == "hook" else {}
        if name == "savefe":
            hook = {"XHDFE_AUTO_ROUTING_RETRY_DIAG": "1"}  # the projection's diagnostic line
        labels = {
            "p2b_off": run(a.p2b_module, name, "p2b_off", out, hook),
            "p2b_again": run(a.p2b_module, name, "p2b_again", out, hook),
            "off": run(a.module, name, "off", out, {**hook, **off}),
            "off_compare": run(a.module, name, "off_compare", out, {**hook, **off, **compare}),
            "on": run(a.module, name, "on", out, hook),
            "on_compare": run(a.module, name, "on_compare", out, {**hook, **compare}),
            "off_nod10": run(a.module, name, "off_nod10", out, {**hook, **off, **nod10}),
            "on_nod10": run(a.module, name, "on_nod10", out, {**hook, **nod10}),
        }
        if name == "groups":
            labels["on_after_fatal"] = run(a.module, name, "on_after_fatal", out, hook, special="fatal")
            labels["on_strict"] = run(a.module, name, "on_strict", out,
                                      {**hook, "XHDFE_TEST_REDUCED_PCG_CUDA_REPLACEMENT_COMPARE": "strict"})
            labels["on_explicit"] = run(a.module, name, "on_explicit", out,
                                        {**hook, "XHDFE_REDUCED_PCG_CUDA_REPLACEMENT": "1"})
            labels["on_other"] = run(a.module, name, "on_other", out,
                                     {**hook, "XHDFE_REDUCED_PCG_CUDA_REPLACEMENT": "off"})
            labels["on_late0"] = run(a.module, name, "on_late0", out, hook, special="late0")
            labels["on_tight"] = run(a.module, name, "on_tight", out, {**hook, "REPLACEMENT_CONTRACT_TOL": "1e-12"})
        r = {k: load(out, v) for k, v in labels.items()}
        row = {"design": name}

        def check(ok, what):
            if not ok:
                failures.append(f"{name}: {what}")
            return ok
        # What repeats from one process to the next: two runs of build_p2b (reported); the arrays held bit for bit are
        # all but those computed outside the tier (NOT_REPRODUCIBLE), and they must repeat between the two runs.
        from collections import Counter
        import numpy as np
        keys = set(r["p2b_off"][0]) - NOT_REPRODUCIBLE.get(name, set())
        measured = reproducible(r["p2b_off"][0], r["p2b_again"][0])
        check(keys <= measured, f"build_p2b itself does not repeat on {sorted(keys - measured)}")
        old_runs = [Counter(blank(l) for l in r[k][2]) for k in ("p2b_off", "p2b_again")]
        stable = old_runs[0] & old_runs[1]
        check(old_runs[0] == old_runs[1], f"build_p2b's blanked trace does not repeat: {list((old_runs[0] - stable).elements())[:2]}")
        row["reproducible"] = dict(bit_for_bit=sorted(keys), within_limits=sorted(set(r["p2b_off"][0]) - keys),
                                   p2b_twice_spread={k: float(np.max(np.abs(r["p2b_off"][0][k] - r["p2b_again"][0][k])))
                                                     for k in r["p2b_off"][0] if k not in measured})
        # OFF: package 2's outputs and trace (the reproducible lines of build_p2b, as a multiset; one event=replacement
        # line more per attempt, i.e. per operator line). The hook design has no counterpart in build_p2b.
        new_lines = [blank(l) for l in r["off"][2]]
        added = sum("event=replacement " in l for l in new_lines)
        attempts = sum("reduced_pcg_cuda event=operator " in l for l in new_lines)
        check(added == attempts and added >= 1, f"event=replacement lines {added}, attempts {attempts}")
        if name != "hook":
            ok, rest = agree(r["off"][0], r["p2b_off"][0], keys)
            check(ok and r["off"][1]["method"] == r["p2b_off"][1]["method"] and r["off"][1]["status"] == r["p2b_off"][1]["status"],
                  f"off != p2b_off (outputs; held within the limits: {rest})")
            required = stable
            if name == "savefe":
                # Point 13 (17nov2026): build_p2b's final line for this fit was its capturing sweep's (ineligible by
                # savefe), although the tier publishes the projection; it leaves the required set, and the off run
                # must print exactly one final line, accepted on the device, agreeing with what the run reports.
                old_final = "xhdfe reduced_pcg event=final status=ineligible reason=savefe"
                required = stable - Counter({old_final: stable[old_final]})
                info = r["off"][1]
                said = "status=%s reason=%s" % ({0: "ineligible", 1: "accepted", 2: "declined"}.get(info["status"],
                                                                                                info["status"]), info["reason"])
                finals = [l for l in new_lines if l.startswith("xhdfe reduced_pcg event=final ")]
                check(len(finals) == 1 and info["status"] == 1 and f" event=final {said} " in finals[0] + " "
                      and " backend=cuda " in finals[0] + " ",
                      f"final lines with the switch off {finals}: not one line accepted on the device that agrees with "
                      f"the run's report ({said})")
            new = Counter(l for l in new_lines if "event=replacement " not in l)
            check(not (required - new), f"reproducible p2b trace lines missing or changed with the switch off: "
                                        f"{list((required - new).elements())[:2]}")
            row["off_trace"] = dict(p2b_lines=sum(old_runs[0].values()), lines=len(new_lines), replacement_lines=added,
                                    outputs_within_limits=rest)
            check(field(r["off"][2], "admission", "page_locked_iteration_bytes") ==
                  field(r["p2b_off"][2], "admission", "page_locked_iteration_bytes"), "page_locked_iteration_bytes changed")
        # OFF with compare, ON with compare: the used product's outputs unchanged (same rule).
        ok, rest_off = agree(r["off_compare"][0], r["off"][0], keys)
        check(ok and r["off_compare"][1]["iterations"] == r["off"][1]["iterations"], f"off_compare != off {rest_off}")
        ok, rest_on = agree(r["on_compare"][0], r["on"][0], keys)
        check(ok and r["on_compare"][1]["iterations"] == r["on"][1]["iterations"], f"on_compare != on {rest_on}")
        # ON: accepted on the device (the saved-effects fit: its projection through the tier), replacements on
        # the device, four slices; within the limits of OFF.
        info = r["on"][1]
        if name == "savefe":
            published = {k: sum("savefe_accurate_projection published=1" in l for l in
                                (out / f"{labels[k]}.stderr").read_text().splitlines()) for k in ("off", "on")}
            row["projection_published"] = published
            check(info["gpu_used"] and published["on"] >= 1 and published["off"] >= 1,
                  f"savefe: the projection not published through the tier {published} {info}")
        else:
            check(info["method"] == 8 and info["status"] == 1 and info["gpu_used"], f"on: tier not accepted on the device {info}")
        rep_on, rep_off = replacements(r["on"][2]), replacements(r["off"][2])
        check(rep_on["in_force"] == ["1"] and rep_on["device"] > 0 and rep_on["cpu"] == 0,
              f"on: replacements not on the device {rep_on}")
        check(rep_off["in_force"] == ["0"] and rep_off["device"] == 0 and rep_off["cpu"] > 0,
              f"off: replacements not on the CPU {rep_off}")
        check(len(rep_on["registered"]) >= 1 and len(rep_off["registered"]) >= 1 and
              [2 * v for v in rep_off["registered"]] == rep_on["registered"], f"registered: 4 slices on, 2 off {rep_on} {rep_off}")
        # ON against OFF: the runtime contract's limits (b 1e-8, V 1e-8, SE 1e-7). Two designs are degenerate on
        # purpose and are classified, not gated: formed (y's residual is 1e-9 of its scale, so V, made of those
        # residuals, carries the solver's tolerance relative to 1e-9: held to the acceptance references b and SE,
        # with the residuals' largest difference against their scale recorded); span (y and a regressor in the span
        # up to 1e-13: their residuals are at the tolerance itself; numbers recorded only).
        import numpy as np
        db, dv, ds = within(r["on"][0], r["off"][0])
        res_on, res_off = r["on"][0]["res"], r["off"][0]["res"]
        row["on_vs_off_residuals"] = dict(max_abs_diff=float(np.max(np.abs(res_on - res_off))), scale=float(np.max(np.abs(res_off))))
        if name == "formed":
            check(db <= 1e-8 and ds <= 1e-7, f"on vs off beyond the acceptance references: b {db:.2e} SE {ds:.2e} (V {dv:.2e})")
        elif name != "span":
            check(db <= 1e-8 and dv <= 1e-8 and ds <= 1e-7, f"on vs off beyond the limits: b {db:.2e} V {dv:.2e} SE {ds:.2e}")
        if name == "savefe":
            import numpy as np
            dfe = float(np.max(np.abs(r["on"][0]["fe"] - r["off"][0]["fe"])))
            row["fe_on_vs_off"] = dfe
            check(dfe <= 1e-6 * max(1.0, float(np.max(np.abs(r["off"][0]["fe"])))), f"savefe: effects on vs off {dfe:.2e}")
        # Outside D10 the switch changes nothing: not in force; the tier's own lines (plan, columns, final) equal;
        # the outputs bit for bit when the tier published both fits, within the limits otherwise (a fallback route).
        tier_lines = lambda t: [blank(l) for l in t if "xhdfe reduced_pcg event=" in l and "event=phase" not in l]
        check(tier_lines(r["on_nod10"][2]) == tier_lines(r["off_nod10"][2]), "on_nod10: the tier's lines differ from off_nod10")
        both_tier = all(r[k][1]["method"] == 8 and r[k][1]["status"] == 1 for k in ("on_nod10", "off_nod10"))
        ok, rest_nod10 = agree(r["on_nod10"][0], r["off_nod10"][0], keys if both_tier else set())
        check(ok, f"on_nod10 != off_nod10 {rest_nod10}")
        check(replacements(r["on_nod10"][2])["in_force"] == ["0"], "on_nod10: in force outside D10")
        row["nod10"] = dict(route={k: (r[k][1]["method"], r[k][1]["status"], r[k][1]["reason"]) for k in ("off_nod10", "on_nod10")},
                            outputs_within_limits=rest_nod10)
        # The comparison lines.
        for mode in ("off_compare", "on_compare"):
            lines = [l for l in r[mode][2] if "event=replacement_compare " in l]
            summary = next((l for l in r[mode][2] if "event=replacement_compare_summary" in l), "")
            row[mode] = dict(lines=len(lines), summary=summary.split(" ", 2)[-1] if summary else "",
                             differs=sum("decision_differs=1" in l for l in lines),
                             chat_unverified=sum(" chat_verified=0" in l for l in lines),
                             layer2_margin=sum(" layer2=margin" in l for l in lines),
                             layer2_unverified=sum(" layer2=unverified" in l for l in lines),
                             within_bound=sum("threshold_within_bound=1" in l for l in lines),
                             stops=sorted({re.search(r" stop_dev=(\S+)", l).group(1) for l in lines}),
                             formed=sum(" formed_dev=1" in l or " formed_cpu=1" in l for l in lines),
                             max_bound_ratio=max((float(re.search(r"bound_ratio=(\S+)", l).group(1)) for l in lines), default=0.0))
            check(len(lines) > 0, f"{mode}: no comparison line")
            check(" divergent=0" in summary, f"{mode}: a difference above its bound")
        row.update(iterations={k: r[k][1]["iterations"] for k in ("off", "on", "off_nod10")},
                   rel={k: r[k][1]["rel"] for k in ("off", "on")},
                   replacements={"on": rep_on, "off": rep_off}, on_vs_off=dict(b=db, V=dv, SE=ds),
                   stops=[l.split(" ", 3)[-1] for l in r["on"][2] if "reduced_pcg event=column" in l])
        if name == "groups":
            raised = (out / f"{labels['on_after_fatal']}_fatal.txt").read_text()
            check("injected fatal error at a replacement" in raised, f"fatal at a replacement did not raise: {raised!r}")
            check(not (out / f"{labels['on_after_fatal']}_fatal.npz").exists(), "fatal at a replacement published estimates")
            check(same_arrays(r["on_after_fatal"][0], r["on"][0]) and r["on_after_fatal"][1] == r["on"][1],
                  "the fit after the fatal error != on")
            row["fatal"] = raised
            check(field(r["on"][2], "plan", "groups") == "2", "groups: not two groups")
            # The variable at 1 is the default.
            check(same_arrays(r["on_explicit"][0], r["on"][0]) and r["on_explicit"][1] == r["on"][1] and
                  replacements(r["on_explicit"][2]) == rep_on, "the variable at 1 != the default")
            # Round 29: any value but exactly 0 is on; the variable is read once per process; below 1.25e-12 the
            # rule D10 is out of scope, so the replacement stays on the CPU with two registered slices.
            check(same_arrays(r["on_other"][0], r["on"][0]) and replacements(r["on_other"][2]) == rep_on,
                  "the variable at 'off' (not exactly 0) != the default")
            late = replacements(r["on_late0"][2])
            check(same_arrays(r["on_late0"][0], r["on"][0]) and late["in_force"] == ["1"] and late["cpu"] == 0 and
                  late["attempts"] == 2 * rep_on["attempts"],
                  f"the variable set to 0 after the first fit changed the selection {late}")
            tight = replacements(r["on_tight"][2])
            check(tight["attempts"] >= 1 and tight["in_force"] == ["0"] and tight["device"] == 0 and
                  tight["registered"] == rep_off["registered"],
                  f"below 1.25e-12: not the CPU replacement with two slices {tight} {rep_off}")
            row["round29"] = dict(other=replacements(r["on_other"][2]), late0=late, tight=tight,
                                  tight_outcome=r["on_tight"][1])
            # The refusal of a divergence above the bound (round 28, adjudication 4), forced with a zero bound: the
            # attempt declines with its own reason and the fit is published by the next route, within the limits.
            strict = r["on_strict"][1]
            db_s, dv_s, ds_s = within(r["on_strict"][0], r["on"][0])
            check(strict["status"] == 2 and strict["reason"] == "replacement_divergence" and
                  field(r["on_strict"][2], "replacement_compare_summary", "divergent") == "1" and db_s <= 1e-8 and ds_s <= 1e-7,
                  f"strict comparison: {strict} b {db_s:.2e} SE {ds_s:.2e}")
            row["strict"] = dict(outcome=strict, b=db_s, V=dv_s, SE=ds_s)
        rows.append(row)
    receipt = dict(status="PASS" if not failures else "FAIL", failures=failures, designs=rows)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    for row in rows:
        print(json.dumps(row))
    print("REDUCED_CUDA_REPLACEMENT_CONTRACT_" + ("PASS" if not failures else "FAIL " + "; ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
