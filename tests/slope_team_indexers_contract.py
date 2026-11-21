#!/usr/bin/env python3
"""Phase 5a of the Stata-path plan: the identifier builds of the route with
heterogeneous slopes on the team (the indexers of every entry of
absorb_fixed_effects_v6_mixed_impl() and the slope rank's) give every output of
today's serial builds.

Builds nothing; never run during a timing campaign. Synthetic data only, made in
each child process from fixed seeds; nothing is read from disk but the modules.

    python3 -I tests/slope_team_indexers_contract.py --module <this branch's module> \
        --frozen-module <phase chain, packages/cand_sp2_py_cpu> [--only NAME]

Every scenario runs its fits in five fresh processes:
  new     the branch's module (XHDFE_SLOPE_TEAM_INDEXERS unset: the team)
  off     the branch's module with XHDFE_SLOPE_TEAM_INDEXERS=0 (today's builders)
  base    the frozen module
  verify  the branch's module with XHDFE_SPARSE_RANK_VERIFY=1
  bverify the frozen module with XHDFE_SPARSE_RANK_VERIFY=1
new is compared with base, off and verify on the REQUIRED properties of
tests/sparse_rank_contract.py (b, V, residuals, saved effects, levels, degrees of
freedom, iterations, threads_used, ... by dtype, shape and raw bytes), the tier's
diagnostics and the method. Positive controls: the new profile lines
"cpu_profile label=mixed_indexers ... team=T" and "label=slope_rank_indexer team=T"
with T > 1 where the gate admits the team and T = 1 where it does not, and T = 1
in every off line; with the verifier on, no "ok=0" line, and more verified sites
("label=sparse_rank_verify ... ok=1") than the frozen module on the same case
wherever the new code took the team (identical bits with the sites silent prove
nothing).

Scenarios:
  large  about 1.10 M rows (above 2^20, below the 4194304-row gate), a raw sparse
         firm id (range about 8.7 n) carrying the slopes: one term with
         intercept and two terms on the firm, one with and one without intercept
         (the secondreg shape), comparable and fast, explicit thread count;
         weights; saved effects (fast, bounded iterations); a dense firm id
         (control); rows sorted by the sparse id; the automatic thread count
         (below the gate: no team).
  xl     about 4.40 M rows (above the gate): the automatic thread count takes the
         team; one term with intercept, fast.
  small  about 0.17 M rows (below 2^20): explicit (the team is admitted by the
         explicit request) and automatic (no team).
  boundary  the small panel where firms 0, 3, 6, ... carry a slope variable almost
         constant within the firm (1 + eps u, eps 1e-7, 1e-6, 1e-5): the
         determinant boundary of heterogeneous_slope_rank() (det against
         1e-12 * scale); fast, explicit, at most 3000 iterations (ill-conditioned
         on purpose; a first version with these firms in the large comparable
         fits ran for more than ten minutes without finishing one fit).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparse_rank_contract import (DIAGNOSTICS, REQUIRED, captured_stderr, describe,  # noqa: E402
                                  load, spread_codes)

SWITCH = "XHDFE_SLOPE_TEAM_INDEXERS"


def slope_panel(seed, workers, firms, years=10, boundary=False):
    rng = np.random.default_rng(seed)
    rows = rng.integers(3, 9, size=workers)
    worker = np.repeat(np.arange(workers), rows)
    n = worker.size
    year = (rng.integers(0, years - 8, size=workers)[worker] +
            np.concatenate([np.arange(r) for r in rows])) % years
    home = rng.integers(0, firms, size=workers)
    move = rng.random(n) < 0.15
    firm = np.where(move, rng.integers(0, firms, size=n), home[worker])
    z = rng.normal(size=n)
    z2 = rng.normal(size=n)
    if boundary:
        # The determinant boundary: in firms 0, 3, 6, ... the slope variable is
        # 1 + eps * u with eps cycling over 1e-7, 1e-6, 1e-5 (det / scale near
        # 1e-12). Ill-conditioned on purpose: only in the small, fast scenario.
        near = (firm % 3) == 0
        eps = np.array([1e-7, 1e-6, 1e-5])[(firm // 3) % 3]
        z = np.where(near, 1.0 + eps * rng.normal(size=n), z)
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n) + 0.1 * (firm % 7)])
    y = X @ np.array([0.5, -0.25]) + 0.3 * rng.normal(size=n) + 0.2 * z
    for f in (worker, firm, year):
        y = y + rng.normal(size=int(f.max()) + 1)[f]
    y = y + rng.normal(size=firms)[firm] * z
    return {
        "y": y, "X": np.asfortranarray(X), "n": n,
        "worker": worker.astype(np.int32), "firm_code": firm.astype(np.int32),
        "firm_raw": spread_codes(firm, 8.7, rng), "year": year.astype(np.int32),
        "z": z, "z2": z2, "w": rng.uniform(0.5, 2.0, size=n),
    }


def ctor(module, **kw):
    base = dict(se_type="cluster", tol=1e-8, max_iter=100000, fit_intercept=True,
                num_threads=8, drop_singletons=True, symmetric_sweep=False,
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
            except Exception as exc:
                public[name] = {"raises": type(exc).__name__}
        record.update(public=public, missing=missing,
                      diagnostics={k: getattr(reg, k) for k in DIAGNOSTICS if hasattr(reg, k)},
                      method=int(reg.absorption_method_used),
                      threads_used=int(reg.threads_used_))
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["lines"] = [l for l in (log.read_text().splitlines() if log.exists() else [])
                       if "label=mixed_indexers" in l or "label=slope_rank_indexer" in l
                       or "label=sparse_rank" in l]
    (out / f"{label}.json").write_text(json.dumps(record, sort_keys=True))


def fits(module, out, d, cases):
    """One fit per case. A slope term rides on its own absorbed term: the secondreg
    shape (firm##c.z firm#c.z2) is the firm twice, once with z and its intercept
    and once with z2 alone, as the Stata front end passes it."""
    common = (d["y"], d["X"])
    for label, ctor_kw, fes_key, slopes, extra in cases:
        fes = [d["worker"]] + [d[fes_key]] * len(slopes) + [d["year"]]
        terms = [(1 + k, d[v], intercept) for k, (v, intercept) in enumerate(slopes)]
        run_fit(module, out, label, ctor_kw, common, dict(fes=fes, clusters=d["worker"],
                                                          slopes=terms, **extra))


ONE = [("z", True)]
TWO = [("z", True), ("z2", False)]
FAST = dict(tolerance_mode="xhdfe-fast")


def scenario_large(module, out):
    d = slope_panel(20261115, workers=200_000, firms=12_000)
    fits(module, out, d, [
        ("one_cmp", dict(), "firm_raw", ONE, {}),
        ("one_fast", FAST, "firm_raw", ONE, {}),
        ("two_cmp", dict(), "firm_raw", TWO, {}),
        ("two_fast", FAST, "firm_raw", TWO, {}),
        ("weights", dict(), "firm_raw", ONE, dict(weights=d["w"])),
        ("savefe", dict(retain_fes=True, max_iter=3000, **FAST), "firm_raw", ONE, {}),
        ("dense", dict(), "firm_code", ONE, {}),
        ("auto", dict(num_threads=0), "firm_raw", ONE, {}),
    ])
    order = np.argsort(d["firm_raw"], kind="stable")
    s = {k: (np.ascontiguousarray(v[order]) if isinstance(v, np.ndarray) and v.ndim == 1 else v)
         for k, v in d.items()}
    s["X"] = np.asfortranarray(d["X"][order])
    fits(module, out, s, [("sorted", dict(), "firm_raw", ONE, {})])


def scenario_xl(module, out):
    d = slope_panel(20261116, workers=800_000, firms=40_000)
    fits(module, out, d, [("auto_xl", dict(num_threads=0, **FAST), "firm_raw", ONE, {})])


def scenario_small(module, out):
    d = slope_panel(20261117, workers=30_000, firms=2_500)
    fits(module, out, d, [
        ("explicit_small", dict(), "firm_raw", TWO, {}),
        ("auto_small", dict(num_threads=0), "firm_raw", TWO, {}),
    ])


def scenario_boundary(module, out):
    d = slope_panel(20261118, workers=30_000, firms=2_500, boundary=True)
    fits(module, out, d, [
        ("boundary", dict(max_iter=3000, **FAST), "firm_raw", ONE, {}),
    ])


SCENARIOS = {"large": scenario_large, "xl": scenario_xl, "small": scenario_small,
             "boundary": scenario_boundary}
# label -> whether the gate admits the team (explicit count, or 4194304 rows).
TEAM = {"one_cmp": True, "one_fast": True, "two_cmp": True, "two_fast": True, "weights": True,
        "savefe": True, "dense": True, "auto": False, "sorted": True, "auto_xl": True,
        "explicit_small": True, "auto_small": False, "boundary": True}
# The dense firm id takes no index, so the verifier has nothing more to check there.
DENSE = {"dense"}


def child(args) -> int:
    module = load(args.module)
    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    SCENARIOS[args.scenario](module, out)
    return 0


def launch(script, module, out, label, scenario, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("XHDFE_")}
    env.update({"XHDFE_GPU_BACKEND": "cpu", "XHDFE_ABSORPTION_CACHE_MODE": "off",
                "XHDFE_FE_STRUCTURE_MODE": "off", "XHDFE_MOBILITY_MODE": "off",
                "XHDFE_PROFILE_CPU": "1", "OMP_WAIT_POLICY": "passive"})
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
            fail(f"{what}: diagnostics or method differ")

    def teams(r, label):
        return [int(l.split("team=")[1].split()[0]) for l in r.get("lines", []) if f"label={label} " in l]

    def verified(r):
        return sum(1 for l in r.get("lines", []) if "label=sparse_rank_verify" in l and "ok=1" in l)

    only = set(args.only.split(",")) if args.only else None
    with tempfile.TemporaryDirectory(prefix="slope_team_indexers_") as tmp:
        out = Path(tmp)
        for scenario in SCENARIOS:
            if only is not None and scenario not in only:
                continue
            new = records(launch(script, args.module, out, f"{scenario}_new", scenario))
            off = records(launch(script, args.module, out, f"{scenario}_off", scenario, {SWITCH: "0"}))
            base = records(launch(script, args.frozen_module, out, f"{scenario}_base", scenario))
            ver = records(launch(script, args.module, out, f"{scenario}_verify", scenario,
                                 {"XHDFE_SPARSE_RANK_VERIFY": "1"}))
            bver = records(launch(script, args.frozen_module, out, f"{scenario}_bverify", scenario,
                                  {"XHDFE_SPARSE_RANK_VERIFY": "1"}))
            if not new:
                fail(f"{scenario}: no records (vacuous)")
            for label in sorted(new):
                if any(label not in arm for arm in (off, base, ver, bver)):
                    fail(f"{scenario}/{label}: missing in an arm")
                    continue
                if new[label].get("error"):
                    fail(f"{scenario}/{label}: error {new[label]['error']}")
                    continue
                same(new[label], base[label], f"{scenario}/{label}: new against the frozen module")
                same(new[label], off[label], f"{scenario}/{label}: new against the switch off")
                same(new[label], ver[label], f"{scenario}/{label}: new against the verify arm")
                mixed, rank = teams(new[label], "mixed_indexers"), teams(new[label], "slope_rank_indexer")
                mixed_off, rank_off = teams(off[label], "mixed_indexers"), teams(off[label], "slope_rank_indexer")
                if not mixed or not rank:
                    fail(f"{scenario}/{label}: positive control, no mixed_indexers or slope_rank_indexer line")
                    continue
                want_team = TEAM[label]
                if want_team and (min(mixed) <= 1 or min(rank) <= 1):
                    fail(f"{scenario}/{label}: the team was expected, lines {mixed} / {rank}")
                if not want_team and (max(mixed) != 1 or max(rank) != 1):
                    fail(f"{scenario}/{label}: no team was expected, lines {mixed} / {rank}")
                if (mixed_off and max(mixed_off) != 1) or (rank_off and max(rank_off) != 1):
                    fail(f"{scenario}/{label}: the switch off ran a team: {mixed_off} / {rank_off}")
                for l in ver[label].get("lines", []):
                    if "label=sparse_rank_verify" in l and "ok=0" in l:
                        fail(f"{scenario}/{label}: the verifier found a difference: {l}")
                v_new, v_base = verified(ver[label]), verified(bver[label])
                if want_team and label not in DENSE and not v_new > v_base:
                    fail(f"{scenario}/{label}: verified sites {v_new} not above the frozen module's {v_base}")
                print(f"INFO {scenario}/{label}: threads_used {new[label].get('threads_used')} "
                      f"(frozen {base[label].get('threads_used')}), mixed entries {len(mixed)} team "
                      f"{sorted(set(mixed))}, slope ranks {len(rank)} team {sorted(set(rank))}, "
                      f"verified sites {v_new} (frozen {v_base}), method {new[label].get('method')}",
                      flush=True)
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
    args = parser.parse_args()
    if args.child:
        return child(args)
    if not args.frozen_module:
        parser.error("--frozen-module is required")
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
