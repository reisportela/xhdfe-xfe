#!/usr/bin/env python3
"""Validation suite for the xhdfe AKM + leave-out (KSS) module.

Checks the C++ `akm_kss` / `akm_leave_out_set` bindings against:
  1. an independent dense NumPy oracle (pinv/lstsq leverages and quadratic
     forms) replicating the LeaveOutTwoWay (KSS 2020) estimator definition;
  2. a brute-force Python implementation of the leave-out connected set
     (articulation workers of the mover-firm bipartite graph);
  3. exact-vs-JLA convergence and fixed-seed / thread-count determinism;
  4. (optional, --pytwoway) pytwoway as an independent semantic oracle,
     driven in its own venv via subprocess.

Every check prints dataset / oracle / tolerance. Exit code 0 = all pass.

Usage:
  python VALIDATE_AKM_KSS.py [--module-dir build] [--pytwoway VENV_PYTHON]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def load_module(module_dir):
    sys.path.insert(0, os.path.abspath(module_dir))
    import py_hdfe_v11  # noqa: E402

    return py_hdfe_v11


# ---------------------------------------------------------------------------
# Simulated AKM panels
# ---------------------------------------------------------------------------

def sim_akm(n_workers=400, n_periods=5, n_firms=40, p_move=0.35, seed=0,
            n_controls=0, het=False, force_move=False):
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0.0, 0.6, n_workers)
    psi = rng.normal(0.0, 0.4, n_firms)
    rows_i, rows_j = [], []
    for w in range(n_workers):
        j = int(rng.integers(n_firms))
        for _ in range(n_periods):
            rows_i.append(w)
            rows_j.append(j)
            if force_move or rng.random() < p_move:
                nj = int(rng.integers(n_firms - 1))
                j = nj if nj < j else nj + 1
    i = np.asarray(rows_i, dtype=np.int64)
    j = np.asarray(rows_j, dtype=np.int64)
    n = i.size
    X = rng.normal(0.0, 1.0, (n, n_controls)) if n_controls else None
    beta = np.arange(1, n_controls + 1) * 0.1 if n_controls else None
    sd = 0.3 * (1.0 + (np.abs(psi[j]) if het else 0.0))
    y = alpha[i] + psi[j] + rng.normal(0.0, 1.0, n) * sd
    if n_controls:
        y = y + X @ beta
    return y, i, j, X


# ---------------------------------------------------------------------------
# Brute-force leave-out connected set (LeaveOutTwoWay semantics)
# ---------------------------------------------------------------------------

def _components(nodes, edges):
    adj = {v: set() for v in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, comps = set(), []
    for v in nodes:
        if v in seen:
            continue
        comp, stack = set(), [v]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(adj[u] - comp)
        seen |= comp
        comps.append(comp)
    return comps


def bruteforce_leave_out_keep(i, j):
    """Largest connected set -> iterate {drop articulation movers, largest CC}
    -> drop workers observed once. Returns a boolean keep mask."""
    n = i.size
    keep = np.ones(n, dtype=bool)

    def largest_cc(keep):
        nodes = set()
        edges = set()
        for k in range(n):
            if not keep[k]:
                continue
            wn, fn = ("w", int(i[k])), ("f", int(j[k]))
            nodes.add(wn)
            nodes.add(fn)
            edges.add((wn, fn))
        if not nodes:
            return keep
        comps = _components(nodes, edges)
        comps.sort(key=lambda c: (sum(1 for v in c if v[0] == "f"),
                                  sum(1 for k in range(n) if keep[k] and ("f", int(j[k])) in c)),
                   reverse=True)
        best = comps[0]
        out = keep.copy()
        for k in range(n):
            if out[k] and ("f", int(j[k])) not in best:
                out[k] = False
        return out

    keep = largest_cc(keep)
    while True:
        # mover-firm bipartite graph on unique pairs
        pairs = {}
        firms_of = {}
        for k in range(n):
            if keep[k]:
                firms_of.setdefault(int(i[k]), set()).add(int(j[k]))
        movers = {w for w, fs in firms_of.items() if len(fs) >= 2}
        nodes, edges = set(), set()
        for w in movers:
            for f in firms_of[w]:
                nodes.add(("w", w))
                nodes.add(("f", f))
                edges.add((("w", w), ("f", f)))
        if not movers:
            break
        base = len(_components(nodes, edges))
        bad = set()
        for w in movers:
            nodes2 = nodes - {("w", w)}
            edges2 = {e for e in edges if ("w", w) not in e}
            if nodes2 and len(_components(nodes2, edges2)) > base - 0:
                # removing w must not increase the number of components among
                # the remaining vertices
                if len(_components(nodes2, edges2)) > base:
                    bad.add(w)
        if not bad:
            break
        for k in range(n):
            if keep[k] and int(i[k]) in bad:
                keep[k] = False
        keep = largest_cc(keep)
    # drop workers observed once
    counts = {}
    for k in range(n):
        if keep[k]:
            counts[int(i[k])] = counts.get(int(i[k]), 0) + 1
    for k in range(n):
        if keep[k] and counts[int(i[k])] == 1:
            keep[k] = False
    return keep


# ---------------------------------------------------------------------------
# Dense NumPy oracle for the estimator on a given (already kept) sample
# ---------------------------------------------------------------------------

def dense_oracle(y, i, j, X, level):
    """Replicates LeaveOutTwoWay exactly on a leave-out-connected sample."""
    y = np.asarray(y, dtype=float)
    _, iw = np.unique(i, return_inverse=True)
    _, jf = np.unique(j, return_inverse=True)
    n = y.size
    N = iw.max() + 1
    J = jf.max() + 1

    D = np.zeros((n, N))
    D[np.arange(n), iw] = 1.0
    F = np.zeros((n, J))
    F[np.arange(n), jf] = 1.0

    # FWL at the person-year level
    if X is not None and X.shape[1] > 0:
        Z = np.hstack([D, F[:, :-1], X])
        b = np.linalg.lstsq(Z, y, rcond=None)[0]
        y = y - X @ b[N + J - 1:]

    # collapse
    if level == "match":
        keys = iw.astype(np.int64) * J + jf
        uk, inv = np.unique(keys, return_inverse=True)
        M = uk.size
        peso = np.bincount(inv).astype(float)
        ybar = np.bincount(inv, weights=y) / peso
        mw = (uk // J).astype(int)
        mf = (uk % J).astype(int)
    else:
        M = n
        inv = np.arange(n)
        peso = np.ones(n)
        ybar = y.copy()
        mw, mf = iw.copy(), jf.copy()

    sw = np.sqrt(peso)
    A = np.zeros((M, N + J))
    A[np.arange(M), mw] = sw
    A[np.arange(M), N + mf] = sw
    yt = sw * ybar

    G = np.linalg.pinv(A.T @ A)
    b = G @ (A.T @ yt)  # min-norm solution; fits/centered forms invariant
    alpha_full, psi_full = b[:N], b[N:]

    fit = A @ b
    eta = yt - fit
    Pii = np.einsum("ij,jk,ik->i", A, G, A)

    # person-year expanded quadratic-form designs
    Xfe_py = np.zeros((n, N + J))
    Xfe_py[np.arange(n), N + jf] = 1.0
    Xpe_py = np.zeros((n, N + J))
    Xpe_py[np.arange(n), iw] = 1.0

    GA = G @ A.T  # (N+J) x M
    Zfe = Xfe_py @ GA  # n x M
    Zpe = Xpe_py @ GA
    Zfe_c = Zfe - Zfe.mean(axis=0, keepdims=True)
    Zpe_c = Zpe - Zpe.mean(axis=0, keepdims=True)
    Bfe = np.einsum("im,im->m", Zfe_c, Zfe_c)
    Bpe = np.einsum("im,im->m", Zpe_c, Zpe_c)
    Bcov = np.einsum("im,im->m", Zpe_c, Zfe_c)

    # sigma_i with the stayer replacement at match level
    Mii = 1.0 - Pii
    counts_matches = np.bincount(mw, minlength=N)
    stayer_row = counts_matches[mw] < 2 if level == "match" else np.zeros(M, dtype=bool)
    sigma = np.zeros(M)
    ok = ~stayer_row
    sigma[ok] = (yt[ok] - yt.mean()) * eta[ok] / Mii[ok]
    if level == "match" and stayer_row.any():
        T = np.bincount(iw, minlength=N).astype(float)
        fit_py = alpha_full[iw] + psi_full[jf]
        mii_py = 1.0 - 1.0 / T[iw]
        sig_py = (y - y.mean()) * (y - fit_py) / mii_py
        acc = np.bincount(inv, weights=sig_py) / peso
        sigma[stayer_row] = acc[stayer_row]

    left_fe = Xfe_py @ b
    left_pe = Xpe_py @ b
    dof = n - 1.0

    def cc(a, bb):
        return ((a - a.mean()) * (bb - bb.mean())).sum() / dof

    plug = dict(var_psi=cc(left_fe, left_fe), var_alpha=cc(left_pe, left_pe),
                cov_alpha_psi=cc(left_pe, left_fe))
    kss = dict(var_psi=plug["var_psi"] - (Bfe * sigma).sum() / dof,
               var_alpha=plug["var_alpha"] - (Bpe * sigma).sum() / dof,
               cov_alpha_psi=plug["cov_alpha_psi"] - (Bcov * sigma).sum() / dof)
    rss_py = ((y - alpha_full[iw] - psi_full[jf]) ** 2).sum()
    s2ho = rss_py / (n - (N + J - 1)) if n > N + J - 1 else 0.0
    agsu = dict(var_psi=plug["var_psi"] - s2ho * Bfe.sum() / dof,
                var_alpha=plug["var_alpha"] - s2ho * Bpe.sum() / dof,
                cov_alpha_psi=plug["cov_alpha_psi"] - s2ho * Bcov.sum() / dof)
    return dict(plugin=plug, kss=kss, agsu=agsu, pii=Pii, sigma2_ho=s2ho,
                mw=mw, mf=mf, stayer=stayer_row)


def compare_components(tag, got, want, tol):
    for key in ("var_psi", "var_alpha", "cov_alpha_psi"):
        diff = abs(got[key] - want[key])
        scale = max(1.0, abs(want[key]))
        check(f"{tag}:{key}", diff / scale <= tol, f"|Δ|={diff:.3e} (tol {tol:g})")


# ---------------------------------------------------------------------------
# pytwoway oracle (subprocess in its own venv)
# ---------------------------------------------------------------------------

PT_SCRIPT = r"""
import json, sys
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pytwoway as tw

payload = json.load(open(sys.argv[1]))
df = pd.DataFrame({'i': payload['i'], 'j': payload['j'],
                   'y': payload['y'], 't': payload['t']})
mode = payload['mode']
if mode == 'worker_set':
    cp = bpd.clean_params({'connectedness': 'leave_out_worker',
                           'drop_single_stayers': False, 'drop_returns': False,
                           'copy': False, 'verbose': False})
    bdf = bpd.BipartiteDataFrame(df, track_id_changes=True).clean(cp)
    orig = bdf.original_ids()
    def col(name):
        oname = 'original_' + name
        return orig[oname] if oname in orig.columns else orig[name]
    out = {'kept': sorted(set(zip(col('i').tolist(), col('j').tolist())))}
else:
    cp = bpd.clean_params({'connectedness': 'leave_out_observation',
                           'drop_single_stayers': True, 'drop_returns': False,
                           'copy': False, 'verbose': False})
    bdf = bpd.BipartiteDataFrame(df, track_id_changes=True).clean(cp)
    orig = bdf.original_ids()
    def col(name):
        oname = 'original_' + name
        return orig[oname] if oname in orig.columns else orig[name]
    fe_params = tw.fe_params({'he': True,
                              'Q_var': [tw.Q.VarPsi(), tw.Q.VarAlpha()],
                              'Q_cov': [tw.Q.CovPsiAlpha()],
                              'exact_trace_sigma_2': True, 'exact_trace_ho': True,
                              'exact_trace_he': True, 'exact_lev_he': True,
                              'attach_fe_estimates': True,
                              'progress_bars': False})
    fe = tw.FEEstimator(bdf.copy(), fe_params)
    fe.fit(rng=np.random.default_rng(4321))
    res = {k: fe.summary[k] for k in fe.summary}
    # Second fit with approximate (Hutchinson) traces: pytwoway's exact-trace
    # path mis-corrects Q.VarAlpha (it reuses the covariance trace), so the
    # alpha components are cross-checked against the approximate path.
    fe_params_approx = tw.fe_params({'he': True,
                                     'Q_var': [tw.Q.VarPsi(), tw.Q.VarAlpha()],
                                     'Q_cov': [tw.Q.CovPsiAlpha()],
                                     'exact_trace_sigma_2': True,
                                     'exact_trace_ho': False, 'exact_trace_he': False,
                                     'ndraw_trace_ho': 400, 'ndraw_trace_he': 400,
                                     'exact_lev_he': True,
                                     'progress_bars': False})
    fe2 = tw.FEEstimator(bdf, fe_params_approx)
    fe2.fit(rng=np.random.default_rng(4321))
    res_approx = {k: fe2.summary[k] for k in fe2.summary}
    out = {'rows_i': col('i').tolist(),
           'rows_j': col('j').tolist(),
           'rows_y': orig['y'].tolist(),
           'res': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in res.items()},
           'res_approx': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in res_approx.items()}}
json.dump(out, open(sys.argv[2], 'w'))
"""


def run_pytwoway(venv_python, y, i, j, t, mode):
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "pt_run.py")
        fin = os.path.join(td, "in.json")
        fout = os.path.join(td, "out.json")
        with open(script, "w") as fh:
            fh.write(PT_SCRIPT)
        json.dump({"i": i.tolist(), "j": j.tolist(), "y": y.tolist(),
                   "t": t.tolist(), "mode": mode}, open(fin, "w"))
        subprocess.run([venv_python, script, fin, fout], check=True,
                       capture_output=True)
        return json.load(open(fout))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", default="build")
    ap.add_argument("--pytwoway", default=None,
                    help="path to a venv python with pytwoway installed")
    ap.add_argument("--matlab-lot2w", default=None,
                    help="path to a LeaveOutTwoWay clone with CMG installed; "
                         "runs the canonical MATLAB oracle comparison")
    ap.add_argument("--felsdvsimul", default="data/felsdvsimul.dta")
    args = ap.parse_args()

    mod = load_module(args.module_dir)
    print(f"module: {mod.__file__}")
    have = [n for n in ("akm_kss", "akm_leave_out_set") if hasattr(mod, n)]
    check("bindings-present", len(have) == 2, ",".join(have))
    if len(have) != 2:
        sys.exit(1)

    # ---- 1. Brute-force leave-out set on random graphs --------------------
    for seed in range(6):
        y, i, j, _ = sim_akm(n_workers=60, n_periods=3, n_firms=12,
                             p_move=0.25, seed=100 + seed)
        got = mod.akm_leave_out_set(i, j)["keep"]
        want = bruteforce_leave_out_keep(i, j)
        check(f"leaveout-set:sim{seed}", bool((got == want).all()),
              f"kept {int(got.sum())}/{y.size} (bruteforce {int(want.sum())})")

    # ---- 2. Dense oracle: felsdvsimul + simulations ------------------------
    cases = []
    if os.path.exists(args.felsdvsimul):
        import pandas as pd
        df = pd.read_stata(args.felsdvsimul)
        cases.append(("felsdvsimul", df["y"].to_numpy(float),
                      df["i"].to_numpy(np.int64), df["j"].to_numpy(np.int64), None))
        cases.append(("felsdvsimul+X", df["y"].to_numpy(float),
                      df["i"].to_numpy(np.int64), df["j"].to_numpy(np.int64),
                      df[["x1", "x2"]].to_numpy(float)))
    y, i, j, X = sim_akm(n_workers=300, n_periods=4, n_firms=30, p_move=0.3,
                         seed=7, n_controls=2, het=True)
    cases.append(("sim-het+X", y, i, j, X))
    y2, i2, j2, _ = sim_akm(n_workers=250, n_periods=4, n_firms=25,
                            p_move=0.25, seed=11, het=True)
    cases.append(("sim-het", y2, i2, j2, None))

    for name, yy, ii, jj, XX in cases:
        for level in ("match", "obs"):
            r = mod.akm_kss(yy, ii, jj, X=XX, leave_out_level=level,
                            leverages="exact")
            check(f"exact:{name}:{level}:converged", bool(r["converged"]),
                  r["notes"])
            keep = r["sample"]["keep"]
            o = dense_oracle(yy[keep], ii[keep], jj[keep],
                             XX[keep] if XX is not None else None, level)
            tol = 5e-7 if XX is not None else 1e-9
            compare_components(f"exact-vs-dense:{name}:{level}:plugin",
                               r["plugin"], o["plugin"], tol)
            compare_components(f"exact-vs-dense:{name}:{level}:kss",
                               r["kss"], o["kss"], tol)
            compare_components(f"exact-vs-dense:{name}:{level}:agsu",
                               r["agsu"], o["agsu"], tol)
            # row-level leverages (row order: C++ sorts by final labels; the
            # oracle sorts by np.unique codes — compare as sorted multisets
            # keyed by (worker, firm) original ids)
            got = sorted(zip(r["row_worker"].tolist(), r["row_firm"].tolist(),
                             np.round(r["pii"], 9).tolist()))
            uw = np.unique(ii[keep])
            uf = np.unique(jj[keep])
            want = sorted(zip(uw[o["mw"]].tolist(), uf[o["mf"]].tolist(),
                              np.round(o["pii"], 9).tolist()))
            pii_ok = len(got) == len(want) and all(
                a[0] == b[0] and a[1] == b[1] and abs(a[2] - b[2]) <= 1e-7
                for a, b in zip(got, want))
            check(f"exact-vs-dense:{name}:{level}:pii", pii_ok,
                  f"{len(got)} rows, max_pii={r['max_pii']:.4f}")

    # ---- 3. JLA convergence + determinism ----------------------------------
    y, i, j, _ = sim_akm(n_workers=800, n_periods=4, n_firms=60, p_move=0.3,
                         seed=21, het=True)
    exact = mod.akm_kss(y, i, j, leverages="exact")
    errs = []
    for draws in (200, 800, 3200):
        r = mod.akm_kss(y, i, j, leverages="jla", jla_draws=draws, seed=99)
        errs.append(abs(r["kss"]["var_psi"] - exact["kss"]["var_psi"]))
    scale = max(abs(exact["kss"]["var_psi"]), 1e-3)
    check("jla-convergence", all(e <= 0.01 * scale for e in errs),
          f"|jla-exact| var_psi @200/800/3200 draws: "
          f"{errs[0]:.2e}/{errs[1]:.2e}/{errs[2]:.2e} (exact={exact['kss']['var_psi']:.4f}, "
          f"tol 1% of scale)")

    r1 = mod.akm_kss(y, i, j, leverages="jla", jla_draws=200, seed=5, num_threads=1)
    r2 = mod.akm_kss(y, i, j, leverages="jla", jla_draws=200, seed=5, num_threads=8)
    same = all(r1["kss"][k] == r2["kss"][k] for k in r1["kss"])
    check("jla-determinism-threads", same,
          f"1-thread vs 8-thread var_psi: {r1['kss']['var_psi']:.12e} vs "
          f"{r2['kss']['var_psi']:.12e}")
    r3 = mod.akm_kss(y, i, j, leverages="jla", jla_draws=200, seed=5, num_threads=8)
    check("jla-determinism-repeat",
          all(r2["kss"][k] == r3["kss"][k] for k in r2["kss"]))

    # A failed/under-iterated JLA solve can put many working rows at the
    # numerical unit-leverage guard.  Diagnostics must report the affected
    # count once: repeating one sentence per row used to make Stata expand an
    # >645k local macro before it could print the non-convergence warning.
    n_ring = 25_000
    ring_w = np.repeat(np.arange(n_ring, dtype=np.int64), 2)
    ring_f = np.empty(2 * n_ring, dtype=np.int64)
    ring_f[0::2] = np.arange(n_ring, dtype=np.int64)
    ring_f[1::2] = (np.arange(n_ring, dtype=np.int64) + 1) % n_ring
    ring_y = np.sin(np.arange(2 * n_ring, dtype=float))
    r_bad_jla = mod.akm_kss(
        ring_y, ring_w, ring_f, leverages="jla", jla_draws=2, seed=1,
        direct_max_firms=0, cg_tol=1e-30, cg_max_iter=1, num_threads=4)
    n_unit = int(np.count_nonzero(
        (1.0 - np.asarray(r_bad_jla["pii"], dtype=float)) <= 1e-12))
    bounded_note = r_bad_jla["notes"]
    check(
        "jla-unit-leverage-note-is-bounded",
        (not r_bad_jla["converged"] and n_unit > 0 and
         bounded_note.count("leverage ~1 on a non-stayer row") == 1 and
         f"affected working rows: {n_unit}" in bounded_note and
         len(bounded_note) < 4096),
        f"affected={n_unit}, note occurrences="
        f"{bounded_note.count('leverage ~1 on a non-stayer row')}, "
        f"note length={len(bounded_note)}")

    # A controls() FWL fit has its own problem-size thread tuning.  It must
    # not leak that process-wide OpenMP setting into the subsequent KSS
    # solver.  Uncap the solver's work heuristic only for this small fixture
    # so the requested/effective contract is observable without a huge graph.
    old_team = os.environ.get("XHDFE_AKM_TEAM")
    os.environ["XHDFE_AKM_TEAM"] = "0"
    try:
        x_ctrl = np.sin(np.arange(y.size, dtype=float))[:, None]
        rt = mod.akm_kss(y, i, j, X=x_ctrl, leverages="jla", jla_draws=2,
                         seed=5, num_threads=4)
    finally:
        if old_team is None:
            os.environ.pop("XHDFE_AKM_TEAM", None)
        else:
            os.environ["XHDFE_AKM_TEAM"] = old_team
    check("controls-fwl-restores-kss-thread-budget",
          rt["converged"] and rt["fwl_threads_used"] >= 1 and
          rt["threads_used"] == 4,
          f"FWL={rt['fwl_threads_used']}, KSS={rt['threads_used']} (requested 4)")

    # ---- 3d. GPU solver parity (opt-in path; skips on CPU-only builds) ----
    yg, ig, jg, _ = sim_akm(n_workers=800, n_periods=4, n_firms=60,
                            p_move=0.3, seed=21, het=True)
    rgpu = mod.akm_kss(yg, ig, jg, leverages="jla", jla_draws=100, seed=99,
                       direct_max_firms=0, gpu=True)
    if rgpu["gpu_used"]:
        rcpu = mod.akm_kss(yg, ig, jg, leverages="jla", jla_draws=100, seed=99,
                           direct_max_firms=0)
        dmax = max(abs(rgpu["kss"][k] - rcpu["kss"][k]) for k in rcpu["kss"])
        check("gpu:cpu-parity", dmax <= 1e-10, f"max|diff|={dmax:.2e}")
        rgpu2 = mod.akm_kss(yg, ig, jg, leverages="jla", jla_draws=100, seed=99,
                            direct_max_firms=0, gpu=True)
        check("gpu:run-to-run", rgpu["kss"] == rgpu2["kss"])
    else:
        print("[SKIP] gpu checks — CUDA not available in this build")

    # ---- 3c. Component standard errors (M5, leave_out_COMPLETE machinery).
    # Oracle pins recorded 2026-07-06 from leave_out_COMPLETE(do_SE=1,
    # subsample_llr_fit=4, exact) run 3-5x on the committed DGPs (MATLAB
    # NSIM=1000 is unseeded, so the oracle itself carries ~0.5-1.5% MC
    # noise; xhdfe simulations are seeded). theta_c at 1e-9; SEs at 2.5%.
    def dgp_movers(seed=21):
        rng = np.random.default_rng(seed)
        nw, T, nf = 350, 4, 35
        rows_i, rows_j = [], []
        for w in range(nw):
            f = int(rng.integers(nf))
            for t in range(T):
                rows_i.append(w)
                rows_j.append(f)
                nf2 = int(rng.integers(nf - 1))
                f = nf2 if nf2 < f else nf2 + 1
        i = np.array(rows_i, dtype=np.int64)
        j = np.array(rows_j, dtype=np.int64)
        alpha = rng.normal(0, .6, nw)
        psi = rng.normal(0, .4, nf)
        y = alpha[i] + psi[j] + rng.normal(size=i.size) * 0.3 * (1 + np.abs(psi[j]))
        return y, i, j
    ym, im, jm = dgp_movers()
    r = mod.akm_kss(ym, im, jm, leverages="exact", compute_se=True)
    cse = r["component_se"]
    pins = {  # match level: oracle reports firm + covariance only
        "var_psi": (0.172389827899, 0.011662, 0.025),
        "cov_alpha_psi": (-0.0146727400976, 0.0029463, 0.025),
    }
    for key, (th_o, se_o, tol) in pins.items():
        check(f"component-se:movers:theta_{key}",
              abs(float(cse[f"theta_{key}"]) - th_o) <= 1e-9,
              f"{float(cse[f'theta_{key}']):.12g} vs {th_o:.12g}")
        check(f"component-se:movers:se_{key}",
              abs(float(cse[f"se_{key}"]) - se_o) <= tol * se_o,
              f"{float(cse[f'se_{key}']):.6f} vs oracle ~{se_o:.6f}")
    check("component-se:match:var-alpha-missing-oracle-rule",
          np.isnan(float(cse["se_var_alpha"])) and
          np.isnan(float(cse["theta_var_alpha"])) and
          "not identified at match level" in r["notes"], r["notes"])

    # Observation level remains the supported/calibrated var(alpha)
    # inference surface.  Pin every component to the pre-mitigation values so
    # the match-level safety rule cannot perturb healthy obs-level arithmetic.
    r_obs_se = mod.akm_kss(ym, im, jm, leverages="exact", compute_se=True,
                           leave_out_level="obs")
    cse_obs = r_obs_se["component_se"]
    obs_pins = {
        "var_psi": (0.17236228757171249, 0.01165956740789163),
        "cov_alpha_psi": (-0.014674007058685686, 0.002955741533289813),
        "var_alpha": (0.31802478393891886, 0.007378431330052438),
    }
    for key, (th_o, se_o) in obs_pins.items():
        check(f"component-se:obs:theta-{key}-regression-pin",
              abs(float(cse_obs[f"theta_{key}"]) - th_o) <= 5e-12)
        check(f"component-se:obs:se-{key}-regression-pin",
              abs(float(cse_obs[f"se_{key}"]) - se_o) <= 5e-12)
    r2 = mod.akm_kss(ym, im, jm, leverages="exact", compute_se=True)
    check("component-se:deterministic",
          all(float(cse[k]) == float(r2["component_se"][k]) or
              (np.isnan(float(cse[k])) and
               np.isnan(float(r2["component_se"][k]))) for k in cse))

    # Collinear controls are an intentional reduced-model fit, but the
    # omission must be observable instead of looking like a true zero.
    xdup = np.column_stack([np.arange(ym.size, dtype=float) % 17,
                            np.arange(ym.size, dtype=float) % 17])
    r_drop = mod.akm_kss(ym, im, jm, X=xdup, leverages="exact")
    check("controls:collinear-drop-is-audible",
          float(r_drop["beta"][1]) == 0.0 and
          "control column(s) 2 omitted" in r_drop["notes"],
          r_drop["notes"])

    # Fixed KSSMC-2 regression fixture (D3, rep 2): the negative simulated
    # variance clamp stays numerically 0, but is now noted; the leaked
    # [NaN,-1e300] AM sentinel must become [NaN,NaN].
    def dgp_kssmc2():
        for attempt in range(50):
            rg = np.random.default_rng(np.random.SeedSequence([77003, attempt]))
            wi, fj = [], []
            for ww in range(70):
                f1, f2 = rg.choice(15, size=2, replace=False)
                wi += [ww] * 4
                fj += [f1, f1, f2, f2]
            for ss in range(30):
                ww = 70 + ss
                ff = int(rg.integers(15))
                wi += [ww] * 4
                fj += [ff] * 4
            wi = np.asarray(wi, dtype=np.int64)
            fj = np.asarray(fj, dtype=np.int64)
            if int(np.asarray(mod.akm_leave_out_set(wi, fj)["keep"]).sum()) != wi.size:
                continue
            psi0 = rg.normal(0.0, 0.40, size=15)
            alpha0 = np.empty(100)
            for ww in range(100):
                fs = np.unique(fj[wi == ww])
                alpha0[ww] = 0.5 * psi0[fs].mean() + rg.normal(0.0, 0.35)
            sigma0 = 0.12 + 0.55 * rg.uniform(size=wi.size)
            mu0 = alpha0[wi] + psi0[fj]
            er = np.random.default_rng(
                np.random.SeedSequence([881100, 77003, 2])).standard_normal(wi.size)
            return mu0 + sigma0 * er, wi, fj
        raise RuntimeError("could not build KSSMC-2 fixture")

    yc, ic, jc = dgp_kssmc2()
    rc = mod.akm_kss(yc, ic, jc, leave_out_level="obs", leverages="exact",
                     compute_se=True, eigen_diagnostics=True, se_nsim=1000,
                     eig_trace_nsim=100, num_threads=2, seed=424244)
    wc = rc["weak_id"]["cov_alpha_psi"]
    check("component-se:negative-variance-clamp-is-audible",
          float(rc["component_se"]["se_cov_alpha_psi"]) == 0.0 and
          "truncated to zero" in rc["notes"], rc["notes"])
    check("am-ci:undefined-bounds-sanitized-and-audible",
          np.isnan(float(wc["ci_lb"])) and np.isnan(float(wc["ci_ub"])) and
          "bounds are missing" in rc["notes"], rc["notes"])

    # ---- 2b. Weak-identification diagnostics + Andrews-Mikusheva q=1 CIs --
    # eigen_diagnostics=True (leave_out_COMPLETE eigen_diagno path). Oracle =
    # leave_out_COMPLETE(..., eigen_diagno=1, do_SE=1) on this same movers
    # panel (MATLAB run 06jul2026; diary-printed at 5 significant digits).
    # Deterministic pieces (Lindeberg, COV_R1(1,1), curvature, CI bounds, F)
    # agree to print precision / eigAux's internal pcg tol (1e-5); COV_R1(2,2)
    # and gamma^2 carry the 1000-draw simulation noise; |cov12| compared in
    # absolute value (the eigenvector sign is arbitrary and the CI is
    # sign-invariant). Upstream note: leave_out_COMPLETE's SUM_EIG block
    # references an undefined trace_pe for n_of_parameters==2 (and swaps
    # cov/pe for ==3); the oracle runs used a shadow with the pp-consistent
    # assignment -- this only affects the printed eigenvalue shares.
    def am_block(level, oracle):
        r = mod.akm_kss(ym, im, jm, leverages="exact", eigen_diagnostics=True,
                        leave_out_level=level)
        wk = r["weak_id"]
        for key, o in oracle.items():
            g = wk[key]
            tag = f"am-ci:{level}:{key}"
            check(f"{tag}:lindeberg",
                  abs(g["lindeberg_max_x1bar_sq"] - o["lind"]) <= 2e-6,
                  f"{g['lindeberg_max_x1bar_sq']:.7f} vs {o['lind']}")
            check(f"{tag}:cov11",
                  abs(g["cov_r1_11"] - o["c11"]) <= 2e-4 * abs(o["c11"]),
                  f"{g['cov_r1_11']:.6f} vs {o['c11']}")
            check(f"{tag}:|cov12|",
                  abs(abs(g["cov_r1_12"]) - abs(o["c12"])) <=
                  2e-2 * abs(o["c12"]),
                  f"{g['cov_r1_12']:.8f} vs {o['c12']}")
            check(f"{tag}:cov22",
                  abs(g["cov_r1_22"] - o["c22"]) <= 1e-2 * abs(o["c22"]),
                  f"{g['cov_r1_22']:.4e} vs {o['c22']}")
            check(f"{tag}:gamma_sq",
                  abs(g["gamma_sq"] - o["g2"]) <= 1e-2 * o["g2"],
                  f"{g['gamma_sq']:.4e} vs {o['g2']}")
            check(f"{tag}:f_stat",
                  abs(g["f_stat"] - o["f"]) <= 2e-3 * o["f"],
                  f"{g['f_stat']:.5f} vs {o['f']}")
            check(f"{tag}:ci",
                  abs(g["ci_lb"] - o["ci"][0]) <= 5e-5 and
                  abs(g["ci_ub"] - o["ci"][1]) <= 5e-5,
                  f"[{g['ci_lb']:.6f},{g['ci_ub']:.6f}] vs {o['ci']}")
            check(f"{tag}:curvature",
                  abs(g["curvature"] - o["curv"]) <= 2e-3 * o["curv"],
                  f"{g['curvature']:.6f} vs {o['curv']}")
        return r
    # match level: the oracle reports fe + cov only (n_of_parameters forced
    # to 2 at match level regardless of stayers).
    r_am = am_block("match", {
        "var_psi": dict(lind=0.0074311, c11=0.16081, c12=0.00027165,
                        c22=0.00013942, g2=0.00023182, f=1.9895,
                        ci=(0.14926, 0.19588), curv=0.030502),
        "cov_alpha_psi": dict(lind=0.0075409, c11=0.15902, c12=0.00014425,
                              c22=9.297e-06, g2=0.00078631, f=0.12663,
                              ci=(-0.020812, -0.0087256), curv=0.056481),
    })
    # obs level: all three components, validating the pe path.
    am_block("obs", {
        "var_psi": dict(lind=0.0074311, c11=0.16187, c12=0.00026717,
                        c22=0.00013933, g2=0.00023501, f=1.9766,
                        ci=(0.14924, 0.19585), curv=0.030709),
        "cov_alpha_psi": dict(lind=0.0075409, c11=0.15946, c12=0.0001438,
                              c22=9.2893e-06, g2=0.00079135, f=0.12628,
                              ci=(-0.020811, -0.0087296), curv=0.056659),
        "var_alpha": dict(lind=0.0071027, c11=0.15751, c12=0.00018354,
                          c22=5.2741e-05, g2=0.00058785, f=0.61727,
                          ci=(0.30379, 0.33261), curv=0.04859),
    })
    r_am2 = mod.akm_kss(ym, im, jm, leverages="exact", eigen_diagnostics=True,
                        leave_out_level="match")
    check("am-ci:deterministic",
          all(float(r_am["weak_id"][c][k]) == float(r_am2["weak_id"][c][k])
              for c in ("var_psi", "cov_alpha_psi")
              for k in r_am["weak_id"][c]))
    # Hutchinson trace converges to the oracle's estimate: pe trace/NT^2 with
    # 2000 draws vs the oracle's 0.00019086 (its own 100-draw estimate).
    r_tr = mod.akm_kss(ym, im, jm, leverages="exact", eigen_diagnostics=True,
                       leave_out_level="obs", eig_trace_nsim=2000)
    NT_obs = float(r_tr["sample"]["n_obs"])
    wkpe = r_tr["weak_id"]["var_alpha"]
    tr_pe = wkpe["lambda1"] ** 2 / wkpe["eig_share1"] / NT_obs ** 2
    check("am-ci:trace-pe-2000draws",
          abs(tr_pe - 0.00019086) <= 2e-2 * 0.00019086,
          f"{tr_pe:.6e} vs oracle 1.9086e-4")

    # ---- 2c. Frequency weights (match level): the fweighted run equals the
    # row-expanded run (components to machine precision; the JLA streams are
    # keyed per match with person-year counts, so they coincide by
    # construction). Unsupported combinations raise.
    rngw = np.random.default_rng(42)
    wgt = rngw.integers(1, 5, size=ym.size).astype(float)
    idx = np.repeat(np.arange(ym.size), wgt.astype(int))
    Xw = np.column_stack([rngw.normal(size=ym.size),
                          rngw.normal(size=ym.size)])
    for tag, kw in (("exact", dict(leverages="exact")),
                    ("exact+X", dict(leverages="exact", X=Xw)),
                    ("jla", dict(leverages="jla", jla_draws=200))):
        Xa = kw.pop("X", None)
        rw = mod.akm_kss(ym, im, jm, X=Xa, fweights=wgt, **kw)
        rex = mod.akm_kss(ym[idx], im[idx], jm[idx],
                          X=(Xa[idx] if Xa is not None else None), **kw)
        worst = max(abs(float(rw[t][k]) - float(rex[t][k]))
                    for t in ("plugin", "agsu", "kss") for k in rw[t])
        check(f"fweights:{tag}:components==expanded", worst <= 1e-12,
              f"worst |delta|={worst:.2e}")
        check(f"fweights:{tag}:n_obs", int(rw["sample"]["n_obs"]) ==
              int(rex["sample"]["n_obs"]))
    for bad_kw, msg in ((dict(leave_out_level="obs"), "obs"),
                        (dict(compute_se=True), "compute_se"),
                        (dict(Z=np.ones((ym.size, 1))), "lincom")):
        try:
            mod.akm_kss(ym, im, jm, fweights=wgt, **bad_kw)
            check(f"fweights:gate:{msg}", False, "no error raised")
        except RuntimeError:
            check(f"fweights:gate:{msg}", True)
    try:
        mod.akm_kss(ym, im, jm, fweights=wgt + 0.5)
        check("fweights:gate:non-integer", False, "no error raised")
    except RuntimeError:
        check("fweights:gate:non-integer", True)
    # weighted leave-out set: T=1 rule counts person-years (a single row
    # with weight >= 2 is a T>1 worker).
    iw = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    jw = np.array([0, 1, 0, 0, 1], dtype=np.int64)
    fw_t = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    s_unw = mod.akm_leave_out_set(iw, jw)
    s_w = mod.akm_leave_out_set(iw, jw, fweights=fw_t)
    check("fweights:leaveout-T1-rule",
          bool(not s_unw["keep"][2]) and bool(s_w["keep"][2]),
          f"unweighted keep={list(map(bool, s_unw['keep']))} "
          f"weighted keep={list(map(bool, s_w['keep']))}")

    # ---- 2d. Lowess sigma-tilde (llr_fit mode 0 port; optional, no oracle
    # on this workstation -- MATLAB here lacks the Curve Fitting Toolbox, so
    # this is a semantics port validated for determinism and sanity; the
    # binned mode-4 fit stays the validated default).
    r_lw = mod.akm_kss(ym, im, jm, leverages="exact", compute_se=True,
                       leave_out_level="obs", se_sigma_lowess=True)
    r_lw2 = mod.akm_kss(ym, im, jm, leverages="exact", compute_se=True,
                        leave_out_level="obs", se_sigma_lowess=True,
                        num_threads=3)
    cl = r_lw["component_se"]
    check("lowess-se:deterministic-threads",
          all(float(cl[k]) == float(r_lw2["component_se"][k]) for k in cl))
    check("lowess-se:finite-positive",
          all(np.isfinite(float(cl[k])) and float(cl[k]) > 0
              for k in ("se_var_psi", "se_cov_alpha_psi", "se_var_alpha")))
    check("lowess-se:same-scale-as-binned",
          all(0.3 <= float(cl[k]) / float(cse_obs[k]) <= 3.0
              for k in ("se_var_psi", "se_cov_alpha_psi", "se_var_alpha")),
          "ratios " + ",".join(f"{float(cl[k]) / float(cse_obs[k]):.3f}"
                               for k in ("se_var_psi", "se_cov_alpha_psi",
                                         "se_var_alpha")))
    check("lowess-se:thetas-unchanged",
          all(float(cl[f"theta_{k}"]) == float(cse_obs[f"theta_{k}"])
              for k in ("var_psi", "cov_alpha_psi", "var_alpha")))

    # ---- 3a. KSS lincom (Proposition 1): numpy oracle for the numerator and
    # the sigma-weighted denominator; cross-checked against LeaveOutTwoWay's
    # lincom_KSS on a no-returner panel (see PROGRESS; the MATLAB pairing is
    # match-major and misaligns Z for returners, ours pairs original rows).
    y, i, j, _ = sim_akm(n_workers=250, n_periods=4, n_firms=25, p_move=0.25,
                         seed=11, het=True)
    rngz = np.random.default_rng(3)
    Z = (0.5 * rngz.normal(size=25)[j] +
         np.random.default_rng(4).normal(size=y.size)).reshape(-1, 1)
    r = mod.akm_kss(y, i, j, Z=Z, leverages="exact")
    keep = r["sample"]["keep"]
    Zt = np.column_stack([np.ones(int(keep.sum())), Z[keep]])
    num = np.linalg.lstsq(Zt, np.asarray(r["psi"]), rcond=None)[0]
    check("lincom:numerator-vs-numpy",
          abs(float(np.asarray(r["lincom"]["coef"])[0]) - num[1]) <= 1e-10,
          f"coef={float(np.asarray(r['lincom']['coef'])[0]):.8f}")
    check("lincom:finite-se",
          np.isfinite(np.asarray(r["lincom"]["se_kss"])).all() and
          float(np.asarray(r["lincom"]["se_kss"])[0]) > 0)

    # ---- 3b. Front-end helpers (M6 subsampling + M7 exports) ---------------
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import xhdfe.akm as xakm
        have_helpers = True
    except ImportError as exc:
        have_helpers = False
        print(f"[SKIP] helpers — xhdfe.akm unavailable ({exc})")
    if have_helpers:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            xakm.akm_kss(ym, im, jm, leverages="exact", compute_se=True)
        check("helpers:match-var-alpha-warning-surfaced",
              any("not identified at match level" in str(w.message)
                  for w in caught))
        ys, iw, jf, _ = sim_akm(n_workers=200, n_periods=4, n_firms=20,
                                p_move=0.3, seed=13, het=True)
        rec1 = xakm.subsampling_diagnostic(ys, iw, jf, fractions=(0.0, 0.25),
                                           n_reps=2, seed=7, leverages="exact")
        rec2 = xakm.subsampling_diagnostic(ys, iw, jf, fractions=(0.0, 0.25),
                                           n_reps=2, seed=7, leverages="exact")
        check("helpers:subsampling-deterministic", rec1 == rec2,
              f"{len(rec1)} records")
        check("helpers:subsampling-converged",
              all(r["converged"] for r in rec1))
        import pandas as pd
        with tempfile.TemporaryDirectory() as td:
            fr = xakm.to_pytwoway_frame(ys, iw, jf)
            n_lo = int(xakm.leave_out_set(iw, jf)["n_obs"])
            check("helpers:pytwoway-frame", len(fr) == n_lo and
                  set(fr.columns) == {"i", "j", "y", "t"}, f"{len(fr)} rows")
            p = xakm.export_leaveout_csv(os.path.join(td, "lo.csv"), ys, iw, jf)
            back = pd.read_csv(p, header=None, names=["y", "i", "j"])
            check("helpers:leaveout-csv-roundtrip",
                  len(back) == n_lo and abs(back["y"].sum() - fr["y"].sum()) < 1e-9)
            rres = xakm.akm_kss(ys, iw, jf, leverages="exact")
            paths = xakm.export_results(rres, ys, iw, jf,
                                        os.path.join(td, "res"), fmt="csv")
            eff = pd.read_csv(paths[0])
            rows = pd.read_csv(paths[1])
            comps = pd.read_csv(paths[2])
            check("helpers:export-results",
                  len(eff) == int(rres["sample"]["n_obs"]) and
                  len(rows) == int(rres["n_rows"]) and len(comps) == 3)

    # ---- 4. pytwoway cross-checks (optional) -------------------------------
    if args.pytwoway:
        # 4a. leave-out-worker sample equality on a general DGP
        y, i, j, _ = sim_akm(n_workers=200, n_periods=4, n_firms=20,
                             p_move=0.3, seed=31)
        t = np.zeros_like(i)
        pos = {}
        for k in range(i.size):
            t[k] = pos.get(int(i[k]), 0)
            pos[int(i[k])] = int(t[k]) + 1
        pt = run_pytwoway(args.pytwoway, y, i, j, t, "worker_set")
        keep = mod.akm_leave_out_set(i, j)["keep"]
        # pytwoway leave_out_worker does not apply the T>1 filter; drop
        # single-observation workers from their kept pair set for comparison.
        mine = set(zip(i[keep].tolist(), j[keep].tolist()))
        theirs_pairs = set(map(tuple, pt["kept"]))
        kept_workers = {}
        for k in range(i.size):
            if (int(i[k]), int(j[k])) in theirs_pairs:
                kept_workers[int(i[k])] = kept_workers.get(int(i[k]), 0) + 1
        theirs = {(w, f) for (w, f) in theirs_pairs if kept_workers.get(w, 0) > 1}
        # bipartitepandas' leave_out_worker drops only the offending ROWS of an
        # articulation worker; LeaveOutTwoWay (canonical, what xhdfe follows)
        # drops the worker's entire history. Both yield leave-worker-out
        # connected samples; ours must therefore be a subset of theirs.
        check("pytwoway:leave-out-worker-subset", mine <= theirs,
              f"mine {len(mine)} pairs subset of pytwoway(+T>1) {len(theirs)} pairs")

        # 4b. components on an all-movers DGP (no stayers, obs level, both
        # exact): compare plug-in / HO / HE with the (n-1)/n dof rescale.
        y, i, j, _ = sim_akm(n_workers=500, n_periods=4, n_firms=40,
                             p_move=1.0, seed=41, het=True, force_move=True)
        t = np.zeros_like(i)
        pos = {}
        for k in range(i.size):
            t[k] = pos.get(int(i[k]), 0)
            pos[int(i[k])] = int(t[k]) + 1
        pt = run_pytwoway(args.pytwoway, y, i, j, t, "components")
        ri = np.asarray(pt["rows_i"], dtype=np.int64)
        rj = np.asarray(pt["rows_j"], dtype=np.int64)
        ry = np.asarray(pt["rows_y"], dtype=float)
        r = mod.akm_kss(ry, ri, rj, leave_out_level="obs", leverages="exact",
                        prune=False)
        n = ry.size
        resc = (n - 1.0) / n
        pr = pt["res"]
        pra = pt["res_approx"]
        # (source, my table, my key, pytwoway key, abs tolerance). Alpha
        # corrections use pytwoway's approximate-trace path (400 Hutchinson
        # draws; JL noise ~1e-4) because its exact-trace path mis-corrects
        # Q.VarAlpha with the covariance trace. cov_he carries a small
        # documented deviation (~1e-5) of pytwoway's exact quadrant algebra
        # from the canonical LeaveOutTwoWay per-row formula xhdfe implements.
        pairs = [(pr, "plugin", "var_psi", "var(psi)_fe", 1e-8),
                 (pr, "plugin", "var_alpha", "var(alpha)_fe", 1e-8),
                 (pr, "plugin", "cov_alpha_psi", "cov(psi, alpha)_fe", 1e-8),
                 (pr, "agsu", "var_psi", "var(psi)_ho", 1e-8),
                 (pra, "agsu", "var_alpha", "var(alpha)_ho", 1e-3),
                 (pr, "agsu", "cov_alpha_psi", "cov(psi, alpha)_ho", 1e-8),
                 (pr, "kss", "var_psi", "var(psi)_he", 1e-8),
                 (pra, "kss", "var_alpha", "var(alpha)_he", 1e-3),
                 (pr, "kss", "cov_alpha_psi", "cov(psi, alpha)_he", 2e-5)]
        for src, table, key, pk, tol in pairs:
            mine_v = r[table][key] * resc
            their_v = float(src[pk])
            diff = abs(mine_v - their_v)
            check(f"pytwoway:{pk}", diff <= tol,
                  f"mine {mine_v:.8f} vs pytwoway {their_v:.8f} (|Δ|={diff:.2e}, tol {tol:g})")

    # ---- 5. LeaveOutTwoWay (MATLAB, canonical) -----------------------------
    if args.matlab_lot2w:
        import pandas as pd
        import shutil
        lot2w = os.path.abspath(args.matlab_lot2w)
        driver = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              "tools", "akm_kss_matlab_driver.m"))
        mat_cases = []
        if os.path.exists(args.felsdvsimul):
            df = pd.read_stata(args.felsdvsimul).sort_values(["i", "t"])
            mat_cases.append(("felsdvsimul", df["y"].to_numpy(float),
                              df["i"].to_numpy(np.int64), df["j"].to_numpy(np.int64)))
        ym, im, jm, _ = sim_akm(n_workers=300, n_periods=4, n_firms=30,
                                p_move=0.3, seed=7, het=True)
        mat_cases.append(("sim-het", ym, im, jm))
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(driver, os.path.join(td, "akm_kss_matlab_driver.m"))
            for tag, yy, ii, jj in mat_cases:
                stem = os.path.join(td, f"o_{tag}")
                pd.DataFrame({"y": yy, "i": ii, "j": jj}).to_csv(
                    stem + "_in.csv", index=False, header=False)
                cmd = (f"cd('{td}'); akm_kss_matlab_driver('{lot2w}', "
                       f"'{stem}_in.csv', '{stem}')")
                subprocess.run(["matlab", "-batch", cmd], check=True,
                               capture_output=True, timeout=1800)
                ml = list(map(float, open(stem + "_final.txt").read().split()))
                r = mod.akm_kss(yy, ii, jj, leave_out_level="match",
                                leverages="exact")
                for got, want, key in [(r["kss"]["var_psi"], ml[0], "var_psi"),
                                       (r["kss"]["cov_alpha_psi"], ml[1], "cov"),
                                       (r["kss"]["var_alpha"], ml[2], "var_alpha")]:
                    diff = abs(got - want)
                    tol = 1e-8 * max(1.0, abs(want))
                    check(f"matlab-lot2w:{tag}:{key}", diff <= tol,
                          f"mine {got:.10f} vs LeaveOutTwoWay {want:.10f} "
                          f"(|delta|={diff:.2e})")
                # LeaveOutTwoWay exports its leave-out sample + Pii to
                # <stem>.csv (tab-separated): compare rows and leverages.
                lot_out = pd.read_csv(stem + ".csv", sep="\t", header=None,
                                      names=["y", "id", "firmid", "pii"])
                mine_rows = pd.DataFrame({"w": r["row_worker"],
                                          "f": r["row_firm"],
                                          "pii": r["pii"]})
                a = mine_rows.sort_values(["w", "f"]).reset_index(drop=True)
                b = lot_out.sort_values(["id", "firmid"]).reset_index(drop=True)
                ok = (len(a) == len(b) and
                      (a["w"].to_numpy() == b["id"].to_numpy()).all() and
                      (a["f"].to_numpy() == b["firmid"].to_numpy()).all() and
                      np.abs(a["pii"].to_numpy() - b["pii"].to_numpy()).max() <= 1e-8)
                check(f"matlab-lot2w:{tag}:sample+pii", bool(ok),
                      f"{len(a)} vs {len(b)} match rows")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
