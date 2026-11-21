#!/usr/bin/env python3
"""Phase 5b of the Stata-path plan: multi-way clustering without the repeated
passes (Verifications/perf_stata_path_20261114/fase5/DESENHO_FASE5B.md, sections
1, 2 and 5, with the conditions of round 23). Every distinct count of a
multi-way fit comes from the meats of the OLS stage, and the pair of two cluster
dimensions is built once per fit and lent to the intercept's meats and to the
reconstruction; every published number must keep its bits.

Builds nothing; never run during a timing campaign. Synthetic data only, made in
each child process from fixed seeds.

    python3 -I tests/cluster_shared_passes_contract.py --module <branch module> \
        --frozen-module <module of a previous build that has phase 5b> [--only NAME] \
        [--n1-hook <tests/analytic_precision/n1_fault_injection.cpp built as a .so>] \
        [--cuda-module M --frozen-cuda-module M] [--harness-proof]

Arms, every scenario in a fresh process per arm (the switch is read once per
process), XHDFE_PROFILE_CPU=1, the CPU backend and the caches off unless a
scenario sets them:
  new     --module, XHDFE_CLUSTER_SHARED_PASSES unset
  off     --module, XHDFE_CLUSTER_SHARED_PASSES=0
  frozen  --frozen-module, also with XHDFE_CLUSTER_SHARED_PASSES=0: a previous
          build that already has phase 5b (17nov2026, point 27: a build from before
          phase 5b is no longer comparable in bits with today's code), whose switch
          restores today's path there as it does in the off arm
new is compared with frozen and with off, bit for bit (dtype, shape, raw bytes),
on every REQUIRED property of tests/sparse_rank_contract.py (b, the complete V,
SE, residuals, num_clusters_, cluster_counts_, cluster_combo_counts_,
cluster_scale_, df_resid_, threads_used_, saved effects, sample_index_, ...), on
the error if any, on the method and on the N05 audit line where the audit runs.

Checks beyond the arms:
  counts    cluster_counts_ and cluster_combo_counts_ against counts made here on
            the estimation sample (sample_index_), mask by mask (masks 1 .. 2^m-1
            ascending, the layout of cluster_combo_counts_), and num_clusters_
            against the minimum over the single dimensions
  branches  the branch every meat takes, predicted from today's rule
            (ols.cpp:1505: minimum >= 0, range <= 50 M and <= 2n give the dense
            branch; there run_frac >= 0.20 gives the sums of contiguous runs, in
            the fixed buffer for p <= 8; the intercept's dense branch has no
            runs; the reconstruction has p + 1 columns): the grid must cover dense
            with runs in both buffers, dense row by row and sparse
  controls  the lines "cpu_profile label=cluster_passes site=S ..." (sites ols,
            counts, intercept, recon, release): off and frozen print, at every
            site that runs, counts=recount and pair=built (today's code) and free
            nothing, and never read a carrier; new prints the path of the route
            table of the design note. With --harness-proof every arm runs the
            frozen module (--module is not read): the run proves that the
            comparisons and the controls above hold on the reference build against
            itself (its switch on in new, off in off and frozen), so that a failure
            of a real run comes from the module under test, not from the harness
  repeats   within an arm, the same fit twice, and a list fitted, changed and
            fitted back, give the same bytes; success, failure, success on one
            regressor gives the first bytes again

Scenarios (design note section 5 and round 23; R12, xfepout, is in the Stata
contract): ids (R1), weights (R2, with clusters only in zero-weight rows), factors
(R3), constant (R4), shift (R5), wide (R6), iv (R7), grouped (R8, sum, mean and
group filtering), dims (R9, the minimum at every position up to m = 4), repeat
(R10), audit (R11, XHDFE_CERTIFY=1), savefe (two clusters, effects compared),
cache (clusters only in rows removed as singletons, with the FE-structure cache
writing then reading), n1 (with --n1-hook: a forced N1 re-entry, observed, and
success, failure, success), cuda (with --cuda-module: complete fits with real GPU
use, compared numerically; the bit-for-bit inference on frozen post-absorption
inputs is tests/cluster_shared_passes_inference.cpp).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparse_rank_contract import (DIAGNOSTICS, REQUIRED, captured_stderr, describe,  # noqa: E402
                                  spread_codes)

SWITCH = "XHDFE_CLUSTER_SHARED_PASSES"
N1_SYMBOL = "_ZNK4hdfe6detail16N1NormalEvidence7requireEd"
LINE = re.compile(r"cpu_profile label=cluster_passes site=(\S+)(.*)")

# Expected cluster_passes lines in the new arm, per pass of the fit (design note,
# table of routes; round 23 for the release of the pair). Each entry is
# (site, counts, pair); ("release", ">0") / ("release", "0") is the size of the pair
# freed after the intercept block; ("ols-wide",) the catch of the OLS stage.
ROUTES = {
    "pair": [("ols", "carrier", "built"), ("counts", "carrier", "none"),
             ("intercept", "carrier", "shared"), ("recon", "carrier", "shared"), ("release", ">0")],
    "pair_norecon": [("ols", "carrier", "built"), ("counts", "carrier", "none"),
                     ("intercept", "carrier", "shared"), ("release", ">0")],
    "saturated": [("ols", "carrier", "built"), ("counts", "carrier", "none"), ("release", ">0")],
    "nopair": [("ols", "carrier", "built"), ("counts", "carrier", "none"), ("release", "0")],
    "m3": [("ols", "carrier", "built"), ("counts", "carrier", "none"),
           ("intercept", "carrier", "built"), ("recon", "carrier", "built"), ("release", "0")],
    "wide_catch": [("ols-wide",), ("ols", "recount", "built"), ("counts", "recount", "built"),
                   ("intercept", "recount", "built"), ("recon", "recount", "built"), ("release", "0")],
    "wide_direct": [("ols", "recount", "built"), ("counts", "recount", "built"),
                    ("intercept", "recount", "built"), ("recon", "recount", "built"), ("release", "0")],
    "wide_catch_noint": [("ols-wide",), ("ols", "recount", "built"), ("counts", "recount", "built"),
                         ("release", "0")],
    "iv_wide": [("ols", "recount", "built"), ("counts", "recount", "built"), ("release", "0")],
    "iv_wide_catch": [("ols-wide",), ("ols", "recount", "built"), ("counts", "recount", "built"),
                      ("release", "0")],
    "noreg": [("counts", "recount", "built"), ("intercept", "recount", "built"), ("release", "0")],
    "m1": [],
    "error": [],
}


def today(route):
    """The off and frozen arms (the switch off): the same sites with today's code
    (nothing from a carrier)."""
    out = []
    for entry in ROUTES[route.split("|")[0]]:
        if entry[0] == "release":
            out.append(("release", "0"))
        elif entry[0] != "ols-wide":  # the catch prints only with a carrier
            out.append((entry[0], "recount", "built"))
    return out


def load(path: str, hook: str | None = None):
    """The module; with an N1 hook, the hook first (RTLD_GLOBAL), pointed at the
    module's own definition of N1NormalEvidence::require (as in
    tests/analytic_precision/n1_fault_injection_contract.py)."""
    import importlib.util
    handle = None
    if hook:
        handle = ctypes.CDLL(str(Path(hook).resolve()), mode=ctypes.RTLD_GLOBAL)
        handle.n1_test_set_original.argtypes = [ctypes.c_void_p]
        handle.n1_test_set_original.restype = ctypes.c_int
        handle.n1_test_hook_address.restype = ctypes.c_void_p
        handle.n1_test_arm.argtypes = [ctypes.c_int]
        handle.n1_test_calls.restype = ctypes.c_int
        handle.n1_test_forwarded.restype = ctypes.c_int
        native = ctypes.CDLL(str(Path(path).resolve()), mode=ctypes.RTLD_LOCAL)
        original = ctypes.cast(getattr(native, N1_SYMBOL), ctypes.c_void_p).value
        assert original != handle.n1_test_hook_address(), "dlsym resolved the hook, not the module"
        assert handle.n1_test_set_original(original) == 1
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, handle


# --- Synthetic data -------------------------------------------------------------

def panel(seed=20261115, workers=20_000, firms=1_500, years=8, k=2, singles=0):
    """A worker-firm-year panel, about 110 k rows, rows grouped by worker. Cluster
    candidates with distinct counts: worker (~20 k), firm (1 500), region (50),
    sector (11), year (8); sparse versions of worker and firm (range ~8.7 n)."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(3, 9, size=workers)
    if singles:
        rows[:singles] = 1  # one-row workers: singletons of the worker effect
    worker = np.repeat(np.arange(workers), rows)
    n = worker.size
    t = np.concatenate([np.arange(r) for r in rows])
    year = (rng.integers(0, years, size=workers)[worker] + t) % years
    home = rng.integers(0, firms, size=workers)
    firm = np.where(rng.random(n) < 0.15, rng.integers(0, firms, size=n), home[worker])
    X = rng.normal(size=(n, k))
    if k > 1:
        X[:, 1] += 0.1 * (firm % 7)
    y = X @ np.linspace(0.5, -0.25, k) + 0.3 * rng.normal(size=n)
    for f in (worker, firm, year):
        y = y + rng.normal(size=int(f.max()) + 1)[f]
    d = {"y": y, "X": np.asfortranarray(X), "n": n, "rng": rng,
         "worker": worker.astype(np.int32), "firm": firm.astype(np.int32),
         "year": year.astype(np.int32), "region": (firm // 30).astype(np.int32),
         "sector": ((firm * 7) % 11).astype(np.int32),
         "const": np.zeros(n, dtype=np.int32),
         "firm_relabel": rng.permutation(firms)[firm].astype(np.int32),
         "firm_sparse": spread_codes(firm, 8.7, rng),
         "worker_sparse": spread_codes(worker, 8.7, rng),
         "w": rng.uniform(0.5, 2.0, size=n), "fw": rng.integers(1, 4, size=n).astype(np.float64),
         "e": rng.normal(size=n), "z1": rng.normal(size=n), "z2": rng.normal(size=n)}
    return d


def shuffled(d, seed=99):
    order = np.random.default_rng(seed).permutation(d["n"])
    out = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray) and value.ndim == 1 and value.size == d["n"]:
            out[key] = np.ascontiguousarray(value[order])
        elif isinstance(value, np.ndarray) and value.ndim == 2:
            out[key] = np.asfortranarray(value[order])
        else:
            out[key] = value
    return out


def ctor(module, **kw):
    base = dict(se_type="cluster", tol=1e-8, max_iter=100000, fit_intercept=True,
                num_threads=4, drop_singletons=True, symmetric_sweep=False,
                absorption_method="auto", tolerance_mode="reghdfe-comparable")
    base.update(kw)
    return module.HdfeRegressor(**base)


# --- Independent counts and predicted branches ------------------------------------

def first_appearance(values):
    _, first, inverse = np.unique(values, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size)
    return rank[inverse.ravel()]


def combined(sample, dims):
    """combine_clusters(): uint32 of each int32, concatenated into a uint64, ids by
    first appearance (ols.cpp:1654)."""
    if len(dims) == 1:
        return sample[dims[0]].astype(np.int64)
    current = sample[dims[0]].astype(np.int64)
    for d in dims[1:]:
        a = current.astype(np.int32).view(np.uint32).astype(np.uint64)
        b = sample[d].astype(np.int32).view(np.uint32).astype(np.uint64)
        current = first_appearance((a << np.uint64(32)) | b)
    return current


def branch(ids, p, intercept=False):
    n = ids.size
    lo, hi = int(ids.min()), int(ids.max())
    span = hi - lo + 1
    if not (lo >= 0 and span <= 50_000_000 and span <= 2 * n):
        return "sparse"
    if intercept:
        return "dense-rows"
    run_frac = np.count_nonzero(ids[1:] == ids[:-1]) / (n - 1) if n > 1 else 0.0
    if run_frac >= 0.20:
        return "dense-runs-fixed" if p <= 8 else "dense-runs-vector"
    return "dense-rows"


def independent(clusters, sample_index, p):
    sample = [np.asarray(c)[sample_index] for c in clusters]
    m = len(sample)
    masks, pred = [], {}
    for mask in range(1, 1 << m):
        dims = [j for j in range(m) if mask >> j & 1]
        ids = combined(sample, dims)
        masks.append(int(np.unique(ids).size))
        pred[str(mask)] = {"ols": branch(ids, p), "intercept": branch(ids, p, True),
                           "recon": branch(ids, p + 1)}
    return {"dims": [masks[(1 << j) - 1] for j in range(m)], "masks": masks}, pred


# --- One fit --------------------------------------------------------------------

def record_of(reg, log: Path, error=None):
    rec = {"error": error}
    if error is None:
        public, missing = {}, []
        for name in REQUIRED:
            try:
                public[name] = describe(getattr(reg, name))
            except AttributeError:
                missing.append(name)
            except Exception as exc:
                public[name] = {"raises": type(exc).__name__}
        rec.update(public=public, missing=missing,
                   diagnostics={k: getattr(reg, k) for k in DIAGNOSTICS if hasattr(reg, k)},
                   method=int(reg.absorption_method_used), threads_used=int(reg.threads_used_))
        values = {}
        for name in ("num_clusters_", "cluster_counts_", "cluster_combo_counts_", "omitted_",
                     "nobs_", "num_singletons_", "gpu_used_", "coef_", "stderr_", "covariance_"):
            try:
                values[name] = np.asarray(getattr(reg, name)).tolist()
            except Exception:
                values[name] = None
        rec["values"] = values
    text = log.read_text().splitlines() if log.exists() else []
    rec["lines"] = [l for l in text if "label=cluster_passes" in l or "XHDFE_N05_AUDIT" in l]
    rec["ols_passes"] = sum(1 for l in text if l.startswith("cpu_profile label=ols "))
    # Without a singleton scan the kept rows came from the FE-structure cache.
    rec["singleton_scan"] = sum(1 for l in text if l.startswith("cpu_profile label=singleton_scan "))
    return rec


def fit(module, out: Path, label, d, clusters, route, ctor_kw=None, fit_kw=None,
        fes="default", X=None, y=None, reg=None, check=True):
    """One fit; the record carries the independent counts and the predicted branches."""
    X = d["X"] if X is None else X
    y = d["y"] if y is None else y
    fes = [d["worker"], d["firm"], d["year"]] if isinstance(fes, str) else fes
    log = out / f"{label}.stderr"
    fit_kw = dict(fit_kw or {})
    error = None
    try:
        reg = reg if reg is not None else ctor(module, **(ctor_kw or {}))
        with captured_stderr(log):
            reg.fit(y, X, fes=fes, clusters=clusters, **fit_kw)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    rec = record_of(reg, log, error)
    rec["route"] = route
    rec["m"] = len(clusters) if isinstance(clusters, list) else 1
    if error is None and check and "group" not in fit_kw:
        omitted = rec["values"].get("omitted_") or []
        k = X.shape[1]
        p = sum(1 for j in range(k) if not (j < len(omitted) and omitted[j]))
        if fes is None and (ctor_kw or {}).get("fit_intercept", True):
            p += 1
        rec["indep"], rec["pred"] = independent(clusters if isinstance(clusters, list) else [clusters],
                                                np.asarray(reg.sample_index_), p)
        rec["p"] = p
    (out / f"{label}.json").write_text(json.dumps(rec, sort_keys=True))
    return reg


FAST = dict(tolerance_mode="xhdfe-fast")


def scenario_ids(module, out, **_):
    d = panel(1)
    s = shuffled(d)
    d8 = panel(2, k=8)
    fit(module, out, "r1_sorted_dense", d, [d["worker"], d["firm"]], "pair")
    fit(module, out, "r1_shuffled_dense", s, [s["worker"], s["firm"]], "pair")
    fit(module, out, "r1_dense_sparse", d, [d["worker"], d["firm_sparse"]], "pair")
    fit(module, out, "r1_sparse_sparse", d, [d["worker_sparse"], d["firm_sparse"]], "pair")
    fit(module, out, "r1_shuffled_sparse", s, [s["worker_sparse"], s["firm_sparse"]], "pair")
    fit(module, out, "r1_g1", d, [d["worker"], d["const"]], "pair")
    fit(module, out, "r1_relabel", d, [d["firm"], d["firm_relabel"]], "pair")
    fit(module, out, "r1_nested", d, [d["firm"], d["region"]], "pair")
    fit(module, out, "r1_p8", d8, [d8["worker"], d8["firm"]], "pair")
    # The public refusal of negative codes (round 23): the meats with INT_MIN,
    # INT_MAX and negatives are reached by the direct C++ calls of
    # tests/cluster_shared_passes_inference.cpp.
    fit(module, out, "r1_negative_public", d, [d["worker"], (d["firm"] - 700).astype(np.int32)], "error",
        check=False)


def scenario_weights(module, out, **_):
    d = panel(3)
    cl = [d["worker"], d["firm_sparse"]]
    fit(module, out, "r2_aweight", d, cl, "pair", fit_kw=dict(weights=d["w"]))
    fit(module, out, "r2_pweight", d, cl, "pair", fit_kw=dict(weights=d["w"], pweights=True))
    fit(module, out, "r2_fweight", d, cl, "pair", fit_kw=dict(weights=d["fw"], fweights=True))
    # Firms 0..9 only in rows of weight zero: removed before the OLS (hr:10936), so
    # absent from every count.
    w0 = np.where(d["firm"] < 10, 0.0, d["w"])
    fit(module, out, "r2_zero_only", d, cl, "pair", fit_kw=dict(weights=w0))


def scenario_factors(module, out, **_):
    d = panel(4)
    cl = [d["worker"], d["firm_sparse"]]
    for df in ("min", "conventional"):
        for adj in (True, False):
            fit(module, out, f"r3_{df}_{'adj' if adj else 'noadj'}", d, cl, "pair",
                ctor_kw=dict(ssc_g_df=df, ssc_g_adj=adj))
    fit(module, out, "r3_legacy", d, cl, "pair", ctor_kw=dict(stats_style="legacy"))


def scenario_constant(module, out, **_):
    d = panel(5)
    cl = [d["worker"], d["firm_sparse"]]
    fit(module, out, "r4_noconstant", d, cl, "nopair", ctor_kw=dict(fit_intercept=False))
    fit(module, out, "r4_no_fe", d, cl, "nopair", fes=None)
    X3 = np.asfortranarray(np.column_stack([d["X"], d["X"][:, 0] + d["X"][:, 1]]))
    # An omitted regressor (no reconstruction: hr:13615 widens V). The clusters are
    # firm and region: with worker and a firm id the multiway intercept fallback
    # (var <= 0) crashes today in every arm, cand_sp5a and 2.31.1 alike
    # (hr:1446-1449, a 3x3 COD solved against a 2-vector; reported, not touched
    # in this phase: Verifications/perf_stata_path_20261114/fase5/repro_omitido_dois_clusters.py).
    fit(module, out, "r4_omitted", d, [d["firm"], d["region"]], "pair_norecon", X=X3)
    Xw = np.asfortranarray(np.random.default_rng(55).normal(size=int(d["worker"].max()) + 1)[d["worker"]][:, None])
    fit(module, out, "r4_no_regressors", d, cl, "noreg", X=Xw)
    # Saturated: 10 workers of two rows, the worker effect and 10 regressors, with
    # clusters that do not nest the worker (no nested adjustment of the degrees of
    # freedom): df_r = 20 - 10 - 10 = 0, so neither the intercept nor the
    # reconstruction runs and the pair is freed unused.
    rng = np.random.default_rng(56)
    wk = np.repeat(np.arange(10), 2).astype(np.int32)
    c1 = (np.arange(20) % 5).astype(np.int32)
    c2 = (np.arange(20) % 3).astype(np.int32)
    Xs = np.asfortranarray(rng.normal(size=(20, 10)))
    ys = Xs @ rng.normal(size=10) + rng.normal(size=10)[wk]
    fit(module, out, "r4_saturated", {"X": Xs, "y": ys}, [c1, c2], "saturated", fes=[wk])


def scenario_shift(module, out, **_):
    d = panel(6)
    Xs = np.asfortranarray(np.column_stack([d["X"][:, 0], 1e6 + d["X"][:, 1]]))
    fit(module, out, "r5_shift", d, [d["worker"], d["firm_sparse"]], "nopair", fes=None, X=Xs)


def scenario_wide(module, out, **_):
    d = panel(7)
    cl = [d["worker"], d["firm_sparse"]]
    # Near-collinear regressors (1 - r ~ 5e-9, kept by the 1e-6 rank rule): the
    # normal equations go to the wide route directly (rcond < 1e-8).
    Xn = np.asfortranarray(np.column_stack([d["X"][:, 0], d["X"][:, 0] + 1e-4 * d["e"]]))
    fit(module, out, "r6_direct", d, cl, "wide_direct|wide_catch", X=Xn, ctor_kw=dict(max_iter=3000, **FAST))
    # A response scaled to 1e-147 without a recovered constant: the meats fall below
    # the floor of ols.cpp:185, the OLS stage goes to the wide route through the
    # catch, and no reconstruction follows (ill-conditioned on purpose).
    fit(module, out, "r6_tiny_noconstant", d, cl, "wide_catch_noint|nopair", y=d["y"] * 1e-147,
        ctor_kw=dict(max_iter=3000, fit_intercept=False, **FAST))
    # The same scale with the constant recovered: the reconstruction's meat (the
    # constant's column, sums of residuals within worker clusters nested in the
    # worker effect) leaves its range; since 23e9d5f27 the reconstruction repeats it
    # after an exact power-of-two change of the residuals' units
    # (docs/LICOES_COVARIANCIA_20261115.md, "Gama na reconstrução"), and the fit
    # succeeds in every arm (it used to end without estimates: finding of the design
    # note, section 4).
    fit(module, out, "r6_recon_range", d, cl, "pair", y=d["y"] * 1e-140, ctor_kw=dict(max_iter=3000, **FAST))
    # 17nov2026 (point 27): the reconstruction can still refuse. At 2^-496.5 the OLS
    # stage takes the wide route through the catch, and the reconstruction, past its
    # change of units, cannot represent its result: the fit ends without estimates in
    # every arm after reaching the reconstruction (between 2^-496 and 2^-497 here;
    # from 2^-498 the OLS stage itself fails, as r6_wide_fails).
    fit(module, out, "r6_recon_refused", d, cl, "error", y=d["y"] * 2.0 ** -496.5,
        ctor_kw=dict(max_iter=3000, **FAST), check=False)
    # A response at 1e-160: the wide route itself fails (an error in every arm).
    fit(module, out, "r6_wide_fails", d, cl, "error", y=d["y"] * 1e-160, ctor_kw=dict(max_iter=3000, **FAST),
        check=False)


def scenario_iv(module, out, **_):
    d = panel(8)
    cl = [d["worker"], d["firm_sparse"]]
    endog = d["z1"] + 0.5 * d["z2"] + 0.5 * d["e"]
    X = np.asfortranarray(np.column_stack([endog, d["X"][:, 1]]))
    Z = np.asfortranarray(np.column_stack([d["z1"], d["z2"]]))
    fit(module, out, "r7_iv", d, cl, "pair", X=X, fit_kw=dict(instruments=Z, endogenous_idx=[0]))
    # An instrument almost collinear with the exogenous regressor: the wide route
    # with the recovered constant captured (no intercept site, no reconstruction).
    zc = d["X"][:, 1] + 1e-7 * d["z1"]
    Xc = np.asfortranarray(np.column_stack([zc + 0.1 * d["e"], d["X"][:, 1]]))
    fit(module, out, "r7_iv_wide", d, cl, "iv_wide|iv_wide_catch", X=Xc,
        fit_kw=dict(instruments=np.asfortranarray(zc[:, None]), endogenous_idx=[0]))


def group_data(seed=11, n_groups=30_000, pool=15_000):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 5, size=n_groups)
    group = np.repeat(np.arange(n_groups), sizes).astype(np.int64)
    individual = np.concatenate([rng.choice(pool, size=s, replace=False) for s in sizes])
    # A few individuals seen once: their groups' singleton filtering.
    individual[rng.choice(individual.size, size=300, replace=False)] = pool + np.arange(300)
    gfe = spread_codes((np.arange(n_groups) % 700)[group], 9.0, rng)
    c1 = ((np.arange(n_groups) % 700)[group]).astype(np.int32)
    c2 = (group // 13).astype(np.int32)
    gy = rng.normal(size=n_groups)[group]
    gx = np.asfortranarray(rng.normal(size=(n_groups, 2))[group])
    return {"y": gy, "X": gx, "group": group, "individual": individual.astype(np.int64),
            "fes": [gfe, individual.astype(np.int32)], "c1": c1, "c2": c2}


def scenario_grouped(module, out, **_):
    g = group_data()
    for agg in ("mean", "sum"):
        fit(module, out, f"r8_group_{agg}", g, [g["c1"], g["c2"]], "pair", fes=g["fes"],
            fit_kw=dict(group=g["group"], individual=g["individual"], aggregation=agg))


def scenario_dims(module, out, **_):
    d = panel(9)
    w, f, r, s = d["worker"], d["firm_sparse"], d["region"], d["sector"]
    fit(module, out, "r9_m1", d, [w], "m1")
    fit(module, out, "r9_m2_min0", d, [f, w], "pair")
    fit(module, out, "r9_m2_min1", d, [w, f], "pair")
    for pos in range(3):
        order = [w, f]
        order.insert(pos, r)
        fit(module, out, f"r9_m3_min{pos}", d, order, "m3")
    for pos in range(4):
        order = [w, f, r]
        order.insert(pos, s)
        fit(module, out, f"r9_m4_min{pos}", d, order, "m3")


def scenario_repeat(module, out, **_):
    d = panel(10)
    a = [d["worker"], d["firm_sparse"]]
    fit(module, out, "r10_twice_1", d, a, "pair")
    fit(module, out, "r10_twice_2", d, a, "pair")
    fit(module, out, "r10_alt_a", d, a, "pair")
    fit(module, out, "r10_alt_b", d, [d["worker"], d["region"]], "pair")
    fit(module, out, "r10_alt_c", d, a, "pair")
    # The same list object: its content changed and changed back between fits.
    lst = [d["worker"].copy(), d["firm_sparse"].copy()]
    fit(module, out, "r10_list_1", d, lst, "pair")
    lst[1][:] = d["region"]
    fit(module, out, "r10_list_2", d, lst, "pair")
    lst[1][:] = d["firm_sparse"]
    fit(module, out, "r10_list_3", d, lst, "pair")
    fit(module, out, "r10_auto", d, a, "pair", ctor_kw=dict(num_threads=0))
    fit(module, out, "r10_t1", d, a, "pair", ctor_kw=dict(num_threads=1))
    # Success, failure, success on one regressor object.
    reg = ctor(module)
    fit(module, out, "r10_sfs_1", d, a, "pair", reg=reg)
    fit(module, out, "r10_sfs_2", d, [d["worker"], (d["firm"] - 700).astype(np.int32)], "error",
        reg=reg, check=False)
    fit(module, out, "r10_sfs_3", d, a, "pair", reg=reg)


def scenario_audit(module, out, **_):
    d = panel(12)
    fit(module, out, "r11_audit", d, [d["worker"], d["firm_sparse"]], "pair")


def scenario_savefe(module, out, **_):
    d = panel(13)
    fit(module, out, "savefe_two", d, [d["worker"], d["firm_sparse"]], "pair",
        ctor_kw=dict(retain_fes=True, max_iter=3000, **FAST))


def scenario_cache(module, out, **_):
    # Firms 0..9 only in one-row workers (singletons of the worker effect, removed
    # before the fit); the FE-structure cache writes on the first fit, the second
    # reads the kept rows from it.
    d = panel(14, singles=400)
    single_rows = np.bincount(d["worker"])[d["worker"]] == 1
    firm = np.where(single_rows, np.arange(d["n"]) % 10, np.maximum(d["firm"], 10)).astype(np.int32)
    cl = [d["worker"], spread_codes(firm, 8.7, np.random.default_rng(141))]
    fes = [d["worker"], firm, d["year"]]
    os.environ["XHDFE_FE_STRUCTURE_CACHE"] = str(out / "fe_structure.cache")
    os.environ["XHDFE_FE_STRUCTURE_MODE"] = "auto"
    fit(module, out, "cache_write", d, cl, "pair", fes=fes)
    fit(module, out, "cache_read", d, cl, "pair", fes=fes)
    # The same with firms 10..19 only in rows of weight zero (another cache file:
    # the signature includes the weights).
    w0 = np.where((firm >= 10) & (firm < 20), 0.0, d["w"])
    os.environ["XHDFE_FE_STRUCTURE_CACHE"] = str(out / "fe_structure_w0.cache")
    fit(module, out, "cache_w0_write", d, cl, "pair", fes=fes, fit_kw=dict(weights=w0))
    fit(module, out, "cache_w0_read", d, cl, "pair", fes=fes, fit_kw=dict(weights=w0))
    os.environ["XHDFE_FE_STRUCTURE_MODE"] = "off"


def scenario_n1(module, out, hook=None, **_):
    if hook is None:
        return
    d = panel(15)
    a = [d["worker"], d["firm_sparse"]]
    X3 = np.asfortranarray(np.column_stack([d["X"], d["X"][:, 0] + d["X"][:, 1]]))
    for label, clusters, route, kw in (("n1_pair", a, "pair", {}),
                                       ("n1_omitted", [d["firm"], d["region"]], "pair_norecon", dict(X=X3))):
        hook.n1_test_arm(1)
        fit(module, out, label, d, clusters, route, **kw)
        rec = json.loads((out / f"{label}.json").read_text())
        rec["hook"] = [hook.n1_test_calls(), hook.n1_test_forwarded()]
        (out / f"{label}.json").write_text(json.dumps(rec, sort_keys=True))
    reg = ctor(module)
    for label, armed, route in (("n1_sfs_1", 0, "pair"), ("n1_sfs_2", 2, "error"), ("n1_sfs_3", 0, "pair")):
        hook.n1_test_arm(armed)
        fit(module, out, label, d, a, route, reg=reg, check=armed == 0)
        rec = json.loads((out / f"{label}.json").read_text())
        rec["hook"] = [hook.n1_test_calls(), hook.n1_test_forwarded()]
        (out / f"{label}.json").write_text(json.dumps(rec, sort_keys=True))
    hook.n1_test_arm(0)


def scenario_cuda(module, out, **_):
    d = panel(16)
    fit(module, out, "cuda_pair", d, [d["worker"], d["firm_sparse"]], "pair")
    fit(module, out, "cuda_m3", d, [d["worker"], d["firm_sparse"], d["region"]], "m3")


SCENARIOS = {"ids": scenario_ids, "weights": scenario_weights, "factors": scenario_factors,
             "constant": scenario_constant, "shift": scenario_shift, "wide": scenario_wide,
             "iv": scenario_iv, "grouped": scenario_grouped, "dims": scenario_dims,
             "repeat": scenario_repeat, "audit": scenario_audit, "savefe": scenario_savefe,
             "cache": scenario_cache, "n1": scenario_n1, "cuda": scenario_cuda}
# The N1 evidence (N1NormalEvidence::require, the hook's target) is checked only
# with the audit on (hr:9976).
EXTRA_ENV = {"audit": {"XHDFE_CERTIFY": "1"}, "n1": {"XHDFE_CERTIFY": "1"}}
# Pairs that must give the same bytes inside one arm.
SAME_IN_ARM = [("r10_twice_1", "r10_twice_2"), ("r10_alt_a", "r10_alt_c"),
               ("r10_list_1", "r10_list_3"), ("r10_sfs_1", "r10_sfs_3"), ("r10_twice_1", "r10_alt_a"),
               ("r10_list_1", "r10_alt_a"), ("n1_sfs_1", "n1_sfs_3")]


def child(args) -> int:
    module, hook = load(args.module, args.n1_hook if args.scenario == "n1" else None)
    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    SCENARIOS[args.scenario](module, out, hook=hook)
    return 0


def launch(script, module, out, label, scenario, extra=None, hook=None, backend="cpu"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("XHDFE_")}
    env.update({"XHDFE_GPU_BACKEND": backend, "XHDFE_ABSORPTION_CACHE_MODE": "off",
                "XHDFE_FE_STRUCTURE_MODE": "off", "XHDFE_MOBILITY_MODE": "off",
                "XHDFE_PROFILE_CPU": "1", "OMP_WAIT_POLICY": "passive"})
    env.update(EXTRA_ENV.get(scenario, {}))
    if extra:
        env.update(extra)
    command = [sys.executable, "-B", "-I", str(script), "--child", "--module", str(module),
               "--scenario", scenario, "--label", label, "--out", str(out)]
    if hook:
        command += ["--n1-hook", str(hook)]
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{label}: child failed ({done.returncode}): {done.stderr[-2000:]}")
    return out / label


def records(folder: Path):
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


def parse(lines):
    out = []
    for l in lines:
        m = LINE.search(l)
        if not m:
            continue
        site, rest = m.group(1), dict(kv.split("=", 1) for kv in m.group(2).split() if "=" in kv)
        if site == "release":
            out.append(("release", ">0" if int(rest.get("pair_rows", "0")) > 0 else "0"))
        elif rest.get("route") == "wide":
            out.append(("ols-wide",))
            if int(rest.get("ready", "0")) or int(rest.get("pair_rows", "0")):
                out.append(("ols-wide-left-state", rest.get("ready"), rest.get("pair_rows")))
        else:
            out.append((site, rest.get("counts"), rest.get("pair")))
    return out


def parent(args) -> int:
    script = Path(__file__).resolve()
    failures = []
    covered = set()

    def fail(message):
        failures.append(message)
        print("FAIL", message, flush=True)

    def same(a, b, what, numeric=False):
        if a.get("error") or b.get("error"):
            if a.get("error") != b.get("error"):
                fail(f"{what}: errors differ: {a.get('error')} != {b.get('error')}")
            return
        for r in (a, b):
            if r.get("missing"):
                fail(f"{what}: missing properties {r['missing']}")
        if numeric:
            return
        diff = sorted(k for k in REQUIRED if a["public"].get(k) != b["public"].get(k))
        if diff:
            fail(f"{what}: differs in {diff}")
        if a.get("diagnostics") != b.get("diagnostics") or a.get("method") != b.get("method"):
            fail(f"{what}: diagnostics or method differ")
        audit_a = [l for l in a.get("lines", []) if "XHDFE_N05_AUDIT" in l]
        audit_b = [l for l in b.get("lines", []) if "XHDFE_N05_AUDIT" in l]
        if audit_a != audit_b:
            fail(f"{what}: audit lines differ")

    def consistent(rec, arm, what):
        """Inside one pass, a consumer reads the carrier only after the OLS stage
        produced it; nothing reads it after the catch sent the OLS to the wide route."""
        produced = False
        for g in parse(rec.get("lines", [])):
            if g[0] == "ols" and len(g) == 3:
                produced = g[1] == "carrier"
            elif g[0] == "ols-wide":
                produced = False
            elif g[0] == "release":
                produced = False
            elif len(g) == 3 and (g[1] == "carrier" or g[2] == "shared") and not produced:
                fail(f"{what} [{arm}]: {g[0]} read a carrier no OLS stage had produced")

    def control(rec, arm, what):
        got = sorted(map(tuple, parse(rec.get("lines", []))))
        route = rec.get("route", "error")
        passes = max(1, rec.get("ols_passes", 1)) if route != "error" else 1
        if rec.get("error"):
            # A failed fit: the sites it reached, in order, consistent with the arm;
            # with the switch off (off and frozen) no carrier is ever read.
            if arm in ("off", "frozen") and any(len(g) == 3 and (g[1] == "carrier" or g[2] == "shared")
                                                for g in got):
                fail(f"{what} [{arm}]: the switch off read a carrier: {got}")
            consistent(rec, arm, what)
            return "failed fit (sites reached: " + (", ".join(g[0] for g in parse(rec.get("lines", []))) or "none") + ")"
        if arm in ("off", "frozen"):
            options = [today(r) * passes for r in route.split("|")]
            options = [o for i, o in enumerate(options) if o not in options[:i]]
            label = "today's path (counts=recount pair=built)"
        else:
            options = [ROUTES[r] * passes for r in route.split("|")]
            label = f"route {route}"
        consistent(rec, arm, what)
        for want in options:
            if got == sorted(map(tuple, want)):
                return label if "|" not in route else f"{label} (matched {options.index(want)})"
        fail(f"{what} [{arm}]: control lines {got} != expected {options} ({label})")
        return label

    only = set(args.only.split(",")) if args.only else None
    with tempfile.TemporaryDirectory(prefix="cluster_shared_passes_") as tmp:
        out = Path(tmp)
        for scenario in SCENARIOS:
            if only is not None and scenario not in only:
                continue
            if scenario == "n1" and not args.n1_hook:
                print("SKIP n1: no --n1-hook", flush=True)
                continue
            if scenario == "cuda" and not args.cuda_module:
                print("SKIP cuda: no --cuda-module", flush=True)
                continue
            numeric = scenario == "cuda"
            mod, frozen = ((args.cuda_module, args.frozen_cuda_module) if numeric
                           else (args.module, args.frozen_module))
            if args.harness_proof:
                mod = frozen  # every arm on the reference build
            backend = "cuda" if numeric else "cpu"
            hook = args.n1_hook if scenario == "n1" else None
            new = records(launch(script, mod, out, f"{scenario}_new", scenario, hook=hook, backend=backend))
            off = records(launch(script, mod, out, f"{scenario}_off", scenario, {SWITCH: "0"}, hook=hook,
                                 backend=backend))
            base = records(launch(script, frozen, out, f"{scenario}_frozen", scenario, {SWITCH: "0"}, hook=hook,
                                  backend=backend))
            if not new:
                fail(f"{scenario}: no records (vacuous)")
            for label in sorted(new):
                if label not in off or label not in base:
                    fail(f"{scenario}/{label}: missing in an arm")
                    continue
                r_new, r_off, r_base = new[label], off[label], base[label]
                same(r_new, r_base, f"{scenario}/{label}: new against frozen", numeric)
                same(r_new, r_off, f"{scenario}/{label}: new against off", numeric)
                if numeric and not r_new.get("error"):
                    # CUDA absorption is not bit-reproducible between runs (AUDITORIA_FASE5A
                    # line 30): complete GPU fits are compared numerically; the bit-for-bit
                    # inference on frozen post-absorption inputs is the C++ test's.
                    for r, arm in ((r_new, "new"), (r_off, "off"), (r_base, "frozen")):
                        if r["values"].get("gpu_used_") is not True:
                            fail(f"{scenario}/{label} [{arm}]: no real GPU use")
                        if (r["values"]["cluster_combo_counts_"] != r_base["values"]["cluster_combo_counts_"]
                                or r["values"]["num_clusters_"] != r_base["values"]["num_clusters_"]):
                            fail(f"{scenario}/{label} [{arm}]: counts differ from frozen")
                        for name, rtol in (("coef_", 1e-8), ("stderr_", 1e-7), ("covariance_", 1e-7)):
                            a = np.asarray(r["values"][name], dtype=float)
                            b = np.asarray(r_base["values"][name], dtype=float)
                            if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=0, equal_nan=True):
                                fail(f"{scenario}/{label} [{arm}]: {name} beyond rtol {rtol} of frozen")
                labels = [control(r, arm, f"{scenario}/{label}")
                          for r, arm in ((r_new, "new"), (r_off, "off"), (r_base, "frozen"))]
                if label == "r6_recon_refused":
                    # The refusal is the reconstruction's: every arm reached it and
                    # none freed the pair after it.
                    for r, arm in ((r_new, "new"), (r_off, "off"), (r_base, "frozen")):
                        sites = [g[0] for g in parse(r.get("lines", []))]
                        if not r.get("error") or "recon" not in sites or "release" in sites:
                            fail(f"{scenario}/{label} [{arm}]: not a refusal of the reconstruction "
                                 f"(error {bool(r.get('error'))}, sites {sites})")
                if label.startswith("r8_") and not r_new.get("error"):
                    # Group filtering happened (singleton individuals): the grouped
                    # counts themselves are compared across the arms only.
                    for r, arm in ((r_new, "new"), (r_off, "off"), (r_base, "frozen")):
                        if not (r["values"].get("num_singletons_") or 0) > 0:
                            fail(f"{scenario}/{label} [{arm}]: no group was filtered")
                if label in ("cache_read", "cache_w0_read"):
                    for r, arm in ((r_new, "new"), (r_off, "off"), (r_base, "frozen")):
                        if r.get("singleton_scan") != 0:
                            fail(f"{scenario}/{label} [{arm}]: the FE-structure cache was not read")
                if label in ("cache_write", "cache_w0_write"):
                    for r, arm in ((r_new, "new"), (r_off, "off"), (r_base, "frozen")):
                        if r.get("singleton_scan") != 1:
                            fail(f"{scenario}/{label} [{arm}]: no singleton scan before the cache write")
                if "hook" in r_new:
                    want_calls = {"n1_pair": [2, 1], "n1_omitted": [2, 1], "n1_sfs_1": [1, 1],
                                  "n1_sfs_2": [2, 0], "n1_sfs_3": [1, 1]}[label]
                    for r, arm in ((r_new, "new"), (r_off, "off"), (r_base, "frozen")):
                        if r.get("hook") != want_calls:
                            fail(f"{scenario}/{label} [{arm}]: N1 hook calls {r.get('hook')} != {want_calls}")
                    if label in ("n1_pair", "n1_omitted") and r_new.get("ols_passes") != 2:
                        fail(f"{scenario}/{label}: the N1 re-entry was not observed "
                             f"({r_new.get('ols_passes')} OLS passes)")
                info = ""
                if not r_new.get("error") and "indep" in r_new:
                    v = r_new["values"]
                    ind = r_new["indep"]
                    if v["cluster_counts_"] != ind["dims"] or v["cluster_combo_counts_"] != ind["masks"]:
                        if not (r_new["m"] == 1 and v["cluster_counts_"] == ind["dims"]):
                            fail(f"{scenario}/{label}: counts {v['cluster_counts_']} / "
                                 f"{v['cluster_combo_counts_']} != independent {ind['dims']} / {ind['masks']}")
                    if v["num_clusters_"] != min(ind["dims"]):
                        fail(f"{scenario}/{label}: num_clusters {v['num_clusters_']} != min {min(ind['dims'])}")
                    for mask, pred in r_new["pred"].items():
                        covered.add(pred["ols"])
                        if r_new["route"] in ("pair",):
                            covered.add(pred["recon"])
                    info = (f"counts {ind['dims']} masks {ind['masks']} p {r_new.get('p')} branches "
                            + " ".join(f"{k}:{v['ols']}/{v['intercept']}/{v['recon']}"
                                       for k, v in sorted(r_new["pred"].items(), key=lambda kv: int(kv[0]))))
                print(f"INFO {scenario}/{label}: {r_new.get('error', {}) and r_new['error'].get('type') or 'ok'} "
                      f"threads_used {r_new.get('threads_used')} passes {r_new.get('ols_passes')} "
                      f"controls new={labels[0]} off={labels[1]} frozen={labels[2]} {info}", flush=True)
            for a, b in SAME_IN_ARM:
                for arm_name, arm in (("new", new), ("off", off), ("frozen", base)):
                    if a in arm and b in arm:
                        ra, rb = arm[a], arm[b]
                        if ra.get("error") or rb.get("error"):
                            fail(f"{scenario}: {a}/{b} [{arm_name}]: an error")
                            continue
                        diff = sorted(k for k in REQUIRED if ra["public"].get(k) != rb["public"].get(k)
                                      and k not in ("generation_",))
                        if diff:
                            fail(f"{scenario}: {a} and {b} differ inside the {arm_name} arm in {diff}")
        need = {"dense-runs-fixed", "dense-runs-vector", "dense-rows", "sparse"}
        if (only is None or "ids" in only) and not need <= covered:
            fail(f"branch coverage: {sorted(need - covered)} never predicted")
        print(f"INFO branches covered: {sorted(covered)}", flush=True)
    print("PASS" if not failures else f"{len(failures)} FAILURES", flush=True)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--module", required=True)
    parser.add_argument("--frozen-module")
    parser.add_argument("--cuda-module")
    parser.add_argument("--frozen-cuda-module")
    parser.add_argument("--n1-hook")
    parser.add_argument("--scenario", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--only", default="")
    parser.add_argument("--harness-proof", action="store_true",
                        help="every arm runs the frozen module: the comparisons and the controls must "
                             "hold on the reference build against itself")
    args = parser.parse_args()
    if args.child:
        return child(args)
    if not args.frozen_module:
        parser.error("--frozen-module is required")
    if bool(args.cuda_module) != bool(args.frozen_cuda_module):
        parser.error("--cuda-module and --frozen-cuda-module go together")
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
