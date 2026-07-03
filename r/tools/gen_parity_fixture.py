#!/usr/bin/env python3
"""Generate the R<->Python parity reference fixture.

Reads the binary dataset exported by gen_parity_data.R, runs the canonical
spec battery through the reference Python module (build/py_hdfe_v11), and
writes tests/testthat/fixtures/parity_reference.json.

Usage:
  python3 r/tools/gen_parity_fixture.py <datadir> <build_dir> <out_json>
"""
import json
import sys
from pathlib import Path

import numpy as np

DATADIR = Path(sys.argv[1] if len(sys.argv) > 1 else "r/tools/parity_work")
BUILD = Path(sys.argv[2] if len(sys.argv) > 2 else "build")
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else
           "r/xhdfe/tests/testthat/fixtures/parity_reference.json")

sys.path.insert(0, str(BUILD))
import py_hdfe_v11  # noqa: E402


def load(prefix):
    cols = (DATADIR / f"{prefix}_columns.txt").read_text().strip().split(",")
    n = int((DATADIR / f"{prefix}_nrow.txt").read_text().strip())
    out = {}
    for c in cols:
        arr = np.fromfile(DATADIR / f"{prefix}_{c}.bin", dtype="<f8")
        assert arr.size == n, (c, arr.size, n)
        out[c] = arr
    return out


d = load("main")
g = load("grouped")

ids = {k: d[k].astype(np.int32) for k in ("id1", "id2", "id3", "id_hi")}
gid = {k: g[k].astype(np.int32) for k in ("grp", "ind", "year")}

X2 = np.column_stack([d["x1"], d["x2"]])
XIV = np.column_stack([d["x1"], d["x2"], d["xend"]])
Z = np.column_stack([d["ze1"], d["ze2"]])
XG = g["xg"].reshape(-1, 1)


def digest(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "sum": float(np.sum(a)),
            "sumsq": float(np.sum(a * a)),
            "min": float(np.min(a)), "max": float(np.max(a))}


SPECS = []


def spec(name, ctor=None, fit=None):
    SPECS.append({"name": name, "ctor": ctor or {}, "fit": fit or {}})


spec("nofe_unadj", {"se_type": "unadjusted"}, {"X": "X2"})
spec("fe2_unadj", {"se_type": "unadjusted"}, {"X": "X2", "fes": ["id1", "id2"]})
spec("fe2_robust", {"se_type": "robust"}, {"X": "X2", "fes": ["id1", "id2"]})
spec("fe2_cluster1", {"se_type": "cluster"},
     {"X": "X2", "fes": ["id1", "id2"], "clusters": ["id1"]})
spec("fe3_cluster2", {"se_type": "cluster"},
     {"X": "X2", "fes": ["id1", "id2", "id3"], "clusters": ["id1", "id2"]})
spec("fe2_weights_robust", {"se_type": "robust"},
     {"X": "X2", "fes": ["id1", "id2"], "weights": "w"})
spec("iv_fe2_cluster", {"se_type": "cluster"},
     {"X": "XIV", "fes": ["id1", "id2"], "clusters": ["id1"],
      "instruments": True, "endogenous_idx": [2]})
spec("slopes_intercept", {"se_type": "cluster"},
     {"X": "X2", "fes": ["id1", "id2"], "clusters": ["id1"],
      "slopes": [{"fe_index": 1, "values": "z", "include_intercept": True}]})
spec("slopes_only", {"se_type": "robust"},
     {"X": "X2", "fes": ["id1", "id2"],
      "slopes": [{"fe_index": 1, "values": "z", "include_intercept": False}]})
spec("keepsing", {"se_type": "robust", "keepsingletons": True},
     {"X": "X2", "fes": ["id_hi", "id2"]})
spec("dropsing", {"se_type": "robust"}, {"X": "X2", "fes": ["id_hi", "id2"]})
spec("dof_none", {"se_type": "cluster", "dofadjustments": "none"},
     {"X": "X2", "fes": ["id1", "id2"], "clusters": ["id1"]})
spec("ssc_variants",
     {"se_type": "cluster", "ssc_k_fixef": "nonnested", "ssc_g_df": "conventional",
      "ssc_k_adj": False},
     {"X": "X2", "fes": ["id1", "id2"], "clusters": ["id1"]})
spec("tolmode_fast", {"se_type": "unadjusted", "tolerance_mode": "xhdfe-fast"},
     {"X": "X2", "fes": ["id1", "id2"]})
spec("tolmode_strict", {"se_type": "unadjusted", "tolerance_mode": "strict-residual"},
     {"X": "X2", "fes": ["id1", "id2"]})
spec("noconstant", {"se_type": "unadjusted", "fit_intercept": False},
     {"X": "X2"})
spec("level90", {"se_type": "robust", "level": 90.0},
     {"X": "X2", "fes": ["id1", "id2"]})
spec("savefe", {"se_type": "unadjusted", "retain_fes": True},
     {"X": "X2", "fes": ["id1", "id2"]})
spec("groupvar", {"se_type": "unadjusted", "groupvar": True},
     {"X": "X2", "fes": ["id1", "id2"]})
spec("method_mlsmr", {"se_type": "unadjusted", "absorption_method": "mlsmr"},
     {"X": "X2", "fes": ["id1", "id2"]})
spec("method_symgs", {"se_type": "unadjusted", "absorption_method": "symmetric-gauss-seidel",
                      "symmetric_sweep": True},
     {"X": "X2", "fes": ["id1", "id2"]})
# NOTE: no frequency-weight spec here -- the Python binding does not expose
# weights_are_frequencies. The R test suite validates fweights against the
# exact replicated-rows equivalence instead.

MATS = {"X2": X2, "XIV": XIV, "XG": XG}

results = {}
for s in SPECS:
    ctor = dict(s["ctor"])
    fit = s["fit"]
    reg = py_hdfe_v11.HdfeRegressor(**ctor)
    kwargs = {}
    if "fes" in fit:
        kwargs["fes"] = [ids[k] for k in fit["fes"]]
    if "clusters" in fit:
        kwargs["clusters"] = [ids[k] for k in fit["clusters"]]
    if "weights" in fit:
        kwargs["weights"] = d[fit["weights"]]
    if fit.get("instruments"):
        kwargs["instruments"] = Z
        kwargs["endogenous_idx"] = fit["endogenous_idx"]
    if "slopes" in fit:
        kwargs["slopes"] = [
            {"fe_index": t["fe_index"], "values": d[t["values"]],
             "include_intercept": t["include_intercept"]}
            for t in fit["slopes"]
        ]
    reg.fit(d["y"], MATS[fit["X"]], **kwargs)
    entry = {
        "coef": list(map(float, reg.coef_)),
        "se": list(map(float, reg.stderr_)),
        "tvalues": list(map(float, reg.tvalues_)),
        "pvalues": list(map(float, reg.pvalues_)),
        "conf_int": [list(map(float, row)) for row in reg.conf_int_],
        "cov_digest": digest(reg.covariance_),
        "residual_digest": digest(reg.residuals_),
        "nobs": int(reg.nobs_),
        "nobs_full": int(reg.nobs_full_),
        "num_singletons": int(reg.num_singletons_),
        "df_resid": float(reg.df_resid_),
        "df_resid_unadj": float(reg.df_resid_unadj_),
        "df_m": float(reg.df_m_),
        "df_a": float(reg.df_a_),
        "df_a_levels": float(reg.df_a_levels_),
        "df_a_exact": float(reg.df_a_exact_),
        "df_a_nested": float(reg.df_a_nested_),
        "r2": float(reg.r2_),
        "r2_within": float(reg.r2_within_),
        "rss": float(reg.rss_),
        "tss": float(reg.tss_),
        "tss_within": float(reg.tss_within_),
        "saturated": bool(reg.saturated_),
        "converged": bool(reg.converged_),
        "num_iterations": int(reg.num_iterations_),
        "fe_num_levels": list(map(int, reg.fe_num_levels_)),
        "num_clusters": int(reg.num_clusters_),
        "cluster_counts": list(map(int, reg.cluster_counts_)),
        "cluster_scale": float(reg.cluster_scale_),
        "sample_index_digest": digest(reg.sample_index_),
        "method_used": str(reg.absorption_method_used).split(".")[-1],
    }
    if s["name"] == "savefe":
        entry["fe_effect_digests"] = [digest(v) for v in reg.fe_effects_]
        entry["fe_recovery_converged"] = bool(reg.fe_recovery_converged_)
    if s["name"] == "groupvar":
        entry["groupvar_digest"] = digest(reg.groupvar_)
    results[s["name"]] = entry

# Group-level outcome specs.
reg = py_hdfe_v11.HdfeRegressor(se_type="unadjusted")
reg.fit(g["yg"], XG, fes=[gid["year"]], group=gid["grp"])
results["group_only"] = {
    "coef": list(map(float, reg.coef_)),
    "se": list(map(float, reg.stderr_)),
    "nobs": int(reg.nobs_),
    "r2": float(reg.r2_),
    "rss": float(reg.rss_),
    "residual_digest": digest(reg.residuals_),
}

reg = py_hdfe_v11.HdfeRegressor(se_type="unadjusted")
reg.fit(g["yg"], XG, fes=[gid["ind"]], group=gid["grp"],
        individual=gid["ind"], aggregation="mean")
results["group_individual"] = {
    "coef": list(map(float, reg.coef_)),
    "se": list(map(float, reg.stderr_)),
    "nobs": int(reg.nobs_),
    "r2": float(reg.r2_),
    "rss": float(reg.rss_),
}

fes_dec = reg.extract_group_individual_fes(
    g["yg"], XG, [gid["ind"]], gid["grp"], gid["ind"], aggregation="mean")
results["group_fes_decomposition"] = {
    "individual_effects_digest": digest(fes_dec["individual_effects"]),
    "iterations": int(fes_dec["iterations"]),
    "converged": bool(fes_dec["converged"]),
    "mse": float(fes_dec["mse"]),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"specs": results}, indent=1))
print(f"wrote {OUT} with {len(results)} spec entries")
