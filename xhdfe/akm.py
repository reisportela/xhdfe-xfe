"""AKM + leave-out (KSS) front-end helpers for xhdfe.

Thin Python-idiom layer over the compiled ``py_hdfe_v11.akm_kss`` /
``akm_leave_out_set`` bindings:

- :func:`akm_kss` / :func:`leave_out_set` — passthroughs to the C++ core.
- :func:`subsampling_diagnostic` — Andrews, Gill, Schank & Upward (2012)
  diagnostic: iteratively drop random fractions of movers and track the
  variance components (deterministic seeding).
- :func:`plot_subsampling` — matplotlib rendering of the trajectory.
- :func:`to_pytwoway_frame`, :func:`export_leaveout_csv`,
  :func:`export_results` — interoperability exports (pytwoway /
  LeaveOutTwoWay / generic CSV or Parquet). Exports only: the CRE /
  structural branch is out of scope by design.

Everything here is opt-in and does not touch the existing estimation paths.
"""

from __future__ import annotations

import numpy as np


def _core():
    import sys

    # Reuse an already-imported extension (e.g. a benchmark or validation
    # script that loaded py_hdfe_v11 from a build directory): importing a
    # second copy of the same pybind module would re-register its types.
    core = sys.modules.get("py_hdfe_v11")
    if core is None:
        core = sys.modules.get("xhdfe.py_hdfe_v11")
    if core is None:
        from . import py_hdfe_v11 as core

    if not hasattr(core, "akm_kss"):
        raise ImportError(
            "the compiled xhdfe extension predates the AKM/KSS module; "
            "rebuild the package to use xhdfe.akm"
        )
    return core


def akm_kss(y, worker, firm, X=None, **kwargs):
    """AKM two-way estimation + plug-in/AGSU/KSS variance decomposition.

    See ``help(xhdfe.py_hdfe_v11.akm_kss)`` for the full argument list
    (leave_out_level, leverages, jla_draws, seed, prune, ...).
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    return _core().akm_kss(y, worker, firm, X=X, **kwargs)


def leave_out_set(worker, firm):
    """Largest leave-one-out connected set (KSS / LeaveOutTwoWay semantics)."""
    return _core().akm_leave_out_set(worker, firm)


def subsampling_diagnostic(y, worker, firm, X=None, fractions=(0.0, 0.1, 0.2,
                                                               0.3, 0.4, 0.5),
                           n_reps=3, seed=20260705, **akm_kwargs):
    """Andrews et al. (2012) subsampling diagnostic.

    For each fraction f, drops f of the movers (whole workers, chosen at
    random with deterministic seeding) from the leave-out sample and re-runs
    the full decomposition on the remaining rows. Returns a list of records
    ``{fraction, rep, n_obs, n_movers, plugin, agsu, kss, converged}`` whose
    trajectory as f grows reveals limited-mobility bias in the plug-in
    components (the corrected ones should stay ~flat).
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    worker = np.asarray(worker)
    firm = np.asarray(firm)
    base = leave_out_set(worker, firm)
    keep0 = base["keep"]
    # movers of the leave-out sample
    kw = worker[keep0]
    kf = firm[keep0]
    order = np.argsort(kw, kind="stable")
    uw, start = np.unique(kw[order], return_index=True)
    n_firms_of = np.array([
        np.unique(kf[order][s:e]).size
        for s, e in zip(start, np.append(start[1:], kw.size))
    ])
    movers = uw[n_firms_of >= 2]

    records = []
    for frac in fractions:
        reps = 1 if frac == 0.0 else n_reps
        for rep in range(reps):
            if frac == 0.0:
                mask = keep0.copy()
            else:
                rng = np.random.default_rng(
                    np.random.SeedSequence([int(seed), int(round(frac * 1e6)),
                                            rep]))
                n_drop = int(round(frac * movers.size))
                dropped = rng.choice(movers, size=n_drop, replace=False)
                mask = keep0 & ~np.isin(worker, dropped)
            if mask.sum() < 3:
                continue
            r = akm_kss(y[mask], worker[mask], firm[mask],
                        X=X[mask] if X is not None else None, **akm_kwargs)
            records.append({
                "fraction": float(frac),
                "rep": int(rep),
                "n_obs": int(r["sample"]["n_obs"]),
                "n_movers": int(r["sample"]["n_movers"]),
                "plugin": dict(r["plugin"]),
                "agsu": dict(r["agsu"]),
                "kss": dict(r["kss"]),
                "converged": bool(r["converged"]),
            })
    return records


def plot_subsampling(records, component="var_psi", ax=None):
    """Plot the subsampling trajectory of one component (plug-in vs KSS)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    for table, style in (("plugin", "o--"), ("agsu", "s-."), ("kss", "d-")):
        xs, ys = [], []
        for rec in records:
            xs.append(rec["fraction"])
            ys.append(rec[table][component])
        ax.plot(xs, ys, style, label=table)
    ax.set_xlabel("fraction of movers dropped")
    ax.set_ylabel(component)
    ax.legend()
    return ax


def to_pytwoway_frame(y, worker, firm, keep=None, t=None):
    """Leave-out sample as a pandas long frame (columns i, j, y, t) directly
    consumable by ``bipartitepandas.BipartiteDataFrame`` / pytwoway."""
    import pandas as pd

    y = np.asarray(y, dtype=float)
    worker = np.asarray(worker)
    firm = np.asarray(firm)
    if keep is None:
        keep = leave_out_set(worker, firm)["keep"]
    i = worker[keep]
    j = firm[keep]
    yy = y[keep]
    if t is None:
        # within-worker running period in input order
        df = pd.DataFrame({"i": i, "j": j, "y": yy})
        df["t"] = df.groupby("i").cumcount()
    else:
        df = pd.DataFrame({"i": i, "j": j, "y": yy,
                           "t": np.asarray(t)[keep]})
    return df


def export_leaveout_csv(path, y, worker, firm, keep=None):
    """Headerless CSV (y, worker, firm) of the leave-out sample, sorted by
    worker — the input format LeaveOutTwoWay's ``leave_out_KSS`` expects."""
    df = to_pytwoway_frame(y, worker, firm, keep=keep)
    df = df.sort_values(["i", "t"], kind="stable")
    df[["y", "i", "j"]].to_csv(path, index=False, header=False)
    return path


def export_results(result, y, worker, firm, path_prefix, fmt="csv"):
    """Generic export of an :func:`akm_kss` result.

    Writes ``<prefix>_effects`` (per kept observation: worker, firm, y,
    alpha, psi), ``<prefix>_rows`` (per leave-out row: worker, firm, weight,
    pii, sigma_i) and ``<prefix>_components`` (plug-in/AGSU/KSS table) as CSV
    or Parquet.
    """
    import pandas as pd

    if fmt not in ("csv", "parquet"):
        raise ValueError("fmt must be 'csv' or 'parquet'")
    y = np.asarray(y, dtype=float)
    worker = np.asarray(worker)
    firm = np.asarray(firm)
    keep = result["sample"]["keep"]
    effects = pd.DataFrame({
        "worker": worker[keep],
        "firm": firm[keep],
        "y": y[keep],
        "alpha": result["alpha"],
        "psi": result["psi"],
    })
    rows = pd.DataFrame({
        "worker": result["row_worker"],
        "firm": result["row_firm"],
        "weight": result["row_weight"],
        "pii": result["pii"],
        "sigma_i": result["sigma_i"],
    })
    comps = pd.DataFrame([
        {"estimator": name, **result[name]}
        for name in ("plugin", "agsu", "kss")
    ])
    paths = []
    for tag, df in (("effects", effects), ("rows", rows),
                    ("components", comps)):
        p = f"{path_prefix}_{tag}.{fmt}"
        if fmt == "csv":
            df.to_csv(p, index=False)
        else:
            df.to_parquet(p, index=False)
        paths.append(p)
    return paths
