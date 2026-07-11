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


def _id_codes(x):
    """Worker/firm ids as dense int32 codes for the compiled core.

    The core relabels ids to dense indices internally and only accepts int32,
    so ids that fall outside int32 range (e.g. NISS/NIF person codes) or that
    are non-integer / string labels are compacted to 0..N-1 codes here. Because
    the core relabels anyway, this never changes the estimates — it only keeps
    large or non-integer ids from being rejected.

    Returns ``(codes, uniques)`` where ``codes`` is a length-n int32 vector and
    ``uniques`` maps a code back to the caller's original id (``uniques[codes]
    == x``), or ``None`` when the ids were already valid int32 and passed
    through unchanged.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.ravel()
    i32 = np.iinfo(np.int32)
    if (
        np.issubdtype(x.dtype, np.integer)
        and x.size
        and x.min() >= i32.min
        and x.max() <= i32.max
    ):
        return np.ascontiguousarray(x, dtype=np.int32), None
    # Reject missing / non-finite ids instead of fusing them into one spurious
    # worker/firm node (mirrors the R front-end, which stops on missing ids).
    if np.issubdtype(x.dtype, np.floating) and not np.all(np.isfinite(x)):
        raise ValueError("worker/firm ids contain missing or non-finite values")
    uniques, codes = np.unique(x, return_inverse=True)
    return np.ascontiguousarray(np.ravel(codes), dtype=np.int32), uniques


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

    Advanced performance environment variables (defaults are tuned; none
    changes the default numeric output):

    - ``XHDFE_AKM_TEAM`` caps the OpenMP team size of the per-iteration solver
      regions. The default caps it by the edge work so a large thread pool does
      not oversubscribe small/medium graphs (the dominant speed lever below
      ~10M rows); ``0`` restores the uncapped team, ``k`` forces ``k`` threads.
    - ``XHDFE_AKM_JLA_BLOCK`` overrides the JLA multi-RHS block size (default 8;
      ``0`` selects the pre-2.14 sequential solver, last-ulp different).
    - ``XHDFE_AKM_SE_BLOCK`` does the same for the component-SE / eigen-
      diagnostics / lincom solves (default 8; ``0`` = sequential).
    - ``XHDFE_AKM_SCATTER_CSR`` (default on) selects the parallel CSR-ordered
      Rademacher scatter at scale; ``0`` restores the sequential scatter.

    The leverage and SE solves are batched without changing the estimator or
    solver tolerances. Different thread/backend reduction schedules can differ
    at the last-ulp level. A ``RuntimeWarning`` is emitted if the decomposition
    does not converge (check ``res['converged']``) or if ``notes`` contains an
    inferential warning (collinear control omission, a truncated negative
    simulated variance, or an undefined AM interval).

    Under the canonical ``leave_out_COMPLETE`` rule, match-level component
    inference reports only ``var_psi`` and ``cov_alpha_psi``. Therefore
    ``se_var_alpha`` and its AM diagnostics are missing at match level even on
    movers-only samples; use ``leave_out_level='obs'`` for var(alpha)
    inference. Point decompositions remain available at both levels.

    Pass ``verbose=1`` to print phase-by-phase progress to stderr (leave-out
    set, FWL, solver choice, JLA draws d/D with elapsed time and an ETA, SE
    simulations, eigen diagnostics) — useful on multi-hour panels to see
    where the estimation is. Output only; results are unaffected.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    wc, wu = _id_codes(worker)
    fc, fu = _id_codes(firm)
    res = _core().akm_kss(y, wc, fc, X=X, **kwargs)
    # When ids were recoded, the core echoes back the compact codes in the
    # per-row id vectors; map them to the caller's original ids so row_worker /
    # row_firm keep their documented meaning.
    if isinstance(res, dict):
        if wu is not None and res.get("row_worker") is not None:
            res["row_worker"] = wu[np.asarray(res["row_worker"])]
        if fu is not None and res.get("row_firm") is not None:
            res["row_firm"] = fu[np.asarray(res["row_firm"])]
        # Surface non-convergence loudly, matching the Stata front-end
        # (di as err) and the Gelbach front-end's warning — a silently
        # returned non-converged decomposition is a footgun.
        note = str(res.get("notes", "")).strip()
        if res.get("converged") is False:
            import warnings

            warnings.warn(
                "xhdfe.akm.akm_kss: the AKM/KSS decomposition did not "
                "converge — results are unreliable."
                + (f" notes: {note}" if note else ""),
                RuntimeWarning,
                stacklevel=2,
            )
        elif "warning:" in note.lower():
            import warnings

            warnings.warn(
                "xhdfe.akm.akm_kss: inferential diagnostic. " + note,
                RuntimeWarning,
                stacklevel=2,
            )
    return res


def leave_out_set(worker, firm):
    """Largest leave-one-out connected set (KSS / LeaveOutTwoWay semantics)."""
    wc, _ = _id_codes(worker)
    fc, _ = _id_codes(firm)
    return _core().akm_leave_out_set(wc, fc)


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
    sorted_worker = kw[order]
    sorted_firm = kf[order]
    uw, start = np.unique(sorted_worker, return_index=True)
    n_firms_of = np.array([
        np.unique(sorted_firm[s:e]).size
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
