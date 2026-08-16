"""Zero-coupling maketables plug-in hooks for fitted xhdfe results.

`maketables <https://github.com/py-econometrics/maketables>`_ renders
publication-ready regression tables from fitted models.  Its plug-in format is
duck-typed: a result class becomes tabulatable by exposing a small set of
``__maketables_*`` attributes.  Nothing in this module imports maketables, so
xhdfe gains no dependency on it and no coupling to its release cycle -- a user
who has both installed can pass an xhdfe result straight to ``mt.ETable`` and
maketables discovers the hooks on its own.

The hooks are attached to both public result surfaces:

* ``FormulaRegressor`` (the :func:`xhdfe.feols` frontend), which carries the
  coefficient names, the dependent-variable name, the fixed-effect names and
  the cluster names.
* the native ``HdfeRegressor`` (the array API), which has none of that naming
  metadata; its coefficients are reported as ``b0``, ``b1``, ... and its
  dependent variable as ``y``.

pandas is imported lazily here, exactly as in :mod:`xhdfe._formula`, so the
hooks add nothing to the import cost of the array API.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


_ATTACHED_FLAG = "_xhdfe_maketables_attached"

# Canonical maketables statistic keys this backend answers.  Keys outside this
# set resolve to None, which maketables renders as an omitted row.
_CANONICAL_STATS = ("N", "r2", "r2_within", "rmse", "n_clusters", "se_type")

# xhdfe-specific rows, available on request via ``model_stats=[...]``.  They are
# not offered by default because a table mixing xhdfe with another backend would
# show them as blanks for the other models.
_EXTRA_STATS = ("N_full", "n_singletons", "df_absorbed")

# Only the xhdfe-specific keys need labels; maketables already labels the
# canonical ones, and overriding those would desynchronize mixed tables.
_STAT_LABELS = {
    "N_full": "Observations (before singletons)",
    "n_singletons": "Singletons dropped",
    "df_absorbed": "Absorbed d.f.",
}

_DEFAULT_STAT_KEYS = ["N", "r2", "r2_within"]


def _pandas():
    """Import pandas on demand, mirroring the formula frontend's lazy policy."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas ships with maketables
        raise ImportError(
            "Rendering an xhdfe result with maketables requires pandas. Install "
            "it with `python -m pip install pandas`, or `python -m pip install "
            "'xhdfe[formula]'` for the full formula frontend."
        ) from exc
    return pd


def _coefficient_names(model: Any) -> tuple[str, ...]:
    """Return coefficient names, falling back to positional names.

    The array API never sees column names, so a fit made through it is reported
    as ``b0``, ``b1``, ... rather than being refused.
    """
    names = getattr(model, "coef_names_", None)
    count = len(np.asarray(model.coef_))
    if names is None:
        return tuple(f"b{index}" for index in range(count))
    names = tuple(str(name) for name in names)
    if len(names) != count:
        raise RuntimeError("coefficient names are not aligned with native results")
    return names


def _fixed_effect_names(model: Any) -> tuple[str, ...]:
    """Return fixed-effect names, or an empty tuple when there are none."""
    names = getattr(model, "fe_names_", None)
    if names is None:
        return ()
    return tuple(str(name) for name in names)


def _absorbs_fixed_effects(model: Any) -> bool:
    """Report whether the fit actually absorbed any fixed effect.

    The array API carries no fixed-effect names, so fall back to the native
    level counts.  This gates the within-R2, which the core reports equal to the
    overall R2 when nothing was absorbed.
    """
    if _fixed_effect_names(model):
        return True
    levels = getattr(model, "fe_num_levels_", None)
    if levels is None:
        return False
    try:
        return len(levels) > 0
    except TypeError:
        return False


def _cluster_names(model: Any) -> tuple[str, ...]:
    names = getattr(model, "cluster_names_", None)
    if names is None:
        return ()
    return tuple(str(name) for name in names)


def _se_type_label(model: Any) -> Optional[str]:
    """Describe the standard errors in maketables' vocabulary.

    Clustered fits are spelled ``by: firm`` to match the string pyfixest reports,
    so a table mixing the two backends stays consistent.  ``homoskedastic`` is
    reported as ``iid`` for the same reason.
    """
    if getattr(model, "num_clusters_", 0):
        names = _cluster_names(model)
        return ("by: " + "+".join(names)) if names else "by: cluster"
    se_type = getattr(model, "se_type_", None)
    if se_type is None:
        return None
    return "iid" if se_type in ("homoskedastic", "unadjusted") else str(se_type)


def coef_table(model: Any):
    """Return the coefficient table in maketables' canonical column names.

    Built from the native result arrays rather than from
    ``FormulaRegressor.tidy`` so that the array API is served by the same code
    path; the two agree by construction and a test pins that.
    """
    pd = _pandas()
    names = _coefficient_names(model)
    columns = {
        "b": np.asarray(model.coef_, dtype=float),
        "se": np.asarray(model.stderr_, dtype=float),
        "t": np.asarray(model.tvalues_, dtype=float),
        "p": np.asarray(model.pvalues_, dtype=float),
    }
    conf_int = np.asarray(model.conf_int_, dtype=float)
    if conf_int.ndim == 2 and conf_int.shape[1] == 2:
        columns["ci95l"] = conf_int[:, 0]
        columns["ci95u"] = conf_int[:, 1]
    return pd.DataFrame(columns, index=pd.Index(names, name="Coefficient"))


def depvar(model: Any) -> str:
    """Return the dependent-variable name, or ``y`` for an array-API fit."""
    formula = getattr(model, "formula_", None)
    if not formula or "~" not in formula:
        return "y"
    return formula.split("~", 1)[0].strip() or "y"


def fixef_string(model: Any) -> Optional[str]:
    """Return the absorbed fixed effects as ``'firm+year'``, or None."""
    names = _fixed_effect_names(model)
    return "+".join(names) if names else None


def stat(model: Any, key: str) -> Any:
    """Return a model-level statistic by maketables key, or None if unavailable."""
    if key == "N":
        nobs = getattr(model, "nobs_", None)
        return None if nobs is None else int(nobs)
    if key == "r2":
        return _finite(getattr(model, "r2_", None))
    if key == "r2_within":
        # The core sets the within-R2 equal to the overall R2 when nothing was
        # absorbed; reporting it there would put a meaningless row in the table.
        if not _absorbs_fixed_effects(model):
            return None
        return _finite(getattr(model, "r2_within_", None))
    if key == "rmse":
        rss = getattr(model, "rss_", None)
        nobs = getattr(model, "nobs_", None)
        if rss is None or not nobs:
            return None
        # sqrt(RSS/N), the convention pyfixest reports.  Note df_resid_ is the
        # inference denominator (G-1 under clustering), not a residual count.
        return float(np.sqrt(rss / nobs))
    if key == "n_clusters":
        clusters = getattr(model, "num_clusters_", 0)
        return int(clusters) if clusters else None
    if key == "se_type":
        return _se_type_label(model)
    if key == "N_full":
        nobs_full = getattr(model, "nobs_full_", None)
        return None if nobs_full is None else int(nobs_full)
    if key == "n_singletons":
        singletons = getattr(model, "num_singletons_", None)
        return None if singletons is None else int(singletons)
    if key == "df_absorbed":
        df_a = getattr(model, "df_a_", None)
        return None if df_a is None else int(df_a)
    return None


def _finite(value: Any) -> Optional[float]:
    """Drop NaN fit statistics, which the core reports for a constant outcome."""
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else value


def supported_stats(model: Any) -> set:
    """Return the statistic keys that resolve to a value for this fit.

    A helper for callers and tests, not part of the plug-in format: maketables
    probes the canonical keys itself and never reads a ``supported_stats`` hook
    from a plug-in model.
    """
    keys = _CANONICAL_STATS + _EXTRA_STATS
    return {key for key in keys if stat(model, key) is not None}


def vcov_info(model: Any) -> dict:
    """Return variance-estimator metadata for maketables' table notes."""
    info: dict = {"se_type": _se_type_label(model)}
    names = _cluster_names(model)
    if names:
        info["cluster_var"] = "+".join(names)
    return info


def var_labels(model: Any) -> Optional[dict]:
    """Return variable labels captured from the estimation frame, if any.

    Populated when the data carried maketables-style labels, which is what
    ``maketables.import_dta`` writes when reading a labelled Stata file.
    """
    labels = getattr(model, "var_labels_", None)
    if not labels:
        return None
    return dict(labels)


def stat_labels(model: Any) -> dict:
    """Return display labels for the xhdfe-specific statistics."""
    return dict(_STAT_LABELS)


def default_stat_keys(model: Any) -> list:
    """Return the rows shown when the caller does not pass ``model_stats``."""
    return list(_DEFAULT_STAT_KEYS)


def attach(cls: type) -> type:
    """Install the maketables plug-in hooks on a result class, once.

    Idempotent, so it is safe to call on every core load and on the cached
    formula-regressor class.
    """
    if cls.__dict__.get(_ATTACHED_FLAG, False):
        return cls
    cls.__maketables_coef_table__ = property(coef_table)
    cls.__maketables_depvar__ = property(depvar)
    cls.__maketables_fixef_string__ = property(fixef_string)
    cls.__maketables_vcov_info__ = property(vcov_info)
    cls.__maketables_var_labels__ = property(var_labels)
    cls.__maketables_stat_labels__ = property(stat_labels)
    cls.__maketables_default_stat_keys__ = property(default_stat_keys)
    cls.__maketables_stat__ = stat
    setattr(cls, _ATTACHED_FLAG, True)
    return cls


__all__ = ["attach", "coef_table", "depvar", "fixef_string", "stat", "supported_stats"]
