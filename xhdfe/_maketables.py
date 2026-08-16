"""Zero-coupling maketables hooks for fitted xhdfe results.

``maketables`` discovers compatible models through duck-typed
``__maketables_*`` attributes.  This module implements that protocol without
importing maketables.  pandas is also imported only when a coefficient table is
actually requested, so the native array API keeps its existing import profile.

Formula fits retain their coefficient, dependent-variable, fixed-effect, and
cluster names.  Native array fits have no such metadata, so the hooks use
positional coefficient names and explicit generic fixed-effect names rather
than silently presenting an absorbed model as one without fixed effects.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


_ATTACHED_FLAG = "_xhdfe_maketables_attached"
_MISSING = object()

_CANONICAL_STATS = ("N", "r2", "r2_within", "rmse", "n_clusters", "se_type")
_EXTRA_STATS = ("N_full", "n_singletons", "df_absorbed")
_STAT_LABELS = {
    "N_full": "Observations (before singletons)",
    "n_singletons": "Singletons dropped",
    "df_absorbed": "Absorbed d.f.",
}
_DEFAULT_STAT_KEYS = ["N", "r2", "r2_within"]


def _pandas():
    """Import pandas only when maketables requests tabular output."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - maketables itself needs pandas
        raise ImportError(
            "Rendering an xhdfe result with maketables requires pandas. Install "
            "pandas, or install `xhdfe[formula]` for the formula frontend."
        ) from exc
    return pd


def _coefficient_names(model: Any) -> tuple[str, ...]:
    names = getattr(model, "coef_names_", None)
    count = len(np.asarray(model.coef_))
    if names is None:
        return tuple(f"b{index}" for index in range(count))
    names = tuple(str(name) for name in names)
    if len(names) != count:
        raise RuntimeError("coefficient names are not aligned with native results")
    return names


def _fixed_effect_names(model: Any) -> tuple[str, ...]:
    names = getattr(model, "fe_names_", None)
    if names is not None:
        return tuple(str(name) for name in names)
    levels = getattr(model, "fe_num_levels_", None)
    if levels is None:
        return ()
    try:
        return tuple(f"fe{index + 1}" for index in range(len(levels)))
    except TypeError:
        return ()


def _cluster_names(model: Any) -> tuple[str, ...]:
    names = getattr(model, "cluster_names_", None)
    if names is not None:
        return tuple(str(name) for name in names)
    counts = getattr(model, "cluster_counts_", None)
    if counts is None:
        return ()
    try:
        count = len(counts)
    except TypeError:
        return ()
    if count == 1:
        return ("cluster",)
    return tuple(f"cluster{index + 1}" for index in range(count))


def _absorbs_fixed_effects(model: Any) -> bool:
    return bool(_fixed_effect_names(model))


def _se_type_label(model: Any) -> Optional[str]:
    if getattr(model, "num_clusters_", 0):
        names = _cluster_names(model)
        return ("by: " + "+".join(names)) if names else "by: cluster"
    se_type = getattr(model, "se_type_", None)
    if se_type is None:
        return None
    text = str(getattr(se_type, "name", se_type)).lower()
    if text in {"homoskedastic", "classical", "unadjusted", "unadj", "ols"}:
        return "iid"
    if text in {"robust", "hc1", "heteroskedastic"}:
        return "robust"
    return text


def coef_table(model: Any):
    """Return the canonical maketables coefficient columns.

    ``conf_int_`` is deliberately not exposed as ``ci95l``/``ci95u`` because
    xhdfe supports arbitrary confidence levels and the native result currently
    does not retain the requested level as public metadata.
    """
    pd = _pandas()
    columns = {
        "b": np.asarray(model.coef_, dtype=float),
        "se": np.asarray(model.stderr_, dtype=float),
        "t": np.asarray(model.tvalues_, dtype=float),
        "p": np.asarray(model.pvalues_, dtype=float),
    }
    return pd.DataFrame(
        columns,
        index=pd.Index(_coefficient_names(model), name="Coefficient"),
    )


def _formula_lhs(formula: str) -> Optional[str]:
    """Return the text before the top-level formula separator."""
    depth = 0
    quote: Optional[str] = None
    escaped = False
    for index, char in enumerate(formula):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\" and quote != "`":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"', "`"):
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "~" and depth == 0:
            return formula[:index].strip()
    return None


def depvar(model: Any) -> str:
    formula = getattr(model, "formula_", None)
    if not formula:
        return "y"
    lhs = _formula_lhs(str(formula))
    if not lhs:
        return "y"
    if len(lhs) >= 2 and lhs[0] == lhs[-1] == "`":
        return lhs[1:-1]
    return lhs


def fixef_string(model: Any) -> Optional[str]:
    names = _fixed_effect_names(model)
    return "+".join(names) if names else None


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def stat(model: Any, key: str) -> Any:
    """Return a model statistic by maketables key, or ``None`` if unavailable."""
    if key == "N":
        nobs = getattr(model, "nobs_", None)
        return None if nobs is None else int(nobs)
    if key == "r2":
        return _finite(getattr(model, "r2_", None))
    if key == "r2_within":
        if not _absorbs_fixed_effects(model):
            return None
        return _finite(getattr(model, "r2_within_", None))
    if key == "rmse":
        rss = getattr(model, "rss_", None)
        nobs = getattr(model, "nobs_", None)
        if rss is None or not nobs:
            return None
        return _finite(np.sqrt(float(rss) / float(nobs)))
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


def supported_stats(model: Any) -> set[str]:
    keys = _CANONICAL_STATS + _EXTRA_STATS
    return {key for key in keys if stat(model, key) is not None}


def vcov_info(model: Any) -> dict[str, str]:
    info: dict[str, str] = {}
    se_type = _se_type_label(model)
    if se_type is not None:
        info["se_type"] = se_type
    names = _cluster_names(model)
    if names:
        info["cluster_var"] = "+".join(names)
    return info


def var_labels(model: Any) -> Optional[dict[str, str]]:
    labels = getattr(model, "var_labels_", None)
    return dict(labels) if labels else None


def stat_labels(model: Any) -> dict[str, str]:
    del model
    return dict(_STAT_LABELS)


def default_stat_keys(model: Any) -> list[str]:
    del model
    return list(_DEFAULT_STAT_KEYS)


def attach(cls: type) -> type:
    """Install the plug-in hooks idempotently and without partial mutation."""
    if cls.__dict__.get(_ATTACHED_FLAG, False):
        return cls
    attributes = {
        "__maketables_coef_table__": property(coef_table),
        "__maketables_depvar__": property(depvar),
        "__maketables_fixef_string__": property(fixef_string),
        "__maketables_vcov_info__": property(vcov_info),
        "__maketables_var_labels__": property(var_labels),
        "__maketables_stat_labels__": property(stat_labels),
        "__maketables_default_stat_keys__": property(default_stat_keys),
        "__maketables_stat__": stat,
        _ATTACHED_FLAG: True,
    }
    previous: list[tuple[str, Any]] = []
    try:
        for name, value in attributes.items():
            previous.append((name, cls.__dict__.get(name, _MISSING)))
            setattr(cls, name, value)
    except (AttributeError, TypeError):
        for name, old_value in reversed(previous):
            try:
                if old_value is _MISSING:
                    delattr(cls, name)
                else:
                    setattr(cls, name, old_value)
            except (AttributeError, TypeError):
                pass
        raise
    return cls


__all__ = ["attach", "coef_table", "depvar", "fixef_string", "stat", "supported_stats"]
