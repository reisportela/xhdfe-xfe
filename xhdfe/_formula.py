"""Optional Formulaic frontend for the native xhdfe estimator.

The array API remains the canonical low-overhead interface.  This module is
loaded lazily and imports Formulaic and pandas only when a formula is parsed.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from numbers import Integral, Real
import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import _load_core
from . import _maketables


_FORMULA_METADATA = (
    "formula_",
    "coef_names_",
    "intercept_index_",
    "fe_names_",
    "fe_levels_",
    "cluster_levels_",
    "cluster_names_",
    "se_type_",
    "var_labels_",
    "model_spec_",
    "data_index_",
    "estimation_index_",
    "used_fast_path_",
)


def _formula_dependencies():
    try:
        from formulaic import Formula
        import pandas as pd
    except ModuleNotFoundError as exc:
        if exc.name in {"formulaic", "pandas"}:
            raise ImportError(
                "The xhdfe formula interface requires the optional formula "
                "dependencies. From a source checkout, run `python -m pip "
                "install '.[formula]'`; for an installed release wheel, run "
                "`python -m pip install 'formulaic>=1.2.1,<2' "
                "'pandas>=1.3'`."
            ) from exc
        raise ImportError(
            "The xhdfe formula dependency stack could not be imported because "
            f"{exc.name or 'a transitive module'} is missing. Reinstall the "
            "optional Formulaic dependency stack and inspect the chained "
            "import error."
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "The xhdfe formula dependency stack is installed but could not be "
            "imported. Reinstall the optional Formulaic dependency stack and "
            f"inspect the chained error: {exc}"
        ) from exc
    return Formula, pd


def _top_level_pipe_count(formula: str) -> int:
    """Count formula separators without splitting quoted or nested content."""
    depth = 0
    quote: Optional[str] = None
    escaped = False
    count = 0
    for char in formula:
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
        elif char == "|" and depth == 0:
            count += 1
    return count


def _parse_formula(formula: str):
    if not isinstance(formula, str) or not formula.strip():
        raise TypeError("formula must be a non-empty string")
    return _parse_formula_string(formula)


@lru_cache(maxsize=128)
def _parse_formula_string(formula: str):
    if _top_level_pipe_count(formula) > 1:
        raise NotImplementedError(
            "The first formula release supports at most one `|` separator. "
            "Use the array API for IV specifications."
        )
    Formula, _ = _formula_dependencies()
    try:
        parsed = Formula(formula)
    except Exception as exc:
        if "`.` operator" in str(exc):
            raise NotImplementedError(
                "The `.` shorthand is not supported in the first formula "
                "release; list the regression columns explicitly"
            ) from exc
        raise ValueError(f"Invalid xhdfe formula {formula!r}: {exc}") from exc
    if not hasattr(parsed, "lhs") or not hasattr(parsed, "rhs"):
        raise ValueError("formula must have the form `response ~ regressors [| fixed effects]`")
    return parsed


def _is_intercept_term(term: Any) -> bool:
    factors = tuple(term.factors)
    return (
        len(factors) == 1
        and getattr(factors[0].eval_method, "name", "") == "LITERAL"
        and str(factors[0].expr) == "1"
    )


def _lookup_factor(term: Any):
    factors = tuple(term.factors)
    if (
        len(factors) == 1
        and getattr(factors[0].eval_method, "name", "") == "LOOKUP"
    ):
        return factors[0]
    return None


def _formula_parts(parsed: Any):
    rhs = parsed.rhs
    if isinstance(rhs, tuple):
        if len(rhs) != 2:
            raise NotImplementedError(
                "Only `y ~ x | fe1 + fe2` is supported in the first formula release"
            )
        regression_rhs, fixed_effect_rhs = rhs
    else:
        regression_rhs, fixed_effect_rhs = rhs, None
    return parsed.lhs, regression_rhs, fixed_effect_rhs


def _fixed_effect_names(fixed_effect_rhs: Any) -> tuple[str, ...]:
    if fixed_effect_rhs is None:
        return ()
    names: list[str] = []
    for term in fixed_effect_rhs:
        if _is_intercept_term(term):
            continue
        factor = _lookup_factor(term)
        if factor is None:
            raise NotImplementedError(
                "Fixed effects after `|` must be bare column names in the first "
                "formula release; interactions and transforms belong on the RHS"
            )
        names.append(str(factor.expr))
    return tuple(names)


def _data_length(data: Any) -> int:
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    if isinstance(data, Mapping):
        if not data:
            return 0
        return len(next(iter(data.values())))
    raise TypeError("data must be a dataframe or a mapping of column names to arrays")


def _data_index(data: Any, n_rows: int) -> np.ndarray:
    index = getattr(data, "index", None)
    if index is None:
        return np.arange(n_rows, dtype=np.int64)
    try:
        return np.asarray(index.to_numpy(copy=True))
    except (AttributeError, TypeError):
        return np.asarray(index).copy()


def _column(data: Any, name: str):
    try:
        return data[name]
    except Exception as exc:
        raise ValueError(f"Column {name!r} was not found in data") from exc


def _as_array(value: Any) -> np.ndarray:
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        try:
            return np.asarray(to_numpy(copy=False))
        except TypeError:
            return np.asarray(to_numpy())
    return np.asarray(value)


def _as_vector(value: Any, n_rows: int, label: str) -> np.ndarray:
    array = _as_array(value)
    if array.ndim != 1 or array.shape[0] != n_rows:
        raise ValueError(f"{label} must be a one-dimensional vector of length {n_rows}")
    return array


def _missing_positions(array: np.ndarray, pd: Any) -> np.ndarray:
    missing = np.asarray(pd.isna(array))
    if missing.ndim > 1:
        missing = missing.any(axis=tuple(range(1, missing.ndim)))
    return np.flatnonzero(missing)


def _require_no_missing(array: np.ndarray, label: str, pd: Any) -> None:
    positions = _missing_positions(array, pd)
    if positions.size:
        preview = ", ".join(str(int(pos)) for pos in positions[:5])
        suffix = "..." if positions.size > 5 else ""
        raise ValueError(f"{label} contains missing values at row(s) {preview}{suffix}")


def _is_pandas_categorical(value: Any, pd: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    return isinstance(dtype, pd.CategoricalDtype)


def _is_plain_numeric(value: Any, pd: Any) -> bool:
    if _is_pandas_categorical(value, pd):
        return False
    array = _as_array(value)
    return array.dtype.kind in "iuf"


def _numeric_vector(value: Any, n_rows: int, label: str, pd: Any) -> np.ndarray:
    array = _as_vector(value, n_rows, label)
    _require_no_missing(array, label, pd)
    try:
        return np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc


def _encode_ids(value: Any, n_rows: int, label: str, pd: Any):
    array = _as_vector(value, n_rows, label)
    _require_no_missing(array, label, pd)
    if array.dtype.kind == "i":
        negative = np.flatnonzero(array < 0)
        if negative.size:
            raise ValueError(
                f"{label} contains a negative integer ID at row "
                f"{int(negative[0])}; -1 is reserved as a missing-category sentinel"
            )
        return np.ascontiguousarray(array, dtype=np.int64), None
    if array.dtype.kind == "u":
        if array.size == 0 or int(np.max(array)) <= np.iinfo(np.int64).max:
            return np.ascontiguousarray(array, dtype=np.int64), None
    elif array.dtype.kind == "f":
        nonfinite = np.flatnonzero(~np.isfinite(array))
        if nonfinite.size:
            raise ValueError(
                f"{label} contains a non-finite ID at row {int(nonfinite[0])}"
            )
        negative = np.flatnonzero(array < 0)
        if negative.size:
            raise ValueError(
                f"{label} contains a negative numeric ID at row {int(negative[0])}"
            )
    elif array.dtype.kind == "c":
        raise ValueError(f"{label} must contain real-valued category IDs")
    elif array.dtype.kind == "O":
        for position, raw in enumerate(array):
            if isinstance(raw, Integral) and not isinstance(raw, (bool, np.bool_)):
                if int(raw) < 0:
                    raise ValueError(
                        f"{label} contains a negative integer ID at row {position}; "
                        "-1 is reserved as a missing-category sentinel"
                    )
            elif isinstance(raw, Real):
                numeric = float(raw)
                if not np.isfinite(numeric):
                    raise ValueError(
                        f"{label} contains a non-finite ID at row {position}"
                    )
                if numeric < 0:
                    raise ValueError(
                        f"{label} contains a negative numeric ID at row {position}"
                    )
            elif isinstance(raw, complex):
                raise ValueError(f"{label} must contain real-valued category IDs")
    try:
        codes, levels = pd.factorize(array, sort=False, use_na_sentinel=True)
    except TypeError:  # pandas < 1.5, still supported by Formulaic 1.2
        codes, levels = pd.factorize(array, sort=False, na_sentinel=-1)
    if np.any(codes < 0):
        raise ValueError(f"{label} contains missing values")
    return np.ascontiguousarray(codes, dtype=np.int64), np.asarray(levels)


def _frequency_weight_vector(value: Any, n_rows: int, pd: Any) -> np.ndarray:
    array = _as_vector(value, n_rows, "weights")
    _require_no_missing(array, "weights", pd)
    converted = np.empty(n_rows, dtype=np.float64)
    total = 0
    int64_max = int(np.iinfo(np.int64).max)
    for position, raw in enumerate(array):
        scalar = raw.item() if isinstance(raw, np.generic) else raw
        if isinstance(scalar, bool) or not isinstance(scalar, Real):
            raise ValueError(
                f"frequency weights must be numeric positive integers (row {position})"
            )
        numeric = float(scalar)
        if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
            raise ValueError(
                f"frequency weights must be finite positive integers (row {position})"
            )
        exact = int(scalar) if isinstance(scalar, Integral) else int(numeric)
        if exact > int64_max:
            raise ValueError(
                f"frequency weight at row {position} is not representable as int64"
            )
        if int(numeric) != exact:
            raise ValueError(
                f"frequency weight at row {position} cannot be represented exactly "
                "by the current float64 Python binding"
            )
        total += exact
        if total > int64_max:
            raise ValueError("the cumulative frequency-weight total overflows int64")
        converted[position] = numeric
    return converted


def _resolve_weights(
    weights: Any,
    data: Any,
    n_rows: int,
    pd: Any,
    *,
    fweights: bool,
):
    if weights is None:
        return None
    value = _column(data, weights) if isinstance(weights, str) else weights
    if fweights:
        return _frequency_weight_vector(value, n_rows, pd)
    return _numeric_vector(value, n_rows, "weights", pd)


def _variable_labels(data: Any) -> Optional[Mapping[str, str]]:
    """Snapshot descriptive variable labels carried on the estimation frame.

    pandas propagates ``DataFrame.attrs``, and ``maketables.import_dta`` writes
    Stata variable labels under the ``variable_labels`` key there. Reading that
    key is a plain dict lookup, so the labels reach a rendered table without
    xhdfe depending on, or importing, any table package.
    """
    attrs = getattr(data, "attrs", None)
    if not isinstance(attrs, Mapping):
        return None
    labels = attrs.get("variable_labels")
    if not isinstance(labels, Mapping):
        return None
    snapshot = {
        str(key): str(value)
        for key, value in labels.items()
        if isinstance(key, str) and isinstance(value, str)
    }
    return MappingProxyType(snapshot) if snapshot else None


def _normalized_se_type(regressor_options: Mapping[str, Any]) -> str:
    """Name the standard-error family a fit requested.

    The native constructor takes ``se_type`` as a string and defaults it to
    ``unadjusted``, which is its spelling of homoskedastic errors; both spellings
    collapse to ``homoskedastic`` here so downstream consumers see one name. A
    ``StandardErrorType`` member is accepted defensively and read by name.
    """
    raw = regressor_options.get("se_type", "unadjusted")
    name = getattr(raw, "name", None)
    text = str(name if name is not None else raw).lower()
    return "homoskedastic" if text == "unadjusted" else text


def _resolve_clusters(clusters: Any, data: Any, n_rows: int, pd: Any):
    if clusters is None:
        return None, {}, ()

    labels: list[str] = []
    values: list[Any] = []
    if isinstance(clusters, str):
        labels = [clusters]
        values = [_column(data, clusters)]
    elif isinstance(clusters, Sequence) and not isinstance(clusters, np.ndarray):
        sequence = list(clusters)
        if not sequence:
            raise ValueError("clusters cannot be an empty sequence")
        if sequence and all(isinstance(item, str) for item in sequence):
            candidate_labels = [str(item) for item in sequence]
            try:
                candidate_values = [_column(data, item) for item in candidate_labels]
            except ValueError:
                if len(sequence) != n_rows:
                    raise ValueError(
                        "a string cluster sequence must either name data columns "
                        "or contain one label per observation"
                    ) from None
                labels = ["clusters"]
                values = [np.asarray(sequence, dtype=object)]
            else:
                labels = candidate_labels
                values = candidate_values
        elif all(
            _as_array(item).ndim == 1
            and _as_array(item).shape[0] == n_rows
            for item in sequence
        ):
            labels = [f"clusters[{j}]" for j in range(len(sequence))]
            values = sequence
        else:
            array = np.asarray(sequence)
            if array.ndim == 1 and array.shape[0] == n_rows:
                labels = ["clusters"]
                values = [array]
            elif array.ndim == 2 and array.shape[0] == n_rows:
                labels = [f"clusters[{j}]" for j in range(array.shape[1])]
                values = [array[:, j] for j in range(array.shape[1])]
            else:
                raise ValueError(
                    "clusters must be a length-n vector, an (n, q) array, or "
                    "a sequence of length-n vectors"
                )
    else:
        array = _as_array(clusters)
        if array.ndim == 1 and array.shape[0] == n_rows:
            labels = ["clusters"]
            values = [array]
        elif array.ndim == 2 and array.shape[0] == n_rows:
            labels = [f"clusters[{j}]" for j in range(array.shape[1])]
            values = [array[:, j] for j in range(array.shape[1])]
        else:
            raise ValueError("clusters must have shape (n,) or (n, q)")

    encoded: list[np.ndarray] = []
    levels: dict[str, np.ndarray] = {}
    for label, value in zip(labels, values):
        codes, category_levels = _encode_ids(value, n_rows, label, pd)
        encoded.append(codes)
        if category_levels is not None:
            levels[label] = category_levels
    # ``levels`` only records categorical clusters, so the labels are returned
    # separately: they are the sole record of which variables were clustered on.
    return encoded, levels, tuple(labels)


def _simple_numeric_matrix(lhs: Any, rhs: Any, data: Any, n_rows: int, pd: Any):
    lhs_terms = list(lhs)
    if len(lhs_terms) != 1:
        return None
    lhs_factor = _lookup_factor(lhs_terms[0])
    if lhs_factor is None:
        return None
    y_source = _column(data, str(lhs_factor.expr))
    if not _is_plain_numeric(y_source, pd):
        return None

    has_intercept = False
    term_sources: list[tuple[Any, list[tuple[str, Any]]]] = []
    for term in rhs:
        if _is_intercept_term(term):
            has_intercept = True
            continue
        factors = list(term.factors)
        if not factors or any(
            getattr(factor.eval_method, "name", "") != "LOOKUP" for factor in factors
        ):
            return None
        sources = [
            (str(factor.expr), _column(data, str(factor.expr)))
            for factor in factors
        ]
        if not all(_is_plain_numeric(source, pd) for _, source in sources):
            return None
        term_sources.append((term, sources))

    y = _numeric_vector(y_source, n_rows, "response", pd)
    columns: list[np.ndarray] = []
    names: list[str] = []
    converted_sources: dict[str, np.ndarray] = {}
    for term, sources in term_sources:
        column = np.ones(n_rows, dtype=np.float64)
        for source_name, source in sources:
            values = converted_sources.get(source_name)
            if values is None:
                values = _numeric_vector(source, n_rows, f"term {term}", pd)
                converted_sources[source_name] = values
            column *= values
        columns.append(column)
        names.append(str(term))
    X = (
        np.asfortranarray(np.column_stack(columns), dtype=np.float64)
        if columns
        else np.empty((n_rows, 0), dtype=np.float64, order="F")
    )
    return y, X, tuple(names), has_intercept


_BACKTICK_NAME_RE = re.compile(r"`([^`]+)`")


def _python_factor_columns(
    expression: str,
    available_columns: set[str],
) -> tuple[set[str], set[str], bool]:
    token_to_column: dict[str, str] = {}

    def replace_backtick(match: re.Match[str]) -> str:
        token = f"__xhdfe_column_{len(token_to_column)}"
        token_to_column[token] = match.group(1)
        return token

    parseable = _BACKTICK_NAME_RE.sub(replace_backtick, expression)
    try:
        tree = ast.parse(parseable, mode="eval")
    except SyntaxError:
        referenced = {
            name for name in token_to_column.values() if name in available_columns
        }
        categorical = referenced if expression.lstrip().startswith("C(") else set()
        return referenced, categorical, bool(categorical)

    def column_name(node: ast.Name) -> Optional[str]:
        name = token_to_column.get(node.id, node.id)
        return name if name in available_columns else None

    def quoted_column(node: ast.AST) -> Optional[str]:
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "Q"
            or len(node.args) != 1
            or node.keywords
            or not isinstance(node.args[0], ast.Constant)
            or not isinstance(node.args[0].value, str)
        ):
            return None
        name = node.args[0].value
        return name if name in available_columns else None

    def referenced_columns(node: ast.AST) -> set[str]:
        call_functions = {
            id(child.func)
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
        }
        columns: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and id(child) not in call_functions:
                name = column_name(child)
                if name is not None:
                    columns.add(name)
            quoted = quoted_column(child)
            if quoted is not None:
                columns.add(quoted)
        return columns

    referenced = referenced_columns(tree)
    categorical: set[str] = set()
    has_derived_categorical = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "C":
            continue
        data_argument: Optional[ast.AST] = node.args[0] if node.args else None
        if data_argument is None:
            for keyword in node.keywords:
                if keyword.arg == "data":
                    data_argument = keyword.value
                    break
        if not (
            isinstance(data_argument, ast.Name)
            and column_name(data_argument) is not None
        ) and quoted_column(data_argument) is None:
            has_derived_categorical = True
        for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
            categorical.update(referenced_columns(argument))
    return referenced, categorical, has_derived_categorical


def _formulaic_input_data(lhs: Any, rhs: Any, data: Any, n_rows: int, pd: Any):
    if isinstance(data, Mapping):
        columns: dict[Any, np.ndarray] = {}
        for name, value in data.items():
            columns[name] = _as_vector(value, n_rows, f"column {name!r}")
        try:
            formula_data = pd.DataFrame(columns, copy=False)
        except Exception as exc:
            raise ValueError(f"Could not construct formula data from the mapping: {exc}") from exc
    else:
        formula_data = data

    if not isinstance(formula_data, pd.DataFrame):
        return formula_data

    available_columns = {str(name) for name in formula_data.columns}
    numeric_candidates: set[str] = set()
    categorical_candidates: set[str] = set()
    for formula_side in (lhs, rhs):
        for term in formula_side:
            for factor in term.factors:
                method = getattr(factor.eval_method, "name", "")
                if method == "LOOKUP":
                    numeric_candidates.add(str(factor.expr))
                elif method == "PYTHON":
                    referenced, categorical, derived_categorical = (
                        _python_factor_columns(
                            str(factor.expr),
                            available_columns,
                        )
                    )
                    if derived_categorical:
                        raise NotImplementedError(
                            "The first argument to C(...) must be a bare column "
                            "name in the first formula release; create the "
                            "derived category explicitly in data"
                        )
                    numeric_candidates.update(referenced - categorical)
                    categorical_candidates.update(categorical)

    numeric_lookups: dict[str, np.ndarray] = {}
    for name in numeric_candidates:
        if name not in available_columns:
            continue
        source = _column(formula_data, name)
        if _is_plain_numeric(source, pd):
            numeric_lookups[name] = _numeric_vector(
                source,
                n_rows,
                f"term {name!r}",
                pd,
            )

    for name in categorical_candidates.intersection(numeric_lookups):
        source = _as_array(_column(formula_data, name))
        if source.dtype != np.dtype(np.float64):
            raise ValueError(
                f"Column {name!r} is used both categorically through C(...) "
                "and numerically in the same formula. Create a separate "
                "float64 column for its numeric role to preserve category IDs."
            )
    if not numeric_lookups:
        return formula_data

    promoted = formula_data.copy(deep=False)
    for name, values in numeric_lookups.items():
        promoted[name] = values
    return promoted


def _formulaic_matrix(lhs: Any, rhs: Any, data: Any, context: Mapping[str, Any]):
    Formula, pd = _formula_dependencies()
    regression_formula = Formula({"lhs": lhs, "rhs": rhs})
    n_rows = _data_length(data)
    formula_data = _formulaic_input_data(lhs, rhs, data, n_rows, pd)
    try:
        matrices = regression_formula.get_model_matrix(
            formula_data,
            context=dict(context),
            output="numpy",
            na_action="raise",
            ensure_full_rank=True,
        )
    except Exception as exc:
        raise ValueError(f"Could not materialize the xhdfe formula: {exc}") from exc

    lhs_matrix = np.asarray(matrices.lhs)
    if lhs_matrix.ndim != 2 or lhs_matrix.shape[1] != 1:
        raise ValueError("The formula response must materialize to one numeric column")
    if np.iscomplexobj(lhs_matrix):
        raise ValueError("The formula response must be real-valued")
    try:
        y = np.ascontiguousarray(lhs_matrix[:, 0], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("The formula response must be numeric") from exc

    rhs_matrix = np.asarray(matrices.rhs)
    if np.iscomplexobj(rhs_matrix):
        raise ValueError("The formula RHS must be real-valued")
    column_names = list(matrices.rhs.model_spec.column_names)
    intercept_indices: list[int] = []
    for term, indices in matrices.rhs.model_spec.term_indices.items():
        if _is_intercept_term(term):
            intercept_indices.extend(int(index) for index in indices)
    intercept_set = set(intercept_indices)
    keep = [index for index in range(rhs_matrix.shape[1]) if index not in intercept_set]
    try:
        X = np.asfortranarray(rhs_matrix[:, keep], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The formula RHS did not materialize to numeric columns. Wrap "
            "categorical variables explicitly in C(...)."
        ) from exc
    names = tuple(column_names[index] for index in keep)
    return y, X, names, bool(intercept_indices), matrices.model_spec


@dataclass(frozen=True)
class _MaterializedFormula:
    formula: str
    y: np.ndarray
    X: np.ndarray
    fes: tuple[np.ndarray, ...]
    weights: Optional[np.ndarray]
    clusters: Optional[tuple[np.ndarray, ...]]
    fweights: bool
    fit_intercept: bool
    coef_names: tuple[str, ...]
    intercept_index: Optional[int]
    fe_names: tuple[str, ...]
    fe_levels: Mapping[str, np.ndarray]
    cluster_levels: Mapping[str, np.ndarray]
    cluster_names: tuple[str, ...]
    var_labels: Optional[Mapping[str, str]]
    model_spec: Any
    data_index: np.ndarray
    used_fast_path: bool


def _materialize_formula(
    formula: str,
    data: Any,
    *,
    weights: Any,
    clusters: Any,
    fweights: bool,
    na_action: str,
    context: Optional[Mapping[str, Any]],
    force_formulaic: bool = False,
) -> _MaterializedFormula:
    if na_action != "raise":
        raise NotImplementedError(
            "The first formula release supports only na_action='raise'; drop or "
            "impute missing values explicitly before estimation"
        )
    if context is not None and not isinstance(context, Mapping):
        raise TypeError("context must be a mapping of explicitly allowed transforms")
    if not isinstance(fweights, (bool, np.bool_)):
        raise TypeError("fweights must be a boolean")
    fweights = bool(fweights)

    parsed = _parse_formula(formula)
    lhs, rhs, fixed_effect_rhs = _formula_parts(parsed)
    fe_names = _fixed_effect_names(fixed_effect_rhs)
    _, pd = _formula_dependencies()
    n_rows = _data_length(data)
    if n_rows == 0:
        raise ValueError("data contains no observations")

    simple = None if force_formulaic else _simple_numeric_matrix(lhs, rhs, data, n_rows, pd)
    if simple is None:
        y, X, slope_names, fit_intercept, model_spec = _formulaic_matrix(
            lhs, rhs, data, context or {}
        )
        used_fast_path = False
    else:
        y, X, slope_names, fit_intercept = simple
        model_spec = None
        used_fast_path = True

    fes: list[np.ndarray] = []
    fe_levels: dict[str, np.ndarray] = {}
    for name in fe_names:
        codes, levels = _encode_ids(_column(data, name), n_rows, f"fixed effect {name!r}", pd)
        fes.append(codes)
        if levels is not None:
            fe_levels[name] = levels

    resolved_weights = _resolve_weights(
        weights,
        data,
        n_rows,
        pd,
        fweights=fweights,
    )
    if fweights and resolved_weights is None:
        raise ValueError("fweights=True requires weights")
    resolved_clusters, cluster_levels, cluster_names = _resolve_clusters(
        clusters, data, n_rows, pd
    )
    intercept_index: Optional[int] = None
    coef_names = slope_names
    if fit_intercept:
        intercept_index = len(slope_names)
        intercept_name = "Intercept"
        suffix = 1
        while intercept_name in slope_names:
            qualifier = "xhdfe" if suffix == 1 else f"xhdfe {suffix}"
            intercept_name = f"Intercept [{qualifier}]"
            suffix += 1
        coef_names = slope_names + (intercept_name,)
    return _MaterializedFormula(
        formula=formula,
        y=y,
        X=X,
        fes=tuple(fes),
        weights=resolved_weights,
        clusters=tuple(resolved_clusters) if resolved_clusters is not None else None,
        fweights=fweights,
        fit_intercept=fit_intercept,
        coef_names=coef_names,
        intercept_index=intercept_index,
        fe_names=fe_names,
        fe_levels=MappingProxyType(fe_levels),
        cluster_levels=MappingProxyType(cluster_levels),
        cluster_names=cluster_names,
        var_labels=_variable_labels(data),
        model_spec=model_spec,
        data_index=_data_index(data, n_rows),
        used_fast_path=used_fast_path,
    )


def _readonly_copy(array: np.ndarray, *, order: str = "C") -> np.ndarray:
    copied = np.array(array, copy=True, order=order)
    copied.setflags(write=False)
    return copied


def _freeze_materialized(materialized: _MaterializedFormula) -> _MaterializedFormula:
    levels = {
        name: _readonly_copy(np.asarray(value))
        for name, value in materialized.fe_levels.items()
    }
    cluster_levels = {
        name: _readonly_copy(np.asarray(value))
        for name, value in materialized.cluster_levels.items()
    }
    return replace(
        materialized,
        y=_readonly_copy(materialized.y),
        X=_readonly_copy(materialized.X, order="F"),
        fes=tuple(_readonly_copy(fe) for fe in materialized.fes),
        weights=(
            _readonly_copy(materialized.weights)
            if materialized.weights is not None
            else None
        ),
        clusters=(
            tuple(_readonly_copy(cluster) for cluster in materialized.clusters)
            if materialized.clusters is not None
            else None
        ),
        fe_levels=MappingProxyType(levels),
        cluster_levels=MappingProxyType(cluster_levels),
        data_index=_readonly_copy(materialized.data_index),
    )


@lru_cache(maxsize=1)
def _formula_regressor_class():
    native_class = _load_core().HdfeRegressor

    class FormulaRegressor(native_class):
        """Native regressor with formula metadata attached after fitting."""

        def fit(self, *args, **kwargs):
            for name in _FORMULA_METADATA:
                self.__dict__.pop(name, None)
            return super().fit(*args, **kwargs)

        def tidy(self):
            """Return named coefficient results as a pandas DataFrame."""
            _, pd = _formula_dependencies()
            names = getattr(
                self,
                "coef_names_",
                tuple(f"b{index}" for index in range(len(self.coef_))),
            )
            if len(names) != len(self.coef_):
                raise RuntimeError("coefficient names are not aligned with native results")
            return pd.DataFrame(
                {
                    "Estimate": np.asarray(self.coef_),
                    "Std. Error": np.asarray(self.stderr_),
                    "t value": np.asarray(self.tvalues_),
                    "Pr(>|t|)": np.asarray(self.pvalues_),
                    "CI Low": np.asarray(self.conf_int_)[:, 0],
                    "CI High": np.asarray(self.conf_int_)[:, 1],
                },
                index=pd.Index(names, name="Coefficient"),
            )

    FormulaRegressor.__name__ = "FormulaRegressor"
    FormulaRegressor.__qualname__ = "FormulaRegressor"
    FormulaRegressor.__module__ = __name__
    _maketables.attach(FormulaRegressor)
    return FormulaRegressor


def _fit_materialized(
    materialized: _MaterializedFormula,
    regressor_options: Mapping[str, Any],
):
    options = dict(regressor_options)
    if "fit_intercept" in options:
        raise TypeError(
            "fit_intercept is controlled by the formula; use `0` or `-1` to "
            "remove the intercept"
        )
    options["fit_intercept"] = materialized.fit_intercept
    regressor_class = _formula_regressor_class()
    model = regressor_class(**options)
    fit_options = {
        "fes": list(materialized.fes) if materialized.fes else None,
        "weights": materialized.weights,
        "clusters": (
            list(materialized.clusters)
            if materialized.clusters is not None
            else None
        ),
    }
    if materialized.fweights:
        fit_options["fweights"] = True
    model.fit(materialized.y, materialized.X, **fit_options)
    if len(materialized.coef_names) != len(model.coef_):
        raise RuntimeError(
            "Formula coefficient names do not align with the native coefficient vector"
        )
    model.formula_ = materialized.formula
    model.coef_names_ = materialized.coef_names
    model.intercept_index_ = materialized.intercept_index
    model.fe_names_ = materialized.fe_names
    model.fe_levels_ = materialized.fe_levels
    model.cluster_levels_ = materialized.cluster_levels
    model.cluster_names_ = materialized.cluster_names
    model.se_type_ = _normalized_se_type(regressor_options)
    model.var_labels_ = materialized.var_labels
    model.model_spec_ = materialized.model_spec
    model.data_index_ = materialized.data_index.copy()
    sample = np.asarray(model.sample_index_, dtype=np.int64)
    model.estimation_index_ = materialized.data_index[sample].copy()
    model.used_fast_path_ = materialized.used_fast_path
    return model


class PreparedFormula:
    """An explicit, read-only snapshot for repeated formula fits."""

    __slots__ = ("_materialized", "_regressor_options")

    def __init__(
        self,
        materialized: _MaterializedFormula,
        regressor_options: Mapping[str, Any],
    ) -> None:
        self._materialized = _freeze_materialized(materialized)
        self._regressor_options = MappingProxyType(deepcopy(dict(regressor_options)))

    @property
    def formula(self) -> str:
        return self._materialized.formula

    @property
    def y(self) -> np.ndarray:
        return self._materialized.y

    @property
    def X(self) -> np.ndarray:
        return self._materialized.X

    @property
    def fes(self) -> tuple[np.ndarray, ...]:
        return self._materialized.fes

    @property
    def coef_names(self) -> tuple[str, ...]:
        return self._materialized.coef_names

    @property
    def intercept_index(self) -> Optional[int]:
        return self._materialized.intercept_index

    @property
    def used_fast_path(self) -> bool:
        return self._materialized.used_fast_path

    def fit(self):
        """Fit a fresh native regressor against the prepared read-only arrays."""
        return _fit_materialized(self._materialized, self._regressor_options)


def prepare_formula(
    formula: str,
    data: Any,
    *,
    weights: Any = None,
    clusters: Any = None,
    fweights: bool = False,
    na_action: str = "raise",
    context: Optional[Mapping[str, Any]] = None,
    **regressor_options: Any,
) -> PreparedFormula:
    """Materialize a read-only formula design snapshot for repeated fits."""
    materialized = _materialize_formula(
        formula,
        data,
        weights=weights,
        clusters=clusters,
        fweights=fweights,
        na_action=na_action,
        context=context,
    )
    return PreparedFormula(materialized, regressor_options)


def feols(
    formula: str,
    data: Any,
    *,
    weights: Any = None,
    clusters: Any = None,
    fweights: bool = False,
    na_action: str = "raise",
    context: Optional[Mapping[str, Any]] = None,
    **regressor_options: Any,
):
    """Fit xhdfe from an R/Formulaic-style formula.

    The supported fixed-effect syntax is ``y ~ x1 + x2 | firm + year``.
    Formulaic handles RHS categories and interactions; the fixed-effect branch
    accepts bare column names and is passed to the unchanged native absorber.
    """
    materialized = _materialize_formula(
        formula,
        data,
        weights=weights,
        clusters=clusters,
        fweights=fweights,
        na_action=na_action,
        context=context,
    )
    return _fit_materialized(materialized, regressor_options)


__all__ = ["PreparedFormula", "feols", "prepare_formula"]
