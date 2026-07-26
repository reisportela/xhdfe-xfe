#!/usr/bin/env python3
"""Adversarial retained-sample provenance checks for xhdfe Gelbach.

Set XHDFE_PY_MODULE to the exact py_hdfe_v11 shared library to validate.
Without it, the script validates the package-local extension that
``import xhdfe`` actually ships.  The hash oracle below is independent of the
C++ implementation and documents the canonical byte stream.
"""

from __future__ import annotations

import importlib.util
import hashlib
import os
import sys
import warnings
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CHECKS = 0


def require(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def module_path() -> Path:
    explicit = os.environ.get("XHDFE_PY_MODULE")
    if explicit:
        return Path(explicit).resolve()
    candidates = sorted((ROOT / "xhdfe").glob("py_hdfe_v11*.so"))
    if not candidates:
        raise FileNotFoundError(
            "no py_hdfe_v11 extension found; set XHDFE_PY_MODULE"
        )
    return candidates[0].resolve()


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extension at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["py_hdfe_v11"] = module
    spec.loader.exec_module(module)
    return module


def fnv1a64_retained_sample(n_input: int, keep: np.ndarray) -> str:
    value = 14695981039346656037
    prime = 1099511628211

    def update_byte(byte: int) -> None:
        nonlocal value
        value ^= byte
        value = (value * prime) & ((1 << 64) - 1)

    def update_u64(number: int) -> None:
        for shift in range(0, 64, 8):
            update_byte((number >> shift) & 0xFF)

    for byte in b"xhdfe-gelbach-sample-v1":
        update_byte(byte)
    update_u64(n_input)
    update_u64(len(keep))
    for index in keep:
        update_u64(int(index))
    return f"{value:016x}"


def assert_same_estimates(left: dict, right: dict) -> None:
    arrays = [
        (left["b_base"], right["b_base"], "b_base"),
        (left["b_full"], right["b_full"], "b_full"),
        (left["cov"], right["cov"], "cov"),
        (left["total"]["coef"], right["total"]["coef"], "total coef"),
        (left["total"]["cov"], right["total"]["cov"], "total cov"),
    ]
    for name in left["names"]:
        arrays.append((
            left["delta"][name]["coef"],
            right["delta"][name]["coef"],
            f"delta {name}",
        ))
    for a, b, label in arrays:
        require(
            np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True),
            f"sample_info changed {label}",
        )


def main() -> None:
    path = module_path()
    module = load_core(path)
    loaded_path = Path(module.__file__).resolve()
    require(loaded_path == path,
            f"loaded extension {loaded_path} differs from selected {path}")
    module_sha256 = hashlib.sha256(loaded_path.read_bytes()).hexdigest()
    from xhdfe import gelbach

    rng = np.random.default_rng(20260725)
    n = 49
    fe = np.r_[np.repeat(np.arange(12), 4), 12]
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 0.8 * x + 0.35 * z + rng.normal(size=n)
    kwargs = {
        "x2_groups": {"observed": z},
        "fes": {"group": fe},
        "x1_names": ["target"],
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        default = gelbach.decompose(y, x, **kwargs)
        audited = gelbach.decompose(y, x, sample_info=True, **kwargs)
        audited_t1 = gelbach.decompose(
            y, x, sample_info=True, num_threads=1, **kwargs
        )

    require(default["sample_info_requested"] is False,
            "default result does not record opt-out")
    for field in (
        "sample_index", "sample_mask", "sample_hash",
        "sample_hash_algorithm", "sample_index_scope",
    ):
        require(default[field] is None, f"default unexpectedly materialized {field}")
    assert_same_estimates(default, audited)

    expected = np.arange(n - 1, dtype=np.int64)
    require(np.array_equal(audited["sample_index"], expected),
            "recursive singleton removal is not reflected in sample_index")
    require(
        np.array_equal(np.flatnonzero(audited["sample_mask"]), expected),
        "sample_mask and sample_index disagree",
    )
    require(audited["sample_mask"].dtype == np.bool_,
            "sample_mask is not Boolean")
    require(audited["n_obs"] == int(audited["sample_mask"].sum()),
            "sample count and mask disagree")
    require(audited["n_singletons_dropped"] == 1,
            "fixture did not exercise singleton dropping")
    require(audited["sample_hash_algorithm"] == "fnv1a64-le-v1",
            "hash algorithm metadata changed")
    require(audited["sample_index_scope"] == "input_rows_zero_based",
            "sample scope metadata changed")
    require(
        audited["sample_hash"] == fnv1a64_retained_sample(n, expected),
        "C++ hash disagrees with the independent canonical oracle",
    )
    require(audited_t1["sample_hash"] == audited["sample_hash"],
            "sample hash depends on thread request")
    require(np.array_equal(audited_t1["sample_index"], expected),
            "sample index depends on thread request")
    assert_same_estimates(audited, audited_t1)

    # The identifier intentionally binds row positions as well as membership.
    permutation = rng.permutation(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        permuted = gelbach.decompose(
            y[permutation], x[permutation],
            x2_groups={"observed": z[permutation]},
            fes={"group": fe[permutation]},
            x1_names=["target"], sample_info=True,
        )
    singleton_position = int(np.flatnonzero(permutation == n - 1)[0])
    expected_permuted = np.delete(np.arange(n, dtype=np.int64),
                                  singleton_position)
    require(np.array_equal(permuted["sample_index"], expected_permuted),
            "permuted singleton was not mapped to input positions")
    require(
        permuted["sample_hash"] ==
        fnv1a64_retained_sample(n, expected_permuted),
        "permuted hash disagrees with oracle",
    )
    require(permuted["sample_hash"] != audited["sample_hash"],
            "position-bound hash failed to distinguish row order")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        no_fe = gelbach.decompose(
            y[:-1], x[:-1], x2_groups={"observed": z[:-1]},
            sample_info=True,
        )
    all_rows = np.arange(n - 1, dtype=np.int64)
    require(np.array_equal(no_fe["sample_index"], all_rows),
            "all-retained case did not return every input position")
    require(np.all(no_fe["sample_mask"]),
            "all-retained mask contains false entries")

    try:
        gelbach.decompose(y, x, sample_info="yes", **kwargs)
    except TypeError:
        pass
    else:
        raise AssertionError("non-Boolean sample_info did not fail closed")

    print(
        "PASS: Gelbach sample provenance "
        f"({CHECKS} checks; module={loaded_path}; "
        f"module_sha256={module_sha256}; hash={audited['sample_hash']})"
    )


if __name__ == "__main__":
    main()
