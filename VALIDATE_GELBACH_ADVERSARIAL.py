#!/usr/bin/env python3
"""Adversarial HDFE gates for Gelbach inference and FE normalization."""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

FAIL = []
decompose = None
tidy = None


def check(name, got, expected, tol):
    diff = float(np.max(np.abs(np.asarray(got) - np.asarray(expected))))
    ok = diff <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max|diff|={diff:.2e}")
    if not ok:
        FAIL.append(name)


def check_condition(name, condition, detail=""):
    ok = bool(condition)
    suffix = f": {detail}" if detail else ""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{suffix}")
    if not ok:
        FAIL.append(name)


def check_raises(name, fn, text):
    try:
        fn()
    except Exception as exc:
        check_condition(name, text in str(exc),
                        f"{type(exc).__name__}: {exc}")
    else:
        check_condition(name, False, "no error raised")


def project(x, h):
    X = np.column_stack([x, np.ones(x.size)])
    return np.linalg.lstsq(X, h, rcond=None)[0]


def slow_chain():
    rng = np.random.default_rng(8)
    m = 1000
    worker = np.repeat(np.arange(m), 4)
    firm = np.column_stack([np.arange(m), np.arange(m),
                            np.arange(m) + 1, np.arange(m) + 1]).ravel()
    n = worker.size
    x = rng.normal(size=n)
    z = 0.3 * x + rng.normal(size=n)
    y = (0.7 * x + 0.5 * z + rng.normal(size=m)[worker] +
         rng.normal(size=m + 1)[firm] + rng.normal(scale=0.1, size=n))

    current = os.environ.pop("XHDFE_GELBACH_FAST_FIT", None)
    try:
        fast = decompose(y, x, {"z": z}, {"worker": worker, "firm": firm},
                         tol=1e-8)
        os.environ["XHDFE_GELBACH_FAST_FIT"] = "0"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            legacy = decompose(y, x, {"z": z},
                               {"worker": worker, "firm": firm}, tol=1e-8)
    finally:
        if current is None:
            os.environ.pop("XHDFE_GELBACH_FAST_FIT", None)
        else:
            os.environ["XHDFE_GELBACH_FAST_FIT"] = current

    if fast["converged"]:
        print("[PASS] slow-chain:default-converged")
    else:
        print(f"[FAIL] slow-chain:default-converged: {fast['notes']}")
        FAIL.append("slow-chain:default-converged")
    if (not legacy["converged"] and
            "exact-normal-equations cross-check" in legacy["notes"]):
        print("[PASS] slow-chain:legacy-fails-closed")
    else:
        print(f"[FAIL] slow-chain:legacy-fails-closed: {legacy['notes']}")
        FAIL.append("slow-chain:legacy-fails-closed")

    # Independent explicit-dummy sparse oracle. With a constant, omit one
    # level from each FE dimension in this connected graph.
    rows = np.arange(n)
    Dw = sparse.csr_matrix((np.ones(n), (rows, worker)), shape=(n, m))[:, 1:]
    Df = sparse.csr_matrix((np.ones(n), (rows, firm)), shape=(n, m + 1))[:, 1:]
    X = sparse.hstack([x[:, None], z[:, None], np.ones((n, 1)), Dw, Df],
                      format="csr")
    gram = (X.T @ X).tocsc()
    b = spsolve(gram, np.asarray(X.T @ y).ravel())
    normal_resid = float(np.max(np.abs(gram @ b - np.asarray(X.T @ y).ravel())))
    if np.all(np.isfinite(b)) and normal_resid < 1e-8:
        print(f"[PASS] slow-chain:sparse-oracle-normal-equations: {normal_resid:.2e}")
    else:
        print(f"[FAIL] slow-chain:sparse-oracle-normal-equations: {normal_resid:.2e}")
        FAIL.append("slow-chain:sparse-oracle-normal-equations")
    hfe = np.asarray(Dw @ b[3:3 + m - 1] + Df @ b[3 + m - 1:]).ravel()
    check("slow-chain:b_full-vs-LSDV", fast["b_full"][0], b[0], 5e-7)
    check("slow-chain:observed-delta-vs-LSDV", fast["delta"]["z"]["coef"],
          project(x, z * b[1]), 2e-7)
    # As in the b1x2 strong-HDFE oracle, compare focal-regressor rows. The
    # intercept row moves with the omitted-dummy/FE normalization convention.
    check("slow-chain:aggregate-FE-x1-vs-LSDV", fast["fe_total"]["coef"][:1],
          project(x, hfe)[:1], 2e-7)


def normalization_convention():
    rng = np.random.default_rng(77)
    n = 400
    component = np.repeat([0, 1], n // 2)
    x = component + 0.2 * rng.normal(size=n)
    alpha = rng.normal(size=n)
    psi = rng.normal(size=n)
    before_a, before_p = project(x, alpha), project(x, psi)
    shift = np.where(component == 0, 2.5, -1.75)
    after_a, after_p = project(x, alpha + shift), project(x, psi - shift)
    check("normalization:multi-component-aggregate-invariant",
          after_a + after_p, before_a + before_p, 1e-12)
    movement = abs(after_a[0] - before_a[0])
    if movement > 0.1:
        print(f"[PASS] normalization:multi-component-split-moves: {movement:.2e}")
    else:
        print(f"[FAIL] normalization:multi-component-split-moves: {movement:.2e}")
        FAIL.append("normalization:multi-component-split-moves")

    # In one connected component, the admissible ambiguity is a global
    # constant; the x1 row is unchanged because the auxiliary model includes
    # an intercept (only the constant row moves).
    global_shift = np.full(n, 3.0)
    one_a = project(x, alpha + global_shift)
    one_p = project(x, psi - global_shift)
    check("normalization:single-component-x1-split-invariant",
          [one_a[0], one_p[0]], [before_a[0], before_p[0]], 1e-12)


def inference_contract_attacks():
    """Attack the additive post-audit warning and covariance contracts."""
    rng = np.random.default_rng(20260723)
    n_groups, periods = 30, 12
    fe = np.repeat(np.arange(n_groups), periods)
    group_signal = rng.normal(size=n_groups)[fe]
    eps = rng.normal(size=fe.size)
    eps -= np.bincount(
        fe, weights=eps, minlength=n_groups)[fe] / periods
    target_ratio = 5e-8
    scale = np.sqrt(
        target_ratio * (group_signal @ group_signal)
        / ((eps @ eps) * (1.0 - target_ratio))
    )
    x = group_signal + scale * eps
    z = rng.normal(size=fe.size)
    y = (
        0.8 * x + 0.5 * z + rng.normal(size=n_groups)[fe]
        + rng.normal(scale=0.5, size=fe.size)
    )
    marker = "near-FE-collinear focal"
    with warnings.catch_warnings(record=True) as caught_default:
        warnings.simplefilter("always", RuntimeWarning)
        default = decompose(y, x, {"z": z}, {"firm": fe})
    check_condition(
        "near-fe:default-warning-contract",
        default["x1_near_collinear_mask"] == [True]
        and default["fe_collinear_ss_ratio_tol"]
        < default["x1_fe_collinear_ratio"][0]
        <= default["near_fe_collinear_ss_ratio_warn_upper"]
        and marker in default["notes"]
        and any(marker in str(item.message) for item in caught_default),
        f"ratio={default['x1_fe_collinear_ratio'][0]:.3e}",
    )

    old_switch = os.environ.get("XHDFE_GELBACH_NEAR_COLLINEAR_WARN")
    os.environ["XHDFE_GELBACH_NEAR_COLLINEAR_WARN"] = "0"
    try:
        with warnings.catch_warnings(record=True) as caught_kill:
            warnings.simplefilter("always", RuntimeWarning)
            killed = decompose(y, x, {"z": z}, {"firm": fe})
    finally:
        if old_switch is None:
            os.environ.pop("XHDFE_GELBACH_NEAR_COLLINEAR_WARN", None)
        else:
            os.environ["XHDFE_GELBACH_NEAR_COLLINEAR_WARN"] = old_switch
    check_condition(
        "near-fe:kill-switch-is-output-only",
        killed["x1_near_collinear_mask"] == [True]
        and marker not in killed["notes"]
        and not any(marker in str(item.message) for item in caught_kill)
        and np.array_equal(killed["b_base"], default["b_base"])
        and np.array_equal(killed["b_full"], default["b_full"])
        and np.array_equal(killed["cov"], default["cov"])
        and np.array_equal(
            killed["cov_delta_bbase"], default["cov_delta_bbase"]),
    )

    # Metadata is independently reconstructible from the same retained rows.
    n = 300
    x1 = rng.normal(size=(n, 2))
    a = np.column_stack([
        0.2 * x1[:, 0] + rng.normal(size=n),
        rng.normal(size=n),
    ])
    b = 0.3 * x1[:, 1] + rng.normal(size=n)
    y2 = (
        x1 @ np.array([0.7, -0.2]) + a @ np.array([0.6, -0.4])
        + 0.5 * b + rng.normal(size=n)
    )
    cluster3 = np.arange(n) % 3
    with warnings.catch_warnings(record=True) as caught_g:
        warnings.simplefilter("always", RuntimeWarning)
        meta = decompose(
            y2, x1, {"A": a, "B": b},
            vce="cluster", cluster=cluster3)
    full_coef = np.linalg.lstsq(
        np.column_stack([x1, a, b, np.ones(n)]), y2, rcond=None)[0]
    check("metadata:gamma-A-independent-full-fit",
          meta["gamma"]["A"], full_coef[2:4], 2e-12)
    check("metadata:gamma-B-independent-full-fit",
          meta["gamma"]["B"], full_coef[4:5], 2e-12)
    check_condition(
        "metadata:G-df-and-few-cluster-note",
        meta["n_clusters"] == np.unique(cluster3).size
        and meta["df_base"] == n - 3
        and meta["few_cluster_warning_threshold"] == 30
        and "few clusters (G < 30)" in meta["notes"]
        and any("few clusters (G < 30)" in str(item.message)
                for item in caught_g),
    )

    # Recompute one delta/base ratio variance from the public covariance
    # blocks.  This detects sign, transpose, and wrong-multiplier errors.
    rows = tidy(
        meta, focal=0, share="base",
        include_total=False, include_full=False)
    denom = float(meta["b_base"][0])
    delta_a = float(meta["delta"]["A"]["coef"][0])
    k1 = len(meta["labels"])
    expected_var = (
        float(meta["cov"][0, 0]) / denom ** 2
        + delta_a ** 2 * float(meta["base_cov"][0, 0]) / denom ** 4
        - 2.0 * delta_a
        * float(meta["cov_delta_bbase"][0, 0]) / denom ** 3
    )
    check("share-base:manual-joint-delta-method",
          rows[0]["share_std_error"],
          np.sqrt(max(0.0, expected_var)), 2e-15)
    check_condition(
        "share-base:joint-label",
        rows[0]["share_se_type"] == "joint_base_covariance_delta_method"
        and np.isfinite(rows[0]["share_std_error"])
        and meta["cov_delta_bbase"].shape == (
            len(meta["names"]) * k1, k1),
    )

    with warnings.catch_warnings(record=True) as caught_share:
        warnings.simplefilter("always", RuntimeWarning)
        undefined = tidy(
            meta, focal=0, share="base", share_tol=1e100,
            include_total=False, include_full=False)
    share_warnings = [
        item for item in caught_share
        if "share denominator is undefined" in str(item.message)
    ]
    check_condition(
        "share-base:undefined-warning-once",
        len(share_warnings) == 1
        and all(not row["share_defined"] for row in undefined),
    )

    # A saturated fit must throw through the language boundary and leave the
    # process healthy for the next call.
    sat_x1 = rng.normal(size=5)
    sat_x2 = rng.normal(size=(5, 3))
    sat_y = rng.normal(size=5)
    check_raises(
        "saturated:catchable",
        lambda: decompose(sat_y, sat_x1, {"sat": sat_x2}),
        "df_full must be positive",
    )
    healthy = decompose(y2, x1, {"A": a})
    check_condition(
        "saturated:process-remains-healthy",
        healthy["converged"] and np.all(np.isfinite(healthy["cov"])),
    )

    # In absorbed-target mode total_j and b_base_j are the same estimator.
    invariant = rng.normal(size=n_groups)[fe]
    z3 = rng.normal(size=fe.size)
    y3 = (
        0.5 * invariant + 0.4 * z3 + rng.normal(size=n_groups)[fe]
        + rng.normal(size=fe.size)
    )
    absorbed = decompose(
        y3, invariant, {"z": z3}, {"firm": fe},
        vce="cluster", cluster=fe, absorbed_targets=[0])
    check("absorbed-total:cross-covariance-identity",
          absorbed["cov_total_bbase"][0, :],
          absorbed["base_cov"][0, :], 0.0)


def main():
    global decompose, tidy
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", default=None,
                    help="directory containing the py_hdfe_v11 extension to validate")
    args = ap.parse_args()
    if args.module_dir:
        sys.path.insert(0, os.path.abspath(args.module_dir))
        __import__("py_hdfe_v11")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from xhdfe.gelbach import decompose as loaded_decompose
    from xhdfe.gelbach import tidy as loaded_tidy
    decompose = loaded_decompose
    tidy = loaded_tidy

    slow_chain()
    normalization_convention()
    inference_contract_attacks()
    if FAIL:
        raise SystemExit(f"{len(FAIL)} adversarial check(s) failed: {FAIL}")
    print("ALL ADVERSARIAL GELBACH CHECKS PASSED")


if __name__ == "__main__":
    main()
