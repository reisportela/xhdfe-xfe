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
from scipy.stats import chi2

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, REPO_ROOT)
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


def input_contract():
    x = np.arange(12, dtype=float)[:, None]
    y = 0.5 * x[:, 0] + np.sin(x[:, 0])
    z = np.cos(x[:, 0])
    check_raises(
        "inputs:negative-num-threads-fails-closed",
        lambda: decompose(y, x, {"z": z}, num_threads=-1),
        "nonnegative integer",
    )
    check_raises(
        "inputs:fractional-num-threads-fails-closed",
        lambda: decompose(y, x, {"z": z}, num_threads=1.5),
        "nonnegative integer",
    )


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


def common_fe_contract():
    """Common FEs condition base/full rather than becoming omitted blocks."""
    rng = np.random.default_rng(20260726)
    n = 720
    p = 2
    q = 2
    common = np.repeat(np.arange(24), n // 24)
    added = rng.integers(0, 18, size=n)
    cluster = np.repeat(np.arange(60), n // 60)
    x1 = rng.normal(size=(n, p))
    x2 = np.column_stack([
        0.35 * x1[:, 0] + rng.normal(size=n),
        -0.20 * x1[:, 1] + rng.normal(size=n),
    ])
    y = (
        x1 @ np.array([1.1, -0.6])
        + x2 @ np.array([0.7, -0.25])
        + rng.normal(size=24)[common]
        + rng.normal(size=18)[added]
        + rng.normal(scale=0.25, size=n)
    )

    rows = np.arange(n)
    Dc = sparse.csr_matrix(
        (np.ones(n), (rows, common)), shape=(n, 24)
    ).toarray()[:, 1:]
    Da = sparse.csr_matrix(
        (np.ones(n), (rows, added)), shape=(n, 18)
    ).toarray()[:, 1:]
    A_base = np.column_stack([np.ones(n), x1, Dc])
    A_full = np.column_stack([np.ones(n), x1, x2, Dc, Da])
    b_base = np.linalg.lstsq(A_base, y, rcond=None)[0]
    b_full = np.linalg.lstsq(A_full, y, rcond=None)[0]
    h_observed = x2 @ b_full[1 + p:1 + p + q]
    added_start = 1 + p + q + Dc.shape[1]
    h_added = Da @ b_full[added_start:]
    d_observed = np.linalg.lstsq(
        A_base, h_observed, rcond=None
    )[0][1:1 + p]
    d_added = np.linalg.lstsq(
        A_base, h_added, rcond=None
    )[0][1:1 + p]

    results = {}
    for vce in ("unadjusted", "robust", "cluster"):
        kwargs = {"vce": vce, "tol": 1e-10}
        if vce == "cluster":
            kwargs["cluster"] = cluster
        # Independent parameterization oracle: the common FE is represented by
        # a full-rank set of explicit base/full dummies. The focal-slope
        # contribution covariance must equal the absorbed-common-FE result.
        x1_explicit = np.column_stack([x1, Dc])
        explicit = decompose(
            y, x1_explicit, {"observed": x2}, {"added": added}, **kwargs
        )
        got = decompose(
            y, x1, {"observed": x2}, {"added": added},
            common_fes={"cohort": common}, x1_names=["x", "w"], **kwargs
        )
        results[vce] = got
        check(f"common-fe:{vce}:b-base-vs-LSDV",
              got["b_base"], b_base[1:1 + p], 2e-10)
        check(f"common-fe:{vce}:b-full-vs-LSDV",
              got["b_full"], b_full[1:1 + p], 2e-10)
        check(f"common-fe:{vce}:observed-delta-vs-LSDV",
              got["delta"]["observed"]["coef"][:p], d_observed, 2e-9)
        check(f"common-fe:{vce}:added-delta-vs-LSDV",
              got["delta"]["added"]["coef"][:p], d_added, 2e-9)
        check(f"common-fe:{vce}:b-base-vs-explicit-dummies",
              got["b_base"], explicit["b_base"][:p], 2e-10)
        check(f"common-fe:{vce}:b-full-vs-explicit-dummies",
              got["b_full"], explicit["b_full"][:p], 2e-10)
        for name in ("observed", "added"):
            check(f"common-fe:{vce}:{name}-delta-vs-explicit-dummies",
                  got["delta"][name]["coef"][:p],
                  explicit["delta"][name]["coef"][:p], 2e-9)
        k_common = p + 1
        k_explicit = x1_explicit.shape[1] + 1
        common_rows = np.concatenate([
            g * k_common + np.arange(p) for g in range(2)
        ])
        explicit_rows = np.concatenate([
            g * k_explicit + np.arange(p) for g in range(2)
        ])
        check(f"common-fe:{vce}:cov-vs-explicit-dummies",
              got["cov"][np.ix_(common_rows, common_rows)],
              explicit["cov"][np.ix_(explicit_rows, explicit_rows)], 2e-9)
        check(f"common-fe:{vce}:base-cov-vs-explicit-dummies",
              got["base_cov"][:p, :p],
              explicit["base_cov"][:p, :p], 2e-9)
        check(f"common-fe:{vce}:delta-base-cross-vs-explicit-dummies",
              got["cov_delta_bbase"][np.ix_(common_rows, np.arange(p))],
              explicit["cov_delta_bbase"][
                  np.ix_(explicit_rows, np.arange(p))
              ], 2e-9)
        check(f"common-fe:{vce}:total-cov-vs-explicit-dummies",
              got["total_cov"][:p, :p],
              explicit["total_cov"][:p, :p], 2e-9)
        check(f"common-fe:{vce}:total-base-cross-vs-explicit-dummies",
              got["cov_total_bbase"][:p, :p],
              explicit["cov_total_bbase"][:p, :p], 2e-9)
        check(
            f"common-fe:{vce}:slope-identity",
            got["total"]["coef"][:p],
            got["b_base"] - got["b_full"],
            2e-10,
        )
        slope_positions = np.array([0, 1, 3, 4])
        check_condition(
            f"common-fe:{vce}:public-contract",
            got["converged"]
            and got["identity_gap"] < 2e-10
            and got["common_fe_names"] == ["cohort"]
            and got["n_common_fes"] == 1
            and got["common_fes_applied"] is True
            and got["intercept_inference_available"] is False
            and got["intercept_status"] == "not_certified_common_fes"
            and got["identity_status"] == "exact_ols_conditional_common_fes"
            and np.isnan(got["total"]["coef"][p])
            and np.isnan(got["total"]["se"][p])
            and np.isnan(got["base_cov"][p, p])
            and np.all(np.isfinite(
                got["cov"][np.ix_(slope_positions, slope_positions)]
            ))
            and got["regular_inference_status"]["observed"][p]
            == "not_applicable_common_fe_intercept",
        )

    check(
        "common-fe:vce-does-not-change-points",
        results["robust"]["total"]["coef"][:p],
        results["unadjusted"]["total"]["coef"][:p],
        0.0,
    )
    check_raises(
        "common-fe:common-only-is-not-a-decomposition",
        lambda: decompose(y, x1, common_fes={"cohort": common}),
        "at least one x2 group or added fixed-effect dimension",
    )
    check_raises(
        "common-fe:names-must-be-globally-unique",
        lambda: decompose(
            y, x1, {"same": x2}, {"added": added},
            common_fes={"same": common}
        ),
        "names must be unique",
    )
    check_raises(
        "common-fe:connected-require-fails-closed",
        lambda: decompose(
            y, x1, {"observed": x2},
            {"added": added, "added2": (added + common) % 19},
            common_fes={"cohort": common}, connected="require"
        ),
        "no common FEs",
    )

    diagnosed = decompose(
        y, x1, {"observed": x2},
        {"added": added, "added2": (added + common) % 19},
        common_fes={"cohort": common}
    )
    check_condition(
        "common-fe:added-fe-split-not-over-certified",
        diagnosed["fe_split_identified"] is False
        and diagnosed["fe_split_status"] == "not_certified_with_common_fes"
        and diagnosed["mobility_component_scope"]
        == "first_two_added_fe_dimensions"
        and diagnosed["connectivity_fe_names"] == ["added", "added2"],
    )


def connectivity_contract():
    """Exercise the executable retained-sample FE-identification contract."""
    rng = np.random.default_rng(20260724)

    # One connected worker-firm ring.
    workers = np.repeat(np.arange(16), 4)
    period = np.tile(np.arange(4), 16)
    firms = np.where(period < 2, workers % 8, (workers + 1) % 8)
    n = workers.size
    x = rng.normal(size=n)
    z = 0.25 * x + rng.normal(size=n)
    y = (
        0.7 * x + 0.4 * z + rng.normal(size=16)[workers]
        + rng.normal(size=8)[firms] + rng.normal(scale=0.3, size=n)
    )
    connected = decompose(
        y, x, {"observed": z}, {"worker": workers, "firm": firms}
    )
    check_condition(
        "connectivity:connected-two-way-certified",
        connected["converged"]
        and connected["n_mobility_components"] == 1
        and connected["largest_mobility_component_n_obs"] == n
        and connected["largest_mobility_component_share"] == 1.0
        and connected["largest_mobility_component_weight_share"] == 1.0
        and connected["fe_split_identified"] is True
        and connected["fe_split_status"] == "identified_two_way"
        and connected["connectivity_fe_indices"] == [0, 1]
        and connected["connectivity_fe_names"] == ["worker", "firm"]
        and connected["connectivity_pair_explicit"] is False
        and connected["connectivity_pair_status"] == "connected"
        and connected["connected_mode"] == "diagnose"
        and connected["mobility_component_scope"]
        == "first_two_fe_dimensions"
        and "normalization-dependent" not in connected["notes"],
    )
    required = decompose(
        y, x, {"observed": z}, {"worker": workers, "firm": firms},
        connected="require", connectivity_fes=("firm", "worker"),
    )
    check_condition(
        "connectivity:require-connected-two-way-passes",
        required["converged"]
        and required["fe_split_identified"] is True
        and required["connected_mode"] == "require"
        and required["connectivity_fe_indices"] == [1, 0]
        and required["connectivity_fe_names"] == ["firm", "worker"]
        and required["connectivity_pair_explicit"] is True
        and required["connectivity_pair_status"] == "connected"
        and required["mobility_component_scope"] == "selected_fe_pair",
    )
    check(
        "connectivity:require-is-numerically-inert",
        required["cov"], connected["cov"], 0.0,
    )

    # Two equal-sized disconnected rings plus one raw singleton. The singleton
    # carries a deliberately huge weight: exact 0.5/0.75 retained-sample
    # shares prove that diagnostics are computed after recursive singleton
    # removal, not on the input rows.
    local_worker = np.repeat(np.arange(8), 4)
    local_period = np.tile(np.arange(4), 8)
    local_firm = np.where(
        local_period < 2, local_worker % 4, (local_worker + 1) % 4
    )
    worker = np.concatenate([local_worker, 8 + local_worker, [16]])
    firm = np.concatenate([local_firm, 4 + local_firm, [8]])
    component = np.concatenate([
        np.zeros(local_worker.size, dtype=int),
        np.ones(local_worker.size, dtype=int),
        [2],
    ])
    n_input = worker.size
    x = component.astype(float) + 0.2 * rng.normal(size=n_input)
    z = 0.3 * x + rng.normal(size=n_input)
    y = (
        0.8 * x + 0.5 * z + rng.normal(size=17)[worker]
        + rng.normal(size=9)[firm] + rng.normal(scale=0.3, size=n_input)
    )
    weights = np.where(component == 0, 1.0, 3.0)
    weights[-1] = 100.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        disconnected = decompose(
            y, x, {"observed": z}, {"worker": worker, "firm": firm},
            weights=weights,
        )
    marker = "normalization-dependent"
    check_condition(
        "connectivity:disconnected-two-way-fails-loud",
        disconnected["converged"]
        and disconnected["n_obs_input"] == n_input
        and disconnected["n_singletons_dropped"] == 1
        and disconnected["n_obs"] == n_input - 1
        and disconnected["n_mobility_components"] == 2
        and disconnected["largest_mobility_component_n_obs"] == 32
        and disconnected["largest_mobility_component_share"] == 0.5
        and disconnected["largest_mobility_component_weight_share"] == 0.75
        and disconnected["fe_split_identified"] is False
        and disconnected["fe_split_status"] == "normalization_dependent"
        and disconnected["connectivity_pair_status"] == "disconnected"
        and disconnected["connected_mode"] == "diagnose"
        and marker in disconnected["notes"]
        and any(marker in str(item.message) for item in caught),
    )
    check(
        "connectivity:normalization-safe-fe-total",
        disconnected["fe_total"]["coef"],
        (disconnected["delta"]["worker"]["coef"]
         + disconnected["delta"]["firm"]["coef"]),
        0.0,
    )
    check_raises(
        "connectivity:require-disconnected-two-way-fails-closed",
        lambda: decompose(
            y, x, {"observed": z}, {"worker": worker, "firm": firm},
            weights=weights, connected="require",
        ),
        "connected(require) failed",
    )

    # A connected selected pair does not over-certify a three-way FE design.
    occupation = (workers + period) % 5
    with warnings.catch_warnings(record=True) as caught_multi:
        warnings.simplefilter("always", RuntimeWarning)
        multiway = decompose(
            y=(
                0.7 * connected["b_base"][0] * np.ones(n)
                + 0.4 * z[:n] + rng.normal(size=n)
            ),
            x1=x[:n],
            x2_groups={"observed": z[:n]},
            fes={
                "worker": workers,
                "firm": firms,
                "occupation": occupation,
            },
        )
    multi_marker = "not connectivity-certified"
    check_condition(
        "connectivity:multiway-does-not-over-certify",
        multiway["n_mobility_components"] == 1
        and multiway["fe_split_identified"] is False
        and multiway["fe_split_status"] == "not_certified_multiway"
        and multiway["connectivity_pair_status"] == "connected"
        and multi_marker in multiway["notes"]
        and any(multi_marker in str(item.message)
                for item in caught_multi),
    )

    # In a three-way design the first pair can be disconnected while another
    # selected pair is connected through its shared FE levels. Selection must
    # change only the pair diagnostic, never the decomposition or the global
    # multiway certification state.
    bridge = np.concatenate([
        local_period % 2,
        local_period % 2,
        [0],
    ])
    three_fes = {"worker": worker, "firm": firm, "bridge": bridge}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        first_pair = decompose(
            y, x, {"observed": z}, three_fes, weights=weights
        )
        selected_pair = decompose(
            y, x, {"observed": z}, three_fes, weights=weights,
            connectivity_fes=("worker", "bridge"),
        )
    check_condition(
        "connectivity:explicit-pair-selects-diagnostic-only",
        first_pair["n_mobility_components"] == 2
        and first_pair["connectivity_pair_status"] == "disconnected"
        and first_pair["fe_split_status"] == "not_certified_multiway"
        and selected_pair["n_mobility_components"] == 1
        and selected_pair["connectivity_pair_status"] == "connected"
        and selected_pair["connectivity_fe_indices"] == [0, 2]
        and selected_pair["connectivity_fe_names"] == ["worker", "bridge"]
        and selected_pair["connectivity_pair_explicit"] is True
        and selected_pair["mobility_component_scope"] == "selected_fe_pair"
        and selected_pair["fe_split_identified"] is False
        and selected_pair["fe_split_status"] == "not_certified_multiway",
    )
    check(
        "connectivity:pair-selection-does-not-change-numerics",
        selected_pair["cov"], first_pair["cov"], 0.0,
    )
    check_raises(
        "connectivity:require-rejects-multiway-even-when-pair-connected",
        lambda: decompose(
            y, x, {"observed": z}, three_fes,
            connectivity_fes=("worker", "bridge"),
            connected="require",
        ),
        "requires exactly two FE dimensions",
    )
    check_raises(
        "connectivity:selector-rejects-unknown-name",
        lambda: decompose(
            y, x, {"observed": z}, three_fes,
            connectivity_fes=("worker", "unknown"),
        ),
        "unknown FE name",
    )
    check_raises(
        "connectivity:selector-rejects-duplicate",
        lambda: decompose(
            y, x, {"observed": z}, three_fes,
            connectivity_fes=("worker", "worker"),
        ),
        "must be distinct",
    )

    one_way = decompose(
        y[:n], x[:n], {"observed": z[:n]}, {"worker": workers}
    )
    check_condition(
        "connectivity:single-fe-is-not-a-between-dimension-claim",
        one_way["n_mobility_components"] == 0
        and one_way["fe_split_identified"] is False
        and one_way["fe_split_status"] == "single_fe_dimension"
        and one_way["connectivity_fe_indices"] == []
        and one_way["connectivity_fe_names"] == []
        and one_way["connectivity_pair_status"] == "not_applicable"
        and one_way["connected_mode"] == "diagnose"
        and one_way["mobility_component_scope"] == "not_applicable",
    )


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
    check_condition(
        "fe-variance:between-fe-dominant-gated",
        default["fe_variance_status"]
        == ["conditional_only_between_fe_dominant"]
        and default["fe_variance_ratio_min"] == 0.35
        and default["delta"]["firm"]["se_type"].endswith(
            "_conditional_only_diagnostic"
        )
        and default["total"]["se_type"].endswith(
            "_conditional_only_diagnostic"
        ),
    )
    valid_x = 0.2 * group_signal + rng.normal(size=fe.size)
    valid_y = (
        0.8 * valid_x + 0.5 * z + rng.normal(size=n_groups)[fe]
        + rng.normal(scale=0.5, size=fe.size)
    )
    valid_fe = decompose(valid_y, valid_x, {"z": z}, {"firm": fe})
    check_condition(
        "fe-variance:within-dominant-valid",
        valid_fe["x1_fe_collinear_ratio"][0] > 0.35
        and valid_fe["fe_variance_status"] == ["valid_first_order"]
        and not valid_fe["delta"]["firm"]["se_type"].endswith(
            "_conditional_only_diagnostic"
        )
        and not valid_fe["total"]["se_type"].endswith(
            "_conditional_only_diagnostic"
        ),
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

    weak_x = rng.normal(size=(600, 1))
    weak_z = rng.normal(size=600)
    weak_design = np.column_stack([weak_x[:, 0], np.ones(600)])
    weak_y0 = 0.7 * weak_z + rng.normal(size=600)
    weak_resid = weak_y0 - weak_design @ np.linalg.lstsq(
        weak_design, weak_y0, rcond=None
    )[0]
    weak_fit = decompose(
        weak_resid + 1e-11 * weak_x[:, 0],
        weak_x, {"weak": weak_z}, x1_names=["x"],
    )
    with warnings.catch_warnings(record=True) as caught_weak:
        warnings.simplefilter("always", RuntimeWarning)
        weak_rows = tidy(
            weak_fit, focal="x", share="base",
            include_total=False, include_full=False,
        )
    check_condition(
        "share-base:weak-denominator-diagnostic-gate",
        all(row["share_defined"] for row in weak_rows)
        and all(row["share_denominator_t"] < 3.0 for row in weak_rows)
        and all(
            row["share_interval_status"]
            == "weak_denominator_delta_method_unreliable"
            for row in weak_rows
        )
        and all(
            row["share_se_type"].endswith(
                "_weak_denominator_diagnostic_only"
            )
            for row in weak_rows
        )
        and sum(
            "|t| < share_t_min" in str(item.message)
            for item in caught_weak
        ) == 1,
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


def regularity_contract():
    """Exercise Gelbach's nonregular product-inference boundary."""
    n = 256
    index = np.arange(n, dtype=np.uint64)

    def walsh(code):
        return np.asarray([
            1.0 if int(value & np.uint64(code)).bit_count() % 2 == 0
            else -1.0
            for value in index
        ])

    x1 = walsh(1)[:, None]
    orthogonal_x2 = walsh(2)
    residual = walsh(4)
    y0 = 1.2 * x1[:, 0] + 0.7 * residual
    cluster = (np.arange(n) * 17) % 31

    for vce, cluster_arg in (
            ("unadjusted", None), ("robust", None), ("cluster", cluster)):
        kwargs = {"vce": vce}
        if cluster_arg is not None:
            kwargs["cluster"] = cluster_arg
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            fit = decompose(
                y0, x1, {"orthogonal": orthogonal_x2},
                x1_names=["focal"], **kwargs
            )
        diagnostic = fit["regularity"]["orthogonal"]
        rows = tidy(
            fit, focal="focal", include_total=False, include_full=False
        )
        check_condition(
            f"regularity:{vce}:degenerate-product-fails-loud",
            fit["converged"]
            and fit["regular_inference_all_valid"] is False
            and diagnostic["regular_inference_valid"].tolist()
            == [False, False]
            and diagnostic["regular_inference_status"]
            == ["nonregular_not_ruled_out",
                "nonregular_not_ruled_out"]
            and diagnostic["beta2_wald_pvalue"] >
            fit["regularity_test_alpha"]
            and np.all(
                diagnostic["auxiliary_loading_pvalue"]
                > fit["regularity_test_alpha"]
            )
            and "regular first-order delta-method inference is not established"
            in fit["notes"]
            and any(
                "regular first-order delta-method inference is not established"
                in str(item.message) for item in caught
            )
            and rows[0]["regular_inference_valid"] is False
            and rows[0]["confidence_interval_status"]
            == "diagnostic_only_nonregular_not_ruled_out"
            and rows[0]["se_type"].endswith(
                "_nonregular_diagnostic_only"
            ),
        )
        check(
            f"regularity:{vce}:true-auxiliary-loadings",
            diagnostic["auxiliary_loadings"],
            np.zeros((2, 1)), 2e-14,
        )
        check(
            f"regularity:{vce}:zero-product-gradient",
            diagnostic["contribution_gradient_norm"],
            np.zeros(2), 2e-14,
        )

    # A nonzero full-model block coefficient makes every contribution row a
    # regular first-order product, even when its auxiliary loading is zero.
    y_beta = y0 + 0.5 * orthogonal_x2
    beta_regular = decompose(
        y_beta, x1, {"beta_signal": orthogonal_x2},
        x1_names=["focal"]
    )
    beta_diag = beta_regular["regularity"]["beta_signal"]
    check_condition(
        "regularity:beta-signal-certifies-all-rows",
        beta_regular["regular_inference_all_valid"] is True
        and beta_diag["regular_inference_valid"].tolist() == [True, True]
        and beta_diag["regular_inference_status"]
        == ["regular_beta_nonzero", "regular_beta_nonzero"]
        and beta_diag["beta2_wald_pvalue"]
        < beta_regular["regularity_test_alpha"]
        and not np.any(beta_diag["auxiliary_loading_test_evaluated"]),
    )

    # If beta2 is compatible with zero, regularity is contribution-specific:
    # a nonzero focal-row loading certifies that row, while the zero intercept
    # loading remains conservatively flagged.
    loaded_x2 = 0.8 * x1[:, 0] + orthogonal_x2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        loading_regular = decompose(
            y0, x1, {"loading_signal": loaded_x2},
            x1_names=["focal"]
        )
    loading_diag = loading_regular["regularity"]["loading_signal"]
    check_condition(
        "regularity:loading-signal-is-cell-specific",
        loading_diag["regular_inference_valid"].tolist() == [True, False]
        and loading_diag["regular_inference_status"][0]
        == "regular_loading_nonzero"
        and loading_diag["regular_inference_status"][1]
        == "nonregular_not_ruled_out"
        and loading_diag["auxiliary_loading_test_evaluated"].tolist()
        == [True, True]
        and loading_diag["auxiliary_loading_pvalue"][0]
        < loading_regular["regularity_test_alpha"]
        and loading_diag["beta2_wald_pvalue"]
        > loading_regular["regularity_test_alpha"],
    )

    # The shared-core chi-square survival probability is independently
    # checked against SciPy for a two-column block.
    rng = np.random.default_rng(20260725)
    x2_pair = np.column_stack([
        0.2 * x1[:, 0] + rng.normal(size=n),
        -0.1 * x1[:, 0] + rng.normal(size=n),
    ])
    y_pair = (
        0.7 * x1[:, 0] + x2_pair @ np.array([0.3, -0.2])
        + rng.normal(size=n)
    )
    pair = decompose(y_pair, x1, {"pair": x2_pair})
    pair_diag = pair["regularity"]["pair"]
    expected_p = chi2.sf(
        pair_diag["beta2_wald_stat"], pair_diag["beta2_wald_df"]
    )
    check(
        "regularity:shared-core-chi-square-tail",
        pair_diag["beta2_wald_pvalue"], expected_p, 2e-13,
    )
    check_condition(
        "regularity:public-schema-complete",
        pair["beta2"].shape == (2,)
        and pair["beta2_cov"].shape == (2, 2)
        and pair["auxiliary_loadings"].shape == (2, 2)
        and pair_diag["auxiliary_loading_rank"] >= 1
        and np.isfinite(
            pair_diag["auxiliary_loading_condition_number"]
        ),
    )

    # Under the joint null beta2=0 and Gamma=0, certification is the union of
    # the two component tests.  Each must therefore use alpha/2 while the
    # public regularity_test_alpha remains the family-wise level.
    rng = np.random.default_rng(2222)
    regular = 0
    reps = 500
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(reps):
            m = 300
            null_x = rng.normal(size=(m, 1))
            null_z = rng.normal(size=m)
            null_y = 2.0 * null_x[:, 0] + rng.normal(size=m)
            null_fit = decompose(
                null_y, null_x, {"null": null_z},
                vce="unadjusted", x1_names=["x"],
            )
            regular += (
                null_fit["regular_inference_status"]["null"][0]
                != "nonregular_not_ruled_out"
            )
    familywise_size = regular / reps
    check_condition(
        "regularity:family-wise-size-alpha-half",
        0.04 <= familywise_size <= 0.06,
        f"regular={regular}/{reps}, size={familywise_size:.3f}",
    )


def main():
    global decompose, tidy
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", default=None,
                    help="directory containing the py_hdfe_v11 extension to validate")
    args = ap.parse_args()
    if args.module_dir:
        sys.path.insert(0, os.path.abspath(args.module_dir))
        __import__("py_hdfe_v11")
    sys.path.insert(0, REPO_ROOT)
    from xhdfe.gelbach import decompose as loaded_decompose
    from xhdfe.gelbach import tidy as loaded_tidy
    decompose = loaded_decompose
    tidy = loaded_tidy

    input_contract()
    slow_chain()
    normalization_convention()
    common_fe_contract()
    connectivity_contract()
    inference_contract_attacks()
    regularity_contract()
    if FAIL:
        raise SystemExit(f"{len(FAIL)} adversarial check(s) failed: {FAIL}")
    print("ALL ADVERSARIAL GELBACH CHECKS PASSED")


if __name__ == "__main__":
    main()
