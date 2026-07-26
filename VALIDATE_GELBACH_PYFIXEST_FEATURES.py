#!/usr/bin/env python3
"""Adversarial gates for the PyFixest-derived Gelbach feature surface.

This validator targets the capabilities reviewed in
``literature/gelbach_methods/pyfixest_v0.50.1_decomposition.py`` that sit
above the decomposition estimator:

* iid and declared-cluster pairs bootstrap;
* reproducible, independent replication streams and auditable failures;
* percentile/basic intervals for levels and both share denominators;
* table rendering; and
* identity-preserving waterfall data/plots with keep/drop/labels.

Every real replication calls the ordinary public ``gelbach.decompose``.
There is no bootstrap-only estimator.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    suffix = f": {detail}" if detail else ""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{suffix}")
    if not ok:
        FAILURES.append(name)


def check_array(name: str, got, expected, tol: float = 0.0) -> None:
    got = np.asarray(got)
    expected = np.asarray(expected)
    same_shape = got.shape == expected.shape
    same_special = (
        same_shape
        and np.array_equal(np.isnan(got), np.isnan(expected))
        and np.array_equal(np.isposinf(got), np.isposinf(expected))
        and np.array_equal(np.isneginf(got), np.isneginf(expected))
    )
    if same_special:
        finite = np.isfinite(got) & np.isfinite(expected)
        diff = (
            float(np.max(np.abs(got[finite] - expected[finite])))
            if np.any(finite) else 0.0
        )
    else:
        diff = float("inf")
    check(
        name,
        same_shape and same_special and diff <= tol,
        f"shape={got.shape}, max|diff|={diff:.3e}, tol={tol:.1e}",
    )


def check_raises(name: str, function, text: str) -> None:
    try:
        function()
    except Exception as exc:  # validation spans Python and C++ exceptions
        check(name, text in str(exc), f"{type(exc).__name__}: {exc}")
    else:
        check(name, False, "no exception raised")


def point_arrays(result):
    return {
        "delta": np.column_stack([
            result["delta"][name]["coef"] for name in result["names"]
        ]),
        "total": np.asarray(result["total"]["coef"]),
        "b_base": np.asarray(result["b_base"]),
        "b_full": np.asarray(result["b_full"]),
    }


def rng_states_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def fixture(seed=20260725):
    rng = np.random.default_rng(seed)
    n_clusters = 36
    cluster_size = 8
    n = n_clusters * cluster_size
    cluster = np.repeat(np.arange(n_clusters), cluster_size)
    common = np.tile(np.arange(cluster_size), n_clusters)
    added = rng.integers(0, 18, n)
    x1 = np.column_stack([
        rng.normal(size=n),
        rng.normal(size=n),
    ])
    human = np.column_stack([
        0.35 * x1[:, 0] + rng.normal(size=n),
        -0.20 * x1[:, 1] + rng.normal(size=n),
    ])
    job = 0.25 * x1[:, 0] - 0.15 * x1[:, 1] + rng.normal(size=n)
    y = (
        x1 @ np.array([0.9, -0.45])
        + human @ np.array([0.6, -0.25])
        + 0.4 * job
        + rng.normal(scale=0.6, size=n_clusters)[cluster]
        + rng.normal(scale=0.4, size=cluster_size)[common]
        + rng.normal(scale=0.3, size=18)[added]
        + rng.normal(scale=0.35, size=n)
    )
    weights = rng.uniform(0.4, 2.2, n)
    return {
        "y": y,
        "x1": x1,
        "groups": {"human": human, "job": job},
        "common": {"year": common},
        "added": {"occupation": added},
        "cluster": cluster,
        "weights": weights,
    }


def manual_iid_first_draw(gelbach, data, seed):
    child = np.random.SeedSequence(seed).spawn(1)[0]
    rng = np.random.Generator(np.random.PCG64(child))
    n = data["y"].size
    index = rng.integers(0, n, size=n, dtype=np.int64)
    return gelbach.decompose(
        data["y"][index],
        data["x1"][index],
        x2_groups={
            name: np.asarray(values)[index]
            for name, values in data["groups"].items()
        },
        common_fes={
            name: np.asarray(values)[index]
            for name, values in data["common"].items()
        },
        vce="unadjusted",
        x1_names=["target", "baseline_control"],
    )


def manual_cluster_first_draw(gelbach, data, seed):
    cluster = data["cluster"]
    unique, codes = np.unique(cluster, return_inverse=True)
    blocks = [np.flatnonzero(codes == code) for code in range(unique.size)]
    child = np.random.SeedSequence(seed).spawn(1)[0]
    rng = np.random.Generator(np.random.PCG64(child))
    selected = rng.integers(0, len(blocks), size=len(blocks), dtype=np.int64)
    index = np.concatenate([blocks[int(code)] for code in selected])
    return gelbach.decompose(
        data["y"][index],
        data["x1"][index],
        x2_groups={
            name: np.asarray(values)[index]
            for name, values in data["groups"].items()
        },
        common_fes={
            name: np.asarray(values)[index]
            for name, values in data["common"].items()
        },
        vce="unadjusted",
        x1_names=["target", "baseline_control"],
    )


def validate_bootstrap(gelbach, data, *, cuda_smoke=False):
    common = dict(
        x2_groups=data["groups"],
        common_fes=data["common"],
        x1_names=["target", "baseline_control"],
        vce="cluster",
        cluster=data["cluster"],
    )
    point = gelbach.decompose(data["y"], data["x1"], **common)

    np.random.seed(314159)
    state_before = np.random.get_state()
    pairs = gelbach.bootstrap(
        data["y"], data["x1"], method="pairs", reps=19, seed=12345,
        min_valid_reps=17, **common,
    )
    state_after = np.random.get_state()

    boot = pairs["bootstrap"]
    check("pairs:all-valid", boot["reps_valid"] == 19)
    check("pairs:ledger-complete", len(boot["ledger"]) == 19)
    check(
        "pairs:independent-stream-ledger",
        [entry["spawn_key"] for entry in boot["ledger"]]
        == [(index,) for index in range(19)],
    )
    check("pairs:does-not-touch-global-numpy-rng",
          rng_states_equal(state_before, state_after))
    check("pairs:point-vce-retained", boot["point_vce"] == "cluster")
    check(
        "pairs:replicate-vce-is-point-functional-only",
        boot["replicate_vce"] == "unadjusted_point_functional_only",
    )
    check(
        "pairs:nonregularity-caution-retained",
        boot["interval_status"]
        == "resampling_based_not_a_nonregularity_cure",
    )
    check_array("pairs:point-b-base-unchanged", pairs["b_base"],
                point["b_base"], 0.0)
    check_array("pairs:point-b-full-unchanged", pairs["b_full"],
                point["b_full"], 0.0)

    repeated = gelbach.bootstrap(
        data["y"], data["x1"], method="pairs", reps=19, seed=12345,
        min_valid_reps=17, **common,
    )
    check_array(
        "pairs:same-seed-bitwise-reproducible",
        repeated["bootstrap"]["draws"]["delta"],
        boot["draws"]["delta"],
        0.0,
    )
    changed = gelbach.bootstrap(
        data["y"], data["x1"], method="pairs", reps=5, seed=12346,
        min_valid_reps=5, **common,
    )
    check(
        "pairs:different-seed-different-draws",
        not np.array_equal(
            changed["bootstrap"]["draws"]["delta"],
            boot["draws"]["delta"][:5],
        ),
    )

    oracle = point_arrays(manual_iid_first_draw(gelbach, data, 12345))
    for key in ("delta", "total", "b_base", "b_full"):
        check_array(
            f"pairs:first-draw-full-refit-oracle:{key}",
            boot["draws"][key][0], oracle[key], 0.0,
        )

    alpha = 1.0 - boot["conf_level"]
    values = boot["draws"]["delta"][:, 0, 0]
    expected = np.quantile(
        values, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
    )
    check_array(
        "pairs:percentile-ci-oracle",
        [
            boot["intervals"]["delta"]["low"][0, 0],
            boot["intervals"]["delta"]["high"][0, 0],
        ],
        expected,
        0.0,
    )
    check(
        "pairs:all-level-and-share-intervals",
        set(boot["intervals"])
        == {
            "delta", "total", "b_base", "b_full", "share_base",
            "share_movement", "full_share_base", "total_share_base",
        },
    )

    cluster_seed = 54321
    clustered = gelbach.bootstrap(
        data["y"], data["x1"], method="cluster_pairs",
        bootstrap_cluster=data["cluster"], reps=13, seed=cluster_seed,
        min_valid_reps=12, ci_method="basic", **common,
    )
    cboot = clustered["bootstrap"]
    check("cluster:declared-unit",
          cboot["resampling_unit"] == "declared_cluster")
    cluster_oracle = point_arrays(
        manual_cluster_first_draw(gelbach, data, cluster_seed)
    )
    for key in ("delta", "total", "b_base", "b_full"):
        check_array(
            f"cluster:first-draw-full-refit-oracle:{key}",
            cboot["draws"][key][0], cluster_oracle[key], 0.0,
        )
    values = cboot["draws"]["total"][:, 0]
    q_low, q_high = np.quantile(
        values,
        [(1.0 - cboot["conf_level"]) / 2.0,
         1.0 - (1.0 - cboot["conf_level"]) / 2.0],
        method="linear",
    )
    point_total = clustered["total"]["coef"][0]
    check_array(
        "cluster:basic-ci-oracle",
        [
            cboot["intervals"]["total"]["low"][0],
            cboot["intervals"]["total"]["high"][0],
        ],
        [2.0 * point_total - q_high, 2.0 * point_total - q_low],
        0.0,
    )

    no_draws = gelbach.bootstrap(
        data["y"], data["x1"], method="pairs", reps=3, seed=99,
        min_valid_reps=3, store_draws=False, **common,
    )
    check(
        "pairs:store-draws-false",
        no_draws["bootstrap"]["draws"] is None
        and not no_draws["bootstrap"]["draws_stored"],
    )

    weighted = gelbach.bootstrap(
        data["y"], data["x1"], x2_groups=data["groups"],
        common_fes=data["common"], x1_names=["target", "baseline_control"],
        weights=data["weights"], method="pairs", reps=5, seed=818,
        min_valid_reps=5,
    )
    weighted_point = gelbach.decompose(
        data["y"], data["x1"], x2_groups=data["groups"],
        common_fes=data["common"], x1_names=["target", "baseline_control"],
        weights=data["weights"],
    )
    check_array("aweights:point-estimate-retained",
                weighted["b_full"], weighted_point["b_full"], 0.0)

    check_raises(
        "cluster:explicit-unit-required",
        lambda: gelbach.bootstrap(
            data["y"], data["x1"], x2_groups=data["groups"],
            method="cluster_pairs", reps=2,
        ),
        "requires bootstrap_cluster",
    )
    check_raises(
        "pairs:rejects-cluster-argument",
        lambda: gelbach.bootstrap(
            data["y"], data["x1"], x2_groups=data["groups"],
            method="pairs", bootstrap_cluster=data["cluster"], reps=2,
        ),
        "only valid",
    )
    check_raises(
        "fweights:ambiguous-resampling-rejected",
        lambda: gelbach.bootstrap(
            data["y"], data["x1"], x2_groups=data["groups"],
            weights=np.ones(data["y"].size), fweights=True, reps=2,
        ),
        "expanded-sample",
    )
    check_raises(
        "gpu:require-flag-needs-explicit-request",
        lambda: gelbach.bootstrap(
            data["y"], data["x1"], x2_groups=data["groups"],
            require_gpu_used=True, reps=2,
        ),
        "requires gpu=True",
    )
    if not cuda_smoke:
        check_raises(
            "gpu:required-use-fails-on-cpu-build",
            lambda: gelbach.bootstrap(
                data["y"], data["x1"], x2_groups=data["groups"],
                gpu=True, require_gpu_used=True, reps=2,
            ),
            "GPU use was required",
        )
    return pairs


def validate_gpu_bootstrap(gelbach, data):
    common = dict(
        x2_groups=data["groups"],
        common_fes=data["common"],
        fes=data["added"],
        x1_names=["target", "baseline_control"],
        gpu=True,
        require_gpu_used=True,
        reps=3,
        min_valid_reps=3,
    )
    for method, extra in (
        ("pairs", {}),
        ("cluster_pairs", {"bootstrap_cluster": data["cluster"]}),
    ):
        result = gelbach.bootstrap(
            data["y"], data["x1"], method=method, seed=919, **extra, **common,
        )
        boot = result["bootstrap"]
        check(
            f"gpu:{method}:point-used-real-cuda",
            result["gpu_used"] is True
            and result["gpu_backend"] == "cuda"
            and result["gpu_status"] == "used",
        )
        check(
            f"gpu:{method}:all-valid-replications-used-real-cuda",
            boot["gpu_required"] is True
            and boot["gpu_used_all_valid"] is True
            and boot["reps_valid"] == 3
            and all(
                row["status"] == "valid" and row["gpu_used"] is True
                for row in boot["ledger"]
            ),
        )


def validate_failure_ledger(gelbach, data):
    from xhdfe._gelbach_features import bootstrap as bootstrap_engine

    calls = {"count": 0}

    def flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] > 1 and calls["count"] % 3 == 0:
            raise RuntimeError("injected_replication_failure")
        return gelbach.decompose(*args, **kwargs)

    result = bootstrap_engine(
        flaky,
        data["y"],
        data["x1"],
        x2_groups=data["groups"],
        common_fes=data["common"],
        x1_names=["target", "baseline_control"],
        reps=9,
        min_valid_reps=6,
        seed=777,
    )
    boot = result["bootstrap"]
    check(
        "ledger:failed-replications-retained",
        boot["reps_valid"] == 6
        and boot["reps_failed"] == 3
        and len(boot["ledger"]) == 9
        and sum(row["status"] == "failed" for row in boot["ledger"]) == 3,
    )
    check(
        "ledger:failure-reasons-counted",
        sum(boot["failure_counts"].values()) == 3
        and all(
            "injected_replication_failure" in reason
            for reason in boot["failure_counts"]
        ),
    )

    calls["count"] = 0
    check_raises(
        "ledger:min-valid-fails-closed",
        lambda: bootstrap_engine(
            flaky,
            data["y"],
            data["x1"],
            x2_groups=data["groups"],
            common_fes=data["common"],
            x1_names=["target", "baseline_control"],
            reps=9,
            min_valid_reps=7,
            seed=777,
        ),
        "failed closed",
    )


def validate_reporting(gelbach, result):
    snapshot = {
        "b_base": np.array(result["b_base"], copy=True),
        "b_full": np.array(result["b_full"], copy=True),
        "delta": {
            name: np.array(result["delta"][name]["coef"], copy=True)
            for name in result["names"]
        },
    }
    records = gelbach.etable(
        result,
        type="records",
        panels=["levels", "share_full", "share_explained"],
        keep="human",
        labels={"human": "Human capital"},
    )
    check(
        "etable:three-panels-and-endpoints",
        {row["panel"] for row in records}
        == {"levels", "share_base", "share_movement"}
        and {"base_model", "total", "full_model"}.issubset(
            {row["component_kind"] for row in records}
        ),
    )
    check(
        "etable:keep-and-label",
        any(row["component"] == "Human capital" for row in records)
        and not any(
            row["component_name"] == "job"
            for row in records
        )
        and any(
            row["component_name"] == "other_filtered"
            and row["component"] == "Other (filtered)"
            for row in records
        ),
    )
    for keep, label in (
        (None, "all"),
        (["human"], "human"),
        (["human", "job"], "human-job"),
    ):
        identity_rows = gelbach.etable(
            result,
            type="records",
            panels=["levels", "share_base", "share_movement"],
            keep=keep,
            exact_match=True,
        )
        for panel in ("levels", "share_base", "share_movement"):
            component_sum = sum(
                row["estimate"] for row in identity_rows
                if row["coefficient"] == "target"
                and row["panel"] == panel
                and row["component_kind"] in {"x2", "fe",
                                               "filtered_aggregate"}
            )
            total = next(
                row["estimate"] for row in identity_rows
                if row["coefficient"] == "target"
                and row["panel"] == panel
                and row["component_kind"] == "total"
            )
            check(
                f"etable:identity:{label}:{panel}",
                abs(component_sum - total) <= 1e-12,
                f"gap={component_sum - total:.3e}",
            )
    other_level = next(
        row for row in records
        if row["coefficient"] == "target"
        and row["panel"] == "levels"
        and row["component_name"] == "other_filtered"
    )
    k1 = len(result["x1_names"]) + 1
    job_index = result["names"].index("job") * k1
    check(
        "etable:other-se-uses-joint-covariance",
        abs(
            other_level["std_error"]
            - np.sqrt(result["cov"][job_index, job_index])
        ) <= 1e-14,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_filtered = gelbach.etable(
            result, type="records", keep=["human"], exact_match=True,
            include_other=False,
        )
    check(
        "etable:include-other-false-warns",
        not any(row["component_name"] == "other_filtered"
                for row in legacy_filtered)
        and any("do not preserve the Gelbach accounting identity"
                in str(item.message) for item in caught),
    )
    check(
        "etable:bootstrap-intervals-auto",
        any(
            row["confidence_method"] == "bootstrap_percentile"
            for row in records
        ),
    )
    markdown = gelbach.etable(result, format="md", caption="A | B")
    latex = gelbach.etable(result, format="tex", caption="A_B")
    html = gelbach.etable(result, format="html", caption="<A>")
    check("etable:markdown", "| Panel |" in markdown and "A \\| B" in markdown)
    check("etable:latex", "\\begin{table}" in latex and "A\\_B" in latex)
    check("etable:html-escaped",
          "<table" in html and "&lt;A&gt;" in html)
    try:
        frame = gelbach.etable(result, format="df")
        check("etable:dataframe",
              list(frame.columns) == list(records[0].keys()))
    except ImportError as exc:
        check("etable:dataframe-optional-dependency",
              "optional pandas" in str(exc))
    try:
        table = gelbach.etable(result, format="gt")
        check("etable:great-tables", table is not None)
    except ImportError as exc:
        check("etable:great-tables-optional-dependency",
              "great_tables" in str(exc))

    waterfall = gelbach.waterfall_data(
        result,
        focal="target",
        keep="human",
        labels={"human": "Human capital"},
    )
    last = waterfall["rows"][-1]
    check(
        "waterfall:filtered-components-aggregated",
        any(row["kind"] == "filtered_aggregate"
            and row["members"] == ["job"] for row in waterfall["rows"]),
    )
    check("waterfall:identity-preserved",
          abs(last["waterfall_residual"]) <= 2e-12)
    check_raises(
        "waterfall:invalid-regex-is-audible",
        lambda: gelbach.waterfall_data(result, focal="target", keep="["),
        "invalid keep/drop regular expression",
    )

    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        with tempfile.TemporaryDirectory(prefix="xhdfe_gelbach_plot_") as tmp:
            mpl_config = Path(tmp) / "matplotlib"
            mpl_config.mkdir()
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
            target = Path(tmp) / "waterfall.png"
            figure, axis = gelbach.coefplot(
                result,
                focal="target",
                keep="human",
                labels={"human": "Human capital"},
                save=target,
            )
            plotted = axis._xhdfe_gelbach_waterfall
            check(
                "coefplot:waterfall-generated",
                target.is_file()
                and abs(plotted["rows"][-1]["waterfall_residual"]) <= 2e-12,
            )
            import matplotlib.pyplot as plt
            plt.close(figure)
    except ImportError as exc:
        check("coefplot:matplotlib-optional-dependency",
              "optional matplotlib" in str(exc))

    check_array("reporting:no-mutation:b-base",
                result["b_base"], snapshot["b_base"], 0.0)
    check_array("reporting:no-mutation:b-full",
                result["b_full"], snapshot["b_full"], 0.0)
    for name in result["names"]:
        check_array(
            f"reporting:no-mutation:delta:{name}",
            result["delta"][name]["coef"],
            snapshot["delta"][name],
            0.0,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module-dir",
        default=None,
        help="directory containing the py_hdfe_v11 extension to validate",
    )
    parser.add_argument(
        "--cuda-smoke",
        action="store_true",
        help=("also require real CUDA use in point estimates and every valid "
              "pairs/cluster-pairs bootstrap replication"),
    )
    args = parser.parse_args()
    if args.module_dir:
        sys.path.insert(0, os.path.abspath(args.module_dir))
        __import__("py_hdfe_v11")
    repo = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo)
    from xhdfe import gelbach

    data = fixture()
    result = validate_bootstrap(
        gelbach, data, cuda_smoke=args.cuda_smoke,
    )
    if args.cuda_smoke:
        validate_gpu_bootstrap(gelbach, data)
    validate_failure_ledger(gelbach, data)
    validate_reporting(gelbach, result)

    if FAILURES:
        raise SystemExit(
            f"{len(FAILURES)} PyFixest-feature gate(s) failed: {FAILURES}"
        )
    print("ALL PYFIXEST-DERIVED GELBACH FEATURE CHECKS PASSED")


if __name__ == "__main__":
    main()
