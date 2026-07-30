#!/usr/bin/env python3
"""Monte Carlo gates for the 25Jul2026 Gelbach remediation.

This validator deliberately lives outside the ordinary fast test suite. Its
strict defaults reproduce the reduced coverage exercises used for the
25Jul2026 Gelbach release certification:

* a weak share denominator at |t| approximately one, with at least 250 outer
  samples and a full-refit pairs bootstrap; and
* a fixed-population-FE random-design exercise for the FE-variance gate.

The FE values and the observation-to-FE assignment are drawn once and then
held fixed.  Only the focal covariate and regression error are redrawn.  This
is not the withdrawn super-population-FE experiment.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, REPO_ROOT)

FAIL = []


def check(name, condition, detail=""):
    ok = bool(condition)
    suffix = f": {detail}" if detail else ""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{suffix}", flush=True)
    if not ok:
        FAIL.append(name)


def _component_row(rows, component, coefficient="x"):
    return next(
        row for row in rows
        if row["component"] == component
        and row["coefficient"] == coefficient
    )


def fieller_gate_sweep(gelbach):
    """Exercise the status boundary on the auditor's weak-share family."""
    n = 800
    results = []
    for signal in (-0.480, -0.470, -0.455, -0.435, 0.200):
        rng = np.random.default_rng(4242)
        x = rng.normal(size=n)
        z = 0.6 * x + rng.normal(size=n)
        y = signal * x + 0.8 * z + rng.normal(size=n)
        fit = gelbach.decompose(
            y, x[:, None], x2_groups={"z": z},
            x1_names=["x"], vce="robust",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row = _component_row(
                gelbach.tidy(
                    fit, focal="x", share="base",
                    include_total=False, include_full=False,
                ),
                "z",
            )
        t_den = float(row["share_denominator_t"])
        status = row["share_interval_status"]
        results.append((signal, t_den, status))

    weak = [item for item in results if item[1] < 1.96]
    strong = results[-1]
    check(
        "share-fieller-sweep:all-t-below-1.96-gated",
        len(weak) >= 4 and all(
            item[2] == "weak_denominator_delta_method_unreliable"
            for item in weak
        ),
        ", ".join(
            f"signal={signal:.3f}:t={t_den:.2f}:{status}"
            for signal, t_den, status in results
        ),
    )
    check(
        "share-fieller-sweep:strong-denominator-silent",
        strong[1] >= 15.0 and strong[2] == "valid_first_order",
        f"signal={strong[0]:.3f}, |t|={strong[1]:.2f}, "
        f"status={strong[2]}",
    )


def weak_share_coverage(gelbach, *, outer, bootstrap_reps, calibration):
    """Compare analytic and pairs-bootstrap share coverage at |t_den| ~= 1."""
    n = 800
    signal = -0.455
    truth = 0.48 / (signal + 0.48)
    analytic_hits = 0
    bootstrap_hits = 0
    weak_statuses = 0
    status_mismatches = 0
    t_denominators = []

    for replication in range(outer):
        rng = np.random.default_rng(10_000 + replication)
        x = rng.normal(size=n)
        z = 0.6 * x + rng.normal(size=n)
        y = signal * x + 0.8 * z + rng.normal(size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fit = gelbach.decompose(
                y, x[:, None], x2_groups={"z": z},
                x1_names=["x"], vce="robust",
            )
            row = _component_row(
                gelbach.tidy(
                    fit, focal="x", share="base",
                    include_total=False, include_full=False,
                ),
                "z",
            )
            boot = gelbach.bootstrap(
                y, x[:, None], x2_groups={"z": z},
                x1_names=["x"], vce="robust",
                method="pairs", reps=bootstrap_reps,
                seed=1000 + replication, store_draws=False,
                min_valid_reps=math.ceil(0.9 * bootstrap_reps),
            )

        t_den = float(row["share_denominator_t"])
        t_denominators.append(t_den)
        expected_status = (
            "valid_first_order"
            if t_den >= 3.0
            else "weak_denominator_delta_method_unreliable"
        )
        status_mismatches += row["share_interval_status"] != expected_status
        weak_statuses += (
            row["share_interval_status"]
            == "weak_denominator_delta_method_unreliable"
        )
        analytic_hits += (
            row["share_conf_low"] <= truth <= row["share_conf_high"]
        )
        interval = boot["bootstrap"]["intervals"]["share_base"]
        low = float(np.asarray(interval["low"])[0, 0])
        high = float(np.asarray(interval["high"])[0, 0])
        bootstrap_hits += low <= truth <= high

    analytic_coverage = analytic_hits / outer
    bootstrap_coverage = bootstrap_hits / outer
    weak_fraction = weak_statuses / outer
    mean_t = float(np.mean(t_denominators))
    print(
        "share coverage: "
        f"outer={outer}, B={bootstrap_reps}, "
        f"mean|t_den|={mean_t:.3f}, "
        f"analytic={analytic_coverage:.3f}, "
        f"bootstrap={bootstrap_coverage:.3f}, "
        f"weak-status={weak_fraction:.3f}",
        flush=True,
    )
    check(
        "share-coverage:status-exactly-follows-threshold",
        status_mismatches == 0,
        f"mismatches={status_mismatches}/{outer}",
    )
    check(
        "share-coverage:weak-design-calibrated-near-one",
        0.75 <= mean_t <= 1.25 and weak_fraction >= 0.90,
        f"mean|t|={mean_t:.3f}, weak_fraction={weak_fraction:.3f}",
    )
    if not calibration:
        check(
            "share-coverage:strict-outer-repetitions",
            outer >= 250,
            f"outer={outer}",
        )
        check(
            "share-coverage:pairs-bootstrap-at-least-0.90",
            bootstrap_coverage >= 0.90,
            f"coverage={bootstrap_coverage:.3f}",
        )


def _fixed_fe_population():
    rng = np.random.default_rng(999)
    # Keep the realized finite-population variance.  Standardizing this draw
    # would change the auditor's reported x1_fe_collinear_ratio calibration
    # (0.302/0.127/0.039 in the gated cells).
    firm_signal = rng.normal(size=40)
    # Keep the outcome FE distinct from the between-firm component of x.  The
    # 1.30 scale reproduces the corrected audit's severity calibration without
    # redrawing FE values across outer samples.
    outcome_fe = 1.30 * rng.normal(size=40)
    return firm_signal, outcome_fe


def _fe_sample(population, between_share, rng, n, firm=None):
    firm_signal, outcome_fe = population
    if firm is None:
        firm = rng.integers(0, firm_signal.size, size=n)
    else:
        firm = np.asarray(firm)
    loading = math.sqrt(between_share / (1.0 - between_share))
    x = loading * firm_signal[firm] + rng.normal(size=n)
    y = x + outcome_fe[firm] + rng.normal(size=n)
    signal = firm_signal[firm]
    effects = outcome_fe[firm]
    signal_variance = float(np.var(signal))
    signal_effect_covariance = float(np.mean(
        (signal - np.mean(signal)) * (effects - np.mean(effects))
    ))
    truth = (
        loading * signal_effect_covariance
        / (loading * loading * signal_variance + 1.0)
    )
    return y, x[:, None], firm, truth


def fe_gate_sweep(gelbach):
    """Check the requested 0.05/.../0.97 between-FE status sweep."""
    population = _fixed_fe_population()
    shares = (0.05, 0.25, 0.50, 0.75, 0.90, 0.97)
    observed = []
    for index, share in enumerate(shares):
        rng = np.random.default_rng(50_000 + index)
        y, x, firm, _ = _fe_sample(population, share, rng, n=6000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fit = gelbach.decompose(
                y, x, fes={"f": firm}, x1_names=["x"],
                vce="unadjusted",
            )
            ungated_label = gelbach.decompose(
                y, x, fes={"f": firm}, x1_names=["x"],
                vce="unadjusted", fe_variance_ratio_min=0.0,
            )
        ratio = float(fit["x1_fe_collinear_ratio"][0])
        status = fit["fe_variance_status"][0]
        observed.append((share, ratio, status))
        check(
            f"fe-sweep:q={share:.2f}:numeric-output-unchanged",
            np.array_equal(fit["b_base"], ungated_label["b_base"])
            and np.array_equal(fit["b_full"], ungated_label["b_full"])
            and np.array_equal(fit["cov"], ungated_label["cov"])
            and np.array_equal(
                fit["delta"]["f"]["se"],
                ungated_label["delta"]["f"]["se"],
            ),
        )

    check(
        "fe-sweep:q-at-most-0.50-valid",
        all(
            status == "valid_first_order"
            for share, _, status in observed if share <= 0.50
        ),
        ", ".join(
            f"q={share:.2f}:ratio={ratio:.3f}:{status}"
            for share, ratio, status in observed
        ),
    )
    check(
        "fe-sweep:q-at-least-0.75-gated",
        all(
            status == "conditional_only_between_fe_dominant"
            for share, _, status in observed if share >= 0.75
        ),
        ", ".join(
            f"q={share:.2f}:ratio={ratio:.3f}:{status}"
            for share, ratio, status in observed
        ),
    )


def fe_bootstrap_coverage(
        gelbach, *, outer, bootstrap_reps, n, calibration
):
    """Coverage of the routed pairs bootstrap in the three gated cells."""
    population = _fixed_fe_population()
    for cell, share in enumerate((0.75, 0.90, 0.97)):
        truth_hits = 0
        gated = 0
        estimates = []
        analytic_ses = []
        lows = []
        highs = []
        ratios = []
        rng = np.random.default_rng(2026)
        # The FE support and its realized composition are part of the fixed
        # population.  Across outer samples only x and the regression errors
        # are redrawn, exactly as required by the corrected audit design.
        firm = np.arange(n, dtype=np.int64) % population[0].size
        for replication in range(outer):
            y, x, _, truth = _fe_sample(
                population, share, rng, n=n, firm=firm
            )
            bootstrap_seed = int(rng.integers(0, 2 ** 32))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = gelbach.bootstrap(
                    y, x, fes={"f": firm}, x1_names=["x"],
                    vce="unadjusted", method="pairs",
                    reps=bootstrap_reps,
                    seed=bootstrap_seed,
                    store_draws=False,
                    min_valid_reps=math.ceil(0.9 * bootstrap_reps),
                )
            interval = result["bootstrap"]["intervals"]["delta"]
            low = float(np.asarray(interval["low"])[0, 0])
            high = float(np.asarray(interval["high"])[0, 0])
            truth_hits += low <= truth <= high
            gated += (
                result["fe_variance_status"][0]
                == "conditional_only_between_fe_dominant"
            )
            estimates.append(float(result["delta"]["f"]["coef"][0]))
            analytic_ses.append(float(result["delta"]["f"]["se"][0]))
            lows.append(low)
            highs.append(high)
            ratios.append(float(result["x1_fe_collinear_ratio"][0]))

        estimates = np.asarray(estimates)
        analytic_ses = np.asarray(analytic_ses)
        lows = np.asarray(lows)
        highs = np.asarray(highs)
        mc_target = float(np.mean(estimates))
        coverage = float(np.mean((lows <= mc_target) & (mc_target <= highs)))
        truth_coverage = truth_hits / outer
        analytic_coverage = float(np.mean(
            np.abs(estimates - mc_target)
            <= 1.959963984540054 * analytic_ses
        ))
        sampling_sd = float(np.std(estimates, ddof=1))
        gated_fraction = gated / outer
        print(
            "FE bootstrap coverage: "
            f"q={share:.2f}, outer={outer}, B={bootstrap_reps}, n={n}, "
            f"analytic={analytic_coverage:.3f}, "
            f"bootstrap_MC_target={coverage:.3f}, "
            f"bootstrap_known_truth={truth_coverage:.3f}, "
            f"SE/MCsd={np.mean(analytic_ses) / sampling_sd:.3f}, "
            f"gated={gated_fraction:.3f}, "
            f"mean_ratio={np.mean(ratios):.3f}, "
            f"bias={mc_target - truth:+.4f}",
            flush=True,
        )
        check(
            f"fe-bootstrap:q={share:.2f}:gate-active",
            gated_fraction >= 0.95,
            f"gated={gated_fraction:.3f}",
        )
        if not calibration:
            check(
                f"fe-bootstrap:q={share:.2f}:coverage-at-least-0.93",
                coverage >= 0.93,
                f"MC-target coverage={coverage:.3f}, "
                f"known-truth coverage={truth_coverage:.3f}",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module-dir", default=None,
        help="directory containing the py_hdfe_v11 extension to validate",
    )
    parser.add_argument("--share-outer", type=int, default=250)
    parser.add_argument("--share-bootstrap-reps", type=int, default=399)
    parser.add_argument("--fe-outer", type=int, default=200)
    parser.add_argument("--fe-bootstrap-reps", type=int, default=199)
    parser.add_argument("--fe-n", type=int, default=1500)
    parser.add_argument(
        "--calibration", action="store_true",
        help="run reduced exploratory counts without asserting coverage floors",
    )
    args = parser.parse_args()
    if args.module_dir:
        sys.path.insert(0, os.path.abspath(args.module_dir))
        __import__("py_hdfe_v11")
    sys.path.insert(0, REPO_ROOT)
    from xhdfe import gelbach

    fieller_gate_sweep(gelbach)
    weak_share_coverage(
        gelbach,
        outer=args.share_outer,
        bootstrap_reps=args.share_bootstrap_reps,
        calibration=args.calibration,
    )
    fe_gate_sweep(gelbach)
    fe_bootstrap_coverage(
        gelbach,
        outer=args.fe_outer,
        bootstrap_reps=args.fe_bootstrap_reps,
        n=args.fe_n,
        calibration=args.calibration,
    )
    if FAIL:
        raise SystemExit(
            f"{len(FAIL)} remediation coverage check(s) failed: {FAIL}"
        )
    qualifier = "CALIBRATION" if args.calibration else "STRICT"
    print(f"ALL {qualifier} GELBACH REMEDIATION COVERAGE CHECKS PASSED")


if __name__ == "__main__":
    main()
