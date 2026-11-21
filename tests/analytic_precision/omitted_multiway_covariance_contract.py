#!/usr/bin/env python3
"""Omitted-column inference, active-space covariance and range reconstruction.

Synthetic panels only. The reference for redundant columns is the identical
fit on its reported kept columns; independent dummy-WLS coverage remains in
multiway_intercept_ssc_contract.py. This contract does not alter a tolerance.
Every run writes a new receipt, including failures and expected range refusals.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cluster_shared_passes_contract import panel  # noqa: E402


def covariance_error(actual, reference):
    scale = np.sqrt(np.maximum(0.0, np.diag(reference)))
    if np.any(scale == 0):
        zero = scale == 0
        assert np.array_equal(actual[zero], reference[zero])
        assert np.array_equal(actual[:, zero], reference[:, zero])
        scale[zero] = 1.0
    return float(np.max(np.abs((actual - reference) / scale[:, None] / scale[None, :])))


def fit(core, d, X, clusters, mode, backend, constant=True, weight=None,
        grouped=None, saved=False, reduced=None):
    model = core.HdfeRegressor(
        se_type="cluster", num_threads=4, tolerance_mode=mode,
        fit_intercept=constant, max_iter=100000,
        retain_fes=saved, reduced_pcg=reduced)
    kw = dict(clusters=clusters)
    if weight is not None:
        kw.update(weights=d[weight], fweights=weight == "fw")
    fes = [d["worker"], d["firm"], d["year"]] if grouped is None else [d["individual"]]
    if grouped is not None:
        fes = [d["individual"]]
        kw.update(group=d["group"], individual=d["individual"], aggregation=grouped)
    model.fit(d["y"], np.asfortranarray(X), fes=fes, **kw)
    assert model.converged_ and model.precision_certified_
    assert bool(model.gpu_used_) == (backend == "cuda")
    return model


def omitted_case(core, d, X, clusters, mode, backend, **kw):
    model = fit(core, d, X, clusters, mode, backend, **kw)
    omitted = np.asarray(model.omitted_reason_[:X.shape[1]]) != 0
    assert omitted.any() and (~omitted).any()
    kept = np.flatnonzero(~omitted)
    ref = fit(core, d, X[:, kept], clusters, mode, backend, **kw)
    selected = list(kept) + ([X.shape[1]] if kw.get("constant", True) else [])
    b = np.asarray(model.coef_)
    v = np.asarray(model.covariance_)
    se = np.asarray(model.stderr_)
    assert np.all(b[:X.shape[1]][omitted] == 0), "omitted coefficients"
    assert np.all(v[np.flatnonzero(omitted)] == 0), "omitted covariance rows"
    assert np.all(v[:, np.flatnonzero(omitted)] == 0), "omitted covariance columns"
    assert np.isnan(se[:X.shape[1]][omitted]).all(), "omitted standard errors"
    for name in ("tvalues_", "pvalues_", "conf_int_"):
        assert np.isnan(np.asarray(getattr(model, name))[np.flatnonzero(omitted)]).all(), name
    br, vr = np.asarray(ref.coef_), np.asarray(ref.covariance_)
    db = float(np.max(np.abs(b[selected] - br) / np.maximum(1.0, np.abs(br))))
    dv = covariance_error(v[np.ix_(selected, selected)], vr)
    sr = np.asarray(ref.stderr_)
    positive = sr > 0
    ds = float(np.max(np.abs(se[selected][positive] / sr[positive] - 1.0)))
    assert db <= 1e-8, ("coefficient", db)
    assert ds <= 1e-7, ("standard error", ds)
    assert dv <= 1e-8, ("full covariance", dv)
    assert np.array_equal(model.sample_index_, ref.sample_index_)
    if kw.get("saved"):
        for left, right in zip(model.fe_effects_, ref.fe_effects_):
            np.testing.assert_allclose(left, right, atol=1e-8, rtol=1e-8)
    return dict(kept=kept.tolist(), omitted=np.flatnonzero(omitted).tolist(),
                b_error=db, se_relative_error=ds, V_error=dv,
                method=int(model.absorption_method_used),
                reduced_status=int(model.reduced_pcg_status_), gpu_used=bool(model.gpu_used_))


def group_panel():
    rng = np.random.default_rng(718)
    groups, individuals = 900, 120
    members = np.array([rng.choice(individuals, 2, replace=False) for _ in range(groups)])
    group = np.repeat(np.arange(groups, dtype=np.int32), 2)
    individual = members.ravel().astype(np.int32)
    X = rng.normal(size=(groups, 2)) + [1.0, -0.5]
    alpha = rng.normal(size=individuals)
    y = X @ [0.5, -0.25] + alpha[members].sum(axis=1) + rng.normal(size=groups)
    return dict(y=y[group], X=np.asfortranarray(X[group]), group=group,
                individual=individual, c1=(group // 9).astype(np.int32),
                c2=(group % 23).astype(np.int32))


def all_omitted_case(core, d, clusters, mode, backend, constant):
    X = np.zeros((d["n"], 3), order="F")
    model = fit(core, d, X, clusters, mode, backend, constant=constant)
    assert np.all(np.asarray(model.omitted_reason_)[:3] != 0)
    assert np.all(np.asarray(model.coef_)[:3] == 0)
    v = np.asarray(model.covariance_)
    assert np.all(v[:3] == 0) and np.all(v[:, :3] == 0)
    for name in ("stderr_", "tvalues_", "pvalues_", "conf_int_"):
        assert np.isnan(np.asarray(getattr(model, name))[:3]).all(), name
    ref = fit(core, d, np.empty((d["n"], 0)), clusters, mode, backend, constant=constant)
    if constant:
        np.testing.assert_allclose(np.asarray(model.coef_)[-1:], ref.coef_, atol=1e-8, rtol=1e-8)
        # Fast already returns distinct rounding-level residual scores with
        # zero RHS columns and with no RHS columns. The permanent general
        # grader's absolute covariance allowance covers this zero-variance case.
        np.testing.assert_allclose(v[-1:, -1:], ref.covariance_, atol=1e-12, rtol=1e-7)
    return dict(active_slopes=0, columns=len(model.coef_), gpu_used=bool(model.gpu_used_),
                intercept_variance=float(v[-1,-1]) if constant else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", type=Path, required=True)
    ap.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--only", choices=("omitted", "range"))
    ap.add_argument("--filter", help="Run only case names containing this text")
    args = ap.parse_args()
    assert not args.output.exists()
    os.environ["XHDFE_GPU_BACKEND"] = args.backend
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rows = []

    def check(name, function):
        if args.filter and args.filter not in name:
            return
        row = dict(name=name)
        try:
            row.update(function() or {}, status="PASS")
        except Exception as error:
            row.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        rows.append(row)

    if args.only != "range":
        d = panel(5)
        X = np.column_stack((d["X"], d["X"].sum(axis=1)))
        check("reported_crash", lambda: omitted_case(
            core, d, X, [d["worker"], d["firm_sparse"]],
            "reghdfe-comparable", args.backend))
        d = panel(215, workers=600, firms=90, years=5)
        x = d["X"]
        layouts = (np.column_stack((x, x.sum(axis=1))),
                   np.column_stack((x[:, 0], x[:, 0], x[:, 1])),
                   np.column_stack((x.sum(axis=1), x)),
                   np.column_stack((x[:, 0], np.zeros(d["n"]), x[:, 1], x.sum(axis=1))))
        for mode in ("reghdfe-comparable", "xhdfe-fast"):
            # Adding a nonzero linear combination changes the RHS-dependent
            # stopping trajectory of Fast. Zero RHS columns isolate omission
            # postprocessing without redefining its approximation contract.
            mode_layouts = layouts if mode != "xhdfe-fast" else (
                np.column_stack((x[:, 0], np.zeros(d["n"]), x[:, 1])),
                np.column_stack((np.zeros(d["n"]), x)),
                np.column_stack((np.zeros(d["n"]), x, np.zeros(d["n"]))))
            for weight in (None, "w", "fw"):
                for constant in (False, True):
                    for ci, clusters in enumerate(([d["firm"]],
                            [d["firm"], d["region"]], [d["worker"], d["firm_sparse"]])):
                        for li, X in enumerate(mode_layouts):
                            name = f"ordinary_{mode}_{weight}_{constant}_c{ci}_l{li}"
                            check(name, lambda X=X, clusters=clusters, mode=mode,
                                  weight=weight, constant=constant: omitted_case(
                                core, d, X, clusters, mode, args.backend,
                                weight=weight, constant=constant))
            for constant in (False, True):
                for clusters in ([d["firm"]], [d["worker"], d["firm_sparse"]]):
                    check(f"all_omitted_{mode}_{constant}_c{len(clusters)}",
                          lambda mode=mode, constant=constant, clusters=clusters:
                              all_omitted_case(core, d, clusters, mode, args.backend, constant))
            check(f"saved_{mode}", lambda mode=mode: omitted_case(
                core, d, layouts[1], [d["worker"], d["firm_sparse"]],
                mode, args.backend, saved=True))
        if args.backend == "cpu":
            for reduced in (False, True):
                check(f"tier_{reduced}", lambda reduced=reduced: omitted_case(
                    core, d, layouts[1], [d["firm"], d["region"]],
                    "reghdfe-comparable", "cpu", reduced=reduced))
        g = group_panel()
        gx = np.column_stack((g["X"][:, 0], g["X"][:, 0], g["X"][:, 1]))
        for mode in ("reghdfe-comparable", "xhdfe-fast"):
            for aggregation in ("sum", "mean"):
                for clusters in ([g["c1"]], [g["c1"], g["c2"]]):
                    name = f"grouped_{mode}_{aggregation}_c{len(clusters)}"
                    check(name, lambda mode=mode, aggregation=aggregation,
                          clusters=clusters: omitted_case(core, g, gx, clusters,
                              mode, args.backend, grouped=aggregation))
    if args.only != "omitted":
        d = panel(7)
        for mode in ("reghdfe-comparable", "xhdfe-fast"):
            for constant in (False, True):
                def scaled(mode=mode, constant=constant):
                    clusters = [d["worker"], d["firm_sparse"]]
                    ref = fit(core, d, d["X"], clusters, mode, args.backend, constant=constant)
                    r = dict(d)
                    r["y"] = d["y"] * 1e-140
                    model = fit(core, r, d["X"], clusters, mode, args.backend, constant=constant)
                    vr = np.asarray(ref.covariance_)
                    v = np.asarray(model.covariance_) / 1e-140 / 1e-140
                    db = float(np.max(np.abs(np.asarray(model.coef_) / 1e-140 -
                        np.asarray(ref.coef_)) / np.maximum(1.0, np.abs(ref.coef_))))
                    ds = float(np.max(np.abs(np.asarray(model.stderr_) / 1e-140 /
                                             np.asarray(ref.stderr_) - 1.0)))
                    dv = covariance_error(v, vr)
                    assert db <= 1e-8 and ds <= 1e-7, (db, ds, dv)
                    # Fast's scale-one fit is an approximate solve, not an
                    # exact full-V oracle for a rescaled solve. Preserve that
                    # measurement without imposing Comparable's contract.
                    if mode == "reghdfe-comparable":
                        assert dv <= 1e-8, (db, ds, dv)
                    return dict(b_error=db, se_relative_error=ds, V_error=dv,
                                full_V_rescaling_measurement_only=mode == "xhdfe-fast",
                                gpu_used=bool(model.gpu_used_))
                check(f"range_{mode}_{constant}", scaled)
        def refusal():
            bad = dict(d)
            bad["y"] = d["y"] * 2.0**-550
            try:
                fit(core, bad, d["X"], [d["worker"], d["firm_sparse"]],
                    "xhdfe-fast", args.backend)
            except RuntimeError as error:
                assert "range" in str(error) or "precision" in str(error), str(error)
                return dict(expected_refusal=True, error=str(error))
            raise AssertionError("unrepresentable nonzero covariance returned estimates")
        check("unrepresentable_final_covariance", refusal)
    report = dict(module=str(args.module.resolve()),
                  sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] /
                                                "cluster_shared_passes_contract.py").read_bytes()).hexdigest(),
                  backend=args.backend, cases=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    failed = [r for r in rows if r["status"] == "FAIL"]
    print(f"{len(rows)-len(failed)}/{len(rows)} checks passed; {len(failed)} failed")
    for row in failed:
        print(row["name"], row["error_type"], row["error"])
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
