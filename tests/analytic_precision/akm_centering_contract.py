#!/usr/bin/env python3
"""Synthetic AKM location contract with an independent dense point oracle.

Use --baseline to collect the old API's results in a separate process.
The alternative's unsupported lowess/lincom combinations are tested, not counted as inference
validation. No input files, confidential data, or production installation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

KEYS = ("var_alpha", "var_psi", "cov_alpha_psi")


def panel(equal=False, stayers=False):
    pairs = [(w, f) for w in range(8) for f in range(4)]
    count = [3 if equal else 1 + (3 * w + 2 * f) % 5 for w, f in pairs]
    if stayers:
        pairs += [(8, 0), (9, 2)]
        count += [4, 5]
    w = np.repeat([v[0] for v in pairs], count).astype(np.int32)
    f = np.repeat([v[1] for v in pairs], count).astype(np.int32)
    rng = np.random.default_rng(18112026)
    x = rng.normal(size=len(w))
    g = rng.integers(0, 4, size=len(w))
    y = .21 * w - .37 * f + .5 * x + .17 * g + rng.normal(size=len(w)) * .3
    return y, w, f, x, g


def oracle(y, w, f, level, mode, controls=None, fw=None):
    """Dense weighted dummy regression, then exact leave-out correction."""
    n = len(y)
    wt = np.ones(n) if fw is None else np.asarray(fw, float)
    total = wt.sum()
    nw, nf = int(w.max()) + 1, int(f.max()) + 1
    Dw, Df = np.eye(nw)[w], np.eye(nf)[f, :nf - 1]
    design = np.column_stack((Dw, Df))
    if controls is not None:
        full = np.column_stack((design, controls))
        beta = np.linalg.lstsq(full * np.sqrt(wt)[:, None], y * np.sqrt(wt), rcond=None)[0]
        y = y - controls @ beta[design.shape[1]:]
    G = np.linalg.inv(design.T @ (wt[:, None] * design))
    coeff = G @ design.T @ (wt * y)
    worker = np.column_stack((Dw, np.zeros_like(Df)))
    firm = np.column_stack((np.zeros_like(Dw), Df))
    C = np.diag(wt) - np.outer(wt, wt) / total
    forms = [worker.T @ C @ worker, firm.T @ C @ firm,
             (worker.T @ C @ firm + firm.T @ C @ worker) / 2]
    pairs, membership = np.unique(np.column_stack((w, f)), axis=0, return_inverse=True)
    counts = np.bincount(membership, weights=wt)
    stayer = np.bincount(pairs[:, 0], minlength=nw)[pairs[:, 0]] == 1
    mean_y = wt @ y / total
    residual = y - design @ coeff
    if level == "match":
        roots = np.sqrt(counts)
        T = np.zeros((len(pairs), n))
        T[membership, np.arange(n)] = wt / roots[membership]
        Z = T @ design
        zy, e = T @ y, T @ residual
        h = np.einsum("ij,jk,ik->i", Z, G, Z)
        centred = zy - (roots * mean_y if mode == "person_year" else zy.mean())
        sigma = np.zeros(len(pairs))
        mover = ~stayer
        sigma[mover] = centred[mover] * e[mover] / (1 - h[mover])
        for j in np.nonzero(stayer)[0]:
            take = membership == j
            sigma[j] = (wt[take] * (y[take] - mean_y) * residual[take]).sum() / (
                counts[j] * (1 - 1 / counts[j]))
    else:
        Z = design
        h = np.einsum("ij,jk,ik->i", Z, G, Z)
        sigma = (y - mean_y) * residual / (1 - h)
    out = []
    for form in forms:
        B = np.einsum("ij,jk,ik->i", Z @ G, form, Z @ G)
        out.append(float((coeff @ form @ coeff - B @ sigma) / (total - 1)))
    return np.array(out)


def component(result, name="kss"):
    return np.array([result[name][k] for k in KEYS])


def numeric_snapshot(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: numeric_snapshot(v) for k, v in value.items()
                if k not in ("notes", "sigma_centering", "thread_limit_reason")}
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshots, checks = {}, []
    fits = 0

    def fit(y, w, f, **options):
        nonlocal fits
        if args.gpu:
            options.update(gpu=True, direct_max_firms=0)
        r = module.akm_kss(y, w, f, seed=18112026, num_threads=1, **options)
        fits += 1
        assert r["converged"] and r["sample"]["n_obs"] == int(
            np.asarray(options.get("fweights", np.ones(len(y)))).sum())
        assert np.all(r["sample"]["keep"])
        if args.gpu:
            assert r["gpu_used"] and r["gpu_backend"] == "cuda"
        return r

    def close(a, b, label):
        error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        assert np.allclose(a, b, atol=1e-11, rtol=1e-10), (label, error)
        checks.append({"name": label, "max_absolute_error": error})

    for kind, equal, stayers in (("unequal", False, False), ("equal", True, False),
                                  ("stayers", False, True)):
        y, w, f, x, g = panel(equal, stayers)
        for level in ("match", "obs"):
            for method in ("exact", "jla"):
                opts = {"leave_out_level": level, "leverages": method}
                key = f"{kind}/{level}/{method}"
                old = fit(y, w, f, **opts)
                snapshots[key] = numeric_snapshot(old)
                if method == "exact":
                    close(component(old), oracle(y, w, f, level, "leaveouttwoway"), key + "/legacy_oracle")
                if args.baseline:
                    continue
                explicit = fit(y, w, f, **opts, sigma_centering="leaveouttwoway")
                assert numeric_snapshot(old) == numeric_snapshot(explicit)
                assert old["sigma_centering"] == "leaveouttwoway"
                inv = fit(y, w, f, **opts, sigma_centering="person_year")
                assert inv["sigma_centering"] == "person_year"
                close(component(inv, "plugin"), component(old, "plugin"), key + "/plugin")
                close(inv["pii"], old["pii"], key + "/leverages")
                if method == "exact":
                    close(component(inv), oracle(y, w, f, level, "person_year"), key + "/invariant_oracle")
                for shift in (1.0, -10.0, 1000.0):
                    shifted = fit(y + shift, w, f, **opts, sigma_centering="person_year")
                    close(component(inv), component(shifted), key + f"/shift{shift}")
                if equal or level == "obs":
                    close(component(inv), component(old), key + "/unchanged_point_formula")

    y, w, f, x, g = panel(stayers=True)
    for method in ("exact", "jla"):
        for base in (0, 2):
            controls = np.column_stack((x, np.eye(4)[g][:, [j for j in range(4) if j != base]]))
            r = fit(y, w, f, X=controls, leverages=method)
            snapshots[f"controls/{method}/base{base}"] = numeric_snapshot(r)
            if not args.baseline:
                inv = fit(y, w, f, X=controls, leverages=method, sigma_centering="person_year")
                if base == 0:
                    first = component(inv)
                else:
                    close(first, component(inv), f"controls/{method}/base_invariance")
                if method == "exact":
                    close(component(inv), oracle(y, w, f, "match", "person_year", controls),
                          f"controls/{method}/base{base}/oracle")
        fw = 1 + np.arange(len(y)) % 3
        old = fit(y, w, f, leverages=method, fweights=fw)
        snapshots[f"fweights/{method}"] = numeric_snapshot(old)
        if not args.baseline:
            inv = fit(y, w, f, leverages=method, fweights=fw, sigma_centering="person_year")
            expanded = fit(np.repeat(y, fw), np.repeat(w, fw), np.repeat(f, fw),
                           leverages=method, sigma_centering="person_year")
            close(component(inv), component(expanded), f"fweights/{method}/replication")
            shifted = fit(y + 10, w, f, leverages=method, fweights=fw, sigma_centering="person_year")
            close(component(inv), component(shifted), f"fweights/{method}/shift")
            if method == "exact":
                close(component(inv), oracle(y, w, f, "match", "person_year", fw=fw), "fweights/oracle")

    for level in ("match", "obs"):
        r = fit(y, w, f, leave_out_level=level, leverages="exact", compute_se=True,
                se_nsim=1000, eigen_diagnostics=True)
        snapshots[f"legacy_inference/{level}"] = numeric_snapshot(r)
    rejected = []
    if not args.baseline:
        for name, opts in (("lowess", {"se_sigma_lowess": True}),
                           ("lincom", {"Z": x[:, None]})):
            try:
                fit(y, w, f, sigma_centering="person_year", **opts)
            except RuntimeError as e:
                assert "does not support" in str(e)
                rejected.append(name)
            else:
                raise AssertionError("unsupported inference was returned: " + name)
        try:
            fit(y, w, f, sigma_centering="invalid")
        except (ValueError, RuntimeError) as e:
            assert "sigma_centering" in str(e)
            rejected.append("unknown_mode")
        else:
            raise AssertionError("invalid mode accepted")
    result = {"status": "PASS_SCOPED_POINT_CONTRACT", "baseline": args.baseline,
              "module": str(args.module.resolve()),
              "module_sha256": hashlib.sha256(args.module.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "fits": fits, "checks": checks, "expected_rejections": rejected,
              "legacy_snapshots": snapshots,
              "limits": ["No alternative-mode inference certified", "No performance claim",
                         "Only the named synthetic cases and selected backend"]}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(result["status"], "fits=", fits, "checks=", len(checks),
          "expected_rejections=", len(rejected))


if __name__ == "__main__":
    main()
