#!/usr/bin/env python3
"""Held-out IV precision, small positive variance, width and thread contracts.

This CPU-only test uses known Walsh score spaces, never the dispatch decision
or the fitted coefficients to construct its oracle. The coefficient, full-V,
residual and RSS limits are inherited unchanged from
iv_projection_conditioning_contract.py. Each output report must be new.
Public collinearity-boundary comparisons require separate baseline workers;
they are deliberately outside this numerical contract.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


DELTAS = (0.02, 0.01, 0.001, 0.0005, 0.00025, 0.0002, 1e-6, 1.5e-6)
MODES = ("xhdfe-fast", "reghdfe-comparable")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fixture(delta, weighted, n=128, beta=3.0, noise=1.0, extra=0):
    """Within every 16-row block, all score/residual weighted moments vanish.

    Extra instruments use bits 4 onward, avoiding the endogenous disturbance
    bit 2 and structural error bit 3. Block-constant weights preserve their
    zero first-stage contribution, even when the extra instruments correlate
    with each other under weighting. The represented instrument's span is
    still exactly span(w,t,extras), including non-dyadic delta values.
    """
    if n % (2 ** (4 + extra)):
        raise ValueError("Walsh fixture needs complete blocks for every extra instrument")
    i = np.arange(n)
    w, t, v, e = (2.0 * ((i // (2**k)) % 2) - 1.0 for k in range(4))
    actual = np.column_stack((w, t + v))
    score = np.column_stack((w, t))
    coefficients = np.array([2.0, beta])
    residual = noise * e
    y = actual @ coefficients + residual
    instruments = [w + delta * t]
    instruments.extend(2.0 * ((i // (2**k)) % 2) - 1.0 for k in range(4, 4 + extra))
    z = np.column_stack(instruments)
    weights = 1.0 + i // 16 if weighted else None
    order = np.random.default_rng(23).permutation(n)
    return dict(y=y[order], X=actual[order], Z=z[order], score=score[order],
                beta=coefficients, residual=residual[order],
                weights=None if weights is None else weights[order], fes=[],
                clusters=(i % 7)[order], endogenous_idx=[1], fe_rank=0)


class ThreadFactory:
    """Override only the test's constructor request; leave the native API intact."""

    def __init__(self, cpp, threads):
        self.cpp = cpp
        self.threads = threads
        self.last = None

    def HdfeRegressor(self, **kwargs):
        kwargs["num_threads"] = self.threads
        self.last = self.cpp.HdfeRegressor(**kwargs)
        return self.last


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def run_case(contract, cpp, rows, kind, delta, weighted, mode, vce,
             n=128, beta=3.0, noise=1.0, extra=0, threads=2, comparisons=None):
    data = make_fixture(delta, weighted, n, beta, noise, extra)
    factory = ThreadFactory(cpp, threads)
    label = dict(kind=kind, delta=delta, weighted=weighted, n=n, beta=beta,
                 noise=noise, extra_instruments=extra, requested_threads=threads,
                 permutation_seed=23)
    row = contract.evaluate(factory, data, "cpu", mode, vce, label)
    if row["status"] == "PASS" and comparisons is not None:
        model = factory.last
        key = (delta, weighted, n, beta, noise, extra, mode, vce)
        current = dict(coef=np.asarray(model.coef_).copy(),
                       covariance=np.asarray(model.covariance_).copy(),
                       rss=float(model.rss_),
                       residual_sha256=digest_array(model.residuals_),
                       sample_sha256=digest_array(model.sample_index_))
        for name in ("threads_requested_", "threads_effective_", "parallel_workers_active_"):
            row[name.rstrip("_")] = int(getattr(model, name))
        reference = comparisons.setdefault(key, current)
        oracle_v, _, _ = contract.oracle(data, vce)
        vscale = np.sqrt(np.outer(np.diag(oracle_v), np.diag(oracle_v)))
        # Each run is independently graded above. Pair differences and exact
        # identity are diagnostics, without introducing another acceptance rule.
        row["thread_comparison"] = dict(
            scaled_beta_difference=float(np.max(np.abs(current["coef"] - reference["coef"]) /
                                                  np.maximum(1.0, np.abs(data["beta"])))),
            scaled_v_difference=float(np.max(np.abs(current["covariance"] - reference["covariance"]) /
                                               vscale)),
            rss_difference=abs(current["rss"] - reference["rss"]),
            coef_identical=np.array_equal(current["coef"], reference["coef"]),
            covariance_identical=np.array_equal(current["covariance"], reference["covariance"]),
            residual_identical=current["residual_sha256"] == reference["residual_sha256"],
            sample_identical=current["sample_sha256"] == reference["sample_sha256"])
    rows.append(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    module_path = args.module.resolve(strict=True)
    helper_path = Path(__file__).resolve().with_name("iv_projection_conditioning_contract.py")
    if args.out.exists():
        raise FileExistsError(f"report already exists: {args.out}")
    for key in tuple(os.environ):
        if key.startswith("XHDFE_"):
            del os.environ[key]
    os.environ["XHDFE_GPU_BACKEND"] = "cpu"
    source_paths = (module_path, Path(__file__).resolve(), helper_path)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    contract = load(helper_path, "iv_projection_conditioning_contract")
    cpp = load(module_path, "py_hdfe_v11")
    rows = []
    # The original 67 held-out cells, including three independent million-row fits.
    for delta in DELTAS:
        for weighted in (False, True):
            for mode in MODES:
                for vce in ("unadjusted", "cluster"):
                    run_case(contract, cpp, rows, "heldout", delta, weighted, mode, vce)
    for delta in (0.02, 0.00025, 1e-6):
        run_case(contract, cpp, rows, "heldout_million", delta, False,
                 "reghdfe-comparable", "unadjusted", n=1_000_000)
    # Dyadic coefficients and noise are exactly represented; covariance is
    # small but positive and retains the same relative full-V requirement.
    for delta in (0.02, 0.00025, 1e-6):
        for weighted in (False, True):
            for mode in MODES:
                for vce in ("unadjusted", "robust"):
                    for beta in (0.0, 0.25, 3.0):
                        run_case(contract, cpp, rows, "small_variance", delta,
                                 weighted, mode, vce, beta=beta, noise=2.0**-20)
    for delta in (0.02, 0.00025, 1e-6):
        for weighted in (False, True):
            for mode in MODES:
                for vce in ("unadjusted", "robust"):
                    run_case(contract, cpp, rows, "wide_instruments", delta,
                             weighted, mode, vce, n=4096, extra=6)
    comparisons = {}
    # Above the new helper's 20,000-row parallel threshold, including width.
    for delta in (0.02, 0.00025, 1e-6):
        for mode in MODES:
            for threads in (1, 2, 16, 48):
                run_case(contract, cpp, rows, "threads", delta, True, mode,
                         "unadjusted", n=65536, extra=6, threads=threads,
                         comparisons=comparisons)
    custody = all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)]
                  for path in source_paths)
    failures = sum(row["status"] != "PASS" for row in rows)
    counts = {}
    for row in rows:
        bucket = counts.setdefault(row["kind"], dict(cases=0, passed=0, failed=0))
        bucket["cases"] += 1
        bucket["passed" if row["status"] == "PASS" else "failed"] += 1
    report = dict(module=str(module_path), module_sha256=hashes[str(module_path)],
                  source_hashes=hashes, custody_unchanged=custody, backend="cpu",
                  oracle="known Walsh score space and structural-residual covariance",
                  limits=dict(coef=1e-9, diagonal_scaled_full_v=1e-8,
                              relative_se=1e-8, residual_absolute=1e-7, rss_absolute=1e-8),
                  scope_exclusions=["CUDA", "Stata/R", "public collinearity boundary versus baseline",
                                    "runtime acceptance"],
                  cases=len(rows), passed=len(rows)-failures, failed=failures,
                  categories=counts, status="PASS" if custody and not failures else "FAIL",
                  results=rows)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "cases", "passed", "failed", "categories")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
