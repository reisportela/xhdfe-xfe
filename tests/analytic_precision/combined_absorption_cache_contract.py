"""Exercise composition, invalidation and read-only behaviour of the two CPU caches."""
import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


def load_module(path):
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exercise(args):
    core = load_module(args.module)
    rng = np.random.default_rng(17)
    n = 2400
    a = (1000 + rng.integers(0, 60, n)).astype(np.int32)
    b = (4000 + rng.integers(0, 45, n)).astype(np.int32)
    a[-1], b[-1] = 9991, 9992
    x = rng.normal(size=(n, 2))
    y = x @ np.array([1.25, -.75]) + rng.normal(size=n) + .01 * (a + b)

    for frequency in (False, True):
        for mode in ("reghdfe-comparable", "xhdfe-fast"):
            case = args.scratch / (("fw_" if frequency else "plain_") + mode)
            case.mkdir()
            w = np.ones(n) if frequency else None
            if w is not None:
                w[-1] = 2

            def fit(label, absmode, femode, hit, *, folder="primary", weights=w,
                    response=y, first_fe=a, threads=2):
                path = case / folder
                path.mkdir(exist_ok=True)
                absorb, structure = path / "abs.bin", path / "fe.bin"
                read_before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                               for p, m in ((absorb, absmode), (structure, femode))
                               if m == "read" and p.exists()}
                os.environ.update(
                    XHDFE_ABSORPTION_CACHE=str(absorb), XHDFE_ABSORPTION_CACHE_MODE=absmode,
                    XHDFE_FE_STRUCTURE_CACHE=str(structure), XHDFE_FE_STRUCTURE_MODE=femode,
                )
                print(f"COMBINED_CACHE {case.name}/{label} {int(hit)}", file=sys.stderr, flush=True)
                model = core.HdfeRegressor(num_threads=threads, tolerance_mode=mode)
                model.fit(response, x, [first_fe, b], weights=weights, fweights=frequency)
                assert model.converged_ and model.precision_certified_, label
                assert all((p.read_bytes(), p.stat().st_mtime_ns) == before
                           for p, before in read_before.items()), label
                return model

            def same(fit_result, reference):
                np.testing.assert_array_equal(fit_result.sample_index_, reference.sample_index_)
                # Compare precision for the mode already selected on this design.
                np.testing.assert_allclose(fit_result.coef_, reference.coef_, rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(fit_result.covariance_, reference.covariance_,
                                           rtol=1e-8, atol=1e-12)

            ref = fit("ref", "off", "off", False)
            np.testing.assert_array_equal(ref.sample_index_, np.arange(n if frequency else n-1))
            same(fit("write", "write", "write", False), ref)
            same(fit("read1", "read", "read", True), ref)
            same(fit("read2", "read", "read", True), ref)
            same(fit("threads", "read", "read", False, threads=1), ref)
            same(fit("isolated_miss", "read", "off", False), ref)
            same(fit("auto1", "auto", "auto", False, folder="auto"), ref)
            same(fit("auto2", "auto", "auto", True, folder="auto"), ref)
            same(fit("isolated_write", "write", "off", False, folder="isolated"), ref)
            same(fit("isolated_hit", "read", "off", True, folder="isolated"), ref)
            same(fit("combined_miss", "read", "auto", False, folder="isolated"), ref)

            changed = a.copy()
            changed[-1] += 10   # leaves the retained design unchanged when unweighted
            same(fit("changed_fe", "read", "read", False, first_fe=changed), ref)
            shifted = y.copy()
            shifted[0] += .5
            shifted_ref = fit("changed_y_ref", "off", "off", False, response=shifted)
            same(fit("changed_y", "read", "read", False, response=shifted), shifted_ref)
            if frequency:
                w1 = w.copy()
                w1[-1] = 1
                w1_ref = fit("changed_fw_ref", "off", "off", False, weights=w1)
                np.testing.assert_array_equal(w1_ref.sample_index_, np.arange(n-1))
                same(fit("changed_fw", "read", "read", False, weights=w1), w1_ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        exercise(args)
        return

    args.scratch.mkdir(parents=True, exist_ok=False)
    env = {name: value for name, value in os.environ.items() if not name.startswith("XHDFE_")}
    env.update(XHDFE_GPU_BACKEND="cpu", XHDFE_MOBILITY_MODE="off",
               XHDFE_DEBUG_SOLVER="1", OPENBLAS_NUM_THREADS="1")
    run = subprocess.run([sys.executable, "-B", __file__, "--worker",
                          "--module", str(args.module), "--scratch", str(args.scratch)],
                         env=env, capture_output=True, text=True, timeout=120)
    (args.scratch / "trace.log").write_text(run.stderr)
    run.check_returncode()
    expected = re.findall(r"^COMBINED_CACHE (\S+) ([01])$", run.stderr, re.M)
    hits = re.findall(r"^xhdfe solver: .* cache_hit=([01])$", run.stderr, re.M)
    assert len(hits) == len(expected) and hits, (len(hits), len(expected))
    wrong = [(label, want, used) for (label, want), used in zip(expected, hits) if want != used]
    assert not wrong, wrong
    print(f"PASS: {len(hits)} combined-cache, precision and invalidation checks")


if __name__ == "__main__":
    main()
