"""Adversarial regressions for CG scale invariance and solved RHS lanes, and
for the packed layout of designs with up to 16 regressors.

Run against a local CMake build, for example:

    XHDFE_TEST_BUILD_DIR=build_cuda python tests/test_cg_breakdown.py

Bit identity of the packed route up to 8 regressors with a build that predates
the wider packed layout is checked only when XHDFE_TEST_REFERENCE_BUILD_DIR
names that build's module directory; otherwise that test is skipped.
XHDFE_TEST_PACKED_LARGE=1 adds the longer packed-against-column cases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / os.environ.get("XHDFE_TEST_BUILD_DIR", "build_cuda")
sys.path.insert(0, str(BUILD_DIR))
import py_hdfe_v11 as cpp


def _fit(y, X, fes, *, convergence, weights=None):
    reg = cpp.HdfeRegressor(
        se_type="unadjusted",
        tol=1e-12,
        max_iter=100,
        fit_intercept=False,
        drop_singletons=False,
        num_threads=1,
        convergence=convergence,
        absorption_method="gauss-seidel",
        tolerance_mode="reghdfe-comparable",
    )
    kwargs = {"fes": fes}
    if weights is not None:
        kwargs["weights"] = weights
    reg.fit(y, X, **kwargs)
    assert reg.converged_
    assert reg.precision_certified_
    return reg


def _packed_fixture(scale):
    os.environ["XHDFE_PACKED"] = "1"
    i = np.arange(120)
    f1 = (i % 20).astype(np.int32)
    f2 = ((i * 7 + i // 20) % 20).astype(np.int32)
    y = 0.7 * np.sin(i / 3) + 0.2 * (f1 - 9.5) + 0.1 * (f2 - 9.5)
    X = np.cos(i / 5)[:, None]
    return _fit(
        scale * y,
        scale * X,
        [f1, f2],
        convergence="reghdfe",
    )


def _soa_fixture(scale):
    os.environ["XHDFE_PACKED"] = "0"
    i = np.arange(40)
    f1 = (i // 2).astype(np.int32)
    f2 = ((i + 1) // 2).astype(np.int32)
    y = np.sin(i / 3) + 0.2 * f1 - 0.1 * f2
    X = np.cos(i / 5)[:, None]
    return _fit(
        scale * y,
        scale * X,
        [f1, f2],
        convergence="auto",
        weights=np.ones(i.size),
    )


class CgBreakdownTest(unittest.TestCase):
    def test_near_redundant_fe_lanes_against_independent_projection(self):
        # Issue #16: most schools have age zero; only late entrants identify
        # the third FE. The rare outcome is solved before the random X lane.
        os.environ["XHDFE_PACKED"] = "1"
        sid = np.repeat(np.arange(1536), 5)
        year = np.tile(np.arange(2015, 2020), 1536)
        first = np.select(
            [sid < 1515, sid < 1522, sid < 1526, sid < 1532],
            [0, 2015, 2016, 2017], default=2018,
        )
        keep = year >= first
        sid, year, first = sid[keep], year[keep], first[keep]
        age = np.where(first == 0, 0, year - first + 1)
        fes = [v.astype(np.int32) for v in (sid, year - 2015, age)]
        rng = np.random.default_rng(16)
        X = rng.normal(size=(sid.size, 1))
        rare = ((sid < 2) | ((sid == 2) & (year != 2017)) |
                ((sid == 3) & (year == 2017))).astype(float)
        aggregate = 10 + rng.normal(size=1536)[sid] + .03*(year-2017)
        aggregate += .02*rng.normal(size=sid.size)

        def within(values):
            # Independent, unaccelerated alternating projections. No xhdfe
            # solver/certificate supplies the expected answer.
            out = values.copy()
            for _ in range(10000):
                before = out.copy()
                for fe in fes:
                    counts = np.bincount(fe)
                    for j in range(out.shape[1]):
                        means = np.bincount(fe, weights=out[:, j]) / counts
                        out[:, j] -= means[fe]
                if np.linalg.norm(out-before) < 1e-13:
                    return out
            self.fail("independent projection did not converge")

        for y in (rare, aggregate):
            projected = within(np.column_stack((y, X)))
            expected = np.linalg.lstsq(projected[:, 1:], projected[:, 0], rcond=None)[0]
            for scale in (1., 1e-10, 1e10):
                for order in (np.arange(sid.size), rng.permutation(sid.size)):
                    for convergence in ("auto", "reghdfe"):
                        with self.subTest(outcome="rare" if y is rare else "aggregate",
                                          scale=scale, convergence=convergence):
                            fit = _fit(scale*y[order], scale*X[order],
                                       [fe[order] for fe in fes],
                                       convergence=convergence)
                            np.testing.assert_allclose(fit.coef_, expected,
                                                       rtol=1e-8, atol=1e-12)

    def test_soa_early_solved_lane_against_explicit_dummies(self):
        i = np.arange(102)
        f1 = np.where(i < 3, 0, np.where(i < 99, 1+(i-3)//2, 49)).astype(np.int32)
        f2 = np.where(i < 2, 0, np.where(i == 2, 1,
                      np.where(i < 99, f1+(i-3)%2, np.where(i == 99, 49, 50)))).astype(np.int32)
        X = np.random.default_rng(91).normal(size=(i.size, 1))
        y = f1/7.0
        y[:2] += [1., -1.]
        dummy = np.column_stack([np.eye(fe.max()+1)[fe] for fe in (f1, f2)])
        expected = np.linalg.lstsq(np.column_stack((X, dummy)), y, rcond=None)[0][:1]
        # Unit weights explicitly select the SoA path rather than the packed
        # implementation, even when another test initialized the packed gate.
        for scale in (1., 1e-10, 1e10):
            fit = _fit(scale*y, scale*X, [f1, f2], convergence="auto",
                       weights=np.ones(i.size))
            np.testing.assert_allclose(fit.coef_, expected, rtol=1e-8, atol=1e-12)

    def test_cg_is_scale_invariant_and_precision_certified(self):
        for fixture in (_packed_fixture, _soa_fixture):
            with self.subTest(fixture=fixture.__name__):
                unit = fixture(1.0)
                tiny = fixture(1e-10)
                self.assertEqual(tiny.num_iterations_, unit.num_iterations_)
                np.testing.assert_allclose(
                    tiny.coef_, unit.coef_, rtol=1e-8, atol=1e-12
                )

    def test_cg_accepts_an_exactly_solved_zero_rhs_lane(self):
        os.environ["XHDFE_PACKED"] = "1"
        i = np.arange(120)
        f1 = (i % 20).astype(np.int32)
        f2 = ((i * 7 + i // 20) % 20).astype(np.int32)
        y = 0.7 * np.sin(i / 3) + 0.2 * (f1 - 9.5) + 0.1 * (f2 - 9.5)
        reg = _fit(
            y,
            np.zeros((i.size, 1)),
            [f1, f2],
            convergence="reghdfe",
        )
        np.testing.assert_array_equal(reg.coef_, np.zeros(1))


MODULE = Path(cpp.__file__).resolve()
PACKED_WIDTH_ROWS = 200_000
# The packed sweep switches to its blocked (fused) forms from 64 MiB of rows.
PACKED_FUSED_MIN_BYTES = 1 << 26


def _fused_expected(rows, p):
    return int(rows * (p + 1) * 8 >= PACKED_FUSED_MIN_BYTES)

# XHDFE_PACKED_MAX_COLS is read once per process, so every fit runs in a fresh
# worker. The design has one contiguous, one high-cardinality non-contiguous
# and one low-cardinality non-contiguous identifier: the three packed modes.
PACKED_WORKER = r"""
import importlib.util
import json
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("py_hdfe_v11", sys.argv[1])
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
n, p, threads = (int(value) for value in sys.argv[2:5])
tol = float(sys.argv[5])
i = np.arange(n, dtype=np.int64)
fes = [(i // 40).astype(np.int32),
       ((i * 7919) % 4999).astype(np.int32),
       ((i * 31) % 37).astype(np.int32)]
X = np.asfortranarray(np.column_stack([
    np.sin(0.001 * (j + 1) * i) + ((i * (17 + 2 * j)) % (101 + j)) / (101 + j)
    for j in range(p)]))
y = (X @ np.linspace(0.5, -0.5, p) + 0.01 * fes[0] - 0.02 * fes[1]
     + 0.03 * fes[2] + np.sin(0.0021 * i))
reg = core.HdfeRegressor(se_type="robust", tol=tol, drop_singletons=False,
                         num_threads=threads, absorption_method="gauss-seidel",
                         tolerance_mode="reghdfe-comparable")
reg.fit(y, X, fes=fes)
print(json.dumps(dict(
    coef=np.asarray(reg.coef_, dtype=np.float64).tolist(),
    covariance=np.asarray(reg.covariance_, dtype=np.float64).tolist(),
    iterations=int(reg.num_iterations_),
    converged=bool(reg.converged_),
    certified=bool(reg.precision_certified_))))
"""


def _packed_run(module, rows, p, threads, max_cols=None, tol=1e-10):
    env = os.environ.copy()
    env.update(XHDFE_PROFILE_CPU="1", XHDFE_GPU_BACKEND="cpu",
               PYTHONDONTWRITEBYTECODE="1")
    for name in ("XHDFE_PACKED", "XHDFE_PACKED_MAX_COLS", "XHDFE_PACKED_FUSED"):
        env.pop(name, None)
    if max_cols is not None:
        env["XHDFE_PACKED_MAX_COLS"] = str(max_cols)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", PACKED_WORKER, str(module), str(rows),
         str(p), str(threads), repr(tol)],
        env=env, capture_output=True, text=True, timeout=900)
    if completed.returncode != 0:
        raise AssertionError(
            f"packed worker rows={rows} p={p} threads={threads} "
            f"max_cols={max_cols} tol={tol!r} failed:\n{completed.stderr[-2000:]}")
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    # The packed sweep reports each dimension and its mode, and whether it uses
    # its fused forms, when profiled.
    result["modes"] = {int(mode) for mode in re.findall(
        r"label=abs_packed_dim dim=\d+ mode=(\d+)", completed.stderr)}
    result["fused"] = {int(active) for active in re.findall(
        r"label=abs_packed_fused active=([01]) bytes=\d+", completed.stderr)}
    return result


def _exact(result):
    return result["coef"], result["covariance"], result["iterations"]


def _assert_comparable(test, candidate, reference):
    # Comparable-mode contract against the reference route:
    # |db_j| <= 1e-9 max(1, |b_j|) and |dV_jk| <= 1e-8 sqrt(V_jj V_kk).
    b = np.asarray(candidate["coef"])
    b_ref = np.asarray(reference["coef"])
    V = np.asarray(candidate["covariance"])
    V_ref = np.asarray(reference["covariance"])
    test.assertEqual(b.shape, b_ref.shape)
    test.assertEqual(V.shape, V_ref.shape)
    b_limit = 1e-9 * np.maximum(1.0, np.abs(b_ref))
    b_ratio = np.max(np.abs(b - b_ref) / b_limit)
    test.assertTrue(b_ratio <= 1.0, f"coefficients: max |db|/limit = {b_ratio:.3g}")
    diagonal = np.diag(V_ref)
    test.assertTrue(np.all(diagonal > 0.0), "reference variances must be positive")
    V_limit = 1e-8 * np.sqrt(np.outer(diagonal, diagonal))
    V_ratio = np.max(np.abs(V - V_ref) / V_limit)
    test.assertTrue(V_ratio <= 1.0, f"covariance: max |dV|/limit = {V_ratio:.3g}")


class PackedWidthTest(unittest.TestCase):
    def test_switch_is_inert_up_to_eight_regressors(self):
        # The bound is inclusive: XHDFE_PACKED_MAX_COLS=p keeps p regressors packed.
        for p in (1, 4, 8):
            for threads in (1, 16):
                wide = _packed_run(MODULE, PACKED_WIDTH_ROWS, p, threads)
                self.assertEqual(wide["modes"], {0, 1, 2})
                for max_cols in sorted({8, p}):
                    with self.subTest(p=p, threads=threads, max_cols=max_cols):
                        narrow = _packed_run(MODULE, PACKED_WIDTH_ROWS, p, threads,
                                             max_cols=max_cols)
                        self.assertEqual(narrow["modes"], {0, 1, 2})
                        self.assertEqual(_exact(wide), _exact(narrow))

    def test_up_to_eight_regressors_match_the_reference_build(self):
        directory = os.environ.get("XHDFE_TEST_REFERENCE_BUILD_DIR")
        if not directory:
            self.skipTest("set XHDFE_TEST_REFERENCE_BUILD_DIR to a module built "
                          "without the wider packed layout to check bit identity")
        modules = sorted(Path(directory).glob("py_hdfe_v11*.so"))
        self.assertEqual(len(modules), 1, f"py_hdfe_v11 modules in {directory}")
        for p in (1, 4, 8):
            for threads in (1, 16):
                with self.subTest(p=p, threads=threads):
                    current = _packed_run(MODULE, PACKED_WIDTH_ROWS, p, threads)
                    reference = _packed_run(modules[0], PACKED_WIDTH_ROWS, p, threads)
                    self.assertEqual(current["modes"], {0, 1, 2})
                    self.assertEqual(reference["modes"], {0, 1, 2})
                    self.assertEqual(_exact(current), _exact(reference))

    def test_nine_to_sixteen_regressors_take_the_packed_route(self):
        # This design mixes slowly: at tol 1e-10 neither route is within the
        # covariance limit of the exact solution, so the routes are compared at
        # 1e-13 (docs/LICOES_ROTAS_VARRIMENTO_20261106.md).
        tol = 1e-13
        large = os.environ.get("XHDFE_TEST_PACKED_LARGE") == "1"
        # (rows, p, run only with XHDFE_TEST_PACKED_LARGE=1)
        cases = [(PACKED_WIDTH_ROWS, 9, False), (PACKED_WIDTH_ROWS, 16, False),
                 (500_000, 16, False), (PACKED_WIDTH_ROWS, 10, True),
                 (PACKED_WIDTH_ROWS, 12, True), (900_000, 9, True)]
        for rows, p, optional in cases:
            with self.subTest(rows=rows, p=p, threads=16):
                if optional and not large:
                    self.skipTest("skipped by default; set XHDFE_TEST_PACKED_LARGE=1 "
                                  "to run this case")
                packed = _packed_run(MODULE, rows, p, 16, tol=tol)
                column = _packed_run(MODULE, rows, p, 16, max_cols=8, tol=tol)
                self.assertEqual(packed["modes"], {0, 1, 2})
                self.assertEqual(column["modes"], set())
                self.assertEqual(packed["fused"], {_fused_expected(rows, p)})
                self.assertEqual(column["fused"], set())
                for result in (packed, column):
                    self.assertTrue(result["converged"] and result["certified"])
                _assert_comparable(self, packed, column)
        with self.subTest(rows=PACKED_WIDTH_ROWS, p=10, threads=(1, 16)):
            one = _packed_run(MODULE, PACKED_WIDTH_ROWS, 10, 1, tol=tol)
            sixteen = _packed_run(MODULE, PACKED_WIDTH_ROWS, 10, 16, tol=tol)
            fused = _fused_expected(PACKED_WIDTH_ROWS, 10)
            for result in (one, sixteen):
                self.assertEqual(result["modes"], {0, 1, 2})
                self.assertEqual(result["fused"], {fused})
                self.assertTrue(result["converged"] and result["certified"])
            self.assertEqual(_exact(one), _exact(sixteen))

    def test_seventeen_regressors_stay_on_the_column_layout(self):
        result = _packed_run(MODULE, PACKED_WIDTH_ROWS, 17, 16)
        self.assertEqual(result["modes"], set())
        self.assertTrue(result["converged"] and result["certified"])


if __name__ == "__main__":
    unittest.main()
