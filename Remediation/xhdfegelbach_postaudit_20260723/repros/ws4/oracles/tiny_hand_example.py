#!/usr/bin/env python3
"""WS4 Task A7: hand-solvable tiny examples in exact rational arithmetic.

Example 1 (n=8, one focal x, one added block z, no weights):
    x = [0,1,2,3,0,1,2,3];  z = [1,3,2,5,2,4,3,7];  y = [2,4,5,9,3,6,6,12]
  Solved with Fractions (exact Cramer solves), then compared to the live
  implementation. All quantities are exact rationals; the memo reproduces
  the numbers.

Example 2 (n=6, absorbed target): workers [0,0,1,1,2,2],
  focal f=[1,1,0,0,1,1] (worker-invariant), t=[0,1,0,1,0,1],
  y=[3,4,1,2,4,5]. Base: y on [1,f,t]. Full: y on [t] + worker dummies with
  f imposed to zero. total_f must equal the exact rational b_base_f.
"""
from fractions import Fraction as F
import sys

import numpy as np

sys.path.insert(0, "/home/mangelo/Documents/GitHub/xhdfe")
import xhdfe.gelbach as gb  # noqa: E402


def solve_frac(A, b):
    """Exact Gaussian elimination over Fractions."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [vr - f * vc for vr, vc in zip(M[r], M[c])]
    return [M[r][n] for r in range(n)]


def gram(cols):
    k = len(cols)
    return [[sum(a * b for a, b in zip(cols[i], cols[j])) for j in range(k)]
            for i in range(k)]


def xty(cols, y):
    return [sum(a * b for a, b in zip(c, y)) for c in cols]


def example1():
    x = [F(v) for v in [0, 1, 2, 3, 0, 1, 2, 3]]
    z = [F(v) for v in [1, 3, 2, 5, 2, 4, 3, 7]]
    y = [F(v) for v in [2, 4, 5, 9, 3, 6, 6, 12]]
    one = [F(1)] * 8
    # base: y on [x, 1]
    bb = solve_frac(gram([x, one]), xty([x, one], y))
    # full: y on [x, z, 1]
    bf = solve_frac(gram([x, z, one]), xty([x, z, one], y))
    # aux: z on [x, 1]
    gz = solve_frac(gram([x, one]), xty([x, one], z))
    delta = [gz[0] * bf[1], gz[1] * bf[1]]      # [x-row, cons-row]
    ident = [bb[0] - bf[0] - delta[0], bb[1] - bf[2] - delta[1]]
    print("Example 1 exact rationals:")
    print("  b_base  =", bb, "=", [float(v) for v in bb])
    print("  b_full  =", [bf[0], bf[2]], "g_z =", bf[1])
    print("  Gamma_z =", gz)
    print("  delta_Z =", delta, "=", [float(v) for v in delta])
    print("  identity residual (must be 0):", ident)
    assert ident == [F(0), F(0)]
    r = gb.decompose(np.array([float(v) for v in y]),
                     np.array([[float(v)] for v in x]),
                     {"Z": np.array([float(v) for v in z])}, None)
    errs = {
        "b_base": abs(float(r["b_base"][0]) - float(bb[0])),
        "b_full": abs(float(r["b_full"][0]) - float(bf[0])),
        "delta_x": abs(float(r["delta"]["Z"]["coef"][0]) - float(delta[0])),
        "delta_cons": abs(float(r["delta"]["Z"]["coef"][1]) - float(delta[1])),
        "total_x": abs(float(r["total"]["coef"][0]) - float(bb[0] - bf[0])),
    }
    print("  implementation abs errors vs exact rationals:", errs)
    assert max(errs.values()) < 1e-13, errs
    # unadjusted SE by hand: v = z - X1t gz; Omega_vv = sum(v^2)/df_full
    # aux residual of the CONTRIBUTION H = g_z * z on [x, 1]: v_H = g_z * v_z
    v = [zi - gz[0] * xi - gz[1] for zi, xi in zip(z, x)]
    u = [yi - bf[0] * xi - bf[1] * zi - bf[2]
         for yi, xi, zi in zip(y, x, z)]
    df_full = F(8 - 3)
    om_vv = bf[1] * bf[1] * sum(vi * vi for vi in v) / df_full
    s2 = sum(ui * ui for ui in u) / df_full
    # P = (X1t'X1t)^{-1}; element (0,0)
    G1 = gram([x, one])
    det = G1[0][0] * G1[1][1] - G1[0][1] * G1[1][0]
    P00 = G1[1][1] / det
    AFinv_zz_num = None
    # (A_F^{-1})[z,z] via cofactor of 3x3 gram of [x, z, 1]
    GF = gram([x, z, one])
    import itertools
    def det3(M):
        return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
                - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
                + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))
    dF = det3(GF)
    cof_zz = GF[0][0]*GF[2][2] - GF[0][2]*GF[2][0]
    AFinv_zz = cof_zz / dF
    var_delta_x = om_vv * P00 + s2 * gz[0] * AFinv_zz * gz[0]
    se_hand = float(var_delta_x) ** 0.5
    se_impl = float(r["delta"]["Z"]["se"][0])
    print(f"  hand unadjusted SE(delta_x) = {se_hand!r} impl = {se_impl!r} "
          f"absdiff = {abs(se_hand - se_impl):.2e}")
    assert abs(se_hand - se_impl) < 1e-13


def example2():
    w = [0, 0, 1, 1, 2, 2]
    f = [F(v) for v in [1, 1, 0, 0, 1, 1]]
    t = [F(v) for v in [0, 1, 0, 1, 0, 1]]
    y = [F(v) for v in [3, 4, 1, 2, 4, 5]]
    one = [F(1)] * 6
    bb = solve_frac(gram([f, t, one]), xty([f, t, one], y))
    print("Example 2 exact rationals:")
    print("  b_base = [f, t, cons] =", bb, "=", [float(v) for v in bb])
    d0 = [F(1), F(1), F(0), F(0), F(0), F(0)]
    d1 = [F(0), F(0), F(1), F(1), F(0), F(0)]
    d2 = [F(0), F(0), F(0), F(0), F(1), F(1)]
    bf = solve_frac(gram([t, d0, d1, d2]), xty([t, d0, d1, d2], y))
    print("  full (f imposed 0): t coef =", bf[0], " worker means =", bf[1:])
    r = gb.decompose(np.array([float(v) for v in y]),
                     np.column_stack([[float(v) for v in f],
                                      [float(v) for v in t]]),
                     None, {"worker": np.array(w)},
                     vce="cluster", cluster=np.array(w),
                     absorbed_targets=[0])
    err_total = abs(float(r["total"]["coef"][0]) - float(bb[0]))
    err_bfull_t = abs(float(r["b_full"][1]) - float(bf[0]))
    print("  impl total_f - exact b_base_f:", err_total)
    print("  impl b_full_t - exact within t coef:", err_bfull_t)
    print("  b_full_status:", r["b_full_status"], "| estimand:", r["estimand"])
    assert err_total < 1e-13 and err_bfull_t < 1e-13
    assert r["b_full"][0] == 0.0
    # base cluster VCE by hand (3 clusters): Stata regress convention
    rres = [yi - bb[0] * fi - bb[1] * ti - bb[2]
            for yi, fi, ti, tiu in zip(y, f, t, t)]
    X1t = [[fi, ti, F(1)] for fi, ti in zip(f, t)]
    A = gram([f, t, one])
    scores = []
    for g in range(3):
        s = [F(0)] * 3
        for i in range(6):
            if w[i] == g:
                for c in range(3):
                    s[c] += X1t[i][c] * rres[i]
        scores.append(s)
    M = [[sum(scores[g][i] * scores[g][j] for g in range(3))
          for j in range(3)] for i in range(3)]
    Ainv_cols = [solve_frac(A, [F(1) if r_ == c else F(0) for r_ in range(3)])
                 for c in range(3)]
    Ainv = [[Ainv_cols[j][i] for j in range(3)] for i in range(3)]
    def matmul(X, Y):
        return [[sum(X[i][k] * Y[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)]
    V = matmul(matmul(Ainv, M), Ainv)
    corr = (F(6 - 1) / F(6 - 3)) * (F(3) / F(2))
    var_f = corr * V[0][0]
    se_hand = float(var_f) ** 0.5
    se_impl = float(r["total"]["se"][0])
    print(f"  hand cluster SE(total_f) = {se_hand!r} impl = {se_impl!r} "
          f"absdiff = {abs(se_hand - se_impl):.2e}")
    assert abs(se_hand - se_impl) < 1e-13
    print("  inference_status:", r["inference_status"])


if __name__ == "__main__":
    example1()
    example2()
    print("TINY HAND EXAMPLES: ALL EXACT CHECKS PASSED")
