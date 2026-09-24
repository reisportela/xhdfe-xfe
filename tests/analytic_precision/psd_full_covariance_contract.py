"""Offline high-precision full covariance, including the translated intercept.

The reference uses 80-digit arithmetic on the represented inputs. No
observed discrepancy is used as an allowance.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

from psd_origin_contract import fixture


def reference_base(y, x, fe, clusters, weights):
    n, p = x.shape
    w = [mp.mpf(float(v)) for v in weights]
    xx = [[mp.mpf(float(v)) for v in row] for row in x]
    yy = [mp.mpf(float(v)) for v in y]
    total = sum(w)
    mx = [sum(w[i]*xx[i][j] for i in range(n))/total for j in range(p)]
    my = sum(w[i]*yy[i] for i in range(n))/total
    wx = [[mp.mpf(0)]*p for _ in range(10)]
    wy, mass = [mp.mpf(0)]*10, [mp.mpf(0)]*10
    for i in range(n):
        g = int(fe[i])
        mass[g] += w[i]
        wy[g] += w[i]*yy[i]
        for j in range(p):
            wx[g][j] += w[i]*xx[i][j]
    xt = [[xx[i][j]-wx[int(fe[i])][j]/mass[int(fe[i])] for j in range(p)] for i in range(n)]
    yt = [yy[i]-wy[int(fe[i])]/mass[int(fe[i])] for i in range(n)]
    gram = mp.matrix(p)
    rhs = mp.matrix(p, 1)
    for j in range(p):
        rhs[j] = sum(w[i]*xt[i][j]*yt[i] for i in range(n))
        for k in range(p):
            gram[j, k] = sum(w[i]*xt[i][j]*xt[i][k] for i in range(n))
    inverse = gram**-1
    beta = inverse*rhs
    coefficient = mp.matrix([*beta, my-sum(mx[j]*beta[j] for j in range(p))])
    scores = []
    for i in range(n):
        residual = yt[i]-sum(xt[i][j]*beta[j] for j in range(p))
        influence = inverse*mp.matrix([w[i]*v for v in xt[i]])
        constant = w[i]/total-sum(mx[j]*influence[j] for j in range(p))
        scores.append([*(residual*v for v in influence), residual*constant])
    covariance = mp.matrix(p+1)
    absolute_terms = mp.matrix(p+1)
    for dimensions, sign in (((0,), 1), ((1,), 1), ((0, 1), -1)):
        totals = {}
        for i in range(n):
            label = tuple(int(clusters[d][i]) for d in dimensions)
            if label not in totals:
                totals[label] = [mp.mpf(0)]*(p+1)
            for j in range(p+1):
                totals[label][j] += scores[i][j]
        for values in totals.values():
            for j in range(p+1):
                for k in range(p+1):
                    term = values[j]*values[k]
                    covariance[j, k] += sign*term
                    absolute_terms[j, k] += abs(term)
    sd = [max(mp.mpf('.001'), mp.sqrt(sum(w[i]*(xx[i][j]-mx[j])**2
          for i in range(n))/(total-1))) for j in range(p)] + [mp.mpf(1)]
    return coefficient, covariance, absolute_terms, sd


def clamp(covariance, scales):
    diagonal = mp.diag(scales)
    inverse = mp.diag([1/v for v in scales])
    values, vectors = mp.eigsy(diagonal*covariance*diagonal)
    return inverse*vectors*mp.diag([max(v, 0) for v in values])*vectors.T*inverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    assert not args.output.exists()
    os.environ.update(XHDFE_GPU_BACKEND=args.backend, XHDFE_MOBILITY_MODE="off",
                      XHDFE_FE_STRUCTURE_MODE="off", XHDFE_ABSORPTION_CACHE_MODE="off")
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", args.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    mp.mp.dps = 80
    rows = []
    for seed in (0, 1, 2, 3, 7, 11):
        y, x, fe, clusters, frequencies = fixture(seed)
        for frequency in (False, True):
            weights = frequencies if frequency else np.ones(y.size)
            b0, v0, magnitudes, scales = reference_base(y, x, fe, clusters, weights)
            slopes = clamp(v0[:3, :3], scales[:3])
            for offset in (0., 2.**30):
                transform = mp.eye(4)
                transform[3, 0] = -mp.mpf(offset)
                b = transform*b0
                v = clamp(transform*v0*transform.T, scales)
                for j in range(3):
                    for k in range(3):
                        v[j, k] = slopes[j, k]
                xx = x.copy()
                xx[:, 0] += offset
                model = core.HdfeRegressor(num_threads=1, se_type="cluster",
                    fit_intercept=True, ssc_g_adj=False, ssc_k_adj=False)
                model.fit(y, xx, [fe], clusters=clusters,
                          weights=weights if frequency else None, fweights=frequency)
                assert bool(model.gpu_used_) == (args.backend == "cuda")
                be = max(abs(mp.mpf(float(model.coef_[j]))-b[j])/max(1, abs(b[j])) for j in range(4))
                errors = []
                for j in range(4):
                    for k in range(4):
                        difference = abs(mp.mpf(float(model.covariance_[j, k]))-v[j, k])
                        scale = mp.sqrt(max(v[j, j], 0)*max(v[k, k], 0))
                        errors.append(difference/scale if scale else (mp.inf if difference else mp.mpf(0)))
                error = max(errors)
                absolute_transform = mp.matrix([[abs(transform[j, k]) for k in range(4)] for j in range(4)])
                transported_magnitudes = absolute_transform*magnitudes*absolute_transform.T
                rows.append(dict(seed=seed, frequency=frequency, offset=offset,
                    b_error=float(be), V_error=float(error) if mp.isfinite(error) else None,
                    V_candidate=np.asarray(model.covariance_).tolist(),
                    V_reference=[[mp.nstr(v[j, k], 50) for k in range(4)] for j in range(4)],
                    operation_magnitudes=[[mp.nstr(transported_magnitudes[j, k], 30) for k in range(4)] for j in range(4)],
                    status="PASS_RAW_CONTRACT" if be <= mp.mpf('1e-9') and error <= mp.mpf('1e-8') else "REVIEW_ARITHMETIC"))
    args.output.write_text(json.dumps(rows, indent=2)+"\n")
    assert all(row["status"] == "PASS_RAW_CONTRACT" for row in rows), rows
    print({status:sum(row["status"] == status for row in rows)
           for status in ("PASS_RAW_CONTRACT", "REVIEW_ARITHMETIC")})


if __name__ == "__main__":
    main()
