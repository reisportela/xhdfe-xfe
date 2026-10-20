"""Independent MP80/MP120 OLS/DK oracle for the upstream public-auto scenario.

Input is the public auto fixture with price/weight/length, turn/t and selected.
The design uses two categorical fixed effects. Lags index observed periods,
as documented by xhdfe/reghdfe; ivreghdfe is a separately graded comparator.
"""
import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
from scipy.linalg import qr


def reference(frame, digits):
    y = frame.price.to_numpy(float)
    X = frame[['weight', 'length']].to_numpy(float)
    g = pd.factorize(frame.turn, sort=True)[0]
    t = pd.factorize(frame.t, sort=True)[0]
    n, periods = len(frame), int(t.max()+1)
    dummy = np.column_stack((np.eye(g.max()+1)[g], np.eye(periods)[t]))
    _, _, pivots = qr(dummy, mode='economic', pivoting=True)
    rank = int(np.linalg.matrix_rank(dummy))
    dummy = dummy[:, pivots[:rank]]
    with mp.workdps(digits):
        D, x, y = mp.matrix(dummy.tolist()), mp.matrix(X.tolist()), mp.matrix(y.tolist())
        gram = D.T*D
        assert mp.det(gram) != 0
        projection = D*(gram**-1)*D.T
        xt, yt = x-projection*x, y-projection*y
        bread = (xt.T*xt)**-1
        beta = bread*xt.T*yt
        residual = yt-xt*beta
        means = [mp.fsum(x[:, j])/n for j in range(2)]
        b = [beta[0], beta[1], mp.fsum(y)/n-sum(means[j]*beta[j] for j in range(2))]
        scores = mp.matrix(periods, 3)
        for i in range(n):
            slopes = bread*xt[i, :].T
            influence = [slopes[0], slopes[1],
                         mp.mpf(1)/n-sum(means[j]*slopes[j] for j in range(2))]
            for j in range(3):
                scores[int(t[i]), j] += influence[j]*residual[i]
        cross = scores[1:, :].T*scores[:-1, :]
        meat = scores.T*scores+(cross+cross.T)/2
        # All time effects are nested in the DK clustering dimension.
        # Dummy rank, rather than candidate-reported DoF, determines this factor.
        non_nested = rank-periods
        factor = mp.mpf(n-1)/(n-2-non_nested)*mp.mpf(periods)/(periods-1)
        V = factor*meat
        return dict(N=n, periods=periods, dummy_rank=rank, non_nested_FE_dof=non_nested,
            b=[mp.nstr(v, digits-5) for v in b],
            V=[[mp.nstr(V[i, j], digits-5) for j in range(3)] for i in range(3)],
            scale=mp.nstr(factor, digits-5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = pd.read_stata(args.fixture)
    data = data[data.selected == 1].copy()
    low, high = reference(data, 80), reference(data, 120)
    with mp.workdps(130):
        flat = lambda r: r['b']+sum(r['V'], [])
        difference = max(abs(mp.mpf(a)-mp.mpf(b)) for a, b in zip(flat(low), flat(high)))
        assert difference < mp.mpf('1e-65')
        difference_text = str(difference)
    b, V = np.array(high['b'], float), np.array(high['V'], float)
    scale = np.sqrt(np.outer(V.diagonal(), V.diagonal()))
    actual = json.loads(args.actual.read_text())
    metrics = {}
    for name, bk, vk in [('xhdfe', 'cb', 'cV'), ('ivreghdfe', 'rb', 'rV')]:
        values, covariance = np.array(actual[bk]).ravel(), np.array(actual[vk])
        size = len(values)
        metrics[name] = dict(
            b_scaled_error=float(np.max(abs(values-b[:size])/np.maximum(1, abs(b[:size])))),
            V_scaled_error=float(np.max(abs(covariance-V[:size, :size])/scale[:size, :size])))
    result = dict(oracle=high, cross_precision_error=difference_text, metrics=metrics,
                  source='Public Stata auto data; captured declared estimation sample',
                  convention='Observed-period Bartlett bandwidth2; independent dummy-rank SSC')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(N=high['N'], periods=high['periods'], metrics=metrics)))
    assert metrics['xhdfe']['b_scaled_error'] <= 1e-9
    assert metrics['xhdfe']['V_scaled_error'] <= 1e-8


if __name__ == '__main__':
    main()
