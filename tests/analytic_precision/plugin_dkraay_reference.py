"""Check the native release gate's independent known-score IV/DK reference.

The represented inputs are dyadic Walsh vectors. Their exact projection is
Q = [9*a/5 + 3*c/5, 9*a/5 + 3*c/5 + 2^-27*b, 1]. The structural residual is
the orthogonal Walsh vector e. The common regressor means (1,1) recover the
constant via A. No xhdfe/reghdfe output enters this calculation.
"""
from pathlib import Path
import re

import mpmath as mp


def reference(digits):
    with mp.workdps(digits):
        q, residual, time = [], [], []
        for i in range(512):
            a, b, c, v, d, e = [2*((i//(2**j)) % 2)-1 for j in range(6)]
            score = mp.mpf(9*a+3*c)/5
            q.append([score, score + mp.mpf(2)**-27*b, 1])
            residual.append(e)
            time.append((73*i + 19*(i//7) + 5*(i//23)) % 17)
        Q = mp.matrix(q)
        G = (Q.T*Q)**-1
        H = mp.matrix([[mp.fsum(q[i][j]*residual[i] for i in range(512) if time[i] == t)
                        for j in range(3)] for t in range(17)])
        meat = H.T*H
        for lag in (1, 2):
            cross = H[lag:, :].T*H[:-lag, :]
            meat += (1-mp.mpf(lag)/3)*(cross+cross.T)
        A = mp.matrix([[1, 0, 0], [0, 1, 0], [-1, -1, 1]])
        covariance = A*G*meat*G.T*A.T*(mp.mpf(511)/509)*(mp.mpf(17)/16)
        mp.cholesky(covariance)
        return [float(covariance[j, k]) for j in range(3) for k in range(3)]


if __name__ == '__main__':
    source = Path(__file__).resolve().parents[2]/'tools/plugin_openmp_probe.cpp'
    block = re.search(r'constexpr double reference\[9\] = \{([^}]+)\}', source.read_text())
    recorded = [float(value) for value in block[1].split(',')]
    assert recorded == reference(90) == reference(120)
    print('PLUGIN_DKRAAY_MP90_MP120_REFERENCE_PASS')
