"""MP80/MP120 explicit reduced-dummy oracle, using represented binary64 data."""
import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
from scipy.linalg import qr

root = Path(__file__).with_name('fixtures')
source = root/'cg_seed1_2017.csv'
frozen = json.loads((root/'cg_seed1_2017_oracle.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest() == frozen['csv_sha256']
frame = pd.read_csv(source, float_precision='round_trip').sort_values(['sid', 'year'])
sid, year, age = [pd.factorize(frame[c], sort=True)[0]
                  for c in ('sid', 'year', 'entry_age')]
clusters = pd.factorize(frame.did, sort=True)[0]
groups = [np.flatnonzero(sid == g).tolist() for g in range(sid.max()+1)]
raw_dummy = np.column_stack((np.eye(year.max()+1)[year], np.eye(age.max()+1)[age]))
reduced_dummy = raw_dummy.copy()
for indices in groups:
    reduced_dummy[indices] -= raw_dummy[indices].mean(axis=0)
_, R, pivots = qr(reduced_dummy, mode='economic', pivoting=True)
rank = np.count_nonzero(abs(R.diagonal()) > abs(R[0, 0])*1e-12)
chosen = pivots[:rank].tolist()


def oracle(digits):
    mp.mp.dps = digits
    n = len(frame)
    D = [[mp.mpf(int(raw_dummy[i, j])) for j in chosen] for i in range(n)]
    values = [[mp.mpf(float(frame[c].iloc[i])) for c in ('dagg', 'x', 'expo', 'z')]
              for i in range(n)]
    means = [mp.fsum(row[j] for row in values)/n for j in range(4)]
    for indices in groups:
        dm = [mp.fsum(D[i][j] for i in indices)/len(indices) for j in range(rank)]
        vm = [mp.fsum(values[i][j] for i in indices)/len(indices) for j in range(4)]
        for i in indices:
            D[i] = [D[i][j]-dm[j] for j in range(rank)]
            values[i] = [values[i][j]-vm[j] for j in range(4)]
    gram = mp.matrix([[mp.fsum(row[j]*row[k] for row in D)
                       for k in range(rank)] for j in range(rank)])
    assert mp.det(gram) != 0
    for j in range(4):
        coefficients = mp.lu_solve(gram, mp.matrix([
            mp.fsum(D[i][k]*values[i][j] for i in range(n)) for k in range(rank)]))
        for i in range(n):
            values[i][j] -= mp.fsum(D[i][k]*coefficients[k] for k in range(rank))
    results = {}
    for name, xcol, zcol in [('ols', 1, 1), ('iv', 2, 3)]:
        cross = mp.fsum(v[zcol]*v[xcol] for v in values)
        beta = mp.fsum(v[zcol]*v[0] for v in values)/cross
        cons = means[0]-means[xcol]*beta
        scores = [[mp.mpf(0), mp.mpf(0)] for _ in range(clusters.max()+1)]
        for i, v in enumerate(values):
            residual = v[0]-v[xcol]*beta
            slope_influence = v[zcol]/cross
            constant_influence = mp.mpf(1)/n-means[xcol]*slope_influence
            scores[clusters[i]][0] += residual*slope_influence
            scores[clusters[i]][1] += residual*constant_influence
        V = [[mp.fsum(s[j]*s[k] for s in scores) for k in range(2)] for j in range(2)]
        results[name] = {'b': [mp.nstr(beta, digits-5), mp.nstr(cons, digits-5)],
                         'V': [[mp.nstr(v, digits-5) for v in row] for row in V]}
    return results


first = oracle(80)
second = oracle(120)
for kind in first:
    a = first[kind]['b'] + sum(first[kind]['V'], [])
    b = second[kind]['b'] + sum(second[kind]['V'], [])
    assert max(abs(mp.mpf(x)-mp.mpf(y)) for x, y in zip(a, b)) < mp.mpf('1e-65')
for kind in second:
    actual = second[kind]['b'] + sum(second[kind]['V'], [])
    expected = frozen['references'][kind]['b'] + sum(frozen['references'][kind]['V'], [])
    assert max(abs(mp.mpf(a)-mp.mpf(b)) for a, b in zip(actual, expected)) < mp.mpf('1e-65')
print('CG_FINAL_STEP_MP80_MP120_ORACLE_PASS')
