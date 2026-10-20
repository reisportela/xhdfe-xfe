"""Regenerate aggregate-only github FWL references from a supplied dataset.

No xhdfe/reghdfe solver is used; no individual records are written.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr, lsqr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    columns = ['y', 'x1', 'x2', 'id1', 'id2', 'id3']
    frame = pd.read_parquet(args.data, columns=columns)
    signature = hashlib.sha256()
    for name in columns:
        a = np.asarray(frame[name], dtype='<f8' if name in columns[:3] else '<i8')
        signature.update(name.encode())
        signature.update(np.asarray(a.shape, dtype='<i8').tobytes())
        signature.update(a.tobytes())
    raw = frame[columns[:3]].to_numpy(dtype=float)
    codes = [pd.factorize(frame[name], sort=True)[0] for name in columns[3:]]
    keep = np.isfinite(raw).all(axis=1) & np.logical_and.reduce([f >= 0 for f in codes])
    while True:
        previous = keep.copy()
        for f in codes:
            count = np.bincount(f[keep], minlength=int(f.max())+1)
            keep &= count[np.maximum(f, 0)] > 1
        if np.array_equal(previous, keep):
            break
    raw = raw[keep]
    fes = [np.unique(f[keep], return_inverse=True)[1] for f in codes]
    n = len(raw)
    offsets = np.cumsum([0]+[int(f.max())+1 for f in fes])
    D = sp.coo_matrix((np.ones(3*n), (np.tile(np.arange(n), 3),
        np.concatenate([f+offsets[k] for k, f in enumerate(fes)]))),
        shape=(n, int(offsets[-1]))).tocsr()
    scale = 1/np.sqrt(np.asarray(D.sum(axis=0)).ravel())
    A = (D@sp.diags(scale)).tocsr()
    DL = D.astype(np.longdouble)
    references = []
    for method, tolerance in [('lsmr', 1e-13), ('lsqr', 1e-14)]:
        projected, diagnostics = [], []
        for rhs in range(3):
            if method == 'lsmr':
                solved = lsmr(A, raw[:, rhs], atol=tolerance, btol=tolerance,
                              conlim=0, maxiter=20000)
            else:
                solved = lsqr(A, raw[:, rhs], atol=tolerance, btol=tolerance,
                              conlim=0, iter_lim=20000)
            assert solved[1] in (1, 2, 4, 5), (method, rhs, solved[1:3])
            q = raw[:, rhs].astype(np.longdouble)-DL@(solved[0]*scale).astype(np.longdouble)
            projected.append(q)
            diagnostics.append(dict(rhs=rhs, stop=int(solved[1]), iterations=int(solved[2]),
                                    max_fe_moment=float(np.max(abs(DL.T@q)))))
        Q = np.column_stack(projected)
        y, X = Q[:, 0], Q[:, 1:]
        gram, rhs = X.T@X, X.T@y
        with mp.workdps(80):
            G = mp.matrix([[mp.mpf(str(v)) for v in row] for row in gram])
            inverse = G**-1
            beta = inverse*mp.matrix([mp.mpf(str(v)) for v in rhs])
            b = np.array([np.longdouble(str(v)) for v in beta])
            bread = np.array([[np.longdouble(str(inverse[i, j])) for j in range(2)]
                              for i in range(2)])
        residual = y-X@b
        means = raw[:, 1:].astype(np.longdouble).mean(axis=0)
        constant = raw[:, 0].astype(np.longdouble).mean()-means@b
        slopes = X@bread
        influence = np.column_stack((slopes, 1/np.longdouble(n)-slopes@means))
        scores = np.zeros((int(fes[0].max())+1, 3), dtype=np.longdouble)
        np.add.at(scores, fes[0], residual[:, None]*influence)
        references.append(dict(method=method, tolerance=tolerance, diagnostics=diagnostics,
            b=np.r_[b, constant].astype(float).tolist(),
            V_no_ssc=(scores.T@scores).astype(float).tolist(),
            n=n, clusters=len(scores), gram_condition=float(np.linalg.cond(np.asarray(gram, float)))))
    a, b = references
    ab, bb = np.asarray(a['b']), np.asarray(b['b'])
    av, bv = np.asarray(a['V_no_ssc']), np.asarray(b['V_no_ssc'])
    db = float(np.max(abs(ab-bb)/np.maximum(1, abs(bb))))
    dv = float(np.max(abs(av-bv)/np.sqrt(np.outer(bv.diagonal(), bv.diagonal()))))
    assert db <= 1e-11 and dv <= 1e-9, 'Independent reference unresolved'
    result = dict(schema=1, columns=columns, canonical_columns_sha256=signature.hexdigest(),
        n_original=len(frame), n_estimation=n, clusters=b['clusters'],
        b=b['b'], V_no_ssc=b['V_no_ssc'], references=references,
        agreement=dict(b_scaled=db, V_scaled=dv), individual_data_included=False,
        reference='Explicit sparse-dummy FWL, LSQR/LSMR, MP80 reduced OLS and long-double cluster sandwich.',
        interpretation='Fast covariance differences are diagnostics; no global Fast forward-error limit is adopted.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='REFERENCE_AGREEMENT_CONFIRMED', b_scaled=db, V_scaled=dv)))


if __name__ == '__main__':
    main()
