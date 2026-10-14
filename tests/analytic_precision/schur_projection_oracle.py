"""Offline sparse projection with exact structural gauges and affine checks.

No observations, FE coefficients, or identifiers are persisted by this module.
"""
from fractions import Fraction
import hashlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def independent_rows(matrix, reverse=False):
    """Choose coordinate constraints by rational elimination of integer rows."""
    basis = []
    chosen = []
    seen = set()
    order = range(len(matrix)-1, -1, -1) if reverse else range(len(matrix))
    for i in order:
        pattern = tuple(int(v) for v in matrix[i])
        if pattern in seen:
            continue
        seen.add(pattern)
        row = [Fraction(v) for v in pattern]
        for pivot, old in basis:
            multiple = row[pivot]
            row = [a-multiple*b for a, b in zip(row, old)]
        pivot = next((j for j, v in enumerate(row) if v), None)
        if pivot is not None:
            multiple = row[pivot]
            basis.append((pivot, [v/multiple for v in row]))
            chosen.append(i)
    return np.array(chosen, dtype=np.int32)


class SchurProjection:
    def __init__(self, fes, counts, components):
        self.fes = fes
        self.counts = counts
        self.levels = [len(c) for c in counts]
        worker, firm, year = fes
        nw, nf, nt = self.levels
        n = len(worker)
        k01, c01 = components(worker, firm, nw, nf)
        k02, c02 = components(worker, year, nw, nt)
        k12, c12 = components(firm, year, nf, nt)
        assert k01+k02+k12 <= 64, 'This offline oracle expects few structural components'
        nulls = np.zeros((nf+nt, k01+k02+k12), dtype=np.int8)
        nulls[np.arange(nf), c01[nw:]] = 1
        nulls[nf+np.arange(nt), k01+c02[nw:]] = 1
        nulls[np.arange(nf), k01+k02+c12[:nf]] = 1
        nulls[nf+np.arange(nt), k01+k02+c12[nf:]] = -1
        # Every proposed null direction must be exactly constant within worker.
        for j in range(nulls.shape[1]):
            values = nulls[firm, j]+nulls[nf+year, j]
            lo = np.full(nw, 3, np.int8)
            hi = np.full(nw, -3, np.int8)
            np.minimum.at(lo, worker, values)
            np.maximum.at(hi, worker, values)
            assert np.array_equal(lo, hi), 'Invalid structural null direction'
        first = independent_rows(nulls)
        second = independent_rows(nulls, reverse=True)
        assert len(first) == len(second) and not np.array_equal(first, second)
        _, old_firm_anchors = np.unique(c01[nw:], return_index=True)
        old_anchors = np.r_[old_firm_anchors, nf]
        old_constraint_rank = len(independent_rows(nulls[old_anchors]))
        raw_other = sparse.hstack([
            sparse.csr_matrix((np.ones(n), (np.arange(n), f)), shape=(n, g))
            for f, g in zip(fes[1:], self.levels[1:])], format='csr')
        raw_worker = sparse.csr_matrix((np.ones(n), (np.arange(n), worker)), shape=(n, nw))
        cross = raw_worker.T@raw_other
        schur = (raw_other.T@raw_other-cross.T@cross.multiply((1./counts[0])[:, None])).tocsc()
        self.factors = []
        factor_records = []
        for anchors in (first, second):
            selected = np.ones(nf+nt, bool)
            selected[anchors] = False
            factor = splu(schur[selected][:, selected].tocsc())
            self.factors.append((selected, factor, raw_other[:, selected].tocsr()))
            diagonal = np.abs(factor.U.diagonal())
            factor_records.append(dict(columns=int(selected.sum()),
                min_abs_pivot=float(diagonal.min()), max_abs_pivot=float(diagonal.max()),
                factor_entries=factor.L.nnz+factor.U.nnz,
                anchor_sha256=hashlib.sha256(anchors.tobytes()).hexdigest()))
        self.metadata = dict(pair_components=[k01, k02, k12],
            known_null_rank=len(first), old_anchor_constraint_rank=old_constraint_rank,
            old_unconstrained_exact_nulls=len(first)-old_constraint_rank,
            factors=factor_records, null_validation='integer within-worker equality')

    def _project(self, v, gauge):
        worker, firm, year = self.fes
        nw, nf, nt = self.levels
        selected, factor, other = self.factors[gauge]
        out = v.copy()
        worker_coef = np.zeros(nw, np.longdouble)
        other_coef = np.zeros(nf+nt, np.longdouble)
        operation_magnitudes = np.zeros(len(v))
        for _ in range(2):
            mean = np.bincount(worker, weights=out, minlength=nw)/self.counts[0]
            within = out-mean[worker]
            coef = factor.solve(other.T@within)
            fit = other@coef
            fit_mean = np.bincount(worker, weights=fit, minlength=nw)/self.counts[0]
            expanded_coef = np.zeros(nf+nt)
            expanded_coef[selected] = coef
            operation_magnitudes += (np.abs(out)+np.abs(mean[worker])+np.abs(fit)+
                np.abs(fit_mean[worker])+np.abs(expanded_coef[firm])+
                np.abs(expanded_coef[nf+year]))
            out = within-(fit-fit_mean[worker])
            worker_coef += mean.astype(np.longdouble)-fit_mean.astype(np.longdouble)
            other_coef[selected] += coef.astype(np.longdouble)
        # Two passes use fewer than 32 FP64 additions/subtractions per row
        # in forming residuals and accumulating/reconstructing coefficients.
        # Check both this operation-derived bound and an absolute norm bound;
        # huge null-space coefficients must not legitimise a poor projection.
        unit = np.finfo(float).eps/2
        gamma = 32*unit/(1-32*unit)
        error2 = np.longdouble(0)
        allowed2 = np.longdouble(0)
        max_ratio = 0.
        for start in range(0, len(v), 131072):
            sl = slice(start, start+131072)
            terms = [v[sl], -out[sl], -worker_coef[worker[sl]],
                     -other_coef[firm[sl]], -other_coef[nf+year[sl]]]
            error = sum(np.asarray(t, np.longdouble) for t in terms)
            scale = operation_magnitudes[sl].astype(np.longdouble)
            scale += sum(np.abs(np.asarray(t, np.longdouble)) for t in terms)
            allowance = gamma*scale
            max_ratio = max(max_ratio, float(np.max(np.abs(error)/np.maximum(allowance, np.finfo(float).tiny))))
            error2 += np.dot(error, error)
            allowed2 += np.dot(allowance, allowance)
        relative = float(np.sqrt(error2)/max(np.linalg.norm(v), np.finfo(float).tiny))
        return out, dict(gauge=gauge, affine_relative_error=relative,
            affine_max_bound_ratio=max_ratio, affine_bound_norm=float(np.sqrt(allowed2)),
            affine_pass=max_ratio<=1 and relative<=1e-11)

    def project(self, v):
        first, a = self._project(v, 0)
        second, b = self._project(v, 1)
        difference = float(np.linalg.norm(first-second)/max(np.linalg.norm(v), np.finfo(float).tiny))
        record = dict(affine=[a, b], gauge_relative_difference=difference,
                      pass_oracle=a['affine_pass'] and b['affine_pass'] and difference<=1e-11)
        assert record['pass_oracle'], record
        return first, record
