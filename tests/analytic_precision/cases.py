"""Deterministic finite-sample OLS oracles, independent of either HDFE package."""
from dataclasses import dataclass, field
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import svd

from precision_contract import (
    covariance_operation_count,
    oracle_covariance_arithmetic_margin,
)

BETA = np.array([-.018, .014, -.0002])


def indicators(ids):
    _, inverse = np.unique(ids, return_inverse=True)
    return np.eye(inverse.max()+1)[inverse]


@dataclass
class Case:
    name: str
    frame: pd.DataFrame
    group_frame: pd.DataFrame
    D: np.ndarray
    fes: list
    absorb: str
    kind: str = 'standard'
    aggregation: str = 'sum'
    weight: str = ''
    vce: str = 'unadjusted'
    clusters: list = field(default_factory=list)
    slopes: list = field(default_factory=list)
    intercept: bool = True
    stress: bool = False
    dropped: int = 0

    def oracle(self):
        f = self.group_frame
        selected = f.expected_sample.to_numpy(bool)
        X = f[['x1', 'x2', 'x3']].to_numpy()[selected]
        y = f.y.to_numpy()[selected]
        e = f.exact_residual.to_numpy()[selected]
        w = f.weight.to_numpy()[selected] if self.weight else np.ones(len(y))
        D = self.D[selected]
        rootw = np.sqrt(w)
        if D.shape[1]:
            U, s, _ = svd(rootw[:, None]*D, full_matrices=False)
            rank_d = int(np.count_nonzero(s > s[0]*1e-12))
            Q = U[:, :rank_d]
            Xw = rootw[:, None]*X
            Xr_w = Xw-Q@(Q.T@Xw)
            yr_w = rootw*y-Q@(Q.T@(rootw*y))
        else:
            rank_d = 0
            Xr_w, yr_w = rootw[:, None]*X, rootw*y
        fit, _, rank_x, sx = np.linalg.lstsq(Xr_w, yr_w, rcond=1e-12)
        C = np.array([[1., 0., 2.], [0., 1., 0.]]) if self.name == 'collinear' else np.eye(3)
        assert np.max(np.abs(C@(fit-BETA))) < 1e-8, (self.name, fit)
        assert np.max(np.abs(X.T@(w*e))) < 1e-8
        if D.shape[1]:
            assert np.max(np.abs(D.T@(w*e))) < 1e-8
        residual = (yr_w-Xr_w@fit)/rootw
        assert np.max(np.abs(residual-e)) < 1e-7, self.name
        Hinv = np.linalg.pinv(Xr_w.T@Xr_w, rcond=1e-12)
        n_effective = float(w.sum()) if self.weight in ('fw','iw') else float(len(y))
        k_exact = rank_d+int(rank_x)
        k_contract = k_exact
        # Documented conservative individual-FE DoF: levels of both effects.
        # This is fixed from the fixture, never inferred from candidate e(V).
        if self.kind == 'group_individual':
            k_contract = (self.frame.individual.nunique()+f.own.nunique()+int(rank_x))
            team_sizes = self.frame.groupby('group').individual.nunique().to_numpy()
            if self.aggregation == 'mean' or np.all(team_sizes == team_sizes[0]):
                # Membership already spans the intercept supplied by own.
                k_contract -= 1
        raw_rss = float(np.dot(w, e*e))
        rss_scale = len(y)/float(w.sum()) if self.weight in ('aw', 'pw') else 1.
        Xr = Xr_w/rootw[:, None]
        if self.vce == 'unadjusted':
            V_exact = Hinv*(raw_rss/(n_effective-k_exact))
            V_contract = Hinv*(raw_rss/(n_effective-k_contract))
            covariance_scale = np.abs(Hinv)*(abs(raw_rss)/(n_effective-k_contract))
            cluster_terms = 0
        elif self.vce == 'robust':
            squared_weight = w if self.weight == 'fw' else w*w
            meat = Xr.T@((squared_weight*e*e)[:, None]*Xr)
            base = Hinv@meat@Hinv
            V_exact = base*n_effective/(n_effective-k_exact)
            V_contract = base*n_effective/(n_effective-k_contract)
            meat_scale = np.abs(Xr).T@((squared_weight*e*e)[:, None]*np.abs(Xr))
            covariance_scale = (np.abs(Hinv)@meat_scale@np.abs(Hinv) *
                                n_effective/(n_effective-k_contract))
            cluster_terms = 0
        else:
            labels = [f[c].to_numpy()[selected] for c in self.clusters]
            meat = np.zeros((3, 3))
            meat_scale = np.zeros((3, 3))
            cluster_terms = 0
            for q in range(1, len(labels)+1):
                for subset in combinations(labels, q):
                    _, ids = np.unique(np.column_stack(subset), axis=0, return_inverse=True)
                    score = np.zeros((ids.max()+1, 3))
                    np.add.at(score, ids, (w*e)[:, None]*Xr)
                    meat += (-1)**(q+1)*(score.T@score)
                    meat_scale += np.abs(score).T@np.abs(score)
                    cluster_terms += 1
            # Default documented reghdfe/xhdfe SSC: one common min(G)/(min(G)-1).
            minimum_clusters = min(len(np.unique(c)) for c in labels)
            meat *= minimum_clusters/(minimum_clusters-1)
            base = Hinv@meat@Hinv
            V_exact = base*(n_effective-1)/(n_effective-k_exact)
            V_contract = base*(n_effective-1)/(n_effective-k_contract)
            covariance_scale = (np.abs(Hinv)@meat_scale@np.abs(Hinv) *
                                minimum_clusters/(minimum_clusters-1) *
                                (n_effective-1)/(n_effective-k_contract))
        # Stata regress multiway VCE may repair the full dummy covariance to PSD;
        # that is not invariant to FE normalization. Its role here is explicit
        # OLS with conventional covariance; multiway inference has its own oracle.
        V_ols = Hinv*(raw_rss/(n_effective-k_exact)) if len(self.clusters)>1 else V_exact
        condition_x = float(sx[0]/sx[int(rank_x)-1])
        arithmetic_operations = covariance_operation_count(
            len(y), X.shape[1], self.vce, cluster_terms)
        arithmetic_margin, arithmetic_model = oracle_covariance_arithmetic_margin(
            covariance_scale, arithmetic_operations, condition_x*condition_x)
        arithmetic_model['scale_source'] = 'absolute products in frozen oracle graph'
        within_gram = Xr_w.T@Xr_w
        within_outcome_norm = float(np.linalg.norm(yr_w))
        return dict(beta=BETA.tolist(), beta_svd=fit.tolist(), contrast=C.tolist(),
                    V_exact=V_exact.tolist(), V_contract=V_contract.tolist(),
                    V_ols=V_ols.tolist(), cluster_ssc='common_min_G',
                    rank_D=rank_d, rank_X=int(rank_x), k_exact=k_exact,
                    k_contract=k_contract, N=n_effective, groups=len(y),
                    dropped=self.dropped, rss=raw_rss*rss_scale,
                    rss_unscaled=raw_rss, residual_max=float(np.max(np.abs(e))),
                    condition_X=condition_x,
                    within_gram=within_gram.tolist(),
                    within_outcome_norm=within_outcome_norm,
                    V_contract_arithmetic_scale=covariance_scale.tolist(),
                    V_contract_arithmetic_margin=(arithmetic_margin.tolist()
                                                  if arithmetic_margin is not None else None),
                    V_arithmetic_model=arithmetic_model,
                    normal_X=float(np.max(np.abs(X.T@(w*e)))),
                    normal_FE=float(np.max(np.abs(D.T@(w*e)))) if D.shape[1] else 0.)

    def write(self, directory):
        directory.mkdir(parents=True, exist_ok=False)
        self.frame.to_stata(directory/'long.dta', write_index=False, version=118)
        groups = pd.concat([self.group_frame.reset_index(drop=True),
                            pd.DataFrame(self.D, columns=[f'd{j+1}' for j in range(self.D.shape[1])])], axis=1)
        groups.to_stata(directory/'explicit.dta', write_index=False, version=118)
        oracle = self.oracle()
        metadata = {k:getattr(self, k) for k in ('name', 'fes', 'absorb', 'kind',
                    'aggregation', 'weight', 'vce', 'clusters', 'slopes', 'intercept', 'stress')}
        (directory/'case.json').write_text(json.dumps(metadata, indent=2)+'\n')
        (directory/'oracle.json').write_text(json.dumps(oracle, indent=2, allow_nan=False)+'\n')
        np.savez(directory/'design.npz', D=self.D)
        return metadata, oracle


def ordinary(name):
    cells, R = 36, 16
    cell = np.repeat(np.arange(cells), R)
    rep = np.tile(np.arange(R), cells)
    a, b, c, d = [2*((rep >> j) & 1)-1 for j in range(4)]
    g1, g2, g3 = cell % 6, cell//6, (cell % 6+2*(cell//6)) % 5
    if name == 'disconnected':
        g2 = cell//6 + 6*(g1//3)
    f = pd.DataFrame(dict(group=np.arange(len(cell)), g1=g1, g2=g2, g3=g3,
                          z=c+.1*g1, weight=1.+cell % 3, expected_sample=1))
    f['x1'] = .2*a + .1*np.sin(cell)
    f['x2'] = 5 + g1*.2+g2*.1+.3*a+.8*b
    f['x3'] = f.x2**2
    f['exact_residual'] = .113*d
    f['cl1'] = np.random.default_rng(651).integers(0, 23, len(f))
    f['cl2'] = np.random.default_rng(652).integers(0, 19, len(f))
    fes = [] if name.startswith('ols') else ['g1'] if name == 'one_fe' else ['g1', 'g2']
    if name == 'three_fe':
        fes.append('g3')
    intercept = name != 'ols_no_constant'
    columns = [np.ones((len(f), 1))] if intercept else []
    slopes = []
    for fe in fes:
        if not (name == 'slope_only' and fe == 'g1'):
            columns.append(indicators(f[fe].to_numpy()))
    absorb = ' '.join(fes)
    if name in ('heterogeneous', 'slope_only'):
        columns.append(indicators(g1)*f.z.to_numpy()[:, None])
        slopes = [[0, 'z', name == 'heterogeneous']]
        absorb = 'g1##c.z g2' if name == 'heterogeneous' else 'g1#c.z g2'
    D = np.column_stack(columns) if columns else np.empty((len(f), 0))
    theta = .2*np.sin(np.arange(D.shape[1]))
    if intercept:
        theta[0] = 1.5
    if name == 'collinear':
        f['x3'] = 2*f.x1
    f['y'] = f[['x1', 'x2', 'x3']].to_numpy()@BETA + D@theta + f.exact_residual
    weight = {'aweights':'aw', 'fweights':'fw', 'pweights':'pw', 'iweights':'iw'}.get(name, '')
    vce = 'robust' if name in ('robust', 'pweights') else 'cluster' if 'cluster' in name else 'unadjusted'
    clusters = ['cl1', 'cl2'] if name == 'multi_cluster' else ['cl1'] if name == 'cluster' else []
    dropped = 0
    if name == 'singletons':
        extra = f.iloc[:2].copy()
        extra['group'] = [len(f), len(f)+1]
        extra['g1'] = [100, 101]
        extra['g2'] = [100, 101]
        extra['expected_sample'] = 0
        extra['exact_residual'] = 0.
        extra['y'] = extra[['x1','x2','x3']].to_numpy()@BETA
        f = pd.concat([f, extra], ignore_index=True)
        D = np.pad(D, ((0, 2), (0, 2)))
        D[-2:, -2:] = np.eye(2)
        dropped = 2
    frame = f.copy()
    kind = 'standard'
    if name == 'group_only':
        frame = f.loc[f.index.repeat(3)].reset_index(drop=True)
        kind = 'group_only'
    return Case(name, frame, f, D, fes, absorb, kind=kind, weight=weight,
                vce=vce, clusters=clusters, slopes=slopes, intercept=intercept, dropped=dropped)


def grouped(name):
    # The quick grid uses eight nodes; the independent difficult case keeps 128.
    n = 128 if name == 'laplacian_sum' else 8
    R = 8
    g = np.arange(4*n*R)
    node, kind, rep = g//(4*R), (g//R) % 4, g % R
    own = node+n*(kind >= 2)
    agg = 'mean' if 'mean' in name else 'sum'
    uniform = 'uniform' in name
    sizes = np.where(kind < 2, 7, 7 if uniform else 6)
    f = pd.DataFrame(dict(group=g, own=own, weight=1.+node % 3, expected_sample=1))
    wave = np.cos(2*np.pi*node/n)
    slow = np.where(kind == 0, -wave, np.where(kind == 1, wave, 0.))
    a, b, c = [2*((rep >> j) & 1)-1 for j in range(3)]
    f['x1'] = .4*slow+.2*a
    f['x2'] = 5+.5*slow+own % 5+.3*a+.8*b
    f['x3'] = f.x2**2
    f['exact_residual'] = .113*c
    edges = []
    for group, i, k in zip(g, node, kind):
        peers = [[(i-1) % n, (i+1) % n], [i, n+i], [i], [n+i]][k]
        peers = peers+list(range(2*n, 2*n+5))
        if uniform and k >= 2:
            peers.append(2*n+5)
        edges.extend((group, p) for p in peers)
    links = pd.DataFrame(edges, columns=['group', 'individual'])
    A = np.zeros((len(g), int(links.individual.max())+1))
    A[links.group, links.individual] = 1.
    lam = 4*np.sin(np.pi/n)**2
    theta_ind = np.zeros(A.shape[1])
    theta_ind[:2*n] = .7*np.cos(2*np.pi*(np.arange(2*n) % n)/n)/lam
    theta_own = np.r_[.35*(1-4/lam)*np.cos(2*np.pi*np.arange(n)/n),
                      -.7/lam*np.cos(2*np.pi*np.arange(n)/n)]
    if agg == 'mean':
        A = A/sizes[:, None]
        theta_own[:n] /= 7
        theta_own[n:] /= 7 if uniform else 6
    D = np.c_[indicators(own), A]
    fe = indicators(own)@theta_own+A@theta_ind
    f['y'] = 1.5+f[['x1','x2','x3']].to_numpy()@BETA+fe+f.exact_residual
    frame = links.merge(f, on='group', validate='many_to_one', sort=False)
    weight = 'fw' if 'fweights' in name else 'aw' if 'aweights' in name else ''
    return Case(name, frame, f, D, ['individual', 'own'], 'individual own',
                kind='group_individual', aggregation=agg, weight=weight,
                stress=name == 'laplacian_sum')


NAMES = ('ols', 'ols_no_constant', 'collinear', 'one_fe', 'two_fe', 'three_fe',
         'disconnected', 'heterogeneous', 'slope_only', 'aweights', 'fweights',
         'pweights', 'iweights', 'robust', 'cluster', 'multi_cluster', 'singletons', 'group_only',
         'group_sum', 'group_mean', 'group_uniform_sum', 'group_uniform_mean',
         'group_fweights_sum', 'group_aweights_mean', 'laplacian_sum')


def generate(name):
    return grouped(name) if name.startswith('group_') and name != 'group_only' or name == 'laplacian_sum' else ordinary(name)
