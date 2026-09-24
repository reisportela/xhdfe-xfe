"""Four ordinary FE: independent projection/sandwich checks for CUDA Auto."""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
os.environ.update(XHDFE_GPU_BACKEND='cuda', XHDFE_MOBILITY_MODE='off',
                  XHDFE_FE_STRUCTURE_MODE='off', XHDFE_ABSORPTION_CACHE_MODE='off')
spec = importlib.util.spec_from_file_location('py_hdfe_v11', args.module)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
rng = np.random.default_rng(20260924)
n = 1024
base_fes = [rng.integers(0, g, n, dtype=np.int32) for g in (16, 16, 8, 4)]
base_x = rng.normal(size=(n, 2))
base_y = 1.25 + base_x @ np.array([.75, -.25])
base_y += sum(rng.normal(size=int(f.max()) + 1)[f] for f in base_fes)
base_y += .2 * rng.normal(size=n)
fixtures = []
for name in ('regular', 'redundant_fe', 'disconnected', 'zero_y', 'absorbed_x1', 'small_y'):
    fes = [f.copy() for f in base_fes]
    x, y = base_x.copy(), base_y.copy()
    active = [0, 1]
    if name == 'redundant_fe':
        fes[2] = fes[0].copy()
    elif name == 'disconnected':
        for f in fes:
            f[n // 2:] += int(f.max()) + 1
    elif name == 'zero_y':
        y[:] = 0
    elif name == 'absorbed_x1':
        x[:, 0] = fes[0] / 8.
        active = [1]
    elif name == 'small_y':
        y *= 2.**-20
    fixtures.append((name, y, x, fes, np.arange(n) % 32, active))

# Exact balanced contrasts exercise the non-small CG implementation too.
n = 32768
bits = [((np.arange(n) >> j) & 1).astype(np.int32) for j in range(4)]
signs = [2*f.astype(float)-1 for f in bits]
x = np.column_stack((signs[0]*signs[1], signs[2]*signs[3]))
cluster = (np.arange(n)//16) % 32
noise_sign = 2*(cluster % 2)-1
y = 1.25 + x @ np.array([.75, -.25]) + sum((j+1)*s/8 for j, s in enumerate(signs))
y += noise_sign * (x[:, 0] + 2*x[:, 1]) / 8
fixtures.append(('balanced_medium', y, x, bits, cluster, [0, 1]))
scaled_x = x * np.array([2.**-20, 2.**10])
fixtures.append(('column_units_medium', y, scaled_x, bits, cluster, [0, 1]))

# A complete fine-group/time grid has a sparse FE and an exact two-way
# projection; the other two FE dimensions are redundant by construction.
fine = np.repeat(np.arange(4096, dtype=np.int32), 8)
period = np.tile(np.arange(8, dtype=np.int32), 4096)
grid_fes = [fine//4, fine, period, period % 4]
grid_cluster = ((fine//16 + period) % 32).astype(np.int32)
grid_x = rng.normal(size=(len(fine), 2))
grid_y = 1.25 + grid_x @ np.array([.75, -.25]) + np.sin(fine/37.) + np.cos(period/5.)
grid_y += .2*rng.normal(size=len(fine))
fixtures.append(('sparse_grid', grid_y, grid_x, grid_fes, grid_cluster, [0, 1]))
absorbed_x = grid_x.copy()
absorbed_x[:, 0] = np.sin(fine/13.) + np.cos(period/5.)
fixtures.append(('sparse_absorbed_x1', grid_y, absorbed_x, grid_fes, grid_cluster, [1]))

rows = []
for name, y, x, fes, cluster, active in fixtures:
    n = len(y)
    assert all(np.bincount(f).min() >= 2 for f in fes)
    z = np.column_stack((y, x))
    if name in ('sparse_grid', 'sparse_absorbed_x1'):
        grid = z.reshape(4096, 8, 3)
        within = (grid - grid.mean(axis=1, keepdims=True)
                  - grid.mean(axis=0, keepdims=True) + grid.mean(axis=(0, 1))).reshape(n, 3)
        rank_fe = 4096 + 8 - 1
    else:
        d = np.column_stack([np.eye(int(f.max())+1)[f] for f in fes])
        u, singular, _ = np.linalg.svd(d, full_matrices=False)
        rank_fe = int(np.sum(singular > singular[0] * max(d.shape) * np.finfo(float).eps))
        qfe = u[:, :rank_fe]
        within = z - qfe @ (qfe.T @ z)
    if name == 'zero_y':
        within[:, 0] = 0
    xt, yt = within[:, np.asarray(active)+1], within[:, 0]
    scales = np.linalg.norm(xt, axis=0)
    xs = xt / scales
    qx, r = np.linalg.qr(xs, mode='reduced')
    beta = np.linalg.solve(r, qx.T @ yt) / scales
    residual = yt - xt @ beta
    inverse_r = np.linalg.inv(r)
    influence_active = ((inverse_r @ qx.T).T / scales).T
    influence = np.zeros((3, n))
    influence[active, :] = influence_active
    means = np.asarray(x.mean(axis=0, dtype=np.longdouble), dtype=float)
    influence[2, :] = 1/n - means[active] @ influence_active
    bref = np.zeros(3)
    bref[active] = beta
    bref[2] = float(y.mean(dtype=np.longdouble)) - means @ bref[:2]
    if name in ('balanced_medium', 'column_units_medium'):
        # These dyadic contrasts are exactly orthogonal to every FE. Avoid
        # treating SVD roundoff in a theoretically zero cluster score as data.
        units = np.array([1., 1.]) if name == 'balanced_medium' else np.array([2.**-20, 2.**10])
        base = x / units
        residual = (2*(cluster % 2)-1) * (base[:, 0] + 2*base[:, 1]) / 8
        influence[:2, :] = x.T / (n * units[:, None]**2)
        influence[2, :] = 1/n
        bref = np.r_[np.array([.75, -.25])/units, 1.25]
    df = n - rank_fe - len(active)
    assert df > 0
    groups = int(cluster.max()) + 1
    # This fixture has no FE nested in the independently assigned cluster.
    for f in fes:
        minimum = np.full(int(f.max())+1, groups)
        maximum = np.full(int(f.max())+1, -1)
        np.minimum.at(minimum, f, cluster)
        np.maximum.at(maximum, f, cluster)
        assert (maximum > minimum).all()
    for se in ('unadjusted', 'robust', 'cluster'):
        if se == 'unadjusted':
            vref = (residual @ residual) / df * (influence @ influence.T)
        else:
            scores = influence * residual
            if se == 'robust':
                vref = n/df * (scores @ scores.T)
            else:
                grouped = np.column_stack([np.bincount(cluster, weights=s, minlength=groups) for s in scores])
                vref = groups/(groups-1) * (n-1)/df * (grouped.T @ grouped)
        model = core.HdfeRegressor(se_type=se, num_threads=2, tol=1e-8, max_iter=100000,
                                  tolerance_mode='reghdfe-comparable', absorption_method='auto')
        kwargs = {'clusters': cluster.astype(np.int32)} if se == 'cluster' else {}
        started = time.perf_counter()
        try:
            model.fit(y, np.asfortranarray(x), fes, **kwargs)
        except RuntimeError as error:
            rows.append(dict(case=name, se=se, status='REFUSED', error=str(error),
                             fit_seconds=time.perf_counter()-started))
            continue
        fit_seconds = time.perf_counter()-started
        actual_b, actual_v = np.asarray(model.coef_), np.asarray(model.covariance_)
        be = float(np.max(abs(actual_b-bref)/np.maximum(1, abs(bref))))
        scale = np.sqrt(np.outer(vref.diagonal(), vref.diagonal()))
        positive = scale > 0
        ve = float(np.max(abs(actual_v-vref)[positive]/scale[positive])) if positive.any() else 0.
        zeros = bool(np.all(actual_v[~positive] == vref[~positive]))
        passed = model.converged_ and model.gpu_used_ and be <= 1e-9 and ve <= 1e-8 and zeros
        rows.append(dict(case=name, se=se, b_error=be, V_error=ve,
                         fit_seconds=fit_seconds,
                         zero_entries_equal=zeros, iterations=int(model.num_iterations_),
                         converged=bool(model.converged_), gpu_used=bool(model.gpu_used_),
                         status='PASS' if passed else 'REVIEW'))
args.output.write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps({status: sum(row['status'] == status for row in rows)
                  for status in ('PASS', 'REVIEW', 'REFUSED')}))
raise SystemExit(any(row['status'] != 'PASS' for row in rows))
