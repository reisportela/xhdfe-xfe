"""Independent dense synthetic oracle for completed-residual inference.

Research only: no production imports, no tuning on coverage, no source data.
K=4 is fixed in the round-3 proposal. The normal interval is diagnostic only;
the projected interval is always used, including weak/zero signals.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, null_space
from scipy.optimize import brentq, minimize
from scipy.stats import chi2, norm


def trust_min(mat, linear, radius):
    """Global minimum of x'mat x+2 linear'x, ||x||<=radius.

    Eigensystem/secular oracle, including PSD interior and the hard case.
    Returns primal and Lagrange-dual bounds, not just optimiser success.
    """
    d, u = np.linalg.eigh((mat + mat.T) / 2)
    g = u.T @ linear
    scale = max(np.max(np.abs(d)), np.linalg.norm(g) / radius,
                np.finfo(float).tiny)
    tol = 64 * np.finfo(float).eps * scale
    shift = max(0., -d[0])
    denom = d + shift
    zero = denom <= tol
    x = np.zeros_like(g)
    x[~zero] = -g[~zero] / denom[~zero]
    gsmall = np.linalg.norm(g[zero]) <= tol * radius
    if gsmall and np.linalg.norm(x) <= radius:
        if shift > 0:
            x[np.flatnonzero(zero)[0]] = np.sqrt(max(0., radius**2 - x @ x))
        lam = shift
        method = "hard" if shift else "interior"
    else:
        # Solve for the distance from the leftmost pole, not for lambda;
        # this retains resolution near a hard case.
        def secular(t):
            dd = denom + t
            if np.any((dd <= 0) & (g != 0)):
                return np.inf
            z = np.divide(g, dd, out=np.zeros_like(g), where=dd > 0)
            return np.linalg.norm(z) / radius - 1
        high = np.linalg.norm(g) / radius
        high = max(high, np.finfo(float).eps * scale)
        while secular(high) > 0:
            high *= 2
        t = brentq(secular, 0., high, xtol=np.finfo(float).tiny,
                   rtol=8*np.finfo(float).eps)
        lam = shift + t
        x = -g / (denom + t)
        method = "boundary"
    primal = float(np.sum(d * x*x + 2*g*x))
    dual_denom = d + lam
    good = dual_denom > tol
    dual = float(-np.sum(g[good]**2 / dual_denom[good]) - lam*radius**2)
    point = u @ x
    residual = float(np.linalg.norm(mat @ point + linear + lam*point))
    return dict(value=primal, dual=dual, gap=primal-dual,
                stationarity=residual, radius=float(np.linalg.norm(point)),
                method=method, point=point)


def project(kappa, covariance, amat, critical):
    root = np.linalg.cholesky(covariance)
    mat = root.T @ amat @ root
    linear = root.T @ amat @ kappa
    const = float(kappa @ amat @ kappa)
    low = trust_min(mat, linear, np.sqrt(critical))
    high = trust_min(-mat, -linear, np.sqrt(critical))
    return const+low["value"], const-high["value"], max(low["gap"], high["gap"])


def make_design(kind, workers, firms, between, weak, seed):
    rng = np.random.default_rng(seed)
    if kind == "grid":
        w = np.repeat(np.arange(workers), 2*firms)
        f = np.tile(np.repeat(np.arange(firms), 2), workers)
    else:
        ws, fs = [], []
        for worker in range(workers):
            stay = kind == "stayers" and worker % 3 != 0
            first = worker % firms
            second = first if stay else (first + 1 + worker // firms % (firms-1)) % firms
            ws.extend([worker]*4)
            fs.extend([first, first, second, second])
        w, f = np.array(ws), np.array(fs)
    ordering = np.lexsort((f, w))
    w, f = w[ordering], f[ordering]
    n = len(w)
    df = np.column_stack([np.eye(workers)[w], np.eye(firms)[f, :firms-1]])
    q, _ = np.linalg.qr(df)
    raw = rng.normal(size=n)
    within = raw - q @ (q.T @ raw)
    within /= np.std(within)
    firm_signal = (f-f.mean())/f.std() * (0. if weak else np.sqrt(.08))
    control = between*((f-f.mean())/f.std()*np.sqrt(.08)) + .3*within
    cs = [control]
    if kind == "local":
        local = np.zeros(n)
        local[:2] = [1., -1.]
        cs.append(local)
    x = np.column_stack([df, *cs])
    inverse = np.linalg.solve(x.T @ x, x.T)
    projector = x @ inverse
    m = np.eye(n)-projector
    probs = np.bincount(f, minlength=firms)/n
    contrast = null_space(np.sqrt(probs)[None, :]).T @ np.diag(np.sqrt(probs))
    psi = np.zeros((firms, x.shape[1]))
    psi[:firms-1, workers:workers+firms-1] = np.eye(firms-1)
    worker_average = np.zeros((firms, x.shape[1]))
    np.add.at(worker_average, (f, w), 1.)
    worker_average /= (n*probs[:, None])
    raw_map = np.vstack([contrast @ psi, contrast @ worker_average]) @ inverse
    d, u = np.linalg.eigh(raw_map @ raw_map.T)
    active = d > max(d)*1e-11
    basis = u[:, active]
    kmap = basis.T @ raw_map
    rp = firms-1
    avar_raw = np.zeros((2*rp, 2*rp))
    avar_raw[:rp, :rp] = np.eye(rp)*n/(n-1)
    acov_raw = np.zeros_like(avar_raw)
    acov_raw[:rp, rp:] = np.eye(rp)*n/(n-1)/2
    acov_raw[rp:, :rp] = np.eye(rp)*n/(n-1)/2
    avar, acov = basis.T @ avar_raw @ basis, basis.T @ acov_raw @ basis
    signal = .3*np.sin(w) + firm_signal + .5*control
    if kind == "local":
        signal += .1*cs[-1]
    clusters = [np.flatnonzero((w == ww) & (f == ff))
                for ww, ff in sorted(set(zip(w, f)))]
    lmap = np.zeros((n, n))
    omegas = []
    min_local_residual_eigenvalue = 1.
    for g in clusters:
        pg = projector[np.ix_(g, g)]
        mg = np.eye(len(g))-pg
        min_local_residual_eigenvalue = min(min_local_residual_eigenvalue,
                                          np.linalg.eigvalsh(mg)[0])
        power = np.eye(len(g))
        h = power.copy()
        for _ in range(4):
            power = power @ pg
            h += power
        c = power @ pg
        other = np.ones(n)/(n-len(g))
        other[g] = 0
        mapping = np.eye(n)[g] - other[None, :]
        lmap[g] = h @ m[g] + c @ mapping
        variance = .09*(1 + .8*(f[g[0]] % 2))
        rho = .6 if kind in ("stayers", "serial", "local") else 0.
        omegas.append(variance*rho**np.abs(np.subtract.outer(np.arange(len(g)), np.arange(len(g)))))
    # g order follows original rows for these fixtures.
    order = np.concatenate(clusters)
    assert np.array_equal(order, np.arange(n))
    omega = block_diag(*omegas)
    vtrue = kmap @ omega @ kmap.T
    br = lmap @ signal
    vr = lmap @ omega @ lmap.T
    expected = np.zeros_like(vtrue)
    self_error, shift_error = 0., np.max(np.abs(lmap @ np.ones(n)))
    for g in clusters:
        self_error = max(self_error, np.max(np.abs(lmap[np.ix_(g,g)]-np.eye(len(g)))))
        expected += kmap[:,g] @ (vr[np.ix_(g,g)] + np.outer(br[g], br[g])) @ kmap[:,g].T
    whitener = np.linalg.inv(np.linalg.cholesky(vtrue))
    eig_slack = np.linalg.eigvalsh(whitener @ (expected-vtrue) @ whitener.T)
    identity_bread = kmap @ kmap.T
    design_white = np.linalg.solve(np.linalg.cholesky(identity_bread), kmap)
    concentration = max(np.linalg.norm(design_white[:,g], 2)**2 for g in clusters)
    return locals()


def run_case(case, reps, seed):
    start = time.monotonic()
    d = make_design(**case, seed=seed)
    rng = np.random.default_rng(seed+991)
    kmap, lmap, signal = d["kmap"], d["lmap"], d["signal"]
    truth_k = kmap @ signal
    truth = [float(truth_k @ a @ truth_k) for a in (d["avar"], d["acov"])]
    critical = chi2.ppf(.95, len(truth_k))
    cov_root = np.linalg.cholesky(d["omega"])
    count_ellipsoid = 0
    projected = [0, 0]
    normal = [0, 0]
    widths = [[], []]
    squared_se = [[], []]
    maximum_gap = 0.
    for begin in range(0, reps, 100):
        size = min(100, reps-begin)
        innovation = rng.normal(size=(size, d["n"]))
        if case["kind"] == "serial":
            innovation = (rng.exponential(size=(size, d["n"]))-1)
        ys = signal + innovation @ cov_root.T
        khat = ys @ kmap.T
        residual = ys @ lmap.T
        covariance = np.zeros((size, len(truth_k), len(truth_k)))
        for g in d["clusters"]:
            score = residual[:,g] @ kmap[:,g].T
            covariance += np.einsum("bi,bj->bij", score, score)
        diff = khat-truth_k
        stats = np.einsum("bi,bi->b", diff, np.linalg.solve(covariance, diff[...,None])[...,0])
        count_ellipsoid += int(np.sum(stats <= critical))
        for j, a in enumerate((d["avar"], d["acov"])):
            for b in range(size):
                low, high, gap = project(khat[b], covariance[b], a, critical)
                maximum_gap = max(maximum_gap, gap)
                projected[j] += int(low-1e-11 <= truth[j] <= high+1e-11)
                widths[j].append(high-low)
                estimate = float(khat[b] @ a @ khat[b])
                se2 = float(4*(a @ khat[b]) @ covariance[b] @ (a @ khat[b]))
                squared_se[j].append(se2)
                normal[j] += int(abs(estimate-truth[j]) <= norm.ppf(.975)*np.sqrt(max(0,se2)))
    return dict(case=case, n=d["n"], rank=len(truth_k), repetitions=reps,
                independent_clusters=len(d["clusters"]), truth=truth,
                maximum_diagonal_identity_error=d["self_error"],
                shift_identity_error=float(d["shift_error"]),
                min_local_M_eigenvalue=float(d["min_local_residual_eigenvalue"]),
                minimum_whitened_expectation_slack=float(d["eig_slack"][0]),
                design_concentration=float(d["concentration"]),
                ellipsoid_coverage=count_ellipsoid/reps,
                projection_coverage=np.array(projected).tolist() if not reps else (np.array(projected)/reps).tolist(),
                plugin_normal_coverage=(np.array(normal)/reps).tolist(),
                mean_projection_width=[float(np.mean(v)) for v in widths],
                rms_delta_se=[float(np.sqrt(np.mean(v))) for v in squared_se],
                maximum_primal_dual_gap=maximum_gap,
                elapsed_seconds=time.monotonic()-start)


def solver_tests():
    rng = np.random.default_rng(181818)
    tests = [(np.diag([-2., 1.]), np.array([0., .3]), 1.),
             (np.eye(2), np.zeros(2), 1.),
             (np.diag([0., 1.]), np.array([0., .2]), 1.),
             (np.diag([-2., 1.]), np.array([1e-12, .3]), 1.)]
    for r in (1,2,5,12):
        for _ in range(8):
            a = rng.normal(size=(r,r)); a = (a+a.T)/2
            tests.append((a, rng.normal(size=r), .1+rng.random()*3))
    max_gap, max_residual, max_optimizer = 0., 0., 0.
    for a,g,rad in tests:
        result = trust_min(a,g,rad)
        scale = max(1., np.linalg.norm(a)*rad**2, np.linalg.norm(g)*rad)
        assert result["radius"] <= rad*(1+1e-10)
        assert abs(result["gap"]) < 1e-9*scale
        assert result["stationarity"] < 1e-9*scale/rad
        max_gap = max(max_gap, abs(result["gap"])/scale)
        max_residual = max(max_residual, result["stationarity"]*rad/scale)
        # Independent constrained optimiser, started at the certified point;
        # random feasible points must not improve it either.
        other = minimize(lambda z: z @ a @ z+2*g @ z, result["point"],
                         jac=lambda z: 2*(a @ z+g), method="SLSQP",
                         constraints=[dict(type="ineq", fun=lambda z: rad**2-z @ z,
                                           jac=lambda z: -2*z)],
                         options=dict(ftol=1e-12, maxiter=100))
        assert result["value"] <= other.fun+1e-8*scale
        max_optimizer = max(max_optimizer, (result["value"]-other.fun)/scale)
    return dict(cases=len(tests), max_scaled_gap=max_gap,
                max_scaled_stationarity=max_residual,
                max_optimizer_improvement=max_optimizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    cases = [dict(kind="grid",workers=20,firms=10,between=a,weak=False)
             for a in (0.,1.,5.)]
    cases += [dict(kind=k,workers=100,firms=5,between=5.,weak=weak)
              for k,weak in [("thin",False),("stayers",False),("serial",False),
                             ("local",False),("thin",True),("stayers",True)]]
    numerical = solver_tests()
    print(json.dumps(dict(solver=numerical)), flush=True)
    rows = []
    for index, case in enumerate(cases):
        result = run_case(case,args.repetitions,1812000+index)
        rows.append(result)
        print(json.dumps(result), flush=True)
    with args.output.open("x") as out:
        json.dump(dict(status="RESEARCH_PILOT_NOT_RELEASE_ACCEPTANCE", rows=rows,
                       solver_tests=numerical, neumann_degree=4,
                       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       scope="Dense full-model synthetic oracle; no production integration"),
                  out, indent=2)


if __name__ == "__main__":
    main()
