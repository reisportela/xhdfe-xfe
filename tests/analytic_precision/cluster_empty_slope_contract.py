#!/usr/bin/env python3
"""Cluster inference with no identified slopes, from explicit FE residual scores."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("py_hdfe_v11", a.module)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rng = np.random.default_rng(48)
    n = 400
    f = np.repeat(np.arange(20), 20).astype(np.int32)
    cluster = [(np.arange(n) % 11).astype(np.int32), ((np.arange(n)//3) % 7).astype(np.int32)]
    y = rng.normal(size=n) + f/20
    rows = []
    for weight_type in ("none", "analytic", "frequency"):
        w = (rng.integers(1, 4, n).astype(float) if weight_type == "frequency"
             else rng.uniform(.5, 2, n) if weight_type == "analytic" else np.ones(n))
        effective = w.sum() if weight_type == "frequency" else n
        means = np.bincount(f, weights=w*y)/np.bincount(f, weights=w)
        residual = y-means[f]
        score = w/w.sum()*residual
        for dimensions in (1, 2):
            clusters = cluster[:dimensions]
            minimum = min(len(np.unique(c)) for c in clusters)
            parts = []
            labels = [clusters[0][:, None]] if dimensions == 1 else [
                clusters[0][:, None], clusters[1][:, None], np.column_stack(clusters)]
            for group in labels:
                _, ids = np.unique(group, axis=0, return_inverse=True)
                total = np.bincount(ids, weights=score)
                parts.append((total.size, total @ total))
            for convention in ("min", "conventional"):
                for adjusted in (False, True):
                    for group_adjusted in (False, True):
                        variance = 0.
                        for sign, (groups, meat) in zip((1, 1, -1), parts):
                            factor = groups/(groups-1) if group_adjusted and convention == "conventional" else 1.
                            variance += sign*factor*meat
                        if group_adjusted and convention == "min":
                            variance *= minimum/(minimum-1)
                        if adjusted:
                            variance *= (effective-1)/(effective-20)
                        variance = max(0., variance)
                        for omitted in (False, True):
                            x = f.astype(float)[:, None] if omitted else np.empty((n, 0))
                            model = core.HdfeRegressor(num_threads=1, se_type="cluster",
                                ssc_g_df=convention, ssc_g_adj=group_adjusted, ssc_k_adj=adjusted)
                            model.fit(y, x, [f], clusters=clusters,
                                weights=None if weight_type == "none" else w,
                                fweights=weight_type == "frequency")
                            assert model.num_clusters_ == minimum
                            assert model.df_resid_ == minimum-1
                            assert abs(model.coef_[-1]-(w @ y/w.sum())) <= 1e-9
                            error = abs(model.covariance_[-1, -1]-variance)
                            assert error <= 1e-8*variance if variance else error == 0.
                            rows.append(dict(weights=weight_type, dimensions=dimensions,
                                convention=convention, k_adjusted=adjusted, g_adjusted=group_adjusted,
                                omitted=omitted, clusters=model.num_clusters_, df=model.df_resid_,
                                variance_error=float(error), status="PASS"))
    m = core.HdfeRegressor(num_threads=1, se_type="cluster")
    m.fit(y, np.empty((n, 0)), [f], clusters=[np.zeros(n, dtype=np.int32)])
    assert m.num_clusters_ == 1 and m.df_resid_ == 0 and not np.isfinite(m.stderr_[-1])
    try:
        m.fit(y, np.empty((n, 0)), [f])
    except RuntimeError:
        pass
    else:
        raise AssertionError("cluster VCE without cluster labels must refuse")
    a.output.write_text(json.dumps(rows, indent=2)+"\n")
    print(f"PASS: {len(rows)} clustered empty-slope cases and missing-inference controls")


if __name__ == "__main__":
    main()
