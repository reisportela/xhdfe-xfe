#!/usr/bin/env python3
"""Generate the pyfixest-DGP benchmark datasets used by the xhdfe core-23 suite.

DGP credit: this reproduces the "simple"/"difficult" fixed-effects benchmark
data-generating process of the **pyfixest** package
(https://github.com/py-econometrics/pyfixest, benchmarks/modular/
dgp_functions.py), which itself credits Kyle Butts' fixest_benchmarks
(https://github.com/kylebutts/fixest_benchmarks). Both are MIT licensed. The
DGP is reimplemented here (plain numpy/pandas) because pyfixest wheels do not
ship the benchmarks module.

The xhdfe core-23 benchmark uses the two n=10,000,000, k=10 datasets, each
generated with seed 1234:

    benchmark_simple_n10000000_k10.parquet     (iid-uniform firm assignment)
    benchmark_difficult_n10000000_k10.parquet  (cyclic firm tiling; the
                                                ill-conditioned FE graph)

Layout: 1,000,000 workers x 10 years; 43,478 firms; y = X[:, :10] @ (1/j)
+ worker FE + firm FE + year FE + N(0,1). Columns: indiv_id, firm_id, year,
y, exp_y, negbin_y, binary_y, x1..x10.

Usage:
    python3 generate_pyfixest_data.py                 # both 10M datasets
    python3 generate_pyfixest_data.py --n 1000000     # smaller variants
    python3 generate_pyfixest_data.py --stata         # also write .dta
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def base_dgp(
    n: int = 1000,
    nb_year: int = 10,
    nb_indiv_per_firm: int = 23,
    type_: str = "simple",
    k: int = 10,
    max_k: int = 10,
    seed: int | None = None,
) -> pd.DataFrame:
    """The pyfixest/fixest simple/difficult benchmark DGP (see module header)."""
    rng = np.random.default_rng(seed)
    nb_indiv = round(n / nb_year)
    nb_firm = round(nb_indiv / nb_indiv_per_firm)
    if nb_indiv < 1 or nb_firm < 1:
        raise ValueError(f"n={n} too small for this DGP")

    n_obs = nb_indiv * nb_year
    if k < 1 or k > max_k:
        raise ValueError(f"k={k} must satisfy 1 <= k <= max_k={max_k}")

    indiv_id = np.repeat(np.arange(1, nb_indiv + 1), nb_year)
    year = np.tile(np.arange(1, nb_year + 1), nb_indiv)
    if type_ == "simple":
        firm_id = rng.integers(1, nb_firm + 1, size=n_obs)
    elif type_ == "difficult":
        firm_id = np.tile(np.arange(1, nb_firm + 1), n_obs // nb_firm + 1)[:n_obs]
    else:
        raise ValueError(f"Unknown DGP type: {type_!r}")

    x = rng.standard_normal((n_obs, max_k))
    betas = 1.0 / np.arange(1, k + 1, dtype=float)
    firm_fe = rng.standard_normal(nb_firm)[firm_id - 1]
    unit_fe = rng.standard_normal(nb_indiv)[indiv_id - 1]
    year_fe = rng.standard_normal(nb_year)[year - 1]
    mu = x[:, :k] @ betas + firm_fe + unit_fe + year_fe
    y = mu + rng.standard_normal(len(mu))

    theta = 0.5
    exp_y = np.exp(y)
    nb_p = theta / (theta + exp_y)
    negbin_y = rng.negative_binomial(n=theta, p=nb_p)

    data = {
        "indiv_id": indiv_id,
        "firm_id": firm_id,
        "year": year,
        "y": y,
        "exp_y": exp_y,
        "negbin_y": negbin_y,
        "binary_y": (y > 0).astype(np.int8),
    }
    for j in range(max_k):
        data[f"x{j + 1}"] = x[:, j]
    return pd.DataFrame(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10_000_000,
                    help="observations per dataset (default 10,000,000 — the core-23 size)")
    ap.add_argument("--seed", type=int, default=1234,
                    help="RNG seed (default 1234 — reproduces the core-23 datasets bit-exactly at n=10,000,000)")
    ap.add_argument("--types", default="simple,difficult",
                    help="comma list of DGP types (default: simple,difficult)")
    ap.add_argument("--stata", action="store_true",
                    help="also write a .dta copy for the Stata runner")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for type_ in [t.strip() for t in args.types.split(",") if t.strip()]:
        name = f"benchmark_{type_}_n{args.n}_k10"
        out = DATA_DIR / f"{name}.parquet"
        print(f"generating {name} (seed={args.seed}) ...", flush=True)
        df = base_dgp(n=args.n, type_=type_, seed=args.seed)
        df.to_parquet(out, index=False)
        print(f"  wrote {out} ({len(df):,} rows)")
        if args.stata:
            dta = DATA_DIR / f"{name}.dta"
            df.drop(columns=["exp_y", "negbin_y", "binary_y"]).to_stata(
                dta, write_index=False)
            print(f"  wrote {dta}")


if __name__ == "__main__":
    main()
