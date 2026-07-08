#!/usr/bin/env python3
"""Generate the simulated worker-firm-occupation panel of the xhdfe core-23 suite.

This is the xhdfe project's own simulated benchmark dataset (authors: the
xhdfe authors). The DGP below is a faithful copy of the canonical generator
(notebook/simulated_panel_data.ipynb in the development repository) that
produced the benchmark file on 08dec2025 with seed 42:

    173,163,263 rows | 20,000,000 workers | 1,000,000 firms
    500 occupations | 15 years | spell length 3 + Poisson(6)

Benchmark model: ln_wage on education experience experience_sq union,
absorbing worker_id + firm_id + occupation_id + year (4-way HDFE), standard
errors clustered by worker_id. Ground-truth fixed effects (fe_worker, fe_firm,
fe_occupation) are stored in the file.

Notes for replicators:
- Full-size generation is a serial pure-Python loop over 20M workers: expect
  multiple hours and a ~2 GB zstd parquet (~16 GB in memory when loaded).
- The stream is deterministic given the seed, but NumPy Generator bit-streams
  are only guaranteed stable within a NumPy release series.
- Use --workers to generate a smaller panel with the same DGP (clearly not the
  benchmark dataset; useful for smoke tests), and --stata to also write a .dta
  copy for the Stata runner (large and slow at full size).

Usage:
    python3 simulate_panel.py                       # the full benchmark panel
    python3 simulate_panel.py --workers 100000      # small same-DGP panel
    python3 simulate_panel.py --stata               # also write .dta
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent

PARAMS = {
    "target_rows": 1_000_000_000,
    "max_rows": 1_000_000_000,
    "n_workers": 20_000_000,
    "n_firms": 1_000_000,
    "n_occupations": 500,
    "n_years": 15,
    "min_tenure": 3,
    "mean_tenure": 9.0,
    "start_decay": 0.35,
    "firm_move_prob": 0.45,
    "occupation_move_prob": 0.3,
    "force_move": True,
    "chunk_size": 2_000_000,
    "seed": 42,
    "output_path": HERE / "data" / "simulated_panel.parquet",
}


def make_start_probs(n_years: int, min_tenure: int, decay: float) -> np.ndarray:
    max_start = n_years - min_tenure
    idx = np.arange(max_start + 1)
    weights = np.exp(-decay * idx)
    return weights / weights.sum()


def simulate_panel(cfg: Dict) -> Dict:
    rng = np.random.default_rng(cfg["seed"])
    start_probs = make_start_probs(cfg["n_years"], cfg["min_tenure"], cfg["start_decay"])
    start_indices = np.arange(len(start_probs))
    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    firm_fes = rng.normal(0, 0.25, size=cfg["n_firms"])
    occupation_fes = rng.normal(0, 0.25, size=cfg["n_occupations"])
    tgrid = np.linspace(0, 2 * np.pi, cfg["n_years"])
    year_fes = 0.06 * np.sin(tgrid) - 0.02 * np.cos(0.5 * tgrid)

    buffers: Dict[str, List[np.ndarray]] = {k: [] for k in (
        "worker_id", "year", "firm_id", "occupation_id", "education",
        "experience", "experience_sq", "union", "ln_wage",
        "fe_worker", "fe_firm", "fe_occupation")}
    buffer_rows = 0
    total_rows = 0
    actual_workers = 0
    writer = None

    def flush():
        nonlocal buffer_rows, writer
        if buffer_rows == 0:
            return
        table = pa.table({k: np.concatenate(v) for k, v in buffers.items()})
        nonlocal_writer_init(table)
        buffer_rows = 0
        for k in buffers:
            buffers[k] = []

    def nonlocal_writer_init(table):
        nonlocal writer
        if writer is None:
            writer = pq.ParquetWriter(
                output_path, table.schema, compression="zstd",
                use_dictionary=False, write_statistics=True)
        writer.write_table(table)

    lambda_tenure = max(cfg["mean_tenure"] - cfg["min_tenure"], 1e-3)
    for worker_id in range(cfg["n_workers"]):
        remaining = cfg["max_rows"] - total_rows
        if remaining < cfg["min_tenure"]:
            break

        start_year = int(rng.choice(start_indices, p=start_probs))
        available_years = cfg["n_years"] - start_year
        if available_years < cfg["min_tenure"]:
            continue

        tenure_draw = cfg["min_tenure"] + rng.poisson(lambda_tenure)
        n_obs = min(available_years, tenure_draw)
        if n_obs < cfg["min_tenure"]:
            n_obs = available_years
        if n_obs < cfg["min_tenure"]:
            continue
        if n_obs > remaining:
            n_obs = remaining
        if n_obs < cfg["min_tenure"]:
            continue

        possible_years = np.arange(start_year, cfg["n_years"], dtype=np.int16)
        if possible_years.size < cfg["min_tenure"]:
            continue
        if n_obs > possible_years.size:
            n_obs = possible_years.size
        chosen_rest = rng.choice(possible_years[1:], size=n_obs - 1, replace=False) if n_obs > 1 else np.array([], dtype=np.int16)
        years = np.sort(np.concatenate(([possible_years[0]], chosen_rest)))

        firm_ids = np.empty(years.size, dtype=np.int32)
        occ_ids = np.empty(years.size, dtype=np.int16)
        firm_ids[0] = rng.integers(0, cfg["n_firms"], dtype=np.int32)
        occ_ids[0] = rng.integers(0, cfg["n_occupations"], dtype=np.int16)
        moved = False

        for t in range(1, years.size):
            prev_f, prev_o = firm_ids[t - 1], occ_ids[t - 1]
            if rng.random() < cfg["firm_move_prob"]:
                candidate = rng.integers(0, cfg["n_firms"], dtype=np.int32)
                firm_ids[t] = candidate if candidate != prev_f else (candidate + 1) % cfg["n_firms"]
            else:
                firm_ids[t] = prev_f

            if rng.random() < cfg["occupation_move_prob"]:
                candidate = rng.integers(0, cfg["n_occupations"], dtype=np.int16)
                occ_ids[t] = candidate if candidate != prev_o else (candidate + 1) % cfg["n_occupations"]
            else:
                occ_ids[t] = prev_o

            moved = moved or (firm_ids[t] != prev_f) or (occ_ids[t] != prev_o)

        if cfg["force_move"] and (not moved) and years.size >= 2:
            firm_ids[-1] = (firm_ids[-1] + 1) % cfg["n_firms"]
            occ_ids[-1] = (occ_ids[-1] + 1) % cfg["n_occupations"]

        fe_w = rng.normal(0, 0.35)
        fe_f = firm_fes[firm_ids]
        fe_o = occupation_fes[occ_ids]

        education_raw = 12.0 + 1.5 * fe_w + 1.1 * fe_f + 0.8 * fe_o + rng.normal(0, 1.0, size=years.size)
        education = np.rint(np.clip(education_raw, 0.0, 20.0)).astype(np.float32)
        experience = np.arange(1, years.size + 1, dtype=np.int16)
        experience_sq = (experience.astype(np.int32) ** 2).astype(np.int32)

        union_latent = -0.4 - 0.6 * fe_w - 0.45 * fe_f - 0.35 * fe_o + rng.normal(0, 0.25, size=years.size)
        union_prob = 1.0 / (1.0 + np.exp(-union_latent))
        union = rng.random(size=years.size) < union_prob

        ln_wage = (
            1.8
            + 0.08 * education
            + 0.02 * experience
            - 0.0002 * experience_sq
            + 0.12 * union
            + fe_w + fe_f + fe_o
            + year_fes[years]
            + rng.normal(0, 0.12, size=years.size)
        )

        buffers["worker_id"].append(np.full(years.size, worker_id, dtype=np.int32))
        buffers["year"].append((years + 1).astype(np.int16))
        buffers["firm_id"].append(firm_ids.astype(np.int32))
        buffers["occupation_id"].append(occ_ids.astype(np.int16))
        buffers["education"].append(education)
        buffers["experience"].append(experience.astype(np.int16))
        buffers["experience_sq"].append(experience_sq.astype(np.int32))
        buffers["union"].append(union.astype(np.bool_))
        buffers["ln_wage"].append(ln_wage.astype(np.float32))
        buffers["fe_worker"].append(np.full(years.size, fe_w, dtype=np.float32))
        buffers["fe_firm"].append(fe_f.astype(np.float32))
        buffers["fe_occupation"].append(fe_o.astype(np.float32))

        total_rows += years.size
        buffer_rows += years.size
        actual_workers += 1
        if buffer_rows >= cfg["chunk_size"]:
            flush()

    flush()
    if writer is not None:
        writer.close()

    return {"rows": total_rows, "workers": actual_workers, "path": output_path}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=PARAMS["n_workers"],
                    help="number of workers (default 20,000,000 — the benchmark panel)")
    ap.add_argument("--seed", type=int, default=PARAMS["seed"],
                    help="RNG seed (default 42 — the benchmark panel)")
    ap.add_argument("--output", type=Path, default=PARAMS["output_path"])
    ap.add_argument("--stata", action="store_true",
                    help="also write a .dta copy for the Stata runner (slow/large at full size)")
    args = ap.parse_args()

    cfg = dict(PARAMS)
    cfg["n_workers"] = args.workers
    cfg["seed"] = args.seed
    cfg["output_path"] = args.output
    if args.workers != PARAMS["n_workers"]:
        print(f"note: --workers {args.workers:,} != 20,000,000 — same DGP but NOT the benchmark dataset")

    res = simulate_panel(cfg)
    print(f"rows={res['rows']:,} workers={res['workers']:,} -> {res['path']}")

    if args.stata:
        import pandas as pd
        dta = Path(str(res["path"]).replace(".parquet", ".dta"))
        print(f"writing {dta} (this can take a while at full size) ...")
        df = pq.read_table(res["path"]).to_pandas()
        df["union"] = df["union"].astype(np.int8)
        df.drop(columns=["fe_worker", "fe_firm", "fe_occupation"]).to_stata(
            dta, write_index=False)
        print(f"wrote {dta}")


if __name__ == "__main__":
    main()
