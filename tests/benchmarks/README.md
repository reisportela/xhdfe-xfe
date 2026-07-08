# xhdfe benchmark replication suite

Public replication material for the `xhdfe` **core-23 benchmark** — the
internal suite the project uses to track performance across engines and
releases. This directory contains everything a researcher needs to replicate
the benchmarks on the **non-proprietary subset (20 of the 23 specifications)**:
download links for the public datasets, deterministic generators for the
simulated ones, and runners for Stata, Python, and R.

> **How this differs from `tests/stata/`:** that directory is the
> *correctness certification* suite (xhdfe vs Stata oracles on small synthetic
> data, runs in seconds). This directory is about *performance replication* on
> real-size benchmark datasets (runs in minutes to hours).

## The datasets

| Group | Specs | Source & credit |
| --- | --- | --- |
| **Sergio** (15) | `credit`, `credit2`, `directors`, `enron`, `github`, `patents`, `schools`, `soccer`, `synthetic-assortative`, `synthetic-complete`, `synthetic-uniform-easy/-hard/-harder`, `synthetic-zigzag`, `workers` | **Sergio Correia's HDFE Benchmark Dataset Collection** — <https://scorreia.com/data/hdfe/>. Downloaded (not redistributed) by `get_sergio_data.sh`. Per the collection's pages, `enron` derives from the SNAP Enron email network, `patents` from the SNAP patent-citation data, `github` from the BigQuery `github_repos` public dataset. |
| **Simulated** (1) | `simulated_panel` — 173,163,263 rows, 20M workers x 1M firms x 500 occupations x 15 years, 4-way HDFE | Simulated by the xhdfe authors; regenerate deterministically (seed 42) with `simulate_panel.py`. Ground-truth fixed effects are stored in the file. |
| **pyfixest DGP** (4 specs, 2 files) | `pf_{difficult,simple}_10m_{3fe,2fe}` — n=10,000,000, k=10, worker/firm/year | DGP by the **pyfixest** package (<https://github.com/py-econometrics/pyfixest>, `benchmarks/modular/dgp_functions.py`), itself adapted from **Kyle Butts'** `fixest_benchmarks` (MIT). Regenerate bit-exactly (seed 1234) with `generate_pyfixest_data.py`. The *difficult* variant tiles firm ids cyclically over the worker-major row order, producing the ill-conditioned FE graph; *simple* assigns firms iid-uniformly. The 2FE specs absorb worker+year on the same file (the firm effect stays in the error), mirroring the pyfixest site benchmark. |

**Excluded (proprietary):** the remaining 3 core-23 specs run on Portuguese
matched employer-employee microdata (*Quadros de Pessoal*): `main_95_21_ready`
(47.6M obs, 3-way HDFE), `akm_v02_firstreg` (57.8M obs), and
`akm_v02_secondreg` (heterogeneous firm-seniority slopes + two-way cluster —
the hardest spec in the suite). These cannot be redistributed; researchers can
apply for access through BPLIM (Banco de Portugal Microdata Research
Laboratory). All published xhdfe results on those specs are produced by the
same runners/options used here.

## Setup

```bash
# 1. the 15 public datasets (~230 MB of .dta)
bash get_sergio_data.sh

# 2. the pyfixest-DGP datasets (a few minutes; ~3 GB; add --stata for the Stata runner)
python3 generate_pyfixest_data.py --stata

# 3. the simulated panel (multi-hour serial generation; ~2 GB parquet;
#    needs a large-memory machine to RUN the benchmark - skip if not replicating it)
python3 simulate_panel.py
```

Install `xhdfe` for the front-end(s) you want to replicate (see the repository
README): Stata `net install xhdfe, from(...)`; Python `pip install .`;
R `remotes::install_github("reisportela/xhdfe-xfe", subdir = "r/xhdfe")`.
For GPU numbers, build the CUDA backend (`xhdfegpu` in Stata;
`XHDFE_ENABLE_CUDA=auto` for Python/R).

## Run

All three runners read the same spec table (`registry.json`), skip datasets
that are not present, and append to `output/*_runs.csv` (timings) and
`output/*_coefficients.csv` (point estimates + SEs, so numerical agreement can
be checked across engines):

```bash
# Python
REPS=3 python3 run_python.py
GROUPS=sergio MODE=fast python3 run_python.py     # subsets / fast mode
GPU=1 python3 run_python.py                       # CUDA (fail-closed)
OTHERS=1 python3 run_python.py                    # also time pyfixest

# R
REPS=3 Rscript run_r.R                            # OTHERS=1 also times fixest

# Stata (from this directory)
stata-mp -b do run_stata.do                       # OTHERS=1 also times reghdfe
```

Knobs (env vars, same across runners): `SPECS` (comma list), `GROUPS`
(sergio|simulated|pyfixest; Python/R), `REPS`, `MODE=comparable|fast`,
`THREADS`, `GPU=1`, `OTHERS=1`.

## Methodology notes

- `MODE=comparable` (default) is xhdfe's default `reghdfe-comparable`
  tolerance mode — coefficients match `reghdfe` at the same nominal tolerance.
  `MODE=fast` is the speed-oriented `xhdfe-fast` mode used in some published
  tables (always labelled as such).
- Timings are wall-clock per estimator call. On shared machines, interleave
  A/B comparisons and use medians over `REPS>=3`; absolute numbers move with
  load, ratios are more stable.
- `synthetic-zigzag` is a saturated design (perfect fit, zero residual df):
  xhdfe, pyfixest and FixedEffectModels.jl agree on it; reghdfe under-converges
  there by default. See the repository docs for the analysis.
- Published xhdfe benchmark tables (e.g. the AKM table in the repository
  README) come from the full core-23 including the proprietary specs; this
  directory replicates the 20 public ones with identical specifications.

## Credits

- **Sergio Correia** — the HDFE benchmark dataset collection, and `reghdfe`,
  whose conventions xhdfe follows.
- **pyfixest** (py-econometrics; Alexander Fischer and contributors) and
  **Kyle Butts** (`fixest_benchmarks`) — the simple/difficult benchmark DGP.
- SNAP (Stanford Network Analysis Project) and Google BigQuery public
  datasets — upstream sources of `enron`, `patents`, and `github`.
