# xhdfe worked examples — AKM / KSS and Gelbach

Self-contained, reproducible example scripts for the worker-firm (AKM) and
Gelbach post-estimation layer, in all three front-ends. Each script generates
its own synthetic data (fixed seed) and prints the key results, so it runs
anywhere the package is installed — no data files needed.

| Topic | Stata | Python | R |
|---|---|---|---|
| AKM + leave-out (KSS) | [`akm_kss_example.do`](akm_kss_example.do) | [`akm_kss_example.py`](akm_kss_example.py) | [`akm_kss_example.R`](akm_kss_example.R) |
| Gelbach decomposition | [`gelbach_example.do`](gelbach_example.do) | [`gelbach_example.py`](gelbach_example.py) | [`gelbach_example.R`](gelbach_example.R) |
| Gelbach absorbed target | [`gelbach_absorbed_target.do`](gelbach_absorbed_target.do) | [`gelbach_absorbed_target.py`](gelbach_absorbed_target.py) | [`gelbach_absorbed_target.R`](gelbach_absorbed_target.R) |

## What they show

**AKM + leave-out (KSS)** — building the leave-one-out connected set
(`xhdfeconnected` / `leave_out_set`), the two-way variance decomposition in
three flavours (plug-in, AGSU homoskedastic, KSS heteroskedastic leave-out),
the derived summaries (correlation of worker and firm effects, shares of
wage variance), component standard errors and Andrews-Mikusheva
weak-identification confidence intervals, controls partialled out by FWL, the
JLA leverage approximation, and (Python) the Andrews-Gill-Schank-Upward
subsampling diagnostic and export to the `pytwoway` / LeaveOutTwoWay format.

**Gelbach** — decomposing the movement of a focal base coefficient between a
short and a long regression into an additive, order-invariant contribution per
declared block, while retaining common controls in both models. The examples
cover unadjusted and one-way cluster-robust inference, observed and absorbed-FE
blocks, focal reporting, signed shares, covariance-aware contrasts, and the
distinct constrained estimand for a target absorbed by an added FE. The latter
prints and stores its full-model zero as imposed, not estimated.

## Running them

Stata (from the repository root, with the plugin built or installed):

```stata
adopath ++ "`c(pwd)'/stata"
do examples/akm_kss_example.do
do examples/gelbach_example.do
do examples/gelbach_absorbed_target.do
```

Python (with the compiled extension installed, `pip install .`):

```bash
python examples/akm_kss_example.py
python examples/gelbach_example.py
python examples/gelbach_absorbed_target.py
```

R (with the installed `xhdfe` package):

```bash
Rscript examples/akm_kss_example.R
Rscript examples/gelbach_example.R
Rscript examples/gelbach_absorbed_target.R
```

For the GPU (CUDA) path, add `gpu` (Stata) / `gpu=True` (Python) / `gpu=TRUE`
(R) to the AKM calls when the plugin/extension was built with CUDA support.
In Stata, `xhdfegelbach, gpu verbose` accelerates the full HDFE absorption and
shows phase progress; `xhdfeconnected, gpu verbose` uses deterministic CUDA
radix sorting on graphs at or above its measured 10-million-row profitability
gate and reports the effective backend in `r(gpu_status)`.

## Background and validation

The numerical semantics follow Kline, Saggio and Solvsten (2020) as
implemented in Saggio's LeaveOutTwoWay (the canonical KSS reference), and the
Gelbach decomposition follows Gelbach (2016) / his `b1x2`. See
[`docs/akm-kss.md`](../docs/akm-kss.md) for a felsdvsimul walkthrough and the
package help (`help xhdfeakm`, `help xhdfeconnected`, `help xhdfegelbach`;
`?xhdfe_akm_kss`, `?xhdfe_gelbach`, `?xhdfe_gelbach_tidy`, and
`?xhdfe_gelbach_contrast` in R; `help(xhdfe.akm.akm_kss)` and
`xhdfe-help gelbach` in Python) for the full option, result, inference, and
limitation reference.
