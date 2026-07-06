# xhdfe worked examples — AKM / KSS and Gelbach

Self-contained, reproducible example scripts for the worker-firm (AKM) and
Gelbach post-estimation layer, in all three front-ends. Each script generates
its own synthetic data (fixed seed) and prints the key results, so it runs
anywhere the package is installed — no data files needed.

| Topic | Stata | Python | R |
|---|---|---|---|
| AKM + leave-out (KSS) | [`akm_kss_example.do`](akm_kss_example.do) | [`akm_kss_example.py`](akm_kss_example.py) | [`akm_kss_example.R`](akm_kss_example.R) |
| Gelbach decomposition | [`gelbach_example.do`](gelbach_example.do) | [`gelbach_example.py`](gelbach_example.py) | [`gelbach_example.R`](gelbach_example.R) |

## What they show

**AKM + leave-out (KSS)** — building the leave-one-out connected set
(`xhdfeconnected` / `leave_out_set`), the two-way variance decomposition in
three flavours (plug-in, AGSU homoskedastic, KSS heteroskedastic leave-out),
the derived summaries (correlation of worker and firm effects, shares of
wage variance), component standard errors and Andrews-Mikusheva
weak-identification confidence intervals, controls partialled out by FWL, the
JLA leverage approximation, and (Python) the Andrews-Gill-Schank-Upward
subsampling diagnostic and export to the `pytwoway` / LeaveOutTwoWay format.

**Gelbach** — decomposing the movement of a base coefficient (a return to
education) between a short and a long regression into an additive,
order-invariant contribution per channel, with unadjusted and cluster-robust
inference.

## Running them

Stata (from the repository root, with the plugin built or installed):

```stata
adopath ++ "`c(pwd)'/stata"
do examples/akm_kss_example.do
do examples/gelbach_example.do
```

Python (with the compiled extension installed, `pip install .`):

```bash
python examples/akm_kss_example.py
python examples/gelbach_example.py
```

R (with the installed `xhdfe` package):

```bash
Rscript examples/akm_kss_example.R
Rscript examples/gelbach_example.R
```

For the GPU (CUDA) path, add `gpu` (Stata) / `gpu=True` (Python) / `gpu=TRUE`
(R) to the AKM calls when the plugin/extension was built with CUDA support.

## Background and validation

The numerical semantics follow Kline, Saggio and Solvsten (2020) as
implemented in Saggio's LeaveOutTwoWay (the canonical KSS reference), and the
Gelbach decomposition follows Gelbach (2016) / his `b1x2`. See
[`docs/akm-kss.md`](../docs/akm-kss.md) for a felsdvsimul walkthrough and the
package help (`help xhdfeakm`, `help xhdfeconnected`, `help xhdfegelbach`;
`?xhdfe_akm_kss` in R; `help(xhdfe.akm.akm_kss)` in Python) for the full
option and return reference.
