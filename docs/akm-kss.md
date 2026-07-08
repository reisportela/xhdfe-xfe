# Worked example — AKM + leave-out (KSS) decomposition and the Gelbach companion

Reproduces a known decomposition end-to-end on the classic `felsdvreg`
simulation shipped with the repository (`data/felsdvsimul.dta`, 100
person-year rows, 20 workers, 15 firms). The exact-path numbers below are
validated at machine precision against Saggio's LeaveOutTwoWay (MATLAB) and
an independent dense oracle (validation record in the development repository).

## Python

```python
import pandas as pd, numpy as np, xhdfe.akm as akm

df = pd.read_stata("data/felsdvsimul.dta")
y = df["y"].to_numpy(float)
w = df["i"].to_numpy(np.int64)
f = df["j"].to_numpy(np.int64)

r = akm.akm_kss(y, w, f, leverages="exact")   # leave-a-match-out (default)
print(r["kss"])
# {'var_alpha': -6.2429443959, 'var_psi': 6.2254112695, 'cov_alpha_psi': 2.6372007926}
# plug-in var(psi) = 34.398203 -> the KSS correction removes the
# limited-mobility bias (2 movers, 15-obs leave-out sample).

# JLA path (large samples): deterministic for any thread count
r = akm.akm_kss(y, w, f, leverages="jla", jla_draws=150, seed=42)
# r['kss']['var_psi'] == 6.13308521068209 in Python, R and Stata alike.

# GPU (CUDA) solves for the JLA/SE machinery — 11.4x at 2M rows on an H100:
r = akm.akm_kss(y, w, f, leverages="jla", gpu=True)      # r['gpu_used']

# Component standard errors (KSS high-rank case; leave_out_COMPLETE
# machinery, validated against the MATLAB oracle):
r = akm.akm_kss(y, w, f, leverages="exact", compute_se=True)
r["component_se"]   # se_var_psi / se_cov_alpha_psi / se_var_alpha + theta_c

# KSS lincom (Proposition 1): projections of firm effects on observables
r = akm.akm_kss(y, w, f, Z=Zmat, leverages="exact")
r["lincom"]         # coef / se_kss / se_white / t

# Subsampling diagnostic (Andrews et al. 2012) and exports
rec = akm.subsampling_diagnostic(y, w, f, fractions=(0.0, 0.2, 0.4), seed=7)
akm.export_results(r, y, w, f, "akm_out", fmt="csv")
```

## Stata

```stata
use data/felsdvsimul, clear
xhdfeakm y, worker(i) firm(j) leverages(exact) generate(akm)
ret list          // r(kss_var_psi) = 6.2254112695 ...
```

## R

```r
library(xhdfe)
d <- foreign::read.dta("data/felsdvsimul.dta")
fit <- xhdfe_akm_kss(d$y, d$i, d$j, leverages = "exact")
print(fit)
```

## Gelbach conditional decomposition (M9B companion)

How much of the change in a coefficient between the base and the full
specification is explained by each covariate group or by an absorbed
fixed-effect block (e.g. the firm-sorting contribution to a gap):

```python
import xhdfe.gelbach as gb
res = gb.decompose(y, x1, x2_groups={"skills": Xs}, fes={"firm": firm_id},
                   vce="robust")           # or 'cluster', cluster=ids
# Stata-style weights (both validated against b1x2 exactly):
res = gb.decompose(y, x1, x2_groups={"skills": Xs}, weights=w)            # aweights
res = gb.decompose(y, x1, x2_groups={"skills": Xs}, weights=k, fweights=True)
res["delta"]["firm"]["coef"]   # contribution over [x1..., _cons]
res["total"]                    # equals b_base - b_full exactly
```

Point estimates are linear functionals of the fixed effects — no KSS-style
correction applies. Inference reproduces Gelbach's b1x2 exactly
(homoskedastic, robust and cluster, aweights and fweights;
`gamma0`/`cov0` options); absorbed FE blocks use the gamma0 treatment.
Stata: `xhdfegelbach y [aw=w], x1(...) x2groups("name = vars : ...") fes(...)`;
R: `xhdfe_gelbach(...)`. Validated against Gelbach's `b1x2` (44 checks at
machine precision).

## References and acknowledgements

The worker-firm leave-out layer follows and is validated at machine precision
against [`LeaveOutTwoWay`](https://github.com/rsaggio87/LeaveOutTwoWay) by
Raffaele Saggio — the canonical KSS implementation — and interoperates with
and is cross-checked against
[`pytwoway`](https://github.com/tlamadon/pytwoway) by Thibaut Lamadon and
Adam A. Oppenheimer; the Gelbach companion is validated against Jonah Gelbach's
`b1x2`. Full credit to their authors.

- Abowd, Kramarz & Margolis (1999), *Econometrica* 67(2): 251-333 — the AKM
  two-way model.
- Andrews, Gill, Schank & Upward (2008), *JRSS-A* 171(3): 673-697 — the AGSU
  homoskedastic correction.
- Kline, Saggio & Sølvsten (2020), *Econometrica* 88(5): 1859-1898 — the KSS
  leave-out heteroskedasticity-robust correction and inference.
- Andrews & Mikusheva (2016), *Econometrica* 84(3): 1249-1264 — the
  weak-identification q=1 confidence intervals.
- Gelbach (2016), *Journal of Labor Economics* 34(2): 509-543 — the
  conditional decomposition of coefficient movements.

