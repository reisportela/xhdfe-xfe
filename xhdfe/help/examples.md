# xhdfe examples with public data

Package documentation version: 2.35.0.20261125.

Run each block independently. The formula examples require the `formula`
extra. The Grunfeld blocks use the public statsmodels edition.

## Absorption, clustered inference and saved effects

```python
import pandas as pd
import xhdfe

data = pd.read_csv(
    "https://raw.githubusercontent.com/statsmodels/statsmodels/main/"
    "statsmodels/datasets/grunfeld/grunfeld.csv"
)
fit = xhdfe.feols(
    "invest ~ value + capital | firm + year", data,
    clusters=["firm", "year"], retain_fes=True,
)
print(fit.tidy())
print(fit.nobs_, len(fit.residuals_), len(fit.fe_effects_))
```

## Analytic, frequency, importance and probability weights

The weights in this block illustrate the four inference conventions.

```python
import pandas as pd
import xhdfe

data = pd.read_csv(
    "https://raw.githubusercontent.com/statsmodels/statsmodels/main/"
    "statsmodels/datasets/grunfeld/grunfeld.csv"
)
data["w"] = 1 + data["capital"] / 100
data["copies"] = 1 + (data["year"] >= 1945).astype(int)
formula = "invest ~ value + capital | firm + year"
analytic = xhdfe.feols(formula, data, weights="w")
frequency = xhdfe.feols(formula, data, weights="copies", fweights=True)
importance = xhdfe.feols(formula, data, weights="w", iweights=True)
probability = xhdfe.feols(formula, data, weights="w", pweights=True)
print(analytic.nobs_, frequency.nobs_, importance.nobs_, probability.nobs_)
```

## Heterogeneous slopes

Give each firm an intercept and a slope on market value. The coefficient on
capital remains common.

```python
import pandas as pd
import xhdfe

data = pd.read_csv(
    "https://raw.githubusercontent.com/statsmodels/statsmodels/main/"
    "statsmodels/datasets/grunfeld/grunfeld.csv"
)
firm = pd.factorize(data["firm"])[0]
fit = xhdfe.HdfeRegressor(se_type="cluster")
fit.fit(
    data["invest"].to_numpy(), data[["capital"]].to_numpy(),
    fes=[firm, data["year"].to_numpy()], clusters=firm,
    slopes=[{"fe_index": 0, "values": data["value"].to_numpy(),
             "include_intercept": True}],
)
print(fit.coef_, fit.stderr_)
```

## IV/2SLS

The Card data illustrate instrumenting education with proximity to a
four-year college, with a regional effect and robust inference.

```python
import pandas as pd
import xhdfe

data = pd.read_csv(
    "https://raw.githubusercontent.com/bashtage/linearmodels/main/"
    "linearmodels/datasets/card/card.csv.bz2",
    compression="bz2",
)
fit = xhdfe.feols(
    "lwage ~ exper + expersq | south66 | educ ~ nearc4",
    data, se_type="robust",
)
print(fit.tidy())
```

## Group outcomes with individual effects

Construct firm-year means on the public worker–firm simulation. Each worker
contributes one membership row to a firm-year. Use `aggregation="sum"` for
the sum of individual effects instead of their mean.

```python
import pandas as pd
import xhdfe

data = pd.read_stata(
    "https://raw.githubusercontent.com/reisportela/xhdfe-xfe/main/"
    "data/felsdvsimul.dta",
    convert_categoricals=False,
)
groups = data.groupby(["j", "t"], sort=False)
data["firm_year"] = groups.ngroup()
data["group_y"] = groups["y"].transform("mean")
data["group_x"] = groups["x1"].transform("mean")
fit = xhdfe.feols(
    "group_y ~ group_x | i", data,
    group="firm_year", individual="i", aggregation="mean",
    dofadjustments="exact",
)
print(fit.tidy())
```

## Connected sample and AKM/KSS

```python
import pandas as pd
from xhdfe import akm

data = pd.read_stata(
    "https://raw.githubusercontent.com/reisportela/xhdfe-xfe/main/"
    "data/felsdvsimul.dta",
    convert_categoricals=False,
)
sample = akm.leave_out_set(data["i"], data["j"])
kept = data.loc[sample["keep"]]
fit = akm.akm_kss(kept["y"], kept["i"], kept["j"], prune=False, leverages="exact")
print(sample["n_obs"], fit["kss"]["var_psi"])
```

## Gelbach decomposition, table and waterfall data

Keep year effects in both models and account for the change in the market
value coefficient when capital and firm effects enter the model.

```python
import pandas as pd
from xhdfe import gelbach

data = pd.read_csv(
    "https://raw.githubusercontent.com/statsmodels/statsmodels/main/"
    "statsmodels/datasets/grunfeld/grunfeld.csv"
)
firm = pd.factorize(data["firm"])[0]
fit = gelbach.decompose(
    data["invest"], data[["value"]],
    x2_groups={"capital": data[["capital"]]},
    common_fes={"year": data["year"]}, fes={"firm": firm},
    vce="cluster", cluster=firm, x1_names=["value"], focal="value",
)
rows = gelbach.tidy(fit, share="movement")
capital_movement = gelbach.contrast(fit, "value", ["capital"])
print(gelbach.etable(fit, focal="value", format="markdown"))
waterfall = gelbach.waterfall_data(fit, focal="value")
```

## Gelbach with an absorbed target

Firm effects absorb the US Steel indicator. The full model imposes its
coefficient at zero. Use this fit for the point decomposition.

```python
import pandas as pd
from xhdfe import gelbach

data = pd.read_csv(
    "https://raw.githubusercontent.com/statsmodels/statsmodels/main/"
    "statsmodels/datasets/grunfeld/grunfeld.csv"
)
data["steel"] = (data["firm"] == "US Steel").astype(int)
firm = pd.factorize(data["firm"])[0]
fit = gelbach.decompose(
    data["invest"], data[["steel", "value"]],
    x2_groups={"capital": data[["capital"]]},
    fes={"firm": firm, "year": data["year"]},
    x1_names=["steel", "value"], absorbed_targets="steel", focal="steel",
)
assert fit["b_full_status"][0] == "imposed_zero"
print(fit["b_base"][0], fit["b_full"][0])
```

## Gelbach cluster-pairs bootstrap

The 39 replications illustrate the API. The resampling unit is the firm.

```python
import pandas as pd
from xhdfe import gelbach

data = pd.read_csv(
    "https://raw.githubusercontent.com/statsmodels/statsmodels/main/"
    "statsmodels/datasets/grunfeld/grunfeld.csv"
)
firm = pd.factorize(data["firm"])[0]
fit = gelbach.bootstrap(
    data["invest"], data[["value"]],
    x2_groups={"capital": data[["capital"]]},
    common_fes={"year": data["year"]}, fes={"firm": firm},
    x1_names=["value"], method="cluster_pairs", bootstrap_cluster=firm,
    reps=39, seed=42,
)
print(gelbach.etable(fit, panels="levels", format="markdown", interval="bootstrap"))
```
