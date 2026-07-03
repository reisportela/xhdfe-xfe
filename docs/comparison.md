# How xhdfe relates to reghdfe, fixest, pyfixest, and FixedEffectModels.jl

`xhdfe` is a reimplementation of high-dimensional fixed-effect (HDFE) estimation
built to **validate against and interoperate with** the established packages in
the ecosystem:

- [reghdfe](https://github.com/sergiocorreia/reghdfe) (Stata, Correia 2016),
- [fixest](https://github.com/lrberge/fixest) (R),
- [pyfixest](https://github.com/py-econometrics/pyfixest) (Python),
- [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl)
  (Julia).

By design, the default output and defaults mirror `reghdfe`: under the
`reghdfe-comparable` tolerance mode (the default), `xhdfe` coefficients match
`reghdfe` at the same nominal tolerance, down to the conditioning of the
problem. `xhdfe` additionally exposes `fixest`/`pyfixest`-style small-sample
corrections (`ssc()` in Stata, `ssc`/`stats_style` in R, `ssc_*` in Python), so
you can also reproduce `fixest`/`pyfixest` conventions when you want them.
`xhdfe` is a distinct implementation; full credit for the estimators and
conventions it interoperates with goes to the authors of the packages above.

## Command / option map — Stata (`xhdfe` vs `reghdfe`)

`xhdfe`'s Stata syntax is deliberately close to `reghdfe`'s; most scripts port
by changing the command name. It also accepts several `reghdfe` options as
compatibility no-ops (e.g. `technique()`, `acceleration()`, `transform()`,
`preconditioner()`, `iterate()` as an alias of `maxiter()`).

| Task                         | reghdfe                                   | xhdfe (Stata)                             |
| ---------------------------- | ----------------------------------------- | ----------------------------------------- |
| Two-way FE                   | `reghdfe y x, absorb(id1 id2)`            | `xhdfe y x, absorb(id1 id2)`              |
| Robust SE                    | `..., vce(robust)`                        | `..., vce(robust)`                        |
| One-way cluster              | `..., vce(cluster id)`                    | `..., vce(cluster id)`                    |
| Multiway cluster             | `..., vce(cluster id1 id2)`               | `..., vce(cluster id1 id2)`               |
| Interaction FE               | `absorb(id1#id2)`                         | `absorb(id1#id2)`                         |
| Heterogeneous slopes         | `absorb(id##c.x)` / `absorb(id#c.x)`      | `absorb(id##c.x)` / `absorb(id#c.x)`      |
| IV / 2SLS                    | `ivreghdfe y (endo = inst), absorb(...)`  | `xhdfe y, absorb(...) endogenous(endo) instruments(inst)` |
| Save fixed effects           | `absorb(..., savefe)` / `savefe`          | `absorb(..., savefe)` / `savefes(prefix)` |
| Save residuals               | `residuals(u)`                            | `residuals(u)`                            |
| Keep singletons              | `keepsingletons`                          | `keepsingletons`                          |
| FE DoF adjustments           | `dofadjustments(...)`                     | `dofadjustments(...)`                     |
| Tolerance / max iterations   | `tolerance(#)` / `maxiter(#)`             | `tolerance(#)` / `maxiter(#)` (`iterate(#)`) |
| Group-level + individual FEs | `group()` / `individual()`                | `group()` / `individual()`                |

## Command / option map — R and Python (`xhdfe` vs `fixest` / `pyfixest`)

The R front end uses the `fixest` formula grammar (`y ~ x | fe1 + fe2`,
`fe[slope]`, `fe[[slope]]`, `f1^f2`, and `| endo ~ inst` for IV). The Python
front end (`xhdfe.HdfeRegressor(...).fit(...)`) takes arrays rather than a
formula.

| Task                 | fixest (R)                          | pyfixest (Python)                       | xhdfe (R)                                | xhdfe (Python)                                            |
| -------------------- | ----------------------------------- | --------------------------------------- | ---------------------------------------- | -------------------------------------------------------- |
| Two-way FE           | `feols(y ~ x \| f1 + f2, data)`     | `feols("y ~ x \| f1 + f2", data)`       | `xhdfe(y ~ x \| f1 + f2, data)`          | `HdfeRegressor().fit(y, X, fes=[f1, f2])`                 |
| Robust SE            | `vcov = "hetero"`                   | `vcov = "hetero"`                       | `vcov = "robust"`                        | `HdfeRegressor(se_type="robust")`                        |
| Cluster SE           | `cluster = ~ f1` / `vcov = ~ f1`    | `vcov = {"CRV1": "f1"}`                 | `cluster = ~ f1`                         | `HdfeRegressor(se_type="cluster")` + `clusters=[f1]`     |
| Multiway cluster     | `cluster = ~ f1 + f2`               | `vcov = {"CRV1": "f1 + f2"}`            | `cluster = ~ f1 + f2`                    | `clusters=[f1, f2]`                                       |
| Heterogeneous slopes | `y ~ x \| f1[s]`                    | `y ~ x \| f1[s]`                        | `y ~ x \| f1[s]` (slopes only: `f1[[s]]`)| `slopes=[{"fe_index": j, "values": s, "include_intercept": True}]` |
| Interaction FE       | `y ~ x \| f1^f2`                    | `y ~ x \| f1^f2`                        | `y ~ x \| f1^f2`                         | pre-build the joint id and pass it in `fes`              |
| IV / 2SLS            | `y ~ x \| f1 \| endo ~ inst`        | `y ~ x \| f1 \| endo ~ inst`            | `y ~ x \| f1 \| endo ~ inst`             | `instruments=Z, endogenous_idx=[j]`                      |
| Weights              | `weights = ~ w`                     | `weights = "w"`                         | `weights = ~ w`                          | `weights=w`                                              |
| Recover fixed effects| `fixef(m)`                          | `m.fixef()`                             | `save_fe = TRUE` then `fixef(m)`         | `HdfeRegressor(retain_fes=True)` then `reg.fe_effects_`  |
| Small-sample control | `ssc(...)`                          | `ssc(...)`                              | `ssc = ...`, `stats_style = ...`         | `ssc_k_adj=`, `ssc_k_fixef=`, `ssc_g_df=`, ...           |

Notes:

- The R/Python `(Intercept)` is a `reghdfe`-compatible constant reported after
  the regressors; remove it with `0 +` in R or `fit_intercept=False` in Python
  (Stata `noconstant`).
- Like `reghdfe`, a categorical-only `f1##f2` / `f1^f2` term is treated as one
  joint fixed effect, **not** expanded into main effects plus interaction; list
  the components explicitly if all three are intended.
- `xhdfe` reports a numeric code for the absorption method actually used
  (`e(absorption_method_used)` in Stata, `absorption_method_used` in R/Python).

## Tolerance modes

The nominal tolerance (`tolerance()` / `tol`) only has a precise meaning
together with the tolerance mode. `xhdfe` exposes three, and records the one used
in `e(tolerance_mode)` (Stata) / the `tolerance_mode` field (R/Python):

- **`reghdfe-comparable`** (default since 2.7.0): the accelerated absorber stops
  when one full sweep moves the working data by less than the tolerance in
  relative norm — the same meaning `reghdfe` attaches to its tolerance — so
  coefficients match `reghdfe` at the same nominal tolerance, down to problem
  conditioning. This is the recommended mode for published results.
- **`xhdfe-fast`**: the pre-2.7.0 stopping rule. It typically uses fewer absorber
  iterations, but its *effective* precision is data-dependent and can be looser
  than the nominal tolerance on ill-conditioned (e.g. sparse bipartite) designs.
  Appropriate for exploration and speed benchmarking — state the mode when citing
  timings.
- **`strict-residual`**: a heavier audit mode that treats the final absolute
  maximum group-mean residual check as authoritative, may use additional
  iterations up to `maxiter`, and reports non-convergence if the check is not
  met. Not supported with heterogeneous slopes.

Under the default `reghdfe-comparable` mode, absorbers on ill-conditioned,
poorly connected multiway FE graphs are transparently handed off to a stable
per-column conjugate-gradient solver on the symmetric demeaning operator (CPU and
CUDA alike). Coefficients, standard errors, and default output are unchanged
(typically *tighter* agreement with `reghdfe`); well-conditioned problems are
numerically identical to previous versions.

## Validation

The R and Python packages share the compiled C++ core, so the R binding is
tested to reproduce the reference Python module across the full estimator
surface (coefficients, inference, iteration counts, degrees of freedom). The
suites also check exact agreement with `lm()` on dummy-expanded designs and with
`fixest::feols` (coefficients and cluster SEs under the matching small-sample
convention). The Stata command is likewise developed against `reghdfe` output.
See the per-package `tests/` and the R showcase in
`r/examples/xhdfe_r_showcase.qmd`.
