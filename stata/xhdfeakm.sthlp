{smcl}
{* *! version 2.14.2  09jul2026}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{vieweralsosee "xhdfeconnected" "help xhdfeconnected"}{...}
{vieweralsosee "xhdfegelbach" "help xhdfegelbach"}{...}
{title:Title}

{p2colset 5 20 22 2}{...}
{p2col :{cmd:xhdfeakm} {hline 2}}AKM worker-firm estimation with leave-out (KSS) variance decomposition{p_end}
{p2colreset}{...}


{title:Syntax}

{p 8 15 2}
{cmd:xhdfeakm} {depvar} {ifin} {weight}{cmd:,}
{opth work:er(varname)} {opth firm(varname)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Model}
{synopt :{opth work:er(varname)}}worker (person) identifier, integer-valued; {bf:required}{p_end}
{synopt :{opth firm(varname)}}firm (establishment) identifier, integer-valued; {bf:required}{p_end}
{synopt :{opth control:s(varlist)}}covariates partialled out (FWL) before the two-way step{p_end}
{synopt :{opt leaveout:level(unit)}}leave-out unit: {cmd:match} (default) or {cmd:obs}{p_end}
{synopt :{opt noprune}}skip the leave-out connected-set pruning{p_end}

{syntab:Leverages (Pii, Bii)}
{synopt :{opt lev:erages(method)}}{cmd:auto} (default), {cmd:exact}, or {cmd:jla}{p_end}
{synopt :{opt draws(#)}}JLA random projections (default 200){p_end}
{synopt :{opt seed(#)}}seed for the JLA draws (default 20260705){p_end}
{synopt :{opt exactmax:rows(#)}}row cap for {cmd:auto} to pick the exact path (default 10000){p_end}

{syntab:Inference}
{synopt :{opt se}}component standard errors (KSS leave-out inference){p_end}
{synopt :{opt sensim(#)}}simulation draws for the SE quadratic part (default 1000){p_end}
{synopt :{opt ci}}weak-id diagnostics + Andrews-Mikusheva q=1 CIs; implies {opt se}{p_end}
{synopt :{opt eigtracensim(#)}}Hutchinson draws for tr(Atilde^2) (default 100){p_end}
{synopt :{opt sigmalow:ess}}lowess sigma-tilde fit instead of the binned default{p_end}

{syntab:Solver / technical}
{synopt :{opt gpu}}solve the two-way systems on the CUDA backend{p_end}
{synopt :{opt threads(#)}}OpenMP threads (0 = library default){p_end}
{synopt :{opt directmax:firms(#)}}firm cap for the direct sparse solve (default 50000){p_end}
{synopt :{opt cgtol(#)}}two-way solver tolerance (default 1e-10){p_end}
{synopt :{opt fwltol(#)}}absorber tolerance for the control step (default 1e-10){p_end}

{syntab:Output}
{synopt :{opth gen:erate(stub)}}save {it:stub}{cmd:_alpha}, {it:stub}{cmd:_psi}, {it:stub}{cmd:_keep}{p_end}
{synopt :{opt replace}}overwrite existing {opt generate()} variables{p_end}
{synoptline}
{p2colreset}{...}
{p 4 6 2}{cmd:fweight}s are allowed (match-level point decomposition only; a
weighted run equals the row-expanded run exactly).{p_end}


{title:Description}

{pstd}
{cmd:xhdfeakm} estimates the two-way AKM model

{p 8 8 2}{it:y_it = alpha_i + psi_j(i,t) + X_it' beta + e_it}{p_end}

{pstd}
(worker effect {it:alpha}, firm effect {it:psi}) on the largest leave-one-out
connected set and reports the wage-variance decomposition in three flavours:
{it:plug-in} (the biased sample decomposition), {it:AGSU} (Andrews, Gill,
Schank and Upward 2008, homoskedastic bias correction) and {it:KSS} (Kline,
Saggio and Soelvsten 2020, heteroskedasticity-robust leave-out correction).
Numerical semantics follow Saggio's {it:LeaveOutTwoWay} (the canonical KSS
implementation this command is validated against): leave-a-match-out by
default, person-year weighted components with denominator {it:n}-1, exact
leverages on small samples and the Johnson-Lindenstrauss (JLA) approximation
with deterministic per-draw seeding on large ones.

{pstd}
The command shares the compiled xhdfe backend with the Python
{cmd:xhdfe.akm.akm_kss} and R {cmd:xhdfe_akm_kss} front-ends and requires the
{cmd:xhdfe.plugin} shipped with this version. It is a self-contained
post-estimation-style command: it does not change any {helpb xhdfe}
estimation path. To only build the leave-out sample without estimating, use
{helpb xhdfeconnected}.

{pstd}
{cmd:xhdfe} is, by design, primarily a high-performance replica of
{helpb reghdfe}; from {cmd:pytwoway} (Lamadon and Oppenheimer's reference
Python toolkit for two-way worker-firm models) and the related AKM literature
it adopts {it:only} what adds value inside that reghdfe universe -- this
leave-out decomposition, the connected set and the Gelbach companion -- rather
than reproducing pytwoway's broader structural models (CRE, BLM). The layer is
validated at machine precision against both {cmd:pytwoway} and Saggio's
{cmd:LeaveOutTwoWay}, and interoperates with pytwoway (the Python front-end can
export the leave-out sample to the pytwoway / bipartitepandas format). When
comparing against pytwoway directly, note two deliberate convention choices
(both follow the canonical {cmd:LeaveOutTwoWay}): the stayer variance rule
matches pytwoway's {it:non-default} {cmd:Sii_stayers='upper_bound'} option
(pytwoway's default {cmd:'firm_mean'} imputes stayer variances from firm-level
mover averages and can differ by 1-2 percent in var(alpha)/cov on
stayer-heavy panels — a convention gap, not a discrepancy), and the plug-in
components use the 1/(n-1) denominator versus pytwoway's 1/n (negligible
above ~10,000 person-years). Avoid pytwoway's {cmd:exact_trace_ho/he=True}
paths as an oracle: their var(alpha) quadrant is broken in pytwoway 0.3.21
(negative variances observed; audited 09jul2026). The
combination is most useful with large linked employer-employee data: do the
fast HDFE regression and the leave-out variance decomposition here in a
familiar reghdfe workflow, and use pytwoway for the structural models outside
{cmd:xhdfe}'s scope.


{title:Options}

{dlgtab:Model}

{phang}{opth worker(varname)} and {opth firm(varname)} are the two
high-dimensional identifiers (integer-valued). Both are {bf:required}.

{phang}{opth controls(varlist)} lists covariates that are partialled out at the
person-year level with the xhdfe absorber (Frisch-Waugh-Lovell) before the
two-way machinery runs; their coefficients are returned in {cmd:e()}/{cmd:r(b)}.
Factor-variable notation is allowed, so {cmd:controls(i.year)} works directly
(base level omitted, as with manual dummies) instead of expanding the dummies by
hand; {cmd:i.}, {cmd:c.}, interactions and a chosen base ({cmd:ib#.}) are all
supported.

{phang}{opt leaveoutlevel(match|obs)} chooses the leave-out unit:
{cmd:match} (worker-firm pair; the default and Saggio's default) or
{cmd:obs} (a single person-year observation). At {cmd:match} level with
stayers, {it:var(alpha)} inference is not point-identified and its SE/CI are
returned as missing (the oracle rule); use {cmd:obs} to identify it.

{phang}{opt noprune} skips the leave-out connected-set computation; use only
when the estimation sample is already leave-out connected (for example after
{helpb xhdfeconnected}).

{dlgtab:Leverages}

{phang}{opt leverages(auto|exact|jla)} selects the statistical leverages
{it:P_ii} and quadratic-form weights {it:B_ii}. {cmd:auto} (default) uses the
{cmd:exact} per-observation solve when the input has at most
{opt exactmaxrows()} rows and the JLA approximation otherwise; {cmd:exact}
and {cmd:jla} force either path.

{phang}{opt draws(#)} sets the number of JLA random projections (default 200,
the LeaveOutTwoWay default). {opt seed(#)} seeds them; results are
bit-reproducible for {it:any} thread count and backend.

{phang}{opt exactmaxrows(#)} is the row cap under which {cmd:auto} picks the
exact path (default 10000, the LeaveOutTwoWay rule).

{dlgtab:Inference}

{phang}{opt se} computes component standard errors by KSS leave-out inference
(person-year block leave-out with a binned nonparametric sigma-tilde fit and a
simulated quadratic term). {opt sensim(#)} sets the number of simulation draws
for the quadratic part (default 1000).

{phang}{opt ci} adds weak-identification diagnostics and the Andrews-Mikusheva
q=1 confidence intervals for each component (the LeaveOutTwoWay
{it:eigen_diagno} path): the top eigenvalue of Atilde and its share of the
sum of squared eigenvalues, the weak-Lindeberg condition, gamma-squared, the
F statistic, the curvature-adjusted point estimate {it:theta_1}, the AM
confidence bounds and the curvature. Implies {opt se}. {opt eigtracensim(#)}
sets the Hutchinson draws for tr(Atilde^2) (default 100, the oracle
default).

{phang}{opt sigmalowess} uses the LeaveOutTwoWay mode-0 lowess surface fit of
sigma-i on ({it:P_ii}, {it:B_ii}) for the SE quadratic part instead of the
binned default. Intended for small/medium-sample sensitivity analysis
(O(n^2)); the binned fit is the validated default.

{dlgtab:Solver / technical}

{phang}{opt gpu} solves the two-way linear systems on the CUDA backend when the
plugin was built with CUDA support; check {cmd:r(gpu_used)}==1. The CPU path is
the numerical reference and the results are identical to solver tolerance.

{phang}{opt threads(#)} sets OpenMP threads (0 = library default).

{phang}{opt directmaxfirms(#)} caps the firm dimension for the direct sparse
Cholesky solve (default 50000); larger problems use preconditioned CG.

{phang}{opt cgtol(#)} is the two-way solver tolerance (default 1e-10);
{opt fwltol(#)} is the absorber tolerance for the {opt controls()} step
(default 1e-10).

{phang}The leverage (JLA) solves are batched: each block of Rademacher draws
is solved as one multi-right-hand-side system, with results identical for any
block size and any thread count. The advanced environment variable
{cmd:XHDFE_AKM_JLA_BLOCK} overrides the block size (default 8; {cmd:0}
selects the pre-2.14 sequential solver, whose per-edge instruction schedule
differs from the batched kernels at the last-ulp level).

{dlgtab:Output}

{phang}{opth generate(stub)} stores {it:stub}{cmd:_alpha} and
{it:stub}{cmd:_psi} (observation-level worker and firm effects on the leave-out
sample; {it:psi} normalized to a zero person-year mean) and
{it:stub}{cmd:_keep} (1 for rows in the leave-out sample). {opt replace}
overwrites existing such variables.


{title:Stored results}

{pstd}{cmd:xhdfeakm} stores the following in {cmd:r()}:

{synoptset 26 tabbed}{...}
{p2col 5 26 30 2: Decomposition (each estimator {it:E} in {cmd:plugin}, {cmd:agsu}, {cmd:kss})}{p_end}
{synopt:{cmd:r(}{it:E}{cmd:_var_alpha)}}variance of worker effects{p_end}
{synopt:{cmd:r(}{it:E}{cmd:_var_psi)}}variance of firm effects{p_end}
{synopt:{cmd:r(}{it:E}{cmd:_cov)}}covariance of worker and firm effects{p_end}
{synopt:{cmd:r(}{it:E}{cmd:_corr)}}correlation of worker and firm effects{p_end}
{synopt:{cmd:r(}{it:E}{cmd:_var_sum)}}variance of {it:alpha}+{it:psi}{p_end}
{synopt:{cmd:r(}{it:E}{cmd:_share_alpha)}}var(alpha) as a share of var(y){p_end}
{synopt:{cmd:r(}{it:E}{cmd:_share_psi)}}var(psi) as a share of var(y){p_end}
{synopt:{cmd:r(}{it:E}{cmd:_share_2cov)}}2*cov as a share of var(y){p_end}

{p2col 5 26 30 2: Sample and diagnostics}{p_end}
{synopt:{cmd:r(var_y)}}person-year variance of {it:y}{p_end}
{synopt:{cmd:r(sigma2_ho)}}homoskedastic error variance (AGSU){p_end}
{synopt:{cmd:r(n_obs)}}person-year observations in the leave-out set{p_end}
{synopt:{cmd:r(n_obs_input)}}input observations{p_end}
{synopt:{cmd:r(n_obs_connected)}}observations after the largest-CC step{p_end}
{synopt:{cmd:r(n_workers)} {cmd:r(n_firms)} {cmd:r(n_matches)}}sample sizes{p_end}
{synopt:{cmd:r(n_movers)} {cmd:r(n_stayers)}}worker mobility counts{p_end}
{synopt:{cmd:r(n_rows)}}working rows (matches or observations){p_end}
{synopt:{cmd:r(max_pii)} {cmd:r(mean_pii)}}leverage diagnostics{p_end}
{synopt:{cmd:r(leverages_exact)}}1 if the exact leverage path was used{p_end}
{synopt:{cmd:r(solver_direct)}}1 if the direct sparse solve was used{p_end}
{synopt:{cmd:r(jla_draws)} {cmd:r(seed)}}JLA draws and seed actually used{p_end}
{synopt:{cmd:r(solver_iterations)} {cmd:r(converged)}}solver diagnostics{p_end}
{synopt:{cmd:r(gpu_used)}}1 if the CUDA backend solved the systems{p_end}

{p2col 5 26 30 2: With {opt se} (component {it:C} in {cmd:var_psi}, {cmd:cov}, {cmd:var_alpha})}{p_end}
{synopt:{cmd:r(se_}{it:C}{cmd:)}}standard error of the KSS component{p_end}
{synopt:{cmd:r(theta_}{it:C}{cmd:)}}the KSS-corrected point estimate{p_end}

{p2col 5 26 30 2: With {opt ci} (component suffix {it:c} in {cmd:_psi}, {cmd:_cov}, {cmd:_alpha})}{p_end}
{synopt:{cmd:r(lambda1_}{it:c}{cmd:)}}top eigenvalue of Atilde{p_end}
{synopt:{cmd:r(eigshare1_}{it:c}{cmd:)}}its share of the sum of squared eigenvalues{p_end}
{synopt:{cmd:r(lindeberg_}{it:c}{cmd:)}}weak-Lindeberg condition{p_end}
{synopt:{cmd:r(gammasq_}{it:c}{cmd:)}}gamma-squared{p_end}
{synopt:{cmd:r(fstat_}{it:c}{cmd:)}}F statistic{p_end}
{synopt:{cmd:r(theta1_}{it:c}{cmd:)}}curvature-adjusted point estimate{p_end}
{synopt:{cmd:r(ci_lb_}{it:c}{cmd:)} {cmd:r(ci_ub_}{it:c}{cmd:)}}AM q=1 confidence bounds{p_end}
{synopt:{cmd:r(curvature_}{it:c}{cmd:)}}curvature{p_end}

{p2col 5 26 30 2: Macros / matrices}{p_end}
{synopt:{cmd:r(cmd)}}{cmd:xhdfeakm}{p_end}
{synopt:{cmd:r(leave_out_level)}}the leave-out unit used{p_end}
{synopt:{cmd:r(notes)}}any solver/identification notes{p_end}
{synopt:{cmd:r(b)}}control coefficients (with {opt controls()}){p_end}
{p2colreset}{...}


{title:Examples}

{pstd}With a worker-firm panel loaded ({it:y} = log wage, {it:i} = worker,
{it:j} = firm), run the default (leave-a-match-out, exact on a small sample):{p_end}
{phang2}{cmd:. xhdfeakm y, worker(i) firm(j)}{p_end}

{pstd}KSS standard errors and Andrews-Mikusheva confidence intervals, with the
worker and firm effects saved:{p_end}
{phang2}{cmd:. xhdfeakm y, worker(i) firm(j) ci generate(akm)}{p_end}
{phang2}{cmd:. display r(kss_var_psi), r(se_var_psi), r(ci_lb_psi), r(ci_ub_psi)}{p_end}

{pstd}Partial out controls, use the JLA leverages on the GPU, and read the
sorting correlation:{p_end}
{phang2}{cmd:. xhdfeakm y, worker(i) firm(j) controls(x1 x2) leverages(jla) gpu}{p_end}
{phang2}{cmd:. display "corr(a,psi) = " r(kss_corr)}{p_end}

{pstd}A fuller walkthrough (Stata/Python/R) ships as
{bf:examples/akm_kss_example.do} and {bf:docs/akm-kss.md}.{p_end}


{title:References}

{pstd}
Abowd, J., F. Kramarz, and D. Margolis. 1999. High wage workers and high wage
firms. {it:Econometrica} 67(2): 251-333.{p_end}
{pstd}
Andrews, M., L. Gill, T. Schank, and R. Upward. 2008. High wage workers and
low wage firms: negative assortative matching or limited mobility bias?
{it:JRSS-A} 171(3): 673-697.{p_end}
{pstd}
Kline, P., R. Saggio, and M. Soelvsten. 2020. Leave-out estimation of variance
components. {it:Econometrica} 88(5): 1859-1898.{p_end}


{title:Also see}

{psee}
{helpb xhdfe}, {helpb xhdfeconnected}, {helpb xhdfegelbach}
{p_end}
