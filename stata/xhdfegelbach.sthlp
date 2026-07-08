{smcl}
{* *! version 2.13.2  08jul2026}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{vieweralsosee "xhdfeakm" "help xhdfeakm"}{...}
{title:Title}

{p2colset 5 22 24 2}{...}
{p2col :{cmd:xhdfegelbach} {hline 2}}Gelbach (2016) conditional decomposition of coefficient movements{p_end}
{p2colreset}{...}


{title:Syntax}

{p 8 15 2}
{cmd:xhdfegelbach} {depvar} {ifin} {weight}{cmd:,}
{opt x1(varlist)}
[{it:options}]

{synoptset 26 tabbed}{...}
{synopthdr}
{synoptline}
{synopt :{opt x1(varlist)}}base covariates whose coefficient movement is decomposed; {bf:required}{p_end}
{synopt :{opt x2groups(spec)}}named covariate groups added in the full model{p_end}
{synopt :{opth fes(varlist)}}fixed-effect dimensions added in the full model{p_end}
{synopt :{opt vce(vcetype)}}{cmd:unadjusted} (default), {cmd:robust}, or {cmd:cluster}{p_end}
{synopt :{opth cluster(varname)}}cluster identifier (with {cmd:vce(cluster)}){p_end}
{synopt :{opt gamma0}}reproduce b1x2's {cmd:gamma0} variant{p_end}
{synopt :{opt cov0}}reproduce b1x2's {cmd:cov0} variant{p_end}
{synopt :{opt threads(#)}}OpenMP threads (0 = library default){p_end}
{synoptline}
{p2colreset}{...}
{p 4 6 2}{cmd:aweight}s and {cmd:fweight}s are allowed (b1x2 conventions).{p_end}


{title:Description}

{pstd}
{cmd:xhdfegelbach} implements the Gelbach (2016) decomposition: it explains the
movement of the {opt x1()} coefficients between the {it:base} regression
({it:y} on {it:x1}) and the {it:full} regression ({it:y} on {it:x1} plus the
covariate groups in {opt x2groups()} and the fixed effects in {opt fes()}) as a
sum of per-group contributions

{p 8 8 2}{it:delta_g = (X1'X1)^-1 X1'X2g beta_g}{p_end}

{pstd}
with a standard error for each contribution. The contributions sum exactly to
the total coefficient movement ({it:base} minus {it:full}); the residual of
that identity is returned in {cmd:r(identity_gap)}. Numerical semantics follow
Gelbach's {cmd:b1x2} (the reference implementation this command is validated
against), including its {opt gamma0}/{opt cov0} variants and the aweight/
fweight conventions. Fixed-effect groups are absorbed with the xhdfe backend,
so high-dimensional FE groups are practical. Shares the compiled backend with
the Python {cmd:xhdfe.gelbach.gelbach} and R {cmd:xhdfe_gelbach} front-ends.


{title:Options}

{phang}{opt x1(varlist)} lists the base covariates whose coefficient movement
is decomposed. {bf:Required}.

{phang}{opt x2groups(spec)} names the covariate groups added in the full model
with the b1x2-style syntax {cmd:x2groups("A = a1 a2 : B = b1 b2 b3")} (groups
separated by colons; each group is {it:name} {cmd:=} {it:varlist}). Each named
group becomes one column of the contribution matrix.

{phang}{opth fes(varlist)} gives fixed-effect dimensions (integer-valued
identifiers) added in the full model, each absorbed with the xhdfe backend and
treated as its own group (always gamma0-style).

{phang}{opt vce(vcetype)} selects the variance estimator: {cmd:unadjusted}
(default), {cmd:robust}, or {cmd:cluster} (supply {opth cluster(varname)}).

{phang}{opt gamma0} and {opt cov0} reproduce the corresponding options of
Gelbach's {cmd:b1x2}.

{phang}{opt threads(#)} sets OpenMP threads (0 = library default).

{pstd}At least one of {opt x2groups()} or {opt fes()} must be supplied.


{title:Stored results}

{pstd}{cmd:xhdfegelbach} stores the following in {cmd:r()}:

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Matrices}{p_end}
{synopt:{cmd:r(delta)}}contributions, one row per {opt x1()} variable, one column per group{p_end}
{synopt:{cmd:r(se)}}standard errors matching {cmd:r(delta)}{p_end}
{synopt:{cmd:r(total)}}total movement per {opt x1()} variable (base minus full) with its SE{p_end}

{p2col 5 20 24 2: Scalars}{p_end}
{synopt:{cmd:r(identity_gap)}}residual of the summation identity (should be ~0){p_end}
{synopt:{cmd:r(n_obs)}}number of observations{p_end}
{synopt:{cmd:r(df_full)}}residual degrees of freedom of the full model{p_end}
{synopt:{cmd:r(converged)}}1 if the computation converged{p_end}

{p2col 5 20 24 2: Macros}{p_end}
{synopt:{cmd:r(vce)}}the variance estimator used{p_end}
{synopt:{cmd:r(groups)}}the group names{p_end}
{synopt:{cmd:r(notes)}}any solver notes{p_end}
{p2colreset}{...}


{title:Examples}

{pstd}Decompose the education coefficient into an ability channel, a job-tenure
channel and a firm fixed-effect channel:{p_end}
{phang2}{cmd:. xhdfegelbach lwage, x1(educ) x2groups("skill = ability : job = tenure exper") fes(firm_id)}{p_end}
{phang2}{cmd:. matrix list r(delta)}{p_end}
{phang2}{cmd:. matrix list r(total)}{p_end}

{pstd}Cluster-robust inference by firm:{p_end}
{phang2}{cmd:. xhdfegelbach lwage, x1(educ) x2groups("job = tenure exper") vce(cluster) cluster(firm_id)}{p_end}

{pstd}A fuller walkthrough (Stata/Python/R) ships as
{bf:examples/gelbach_example.do}.{p_end}


{title:References}

{pstd}
Gelbach, J. B. 2016. When do covariates matter? And which ones, and how much?
{it:Journal of Labor Economics} 34(2): 509-543. Validated against his
{cmd:b1x2} Stata implementation.{p_end}


{title:Also see}

{psee}
{helpb xhdfe}, {helpb xhdfeakm}, {helpb xhdfeconnected}
{p_end}
