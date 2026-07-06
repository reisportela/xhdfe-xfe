{smcl}
{* *! version 1.0.0  06jul2026}{...}
{title:Title}

{p2colset 5 22 24 2}{...}
{p2col :{cmd:xhdfegelbach} {hline 2}}Gelbach (2016) conditional decomposition of coefficient movements{p_end}
{p2colreset}{...}


{title:Syntax}

{p 8 15 2}
{cmd:xhdfegelbach} {varname} {ifin} [{cmd:aweight} {cmd:fweight}]{cmd:,}
{opt x1(varlist)}
[{opt x2groups(spec)} {opt fes(varlist)}
{opt vce(vcetype)} {opth cluster(varname)}
{opt gamma0} {opt cov0} {opt threads(#)}]


{title:Description}

{pstd}
{cmd:xhdfegelbach} implements the Gelbach (2016) decomposition: it explains
the movement of the {opt x1()} coefficients between the "base" regression
(y on x1) and the "full" regression (y on x1 plus the covariate groups in
{opt x2groups()} and the fixed effects in {opt fes()}) as a sum of
contributions, one per group — {it:delta_g = (X1'X1)^-1 X1'X2g beta_g} —
with standard errors for each contribution. Numerical semantics follow
Gelbach's {cmd:b1x2} (the reference implementation this command is validated
against), including its {opt gamma0}/{opt cov0} variants and the aweight /
fweight conventions. Fixed-effect groups are absorbed with the xhdfe
backend, so high-dimensional FE groups are practical.

{phang}{opt x2groups(spec)} names the covariate groups with the b1x2-style
syntax {cmd:x2groups("A = a1 a2 : B = b1 b2 b3")} (groups separated by
colons; each group is {it:name} {cmd:=} {it:varlist}).

{phang}{opt fes(varlist)} gives fixed-effect dimensions (integer-valued
identifiers), each treated as its own group (always gamma0-style).

{phang}{opt vce(vcetype)} is {cmd:unadjusted} (default), {cmd:robust} or
{cmd:cluster} (with {opt cluster()}). {opt gamma0} and {opt cov0} reproduce
the corresponding b1x2 options.


{title:Stored results}

{pstd}{cmd:r(delta)} (matrix of contributions, one row per x1 variable and
one column per group), {cmd:r(se)} (matching standard errors),
{cmd:r(total)} (total movement per x1 variable: base minus full
coefficient), plus {cmd:r(b_base)}, {cmd:r(b_full)} and sample counts.


{title:Example}

{phang2}{cmd:. xhdfegelbach lwage, x1(educ) x2groups("skill = ability : job = tenure exper") fes(firm_id)}{p_end}
{phang2}{cmd:. matrix list r(delta)}{p_end}


{title:References}

{pstd}
Gelbach, J. B. 2016. When do covariates matter? And which ones, and how
much? {it:Journal of Labor Economics} 34(2): 509-543. Validated against his
{cmd:b1x2} Stata implementation.{p_end}


{title:Also see}

{psee}{helpb xhdfe}, {helpb xhdfeakm}{p_end}
