{smcl}
{* *! version 1.1.0  06jul2026}{...}
{title:Title}

{p2colset 5 20 22 2}{...}
{p2col :{cmd:xhdfeakm} {hline 2}}AKM estimation with leave-out (KSS) variance decomposition{p_end}
{p2colreset}{...}


{title:Syntax}

{p 8 15 2}
{cmd:xhdfeakm} {varname} {ifin}{cmd:,}
{opth work:er(varname)} {opth firm(varname)}
[{opth control:s(varlist)}
{opt leaveout:level(match|obs)}
{opt lev:erages(auto|exact|jla)}
{opt draws(#)} {opt seed(#)} {opt noprune}
{opth gen:erate(name)} {opt replace}
{opt threads(#)} {opt exactmax:rows(#)} {opt directmax:firms(#)}
{opt cgtol(#)} {opt fwltol(#)}
{opt se} {opt sensim(#)} {opt ci} {opt eigtracensim(#)} {opt gpu}]


{title:Description}

{pstd}
{cmd:xhdfeakm} estimates the two-way AKM model
{it:y = alpha(worker) + psi(firm) + X*beta + e}
on the largest leave-one-out connected set and reports the variance
decomposition in three flavours: {it:plug-in} (biased baseline), {it:AGSU}
(Andrews et al. 2008, homoskedastic correction) and {it:KSS} (Kline, Saggio
and Soelvsten 2020, heteroskedastic leave-out correction). Numerical
semantics follow Saggio's LeaveOutTwoWay (the canonical KSS implementation):
leave-a-match-out by default, person-year weighted components, and the JLA
leverage approximation with deterministic seeding on large samples. The
command runs on the same compiled xhdfe backend as the Python
{cmd:py_hdfe_v11.akm_kss} and R {cmd:xhdfe_akm_kss} front-ends and requires
the {cmd:xhdfe.plugin} shipped with this version.

{pstd}
Controls in {opt controls()} are partialled out at the person-year level
with the xhdfe absorber (FWL) before the two-way machinery runs.
{opt generate(stub)} stores {it:stub}{cmd:_alpha}, {it:stub}{cmd:_psi}
(observation-level effects on the leave-out sample; psi is normalized to a
zero person-year mean) and {it:stub}{cmd:_keep} (leave-out sample flag).


{title:Options}

{phang}{opth worker(varname)} and {opth firm(varname)} give the two
high-dimensional identifiers (integer-valued). Required.

{phang}{opt leaveoutlevel(match|obs)} chooses the leave-out unit:
{cmd:match} (worker-firm pair, the default and Saggio's default) or
{cmd:obs} (single person-year observation).

{phang}{opt leverages(auto|exact|jla)} selects the exact leverage path or the
Johnson-Lindenstrauss approximation; {cmd:auto} (default) uses exact when the
input has at most {opt exactmaxrows()} rows (10000, the LeaveOutTwoWay rule).

{phang}{opt draws(#)} sets the JLA simulations (default 200);
{opt seed(#)} makes them reproducible for any thread count.

{phang}{opt noprune} skips the leave-out connected-set computation; use only
when the sample is already leave-out connected.

{phang}{opt se} computes component standard errors (KSS leave-out inference,
person-year block leave-out with binned sigma-tilde); {opt sensim(#)} sets the
simulation draws for the quadratic part (default 1000).

{phang}{opt ci} adds weak-identification diagnostics and Andrews-Mikusheva
q=1 confidence intervals per component (the LeaveOutTwoWay
{it:eigen_diagno} path): top eigenvalue of Atilde and its share, the
Lindeberg condition, gamma-squared, the F statistic, the curvature-adjusted
{it:theta_1} and the AM confidence bounds. Implies {opt se}.
{opt eigtracensim(#)} sets the Hutchinson draws for tr(Atilde^2)
(default 100, the oracle default).

{phang}{opt gpu} solves the two-way systems on the CUDA backend when the
plugin was built with CUDA support (check {cmd:r(gpu_used)}).


{title:Stored results}

{pstd}{cmd:xhdfeakm} stores in {cmd:r()} the components
{cmd:r(plugin_var_alpha)}, {cmd:r(plugin_var_psi)}, {cmd:r(plugin_cov)},
{cmd:r(agsu_*)}, {cmd:r(kss_*)}, plus {cmd:r(var_y)}, {cmd:r(sigma2_ho)},
sample counts ({cmd:r(n_obs)}, {cmd:r(n_workers)}, {cmd:r(n_firms)},
{cmd:r(n_matches)}, {cmd:r(n_movers)}, {cmd:r(n_stayers)}), leverage
diagnostics ({cmd:r(max_pii)}, {cmd:r(mean_pii)}, {cmd:r(leverages_exact)})
and solver diagnostics; with controls, {cmd:r(b)} holds the control
coefficients. With {opt se}: {cmd:r(se_var_psi)}, {cmd:r(se_cov)},
{cmd:r(se_var_alpha)} and the matching {cmd:r(theta_*)} point estimates.
With {opt ci}, per component suffix ({cmd:_psi}, {cmd:_cov}, {cmd:_alpha}):
{cmd:r(lambda1_*)}, {cmd:r(eigshare1_*)}, {cmd:r(lindeberg_*)},
{cmd:r(gammasq_*)}, {cmd:r(fstat_*)}, {cmd:r(theta1_*)}, {cmd:r(ci_lb_*)},
{cmd:r(ci_ub_*)} and {cmd:r(curvature_*)}.


{title:Example}

{phang2}{cmd:. use data/felsdvsimul, clear}{p_end}
{phang2}{cmd:. xhdfeakm y, worker(i) firm(j)}{p_end}
{phang2}{cmd:. xhdfeakm y, worker(i) firm(j) controls(x1 x2) generate(akm)}{p_end}


{title:References}

{pstd}
Abowd, Kramarz and Margolis (1999); Andrews, Gill, Schank and Upward (2008);
Kline, Saggio and Soelvsten (2020, Econometrica); Saggio's LeaveOutTwoWay
(the canonical oracle this implementation is validated against).
{p_end}
