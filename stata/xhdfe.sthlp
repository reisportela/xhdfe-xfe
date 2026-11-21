{smcl}
{* *! version 2.34.1 21nov2026}{...}
{vieweralsosee "[R] areg" "help areg"}{...}
{vieweralsosee "[R] xtreg" "help xtreg"}{...}
{vieweralsosee "" "--"}{...}
{vieweralsosee "reghdfe" "help reghdfe"}{...}
{vieweralsosee "ivreghdfe" "help ivreghdfe"}{...}
{vieweralsosee "ppmlhdfe" "help ppmlhdfe"}{...}
{vieweralsosee "" "--"}{...}
{vieweralsosee "xfepout" "help xfepout"}{...}
{vieweralsosee "xhdfeakm" "help xhdfeakm"}{...}
{vieweralsosee "xhdfeconnected" "help xhdfeconnected"}{...}
{vieweralsosee "xhdfegelbach" "help xhdfegelbach"}{...}
{vieweralsosee "Public-data examples" "help xhdfe_examples"}{...}
{vieweralsosee "" "--"}{...}
{viewerjumpto "Syntax" "xhdfe##syntax"}{...}
{viewerjumpto "Absorb syntax" "xhdfe##absorb"}{...}
{viewerjumpto "Description" "xhdfe##description"}{...}
{viewerjumpto "Development process" "xhdfe##development"}{...}
{viewerjumpto "Group FEs" "xhdfe##group_fes"}{...}
{viewerjumpto "Options" "xhdfe##options"}{...}
{viewerjumpto "Postestimation Syntax" "xhdfe##postestimation"}{...}
{viewerjumpto "Known limitations" "xhdfe##known_limitations"}{...}
{viewerjumpto "Examples" "xhdfe##examples"}{...}
{viewerjumpto "Stored results" "xhdfe##results"}{...}
{viewerjumpto "Authors" "xhdfe##contact"}{...}
{viewerjumpto "Acknowledgements" "xhdfe##acknowledgements"}{...}
{viewerjumpto "References" "xhdfe##references"}{...}
{title:Title}

{p2colset 5 18 20 2}{...}
{p2col :{cmd:xhdfe} {hline 2}}Linear regression with multiple fixed effects via a C++ backend{p_end}
{p2colreset}{...}

{marker syntax}{...}
{title:Syntax}

{pstd}
{bf:Least-squares regression without fixed effects:}

{p 8 15 2} {cmd:xhdfe}
{it:depvar} [{it:indepvars}]
{ifin} {it:{weight}}
[{cmd:,} {help xhdfe##options_table:options}]
{p_end}

{pstd}
{bf:Fixed effects regressions:}

{p 8 15 2} {cmd:xhdfe}
{it:depvar} [{it:indepvars}]
{ifin} {it:{weight}} {cmd:,} {opth a:bsorb(xhdfe##absorb:absvars)} [{help xhdfe##options_table:options}]{p_end}

{pstd}
{bf:Fixed effects regressions with group-level outcomes:}

{p 8 15 2} {cmd:xhdfe}
{it:depvar} [{it:indepvars}]
{ifin} {it:{weight}} {cmd:,} {opt g:roup(groupvar)} [{opth a:bsorb(xhdfe##absorb:absvars)} {help xhdfe##options_table:options}]{p_end}

{pstd}
{bf:Fixed effects regressions with group-level outcomes and individual FEs:}

{p 8 15 2} {cmd:xhdfe}
{it:depvar} [{it:indepvars}]
{ifin} {it:{weight}} {cmd:,} {opt g:roup(groupvar)} {opt i:ndividual(indvar)} [{opth a:bsorb(xhdfe##absorb:absvars)} {help xhdfe##options_table:options}]{p_end}

{pstd}
{bf:IV/2SLS with optional fixed effects:}

{p 8 15 2} {cmd:xhdfe}
{it:depvar} [{it:exogvars}] {it:endogvars}
{ifin} {it:{weight}} {cmd:,} [{opth a:bsorb(xhdfe##absorb:absvars)}]
{opt endog:enous(endogvars)} {opt instr:uments(instvars)} [{help xhdfe##options_table:options}]{p_end}

{pstd}
{bf:Show package version:}

{p 8 15 2} {cmd:xhdfe, version}{p_end}


{marker options_table}{...}
{synoptset 22 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Standard FEs {help xhdfe##opt_absorb:[+]}}
{synopt: {opth a:bsorb(xhdfe##absorb:absvars)}}categorical fixed effects, labelled absorbs, categorical interactions, and heterogeneous slopes{p_end}
{synopt: {cmdab:a:bsorb(}{it:label}{cmd:=}{it:absvar}{cmd:)}}label an absorbed effect and use {it:label} as the saved-FE base name{p_end}
{synopt: {cmdab:a:bsorb(}{it:fe}{cmd:#c.}{it:x}{cmd:)}}absorb group-specific slopes without a level FE{p_end}
{synopt: {cmdab:a:bsorb(}{it:fe}{cmd:##c.}{it:x}{cmd:)}}absorb a level FE plus group-specific slopes{p_end}
{synopt: {cmdab:a:bsorb(}{it:...}{cmd:,} {cmd:savefe)}}save all fixed effect contributions with the {it:__hdfe*} prefix{p_end}
{synopt: {opth savefes(prefix)}}save fixed effect contributions using a prefix{p_end}
{synopt: {opt noa:bsorb}}explicitly fit without absorbed effects; incompatible with {cmd:absorb()}{p_end}
{synopt : }{bf:- note:} savefe/savefes are not supported when both {cmd:group()} and {cmd:individual()} are specified{p_end}

{syntab:Group FEs {help xhdfe##opt_group_fes:[+]}}
{synopt : {opth g:roup(xhdfe##opt_groupvar:groupvar)}}categorical variable representing each group (e.g. patent_id){p_end}
{synopt : }{bf:- note:} using {cmd:group()} without {cmd:individual()} is equivalent to running the regression on 1 observation per group{p_end}
{synopt : {opth i:ndividual(xhdfe##opt_indvar:indvar)}}categorical variable representing each individual whose fixed effect will be absorbed{p_end}
{synopt : }{bf:- note:} the {cmd:individual()} option requires {cmd:group()}{p_end}
{synopt : }{bf:- note:} {cmd:individual()} is appended to {cmd:absorb()} if not already listed{p_end}
{synopt : }{bf:- alias:} {cmd:i(}{it:indvar}{cmd:)} is accepted as a shorthand for {cmd:individual(}{it:indvar}{cmd:)}{p_end}
{synopt : {opth ag:gregation(xhdfe##opt_aggregation:str)}}aggregation for individual FEs
within a group: {it:mean} (default), {it:sum}; {it:avg}/{it:average} are aliases for mean{p_end}

{syntab:Model {help xhdfe##opt_model:[+]}}
{synopt : {opth vce:(xhdfe##opt_vce:vcetype)}}{it:vcetype} may be
{opt unadjusted} (default), {opt robust}, {opt cl:uster} {help fvvarlist} (multiway), or {opt dk:raay} [{it:#}]{p_end}
{synopt : {opt rob:ust}}equivalent to {cmd:vce(robust)}{p_end}
{synopt : {opth cl:uster(varlist)}}equivalent to {cmd:vce(cluster ...)};
multiway clustering allowed; supports categorical interactions with {cmd:#}
(two or more components) and two-component {cmd:##}{p_end}
{synopt : {opth res:iduals(newvar)}}save regression residuals{p_end}
{synopt : {opt res:iduals}}save regression residuals in {cmd:_reghdfe_resid}{p_end}
{synopt : {opt keep:singletons}}do not drop singleton observations{p_end}
{synopt : {opt nocon:stant}}suppress the intercept in the slope regression{p_end}
{synopt : {opt cons:tant}}explicitly retain the default intercept; incompatible with {cmd:noconstant}{p_end}

{syntab:IV/2SLS {help xhdfe##opt_iv:[+]}}
{synopt : {opth endog:enous(varlist)}}endogenous regressors (subset of slopes){p_end}
{synopt : {opth instr:uments(varlist)}}instruments for IV/2SLS{p_end}
{synopt : {opt first}}display first-stage regressions and excluded-instrument tests{p_end}
{synopt : {opt ffirst}}display only the first-stage diagnostic summary{p_end}

{syntab:Degrees-of-Freedom Adjustments {help xhdfe##opt_dof:[+]}}
{synopt :{opt dof:adjustments(list)}}select absorbed-DoF adjustment rules; default is {cmd:all}{p_end}
{synopt :{opt ssc(str)}}fixest-style small-sample corrections (K/G/t.df){p_end}
{synopt :{opt statstyle(str)}}fit-statistics style: {it:reghdfe} (default) or {it:legacy} (pre-change xhdfe){p_end}
{synopt: {opth groupvar(newvar)}}identifier for the first mobility group{p_end}

{syntab:Optimization {help xhdfe##opt_optimization:[+]}}
{synopt :{opt tol:erance(#)}}convergence tolerance (default 1e-8){p_end}
{synopt :{opt tolerancemode(str)}}absorber convergence mode:
{it:reghdfe-comparable} (default since 2.7.0), {it:xhdfe-fast} (pre-2.7.0 fast trigger),
or {it:strict-residual}{p_end}
{synopt :{opt convergence(str)}}stopping criterion for iterative sweeps:
{it:auto} (default; follows {cmd:tolerancemode()}),
{it:normchange}, {it:reghdfe}, or {it:both}{p_end}
{synopt :{opt fetol:erance(#)}}fixed-effect recovery tolerance for {cmd:savefe}/{cmd:savefes} MAP fallback (default 1e-6){p_end}
{synopt :{opt ferecoverym:ethod(str)}}fixed-effect recovery method: {it:hybrid} (default) or {it:map}{p_end}
{synopt :{opt maxit:er(#)}}maximum absorber iterations (default 100000){p_end}
{synopt :{opt iterate(#)}}alias of {cmd:maxiter(#)} for reghdfe compatibility; so are {cmd:iter(#)} and {cmd:iterations(#)}{p_end}
{synopt :{opt absorptionm:ethod(str)}}absorption method (auto, gauss-seidel, symmetric-gauss-seidel, jacobi, mlsmr, lsmr, auto-mlsmr; aliases supported){p_end}
{synopt :{opt reducedpcg(on|off)}}reduced-system tier for this fit; default: the global, then the environment variable {cmd:XHDFE_REDUCED_PCG}, else on{p_end}
{synopt :{opt reducedpcgcuda(on|off)}}experimental CUDA variant of the tier for this fit, on CUDA fits only;
default: the global, then the environment variable {cmd:XHDFE_REDUCED_PCG_CUDA}, else off{p_end}
{synopt :{opt sym:metricsweep}}forward + backward sweep per iteration{p_end}
{synopt :{opt jacobirel:axation(#)}}Jacobi relaxation parameter (0 = default){p_end}

{syntab:GPU {help xhdfe##opt_gpu:[+]}}
{synopt :{opt gpubackend(str)}}backend selector: {it:cpu}, {it:cuda}, or {it:metal}; errors if a requested non-CPU backend is unavailable{p_end}
{synopt :{opt weightedcpuroute(on|off)}}request CPU for eligible weighted CUDA fits; default {cmd:off}{p_end}

{syntab:Parallel execution {help xhdfe##opt_parallel:[+]}}
{synopt :{opt numthr:eads(#)}}threads used by the absorber (0 = auto){p_end}
{synopt :{opt defaultthr:eads(#)}}default threads when auto-threading is enabled{p_end}
{synopt :{opt maxthr:eads(#)}}maximum threads when auto-threading is enabled{p_end}
{synopt :{opt minparallelrows(#)}}minimum rows for automatic parallel absorption{p_end}
{synopt :{opt targetrowsperthread(#)}}target rows per thread (0 = auto){p_end}

{syntab:Reporting {help xhdfe##opt_reporting:[+]}}
{synopt :{opt level(#)}}set confidence level; default is {cmd:level(95)}; {cmd:l(#)} is accepted as a shorthand{p_end}
{synopt :{opt noheader}}suppress the header block; forwarded to the xhdfe display layer for reghdfe compatibility{p_end}
{synopt :{opt notable}}suppress the coefficient table; header/footnote behaviour follows Stata e-class display conventions{p_end}
{synopt :{opt nofootnote}}suppress the absorbed-DoF footnote table{p_end}
{synopt :{opt noomitted}}hide omitted coefficients in the coefficient table{p_end}
{synopt :{opt noempty}}hide empty cells in factor-variable displays{p_end}
{synopt :{opt baselevels}}show base levels in factor-variable displays{p_end}
{synopt :{opt allbaselevels}}show all base levels in factor-variable displays{p_end}
{synopt :{opt cformat(fmt)}}display format for coefficients, standard errors and confidence limits{p_end}
{synopt :{opt pformat(fmt)}}p-value display format; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt sformat(fmt)}}display format for the t or z statistic{p_end}
{synopt :{opt fvwrap(#)}}factor-variable wrapping option; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt vsquish}}compact vertical display; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt nolstretch}}disable line stretching in table display; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt nofvlabel}}suppress factor-value labels in table display; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt noci}}hide confidence intervals in the coefficient table{p_end}
{synopt :{opt nopvalues}}hide p-values in the coefficient table{p_end}
{synopt :{opt nosample}}do not save {cmd:e(sample)}; reduces memory use but disables postestimation that requires the sample marker{p_end}
{synopt :{opt nowarn}}suppress singleton and compatibility notes; estimation is unchanged{p_end}

{syntab:Compatibility}
{synopt :{opt timeit}, {opt verbose(#)}, {opt parallel(str)}}accepted for reghdfe compatibility{p_end}
{synopt :{opt technique(str)}, {opt acceleration(str)}, {opt transform(str)}, {opt preconditioner(str)}}accepted for reghdfe compatibility{p_end}
{synopt :{opt prune}, {opt fastregress}, {opt poolsize(#)}, {opt compact}, {opt version(#)}}accepted for reghdfe compatibility{p_end}
{synopt :{opt fvwrapon(style)}}accepted for display compatibility{p_end}

{syntab:Diagnostics {help xhdfe##opt_diagnostics:[+]}}
{synopt :{opt mobilityprofile}}compute mobility profile and save to {cmd:xhdfe_mobility_profile.txt}{p_end}
{synopt :{opth mobfile(filename)}}read or save mobility profile file (aliases: {cmd:mobilityfile()}, {cmd:mobilityprofilefile()}){p_end}
{synopt :{opth absorptioncache(filename)}}read or save an explicit absorption cache file{p_end}
{synopt :{opt absorptioncachemode(str)}}absorption cache mode: off, auto, read, write{p_end}
{synopt :{opt festructurecache}}build/reuse a fixed-effect structure cache in the {cmd:xhdfe.ado} directory{p_end}
{synopt :{opth fescache(filename)}}read or save a fixed-effect structure cache file
at an explicit path (aliases: {cmd:festructurecache()}, {cmd:festructurecachefile()}){p_end}
{synopt :{opt fescachemode(str)}}fixed-effect structure cache mode: off, auto, read, write{p_end}
{synopt : }{bf:- note:} when a profile file exists and matches the data, xhdfe auto-loads it to guide absorber ordering{p_end}
{synopt : }{bf:- note:} FE structure caches speed up repeated runs with the same {cmd:absorb()} structure; use {cmd:fescache()} for a fixed path{p_end}

{synoptline}
{p 4 6 2}The compatibility options in the table leave estimation unchanged.
{cmd:xhdfe, version} (without a number) reports the package version. Use {opt numthreads()} to control
in-process OpenMP parallelism.{p_end}
{p 4 6 2}The dependent variable permits time-series operators. The regressors permit
{help fvvarlist:factor variables} and {help tsvarlist:time-series operators}.
Factor-variable prefixes are supported in slopes. In {cmd:absorb()}, categorical interactions with
{cmd:#}/{cmd:##} are supported, as are heterogeneous slopes of the form {cmd:fe#c.x}
(group-specific slopes) and {cmd:fe##c.x} (group intercepts plus group-specific slopes).
In {cmd:cluster()}, categorical interactions with {cmd:#}/{cmd:##} are supported but continuous-factor
terms such as {cmd:c.x} are rejected. When using {cmd:endogenous()/instruments()},
factor-variable operators in the slope list are not supported.{p_end}
{p 4 6 2}Supported weights are {cmd:aweight}, {cmd:fweight}, {cmd:pweight}, and {cmd:iweight}.
With {cmd:pweight} and no {cmd:vce()}, xhdfe defaults to robust SEs; probability weights cannot
be combined with {cmd:vce(unadjusted)}. Negative importance weights are rejected; zero-weight
rows are excluded. For unadjusted importance-weighted fits, the reported observation count and residual degrees of
freedom follow the Stata importance-weight convention. See {help xhdfe##opt_vce:weights and VCE}.{p_end}

{marker opt_gpu}{...}
{title:Backend selection}

{phang}
{opt gpubackend(cpu|cuda|metal)} selects the absorption backend for this fit.
With the option omitted, selection follows {cmd:XHDFE_GPU_BACKEND}, then the
build default (CPU on standard builds). CUDA requires a CUDA-enabled plugin and
a supported NVIDIA device. Metal is reserved. A required CUDA result that is
unavailable, fails, or would reuse CPU absorption returns error 498, subject to
the explicitly requested weighted route below.

{phang}
Check {cmd:e(gpu_used)}, {cmd:e(gpu_backend)} and {cmd:e(gpu_status)}.
A value of 1 for {cmd:e(gpu_used)} confirms device work. Inspect {cmd:e(cuda_tail)}
for the completion: {cmd:device-lsmr} on the device, {cmd:host-mlsmr} after CUDA
sweeps or on a restart, or {cmd:none} when no separately labelled completion was
reported. Host completion preserves the requested tolerance and shares the
iteration budget with the CUDA stages.

{marker gpu_weighted_cpu_route}{...}
{phang}
{opt weightedcpuroute(on|off)} defaults to {cmd:off}. With {cmd:on}, a fit
requesting available CUDA runs on CPU if it uses weights and at least two
distinct absorbed terms in {it:reghdfe-comparable} or {it:strict-residual} mode.
This includes unit weights and the collapsed {cmd:group()} fit. Combined
{cmd:group()}/{cmd:individual()} fits and {it:xhdfe-fast} retain their own
routing. As an example of how terms are counted, {cmd:absorb(a a)} is one term
and {cmd:absorb(g##c.z)} is one combined term. The routed fit uses CPU cache
and profile rules. It reports {cmd:e(gpu_used)} = 0, {cmd:e(gpu_status_code)} =
7, {cmd:e(gpu_status)} = {cmd:weighted_cpu_route}, {cmd:e(gpu_backend)} =
{cmd:cpu} and {cmd:e(gpu_backend_requested)} = {cmd:cuda}.

{phang}
Ordinary weighted CUDA fits with two to four categorical effects and
unweighted fits with four categorical effects have a per-column FE-moment
check in Comparable and Strict modes. For Comparable analytic, probability
and importance weights, the correction applies up to 2,000,000 estimation
rows. A failed check can trigger device LSMR within the remaining
{cmd:maxiter()} budget. Ordinary unweighted three-effect Comparable fits
have their own completion path. See {help xhdfe##known_limitations:limitations}
and {help xhdfe##opt_reducedpcgcuda:reducedpcgcuda()} for the opt-in reduced tier.

{phang}
The standard online install supplies CPU plugins. See {helpb xhdfegpu} to
build the Linux CUDA plugins. After replacing a plugin or changing its
{cmd:adopath} entry, run {cmd:discard} with no arguments or restart Stata.
See {help xhdfe_examples##cuda:a CUDA example with public data}.

{marker description}{...}
{title:Description}

{pstd}
{cmd:xhdfe} estimates linear models with high-dimensional fixed effects (HDFE) using a C++ backend.
It mirrors the reporting and defaults of {cmd:reghdfe} where possible, while exposing fixest-style
small-sample corrections and ideas from pyfixest, FixedEffectModels.jl, the {it:within} solvers of Fischer
and Schroeder, and {cmd:fereg} by Schmieder (see {help xhdfe##acknowledgements:Acknowledgements}).
The command supports multiway fixed
effects, weights, robust and multiway-cluster inference, reghdfe-style singleton dropping and
DoF adjustments, and (optionally) the group()/individual() machinery.

{pstd}
Additional features include:{p_end}

{phang2}  1. A fast absorber with multiple iterative methods and multi-threading.{p_end}
{phang2}  2. Multiway clustered inference and cluster diagnostics.{p_end}
{phang2}  3. Storage of fixed-effect contributions (savefe/savefes).{p_end}
{phang2}  4. Reghdfe-style DoF logic (dofadjustments) and fixest-style SSC controls (ssc).{p_end}
{phang2}  5. Heterogeneous slopes and IV/2SLS for supported standard-FE designs.{p_end}


{marker development}{...}
{title:Development process}

{pstd}
The authors directed an AI-assisted development and validation process and
are responsible for the econometric specifications, validation criteria and
released code.{p_end}

{marker group_fes}{...}
{title:Description of individual fixed effects in group setting}

{pstd}{cmd:xhdfe} supports estimations that include individual fixed effects with group-level outcomes.
This requires long-format data where the group and individual identifiers uniquely identify observations
(e.g. "isid patent_id inventor_id").

{pstd}If {cmd:group()} is specified without {cmd:individual()}, the regression is run on one observation
per group (equivalent to collapsing to a representative row per group). In group/individual mode,
{cmd:e(sample)} and output variables are written on that representative row.
Singleton bookkeeping in this mode is done after the group-level representative-row construction, so
{cmd:e(num_singletons)} can differ from reghdfe's pre-collapse row count even when the effective estimation
sample is the same.{p_end}

{pstd}On CPU, {cmd:absorptionmethod(auto)} uses a joint matrix-free LSMR operator for the
combined standard and individual fixed effects. {cmd:absorptionmethod(lsmr)} selects that
CPU path explicitly. If an automatic LSMR solve cannot pass its independent certificate,
{cmd:auto} retries the previous sweep path from the original data; the reported method code
always identifies the result actually used. On CUDA, {cmd:auto} retains the GPU group/individual
solver and accepts it only when the authoritative certificate passes. Explicit LSMR with CUDA
is rejected because this LSMR implementation is CPU-only. Explicit {cmd:gauss-seidel} and
{cmd:symmetric-gauss-seidel} remain available; {cmd:jacobi}, {cmd:mlsmr}, and
{cmd:auto-mlsmr} are not supported in group/individual mode.{p_end}


{marker links}{...}
{title:Links to online documentation}

{p2col 8 10 10 2: -}{browse "https://github.com/reisportela/xhdfe-xfe":xhdfe repository}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/sergiocorreia/reghdfe":reghdfe repository}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/lrberge/fixest":fixest repository}{p_end}
{p2col 8 10 10 2: -}{browse "https://pyfixest.org/":pyfixest documentation}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/py-econometrics/pyfixest":pyfixest repository}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/FixedEffects/FixedEffectModels.jl":FixedEffectModels.jl repository}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/py-econometrics/within":within repository}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/py-econometrics/within-paper":Fischer and Schroeder (2026), within paper}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/johannes-schmieder/fereg":fereg repository (Johannes F. Schmieder)}{p_end}
{p2col 8 10 10 2: -}{browse "https://cepr.org/publications/dp22170":Schmieder (2026), fereg paper (CEPR DP22170)}{p_end}
{p2col 8 10 10 2: -}{browse "https://johannes-schmieder.com/papers/fereg.pdf":Schmieder (2026), fereg paper (author's copy)}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/leoliu0/fereg":fereg repository (Leo Liu; a different package)}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/droodman/reghdfejl":reghdfejl repository}{p_end}


{marker absorb}{...}
{title:Absorb() syntax}

{synoptset 22}{...}
{synopthdr:absvar}
{synoptline}
{synopt:{it:varname}}categorical variable to be absorbed{p_end}
{synopt:{it:label}={it:varname}}label the absorbed effect and use {it:label} as the saved-FE base name{p_end}
{synopt:{it:var1#var2#...}}categorical interaction FE with two or more components; categorical-only {cmd:##} is accepted as the same joint FE{p_end}
{synopt:{it:fe#c.var}}group-specific slope without a level FE{p_end}
{synopt:{it:fe##c.var}}level FE plus group-specific slope{p_end}
{synoptline}
{p2colreset}{...}
{p 4 6 2}- Categorical interactions with two or more components are supported via {cmd:#} or {cmd:##}. For reghdfe
compatibility, categorical-only {cmd:##} terms are treated as one joint absorbed FE, not expanded into
main effects plus the interaction. Specify {cmd:var1 var2 var1#var2} explicitly if all three absorbed
components are intended.
Interaction FEs are coded after the estimation sample is defined: with {cmd:egen group()} over their
components when all of them are numeric; when a component is a string, by sorting only the identifier
columns of the estimation sample in a temporary frame, which leaves the order of the data untouched,
with {cmd:egen group()} as the fallback. Plain numeric FE identifiers are passed directly to the plugin;
string identifiers are coded by the same frame sort before the plugin call.{p_end}
{p 4 6 2}- Heterogeneous slopes require exactly one categorical component and one continuous
component. Use {cmd:fe#c.x} for group-specific slopes only, or {cmd:fe##c.x} for a level FE plus
group-specific slopes. Continuous-only interactions, and slope terms with more than two components,
are rejected.{p_end}
{p 4 6 2}- Use {it:label}={it:var} to label absorbed effects in output and saved-FE names.
For {cmd:##}, label the joint term as a single component or specify the components separately if you
explicitly include them separately.{p_end}
{p 4 6 2}- When {it:label}={it:var} is used, xhdfe saves FE contributions even without
{cmd:savefe}/{cmd:savefes}. For categorical FEs the output variable is {it:label}; for
heterogeneous slopes {it:label} is used as the base name, with {cmd:Slope1} added for slope
contributions. The generated variable names must be new. To store group-coded IDs instead,
pre-create them (e.g., {cmd:egen label = group(var)}) and use that variable in {cmd:absorb()}.{p_end}
{p 4 6 2}- To save all absorbed contributions, use {cmd:absorb(..., savefe)} or {cmd:savefes(prefix)}.{p_end}
{p 4 6 2}- Saving FE contributions is not supported when both {cmd:group()} and {cmd:individual()} are specified.{p_end}
{p 4 6 2}- Saved fixed effects are contributions, not ids; they may not be identified in all designs.{p_end}


{marker options}{...}
{title:Options}

{marker opt_absorb}{...}
{dlgtab:Standard FEs}

{phang}
{opth a:bsorb(xhdfe##absorb:absvars)} list of categorical variables representing the fixed effects to be absorbed.
This is equivalent to including an indicator/dummy for each category of each absvar.

{pmore}
Categorical interactions with two or more components can be specified with {cmd:#} or {cmd:##}; categorical-only {cmd:##}
is canonicalized to the same joint FE as {cmd:#}, matching reghdfe's absorbed-DoF presentation.
Interaction FEs are coded after the estimation sample is defined, with {cmd:egen group()} when all their
components are numeric and, when a component is a string, by sorting only the identifier columns in a
temporary frame ({cmd:egen group()} is the fallback). Plain numeric FE identifiers are passed directly to the
plugin; string identifiers are coded by the same frame sort.

{pmore}
For example, {cmd:absorb(a#b#c)} absorbs one effect for each observed joint category,
equivalent to precomputing {cmd:egen abc = group(a b c)} and using {cmd:absorb(abc)}
on the same estimation sample. {cmd:absorb(a b c)} instead absorbs three separate
additive effects; grouping them changes that model. Named effects such as
{cmd:absorb(effect=a#b#c)} are also supported.

{pmore}
Heterogeneous slopes can be specified as {cmd:fe#c.x} or {cmd:fe##c.x}. The first absorbs
group-specific slopes only; the second absorbs both the FE intercept and the group-specific slopes.
They are not supported with {cmd:group()}/{cmd:individual()} or with {cmd:absorptionmethod(lsmr/mlsmr)}.
In {it:reghdfe-comparable} with heterogeneous slopes, the slope-block certificate judges a regressor omitted
as collinear with the absorbed effects in its input scale, so that its rounding residue is not a precision
target and does not cause a refusal; the estimates are those of the fit without that regressor up to the
accuracy of the mode (to rounding when both solves stop at the same iteration).
{cmd:XHDFE_SLOPE_ABSORBED_INPUT_SCALE=0}, read once per process, judges it in its absorbed scale, as every
other column.

{pmore}
Set {cmd:XHDFE_EXPERIMENT_SLOPE_SCHUR=1} to enable the optional secondreg solver for
eligible unweighted OLS with two ordinary fixed effects and two slopes on the same
carrier with an intercept. It applies to automatic absorption and convergence in
{it:reghdfe-comparable} mode without saved effects. An accepted solve reports method
code 9 and passes the original slope certificate. Ineligible fits use the standard
absorber. {cmd:XHDFE_EXPERIMENT_NETWORK_CSR_OWNER=1} enables parallel sparse matrix
assembly. {cmd:XHDFE_EXPERIMENT_NETWORK_SCORES=1} enables eligible two-way cluster
score reuse when the cluster partitions match the worker and firm effects.
On a CUDA fit this solver opens cuBLAS and cuSPARSE at its first use, from the folders recorded in
the plugin; loading the plugin does not need them. If they cannot be opened, the fit stops with an
error and leaves no estimates. The failure is remembered until the Stata session ends, and CPU fits
keep working.

{pmore}
Absorb entries can be labelled as {it:label}={it:var} to control FE names in output and saved-FE names;
for {cmd:##}, label the joint term as one absorb entry or specify separate components explicitly.
Labels also instruct xhdfe to store FE contributions even without {cmd:savefe}/{cmd:savefes}. For
categorical FEs the output variable is {it:label}; for heterogeneous slopes {it:label} is the base
name and xhdfe adds {cmd:Slope1} to slope contributions. Generated names must not already exist.
Only the labelled terms are saved: unlabelled terms in the same {cmd:absorb()} (e.g. {cmd:year} in
{cmd:absorb(year wfe=worker ffe=firm)}) are absorbed as usual but leave no variable behind.

{pmore}
{cmd:absorb(..., savefe)} saves fixed effect contributions into {it:__hdfe1__}, {it:__hdfe2__}, ...
(reghdfe-compatible naming). This option cannot be combined with {cmd:savefes()}.
Saving FE contributions is not supported when both {cmd:group()} and {cmd:individual()} are specified.
For heterogeneous slopes, {cmd:g##c.z} saves the base FE and a {it:Slope1} variable
(for example, {it:__hdfe1__} and {it:__hdfe1__Slope1}); {cmd:g#c.z} saves only
the {it:Slope1} coefficient variable, matching reghdfe's savefe layout.
For {cmd:g##c.(x z)}, the two slope coefficients are named
{it:__hdfe1__Slope1} and {it:__hdfe1__Slope2} within the same FE block.
Repeated calls may refresh automatic FE variables created by xhdfe. A variable
created by the user, or used as an input to the current model, is protected.
Models with only absorbed slopes and no constant or regressors return fit
statistics and predictions without coefficient or covariance matrices.
With {cmd:ferecoverymethod(hybrid)} (default), xhdfe accumulates FE alphas during absorption when feasible,
so savefe/savefes typically avoids a second recovery sweep. {cmd:fetolerance()} applies only to the
fallback MAP recovery.

{pmore}
In {it:reghdfe-comparable} with {cmd:absorptionmethod(auto)}, saved effects ({cmd:savefe}, {cmd:savefes()} or
labelled {cmd:absorb()} terms) come from an effect-capturing sweep, whose exit is looser than the route of the
mode. When that sweep is certified but stopped above the floor of the mode (the Krylov parity floor,
{cmd:1e-11} at the default tolerance, the target of MLSMR and of the
{help xhdfe##reduced_pcg:reduced-system tier}), the projection is computed again without capture by that route
(eligible for the tier), and the coefficients, covariance, standard errors, residuals and constant are those of
that projection, the same as those of the fit without saved effects; {cmd:e(iterations)} then counts both
solves. At or below that floor the sweep itself is published. When the projection is published, the residuals
are those of the accurate fit, so {it:y} = {it:xb} + constant + saved effects + residuals holds only to the
accuracy of the saved effects (see {help xhdfe##known_limitations:Known limitations}).
{cmd:XHDFE_SAVEFE_ACCURATE_PROJECTION=0}, read once per process, publishes the capturing sweep in every case.

{marker opt_savefes}{...}
{phang}
{opt savefes(prefix)} saves fixed effect contributions into new variables named
{it:prefix}{it:fevar}, falling back to {it:prefix}fe{it:#} if names are too long or already exist.
Saving FE contributions is not supported when both {cmd:group()} and {cmd:individual()} are specified.


{marker opt_group_fes}{...}
{dlgtab:Group FEs}

{marker opt_groupvar}{...}
{phang}
{opth g:roup(xhdfe##opt_groupvar:groupvar)} categorical variable representing each group (e.g. patent_id).
If only {cmd:group()} is specified, the regression is run on one observation per group.
A string {it:groupvar} is coded by the same frame sort as string FE identifiers ({cmd:egen group()} is the
fallback); a float or double one is coded with {cmd:egen group()}.

{marker opt_indvar}{...}
{phang}
{opth i:ndividual(xhdfe##opt_indvar:indvar)} categorical variable representing each individual.
If {cmd:individual()} is specified you must also call {cmd:group()}.
In xhdfe, {cmd:individual()} is appended (and placed last) in the absorb list for stability.
{cmd:i(}{it:indvar}{cmd:)} is accepted as a shorthand for {cmd:individual(}{it:indvar}{cmd:)} for reghdfe compatibility.

{marker opt_aggregation}{...}
{phang}
{opt ag:gregation(str)} method of aggregation for individual components in group FEs.
Valid options are {cmd:mean} (default; aliases {cmd:avg}/{cmd:average}) and {cmd:sum}. Values are case-insensitive.

{phang}
{bf:Combining options:} depending on which of {cmd:absorb()}, {cmd:group()}, and {cmd:individual()} you specify:

{pmore}  1. If none is specified, xhdfe runs OLS with a constant.{p_end}
{pmore}  2. If only {cmd:absorb()} is present, xhdfe runs a standard fixed-effects regression.{p_end}
{pmore}  3. If {cmd:group()} is specified (but not {cmd:individual()}), this is equivalent to #1 or #2 with only one observation per group.{p_end}
{pmore}  4. If all are specified, this is a fixed-effects regression at the group level with individual FEs.{p_end}


{marker opt_model}{...}
{dlgtab:Model}

{marker opt_vce}{...}
{phang}
{opth vce:(xhdfe##opt_vce:vcetype)} specifies the type of standard error reported.

{pmore}
{opt unadjusted} (aliases: {cmd:ols}, {cmd:homoskedastic}, {cmd:classical}) estimates conventional standard errors (default).

{pmore}
{opt robust} (alias: {cmd:hc1}) estimates heteroskedasticity-consistent standard errors (HC1-style).
Inside {cmd:vce()}, {cmd:unadjusted}, {cmd:robust} and their aliases must be written in full.

{pmore}
{opt cl:uster} (alias: {cmd:clustered}) {it:clustervars} estimates standard errors clustered by one or more variables.
Multiway clustering is supported by listing multiple cluster variables.

{pmore}
If multiway inclusion-exclusion produces an indefinite covariance matrix,
xhdfe applies the positive-semidefinite correction to the slope block and
preserves that corrected block when repairing the reported constant.
The constant's cross-covariances are projected onto the retained slope space;
its variance is increased only as needed to make the complete covariance
positive semidefinite. Coefficients and degrees of freedom are unchanged.
The constant's standard error and contrasts involving it can change substantially;
{cmd:statstyle(reghdfe)} does not imply equality of that repaired row with reghdfe.
The policy applies in both tolerance modes and to ordinary and grouped models.
Numerically positive IV score covariance is classified before the ill-conditioned
covariance transformation, avoiding an unnecessary repair of a valid constant.

{pmore}
Each {it:clustervar} may be a variable or a categorical interaction with two or more components
specified with {cmd:#}. Two-component {cmd:##} expands to the two main dimensions plus their
interaction; {cmd:##} with more than two components is rejected (list the cluster dimensions explicitly).
Note that {cmd:reghdfe} (6.14.1) treats {cmd:a##b} in {cmd:vce(cluster)} as one joint cluster factor,
exactly like {cmd:a#b}; {cmd:xhdfe} keeps its documented expansion, so {cmd:vce(cluster a##b)} is
three-way clustering here and one-way clustering there. Use {cmd:a#b} for the joint factor in both.
Interaction clusters are coded after sample marking, with {cmd:egen group()} when all their components
are numeric and, when a component is a string, by sorting only the identifier columns in a temporary
frame ({cmd:egen group()} is the fallback). Plain numeric cluster identifiers are passed directly to the
plugin; string identifiers are coded by the same frame sort.

{pmore}
{cmd:robust} and {cmd:cluster()} are equivalent to {cmd:vce(robust)} and {cmd:vce(cluster ...)}; do not combine them with {cmd:vce()}.

{pmore}
{opt dk:raay} [{it:#}] computes Driscoll-Kraay standard errors using cross-sectional
score sums and the Bartlett kernel. Requires {cmd:tsset} or {cmd:xtset} with a time
variable. {it:#} is a positive integer bandwidth: {cmd:vce(dkraay 4)} includes
three lags. With {cmd:vce(dkraay)}, the bandwidth is
{it:floor(4*(T/100)^(2/9))+1}, capped at {it:T}, the number of observed periods in
the final estimation sample. An explicit bandwidth above {it:T} is also capped.

{pmore}
As in reghdfe 6.14.1, lags index chronologically ordered {it:observed} periods.
Entirely absent periods are not inserted as zero scores; this differs from a
calendar-gap HAC calculation. Unbalanced panels are supported. Missing times
are excluded, and periods removed with singletons do not enter {it:T}.
Weights enter the period score sums once; frequency weights represent literal
replication. Coefficients, absorption and FE recovery use the existing estimator.

{pmore}
Under the default {cmd:statstyle(reghdfe)}, the finite-sample multiplier is
{it:(N-1)/(N-df_m-df_a) * T/(T-1)} and inference uses
{it:min(N-df_m-df_a,T-1)} degrees of freedom; {cmd:ssc()} switches still apply.
FE nesting is evaluated against time. When FEs are nested in time, reghdfe's DK
convention omits the extra nested-constant subtraction used by cluster VCE;
thus bandwidth one equals time clustering when that subtraction is inactive.
With fewer than two periods, inference is unavailable as for one cluster.
The covariance includes the reported constant. Serial covariance is computed
on the CPU after the selected CPU or CUDA absorption.

{pmore}
For IV models with nearly collinear projected regressors, the period influences
and recovered constant retain the precision of the IV solve before the
Bartlett covariance is formed. This also covers common shifts of the regressors.

{phang}
{opth res:iduals(newvar)} saves regression residuals in a new variable.
{cmd:residuals} without parentheses is also accepted and saves residuals in {cmd:_reghdfe_resid},
matching reghdfe.
This is required for {cmd:predict, d}, {cmd:predict, xbd}, and {cmd:predict, dresiduals}.

{phang}
{opt keep:singletons} do not drop singleton observations. When enabled, inference may be biased;
xhdfe prints the reghdfe warning at
{browse "http://scorreia.com/reghdfe/nested_within_cluster.pdf":nested_within_cluster.pdf}.

{phang}
{opt nocon:stant} remove the intercept from the slope regression.

{phang}
{opt cons:tant} explicitly requests the default intercept treatment; do not
combine it with {cmd:noconstant}. {opt noa:bsorb} explicitly selects a model
without absorbed effects and cannot be combined with {cmd:absorb()}.


{marker opt_iv}{...}
{dlgtab:IV/2SLS}

{phang}
{opth endog:enous(varlist)} identifies endogenous regressors (subset of slope regressors).

{phang}
{opth instr:uments(varlist)} provides instruments for IV/2SLS.
Both {cmd:endogenous()} and {cmd:instruments()} must be specified together.
Factor-variable operators are not supported with IV, and IV is not supported in {cmd:group()} mode.

{pmore}
Redundant instrument columns are accepted when their retained span identifies the declared endogenous regressors.
An effective zero instrument span or an underidentified model returns an error.
Included exogenous regressors enter the instrument set automatically. If their
variable names also appear in {cmd:instruments()}, the extra entries are removed
before absorption on both CPU and CUDA; they do not count as excluded instruments.
Exactly proportional nonzero instruments are removed before absorption on both
CPU and CUDA, including columns with different variable names. Distinct, nearly
proportional instruments are retained. More general linear dependencies, or
those created only by absorption, can still trigger rank-deficiency errors on CUDA.

{phang}
{opt first} displays the regression of each endogenous variable on the included
exogenous regressors, excluded instruments, and absorbed fixed effects, followed
by the first-stage diagnostic summary. {opt ffirst} displays just the summary.
Both store {cmd:e(first)}, with one column per endogenous regressor and rows
{cmd:pr2}, {cmd:F}, {cmd:df}, {cmd:df_r}, and {cmd:pvalue}. The F-test jointly tests
the excluded-instrument coefficients; redundant restrictions are omitted.
Partial R-squared is {it:1 - RSS_unrestricted/RSS_restricted}, where the restricted
first stage includes the exogenous controls and fixed effects only.

{pmore}
The auxiliary regressions use the final IV estimation sample, weights, fixed
effects (the same {cmd:absorb()} terms, without FE output names), covariance estimator (including
{cmd:vce(dkraay} [{it:#}]{cmd:)} with the same bandwidth), DoF/SSC settings, and
selected CPU/CUDA backend.
They run only when requested, and preserve the structural coefficients,
covariance matrix, estimation sample, and other {cmd:e()} results. These options
perform additional xhdfe regressions and therefore add computation time.
Aliases such as {cmd:absorb(firm_effect=firm)} save the structural effects only;
the auxiliary regressions leave those variables unchanged.

{pmore}
F-statistics follow the same small-sample conventions as a direct xhdfe or
reghdfe first-stage regression followed by {cmd:test}. When fixed effects are
nested within clusters, reghdfe charges an additional intercept degree of
freedom relative to ivreghdfe; the resulting F-statistics can differ by that
known finite-sample factor. With multiple endogenous regressors, these are
per-equation relevance tests, not Sanderson-Windmeijer conditional tests or
Kleibergen-Paap weak-identification statistics. They do not test instrument validity.

{pmore}
With absorbed fixed effects, the reported {cmd:_cons} is the finite
normalization mean({it:y}) - mean({it:X})*{it:b}, under zero-mean absorbed
contributions. It is normalization-dependent and is not a structural IV
parameter.


{marker opt_dof}{...}
{dlgtab:Degrees-of-Freedom Adjustments}

{phang}
{opt dof:adjustments(list)}, also accepted as {opt dof(list)}, sets the absorbed DoF adjustments.
Tokens can be combined and separated by spaces or commas. If {cmd:dofadjustments()} is omitted,
the default is {cmd:all}. When supplied, only the listed tokens are enabled (reghdfe-style).
Valid tokens (case-insensitive) are {cmd:all}, {cmd:none}, {cmd:firstpair} ({cmd:first}),
{cmd:pairwise} ({cmd:pair}), {cmd:clusters} ({cmd:cluster}), and {cmd:continuous} ({cmd:cont}).
{cmd:all} enables the full reghdfe logic. {cmd:none} disables FE DoF correction. {cmd:firstpair}
and {cmd:pairwise} select the FE DoF method. {cmd:clusters} enables adjustments for clusters
nested within absorbed FEs. {cmd:continuous} enables reghdfe-compatible checks for redundant
continuous-interaction slopes. One further token, {cmd:exact}, is accepted only with both {cmd:group()} and
{cmd:individual()}: it computes the exact rank of the absorbed design (see
{help xhdfe##known_limitations:Known limitations}). The calculation is bounded: a design above its size limit
returns an error and no estimates, never an approximate count. Without {cmd:group()} and {cmd:individual()},
{cmd:dofadjustments(exact)} is an error.

{phang}
{opt ssc(str)} fixest-style small-sample corrections. Tokens are case-insensitive; separators may
be spaces or commas; parentheses are optional. Supported tokens:{p_end}

{p2colset 7 24 27 2}{...}
{p2col:{cmd:K.adj}}(0/1/true/false/yes/no; value omitted implies 1){p_end}
{p2col:{cmd:K.fixef}}({cmd:full}|{cmd:none}|{cmd:nonnested}|{cmd:non-nested}|{cmd:non_nested};
{cmd:true}/{cmd:yes} => full, {cmd:false}/{cmd:no} => none){p_end}
{p2col:{cmd:K.exact}}(0/1/true/false/yes/no; value omitted implies 1){p_end}
{p2col:{cmd:G.adj}}(0/1/true/false/yes/no; value omitted implies 1){p_end}
{p2col:{cmd:G.df}}({cmd:min}|{cmd:conventional}|{cmd:conv}){p_end}
{p2col:{cmd:t.df}}(numeric | {cmd:min} | {cmd:conventional}){p_end}
{p2colreset}{...}

{p 4 6 2}Example: {cmd:ssc(K.adj=1 K.fixef=nonnested G.df=min)}. The default is
{cmd:K.fixef=full} with {cmd:statstyle(reghdfe)} and {cmd:K.fixef=nonnested} with
{cmd:statstyle(legacy)}.{p_end}
{p 4 6 2}{cmd:t.df=min} sets df_r to N_clust-1; {cmd:t.df=conventional} sets df_r to df_r_unadj.{p_end}

{phang}
{opt statstyle(str)} selects the convention for fit statistics and small-sample scaling.
{cmd:reghdfe} (default) matches reghdfe's RMSE, adjusted R-squared, and cluster scaling.
{cmd:legacy} reproduces the pre-change xhdfe statistics.{p_end}

{phang}
{opth groupvar(newvar)} identifier for the first mobility group (1-based IDs). Requires at least
two absorbed FEs and {cmd:dofadjustments(all/firstpair/pairwise)}.


{marker opt_optimization}{...}
{dlgtab:Optimization}

{phang}
{opth tol:erance(#)} specifies the tolerance criterion for convergence; default is {cmd:tolerance(1e-8)}.

{phang}
{opt tolerancemode(str)} selects the absorber convergence mode.
{cmd:tolerancemode(reghdfe-comparable)} is the default since version 2.7.0: the accelerated absorber
stops when one full sweep moves the working data by less than {cmd:tolerance()} in relative norm —
the same meaning {helpb reghdfe} attaches to its tolerance — so coefficients match reghdfe at the
same nominal tolerance (down to the conditioning of the problem). Ordinary no-slope LSMR/MLSMR
paths use {cmd:min(effective tolerance,1e-11)} in comparable and strict-residual modes; requested
tolerances already at or below {cmd:1e-11} are unchanged. The {help xhdfe##reduced_pcg:reduced-system tier}
uses a confirmed acceptance-quantity target of {cmd:1e-12} when the requested tolerance is at least
{cmd:1.25e-12}; smaller requested tolerances retain its former stopping rule. Comparable CUDA {cmd:auto}
uses device LSMR on sparse ordinary unweighted two-FE graphs, with the same Krylov floor
and ordinary degrees of freedom. MAP/sweep, fast, group()/individual() and heterogeneous-slope routes, and the
effect-capturing sweep of savefe, retain their established stopping rules (a savefe fit can publish instead the
projection by the route of the mode; see {help xhdfe##opt_absorb:absorb()}).
{cmd:tolerancemode(xhdfe-fast)} restores the pre-2.7.0 stopping rule: typically ~1.5-3x fewer
absorber iterations than the comparable rule on the same route, with an effective precision that is
data-dependent and can be looser than the nominal tolerance on ill-conditioned (e.g. sparse bipartite)
designs — appropriate for exploration and for speed benchmarking (state the mode when citing timings).
For combined {cmd:group()}/{cmd:individual()} CUDA fits, xhdfe nevertheless applies an internal
{cmd:min(tolerance(),1e-12)} floor before certification; fast mode never authorizes a materially
inaccurate early GS stop there.
{cmd:tolerancemode(strict-residual)} is a heavier audit mode: it treats a final absolute check as
authoritative, may use additional iterations up to {cmd:maxiter()}, and reports non-convergence if that check
is not met. On the CPU the check is the largest absolute group mean of the absorbed outcome and of every
absorbed regressor for sweep methods, for fits with both {cmd:group()} and {cmd:individual()}, and for
ordinary LSMR or MLSMR projections without saved effects. In the latter case, a failed absolute check can trigger one
continuation in the same solver within the remaining {cmd:maxiter()} budget. Cached projections
are also checked. For the sparse direct and Krylov-PCG solvers, the solver's stopping rule and the
relative certificate apply.
An ordinary categorical CUDA fit checks the largest absolute group mean of the residuals it delivers,
using the fit's weights. With instruments it reads the structural residuals. With {cmd:savefe}
it reads the reconstruction from the saved effects. An unweighted fit in the automatic Comparable-based
Strict route can continue from its projected state. Other ordinary CUDA fits can take one device LSMR
projection on the original columns when the absolute check fails. Both continuations use the remaining
{cmd:maxiter()} budget. An eligible fit with {cmd:weightedcpuroute(on)} uses the CPU's strict fit.
Fits with both {cmd:group()} and {cmd:individual()} use their grouped-absorption checks.
The check of the delivered residuals is absolute, in the units of the dependent variable. When the residuals are
so large in those units that the tolerance is within a few units of roundoff of them (observed on one synthetic
panel, with {cmd:tolerance(1e-8)}, from residuals of about 2e7 to 4e7), the check may be unattainable in double
precision: the fit can then use iterations up to {cmd:maxiter()} and reports non-convergence, in some runs well
before that limit, and close to that point the same fit can be accepted in one run and refused in the next.
Rescale the dependent variable, raise {cmd:tolerance()}, or use {cmd:tolerancemode(reghdfe-comparable)}, whose
criteria are relative.
The mode used is returned in {cmd:e(tolerance_mode)}.
When ordinary no-slope LSMR/MLSMR runs, {cmd:e(krylov_internal_tolerance)} records
that internal value. {cmd:e(krylov_max_final_backward_error)},
{cmd:e(krylov_max_condition)}, and
{cmd:e(krylov_max_cond_backerr)} are deterministic maxima
over the outcome/design right-hand sides; the condition value is the maximum
running estimate observed by each solve before RHS aggregation. All four are
zero on non-Krylov paths. A nonfinite/DBL_MAX diagnostic is returned as Stata
missing rather than as a finite-looking extreme value.
The separate Krylov-PCG path is not covered by this parity floor and continues
to report zeros in these fields. No additional PCG work target is imposed.
The {help xhdfe##reduced_pcg:reduced-system tier} is a different conjugate-gradient solver with the
stopping rule described below; it leaves these four fields at zero itself.

{phang}
Independently of the stopping mode, xhdfe verifies the returned within transform against the
normal equations. {cmd:e(abs_residual)} is the largest absolute
{cmd:||D' W v_tilde||_2} across the outcome and transformed design columns;
{cmd:e(abs_residual_rel)} is the largest scale-normalized residual
{cmd:||D' W v_tilde||_2 / (||D' W||_F ||v||_2)}, where {cmd:v} is the
corresponding original, pre-absorption right-hand side.
{cmd:e(precision_certified)} is one when this explicit check is finite and no larger than the
documented numerical certificate limit. These diagnostics are computed after CPU, GPU, and
cache-backed absorption. For ordinary heterogeneous slopes, the authoritative certificate is
instead scale-invariant and undiluted: every ordinary-FE intercept and slope block is checked
separately against every transformed {it:y}/{it:X} right-hand side. It records the maximum
Frobenius-scaled, RMS diagonal-scaled, and worst-group moment in
{cmd:e(slope_block_residual_rel)} and the component diagnostics listed below; the legacy
{cmd:e(abs_residual_rel)} remains unchanged for compatibility. In the combined
{cmd:group()}/{cmd:individual()} path the explicit
certificate additionally checks the canonical preconditioned operator {cmd:A=W^(1/2) D C}, using
weighted residuals, moments {cmd:C D' W r}, and a Frobenius norm over active normalized columns.
Each right-hand side must pass either the consistent-system residual ratio
{cmd:||W^(1/2)r||/||W^(1/2)b||} or the least-squares optimality ratio
{cmd:||C D' W r||/(||A||_F ||W^(1/2)r||)}.
The public residual diagnostics above retain their established scale for compatibility, while
{cmd:e(precision_certified)} requires the canonical gate. Additional full sweeps are counted in
{cmd:e(iterations)}, and {cmd:e(converged)} is one only when {cmd:e(precision_certified)} is also
one. Heterogeneous-slope Auto/Auto-MLSMR requests whose first sweep solution misses the new gate
restart from the original {it:y}/{it:X} with the same GS/SGS method at bounded tighter internal tolerances; the public requested
{cmd:tolerance()} does not change, all iterations are cumulative, and exhaustion fails closed.
Explicit {cmd:convergence()} choices keep their stopping rule and receive the honest certificate
without adaptive continuation. Slope-free absorption is unchanged.

{phang}
Estimation returns error 498 if absorption lacks convergence or fails the
required precision checks. Unsuccessful recovery of explicitly saved effects
also returns error 498.

{phang}
Comparable mode can select conjugate gradients or a Krylov solver for
poorly connected multiway FE graphs. The accepted route and convergence
diagnostics are available under {help xhdfe##results:stored results}.

{phang}
{opt convergence(str)} selects the stopping criterion for iterative sweeps in ordinary and
heterogeneous-slope absorption. Direct and Krylov solvers use their own stopping tests.
{cmd:auto} (default) follows {cmd:tolerancemode()}: under
{it:reghdfe-comparable} the slope absorber stops on the reghdfe-style update criterion at the nominal
{cmd:tolerance()}, and under {it:xhdfe-fast} it uses the historical norm-change trigger. For Auto
requests in {it:reghdfe-comparable}, this trigger produces a candidate only: the independent
per-block certificate governs acceptance and may add visible same-method continuation sweeps.
In {it:xhdfe-fast}, the historical aggregate residual certificate governs acceptance; block checks
and block-driven continuation are skipped, including with an explicit {cmd:convergence()} rule.
An aggregate failure can still trigger same-method refinement or refusal.
The estimator, public tolerance and numerical failure guards are retained. Fast mode can therefore
return less accurate absorption than comparable mode on difficult slope designs.
{cmd:tolerancemode(strict-residual)} is not supported with heterogeneous slopes and exits with an
error before estimation; use {cmd:tolerancemode(reghdfe-comparable)} or {cmd:tolerancemode(xhdfe-fast)}
for those designs. Within supported tolerance modes, explicit values override the auto mapping:
{cmd:normchange} stops on the relative change in transformed-vector norms; {cmd:reghdfe} uses the
reghdfe-style vector update criterion based on the maximum weighted mean relative difference between
successive transformed {it:y}/{it:X} columns; {cmd:both} requires both criteria.
Aliases are {cmd:norm}, {cmd:norm_change}, {cmd:current} and {cmd:xhdfe} for
{cmd:normchange}; {cmd:vector}, {cmd:vectors}, {cmd:reghdfe-style} and
{cmd:reghdfe_style} for {cmd:reghdfe}.

{phang}
{opth fetol:erance(#)} specifies the tolerance criterion for fixed-effect recovery (savefe); default is {cmd:fetolerance(1e-6)}.
This tolerance is separate from {cmd:tolerance()} and is only used when the MAP recovery path runs.
It bounds the maximum absolute update in recovered effects, whereas the
absorption tolerance uses its selected convergence criterion. Their numeric
values therefore do not define interchangeable accuracy requirements.

{phang}
{opth ferecoverymethod(str)} selects the fixed-effect recovery strategy. Valid values are
{cmd:map} and {cmd:hybrid} (default). {cmd:hybrid} reuses cached alphas when available; {cmd:map}
forces recovery sweeps on the partial residual.

{phang}
{opth maxit:er(#)} specifies the maximum number of absorber iterations; default is {cmd:maxiter(100000)}.
{cmd:iterate(#)} is accepted as a reghdfe-compatible alias for {cmd:maxiter(#)}. The
{help xhdfe##reduced_pcg:reduced-system tier} runs at most {cmd:min(maxiter(),1500)} iterations; a declined
attempt is not charged to the budget of the route that follows it.

{phang}
{opt absorptionm:ethod(str)} sets the absorption method. Valid values (case-insensitive) are
{cmd:auto}; {cmd:gauss-seidel/gauss_seidel/gs};
{cmd:symmetric/sym/symgs/symmetric-gauss-seidel/symmetric_gauss_seidel};
{cmd:jacobi};
{cmd:mlsmr}/{cmd:modified-lsmr}/{cmd:modified_lsmr}/{cmd:within}/{cmd:within-additive}/{cmd:within_additive};
{cmd:lsmr}/{cmd:plain-lsmr}/{cmd:plain_lsmr}; and
{cmd:auto-mlsmr}/{cmd:auto_mlsmr}/{cmd:mlsmr-auto}/{cmd:mlsmr_auto}.
Explicit LSMR/MLSMR requests use matrix-free Krylov absorbers and are CPU-only.
{cmd:auto} first uses the probe-based auto-MLSMR selector on eligible CPU, standard-FE designs with up to three
fixed-effect dimensions. It promotes
slow-converging large designs to {cmd:mlsmr}; otherwise it resolves to a sweep fallback ({cmd:gauss-seidel} by
default, {cmd:symmetric-gauss-seidel} only with {cmd:symmetricsweep}). This avoids a second survey by the
Schwarz/Jacobi-PCG gate after the selector has already chosen a sweep. Under {cmd:tolerancemode(xhdfe-fast)},
MLSMR promotion is limited to large, many-RHS, multi-way moderate-rho designs. Designs with four or more fixed
effects stay on the full-data gate/sweep path so the 200k-sample probe cannot over-promote large, well-connected
many-way graphs (raise the cap with {cmd:XHDFE_AUTO_MLSMR_DEFAULT_MAX_FES}; {cmd:absorptionmethod(auto-mlsmr)}
is uncapped). Comparable CUDA {cmd:auto} selects device LSMR for ordinary unweighted
two-FE graphs identified by the existing low group-occupancy detector; it reports {cmd:lsmr}
as the method used. Fast, explicit methods/convergence and heterogeneous-slope paths, and the effect-capturing
sweep of savefe, retain their primary routing (a savefe fit can publish instead the projection by the route of
the mode; see {help xhdfe##opt_absorb:absorb()}). A broken ordinary Comparable CG recurrence
restarts with device LSMR within the remaining iteration budget.
Combined {cmd:group()}/{cmd:individual()} fits use their dedicated routing:
joint LSMR for CPU {cmd:auto}, the existing certified solver for CUDA {cmd:auto}, and an optional
certified sweep fallback if automatic CPU LSMR cannot be accepted. Disable MLSMR gate
promotions with {cmd:XHDFE_MLSMR_AUTO_GATE=0}. {cmd:absorptionmethod(mlsmr)} forces MLSMR; {cmd:auto-mlsmr}
runs the same probe-based selector explicitly. For standard FE designs,
{cmd:mlsmr}/{cmd:lsmr}/{cmd:auto-mlsmr} do not support {cmd:savefe} or heterogeneous slopes.
In group/individual mode only plain {cmd:lsmr} is supported, and only on CPU; explicit
{cmd:mlsmr}/{cmd:auto-mlsmr}/{cmd:jacobi} requests are rejected. The Schwarz/CG solver
({cmd:e(absorption_method_used)} = 4) is selected internally and cannot be requested:
{cmd:absorptionmethod(schwarz)} is an error.
{cmd:gs}/{cmd:gauss_seidel}, {cmd:sym}/{cmd:symgs}/{cmd:symmetric}/{cmd:symmetric_gauss_seidel},
{cmd:plain-lsmr}/{cmd:plain_lsmr}, {cmd:modified-lsmr}/{cmd:modified_lsmr}/{cmd:within}/{cmd:within-additive}/{cmd:within_additive},
and {cmd:auto_mlsmr}/{cmd:mlsmr-auto}/{cmd:mlsmr_auto} are aliases for the corresponding methods above.

{marker reduced_pcg}{...}
{pmore}
{it:Reduced-system tier} ({cmd:reduced-pcg}), on by default. Before the route chosen above runs, an ordinary
fit with {cmd:absorptionmethod(auto)}, {cmd:tolerancemode(reghdfe-comparable)}, the default {cmd:convergence()},
two to five absorbed fixed effects and no CUDA or Metal backend request ({cmd:gpubackend()} or
{cmd:XHDFE_GPU_BACKEND}), and without weights, instruments, heterogeneous slopes, saved fixed effects
({cmd:savefe}, {cmd:savefes()} or labelled {cmd:absorb()} terms), {cmd:groupvar()}, {cmd:group()} or the
writing of a mobility profile, first eliminates exactly the fixed effect with the most levels and solves for
the others by preconditioned conjugate gradients on the CPU, with the fit's threads, in at most
{cmd:min(maxiter(),1500)} iterations. For a requested tolerance of at least {cmd:1.25e-12}, a column stops
when its reduced-system acceptance quantity is at most {cmd:1e-12}, confirmed after recomputing the
reduced residual. Existing exits for trivial or compatible systems remain available. Below that
tolerance the former stopping criterion is unchanged. Passing this fixed stopping target does not
replace the independent final precision certificate.
The result is published only if it passes the certificate the other routes pass, including the requested
tolerance, and the unchanged acceptance trigger, with a verified relative residual
({cmd:e(abs_residual_rel)}) of at most 1e-11. A published fit reports {cmd:e(absorption_method_used)} = 8,
{cmd:e(reduced_pcg_status)} = 1 and the tier's iterations in {cmd:e(iterations)} (plus those of a normal-equation
correction, should one follow); the CPU {cmd:auto} retry does not run ({cmd:e(auto_retry_status)} is
{cmd:ineligible}). The method is reported, not requested: {cmd:absorptionmethod(reduced-pcg)} is an error.
With saved fixed effects the tier does not replace the effect-capturing sweep; under the conditions above it
can publish the projection that follows a certified sweep stopped above the floor of the mode (see
{help xhdfe##opt_absorb:absorb()}), and {cmd:e(iterations)} then also counts the sweep's.
The design follows {cmd:fereg} 0.6.0 (Schmieder 2026); see {help xhdfe##acknowledgements:Acknowledgements}.

{pmore}
Otherwise the attempt declines: {cmd:e(reduced_pcg_status)} = 2, {cmd:e(reduced_pcg_reason)} names the reason
(for example {cmd:stall}, {cmd:ceiling}, {cmd:projected}, {cmd:memory}, {cmd:certificate} or {cmd:trigger}) and
{cmd:e(reduced_pcg_discarded_iterations)} counts its iterations. Its work is discarded and the route runs from
the original data with its own {cmd:maxiter()} budget, so the estimates, {cmd:e(iterations)},
{cmd:e(absorption_method_used)} and the thread diagnostics are those of the route, as with the tier off. The
attempt needs working memory besides the data: it declines with reason {cmd:memory} when its estimate exceeds
the larger of 1 GiB and 16 bytes per observation for each of the outcome, the regressors and the absorbed
fixed effects, or when an allocation fails. A fit outside the conditions above, or whose solver a matching
mobility profile forces, reports {cmd:e(reduced_pcg_status)} = 0 and the condition in {cmd:e(reduced_pcg_reason)}.

{pmore}
With an absorption cache, an entry the tier published is read back as such ({cmd:e(reduced_pcg_status)} = 1,
reason {cmd:none}); another route's entry gives reason {cmd:cache}; while the tier is on for a fit, entries
written with it off or by an earlier version are not read, and while it is off, entries written with it on are
not read. For diagnosis, {cmd:XHDFE_REDUCED_PCG_TRACE=1} writes the tier's plan and decision to standard error
({cmd:2} adds the verified residual of each column of a published fit). {helpb xfepout} and the internal
regressions of {helpb xhdfeakm} and {helpb xhdfegelbach} never enter the tier.

{marker reduced_pcg_cuda}{...}
{pmore}
{it:CUDA variant} (experimental, off by default). The one exception to the condition of no CUDA request above: a
fit on CUDA ({cmd:gpubackend(cuda)} or {cmd:XHDFE_GPU_BACKEND=cuda}) that requests the variant with
{help xhdfe##opt_reducedpcgcuda:reducedpcgcuda(on)}, meets the other conditions and uses no mobility profile,
absorption cache or fixed-effect structure cache attempts the tier with the products of the reduced system
computed on the GPU. The recurrence, the stopping rule, the
acceptance trigger and the certificate are those described above, computed on the CPU. A published fit reports
{cmd:e(absorption_method_used)} = 8, {cmd:e(reduced_pcg_status)} = 1, {cmd:e(gpu_used)} = 1 and
{cmd:e(gpu_status)} = {cmd:used}; requesting CUDA alone does not prove entry, so check all of them. A declined
attempt runs the CUDA route that the fit takes without the request. A CUDA fit without the request reports
{cmd:e(reduced_pcg_reason)} = {cmd:gpu}; with the request, a fit kept out by a mobility profile or a cache
reports {cmd:mobility_profile}, {cmd:absorption_cache} or {cmd:fe_structure_cache} (the first that applies, in
that order; a file {cmd:xhdfe_mobility_profile.txt} in the working directory is a mobility profile) and runs that
same route. An attempt that cannot obtain device memory, or cannot page-lock the rows it exchanges with the GPU,
declines with {cmd:e(reduced_pcg_reason)} = {cmd:memory} or {cmd:pinned} and runs that same route.
Where that confirmed stopping rule is in force (a requested tolerance of at least {cmd:1.25e-12} and
{cmd:XHDFE_REDUCED_PCG_CHAT_STOP} not {cmd:0}), the product used to recompute the reduced residual is computed on
the GPU as well; the stopping decisions stay on the CPU. {cmd:XHDFE_REDUCED_PCG_CUDA_REPLACEMENT=0}, set before
starting Stata (read once per process), keeps that product on the CPU. The acceptance trigger and the certificate,
computed on the CPU, are unchanged.

{pmore}
For comparison with the former stopping rule, set {cmd:XHDFE_REDUCED_PCG_CHAT_STOP=0} before starting
Stata. This process switch restores the former stopping criterion; it does not switch the tier off or change its
acceptance trigger or certificate. If the confirmed target is not reached, the attempt declines to the
unchanged route. The switch is read once per process; requested tolerances below {cmd:1.25e-12} retain
the former rule with either setting.

{marker opt_reducedpcg}{...}
{pmore}
{opt reducedpcg(on|off)} switches the tier on or off for this fit, inside the session. Without the option,
the Stata global macro {cmd:XHDFE_REDUCED_PCG} decides: {cmd:global XHDFE_REDUCED_PCG 0} turns the tier off
for the following fits, {cmd:global XHDFE_REDUCED_PCG 1} turns it on, and {cmd:macro drop XHDFE_REDUCED_PCG}
(or any other value) leaves the decision to the process default. The process default is the environment
variable {cmd:XHDFE_REDUCED_PCG}, read once per Stata session ({cmd:0} turns the tier off for every fit that
the option and the global leave undecided, which is convenient for scripts), and otherwise on. The order is
therefore: the option, the global, the environment variable, on. A fit with the tier switched off reports
{cmd:e(reduced_pcg_status)} = 0 and {cmd:e(reduced_pcg_reason)} = {cmd:off}, and gives the results of the
tier switched off for the whole session. The auxiliary first-stage regressions of {cmd:first}, {cmd:ffirst}
and {cmd:estat firststage} follow the global and the environment variable, not the {cmd:reducedpcg()} option of
the IV command (whose own absorption never enters the tier). {helpb xfepout} has no such option.

{marker opt_reducedpcgcuda}{...}
{pmore}
{opt reducedpcgcuda(on|off)} requests the experimental {help xhdfe##reduced_pcg_cuda:CUDA variant} of the tier
for this fit, inside the session. It acts only on a fit on CUDA ({cmd:gpubackend(cuda)} or
{cmd:XHDFE_GPU_BACKEND=cuda}) and is ignored otherwise;
{cmd:reducedpcg(off)}, its global or its environment variable keeps both the tier and its CUDA variant off.
Without the option, the Stata global macro {cmd:XHDFE_REDUCED_PCG_CUDA} decides: {cmd:1} requests the variant
for the following fits, {cmd:0} does not, and {cmd:macro drop XHDFE_REDUCED_PCG_CUDA} (or any other value)
leaves the decision to the process default. The process default is the environment variable
{cmd:XHDFE_REDUCED_PCG_CUDA}, read once per Stata session ({cmd:1} requests the variant for every fit that the
option and the global leave undecided), and otherwise off. The order is therefore: the option, the global, the
environment variable, off. The auxiliary first-stage regressions of {cmd:first}, {cmd:ffirst} and
{cmd:estat firststage} follow the global and the environment variable, not the {cmd:reducedpcgcuda()} option of
the IV command. {helpb xfepout} has no such option.

{phang}
{opt sym:metricsweep} performs a forward + backward sweep per iteration for Gauss-Seidel methods
and can influence {cmd:absorptionmethod(auto)} selection when multiple FEs are present.

{phang}
{opt jacobirel:axation(#)} Jacobi relaxation parameter (<= 0 uses default 2/(J+1), where J is the
number of FE dimensions; values above 1 are capped at 1).

{marker opt_parallel}{...}
{dlgtab:Parallel execution}

{phang}
{opth numthr:eads(#)} requested OpenMP threads (0 = auto). A positive request
is authoritative and bypasses the automatic row-count, FE-shape, and
{cmd:maxthreads()} heuristics. It is clamped only to the logical processors
and OpenMP runtime limit visible to the process.

{phang}
{opth defaultthr:eads(#)} default threads when auto-threading is enabled.

{phang}
{opth maxthr:eads(#)} maximum threads when auto-threading is enabled.
It never reduces a positive {cmd:numthreads()} request.

{phang}
{opth minparallelrows(#)} minimum number of estimation rows for an automatic thread team; the default is
{cmd:minparallelrows(20000)} and # must be positive. Without a positive {cmd:numthreads()}, a fit with fewer
rows runs on one thread ({cmd:e(thread_limit_code)} 2, {cmd:auto_policy}); a positive {cmd:numthreads()} is not
affected. The first-stage fits of IV estimation use the same value.

{phang}
{opth targetrowsperthread(#)} rows per thread when auto-threading sizes the team.
The default, {cmd:targetrowsperthread(0)}, is automatic: one thread per 25,000 rows
when the fixed effects are absorbed without being saved, within the problem-size
cap (one thread below 100,000 rows, at most 8 below 250,000 and at most 16
otherwise), and one thread per 500,000 rows when they are saved ({cmd:savefe},
{cmd:savefes()}) or nothing is absorbed. A positive value applies that fixed
number of rows per thread on every path; {cmd:targetrowsperthread(500000)}
restores the automatic rule of version 2.30.1. Setting
{cmd:XHDFE_AUTO_ROWS_PER_THREAD} to a positive integer before Stata starts
replaces the 25,000.

{phang}
{bf:Simultaneous sessions:} on Linux and macOS, the automatic teams of one
user's simultaneous fits share the machine. Each fit whose team has more than
one thread records it in a small file of that user ({cmd:xhdfe-teams-}{it:uid}
in {cmd:/dev/shm}, or in the temporary directory where {cmd:/dev/shm} does not
exist) when it starts and again when its absorption starts. An automatic fit
that absorbs fixed effects without saving them then takes the smallest of its
automatic team, the logical processors divided by the number of fits running,
and the processors not taken by the absorptions of the other fits, but never
fewer threads than {cmd:targetrowsperthread(500000)} would give. A positive
{cmd:numthreads()} is never reduced; it is recorded so that automatic fits
leave room for it. A fit running alone keeps its automatic team. When the
coordination lowered the team of at least one phase of the fit (its start or
its absorption), {cmd:e(thread_limit_code)} is 6 and
{cmd:e(thread_limit_reason)} is {cmd:auto_team_coordination}; the team then
depends on what else the user is running, so the last bits of the estimates
can differ from those of the same fit run alone. The parallel reading of the
data precedes the fit and keeps its automatic team, so in a batch
{cmd:e(threads_used)} can exceed {cmd:e(threads_effective)}. Other users and
other programs are not seen. Setting {cmd:XHDFE_AUTO_TEAM_COORDINATION=0}
before Stata starts turns the coordination off; it is not available on other
platforms, such as Windows. {cmd:XHDFE_AUTO_TEAM_REGISTRY_DIR}, set to an
absolute directory, moves the file there; it is meant for diagnostics and
tests.

{phang}
{bf:OpenMP:} parallel absorption requires a plugin built with OpenMP support.
If a positive request above one is made against a serial build, xhdfe exits
with an error rather than reporting fictitious parallelism. {cmd:e(threads_used)}
is the largest OpenMP team observed while executing a real xhdfe loop;
{cmd:e(parallel_workers_active)} is the largest number of distinct workers
that processed useful units in the same observed region.
Both include the parallel reading of the data where the plugin uses it, whose
sample selection scans the whole data range, so {cmd:e(threads_used)} can
exceed {cmd:e(threads_effective)} when {cmd:if}/{cmd:in} or singleton dropping
leaves the estimation sample much smaller than the data.
On Linux, the build script will try OpenMP automatically; to force it (recommended for performance), from the repository root:

{phang2}
{cmd:. bash stata/tools/build-plugin.sh --linux --openmp}


{marker opt_reporting}{...}
{dlgtab:Reporting}

{phang}
{opt level(#)} sets confidence level; default is {cmd:level(95)}. Values must be between 0 and 100.
{cmd:l(#)} is accepted as a shorthand.

{phang}
{opt noheader}, {opt notable}, and {opt nofootnote} suppress the header block, coefficient table,
and absorbed-DoF footnote table, respectively. These affect display only; estimation results are still stored.

{phang}
{opt noomitted}, {opt noempty}, {opt baselevels}, {opt allbaselevels}, {opt cformat()},
{opt pformat()}, {opt sformat()}, {opt fvwrap()}, {opt vsquish},
{opt nolstretch}, {opt nofvlabel}, {opt noci}, and {opt nopvalues} are forwarded to Stata's
{cmd:ereturn display} layer to keep the coefficient table behaviour close to {cmd:reghdfe}.
{opt fvwrapon()} is accepted for compatibility and has no effect on display.

{phang}
{opt nosample} prevents {cmd:xhdfe} from posting {cmd:e(sample)}. This reduces memory use, but
postestimation commands that rely on {cmd:e(sample)} may not be available afterward.


{marker opt_diagnostics}{...}
{dlgtab:Profiles and caches}

{phang}
{opt mobilityprofile} writes a method-selection profile for the FE structure.
The default path is {cmd:xhdfe_mobility_profile.txt} in the working directory.
{opth mobfile(filename)} sets the path and enables reuse of a matching profile
unless {opt mobilityprofile} requests a new one. Aliases are {opt mobilityfile()}
and {opt mobilityprofilefile()}. A profile may guide a fit with another outcome
if the relevant signature matches. Writing a profile can make the reduced-system
tier ineligible. Group/individual profiles have their own scope. They keep CPU
Auto on the joint LSMR route and CUDA Auto on the grouped device solver.

{phang}
{opth absorptioncache(filename)} specifies a cache of transformed outcome and
regressors. With a profile path and caching enabled, the default cache path is
the profile path with {cmd:.absorption_cache} appended. Reuse requires matching
inputs, sample, weights, intercept handling, FE structure and solver contract,
including backend, requested tolerance, methods and threads. CUDA absorption,
heterogeneous slopes and combined group/individual fits skip transform-cache
reuse.

{phang}
{opth fescache(filename)} caches only the FE structure and singleton filter.
Aliases are {opt festructurecache()} and {opt festructurecachefile()}.
The bare flag {opt festructurecache} instead selects the default path beside
{cmd:xhdfe.ado}. Specify one form. The structure may be reused when the
outcome or regressors change if the marked sample and FE signature still match.
This cache is unavailable with heterogeneous slopes or combined
{cmd:group()}/{cmd:individual()}. Use distinct files when combining the
two caches. Combined fits use their own absorption-cache signature, based
on the original FE identifiers and singleton policy. To refresh older
entries in this mode, use {cmd:abscachemode(auto)} or {cmd:abscachemode(write)}.

{phang}
{opt abscachemode(mode)} (alias {opt absorptioncachemode()}) and
{opt fecachemode(mode)} (aliases {opt fescachemode()} and
{opt festructurecachemode()}) control the corresponding cache.
{cmd:off} disables that cache, {cmd:read} uses a valid entry without writing,
{cmd:write} computes and saves a fresh entry, and {cmd:auto} reads a valid
entry or computes and can replace a missing, stale or invalid entry.
An absorption-cache write requires a successful, certified result.
Disabling the absorption cache leaves mobility profile selection active.
See {help xhdfe_examples##caches:a cache example}.

{phang}
For debugging, {cmd:XHDFE_FE_DIAG_FILE} appends FE and residual diagnostics
to its path, and {cmd:XHDFE_PROFILE_SAVEFE} enables stage timings for saved FE
recovery and writeback.

{marker postestimation}{...}
{title:Postestimation Syntax}

{pstd}
{cmd:xhdfe} stores estimation results for commands such as {cmd:estimates store},
{cmd:estimates table}, {cmd:test}, {cmd:lincom} and {cmd:nlcom}.
{cmd:margins} uses the default {cmd:xb} prediction. Its disallowed prediction
types include {cmd:residuals} and {cmd:score}.

{pstd}
Use {cmd:estat summarize} for sample summaries, {cmd:estat vce} for the posted
covariance matrix and {cmd:estat ic} for Stata's information criteria with
{cmd:df(e(df_m)+1)}. IV fits also support {cmd:estat firststage}.

{pstd}
{cmd:estat firststage} reports the first-stage summary and returns its matrix in
{cmd:r(first)} without replacing the IV estimates. It reuses {cmd:e(first)} when
available, or estimates the auxiliary regressions on {cmd:e(sample)} otherwise.
{cmd:r(first_gpu_used)} reports GPU use in the unrestricted and restricted first stages.
{cmd:estat firststage, first} refits and displays the full first-stage regressions.
{opt level(#)} sets the confidence level of those displays.
With {cmd:nosample}, request {cmd:first} or {cmd:ffirst} during estimation to retain
the summary for later use; re-estimating full first stages requires the sample.
Replay with {cmd:xhdfe, ffirst} or {cmd:xhdfe, first} is also supported.

{pstd}
The syntax of {it:predict} is:

{p 8 16 2}
{cmd:predict}
{newvar}
{ifin}
[{cmd:,} {it:statistic}]
{p_end}{col 23}Some statistics require {cmd:residuals()}; see {help xhdfe##opt_model:Model options}.
{col 23}Equation: y = xb + d_absorbvars + e

{synoptset 20 tabbed}{...}
{synopthdr:statistic}
{synoptline}
{syntab :Main}
{p2coldent: {opt xb}}xb fitted values; the default{p_end}
{p2coldent: {opt xbd}}xb + d_absorbvars (requires {cmd:residuals()}){p_end}
{p2coldent: {opt d}}d_absorbvars (requires {cmd:residuals()}){p_end}
{p2coldent: {opt r:esiduals}}residuals (requires {cmd:residuals()}){p_end}
{p2coldent: {opt dres:iduals}}y - xb (FE component + residual; requires {cmd:residuals()}){p_end}
{p2coldent: {opt sc:ore}}score; equivalent to {opt residuals} and requires {cmd:residuals()}{p_end}
{p2coldent: {opt stdp}}standard error of the prediction (xb component){p_end}
{synoptline}
{p2colreset}{...}
{p 4 6 2}although {cmd:predict} {help data_types:type} {help newvar} is allowed,
the resulting variable will always be of type {it:double}.{p_end}
{p 4 6 2}Except for {cmd:xb} and {cmd:stdp}, all implemented prediction statistics require that
you requested saved residuals at estimation. Use {cmd:if e(sample)} when only estimation
rows are wanted, especially for {cmd:xb}, {cmd:stdp} or {cmd:dresiduals}.{p_end}


{marker missing_features}{...}
{marker known_limitations}{...}
{title:Restrictions and numerical qualifications}

{p2colset 8 12 12 2}{...}
{p2col: -}Combined {cmd:group()}/{cmd:individual()} cannot be used with saved effects
or heterogeneous slopes. IV/2SLS cannot be used with {cmd:group()} and uses plain
numeric variables in the slope list.
The HAC option is the Bartlett {help xhdfe##opt_vce:Driscoll-Kraay VCE}.{p_end}
{p2col: -}With labelled absorbs, {cmd:savefe} or {cmd:savefes}, the accuracy of published
coefficients and of recovered effects can differ. When Comparable publishes a separate
within projection, the saved effects may reconstruct the outcome only to the FE recovery
accuracy. A fit that publishes the effect-capturing sweep carries that error in its
residuals. Explicit methods, IV, heterogeneous slopes and grouped fits can take this
route. See {help xhdfe##opt_absorb:absorb()} for the applicable rules.{p_end}
{p2col: -}{cmd:residuals()} stores the within-fit residual. In Strict mode CUDA stores
the residual used for its check, including reconstruction from saved effects when
applicable. On CPU without saved effects a further absorption at {opt tolerance()}
may form the stored residual.{p_end}
{p2col: -}Ill-conditioned CUDA fits may fail the precision check with error 498 and
return no estimates. This includes small, difficult group/individual designs for which
the CUDA result can be checked against a direct CPU projection. Ordinary unweighted
Comparable CUDA OLS without saved effects checks the original normal equations and
can apply one bounded correction. Use {opt gpubackend(cpu)} when the device fit is
refused.{p_end}
{p2col: -}Windows uses 53-bit intermediate arithmetic for an extended-precision
additive-cell projection. Its covariance can be inaccurate in near-perfect fits
(residual below a millionth of the signal), as in 2.31.1, even when the coefficients
and reported certificate agree.{p_end}
{p2col: -}When the {help xhdfe##reduced_pcg:reduced-system tier} does not reach its target,
the fit continues on the previous route after paying for the attempt. Against 2.31.1
this was measured on one hard three-effect design of ten million rows as 1.24 times
the time with 16 threads and 1.47 times with 4. {cmd:reducedpcg(off)} avoids the
attempt.{p_end}
{p2col: -}The reduced-system tier gains most with many threads. With one thread it can
be slower than 2.31.1 on panels where it needs tens of iterations.{p_end}
{p2col: -}An explicit thread request above the number of physical cores can be slower
than in 2.31.1 on the same panels; the request is honoured and reported, not
capped.{p_end}
{p2col: -}{cmd:savefe} with weights can take about three times the time and twice the
memory of 2.31.1 on sparse panels, as the price of the correction.{p_end}
{p2col: -}Weighted {it:reghdfe-comparable} fits on CUDA: the repair of 2.33 covers
analytic, probability and importance weights within a size band; outside it a
weighted fit on CUDA can still be certified outside the contract.
{cmd:weightedcpuroute(on)} avoids this at the cost of running on the CPU.{p_end}
{p2col: -}A CUDA library that fails to load is remembered for the life of the process;
recovering needs a new Stata session.{p_end}
{p2col: -}Building from source with {cmd:BUILD_SHARED_LIBS=ON} is not a supported
configuration in this release.{p_end}
{p2col: -}With {cmd:aggregation(sum)} and no ordinary absorbed effect, a separate
projection determines whether the membership columns span a constant. If they do
not, the model reports no {cmd:_cons} and uses the uncentred fit statistics.
By default the grouped absorbed-DoF count adjusts for the known constant
redundancy when applicable. Use {cmd:dofadjustments(exact)} to compute the
exact grouped rank within its size bound.{p_end}
{p2col: -}AKM/KSS centring and outcome-shift behaviour are described in
{helpb xhdfeakm}.{p_end}
{p2colreset}{...}

{marker examples}{...}
{title:Examples}

{pstd}Firm and year fixed effects with firm-clustered standard errors.{p_end}
{phang2}{cmd:. use "https://www.stata-press.com/data/r19/grunfeld.dta", clear}{p_end}
{phang2}{cmd:. xhdfe invest mvalue kstock, absorb(company year) vce(cluster company)}{p_end}

{pstd}See {helpb xhdfe_examples} for runnable public-data examples of OLS, multiple
fixed effects, weights, heterogeneous slopes, group outcomes, IV, postestimation,
caches, CUDA and companion commands.{p_end}

{marker results}{...}
{title:Stored results}

{pstd}
{cmd:xhdfe} stores the following in {cmd:e()}:

{synoptset 24 tabbed}{...}
{syntab:Scalars}
{synopt:{cmd:e(N)}}estimation observations; frequency weights count literal replicas{p_end}
{synopt:{cmd:e(N_full)}}observations before singleton dropping{p_end}
{synopt:{cmd:e(num_singletons)}}singleton count; frequency weights count literal replicas{p_end}
{synopt:{cmd:e(sumweights)}}sum of weights (or {cmd:e(N)} if unweighted){p_end}
{synopt:{cmd:e(drop_singletons)}}1 if the singleton-removal policy is enabled, 0 with {cmd:keepsingletons}{p_end}
{synopt:{cmd:e(df_r)}}residual degrees of freedom (post-SSC){p_end}
{synopt:{cmd:e(df_r_unadj)}}unadjusted residual df (n - k - df_a){p_end}
{synopt:{cmd:e(df_m)}}model degrees of freedom{p_end}
{synopt:{cmd:e(rank)}}effective model rank (omitted coefficients excluded){p_end}
{synopt:{cmd:e(df_a)}}absorbed degrees of freedom (reghdfe-style){p_end}
{synopt:{cmd:e(df_a_levels)}}absorbed levels (M){p_end}
{synopt:{cmd:e(df_a_exact)}}absorbed levels adjusted for collinearity (M_exact){p_end}
{synopt:{cmd:e(df_a_nested)}}absorbed levels nested within clusters (if any){p_end}
{synopt:{cmd:e(df_a_initial)}}absorbed levels before redundancy adjustments (M){p_end}
{synopt:{cmd:e(df_a_redundant)}}redundant absorbed DoF (M - M_exact){p_end}
{synopt:{cmd:e(N_hdfe)}}number of absorbed FE dimensions{p_end}
{synopt:{cmd:e(N_hdfe_extended)}}number of absorbed FE dimensions (extended){p_end}
{synopt:{cmd:e(r2)}}overall R-squared{p_end}
{synopt:{cmd:e(r2_within)}}within R-squared{p_end}
{synopt:{cmd:e(r2_a)}}adjusted R-squared{p_end}
{synopt:{cmd:e(r2_a_within)}}adjusted within R-squared{p_end}
{synopt:{cmd:e(rss)}}residual sum of squares{p_end}
{synopt:{cmd:e(tss)}}total sum of squares{p_end}
{synopt:{cmd:e(tss_within)}}within total sum of squares{p_end}
{synopt:{cmd:e(mss)}}model sum of squares{p_end}
{synopt:{cmd:e(saturated)}}1 if the fitted design is saturated/perfect-fit, 0 otherwise{p_end}
{synopt:{cmd:e(sigma2)}}estimated residual variance{p_end}
{synopt:{cmd:e(rmse)}}root mean squared error{p_end}
{synopt:{cmd:e(F)}}model F-statistic (if defined){p_end}
{synopt:{cmd:e(p)}}model F p-value{p_end}
{synopt:{cmd:e(ll)}}log-likelihood{p_end}
{synopt:{cmd:e(ll_0)}}log-likelihood at zeroed regressors (within){p_end}
{synopt:{cmd:e(iterations)}}absorption iterations; with {cmd:e(absorption_method_used)} = 8, those of the
reduced-system tier and of any correction after it (a declined attempt's are in
{cmd:e(reduced_pcg_discarded_iterations)}); with {cmd:savefe}/{cmd:savefes} in {it:reghdfe-comparable}, when the
projection is published, also those of the effect-capturing sweep that precedes it{p_end}
{synopt:{cmd:e(ic)}}alias of {cmd:e(iterations)}{p_end}
{synopt:{cmd:e(converged)}}1 if converged, 0 otherwise{p_end}
{synopt:{cmd:e(abs_residual)}}verified absolute normal-equation residual after absorption{p_end}
{synopt:{cmd:e(abs_residual_rel)}}verified relative normal-equation residual after absorption{p_end}
{synopt:{cmd:e(krylov_internal_tolerance)}}ordinary no-slope LSMR/MLSMR internal solve tolerance; zero otherwise{p_end}
{synopt:{cmd:e(krylov_max_final_backward_error)}}maximum final Krylov test2/backward error across right-hand sides{p_end}
{synopt:{cmd:e(krylov_max_condition)}}maximum running Krylov condition estimate across right-hand sides; missing if nonfinite/DBL_MAX{p_end}
{synopt:{cmd:e(krylov_max_cond_backerr)}}maximum condition times backward error across right-hand sides{p_end}
{synopt:{cmd:e(auto_retry_policy_enabled)}}1 when the bounded CPU Auto retry policy is enabled{p_end}
{synopt:{cmd:e(auto_retry_eligible)}}1 when all retry eligibility gates passed{p_end}
{synopt:{cmd:e(auto_retry_fired)}}1 when the one cold MLSMR retry ran{p_end}
{synopt:{cmd:e(auto_retry_status_code)}}0 ineligible/disabled (also when the reduced-system tier published the fit),
1 below trigger, 2 retry used, 3 retry failed and primary returned, 4 primary left early and retry used;
its label is the macro {cmd:e(auto_retry_status)}{p_end}
{synopt:{cmd:e(auto_retry_primary_method)}}primary concrete absorption-method code{p_end}
{synopt:{cmd:e(auto_retry_primary_iterations)}}primary sweep iterations; with {cmd:left_early},
the sweep or conjugate-gradient iterations done before leaving{p_end}
{synopt:{cmd:e(auto_retry_primary_abs_res_rel)}}primary canonical certificate metric; with {cmd:left_early},
the predicted final certificate that triggered the leave (0 when the leave was at the hand-off to a conjugate-gradient stage){p_end}
{synopt:{cmd:e(auto_retry_iterations)}}cold MLSMR retry iterations{p_end}
{synopt:{cmd:e(auto_retry_abs_res_rel)}}retry canonical certificate metric{p_end}
{synopt:{cmd:e(auto_retry_elapsed_seconds)}}retry wall time in seconds{p_end}
{synopt:{cmd:e(slope_block_checked)}}1 if the heterogeneous-slope block check ran; 0 in fast mode or without slopes{p_end}
{synopt:{cmd:e(slope_block_residual_rel)}}heterogeneous slopes: maximum undiluted per-block/RHS certificate;
only meaningful when {cmd:e(slope_block_checked)} is 1 (block fields otherwise contain zero placeholders){p_end}
{synopt:{cmd:e(slope_block_frobenius_rel)}}heterogeneous slopes: maximum Frobenius-scaled block moment{p_end}
{synopt:{cmd:e(slope_block_rms_rel)}}heterogeneous slopes: maximum RMS diagonal-scaled block moment{p_end}
{synopt:{cmd:e(slope_block_max_rel)}}heterogeneous slopes: maximum worst-group block moment{p_end}
{synopt:{cmd:e(slope_block_skipped_max_rel)}}heterogeneous slopes: maximum diagnostic cosine in a rank-skipped group{p_end}
{synopt:{cmd:e(slope_certificate_worst_fe)}}0-based internal FE dimension attaining the slope gate; -1 if not applicable{p_end}
{synopt:{cmd:e(slope_certificate_worst_moment)}}0 for plain/intercept, 1 for slope; -1 if not applicable{p_end}
{synopt:{cmd:e(slope_accuracy_retry_stages)}}number of adaptive same-method slope continuation stages{p_end}
{synopt:{cmd:e(slope_accuracy_retry_iterations)}}slope continuation iterations beyond the primary candidate{p_end}
{synopt:{cmd:e(slope_internal_tolerance)}}last internal slope continuation tolerance; public {cmd:tolerance()} is unchanged{p_end}
{synopt:{cmd:e(precision_certified)}}1 if the selected mode's residual check meets its limit; fast slope fits use the aggregate check{p_end}
{synopt:{cmd:e(fe_recovery_converged)}}(only with {cmd:savefe}/{cmd:savefes}) 1 if fixed-effect recovery converged, 0 otherwise{p_end}
{synopt:{cmd:e(fe_recovery_iterations)}}(only with {cmd:savefe}/{cmd:savefes}) iterations used to recover the fixed effects{p_end}
{synopt:{cmd:e(fe_recovery_max_delta)}}(only with {cmd:savefe}/{cmd:savefes}) maximum change in the final FE recovery sweep{p_end}
{synopt:{cmd:e(report_constant)}}1 if the constant is reported, 0 if {cmd:noconstant}{p_end}
{synopt:{cmd:e(threads_requested)}}raw request; 0 denotes automatic selection{p_end}
{synopt:{cmd:e(threads_effective)}}request after the runtime-visible physical limit and auto policy{p_end}
{synopt:{cmd:e(threads_used)}}largest OpenMP team observed executing real xhdfe work{p_end}
{synopt:{cmd:e(parallel_workers_active)}}largest number of distinct workers processing useful work in one observed region{p_end}
{synopt:{cmd:e(thread_capacity)}}logical processors available after the OpenMP runtime limit{p_end}
{synopt:{cmd:e(openmp_enabled)}}1 when the loaded plugin was compiled with OpenMP{p_end}
{synopt:{cmd:e(thread_limit_code)}}0 none, 1 runtime capacity, 2 auto policy, 3 auto maxthreads, 4 OpenMP unavailable,
5 automatic request made inside an active OpenMP region, 6 automatic team shared with the same user's
simultaneous fits; its label is the macro {cmd:e(thread_limit_reason)}{p_end}
{synopt:{cmd:e(gpu_used)}}1 if GPU backend was effectively used, 0 if CPU path was used{p_end}
{synopt:{cmd:e(gpu_status_code)}}GPU status code: 0 not requested, 1 used, 2 unavailable, 3 not converged, 4 failed, 5 CPU cache/profile result,
7 weighted fit run on the CPU by design (see {help xhdfe##gpu_weighted_cpu_route:weighted fits on CUDA}){p_end}
{synopt:{cmd:e(gpu_attempted)}}1 if GPU absorption was attempted, 0 otherwise{p_end}
{synopt:{cmd:e(gpu_absorption_converged)}}1 if attempted GPU absorption converged, 0 if not, missing if not attempted{p_end}
{synopt:{cmd:e(gpu_absorption_iterations)}}GPU absorption iterations, missing if GPU absorption was not attempted{p_end}
{synopt:{cmd:e(cuda_tail_code)}}CUDA completion: 0 none, 1 GPU LSMR, 2 CPU MLSMR after the GPU sweeps{p_end}
{synopt:{cmd:e(absorption_method_used)}}absorption method code: 0 auto,
1 gauss-seidel, 2 symmetric-gauss-seidel, 3 jacobi, 4 Schwarz/CG, 5 LSMR,
6 MLSMR, 7 auto-MLSMR, 8 reduced-PCG (published by the reduced-system tier; never a request),
9 Krylov-PCG (reported when the Krylov-PCG route is forced, for example by a mobility profile that
suggests it; never a request). Codes 1 to 3 name the sweep selected for the fit; they are also reported
when a route without a code of its own served it: the exact projections of weighted designs whose
effects have few levels, the exact sparse solve and the conjugate-gradient accelerator of small path-like two-way
graphs, and the conjugate-gradient stage that can follow the GPU sweeps{p_end}
{synopt:{cmd:e(reduced_pcg_status)}}reduced-system tier: 0 not applicable (off, ineligible, not reached, or
another route's absorption read from the cache), 1 accepted (the tier published this fit, fresh or read back
from the absorption cache), 2 declined (the route ran from the original data){p_end}
{synopt:{cmd:e(reduced_pcg_discarded_iterations)}}iterations of a declined tier attempt; 0 otherwise{p_end}
{synopt:{cmd:e(N_clust)}}minimum number of clusters (if clustered){p_end}
{synopt:{cmd:e(dkraay_bw)}}effective Driscoll-Kraay bandwidth (lags + 1){p_end}
{synopt:{cmd:e(dkraay_lags)}}number of Driscoll-Kraay lags; with {cmd:vce(dkraay)}, {cmd:e(N_clust)} counts observed periods{p_end}
{synopt:{cmd:e(N_clustervars)}}number of cluster dimensions{p_end}
{synopt:{cmd:e(N_clust#)}}cluster counts by dimension{p_end}
{synopt:{cmd:e(cluster_scale)}}cluster scaling factor (SSC){p_end}
{synopt:{cmd:e(vcv_psd_fixed)}}1 if the multiway-cluster VCV required the positive-semidefinite
adjustment; the corrected slope block is preserved when repairing the constant, as described
under {help xhdfe##opt_vce:vce()}. A warning is printed.{p_end}
{synoptline}

{pstd}Notes:{p_end}
{phang} {cmd:e(df_a)} uses the exact FE rank ({cmd:e(df_a_exact)}) and does not follow {cmd:ssc_k_*}.{p_end}
{phang} {cmd:e(df_r_unadj)} = {it:N} - {it:k} - {cmd:e(df_a)}; {cmd:e(rmse)} = sqrt(rss/df_r_unadj).{p_end}
{phang} {cmd:e(r2_a)} and {cmd:e(r2_a_within)} use df_r_unadj.{p_end}
{phang} When {cmd:absorb()} contains only slope terms ({cmd:fe#c.x}, no level dimension), the
model has no intercept: no {cmd:_cons} row is reported and {cmd:e(tss)}/{cmd:e(r2)} use the
uncentered total sum of squares, matching {helpb reghdfe} and {helpb regress}{cmd:, noconstant}.{p_end}
{phang} Factor-variable regressors post the full Stata stripe in {cmd:e(b)}/{cmd:e(V)}, including
base levels (e.g. {cmd:1b.cat}) and omitted terms, matching {helpb reghdfe}.{p_end}
{phang} When the total sum of squares is zero (a constant dependent variable), {cmd:e(r2)} and
{cmd:e(r2_within)} are missing rather than one, matching {helpb regress} and {helpb reghdfe}.
{cmd:e(tss)} and {cmd:e(rss)} are still reported as zero.{p_end}
{phang} With {cmd:vce(cluster ...)} and fewer than two clusters, point estimates are still posted
but the standard errors, {it:t} statistics, {it:p}-values, confidence intervals and the model
{it:F} are missing, again matching {helpb regress} and {helpb reghdfe}. This is the same
convention already used for a saturated design. The displayed table therefore matches
{helpb reghdfe}, but the posted {cmd:e(V)} does not: xhdfe writes a small negative sentinel on the
diagonal so that Stata's {cmd:sqrt(V[j,j])} evaluates to missing, whereas {helpb reghdfe} posts
zero. Code reading {cmd:e(V)} directly in this situation will therefore see a negative variance;
read {cmd:_se[]} or check {cmd:e(N_clust)} instead.{p_end}
{phang} With absorbed fixed effects, {cmd:_cons} is the constant recovered by the FE normalization,
mean({it:y}) - mean({it:X})*{it:b}. When a one-way {cmd:vce(cluster)} nests an absorbed intercept
dimension (every cluster is a union of its levels; a slope-only term {cmd:fe#c.x} has no levels,
whereas {cmd:fe##c.x} keeps them), the cluster sums of the residuals are exactly zero, and the
{cmd:_cons} row and column of {cmd:e(V)} are exactly {it:m}'{it:V}{it:m} and -{it:V}{it:m}, where
{it:m} = mean({it:X}) and {it:V} is the slope block of {cmd:e(V)}; this applies on every absorption
route and backend, but not with {cmd:tolerancemode(xhdfe-fast)}, instruments or {cmd:vce(dkraay)}.
That variance is zero when {it:m} = 0 or there are no regressors: {cmd:_se[_cons]} is then 0,
and the coefficient table shows the standard error, {it:t}, {it:p} and confidence interval of
{cmd:_cons} as missing. This does
not mean a structural parameter known without uncertainty; the constant is a normalization of the
absorbed effects. Saturated designs and fewer than two clusters keep the conventions above.{p_end}
{phang} A converged fit never posts a non-finite coefficient, and never posts a non-finite
standard error for a coefficient whose inference is identified; such a result is rejected with an
error instead of being reported.{p_end}
{phang} Finite inputs can nevertheless overflow FP64 cross-products at extreme scales (roughly
{cmd:abs(X) >= 1e153}). {cmd:xhdfe} then exits with an explicit error instead of treating every
slope as collinear; rescale the dependent variable and/or regressors before fitting again.{p_end}

{synoptset 24 tabbed}{...}
{syntab:Macros}
{synopt:{cmd:e(cmd)}}{cmd:xhdfe}{p_end}
{synopt:{cmd:e(cmdline)}}command as typed{p_end}
{synopt:{cmd:e(title)}}main estimation title{p_end}
{synopt:{cmd:e(title2)}}secondary title reporting the number of absorbed FE groups{p_end}
{synopt:{cmd:e(depvar)}}name of dependent variable{p_end}
{synopt:{cmd:e(indepvars)}}names of independent variables{p_end}
{synopt:{cmd:e(endogenous)}}endogenous regressor list after IV estimation{p_end}
{synopt:{cmd:e(exogenous)}}included exogenous regressor list after IV estimation{p_end}
{synopt:{cmd:e(instruments)}}excluded instrument list after IV estimation{p_end}
{synopt:{cmd:e(firststage_options)}}options passed to the first-stage fits after IV estimation{p_end}
{synopt:{cmd:e(wtype)}}weight type, when weights are specified{p_end}
{synopt:{cmd:e(wexp)}}weight expression ({cmd:=} {it:exp}), when weights are specified{p_end}
{synopt:{cmd:e(absorb)}}absorbed FE list{p_end}
{synopt:{cmd:e(absvars)}}alias for absorbed FE list (reghdfe compatibility){p_end}
{synopt:{cmd:e(extended_absvars)}}extended absorbed FE list (reghdfe compatibility){p_end}
{synopt:{cmd:e(absorb_labels)}}labels shown for absorbed effects in output/savefes(){p_end}
{synopt:{cmd:e(dofmethod)}}DoF adjustment tokens applied; empty with {cmd:dofadjustments(exact)}{p_end}
{synopt:{cmd:e(dof_labels)}}labels used in the absorbed-DoF table, when posted{p_end}
{synopt:{cmd:e(clustvar)}}cluster variable list (if clustered){p_end}
{synopt:{cmd:e(vce)}}vcetype specified in {cmd:vce()}{p_end}
{synopt:{cmd:e(vcetype)}}title used to label Std. Err.{p_end}
{synopt:{cmd:e(dkraay_timevar)}}tsset/xtset time variable used by {cmd:vce(dkraay)}{p_end}
{synopt:{cmd:e(resid)}}residuals variable name (if requested){p_end}
{synopt:{cmd:e(group)}}group variable name (if requested){p_end}
{synopt:{cmd:e(individual)}}individual variable name (if requested){p_end}
{synopt:{cmd:e(aggregation)}}group aggregation method (if requested){p_end}
{synopt:{cmd:e(marginsnotok)}}prediction types disallowed by {cmd:margins}{p_end}
{synopt:{cmd:e(predict)}}program used to implement {cmd:predict}{p_end}
{synopt:{cmd:e(footnote)}}program used to print the absorbed-DoF footnote{p_end}
{synopt:{cmd:e(estat_cmd)}}program used to implement supported {cmd:estat} subcommands{p_end}
{synopt:{cmd:e(model)}}model label currently posted by xhdfe{p_end}
{synopt:{cmd:e(version)}}xhdfe version string{p_end}
{synopt:{cmd:e(tolerance_mode)}}absorber tolerance mode used by the command{p_end}
{synopt:{cmd:e(nowarn)}}posted when {cmd:nowarn} was specified{p_end}
{synopt:{cmd:e(auto_retry_status)}}label of {cmd:e(auto_retry_status_code)}: {cmd:ineligible}, {cmd:below_trigger},
{cmd:used}, {cmd:failed}, or {cmd:left_early}{p_end}
{synopt:{cmd:e(thread_limit_reason)}}label of {cmd:e(thread_limit_code)}: {cmd:none}, {cmd:runtime_capacity},
{cmd:auto_policy}, {cmd:auto_max_threads}, {cmd:openmp_unavailable}, {cmd:active_openmp_region}, or
{cmd:auto_team_coordination}{p_end}
{synopt:{cmd:e(gpu_backend_requested)}}requested backend from {cmd:gpubackend()} or the {cmd:XHDFE_GPU_BACKEND} environment variable, when provided;
{cmd:cuda} for a weighted fit run on the CPU by design ({cmd:e(gpu_status_code)} = 7){p_end}
{synopt:{cmd:e(gpu_backend)}}effective backend when {cmd:gpubackend()} is supplied,
or when the GPU was used via {cmd:XHDFE_GPU_BACKEND}, or {cmd:cpu} when {cmd:e(gpu_status_code)} = 7; otherwise not posted{p_end}
{synopt:{cmd:e(gpu_status)}}GPU status label: {cmd:not_requested}, {cmd:used},
{cmd:backend_unavailable}, {cmd:gpu_absorption_not_converged},
{cmd:gpu_backend_failed}, {cmd:cpu_cache_or_profile_result}, or {cmd:weighted_cpu_route}{p_end}
{synopt:{cmd:e(cuda_tail)}}label of {cmd:e(cuda_tail_code)}: {cmd:none}, {cmd:device-lsmr}, or {cmd:host-mlsmr}{p_end}
{synopt:{cmd:e(reduced_pcg_reason)}}reason behind {cmd:e(reduced_pcg_status)}: {cmd:none} when accepted; the
ineligibility (e.g. {cmd:weights}, {cmd:iv}, {cmd:gpu}; {cmd:mobility_profile}, {cmd:absorption_cache} or
{cmd:fe_structure_cache} when a profile or a cache keeps a {cmd:reducedpcgcuda()} request out) or decline (e.g.
{cmd:stall}, {cmd:trigger}) reason;
{cmd:off} when switched off; {cmd:not_reached} when the fit took a path that never reaches the tier;
{cmd:cache} when another route's absorption was read from the absorption cache
(a tier result read back reports {cmd:none}, like the fit that wrote it){p_end}
{synoptline}

{synoptset 24 tabbed}{...}
{syntab:Matrices}
{synopt:{cmd:e(first)}}first-stage partial R-squared, excluded-instrument F, numerator/denominator df, and p-value; with {cmd:first} or {cmd:ffirst}{p_end}
{synopt:{cmd:e(first_gpu_used)}}GPU use for unrestricted and restricted first stages, one column per endogenous regressor{p_end}
{synopt:{cmd:e(b)}}coefficient vector{p_end}
{synopt:{cmd:e(V)}}variance-covariance matrix of the estimators{p_end}
{synopt:{cmd:e(dof_table)}}absorbed DoF table by FE dimension (if any){p_end}
{synopt:{cmd:e(omit_reason)}}omitted-regressor codes (1=collinear with FEs, 2=other collinearity){p_end}
{synopt:{cmd:e(cluster_diag)}}counts for each cluster combination (bitmask order){p_end}
{synoptline}

{synoptset 24 tabbed}{...}
{syntab:Functions}
{synopt:{cmd:e(sample)}}marks estimation sample unless {cmd:nosample} was specified{p_end}
{synoptline}
{p2colreset}{...}


{marker contact}{...}
{title:Authors}

{pstd}Miguel Portela{break}
NIPE / Universidade do Minho and BPLIM / Banco de Portugal{break}
Email: miguel.portela@eeg.uminho.pt{break}
Website: {browse "https://reisportela.github.io":https://reisportela.github.io}{p_end}

{pstd}Tiago Tavares{break}
NIPE / Universidade do Minho{break}
Email: tgstavares@eeg.uminho.pt{break}
Website: {browse "https://www.tgstavares.com":https://www.tgstavares.com}{p_end}

{pstd}Only the listed human authors are authors or co-authors of {cmd:xhdfe}. No software tool or AI system is credited as an author or co-author.{p_end}


{marker support}{...}
{title:Support and updates}

{pstd}Released builds of {cmd:xhdfe} are installed via {cmd:net install} from the
online Stata site published by {cmd:xhdfe-xfe}:
{browse "https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata"}.{p_end}
{pstd}For local development checkouts or unzipped release bundles,
{cmd:net install} can also point at the local {cmd:stata/} folder that contains
{cmd:stata.toc}, {cmd:xhdfe.pkg}, and the platform plugin.{p_end}
{pstd}For building the plugin or running validation tests, see the repository README.{p_end}


{marker acknowledgements}{...}
{title:Acknowledgements}

{pstd}
xhdfe is a reimplementation built to validate against and interoperate with prior work. Full credit goes to:
{p_end}
{phang2}{bf:reghdfe} by Sergio Correia (Stata) - {browse "https://github.com/sergiocorreia/reghdfe":Github}{p_end}
{phang2}{bf:fixest} by Laurent Berge and contributors (R) - {browse "https://github.com/lrberge/fixest":Github}{p_end}
{phang2}{bf:pyfixest} by Alexander Fischer and collaborators (Python) - {browse "https://pyfixest.org/":pyfixest.org},
{browse "https://github.com/py-econometrics/pyfixest":Github}{p_end}
{phang2}{bf:FixedEffectModels.jl} by Matthieu Gomez and collaborators (Julia) - {browse "https://github.com/FixedEffects/FixedEffectModels.jl":Github}{p_end}
{phang2}{bf:within} by Alexander Fischer and Kristof Schroeder (Rust and Python) - {browse "https://github.com/py-econometrics/within":Github}{p_end}
{phang2}{bf:fereg} by Johannes F. Schmieder (Stata) - {browse "https://github.com/johannes-schmieder/fereg":Github}{p_end}
{phang2}{bf:fereg} by Leo Liu (Stata, Python and R; a different package with the same name) -
{browse "https://github.com/leoliu0/fereg":Github}{p_end}
{phang2}{bf:reghdfejl} by David Roodman (Stata front end to FixedEffectModels.jl) -
{browse "https://github.com/droodman/reghdfejl":Github}{p_end}

{pstd}
The {it:within} materials of Fischer and Schroeder, with their 2026 paper, were the direct starting point for
xhdfe's graph-preconditioned MLSMR/additive-Schwarz route: the factor-pair graph decomposition, the
bipartite-Laplacian representation, the approximate-Cholesky local solves and the additive-Schwarz
preconditioner inside modified LSMR are their contribution.{p_end}

{pstd}
The {help xhdfe##reduced_pcg:reduced-system tier} of version 2.32 ({cmd:reduced-pcg}) follows the design
implemented in {cmd:fereg} 0.6.0 and described in Schmieder (2026), and cites it as such: with two to five fixed
effects, it eliminates exactly the effect with the most levels and solves the remaining system by
Jacobi-preconditioned conjugate gradients. Which effect is eliminated, the range of two to five effects, the
joint treatment of up to eight columns per pass, the limit of 1,500 iterations for that stage and the form of
its stall test (the residual must halve within each window of iterations) follow {cmd:fereg}. As Schmieder
(2026) notes, the elimination is the block reduction of Somaini and Wolak (2016), and iterative solutions
have precedents in Abowd, Creecy and Kramarz (2002) and in Mittag (2019). xhdfe does not implement what
Schmieder (2026) presents as {cmd:fereg}'s distinguishing contribution in this step, the escalation to
stronger preconditioners, including combinatorial multigrid, on poorly connected data: where the
Jacobi-preconditioned iteration stalls, xhdfe hands the fit to its previous routes. xhdfe's part is an
independent C++ implementation inside its own routing, acceptance certificate, fallbacks, diagnostics and
Stata, Python and R front ends. xhdfe contains no {cmd:fereg} source code; its C++ implementation was
written after studying the paper and {cmd:fereg}'s public source. Two changes of version 2.31.0 followed benchmarks against {cmd:fereg}: reading the Stata
data in parallel on qualified hosts, as {cmd:fereg} does, and larger automatic thread teams (one thread per
25,000 rows instead of 500,000), where {cmd:fereg} by default uses all logical processors and sizes the
parallel work of each stage by its amount of work. {cmd:fereg} credits the precedent of parallel copying
from Stata to David Roodman's {cmd:julia} package (Roodman 2025).{p_end}

{pstd}
The performance work of November 2026 (versions 2.32 to 2.34) was measured against successive releases of
Schmieder's {cmd:fereg}, from 0.6.0 to 0.6.7, on public benchmark datasets and on large linked panels. In the
same month we studied {cmd:fereg} by Leo Liu, a different package with the same name for Stata, Python and R,
whose solver, in its author's words, "eliminates the largest fixed effect exactly and solves the remaining
system with a preconditioned conjugate-gradient method on cell-compressed sufficient statistics". We compared
it with xhdfe's reduced-system tier on a large panel to learn where the time of that tier goes. Version 2.34.1
contains no source code and no algorithmic change taken from that package; anything a later version adopts
from that study will be credited here. {cmd:reghdfejl}, by David Roodman, runs {cmd:FixedEffectModels.jl}
from Stata through his {cmd:julia} package and is a comparison engine in the Stata benchmarks.{p_end}

{pstd}
The numerical methods that xhdfe exposes are established ones and are cited as such in the
{help xhdfe##references:References}: partialling out by the Frisch-Waugh-Lovell theorem, alternating
projections as applied to fixed effects by Guimaraes and Portugal (2010) and Gaure (2013) and as implemented
in {cmd:reghdfe} (Correia 2017), conjugate gradients (Hestenes and Stiefel 1952), LSMR (Fong and Saunders
2011) and the Irons-Tuck acceleration (Irons and Tuck 1969). The approximate-Cholesky local solves of the
graph-preconditioned route are those that Fischer and Schroeder take from the Laplacian-solver literature
(Spielman and Teng 2014; Gao, Kyng and Spielman 2026). Multiway clustering follows Cameron, Gelbach and Miller
(2011) and {cmd:vce(dkraay)} follows Driscoll and Kraay (1998).{p_end}

{pstd}
Comparison packages, current releases checked at their sources on 21 November 2026: {cmd:reghdfe} 6.14.1
(8 July 2026), {cmd:fixest} 0.14.2 (CRAN, 26 June 2026), {cmd:pyfixest} 0.70.0 (8 November 2026),
{cmd:FixedEffectModels.jl} 2.2.1 (16 November 2026), {cmd:reghdfejl} 1.1.9 (12 October 2026), {cmd:fereg} by
Schmieder 0.6.7 (20 November 2026) and {cmd:fereg} by Liu 0.5.0 (21 November 2026).{p_end}

{pstd}
We thank Paulo Guimaraes (Banco de Portugal), Marta Silva
(Banco de Portugal), and Nelson Areal (EEG / UMinho) for valuable interactions across
different parts of the project. We especially thank Sergio Correia for feedback on
benchmarking, tolerances, and {cmd:reghdfe}-comparable validation. Nelson Areal and Miguel Portela presented
"Parallel and Cross-Language Computing: A Hands-On Workshop for Empirical Researchers" at BPLIM's Workshop on
{it:Speeding Up Empirical Research: Tools and Techniques for Fast Computing}
({browse "https://github.com/BPLIM/Workshops/tree/master/BPLIM2025":BPLIM2025}), where we
shared an earlier version of this proof of concept. We also
thank Universidade do Minho, Banco de Portugal, and FCT - Portuguese Foundation
for Science and Technology (UID/03182/2025) for financial support. All remaining errors are
ours. The usual disclaimer applies.{p_end}


{marker references}{...}
{title:References}

{pstd}
Selected references for high-dimensional fixed effects and related software include:{p_end}

{phang}
Portela, Miguel, and Tiago Tavares. 2026. "{cmd:xhdfe}: High-dimensional fixed effects
regression via a C++ backend." Version 2.34.1.
{browse "https://github.com/reisportela/xhdfe-xfe":https://github.com/reisportela/xhdfe-xfe}.{p_end}

{phang}
Cornelissen, Thomas. 2008. "The Stata command {cmd:felsdvreg} to fit a linear model with two
high-dimensional fixed effects." {it:Stata Journal} 8(2): 170-189.
{browse "https://doi.org/10.1177/1536867X0800800202":doi:10.1177/1536867X0800800202}.{p_end}

{phang}
Guimaraes, Paulo, and Pedro Portugal. 2010. "A simple feasible procedure to fit models with
high-dimensional fixed effects." {it:Stata Journal} 10(4): 628-649.
{browse "https://doi.org/10.1177/1536867X1101000406":doi:10.1177/1536867X1101000406}.{p_end}

{phang}
Gaure, Simen. 2013. "OLS with multiple high dimensional category variables."
{it:Computational Statistics & Data Analysis} 66: 8-18.
{browse "https://doi.org/10.1016/j.csda.2013.03.024":doi:10.1016/j.csda.2013.03.024}.{p_end}

{phang}
Correia, Sergio. 2016. "{cmd:reghdfe}: Estimating linear models with multi-way fixed effects."
2016 Stata Conference, Stata Users Group.
{browse "https://EconPapers.repec.org/RePEc:boc:scon16:24":https://EconPapers.repec.org/RePEc:boc:scon16:24}.{p_end}

{phang}
Correia, Sergio. 2017. "Linear models with high-dimensional fixed effects: An efficient and feasible
estimator." Working paper (the estimator that {cmd:reghdfe} implements).
{browse "http://scorreia.com/research/hdfe.pdf":http://scorreia.com/research/hdfe.pdf}.{p_end}

{phang}
Constantine, Noah, and Sergio Correia. 2021. "{cmd:reghdfe}: Stata module for linear and
instrumental-variable/GMM regression absorbing multiple levels of fixed effects."
{browse "https://ideas.repec.org/c/boc/bocode/s457874.html":https://ideas.repec.org/c/boc/bocode/s457874.html}.{p_end}

{phang}
Correia, Sergio, Paulo Guimaraes, and Tom Zylkin. 2020. "Fast Poisson estimation with
high-dimensional fixed effects." {it:Stata Journal} 20(1): 95-115.
{browse "https://doi.org/10.1177/1536867X20909691":doi:10.1177/1536867X20909691}.{p_end}

{phang}
Berge, Laurent R., Kyle Butts, and Grant McDermott. 2026. "Fast and user-friendly
econometrics estimations: The R package {cmd:fixest}." {it:arXiv preprint} arXiv:2601.21749.
{browse "https://doi.org/10.48550/arXiv.2601.21749":doi:10.48550/arXiv.2601.21749}.{p_end}

{phang}
The PyFixest Authors. "{cmd:pyfixest}: Fast high-dimensional fixed effect estimation in
Python." Python package.
{browse "https://pyfixest.org/":https://pyfixest.org/} and
{browse "https://github.com/py-econometrics/pyfixest":https://github.com/py-econometrics/pyfixest}.{p_end}

{phang}
FixedEffects.jl contributors. 2025. "{cmd:FixedEffectModels.jl}: Fast Estimation of Linear
Models with IV and High Dimensional Categorical Variables." Julia package.
{browse "https://github.com/FixedEffects/FixedEffectModels.jl":https://github.com/FixedEffects/FixedEffectModels.jl}.{p_end}

{phang}
Abowd, John M., Robert H. Creecy, and Francis Kramarz. 2002. "Computing person and firm effects using linked
longitudinal employer-employee data." Longitudinal Employer-Household Dynamics Technical Paper TP-2002-06,
U.S. Census Bureau.{p_end}

{phang}
Mittag, Nikolas. 2019. "A simple method to estimate large fixed effects models applied to wage determinants."
{it:Labour Economics} 61: 101766.
{browse "https://doi.org/10.1016/j.labeco.2019.101766":doi:10.1016/j.labeco.2019.101766}.{p_end}

{phang}
Somaini, Paulo, and Frank A. Wolak. 2016. "An algorithm to estimate the two-way fixed effects model."
{it:Journal of Econometric Methods} 5(1): 143-152.
{browse "https://doi.org/10.1515/jem-2014-0008":doi:10.1515/jem-2014-0008}.{p_end}

{phang}
Roodman, David. 2025. "Julia as a universal platform for statistical software development."
{it:Stata Journal} 25(2): 255-284.
{browse "https://doi.org/10.1177/1536867X251341105":doi:10.1177/1536867X251341105}.{p_end}

{phang}
Fischer, Alexander, and Kristof Schroeder. 2026. "Graph Preconditioning for High-Dimensional Fixed Effects
Regression." Mimeo, draft of June 2026.
{browse "https://github.com/py-econometrics/within-paper":https://github.com/py-econometrics/within-paper}.{p_end}

{phang}
Fischer, Alexander, and Kristof Schroeder. "{cmd:within}: Fast solvers for high-dimensional fixed effects."
Rust crate and Python package.
{browse "https://github.com/py-econometrics/within":https://github.com/py-econometrics/within}.{p_end}

{phang}
Schmieder, Johannes F. 2026. "{cmd:fereg}: Fast high-dimensional fixed-effects regressions in Stata."
CEPR Discussion Paper No. 22170. CEPR Press, Paris & London.
{browse "https://cepr.org/publications/dp22170":https://cepr.org/publications/dp22170}. Author's copy,
November 2026:
{browse "https://johannes-schmieder.com/papers/fereg.pdf":https://johannes-schmieder.com/papers/fereg.pdf}.
Software:
{browse "https://github.com/johannes-schmieder/fereg":https://github.com/johannes-schmieder/fereg}.{p_end}

{phang}
Liu, Leo. "{cmd:fereg}: Fast high-dimensional fixed-effects regressions for Stata, Python and R, on one
Rust core." {browse "https://github.com/leoliu0/fereg":https://github.com/leoliu0/fereg}.{p_end}

{phang}
Roodman, David. "{cmd:reghdfejl}: High-dimensional fixed-effect estimation in Stata using Julia."
{browse "https://github.com/droodman/reghdfejl":https://github.com/droodman/reghdfejl}.{p_end}

{pstd}
Numerical methods and inference:{p_end}

{phang}
Frisch, Ragnar, and Frederick V. Waugh. 1933. "Partial time regressions as compared with individual
trends." {it:Econometrica} 1(4): 387-401.
{browse "https://doi.org/10.2307/1907330":doi:10.2307/1907330}.{p_end}

{phang}
Lovell, Michael C. 1963. "Seasonal adjustment of economic time series and multiple regression analysis."
{it:Journal of the American Statistical Association} 58(304): 993-1010.
{browse "https://doi.org/10.1080/01621459.1963.10480682":doi:10.1080/01621459.1963.10480682}.{p_end}

{phang}
Hestenes, Magnus R., and Eduard Stiefel. 1952. "Methods of conjugate gradients for solving linear
systems." {it:Journal of Research of the National Bureau of Standards} 49(6): 409-436.
{browse "https://doi.org/10.6028/jres.049.044":doi:10.6028/jres.049.044}.{p_end}

{phang}
Irons, Bruce M., and Robert C. Tuck. 1969. "A version of the Aitken accelerator for computer iteration."
{it:International Journal for Numerical Methods in Engineering} 1(3): 275-277.
{browse "https://doi.org/10.1002/nme.1620010306":doi:10.1002/nme.1620010306}.{p_end}

{phang}
Fong, David Chin-Lung, and Michael Saunders. 2011. "LSMR: An iterative algorithm for sparse least-squares
problems." {it:SIAM Journal on Scientific Computing} 33(5): 2950-2971.
{browse "https://doi.org/10.1137/10079687X":doi:10.1137/10079687X}.{p_end}

{phang}
Spielman, Daniel A., and Shang-Hua Teng. 2014. "Nearly linear time algorithms for preconditioning and
solving symmetric, diagonally dominant linear systems."
{it:SIAM Journal on Matrix Analysis and Applications} 35(3): 835-885.
{browse "https://doi.org/10.1137/090771430":doi:10.1137/090771430}.{p_end}

{phang}
Gao, Yuan, Rasmus Kyng, and Daniel A. Spielman. 2026. "AC(k): Robust solution of Laplacian equations by
randomized approximate Cholesky factorization." {it:SIAM Journal on Scientific Computing} 48(3):
A1643-A1684. {browse "https://doi.org/10.1137/24M1673577":doi:10.1137/24M1673577}.{p_end}

{phang}
Cameron, A. Colin, Jonah B. Gelbach, and Douglas L. Miller. 2011. "Robust inference with multiway
clustering." {it:Journal of Business & Economic Statistics} 29(2): 238-249.
{browse "https://doi.org/10.1198/jbes.2010.07136":doi:10.1198/jbes.2010.07136}.{p_end}

{phang}
Driscoll, John C., and Aart C. Kraay. 1998. "Consistent covariance matrix estimation with spatially
dependent panel data." {it:Review of Economics and Statistics} 80(4): 549-560.
{browse "https://doi.org/10.1162/003465398557825":doi:10.1162/003465398557825}.{p_end}

{pstd}
Worker-firm (AKM) leave-out layer (see {helpb xhdfeakm} and {helpb xhdfegelbach}):{p_end}

{phang}
Saggio, Raffaele. "{cmd:LeaveOutTwoWay}: Leave-out estimation of variance components."
MATLAB package (the canonical Kline-Saggio-Soelvsten implementation).
{browse "https://github.com/rsaggio87/LeaveOutTwoWay":https://github.com/rsaggio87/LeaveOutTwoWay}.{p_end}

{phang}
Lamadon, Thibaut, and Adam A. Oppenheimer. "{cmd:pytwoway}: Two-way fixed-effect models for
worker-firm data in Python."
{browse "https://github.com/tlamadon/pytwoway":https://github.com/tlamadon/pytwoway}.{p_end}

{phang}
Abowd, John M., Francis Kramarz, and David N. Margolis. 1999. "High wage workers and high
wage firms." {it:Econometrica} 67(2): 251-333.{p_end}

{phang}
Andrews, Martyn J., Len Gill, Thorsten Schank, and Richard Upward. 2008. "High wage workers
and low wage firms: negative assortative matching or limited mobility bias?"
{it:Journal of the Royal Statistical Society A} 171(3): 673-697.{p_end}

{phang}
Kline, Patrick, Raffaele Saggio, and Mikkel Soelvsten. 2020. "Leave-out estimation of variance
components." {it:Econometrica} 88(5): 1859-1898.{p_end}

{phang}
Andrews, Isaiah, and Anna Mikusheva. 2016. "A geometric approach to nonlinear
econometric models." {it:Econometrica} 84(3): 1249-1264.{p_end}

{phang}
Gelbach, Jonah B. 2016. "When do covariates matter? And which ones, and how much?"
{it:Journal of Labor Economics} 34(2): 509-543. (Also the {cmd:b1x2} Stata package.){p_end}


{title:Also see}

{psee}
Companion command in the same package: {helpb xfepout} (partials out / residualizes
variables against multiple high-dimensional fixed effects on the same C++ core;
no coefficient table).{p_end}

{psee}
Worker-firm (AKM) post-estimation on the same backend: {helpb xhdfeakm}
(leave-out KSS variance decomposition, component standard errors,
weak-identification confidence intervals), {helpb xhdfeconnected}
(leave-one-out connected set as a sample-preparation utility) and
{helpb xhdfegelbach} (Gelbach decomposition).{p_end}
