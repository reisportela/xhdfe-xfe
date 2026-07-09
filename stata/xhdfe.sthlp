{smcl}
{* *! version 2.14.1 09jul2026}{...}
{vieweralsosee "[R] areg" "help areg"}{...}
{vieweralsosee "[R] xtreg" "help xtreg"}{...}
{vieweralsosee "" "--"}{...}
{vieweralsosee "reghdfe" "help reghdfe"}{...}
{vieweralsosee "ivreghdfe" "help ivreghdfe"}{...}
{vieweralsosee "ppmlhdfe" "help ppmlhdfe"}{...}
{vieweralsosee "" "--"}{...}
{vieweralsosee "xfe" "help xfe"}{...}
{vieweralsosee "xhdfeakm" "help xhdfeakm"}{...}
{vieweralsosee "xhdfeconnected" "help xhdfeconnected"}{...}
{vieweralsosee "xhdfegelbach" "help xhdfegelbach"}{...}
{vieweralsosee "" "--"}{...}
{viewerjumpto "Syntax" "xhdfe##syntax"}{...}
{viewerjumpto "Absorb syntax" "xhdfe##absorb"}{...}
{viewerjumpto "Description" "xhdfe##description"}{...}
{viewerjumpto "Development process" "xhdfe##development"}{...}
{viewerjumpto "Group FEs" "xhdfe##group_fes"}{...}
{viewerjumpto "Options" "xhdfe##options"}{...}
{viewerjumpto "Postestimation Syntax" "xhdfe##postestimation"}{...}
{viewerjumpto "Missing Features" "xhdfe##missing_features"}{...}
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
{bf:Least-square regressions (no fixed effects):}

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
{bf:IV/2SLS with absorbed fixed effects:}

{p 8 15 2} {cmd:xhdfe}
{it:depvar} [{it:exogvars}] {it:endogvars}
{ifin} {it:{weight}} {cmd:,} {opth a:bsorb(xhdfe##absorb:absvars)}
{opt endog:enous(endogvars)} {opt instr:uments(instvars)} [{help xhdfe##options_table:options}]{p_end}

{pstd}
{bf:Show package version:}

{p 8 15 2} {cmd:xhdfe, version}{p_end}


{marker options_table}{...}
{synoptset 22 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Standard FEs {help xhdfe##opt_absorb:[+]}}
{synopt: {opth a:bsorb(xhdfe##absorb:absvars)}}categorical fixed effects, labeled absorbs, two-way interactions, and heterogeneous slopes{p_end}
{synopt: {cmdab:a:bsorb(}{it:label}{cmd:=}{it:absvar}{cmd:)}}label an absorbed effect and use {it:label} as the saved-FE base name{p_end}
{synopt: {cmdab:a:bsorb(}{it:fe}{cmd:#c.}{it:x}{cmd:)}}absorb group-specific slopes without a level FE{p_end}
{synopt: {cmdab:a:bsorb(}{it:fe}{cmd:##c.}{it:x}{cmd:)}}absorb a level FE plus group-specific slopes{p_end}
{synopt: {cmdab:a:bsorb(}{it:...}{cmd:,} {cmd:savefe)}}save all fixed effect contributions with the {it:__hdfe*} prefix{p_end}
{synopt: {opth savefes(prefix)}}save fixed effect contributions using a prefix{p_end}
{synopt : }{bf:- note:} savefe/savefes are not supported when both {cmd:group()} and {cmd:individual()} are specified{p_end}

{syntab:Group FEs {help xhdfe##opt_group_fes:[+]}}
{synopt : {opth g:roup(xhdfe##opt_groupvar:groupvar)}}categorical variable representing each group (e.g. patent_id){p_end}
{synopt : }{bf:- note:} using {cmd:group()} without {cmd:individual()} is equivalent to running the regression on 1 observation per group{p_end}
{synopt : {opth i:ndividual(xhdfe##opt_indvar:indvar)}}categorical variable representing each individual whose fixed effect will be absorbed{p_end}
{synopt : }{bf:- note:} the {cmd:individual()} option requires {cmd:group()}{p_end}
{synopt : }{bf:- note:} {cmd:individual()} is appended to {cmd:absorb()} if not already listed{p_end}
{synopt : }{bf:- alias:} {cmd:i(}{it:indvar}{cmd:)} is accepted as a shorthand for {cmd:individual(}{it:indvar}{cmd:)}{p_end}
{synopt : {opth ag:gregation(xhdfe##opt_aggregation:str)}}aggregation for individual FEs within a group: {it:mean} (default), {it:sum}; {it:avg}/{it:average} are aliases for mean{p_end}

{syntab:Model {help xhdfe##opt_model:[+]}}
{synopt : {opth vce:(xhdfe##opt_vce:vcetype)}}{it:vcetype} may be {opt un:adjusted} (default), {opt r:obust}, or {opt cl:uster} {help fvvarlist} (multiway){p_end}
{synopt : {opt r:obust}}equivalent to {cmd:vce(robust)}{p_end}
{synopt : {opth cl:uster(varlist)}}equivalent to {cmd:vce(cluster ...)}; multiway clustering allowed; supports two-way interactions with {cmd:#}/{cmd:##}{p_end}
{synopt : {opth res:iduals(newvar)}}save regression residuals{p_end}
{synopt : {opt res:iduals}}save regression residuals in {cmd:_reghdfe_resid}{p_end}
{synopt : {opt keep:singletons}}do not drop singleton observations{p_end}
{synopt : {opt nocon:stant}}suppress the intercept in the slope regression{p_end}

{syntab:IV/2SLS {help xhdfe##opt_iv:[+]}}
{synopt : {opth endog:enous(varlist)}}endogenous regressors (subset of slopes){p_end}
{synopt : {opth instr:uments(varlist)}}instruments for IV/2SLS{p_end}

{syntab:Degrees-of-Freedom Adjustments {help xhdfe##opt_dof:[+]}}
{synopt :{opt dof:adjustments(list)}}select absorbed-DoF adjustment rules; default is {cmd:all}{p_end}
{synopt :{opt ssc(str)}}fixest-style small-sample corrections (K/G/t.df){p_end}
{synopt :{opt statstyle(str)}}fit-statistics style: {it:reghdfe} (default) or {it:legacy} (pre-change xhdfe){p_end}
{synopt: {opth groupv:ar(newvar)}}identifier for the first mobility group{p_end}

{syntab:Optimization {help xhdfe##opt_optimization:[+]}}
{synopt :{opt tol:erance(#)}}convergence tolerance (default 1e-8){p_end}
{synopt :{opt tolerancemode(str)}}absorber convergence mode: {it:reghdfe-comparable} (default since 2.7.0), {it:xhdfe-fast} (pre-2.7.0 fast trigger), or {it:strict-residual}{p_end}
{synopt :{opt convergence(str)}}stopping criterion for heterogeneous-slope absorption ({cmd:absorb(fe#c.x)}): {it:auto} (default; follows {cmd:tolerancemode()}), {it:normchange}, {it:reghdfe}, or {it:both}{p_end}
{synopt :{opt fetol:erance(#)}}fixed-effect recovery tolerance for {cmd:savefe}/{cmd:savefes} MAP fallback (default 1e-6){p_end}
{synopt :{opt ferecoverym:ethod(str)}}fixed-effect recovery method: {it:hybrid} (default) or {it:map}{p_end}
{synopt :{opt maxit:er(#)}}maximum absorber iterations (default 100000){p_end}
{synopt :{opt iter:ate(#)}}alias of {cmd:maxiter(#)} accepted for reghdfe compatibility{p_end}
{synopt :{opt absorptionm:ethod(str)}}absorption method (auto, gauss-seidel, symmetric-gauss-seidel, jacobi, mlsmr, lsmr, auto-mlsmr; aliases supported){p_end}
{synopt :{opt sym:metricsweep}}forward + backward sweep per iteration{p_end}
{synopt :{opt jacobirel:axation(#)}}Jacobi relaxation parameter (0 = default){p_end}

{syntab:GPU {help xhdfe##opt_gpu:[+]}}
{synopt :{opt gpubackend(str)}}backend selector: {it:cpu}, {it:cuda}, or {it:metal}; errors if a requested non-CPU backend is unavailable{p_end}

{syntab:Parallel execution {help xhdfe##opt_parallel:[+]}}
{synopt :{opt numthr:eads(#)}}threads used by the absorber (0 = auto){p_end}
{synopt :{opt defaultthr:eads(#)}}default threads when auto-threading is enabled{p_end}
{synopt :{opt maxthr:eads(#)}}maximum threads when auto-threading is enabled{p_end}
{synopt :{opt minparallelr:ows(#)}}minimum rows to enable parallel absorption{p_end}
{synopt :{opt targetrowsper:thread(#)}}target rows per thread{p_end}

{syntab:Reporting {help xhdfe##opt_reporting:[+]}}
{synopt :{opt l:evel(#)}}set confidence level; default is {cmd:level(95)}{p_end}
{synopt :{opt noheader}}suppress the header block; forwarded to the xhdfe display layer for reghdfe compatibility{p_end}
{synopt :{opt notable}}suppress the coefficient table; header/footnote behavior follows Stata e-class display conventions{p_end}
{synopt :{opt nofootnote}}suppress the absorbed-DoF footnote table{p_end}
{synopt :{opt noomitted}}hide omitted coefficients in the coefficient table{p_end}
{synopt :{opt noempty}}hide empty cells in factor-variable displays{p_end}
{synopt :{opt baselevels}}show base levels in factor-variable displays{p_end}
{synopt :{opt allbaselevels}}show all base levels in factor-variable displays{p_end}
{synopt :{opt cformat(fmt)}}coefficient display format; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt pformat(fmt)}}p-value display format; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt sformat(fmt)}}standard-error display format; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt fvwrap(#)}}factor-variable wrapping option; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt fvwrapon(style)}}factor-variable wrapping style; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt vsquish}}compact vertical display; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt nolstretch}}disable line stretching in table display; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt nofvlabel}}suppress factor-value labels in table display; forwarded to {cmd:ereturn display}{p_end}
{synopt :{opt noci}}hide confidence intervals in the coefficient table{p_end}
{synopt :{opt nopvalues}}hide p-values in the coefficient table{p_end}
{synopt :{opt nosample}}do not save {cmd:e(sample)}; reduces memory use but disables postestimation that requires the sample marker{p_end}
{synopt :{opt nowarn}}suppress singleton and compatibility notes; estimation is unchanged{p_end}

{syntab:Compatibility}
{synopt :{opt timeit}}accepted for reghdfe compatibility; currently ignored{p_end}
{synopt :{opt verbose(#)}}accepted for reghdfe compatibility; currently ignored{p_end}
{synopt :{opt technique(str)}}accepted for reghdfe compatibility; currently ignored{p_end}
{synopt :{opt acceleration(str)}}accepted for reghdfe compatibility; currently ignored{p_end}
{synopt :{opt transform(str)}}accepted for reghdfe compatibility; currently ignored{p_end}
{synopt :{opt preconditioner(str)}}accepted for reghdfe compatibility; currently ignored{p_end}
{synopt :{opt prune}}accepted for reghdfe compatibility; currently ignored{p_end}
{synopt :{opt fastregress}}accepted for reghdfe compatibility; currently ignored{p_end}
{synopt :{opt poolsize(#)}}accepted for reghdfe compatibility; currently ignored{p_end}
{synopt :{opt compact}}accepted for reghdfe compatibility; currently ignored{p_end}
{synopt :{opt version(#)}}accepted for reghdfe compatibility; currently ignored{p_end}
{synopt :{opt parallel(str)}}accepted for reghdfe compatibility; currently ignored{p_end}

{syntab:Diagnostics {help xhdfe##opt_diagnostics:[+]}}
{synopt :{opt mobilityprofile}}compute mobility profile and save to {cmd:xhdfe_mobility_profile.txt}{p_end}
{synopt :{opth mobfile(filename)}}read or save mobility profile file (aliases: {cmd:mobilityfile()}, {cmd:mobilityprofilefile()}){p_end}
{synopt :{opth absorptioncache(filename)}}read or save an explicit absorption cache file{p_end}
{synopt :{opt absorptioncachemode(str)}}absorption cache mode: off, auto, read, write{p_end}
{synopt :{opt festructurecache}}build/reuse a fixed-effect structure cache in the {cmd:xhdfe.ado} directory{p_end}
{synopt :{opth fescache(filename)}}read or save a fixed-effect structure cache file at an explicit path (aliases: {cmd:festructurecache()}, {cmd:festructurecachefile()}){p_end}
{synopt :{opt fescachemode(str)}}fixed-effect structure cache mode: off, auto, read, write{p_end}
{synopt : }{bf:- note:} when a profile file exists and matches the data, xhdfe auto-loads it to guide absorber ordering{p_end}
{synopt : }{bf:- note:} FE structure caches speed up repeated runs with the same {cmd:absorb()} structure; use {cmd:fescache()} for a fixed path{p_end}

{synoptline}
{p 4 6 2}{it:depvar} and {it:indepvars} may contain {help tsvarlist:factor variables} and time-series operators.
Factor-variable prefixes are supported in slopes. In {cmd:absorb()}, categorical interactions with
{cmd:#}/{cmd:##} are supported, as are heterogeneous slopes of the form {cmd:fe#c.x}
(group-specific slopes) and {cmd:fe##c.x} (group intercepts plus group-specific slopes).
In {cmd:cluster()}, categorical interactions with {cmd:#}/{cmd:##} are supported but continuous-factor
terms such as {cmd:c.x} are rejected. When using {cmd:endogenous()/instruments()},
factor-variable operators in the slope list are not supported.{p_end}
{p 4 6 2}Supported weights are {cmd:aweight}, {cmd:fweight}, {cmd:pweight}, and {cmd:iweight}. With {cmd:pweight} and no {cmd:vce()}, xhdfe defaults to robust SEs.{p_end}

{marker opt_gpu}{...}
{title:GPU option}

{phang}
{opt gpubackend(str)} selects the backend used by the fixed-effect absorber.

{phang2}
{it:cpu} forces the CPU backend.

{phang2}
{it:cuda} requests the CUDA backend when the plugin was built with CUDA support and a CUDA device is available;
otherwise xhdfe stops with an error instead of silently falling back to CPU.

{phang2}
{it:metal} is reserved for Metal-enabled builds. In builds where Metal is not available, requesting it
stops with an error instead of silently falling back to CPU.

{phang}
The online net-install and the release ZIPs ship {bf:CPU-only} plugins; the GPU cannot be obtained from the
online material. On a Linux machine with an NVIDIA GPU, the easiest way to get CUDA is the companion command
{helpb xhdfegpu}: run it once after {cmd:net install} and it detects the GPU, compiles a plugin for the local
architecture, and installs it over the CPU plugin in place (no file renaming). On a machine without internet
access, download the source zip and pass it in with {cmd:xhdfegpu, zip(}{it:path}{cmd:)}.

{phang}
To build the plugin by hand instead (Linux + NVIDIA only), first read your GPU's
compute capability:

{phang2}
{cmd:. nvidia-smi --query-gpu=compute_cap --format=csv,noheader}

{phang}
Drop the dot to get {cmd:XHDFE_CUDA_ARCH} (for example {cmd:9.0} maps to {cmd:90}, {cmd:8.6} to {cmd:86}; minimum
{it:75}). Then, from the xhdfe repository root:

{phang2}
{cmd:. XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH=90 bash stata/tools/build-plugin.sh --linux --openmp}

{phang}
CUDA builds require NVCC and compute capability {it:sm_75} or newer. Use {cmd:XHDFE_CUDA_ARCH} to set a single
target SM for your card (minimum: 75), or {cmd:XHDFE_CUDA_ARCHS} to set a multi-target list (comma/space/semicolon
separated), for example {cmd:"75,80,86,89,90"} for a shareable multi-GPU binary.

{phang}
After estimation, check {cmd:e(gpu_used)}, {cmd:e(gpu_backend)}, and {cmd:e(gpu_status)} to confirm whether CUDA
was actually used. If {cmd:gpubackend(cuda)} was requested but CUDA is unavailable, fails before a converged
absorption result, or returns a CPU cache/profile result, xhdfe exits with error 498 rather than silently returning
CPU results. Rebuild the plugin with CUDA support for the intended architecture and ensure the requested backend is
available at runtime. If the plugin was rebuilt or switched during the current Stata session, run {cmd:discard}
(with no arguments) before rerunning the command.

{phang}
Practical rule: after rebuilding {cmd:xhdfe.plugin}, after switching between CPU and CUDA plugin binaries, or after
changing the active {cmd:adopath} entry that provides {cmd:xhdfe.ado}/{cmd:xhdfe.plugin}, run {cmd:discard}
(not {cmd:discard xhdfe}) before rerunning {cmd:xhdfe}.

{phang}
Ordinary repeated calls in the same Stata session do {bf:not} require {cmd:discard} when the plugin binary itself
has not changed; for example, rerunning {cmd:xhdfe} first on CPU and then with {cmd:gpubackend(cuda)} now reuses the
same loaded plugin correctly.

{phang}
Advanced: {cmd:XHDFE_GPU_BACKEND} can be set in the environment (cpu|cuda|metal), but {opt gpubackend()} is
recommended.


{marker description}{...}
{title:Description}

{pstd}
{cmd:xhdfe} estimates linear models with high-dimensional fixed effects (HDFE) using a C++ backend.
It mirrors the reporting and defaults of {cmd:reghdfe} where possible, while exposing fixest-style
small-sample corrections and ideas from pyfixest and FixedEffectModels.jl. The command supports multiway fixed
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
{cmd:xhdfe} was developed through an AI-assisted optimization workflow. The
project used AI coding tools to study existing HDFE packages, review relevant
literature, inspect the mixed Stata/C++/Python codebase, draft and revise
patches, instrument slow paths, and improve performance. No AI tool or system
is named or credited as a contributor.{p_end}

{pstd}
The listed human authors did not write the package line by line. Their role was
to define the econometric target, set numerical and performance guardrails,
choose the benchmark and validation criteria, direct the optimization process,
decide which AI-assisted changes to keep, and retain responsibility for the
package. No software tool or AI system is credited as an author or co-author.
{p_end}


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


{marker links}{...}
{title:Links to online documentation}

{p2col 8 10 10 2: -}{browse "https://github.com/reisportela/xhdfe-xfe":xhdfe repository}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/sergiocorreia/reghdfe":reghdfe repository}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/lrberge/fixest":fixest repository}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/py-econometrics/pyfixest":pyfixest repository}{p_end}
{p2col 8 10 10 2: -}{browse "https://github.com/FixedEffects/FixedEffectModels.jl":FixedEffectModels.jl repository}{p_end}


{marker absorb}{...}
{title:Absorb() syntax}

{synoptset 22}{...}
{synopthdr:absvar}
{synoptline}
{synopt:{it:varname}}categorical variable to be absorbed{p_end}
{synopt:{it:label}={it:varname}}label the absorbed effect and use {it:label} as the saved-FE base name{p_end}
{synopt:{it:var1#var2}}two-way categorical interaction FE; {cmd:var1##var2} is accepted as the same joint FE{p_end}
{synopt:{it:fe#c.var}}group-specific slope without a level FE{p_end}
{synopt:{it:fe##c.var}}level FE plus group-specific slope{p_end}
{synoptline}
{p2colreset}{...}
{p 4 6 2}- Two-way categorical interactions are supported via {cmd:#} or {cmd:##}. For reghdfe
compatibility, categorical-only {cmd:##} terms are treated as one joint absorbed FE, not expanded into
main effects plus the interaction. Specify {cmd:var1 var2 var1#var2} explicitly if all three absorbed
components are intended.
Interaction FEs are materialized with {cmd:egen group()} after the estimation sample is defined, and
must have exactly two components. Plain numeric FE identifiers are passed directly to the plugin;
string identifiers are encoded by Stata before the plugin call.{p_end}
{p 4 6 2}- Heterogeneous slopes require exactly one categorical component and one continuous
component. Use {cmd:fe#c.x} for group-specific slopes only, or {cmd:fe##c.x} for a level FE plus
group-specific slopes. Continuous-only interactions and interactions with more than two components
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
Two-way categorical interactions can be specified with {cmd:#} or {cmd:##}; categorical-only {cmd:##}
is canonicalized to the same joint FE as {cmd:#}, matching reghdfe's absorbed-DoF presentation.
Interaction FEs are materialized with {cmd:egen group()} after the estimation sample is defined.
Plain numeric FE identifiers are passed directly to the plugin; string identifiers are encoded by Stata.

{pmore}
Heterogeneous slopes can be specified as {cmd:fe#c.x} or {cmd:fe##c.x}. The first absorbs
group-specific slopes only; the second absorbs both the FE intercept and the group-specific slopes.
They are not supported with {cmd:group()}/{cmd:individual()} or with {cmd:absorptionmethod(lsmr/mlsmr)}.

{pmore}
Absorb entries can be labeled as {it:label}={it:var} to control FE names in output and saved-FE names;
for {cmd:##}, label the joint term as one absorb entry or specify separate components explicitly.
Labels also instruct xhdfe to store FE contributions even without {cmd:savefe}/{cmd:savefes}. For
categorical FEs the output variable is {it:label}; for heterogeneous slopes {it:label} is the base
name and xhdfe adds {cmd:Slope1} to slope contributions. Generated names must not already exist.
Only the labeled terms are saved: unlabeled terms in the same {cmd:absorb()} (e.g. {cmd:year} in
{cmd:absorb(year wfe=worker ffe=firm)}) are absorbed as usual but leave no variable behind.

{pmore}
{cmd:absorb(..., savefe)} saves fixed effect contributions into {it:__hdfe1__}, {it:__hdfe2__}, ...
(reghdfe-compatible naming). This option cannot be combined with {cmd:savefes()}.
Saving FE contributions is not supported when both {cmd:group()} and {cmd:individual()} are specified.
For heterogeneous slopes, {cmd:g##c.z} saves the base FE and a {it:Slope1} variable
(for example, {it:__hdfe1__} and {it:__hdfe1__Slope1}); {cmd:g#c.z} saves only
the {it:Slope1} coefficient variable, matching reghdfe's savefe layout.
With {cmd:ferecoverymethod(hybrid)} (default), xhdfe accumulates FE alphas during absorption when feasible,
so savefe/savefes typically avoids a second recovery sweep. {cmd:fetolerance()} applies only to the
fallback MAP recovery.

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
{opt un:adjusted} (aliases: {cmd:ols}, {cmd:homoskedastic}, {cmd:classical}, {cmd:unadj}) estimates conventional standard errors (default).

{pmore}
{opt r:obust} (aliases: {cmd:hc1}, {cmd:heteroskedastic}) estimates heteroskedasticity-consistent standard errors (HC1-style).

{pmore}
{opt cl:uster} (alias: {cmd:clustered}) {it:clustervars} estimates standard errors clustered by one or more variables.
Multiway clustering is supported by listing multiple cluster variables.

{pmore}
Each {it:clustervar} may be a variable or a two-way interaction specified with {cmd:#}/{cmd:##}.
Interaction clusters are materialized with {cmd:egen group()} after sample marking. Plain numeric
cluster identifiers are passed directly to the plugin; string identifiers are encoded by Stata.

{pmore}
{cmd:robust} and {cmd:cluster()} are equivalent to {cmd:vce(robust)} and {cmd:vce(cluster ...)}; do not combine them with {cmd:vce()}.

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


{marker opt_iv}{...}
{dlgtab:IV/2SLS}

{phang}
{opth endog:enous(varlist)} identifies endogenous regressors (subset of slope regressors).

{phang}
{opth instr:uments(varlist)} provides instruments for IV/2SLS.
Both {cmd:endogenous()} and {cmd:instruments()} must be specified together.
Factor-variable operators are not supported with IV, and IV is not supported in {cmd:group()} mode.


{marker opt_dof}{...}
{dlgtab:Degrees-of-Freedom Adjustments}

{phang}
{opt dof:adjustments(list)} selects how the DoF lost due to absorbed FEs are adjusted.
Tokens can be combined and separated by spaces or commas. If {cmd:dofadjustments()} is omitted,
the default is {cmd:all}. When supplied, only the listed tokens are enabled (reghdfe-style).
Valid tokens (case-insensitive) are {cmd:all}, {cmd:none}, {cmd:firstpair} ({cmd:first}),
{cmd:pairwise} ({cmd:pair}), {cmd:clusters} ({cmd:cluster}), and {cmd:continuous} ({cmd:cont}).
{cmd:all} enables the full reghdfe logic. {cmd:none} disables FE DoF correction. {cmd:firstpair}
and {cmd:pairwise} select the FE DoF method. {cmd:clusters} enables adjustments for clusters
nested within absorbed FEs. {cmd:continuous} reserves reghdfe continuous-interaction checks
(currently a no-op).

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
{opth groupv:ar(newvar)} identifier for the first mobility group (1-based IDs). Requires at least
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
same nominal tolerance (down to the conditioning of the problem). Non-accelerated solver paths use a
calibrated absorber tolerance of {cmd:min(tolerance(),1e-9)}.
{cmd:tolerancemode(xhdfe-fast)} restores the pre-2.7.0 stopping rule: typically ~1.5-3x fewer
absorber iterations, with an effective precision that is data-dependent and can be looser than the
nominal tolerance on ill-conditioned (e.g. sparse bipartite) designs — appropriate for exploration
and for speed benchmarking (state the mode when citing timings).
{cmd:tolerancemode(strict-residual)} is a heavier audit mode: it treats the final absolute maximum
group-mean residual check as authoritative, may use additional iterations up to {cmd:maxiter()}, and
reports non-convergence if that check is not met. The mode used is returned in {cmd:e(tolerance_mode)}.

{phang}
New in version 2.11.0: in {cmd:tolerancemode(reghdfe-comparable)} (the default), absorbers on
ill-conditioned, poorly connected multi-way fixed-effect graphs — where the accelerated
alternating-projections iteration would otherwise stall and fall back to many thousands of plain
sweeps — are automatically handed off to a stable per-column conjugate-gradient solver on the
symmetric demeaning operator. This is transparent: coefficients, standard errors, default output, and
convergence behavior are unchanged (and typically tighter against {helpb reghdfe}), it applies on both
the CPU and CUDA backends, and well-conditioned problems are unaffected and numerically identical to
previous versions. It can cut runtime by an order of magnitude on such designs (e.g. a 540k-observation
3-way graph: Stata CPU ~54s to ~7s, CUDA ~5.5s to ~0.85s) at equal or better precision.

{phang}
{opt convergence(str)} selects the stopping criterion used by the heterogeneous-slope absorber
(active when {cmd:absorb()} contains {cmd:fe#c.x} / {cmd:fe##c.x} terms; standard absorption follows
{cmd:tolerancemode()}). {cmd:auto} (default) follows {cmd:tolerancemode()}: under
{it:reghdfe-comparable} the slope absorber stops on the reghdfe-style update criterion at the nominal
{cmd:tolerance()}, and under {it:xhdfe-fast} it uses the historical norm-change trigger.
{cmd:tolerancemode(strict-residual)} is not supported with heterogeneous slopes and exits with an
error before estimation; use {cmd:tolerancemode(reghdfe-comparable)} or {cmd:tolerancemode(xhdfe-fast)}
for those designs. Within supported tolerance modes, explicit values override the auto mapping:
{cmd:normchange} stops on the relative change in transformed-vector norms; {cmd:reghdfe} uses the
reghdfe-style vector update criterion based on the maximum weighted mean relative difference between
successive transformed {it:y}/{it:X} columns; {cmd:both} requires both criteria.

{phang}
{opth fetol:erance(#)} specifies the tolerance criterion for fixed-effect recovery (savefe); default is {cmd:fetolerance(1e-6)}.
This tolerance is separate from {cmd:tolerance()} and is only used when the MAP recovery path runs.

{phang}
{opth ferecoverymethod(str)} selects the fixed-effect recovery strategy. Valid values are
{cmd:map} and {cmd:hybrid} (default). {cmd:hybrid} reuses cached alphas when available; {cmd:map}
forces recovery sweeps on the partial residual.

{phang}
{opth maxit:er(#)} specifies the maximum number of absorber iterations; default is {cmd:maxiter(100000)}.
{cmd:iterate(#)} is accepted as a reghdfe-compatible alias for {cmd:maxiter(#)}.

{phang}
{opt absorptionm:ethod(str)} sets the absorption method. Valid values (case-insensitive) are
{cmd:auto}; {cmd:gauss-seidel}/{cmd:gauss_seidel}/{cmd:gs}; {cmd:symmetric}/{cmd:sym}/{cmd:symgs}/{cmd:symmetric-gauss-seidel}/{cmd:symmetric_gauss_seidel}; {cmd:jacobi};
{cmd:mlsmr}/{cmd:modified-lsmr}/{cmd:modified_lsmr}/{cmd:within}/{cmd:within-additive}/{cmd:within_additive};
{cmd:lsmr}/{cmd:plain-lsmr}/{cmd:plain_lsmr}; and
{cmd:auto-mlsmr}/{cmd:auto_mlsmr}/{cmd:mlsmr-auto}/{cmd:mlsmr_auto}.
The LSMR/MLSMR family uses matrix-free additive-Schwarz Krylov absorbers and is CPU-only for standard
FE designs.
{cmd:auto} first uses the probe-based auto-MLSMR selector on eligible CPU, standard-FE designs with up to three
fixed-effect dimensions. It promotes
slow-converging large designs to {cmd:mlsmr}; otherwise it resolves to a sweep fallback ({cmd:gauss-seidel} by
default, {cmd:symmetric-gauss-seidel} only with {cmd:symmetricsweep}). This avoids a second survey by the
Schwarz/Jacobi-PCG gate after the selector has already chosen a sweep. Under {cmd:tolerancemode(xhdfe-fast)},
MLSMR promotion is limited to large, many-RHS, multi-way moderate-rho designs. Designs with four or more fixed
effects stay on the full-data gate/sweep path so the 200k-sample probe cannot over-promote large, well-connected
many-way graphs (raise the cap with {cmd:XHDFE_AUTO_MLSMR_DEFAULT_MAX_FES}; {cmd:absorptionmethod(auto-mlsmr)}
is uncapped). Ineligible paths (GPU, savefe,
heterogeneous slopes, {cmd:group()}/{cmd:individual()}) resolve directly to the sweep fallback. Disable MLSMR gate
promotions with {cmd:XHDFE_MLSMR_AUTO_GATE=0}. {cmd:absorptionmethod(mlsmr)} forces MLSMR; {cmd:auto-mlsmr}
runs the same probe-based selector explicitly. {cmd:mlsmr}/{cmd:lsmr}/{cmd:auto-mlsmr} are CPU-only and do not support
{cmd:savefe}/heterogeneous slopes/{cmd:group()}/{cmd:individual()}.

{phang}
{opt sym:metricsweep} performs a forward + backward sweep per iteration for Gauss-Seidel methods
and can influence {cmd:absorptionmethod(auto)} selection when multiple FEs are present.

{phang}
{opt jacobirel:axation(#)} Jacobi relaxation parameter (<= 0 uses default 2/(J+1), where J is the
number of FE dimensions; values above 1 are capped at 1).

{p 4 6 2}For reghdfe script compatibility, xhdfe also accepts {cmd:technique()}, {cmd:acceleration()},
{cmd:transform()}, {cmd:preconditioner()}, {cmd:prune}, {cmd:fastregress}, {cmd:poolsize()}, {cmd:compact},
{cmd:timeit}, {cmd:verbose()}, {cmd:version()}, and {cmd:parallel()}. These are currently accepted as
compatibility options but do not change xhdfe's estimator or backend selection.{p_end}


{marker opt_parallel}{...}
{dlgtab:Parallel execution}

{phang}
{opth numthr:eads(#)} threads used by the absorber (0 = auto).

{phang}
{opth defaultthr:eads(#)} default threads when auto-threading is enabled.

{phang}
{opth maxthr:eads(#)} maximum threads when auto-threading is enabled.

{phang}
{opth minparallelr:ows(#)} minimum rows required to enable parallel absorption.

{phang}
{opth targetrowsper:thread(#)} heuristic target rows per thread.

{phang}
{bf:OpenMP:} parallel absorption requires a plugin built with OpenMP support.
On Linux, the build script will try OpenMP automatically; to force it (recommended for performance), from the repository root:

{phang2}
{cmd:. bash stata/tools/build-plugin.sh --linux --openmp}


{marker opt_reporting}{...}
{dlgtab:Reporting}

{phang}
{opt l:evel(#)} sets confidence level; default is {cmd:level(95)}. Values must be between 0 and 100.

{phang}
{opt noheader}, {opt notable}, and {opt nofootnote} suppress the header block, coefficient table,
and absorbed-DoF footnote table, respectively. These affect display only; estimation results are still stored.

{phang}
{opt noomitted}, {opt noempty}, {opt baselevels}, {opt allbaselevels}, {opt cformat()},
{opt pformat()}, {opt sformat()}, {opt fvwrap()}, {opt fvwrapon()}, {opt vsquish},
{opt nolstretch}, {opt nofvlabel}, {opt noci}, and {opt nopvalues} are forwarded to Stata's
{cmd:ereturn display} layer to keep the coefficient table behavior close to {cmd:reghdfe}.

{phang}
{opt nosample} prevents {cmd:xhdfe} from posting {cmd:e(sample)}. This reduces memory use, but
postestimation commands that rely on {cmd:e(sample)} may not be available afterward.


{marker opt_diagnostics}{...}
{dlgtab:Diagnostics}

{phang}
{opt mobilityprofile} computes mobility diagnostics (connected components, sweep order) once and
saves a profile file to {cmd:xhdfe_mobility_profile.txt} in the current working directory.
On later runs, if the file exists and the FE signature matches, xhdfe reuses it to guide
absorber ordering (and may enable symmetric sweeps under {cmd:absorptionmethod(auto)}).
When {cmd:mobilityfile()} is set, xhdfe computes or reuses the mobility profile at that path. If an
absorption cache is also enabled and its signature matches, xhdfe can skip the absorption step on
later identical model specifications (the cache file can be large).

{phang}
{opth mobfile(filename)} points to a profile file (aliases: {cmd:mobilityfile()}, {cmd:mobilityprofilefile()}).
If the file exists and the FE signature matches, xhdfe reuses it; otherwise it computes a new
profile and overwrites the file. To disable reuse, remove or rename the profile or its
{cmd:.absorption_cache} companion file, or set {cmd:absorptioncachemode(off)}.

{phang}
{opth absorptioncache(filename)} stores the transformed outcome/regressors for reuse across
identical model specifications (same {cmd:y}, {cmd:X}, {cmd:absorb()}, weights, and intercept handling).
When the cache signature matches, xhdfe skips the absorption step. The cache can be large, and it
does not apply when regressors or weights change. This option overrides the default
{cmd:<profile>.absorption_cache} path associated with {cmd:mobilityfile()}.

{phang}
{opt absorptioncachemode(str)} controls how absorption caches are used: {cmd:off} disables caching
entirely (even if {cmd:mobilityfile()} is used), {cmd:read} only reuses existing caches, {cmd:write}
always overwrites, and {cmd:auto} reads when possible and writes only when the cache is missing.
If no explicit {cmd:absorptioncache()} is provided, the mode applies to the {cmd:mobilityfile()}
companion cache when available. {cmd:abscachemode()} is accepted as a shorter alias.

{phang}
{cmd:XHDFE_FE_DIAG_FILE} (environment variable) appends fixed-effect diagnostics for each run
(partial residual stats, FE totals, residual consistency, recovery method) to the specified file.

{phang}
{cmd:XHDFE_PROFILE_SAVEFE} (environment variable) enables savefe timing traces to standard error.
When set to a nonzero value, xhdfe reports stage-level timings for fixed-effect recovery
({cmd:savefe_partial}, cached-hybrid reconstruction/checks, and MAP-fallback stages such as
{cmd:savefe_recover_setup}, {cmd:savefe_recover_loop}, and {cmd:savefe_recover_expand}) and, in
the Stata plugin path, the FE writeback stage ({cmd:plugin_writeback_*}). Disabled by default;
when unset, there is no extra output and no change in default behavior.

{phang}
{opt festructurecache} builds or reuses a fixed-effect structure cache stored alongside {cmd:xhdfe.ado}.
The cache stores normalized FE group ids and singleton filters, so repeated runs with the same
{cmd:absorb()} structure can skip FE indexing and singleton detection even when regressors or
clusters change. It is ignored when the FE signature changes (different absorb variables or
singleton policy) and is not used in {cmd:group()/individual()} mode. This cache is separate from
the absorption cache and does not store transformed {cmd:y} or {cmd:X}.

{phang}
{opth fescache(filename)} uses a specific cache file. {cmd:festructurecache()} and
{cmd:festructurecachefile()} are accepted as aliases for {cmd:fescache()}; bare
{cmd:festructurecache} is the flag form that selects the default cache path alongside
{cmd:xhdfe.ado}. Use an absolute path when the working directory changes or when the ado folder is
read-only.

{phang}
{opt fescachemode(str)} controls how FE structure caches are used: {cmd:off}, {cmd:auto}, {cmd:read},
or {cmd:write}. It requires either {cmd:fescache()} or {cmd:festructurecache} to define the cache path.
{cmd:fecachemode()} and {cmd:festructurecachemode()} are accepted as aliases.


{marker postestimation}{...}
{title:Postestimation Syntax}

{pstd}
{cmd:xhdfe} is an {cmd:eclass} command. Standard commands such as {cmd:estimates store},
{cmd:estimates table}, {cmd:test}, {cmd:lincom}, and {cmd:nlcom} should work.
The {cmd:margins} command uses {cmd:predict} or {cmd:score}; {cmd:predict, score} is mapped to
residuals for compatibility.

{pstd}
Implemented {cmd:estat} subcommands are {cmd:estat summarize}, {cmd:estat vce}, and
{cmd:estat ic}.

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
you estimated the model with {cmd:residuals(newvar)}.{p_end}


{marker missing_features}{...}
{title:Missing Features}

{p2colset 8 12 12 2}{...}
{p2col: -}Heterogeneous slopes with {cmd:group()}/{cmd:individual()} are not supported{p_end}
{p2col: -}{cmd:savefe}/{cmd:savefes} with both {cmd:group()} and {cmd:individual()} are not supported{p_end}
{p2col: -}IV/2SLS with factor-variable slope syntax or with {cmd:group()}/{cmd:individual()} is not supported{p_end}
{p2col: -}HAC or Driscoll-Kraay standard errors (see ivreghdfe in reghdfe ecosystem){p_end}
{p2colreset}{...}


{marker examples}{...}
{title:Examples}

{hline}
{pstd}Setup{p_end}
{phang2}{cmd:. sysuse auto}{p_end}

{pstd}Simple case - one fixed effect{p_end}
{phang2}{cmd:. xhdfe price weight length, absorb(rep78)}{p_end}
{hline}

{pstd}As above, but also compute clustered standard errors{p_end}
{phang2}{cmd:. xhdfe price weight length, absorb(rep78) vce(cluster rep78)}{p_end}
{hline}

{pstd}CUDA backend (requires a CUDA-enabled plugin){p_end}
{phang2}{cmd:. xhdfe price weight length, absorb(rep78) gpubackend(cuda)}{p_end}
{phang2}{cmd:. display e(gpu_used)}{p_end}
{phang2}{cmd:. display "`e(gpu_backend)'"}{p_end}
{hline}

{pstd}Two and three sets of fixed effects{p_end}
{phang2}{cmd:. webuse nlswork, clear}{p_end}
{phang2}{cmd:. xhdfe ln_wage grade age ttl_exp tenure not_smsa south, absorb(idcode year)}{p_end}
{phang2}{cmd:. xhdfe ln_wage grade age ttl_exp tenure not_smsa south, absorb(idcode year occ_code)}{p_end}
{hline}

{title:Advanced examples}

{pstd}Save the FEs as variables{p_end}
{phang2}{cmd:. xhdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year, savefe)}{p_end}

{pstd}Save residuals and FE contributions (prefix-based){p_end}
{phang2}{cmd:. xhdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year) vce(cluster idcode) residuals(u) savefes(fe_)}{p_end}

{pstd}Heterogeneous slopes{p_end}
{phang2}{cmd:. xhdfe ln_wage grade union, absorb(idcode##c.ttl_exp year) vce(cluster idcode)}{p_end}

{pstd}Two-stage least squares after absorbing FEs{p_end}
{phang2}{cmd:. xhdfe ln_wage grade ttl_exp union, absorb(idcode year) endogenous(union) instruments(south)}{p_end}

{pstd}Side-by-side comparison with reghdfe (user example){p_end}
{phang2}{cmd:. webuse nlswork, clear}{p_end}
{phang2}{cmd:. reghdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year) vce(robust)}{p_end}
{phang2}{cmd:. est store reghdfe_rob}{p_end}
{phang2}{cmd:. reghdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year) vce(cluster idcode)}{p_end}
{phang2}{cmd:. est store reghdfe_cl}{p_end}
{phang2}{cmd:. reghdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year, savefe) vce(robust)}{p_end}
{phang2}{cmd:. est store reghdfe3}{p_end}
{phang2}{cmd:. xhdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year) vce(robust)}{p_end}
{phang2}{cmd:. est store xhdfe_rob}{p_end}
{phang2}{cmd:. xhdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year) vce(cluster idcode)}{p_end}
{phang2}{cmd:. est store xhdfe_cl}{p_end}
{phang2}{cmd:. xhdfe ln_wage grade ttl_exp union, absorb(idcode ind_code occ_code year) savefes(fe_) vce(cluster idcode)}{p_end}
{phang2}{cmd:. est tab reghdfe_rob xhdfe_rob reghdfe_cl xhdfe_cl, b(%15.13f) se(%15.13f)}{p_end}

{title:Group Examples}

{hline}
{pstd}Setup (long format with group and individual ids){p_end}
{phang2}{cmd:. clear}{p_end}
{phang2}{cmd:. set obs 40}{p_end}
{phang2}{cmd:. gen patent_id = ceil(_n/2)}{p_end}
{phang2}{cmd:. bysort patent_id: gen inventor_id = mod(patent_id + _n, 10) + 1}{p_end}
{phang2}{cmd:. gen year = 2020 + mod(patent_id, 3)}{p_end}
{phang2}{cmd:. gen funding = mod(patent_id, 2)}{p_end}
{phang2}{cmd:. gen citations = 5 + 0.4*funding + patent_id/10}{p_end}

{pstd}Individual and group fixed effects{p_end}
{phang2}{cmd:. xhdfe citations funding, absorb(inventor_id) group(patent_id) individual(inventor_id)}{p_end}
{hline}

{pstd}Individual and group fixed effects with aggregation(sum){p_end}
{phang2}{cmd:. xhdfe citations funding, absorb(inventor_id) group(patent_id) individual(inventor_id) aggregation(sum)}{p_end}
{hline}

{pstd}Group-level estimation without individual fixed effects{p_end}
{phang2}{cmd:. xhdfe citations funding, absorb(year) group(patent_id)}{p_end}
{hline}


{marker results}{...}
{title:Stored results}

{pstd}
{cmd:xhdfe} stores the following in {cmd:e()}:

{synoptset 24 tabbed}{...}
{syntab:Scalars}
{synopt:{cmd:e(N)}}number of observations{p_end}
{synopt:{cmd:e(N_full)}}observations before singleton dropping{p_end}
{synopt:{cmd:e(num_singletons)}}number of singleton observations dropped{p_end}
{synopt:{cmd:e(sumweights)}}sum of weights (or {cmd:e(N)} if unweighted){p_end}
{synopt:{cmd:e(drop_singletons)}}1 if singletons were dropped, 0 otherwise{p_end}
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
{synopt:{cmd:e(iterations)}}absorption iterations{p_end}
{synopt:{cmd:e(ic)}}alias of {cmd:e(iterations)} for {cmd:estat ic}{p_end}
{synopt:{cmd:e(converged)}}1 if converged, 0 otherwise{p_end}
{synopt:{cmd:e(fe_recovery_converged)}}(only with {cmd:savefe}/{cmd:savefes}) 1 if fixed-effect recovery converged, 0 otherwise{p_end}
{synopt:{cmd:e(fe_recovery_iterations)}}(only with {cmd:savefe}/{cmd:savefes}) iterations used to recover the fixed effects{p_end}
{synopt:{cmd:e(fe_recovery_max_delta)}}(only with {cmd:savefe}/{cmd:savefes}) maximum change in the final FE recovery sweep{p_end}
{synopt:{cmd:e(report_constant)}}1 if the constant is reported, 0 if {cmd:noconstant}{p_end}
{synopt:{cmd:e(threads_used)}}threads used by backend{p_end}
{synopt:{cmd:e(gpu_used)}}1 if GPU backend was effectively used, 0 if CPU path was used{p_end}
{synopt:{cmd:e(gpu_status_code)}}GPU status code: 0 not requested, 1 used, 2 unavailable, 3 not converged, 4 failed, 5 CPU cache/profile result{p_end}
{synopt:{cmd:e(gpu_attempted)}}1 if GPU absorption was attempted, 0 otherwise{p_end}
{synopt:{cmd:e(gpu_absorption_converged)}}1 if attempted GPU absorption converged, 0 if not, missing if not attempted{p_end}
{synopt:{cmd:e(gpu_absorption_iterations)}}GPU absorption iterations, missing if GPU absorption was not attempted{p_end}
{synopt:{cmd:e(absorption_method_used)}}absorption method code: 0 auto, 1 gauss-seidel, 2 symmetric-gauss-seidel, 3 jacobi, 4 Schwarz/CG, 5 LSMR, 6 MLSMR, 7 auto-MLSMR{p_end}
{synopt:{cmd:e(N_clust)}}minimum number of clusters (if clustered){p_end}
{synopt:{cmd:e(N_clustervars)}}number of cluster dimensions{p_end}
{synopt:{cmd:e(N_clust#)}}cluster counts by dimension{p_end}
{synopt:{cmd:e(cluster_scale)}}cluster scaling factor (SSC){p_end}
{synopt:{cmd:e(vcv_psd_fixed)}}1 if the multiway-cluster VCV required the Cameron-Gelbach-Miller
positive-semi-definite adjustment (a warning is printed, matching {helpb reghdfe}){p_end}
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

{synoptset 24 tabbed}{...}
{syntab:Macros}
{synopt:{cmd:e(cmd)}}{cmd:xhdfe}{p_end}
{synopt:{cmd:e(cmdline)}}command as typed{p_end}
{synopt:{cmd:e(title)}}main estimation title{p_end}
{synopt:{cmd:e(title2)}}secondary title reporting the number of absorbed FE groups{p_end}
{synopt:{cmd:e(depvar)}}name of dependent variable{p_end}
{synopt:{cmd:e(indepvars)}}names of independent variables{p_end}
{synopt:{cmd:e(absorb)}}absorbed FE list{p_end}
{synopt:{cmd:e(absvars)}}alias for absorbed FE list (reghdfe compatibility){p_end}
{synopt:{cmd:e(extended_absvars)}}extended absorbed FE list (reghdfe compatibility){p_end}
{synopt:{cmd:e(absorb_labels)}}labels shown for absorbed effects in output/savefes(){p_end}
{synopt:{cmd:e(dofmethod)}}DoF adjustment tokens applied{p_end}
{synopt:{cmd:e(dof_labels)}}labels used in the absorbed-DoF table, when posted{p_end}
{synopt:{cmd:e(clustvar)}}cluster variable list (if clustered){p_end}
{synopt:{cmd:e(vce)}}vcetype specified in {cmd:vce()}{p_end}
{synopt:{cmd:e(vcetype)}}title used to label Std. Err.{p_end}
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
{synopt:{cmd:e(gpu_backend_requested)}}requested backend from {cmd:gpubackend()} or the {cmd:XHDFE_GPU_BACKEND} environment variable, when provided{p_end}
{synopt:{cmd:e(gpu_backend)}}effective backend when {cmd:gpubackend()} is supplied, or when the GPU was used via {cmd:XHDFE_GPU_BACKEND}; otherwise not posted{p_end}
{synopt:{cmd:e(gpu_status)}}GPU status label: {cmd:not_requested}, {cmd:used}, {cmd:backend_unavailable}, {cmd:gpu_absorption_not_converged}, {cmd:gpu_backend_failed}, or {cmd:cpu_cache_or_profile_result}{p_end}
{synoptline}

{synoptset 24 tabbed}{...}
{syntab:Matrices}
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

{pstd}Released builds of {cmd:xhdfe} are installed via {cmd:net install} from the online Stata site published by {cmd:xhdfe-xfe}: {browse "https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata":https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata}.{p_end}
{pstd}For local development checkouts or unzipped release bundles, {cmd:net install} can also point at the local {cmd:stata/} folder that contains {cmd:stata.toc}, {cmd:xhdfe.pkg}, and the platform plugin.{p_end}
{pstd}For building the plugin or running validation tests, see the repository README.{p_end}


{marker acknowledgements}{...}
{title:Acknowledgements}

{pstd}
xhdfe is a reimplementation built to validate against and interoperate with prior work. Full credit goes to:
{p_end}
{phang2}{bf:reghdfe} by Sergio Correia (Stata) - {browse "https://github.com/sergiocorreia/reghdfe":Github}{p_end}
{phang2}{bf:fixest} by Laurent Berge (R) - {browse "https://github.com/lrberge/fixest":Github}{p_end}
{phang2}{bf:pyfixest} by Alexander Fischer and collaborators (Python) - {browse "https://github.com/py-econometrics/pyfixest":Github}{p_end}
{phang2}{bf:FixedEffectModels.jl} by Matthieu Gomez and collaborators (Julia) - {browse "https://github.com/FixedEffects/FixedEffectModels.jl":Github}{p_end}

{pstd}
We thank Paulo Guimaraes (Banco de Portugal), Marta Silva
(Banco de Portugal), and Nelson Areal (EEG / UMinho) for valuable interactions across
different parts of the project. We especially thank Sergio Correia for feedback on
benchmarking, tolerances, and {cmd:reghdfe}-comparable validation. Nelson Areal and Miguel Portela presented
"Parallel and Cross-Language Computing: A Hands-On Workshop for Empirical Researchers" at BPLIM's Workshop on
{it:Speeding Up Empirical Research: Tools and Techniques for Fast Computing}
({browse "https://github.com/BPLIM/Workshops/tree/master/BPLIM2025":BPLIM2025}), where we
shared an earlier version of this proof of concept. We also
thank Universidade do Minho, Banco de Portugal, and FCT - Portuguese Foundation for Science and Technology (UID/03182/2025) for financial support. All remaining errors are
ours. The usual disclaimer applies.{p_end}


{marker references}{...}
{title:References}

{pstd}
Selected references for high-dimensional fixed effects and related software include:{p_end}

{phang}
Portela, Miguel, and Tiago Tavares. 2026. "{cmd:xhdfe}: High-dimensional fixed effects
regression via a C++ backend." Version 2.14.1.
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
Correia, Sergio, Paulo Guimaraes, and Tom Zylkin. 2020. "Fast Poisson estimation with
high-dimensional fixed effects." {it:Stata Journal} 20(1): 95-115.
{browse "https://doi.org/10.1177/1536867X20909691":doi:10.1177/1536867X20909691}.{p_end}

{phang}
Berge, Laurent R., Kyle Butts, and Grant McDermott. 2026. "Fast and user-friendly
econometrics estimations: The R package {cmd:fixest}." {it:arXiv preprint} arXiv:2601.21749.
{browse "https://doi.org/10.48550/arXiv.2601.21749":doi:10.48550/arXiv.2601.21749}.{p_end}

{phang}
Fischer, Alexander, and collaborators. "{cmd:pyfixest}: Fast high-dimensional fixed-effects
regression in Python." Python package.
{browse "https://github.com/py-econometrics/pyfixest":https://github.com/py-econometrics/pyfixest}.{p_end}

{phang}
FixedEffects.jl contributors. 2025. "{cmd:FixedEffectModels.jl}: Fast Estimation of Linear
Models with IV and High Dimensional Categorical Variables." Julia package.
{browse "https://github.com/FixedEffects/FixedEffectModels.jl":https://github.com/FixedEffects/FixedEffectModels.jl}.{p_end}

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
Companion command in the same package: {helpb xfe} (partials out / residualizes
variables against multiple high-dimensional fixed effects on the same C++ core;
no coefficient table).{p_end}

{psee}
Worker-firm (AKM) post-estimation on the same backend: {helpb xhdfeakm}
(leave-out KSS variance decomposition, component standard errors,
weak-identification confidence intervals), {helpb xhdfeconnected}
(leave-one-out connected set as a sample-preparation utility) and
{helpb xhdfegelbach} (Gelbach decomposition).{p_end}
