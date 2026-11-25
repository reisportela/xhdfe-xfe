{smcl}
{* *! version 2.35.0  25nov2026}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{vieweralsosee "xfepout" "help xfepout"}{...}
{title:Title}

{p2colset 5 18 20 2}{...}
{p2col :{cmd:xhdfegpu} {hline 2}}Build and install local CUDA plugins on Linux{p_end}
{p2colreset}{...}

{title:Syntax}

{p 8 17 2}{cmd:xhdfegpu} [{cmd:,} {it:options}]{p_end}

{synoptset 20 tabbed}{...}
{synopthdr}
{synoptline}
{synopt :{opth source(dirname)}}build from an extracted source tree{p_end}
{synopt :{opth zip(filename)}}build from a local source archive{p_end}
{synopt :{opt url(string)}}override the default source archive URL{p_end}
{synopt :{opt arch(#)}}select CUDA compute capability, e.g. 90 for an H100{p_end}
{synopt :{opt keep}}keep the temporary build directory{p_end}
{synopt :{opt force}}accepted for compatibility; build and installation are unchanged{p_end}
{synoptline}

{title:Description}

{pstd}
The command builds OpenMP-enabled Linux plugins for a single CUDA
architecture, with CPU and CUDA support in the same binary. Use source
that matches the installed ado version. Source selection is {opt source()},
then {opt zip()}, then {opt url()}. With all three omitted, the command
downloads the source archive from the package's public net-install site.

{pstd}
Build requirements are Bash, a C++ compiler with OpenMP and the NVIDIA
CUDA toolkit ({cmd:nvcc}), including its cuBLAS and cuSPARSE development
files (the headers {cmd:cublas_v2.h} and {cmd:cusparse.h}): a full toolkit
installation has them, a compiler-only installation does not.
Archive inputs use {cmd:unzip}; online retrieval
may use {cmd:curl} or {cmd:wget}. With {opt arch()} omitted, the command
takes the first compute capability reported by {cmd:nvidia-smi}.
For execution you also need a compatible driver and NVIDIA device
(compute capability 7.5 or later). The local source archive supports an
offline build after these tools are available.

{pstd}
After building, the command replaces the first {cmd:xhdfe.plugin} found on
the {cmd:adopath}. It also tries to build and replace {cmd:xfepout.plugin}
if that plugin is installed. Check the installation message for each binary.
After replacement, run {cmd:discard} with no arguments or restart Stata
before testing a fit.

{title:Examples}

{pstd}Choose a single build command for the source available on your machine.{p_end}
{phang2}{cmd:. xhdfegpu}{p_end}
{phang2}{cmd:. xhdfegpu, zip("/path/to/xhdfe-src.zip") arch(90)}{p_end}
{phang2}{cmd:. xhdfegpu, source("/path/to/xhdfe-src")}{p_end}

{pstd}After reloading the plugin, run the
{help xhdfe_examples##cuda:public-data CUDA example}.
Check {cmd:e(gpu_used)} = 1 and {cmd:e(gpu_status)} = {cmd:used} on the CUDA fit.{p_end}

{title:Stored results}

{synoptset 18 tabbed}{...}
{p2col 5 18 22 2: Scalars}{p_end}
{synopt:{cmd:r(gpu_ready)}}1 when {cmd:xhdfe.plugin} was built and installed{p_end}

{p2col 5 18 22 2: Macros}{p_end}
{synopt:{cmd:r(arch)}}CUDA target, e.g. {cmd:sm_90}{p_end}
{synopt:{cmd:r(plugin)}}installed path of {cmd:xhdfe.plugin}{p_end}
{p2colreset}{...}

{title:Also see}

{psee}{helpb xhdfe}, {helpb xfepout}{p_end}
