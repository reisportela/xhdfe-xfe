{smcl}
{* *! version 2.35.1  25nov2026}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{vieweralsosee "xfepout" "help xfepout"}{...}
{title:Title}

{p2colset 5 18 20 2}{...}
{p2col :{cmd:xhdfegpu} {hline 2}}Build, install, and inspect Stata CUDA plugins on Linux{p_end}
{p2colreset}{...}

{title:Syntax}

{p 8 17 2}{cmd:xhdfegpu} [{cmd:,} {it:build_options}]{p_end}
{p 8 17 2}{cmd:xhdfegpu, status}{p_end}

{synoptset 25 tabbed}{...}
{synopthdr :build_options}
{synoptline}
{synopt :{opth cputarget(string)}}{cmd:portable} or {cmd:native} CPU optimisation{p_end}
{synopt :{opt arch(#)}}CUDA compute capability, e.g. {cmd:90} for an H100{p_end}
{synopt :{opth source(dirname)}}build from an extracted source tree{p_end}
{synopt :{opth zip(filename)}}build from a local source archive{p_end}
{synopt :{opt url(string)}}override the default source archive URL{p_end}
{synopt :{opt keep}}keep the private build directory{p_end}
{synopt :{opt force}}accepted for compatibility{p_end}
{synoptline}

{title:Installation and CPU target}

{pstd}
The standard {cmd:net install xhdfe} installs the portable CPU plugins for
your platform. On Linux x86-64, {cmd:xhdfegpu} builds OpenMP-enabled hybrid
plugins. Each plugin contains a CPU path and a CUDA path.

{phang}
{opt cputarget(portable)} selects the portable CPU target. This is the default
with a clean environment. {opt cputarget(native)} optimises for the CPU of the
build host. Use a separate {cmd:PLUS} directory for a native build if you also
run Stata on nodes with different CPUs. On a shared {cmd:PLUS}, use the
portable target to avoid requiring the build host's CPU instructions on all
nodes.

{phang}
An explicit {opt cputarget()} takes precedence over the process environment.
With the option omitted, the legacy variables select native only when
{cmd:XHDFE_ENABLE_MARCH_NATIVE} is {cmd:on} and
{cmd:XHDFE_FORCE_MARCH_NATIVE_WITH_CUDA} is {cmd:1}, {cmd:ON}, {cmd:on},
{cmd:true}, or {cmd:yes}. Otherwise the CPU target is portable.

{phang}
{opt arch()} controls the GPU architecture independently of the CPU target.
For example, {cmd:xhdfegpu, cputarget(portable) arch(90)} builds a portable
CPU path and CUDA code for H100-class devices. Without {opt arch()},
the installer uses the first compute capability reported by
{cmd:nvidia-smi}.

{pstd}
Use source that matches the installed xhdfe and xfepout versions.
The installer uses {opt source()} first, then {opt zip()}, then {opt url()}
or the public net-install site. A local source tree or archive allows
an offline build.

{pstd}
The build needs Bash, a C++ compiler with OpenMP and a full NVIDIA CUDA
toolkit, including the cuBLAS and cuSPARSE development files. Archives use
{cmd:unzip}, and online retrieval uses {cmd:curl} or {cmd:wget}. The GPU smoke
test requires a compatible driver and device. The build is staged and tested
before replacing the plugin alongside the active {cmd:xhdfe.ado} and the
plugin alongside {cmd:xfepout.ado} when that companion is present. An ordinary
failure during replacement restores the prior files. Byte-identical plugins
stay in place, including when {opt force} is specified.

{pstd}
Open a new Stata session after replacement. Run {cmd:xhdfegpu, status} to
confirm the installed paths and targets, then run the
{help xhdfe_examples##cuda:public-data CUDA example}. For a real CUDA fit
check {cmd:e(gpu_used)} equals one and {cmd:e(gpu_status)} equals {cmd:used}.

{title:Status}

{pstd}
{cmd:xhdfegpu, status} reports the plugin paths that the current ado search
would use. When a plugin is already bound in this Stata session, its build
metadata comes from that module. The bound path and the path on disk may refer
to different builds after an installation. The values
{cmd:by_path}, {cmd:memory}, {cmd:memory_unrecorded}, and {cmd:missing}
distinguish how the plugin was queried. {cmd:memory_unrecorded} means a
registered probe has no recorded path for its loaded module.
Restart Stata after replacement to associate the metadata with the file
currently on disk. A plugin without build metadata reports its text fields as
{cmd:unrecorded} and its capability indicators as missing. Status describes
the build. An actual fit verifies that a GPU is available and used.

{title:Examples}

{phang2}{cmd:. xhdfegpu, status}{p_end}
{phang2}{cmd:. xhdfegpu}{p_end}
{phang2}{cmd:. xhdfegpu, cputarget(native) arch(90)}{p_end}
{phang2}{cmd:. xhdfegpu, zip("/path with spaces/xhdfe-src.zip") cputarget(portable)}{p_end}
{phang2}{cmd:. xhdfegpu, source("/path/to/xhdfe-src")}{p_end}

{title:Stored results}

{pstd}
After a successful install, {cmd:xhdfegpu} returns {cmd:r(gpu_ready)} equal to
one and the macros {cmd:r(plugin)}, {cmd:r(xfepout_plugin)}, {cmd:r(arch)},
{cmd:r(cputarget)}, {cmd:r(backup)}, and {cmd:r(workdir)}. An absent companion
has an empty plugin path.

{pstd}
{cmd:xhdfegpu, status} returns the fields below with
{cmd:xhdfe_} and {cmd:xfepout_} prefixes, e.g. {cmd:r(xhdfe_cputarget)}.

{synoptset 27 tabbed}{...}
{synopt:{it:plugin}}plugin path beside the resolved ado on disk{p_end}
{synopt:{it:bound_plugin}}path used by an already-bound module, if any{p_end}
{synopt:{it:build_source}}{cmd:by_path}, {cmd:memory}, {cmd:memory_unrecorded}, or {cmd:missing}{p_end}
{synopt:{it:cputarget}}{cmd:portable}, {cmd:native}, or {cmd:unrecorded}{p_end}
{synopt:{it:sms}}CUDA compute capabilities, {cmd:none}, or {cmd:unrecorded}{p_end}
{synopt:{it:compiler}}compiler identification or {cmd:unrecorded}{p_end}
{synopt:{it:build_version}}build version or {cmd:unrecorded}{p_end}
{synopt:{it:cuda}}scalar CUDA build flag, missing when unrecorded{p_end}
{synopt:{it:openmp}}scalar OpenMP build flag, missing when unrecorded{p_end}

{title:Also see}

{psee}{helpb xhdfe}, {helpb xfepout}{p_end}
