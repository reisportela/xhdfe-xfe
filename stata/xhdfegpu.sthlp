{smcl}
{* *! version 2.12.0  07jul2026}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{vieweralsosee "xfe" "help xfe"}{...}
{title:Title}

{p2colset 5 18 20 2}{...}
{p2col :{cmd:xhdfegpu} {hline 2}}build and install a CUDA (GPU) xhdfe/xfe plugin for this machine{p_end}
{p2colreset}{...}


{title:Syntax}

{p 8 17 2}
{cmd:xhdfegpu} [{cmd:,} {it:options}]

{synoptset 20 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Source}
{synopt :{opth zip(filename)}}build from a source distribution zip already on
this machine (offline); by default {cmd:xhdfegpu} downloads it from the online site{p_end}
{synopt :{opth source(dirname)}}build from an already-extracted {cmd:xhdfe-src} source directory{p_end}
{synopt :{opth url(string)}}override the download URL of the source distribution zip{p_end}

{syntab:Build}
{synopt :{opt arch(#)}}force the CUDA compute capability (e.g. {cmd:90} for an
H100) instead of auto-detecting it with {cmd:nvidia-smi}{p_end}
{synopt :{opt keep}}keep the temporary build directory instead of deleting it{p_end}
{synoptline}


{title:Description}

{pstd}
{cmd:xhdfegpu} compiles a CUDA (NVIDIA GPU) plugin for {cmd:xhdfe} and
{cmd:xfe} matched to this machine's GPU, and installs it {it:over} the CPU
plugin that {cmd:net install} placed on the adopath. The runtime file names do
not change: the same {cmd:xhdfe.plugin} and {cmd:xfe.plugin} are replaced in
place, so nothing else in your setup changes — GPU runs are then requested the
usual way with {cmd:gpubackend(cuda)}.

{pstd}
{cmd:net install} ships a CPU-only plugin because a precompiled binary cannot
know your GPU. {cmd:xhdfegpu} fills that gap: it runs only when an NVIDIA GPU
is present, detects its compute capability, builds a plugin for that exact
architecture, and swaps it in atomically — the CPU plugin is replaced only if
the build succeeds.

{pstd}
The sources come from a self-contained distribution zip that vendors every
build dependency (Eigen, pybind11, Stata's {cmd:stplugin} inputs), so the
compilation itself needs {bf:no internet access}. {cmd:xhdfegpu} obtains that
zip in one of three ways:

{p 8 11 2}1. {bf:online (default)} — it downloads the zip from the project site;{p_end}
{p 8 11 2}2. {bf:offline} — if the download fails (no internet, or the site is
unreachable), download the zip on another machine, copy it over, and re-run
with {cmd:zip(}{it:path}{cmd:)};{p_end}
{p 8 11 2}3. {bf:pre-extracted} — point {cmd:source(}{it:dir}{cmd:)} at an
already-unpacked {cmd:xhdfe-src} tree.{p_end}


{title:Requirements}

{pstd}
{cmd:xhdfegpu} is supported on Linux. The machine must have an NVIDIA GPU, the
NVIDIA CUDA toolkit ({cmd:nvcc}) and driver, a C++ compiler ({cmd:g++}/{cmd:c++}),
and {cmd:unzip}. The produced plugin is OpenMP-enabled and built for the
detected {cmd:sm_}{it:XX} target only.


{title:Remarks}

{pstd}
After {cmd:xhdfegpu} finishes, the running Stata session still has the old
plugin loaded. Type {cmd:discard} (with no arguments) or restart Stata so the
new CUDA plugin is loaded, then verify a GPU run:

{p 8 8 2}{cmd:. xhdfe y x, absorb(fe1 fe2) gpubackend(cuda)}{p_end}
{p 8 8 2}{cmd:. display e(gpu_used)}   {it:// should be 1}{p_end}

{pstd}
CPU behavior is the numerical reference; the CUDA backend is validated to
agree with it. If a {cmd:gpubackend(cuda)} run reports that CUDA was not
available, this command is how you build the matching plugin.


{title:Examples}

{pstd}Build and install for the local GPU, downloading the sources:{p_end}
{p 8 8 2}{cmd:. xhdfegpu}{p_end}

{pstd}Offline machine — build from a zip you copied over by hand:{p_end}
{p 8 8 2}{cmd:. xhdfegpu, zip("~/downloads/xhdfe-src.zip")}{p_end}

{pstd}Force a specific architecture (e.g. an H100, {cmd:sm_90}):{p_end}
{p 8 8 2}{cmd:. xhdfegpu, arch(90)}{p_end}


{title:Stored results}

{pstd}
{cmd:xhdfegpu} stores the following in {cmd:r()}:

{synoptset 16 tabbed}{...}
{p2col 5 16 20 2: Scalars}{p_end}
{synopt:{cmd:r(gpu_ready)}}1 when a CUDA plugin was built and installed{p_end}

{p2col 5 16 20 2: Macros}{p_end}
{synopt:{cmd:r(arch)}}the CUDA target that was built, e.g. {cmd:sm_90}{p_end}
{synopt:{cmd:r(plugin)}}the adopath location of the installed {cmd:xhdfe.plugin}{p_end}


{title:Also see}

{p 4 13 2}
{helpb xhdfe}, {helpb xfe}
{p_end}
