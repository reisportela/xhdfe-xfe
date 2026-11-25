{smcl}
{* *! version 1.3.0  25nov2026}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{vieweralsosee "xhdfeakm" "help xhdfeakm"}{...}
{vieweralsosee "xhdfegelbach" "help xhdfegelbach"}{...}
{title:Title}

{p2colset 5 24 26 2}{...}
{p2col :{cmd:xhdfeconnected} {hline 2}}leave-one-out connected set (KSS) as a sample-preparation utility{p_end}
{p2colreset}{...}


{title:Syntax}

{p 8 15 2}
{cmd:xhdfeconnected} {it:workervar} {it:firmvar} {ifin} {weight}{cmd:,}
{opth gen:erate(newvar)}
[{opt replace} {opt threads(#)} {opt gpu} {opt verbose}]

{p 4 6 2}Frequency weights are allowed, for example {cmd:[fw=count]}.{p_end}

{synoptset 24 tabbed}{...}
{synopthdr}
{synoptline}
{synopt :{opth gen:erate(newvar)}}save the leave-out-set flag; {bf:required}{p_end}
{synopt :{opt replace}}replace an existing target variable{p_end}
{synopt :{opt threads(#)}}OpenMP threads (0 = library default){p_end}
{synopt :{opt gpu}}use stable CUDA radix sorting when measured profitable{p_end}
{synopt :{opt verbose}}stream deterministic graph-construction progress live{p_end}
{synoptline}
{p2colreset}{...}


{title:Description}

{pstd}
{cmd:xhdfeconnected} computes the largest leave-one-out connected set of the
worker-firm bipartite graph — the sample on which the KSS (Kline, Saggio and
Soelvsten 2020) leave-out variance decomposition is identified — without
running any estimation: largest connected component (component size measured
in firms), iterative removal of workers that are articulation points of the
mover-firm graph, and dropping of workers observed once. The semantics match
Saggio's LeaveOutTwoWay and the sample step of {helpb xhdfeakm}; the flag it
generates equals {it:stub}{cmd:_keep} from
{cmd:xhdfeakm, generate(}{it:stub}{cmd:)}.

{pstd}
With {cmd:fweight}, row {it:i} stands for {it:w_i} identical person-year
observations (the single-observation-worker rule counts person-years, so a
single row with weight 2 or more is a T>1 worker). With {cmd:fweight},
{cmd:r(n_obs_input)}, {cmd:r(n_obs_connected)}, and {cmd:r(n_obs)} count
weighted person-years. The generated flag still marks input rows.

{pstd}
The result is deterministic: it does not depend on the number of threads,
and exact ties between equally-sized connected components (same firm count
and same person-year count — essentially only possible on small or
artificially symmetric graphs) are broken by the smallest firm id in the
component rather than by the physical row order of the data (since 2.14.2;
the reference LeaveOutTwoWay leaves such ties undefined).

{pstd}
Worker and firm variables are numeric categorical identifiers. Codes outside
the signed 32-bit range, and non-integer numeric labels, are compacted
internally without changing the graph or the selected sample.


{title:Options}

{phang}{opth generate(newvar)} saves a byte flag equal to one for evaluated
observations in the leave-one-out connected set and zero for evaluated rows
outside it. Rows excluded by {it:if}/{it:in} or missing worker/firm ids remain
missing; use {cmd:if newvar == 1}, not merely {cmd:if newvar}, in a subsequent
command. {opt replace} drops an existing target before graph estimation.

{phang}{opt threads(#)} sets the OpenMP request. Zero uses the automatic
policy. A positive request bypasses workload heuristics and is limited only
by the logical-processor/OpenMP capacity visible to the process. The selected
sample is deterministic and invariant to the thread count.

{phang}{opt gpu} requests a hybrid CUDA path. Stable 64-bit match keys are
sorted with CUB radix sort on the GPU; exact largest-component, union-find,
Tarjan articulation-point pruning, and final sample construction remain on
CPU. The CUDA sort is used only at 10,000,000 selected input rows or more, the measured
cold-context crossover on the local H100/OpenMP workstation. Below that gate,
the faster CPU path is retained and {cmd:r(gpu_status)} is
{cmd:not_beneficial}. An unavailable or failed GPU also falls back without
changing the sample. {cmd:r(gpu_used)} equals one only after a real CUDA sort.

{phang}{opt verbose} streams phase progress live for id compaction, CUDA
selection, largest-component construction, pruning, and finalization. Each
line is flushed before the next phase; it does not alter the graph or returned
flag.

{pstd}The {cmd:threads()}/{cmd:gpu}/{cmd:verbose} execution controls and
backend diagnostics are also exposed by the Python and R companion
surfaces.{p_end}


{title:Stored results}

{pstd}{cmd:r(n_obs)} (person-year observations in the leave-out set),
{cmd:r(n_obs_input)}, {cmd:r(n_obs_connected)} (after the largest-CC step),
{cmd:r(n_workers)}, {cmd:r(n_firms)}, {cmd:r(n_matches)}, {cmd:r(n_movers)},
{cmd:r(n_stayers)}, {cmd:r(prune_iterations)}, and the macro {cmd:r(cmd)}
({cmd:xhdfeconnected}).

{pstd}Thread diagnostics: {cmd:r(threads_requested)},
{cmd:r(threads_effective)}, {cmd:r(threads_used)},
{cmd:r(parallel_workers_active)}, {cmd:r(thread_capacity)},
{cmd:r(openmp_enabled)}, {cmd:r(thread_limit_code)}, and
{cmd:r(thread_limit_reason)}. Team and active-worker counts are observed
during useful graph/sample work; they are not copies of the request.

{pstd}Backend diagnostics:
{cmd:r(gpu_requested)}, {cmd:r(gpu_used)}, and
{cmd:r(gpu_status_code)} (0 not requested, 1 used, 2 unavailable, 4 failed,
6 not beneficial), plus macros {cmd:r(gpu_backend)} ({cmd:cpu}/{cmd:cuda}) and
{cmd:r(gpu_status)} ({cmd:not_requested}, {cmd:used}, {cmd:unavailable},
{cmd:failed}, or {cmd:not_beneficial}).


{title:Example}

{pstd}Use the largest leave-out connected set of the public simulation.
The same sample can be passed to {cmd:xhdfeakm, noprune}.{p_end}
{phang2}{cmd:. use "https://raw.githubusercontent.com/reisportela/xhdfe-xfe/main/data/felsdvsimul.dta", clear}{p_end}
{phang2}{cmd:. xhdfeconnected i j, generate(in_connected)}{p_end}
{phang2}{cmd:. return list}{p_end}
{pstd}See {help xhdfe_examples##akm:connected sample and AKM/KSS components}
for the complete example.{p_end}

{title:Also see}

{psee}{helpb xhdfeakm}, {helpb xhdfe}{p_end}
