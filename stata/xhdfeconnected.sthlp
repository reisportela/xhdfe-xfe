{smcl}
{* *! version 2.14.1  09jul2026}{...}
{vieweralsosee "xhdfe" "help xhdfe"}{...}
{vieweralsosee "xhdfeakm" "help xhdfeakm"}{...}
{vieweralsosee "xhdfegelbach" "help xhdfegelbach"}{...}
{title:Title}

{p2colset 5 24 26 2}{...}
{p2col :{cmd:xhdfeconnected} {hline 2}}leave-one-out connected set (KSS) as a sample-preparation utility{p_end}
{p2colreset}{...}


{title:Syntax}

{p 8 15 2}
{cmd:xhdfeconnected} {it:workervar} {it:firmvar} {ifin} [{cmd:fweight}]{cmd:,}
{opth gen:erate(newvar)} [{opt replace}]


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
single row with weight 2 or more is a T>1 worker).


{title:Stored results}

{pstd}{cmd:r(n_obs)} (person-year observations in the leave-out set),
{cmd:r(n_obs_input)}, {cmd:r(n_obs_connected)} (after the largest-CC step),
{cmd:r(n_workers)}, {cmd:r(n_firms)}, {cmd:r(n_matches)}, {cmd:r(n_movers)},
{cmd:r(n_stayers)}, {cmd:r(prune_iterations)}.


{title:Example}

{phang2}{cmd:. xhdfeconnected worker_id firm_id, generate(in_leaveout)}{p_end}
{phang2}{cmd:. xhdfeakm y if in_leaveout, worker(worker_id) firm(firm_id) noprune}{p_end}


{title:Also see}

{psee}{helpb xhdfeakm}, {helpb xhdfe}{p_end}
