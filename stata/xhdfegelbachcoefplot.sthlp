{smcl}
{* *! version 1.0.0  25jul2026}{...}
{vieweralsosee "xhdfegelbach" "help xhdfegelbach"}{...}
{vieweralsosee "xhdfegelbachbootstrap" "help xhdfegelbachbootstrap"}{...}
{vieweralsosee "xhdfegelbachetable" "help xhdfegelbachetable"}{...}
{title:Title}

{p2colset 5 28 30 2}{...}
{p2col :{cmd:xhdfegelbachcoefplot} {hline 2}}Identity-preserving Gelbach waterfall{p_end}
{p2colreset}{...}

{pstd}{bf:Version 1.0.0 (25jul2026), distributed with xhdfegelbach 1.5.0.}{p_end}


{title:Syntax}

{p 8 15 2}
{cmd:xhdfegelbachcoefplot}
[{cmd:,} {it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{synopt :{opt focal(name)}}one X1 name; default first {cmd:r(focal_names)}, then first {cmd:r(x1_names)}{p_end}
{synopt :{opt keep(pattern)}}retain component names matching a regular expression{p_end}
{synopt :{opt drop(pattern)}}drop matching retained component names{p_end}
{synopt :{opt exact}}interpret keep/drop as exact name lists{p_end}
{synopt :{opt labels(spec)}}colon-separated {it:group = display label} entries{p_end}
{synopt :{opt noannotate}}omit signed movement-share labels{p_end}
{synopt :{opt noother}}do not draw the filtered aggregate{p_end}
{synopt :{opt title(text)}}custom title{p_end}
{synopt :{opt notes(text)}}custom footer{p_end}
{synopt :{opt name(name)}}store the graph in memory{p_end}
{synopt :{opt saving(filename)}}save a Stata graph{p_end}
{synopt :{opt replace}}replace the saved graph{p_end}
{synopt :{opt nodraw}}construct without displaying{p_end}
{synoptline}
{p2colreset}{...}


{title:Description}

{pstd}
This command must immediately follow {cmd:xhdfegelbach} or
{cmd:xhdfegelbachbootstrap}. It draws the path from the base coefficient,
through the simultaneous component contributions, to the full coefficient.
Positive and negative contributions use distinct colors. Signed percentages
are shares of total movement and are omitted when that denominator is too
small.{p_end}

{pstd}
Selection is evaluated on original group names before labels are applied.
Each colon-separated {opt labels()} entry is parsed independently, so two or
more display labels cannot collapse into the first step.
By default, every component hidden by keep/drop is summed into one
{cmd:Other (filtered)} step. Thus filtering does not falsify the accounting
identity and the waterfall still reaches the full coefficient. The explicit
{opt noother} option suppresses that bridge and the reported filtered residual
then quantifies the omitted movement.{p_end}

{pstd}
The graph is post-processing only and does not refit the model. Component
labels remain specification-accounting labels, not evidence of causal
mediation.{p_end}

{pstd}
Graph commands overwrite Stata's volatile {cmd:r()} results. Create all
{helpb xhdfegelbachetable} outputs before calling
{cmd:xhdfegelbachcoefplot}; the plot should be the final reporting command
after each decomposition. This command returns no stored results.{p_end}


{title:Examples}

{phang2}{cmd:. xhdfegelbach lwage, x1(educ) focal(educ) x2groups("human = ability : job = tenure exper") commonfes(year) fes(firm)}{p_end}
{phang2}{cmd:. xhdfegelbachcoefplot, keep("human") labels("human = Human capital") name(gelbach_educ)}{p_end}


{title:Also see}

{psee}
{helpb xhdfegelbach}, {helpb xhdfegelbachbootstrap},
{helpb xhdfegelbachetable}
{p_end}
