{smcl}
{* *! version 1.0.0  28jul2026}{...}
{vieweralsosee "xhdfegelbach" "help xhdfegelbach"}{...}
{vieweralsosee "xhdfegelbachbootstrap" "help xhdfegelbachbootstrap"}{...}
{vieweralsosee "xhdfegelbachcoefplot" "help xhdfegelbachcoefplot"}{...}
{title:Title}

{p2colset 5 28 30 2}{...}
{p2col :{cmd:xhdfegelbachetable} {hline 2}}Multi-panel Gelbach tables{p_end}
{p2colreset}{...}

{pstd}{bf:Version 1.0.0 (28jul2026), distributed with xhdfegelbach 1.5.0.}{p_end}


{title:Syntax}

{p 8 15 2}
{cmd:xhdfegelbachetable}
[{cmd:,} {it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{synopt :{opt panels(type)}}{cmd:all}, {cmd:levels}, {cmd:share_base}/{cmd:share_full}, or {cmd:share_movement}/{cmd:share_explained}{p_end}
{synopt :{opt format(type)}}{cmd:display} (default), {cmd:markdown}/{cmd:md}, {cmd:latex}/{cmd:tex}, {cmd:html}, or {cmd:csv}{p_end}
{synopt :{opt focal(names)}}X1 names; default {cmd:r(focal_names)}, then {cmd:r(x1_names)}{p_end}
{synopt :{opt keep(pattern)}}retain component names matching a regular expression{p_end}
{synopt :{opt drop(pattern)}}drop matching retained component names{p_end}
{synopt :{opt exact}}interpret keep/drop as exact name lists{p_end}
{synopt :{opt labels(spec)}}colon-separated {it:group = display label} entries{p_end}
{synopt :{opt noother}}suppress the default {cmd:Other (filtered)} aggregate and warn that filtered rows do not close{p_end}
{synopt :{opt digits(#)}}printed decimal places, 0 through 12; default 3{p_end}
{synopt :{opt caption(text)}}optional table caption{p_end}
{synopt :{opt notes(text)}}additional table note{p_end}
{synopt :{opt level(#)}}analytic confidence level when no bootstrap is present; default {cmd:c(level)}{p_end}
{synopt :{opt interval(type)}}{cmd:auto} (default), {cmd:normal}, or require {cmd:bootstrap}{p_end}
{synopt :{opt sharetol(#)}}override the inherited absolute share-denominator threshold{p_end}
{synopt :{opt sharetmin(#)}}override the inherited weak-denominator t threshold{p_end}
{synopt :{opt saving(filename)}}write the selected markup format{p_end}
{synopt :{opt replace}}replace an existing output file{p_end}
{synoptline}
{p2colreset}{...}


{title:Description}

{pstd}
This command must immediately follow {cmd:xhdfegelbach} or
{cmd:xhdfegelbachbootstrap}; any intervening r-class command can overwrite the
result. It reports the base endpoint, selected component contributions, total
movement and full endpoint. The three panels are levels, signed shares of the
base coefficient and signed shares of total movement. The aliases
{cmd:share_full} and {cmd:share_explained} match the reviewed PyFixest
reporting vocabulary. {opt focal()} accepts X1 names only; the normalization-
dependent implicit {cmd:_cons} row is deliberately unavailable in this table
surface.{p_end}

{pstd}
After {cmd:xhdfegelbachbootstrap}, percentile/basic intervals and bootstrap
SEs are used automatically wherever available, including each component row
from {cmd:r(bootstrap_delta_ci)}; {cmd:confidence_method} is stamped
{cmd:bootstrap_percentile} or {cmd:bootstrap_basic}. {opt interval(normal)}
forces analytic reporting, while {opt interval(bootstrap)} fails unless an
immediate bootstrap result is available. After the point command,
analytic normal/delta-method intervals are used where the public covariance
contract permits them. This includes component/base and component/movement
share intervals from the joint covariance, denominator variance and relevant
cross-covariances. A weak denominator retains its numerical interval but
stamps {cmd:normal_delta_diagnostic_only_weak_denominator}; unavailable
intervals remain {cmd:not_available}.{p_end}

{pstd}
Selection uses original group names and relabelling happens afterwards.
Multiple colon-separated {opt labels()} entries remain independent labels.
When {opt keep()} or {opt drop()} filters components, their contributions are
summed by default into one {cmd:Other (filtered)} row in every panel. Its SE
uses the joint covariance from the full omitted-component sub-block of
{cmd:r(cov)}, including cross-component covariances; component SEs are never
added. The row is
reconciled at the requested display precision so displayed component rows
close to the displayed total. {opt noother} restores the former filtered
shape but emits a warning and note that the displayed rows no longer preserve
the accounting identity.
The output is post-processing only: it does not refit or mutate the stored
decomposition. Notes always identify the result as specification accounting,
not causal mediation, and preserve the bootstrap nonregularity warning.{p_end}


{title:Examples}

{phang2}{cmd:. xhdfegelbachbootstrap lwage, x1(educ) x2groups("human = ability : job = tenure exper") commonfes(year) fes(firm) method(pairs) reps(499) seed(42)}{p_end}
{phang2}{cmd:. xhdfegelbachetable, panels(all) format(markdown) keep("human|job") labels("human = Human capital : job = Job controls") saving(gelbach.md) replace}{p_end}


{title:Also see}

{psee}
{helpb xhdfegelbach}, {helpb xhdfegelbachbootstrap},
{helpb xhdfegelbachcoefplot}
{p_end}
