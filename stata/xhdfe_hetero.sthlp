{smcl}
{* 05jun2026}{...}
{title:xhdfe_hetero}

{pstd}
{cmd:xhdfe_hetero} is a quarantined experimental command for PR9 heterogeneous
absorbed slopes. It is not the production {cmd:xhdfe} path.

{title:Syntax}

{p 8 15 2}
{cmd:xhdfe_hetero} {it:depvar} [{it:indepvars}] {cmd:,}
{cmd:absorb(}{it:fe}{cmd:#c.}{it:x}{cmd:)}
[{it:xhdfe_options}]

{pstd}
The command also accepts {cmd:absorb(}{it:fe}{cmd:##c.}{it:x}{cmd:)}.

{title:Guardrail}

{pstd}
The command refuses models without heterogeneous absorbed slopes. Use
{cmd:xhdfe} for all existing production models.

{pstd}
The command also refuses heterogeneous slopes with {cmd:group()} /
{cmd:individual()}.

{title:Example}

{phang2}{cmd:. sysuse auto}{p_end}
{phang2}{cmd:. xhdfe_hetero price length, absorb(rep78#c.weight)}{p_end}

{title:Status}

{pstd}
This command is for testing Tiago Tavares's PR9 implementation without merging
that PR into the production command. The default {cmd:xhdfe} command continues
to reject heterogeneous slopes.
