# N2: cobertura dos controlos adicionais

Complemento de [LICOES_N2_ARENA_20261008.md](LICOES_N2_ARENA_20261008.md).
Os controlos abaixo não alteraram o estimador, os dados ou as tolerâncias.

## MLSMR explícito e savefe

O primeiro controlo grande combinava `absorption_method='mlsmr'` com
`retain_fes=True`. Essa combinação é documentadamente recusada, com a mesma
mensagem, no braço selado de 6 de outubro e na candidata. Classificação:
**evaluation-harness error**; o ensaio pedia uma combinação sem suporte.
A recusa permanece **UNSUPPORTED_EXPECTED**, não sucesso numérico. O fonte
V1 e os logs dos dois braços foram conservados.

O controlo final de armazenamento exerce MLSMR sem savefe, com pesos
analíticos e frequência literal, cache fria/quente e uma assinatura alterada.
O savefe é ensaiado separadamente com Gauss–Seidel, além da bateria permanente
e dos controlos de normalização existentes.

## Check pontual adicional dos resíduos

Um segundo controlo acrescentou arbitrariamente `max(abs(r-noise)) <= 1e-7`
ao desenho grande com savefe. O resultado foi 2,665379001953738e-7 nas duas
versões de 6 de outubro; todos os bits das estimativas, covariância, resíduos
e FE coincidiram entre esses braços. O erro dos coeficientes foi
1,7763568394002506e-16 e o erro relativo da covariância 2,220445202217366e-15,
ambos face ao oráculo Walsh independente. A reconstrução teve erro
4,440892098500626e-16. Na 2.26.2, o check pontual também falhou, com erro
0,249999292649965; os resíduos/FE dessa referência não coincidiam com 6out.

Classificação do novo critério de aceitação: **evaluation-harness error**.
O limite pontual não fazia parte do contrato adotado e não autoriza apertar
o solver. O check adicional continua **FAIL** no registo; não se mudou o seu
limiar. A comparação de não regressão e os oráculos b/V são reportados
separadamente. Isto não transforma o ensaio numa certificação geral de FE.

## Recibos e reprodução

Os fontes dos controlos, versões que falharam, recibos e classificações estão
na [evidência N2](../Verifications/perf_max_20260929/implementacao_bloqueios_20261007/n2_h2/):
`arena_resource_guard.py`, `arena_resource_guard_v2.py`,
`arena_resource_guard_v3.py`, `inspect_savefe_walsh.py`,
`RESOURCE_HARNESS_CLASSIFICATION.json`, `SAVEFE_WALSH_CLASSIFICATION.json` e
`ARENA_RESOURCE_GATE.json`. O controlo final tem 12 fits por braço, com
oráculos aprovados e recibos numéricos iguais; as versões anteriores mantêm
as recusas, falhas e linhas não executadas.

O trace C++ do ensaio Stata está em `process.log`, separado de `run.log`.
O leitor do gate deve consultar esse stream para provar o caminho executado.
