# Driscoll–Kraay — implementação e lições, 11 de outubro de 2026

Candidato 2.29.0.20261006, na pasta
`Verifications/perf_max_20260929/implementacao_fecho_revisao_20261010/dkraay_20261011/`.
O pedido explícito de implementar DK substituiu o adiamento anterior.

## Referência aproximada não é oráculo matemático

**Classe: evaluation-harness error.** `part1/dkraay.do` exige concordância
1e-12 entre aproximações reghdfe/ivreghdfe. No caso público `auto`, dois FE,
o erro de V contra uma projeção racional exata é 3,88e-10 na referência e
4,14e-12 no xhdfe, na escala diagonal do contrato publicado.

Reprodutor: `strict_reference.do`; prova: `rational_reference.py` e
`diagnostics/strict_reference_r1/EXACT_RATIONAL.json`.
Correção: substituir apenas esta comparação, no port documentado, pelo
oráculo racional com o contrato publicado. Preservar a falha bruta e as
asserções restantes. Nunca alterar o solver ou tolerâncias para imitar
erro da referência. Teste permanente: entrada DK em `tests/analytic_precision/supplemental/`.

## Nesting temporal e pequenos ajustamentos

**Classe: evaluation-harness error.** O primeiro oráculo sintético esqueceu
a redundância da constante quando os FE de tempo são nested. Corrigiu-se
G para G−1 no oráculo; o candidato já seguia o contrato do reghdfe.
`oracle/cpu_r1/` conserva a execução inicial. O caso tornou-se um controlo
permanente nos oráculos CPU/CUDA.

O ajuste DK não subtrai a constante nested adicional usada pelo clustering
convencional. A ajuda explica por que DK(1) e cluster(time) podem diferir
nesse caso. Os desfasamentos usam períodos observados, como o upstream
descarregado, e essa escolha tem teste explícito para períodos ausentes.

## Provas e interfaces

**Classe: missing coverage.** A prova interna existente de clustering não
cobre DK. O opt-in XHDFE_CERTIFY identifica explicitamente essa cobertura
como unsupported; os oráculos offline validam DK separadamente.

O controlo negativo Stata foi corrigido para usar programa eclass e
`ereturn matrix ..., copy`; r(152) e r(111) iniciais eram erros desse harness,
preservados nos logs. A versão final deteta as duas perturbações injetadas.

Ao acrescentar uma opção ao struct C++, recompilar e alinhar os dois plugins.
Os testes de carregamento conjunto/cache CPU e CUDA e o smoke de 46.156.187
observações passaram; b/V, amostra e resíduos do percurso anterior coincidiram
exatamente. Windows/macOS, R/Python e core24×8 completo permanecem NOT_RUN.
