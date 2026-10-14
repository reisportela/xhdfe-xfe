# Fecho dos achados N1, ajuda e CUDA/Walsh — 11out2026

Base: candidato privado 2.29.0.20261006, já com PF3FE, IV/PSD, regressões
upstream e Driscoll–Kraay. Evidência nova em
`Verifications/perf_max_20260929/implementacao_fecho_revisao_20261010/fecho_achados_20261011/`.

## N1: verificações que desapareciam com NDEBUG

Classe: **evaluation-harness error**. O teste de contexto usava assert e
podia perder as verificações numa compilação de release. Os recibos anteriores
tinham asserções ativas; não são invalidados por esta fragilidade.

Correção: verificações explícitas, contagem atómica de falhas, exit 1 e
marcadores terminais. A exceção esperada tem tipo próprio, impedindo que uma
falha do avaliador seja confundida com a exceção do predicado. Um modo negativo
corrompe o ponteiro efetivo dentro do contexto e tem de ser detetado.

Reprodutor permanente: `tests/analytic_precision/n1_context_build_contract.py`.
Resultado: dois casos positivos PASS; os dois casos negativos devolvem 1,
com e sem NDEBUG. Nove verificações ativas por execução. Nenhum código de
produto foi alterado.

## Ajuda: informação obsoleta e política PSD incompleta

Classe: **evaluation-harness error**, aplicada à incoerência da documentação
com a implementação e os recibos. A ajuda continuava a listar DK como ausente
e directors/PF3FE como não resolvidos; não explicitava a política da constante.

Correção na nova cópia do candidato: DK retirado da lista de funcionalidades
ausentes; limitações HAC remanescentes preservadas; estados directors/PF3FE
atualizados; política PSD documentada, incluindo preservação do bloco dos
declives já corrigido, projeção das covariâncias cruzadas e ajuste da variância
da constante. Ambos os modos e a consequência para contrastes ficam explícitos.

## CUDA/Walsh: diagnóstico adicional interpretado como defeito do estimador

Classe: **evaluation-harness error**. O diagnóstico absoluto pontual <=1e-7
foi comparado com o contrato publicado antes de qualquer alteração do solver.
Com os mesmos inputs e opções, a 2.26.2 congelada tem erro máximo
1,9101873993676577e-7; a 2.29 tem até 1,910187412690334e-7 nas seis equipas.
Ambas convergem em 14 iterações, com CUDA efetiva e contratos b/V/FE satisfeitos.

O oráculo é independente e exato: padrões Walsh de quatro linhas por aresta
dão X'noise=D'noise=0; as arestas identidade e sucessor tornam o grafo conectado,
logo rank(D)=2G-1. Nos 3.145.728 inputs dyadic representados,
b=(0,5;−1,25), RSS=N e V=I/2.621.439.

No candidato: erro b escalado <=1,78e-16 (limite 1e-9); erro V <=2,89e-15
(limite 1e-8); momento FE máximo <=1,16e-10 (limite 1e-9).
Momentos normais, RSS, amostra, graus de liberdade e uso CUDA também passam.
Perturbações injetadas num FE e num regressor são detetadas.

O FAIL pontual original fica preservado, com o mesmo limite. Não foi
convertido em PASS, não se alteraram tolerâncias, não se acrescentou polimento
ao fit e não se corrigiu o estimador para exceder a referência publicada.
Fecho: interpretação do diagnóstico corrigida e cobertura permanente acrescentada.

Reprodutores: `stata_walsh_residual_contract.{py,do}` e
`walsh_fixture_oracle.py`, em `tests/analytic_precision/`.
Adjudicação: `WALSH_CLASSIFICATION.json`, na pasta desta tranche.

## Custódia e alcance

A pasta DK anterior permanece intacta. O novo staging altera apenas a ajuda;
fontes do estimador, ados e ambos os plugins são byte-idênticos à base DK.
As mudanças na worktree são testes e documentação, com os bytes anteriores
preservados em `before/`. Este fecho não constitui promoção nem certificação
das plataformas/interfaces/matriz de desempenho ainda pendentes da release.
