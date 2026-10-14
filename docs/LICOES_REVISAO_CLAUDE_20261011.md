# Lições da revisão Claude Fable 5.1 — 11 de outubro de 2026

## IV: a correção de partições equivalentes não cobre partições distintas

Classificação: **demonstrated estimator error**, herdado; corrigido no
candidato privado `iv_psd_distinct_20261011` da 2.29.0.20261006.
Na sequência da hipótese B1 do parecer estático, os mesmos inputs Walsh
foram testados com direção fraca 2^-27, partições distintas de 17 e 23
clusters, pesos analíticos/de frequência e modos Fast/Comparable.
O oráculo MP90/MP120 e os limites anteriores foram usados sem alteração.

Nos quatro casos com médias nulas, a variância da constante falha: erro
relativo 0,00680877336 com pesos analíticos e 0,00678100554 com frequências,
contra limite 1e-8. Os quatro controlos com médias (1,-1) passam.
A composição congelada de 9out apresenta os mesmos erros na mesma entrada.
Os coeficientes permanecem exatos dentro do contrato. A correção de partições
equivalentes não entra nestes casos, corretamente, porque as partições diferem.

Na build diagnóstica congelada de 9out, a covariância antes do corte PSD
concorda com o oráculo; depois do corte, a entrada da constante falha.
Esse controlo localiza a contaminação na correção PSD, sem alterar o solver.

Reprodutor permanente:
`tests/analytic_precision/iv_weak_distinct_covariance_contract.py`.
Evidência da reprodução e comparação na pasta
`Verifications/perf_max_20260929/implementacao_fecho_revisao_20261010/claude_followup/`.
Correção: `run_ols_wide` classifica a matriz aumentada dos scores, já acumulada
em precisão alargada, antes da congruência pelo bread mal condicionado e do
arredondamento para binário64. Pivôs LDL' estritamente positivos e finitos
permitem preservar a linha da constante já correta. O ajuste dos declives
mantém-se. Resultados singulares, não positivos ou inconclusivos conservam a
política PSD anterior. Não há novo limiar espectral, passagem pelos dados,
critério de convergência ou recusa de fit. A identidade de partições iguais
continua limitada ao caso exato anterior.

Os oito casos passam em CPU e CUDA, com erro máximo fullV de
1,7497312980638578e-15, b e bloco dos declives idênticos bit a bit aos
anteriores e uso real da H100. Os oito casos Stata com partições distintas
também passam em CPU/CUDA. O teste permanente
`tests/analytic_precision/iv_augmented_meat_contract.py` verifica 16 matrizes
positivas, indefinidas, singulares e reescaladas contra Cholesky MP120,
incluindo sinais guardados apenas na parte baixa e rejeição de valores não
finitos. Os controlos integrados de CGM indefinido e os 80 casos OLS
normal/agrupado preservam a política escolhida.

Evidência, fontes, builds e reversão da correção:
`Verifications/perf_max_20260929/implementacao_fecho_revisao_20261010/iv_psd_distinct_20261011/`.
O parecer Claude anterior antecede esta correção; os novos ficheiros não
foram enviados ao serviço externo.

O exemplo de helper com B=diag(1,1e-32), c=(0,5;1e-15) e a=0,1 devolve
100,25 para a variância da constante. Esse valor é exatamente c'B^-1c sob
a política escolhida de conservar todo o espectro positivo. Tratar a segunda
direção como nula mudaria a política: esse exemplo isolado não é o reprodutor
do erro integrado acima e não autoriza um limiar de valores próprios novo.

## Um parecer estático requer confirmação no fluxo completo

As hipóteses B2, B4 e B5 do parecer não se confirmaram na inspeção local:

- A saída honesta retorna antes da chamada do ramo `ssq==0`; não existe a
  dupla avaliação alegada depois de uma aceitação na saída honesta.
- `requested_tol` é definido pelo mesmo `effective_absorption_tolerance`
  usado no limite por coluna; os dois limites usam também o mesmo piso 512eps.
- O segundo caminho é `fit_grouped`, cuja API C++ não recebe instrumentos.
  As interfaces Python/R/Stata rejeitam a combinação agrupada com IV.

A dependência do teste de contexto N1 em `assert` com NDEBUG, a cobertura de
falha de alocação no dispositivo, a continuação CUDA anteriormente bem-sucedida
e a documentação pública da política da constante ficam registadas para
seguimento. A revisão leu apenas o pacote autorizado e não executou testes.
