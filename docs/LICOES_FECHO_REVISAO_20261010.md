# Lições do fecho da revisão — 10out2026

Base: candidata privada xhdfe 2.29.0.20261006. Plano e evidência em
`Verifications/perf_max_20260929/PLANO_FECHO_REVISAO_XHDFE_20261010.md` e
`Verifications/perf_max_20260929/implementacao_fecho_revisao_20261010/`.

## Seleção vazia no executor da bateria

Classificação: **evaluation-harness error**. `run.py run --filter` com um
identificador inexistente devolvia `RECORDED_ATTEMPTS 0` e rc 0. Nenhum job
tinha sido selecionado; esse estado era indistinguível de uma retoma completa
para um consumidor que só olhasse para o rc ou a contagem de novas execuções.

A seleção passa a ser validada antes de saltar jobs já concluídos. O runner
expõe selecionados, concluídos e pendentes, e distingue ALREADY_COMPLETE,
PARTIAL e SELECTED_COMPLETE. Um diretório interrompido é preservado e dá uma
mensagem explícita. O relatório continua a ser a autoridade do veredicto;
um rc 0 do executor não converte resultados falhados em PASS.

Teste permanente: `tests/analytic_precision/runner_selection_contract.py`,
CLI real com OLS sintético, seleção vazia por filtro/backend, seleção válida,
retoma, limite parcial, interrupção e relatório NOT_RUN. Estimador, fixtures,
oráculos e tolerâncias mantêm o comportamento anterior nesta correção.

## B2: a média quadrática não limita cada momento FE

Classificação: **demonstrated estimator error**. Nos solves group/individual
por sweeps, a média quadrática dos scores normalizados podia passar enquanto
uma coluna excedia 1e-9. O caso group_uniform_sum GS CPU devolvia máximo
1,0174745e-9. A correção acompanha o máximo por coluna nas reduções existentes
e usa a continuação já disponível, com a mesma tolerância e método. Fast e
Krylov não recebem esse gate. O teste permanente
`group_column_moment_contract.cpp` injeta uma perturbação analítica que a RMS
esconde, troca as dimensões standard/individual e conserva o controlo Fast.
A primeira campanha native CPU/CUDA deu 113 PASS e 66 recusas esperadas;
a composição final e os casos Stata exigem recibos próprios.

## B1: repor o bloco dos declives pode tornar V indefinida

Classificação: **demonstrated estimator error**, com decisão de política
explícita do mantenedor: conservar os declives e reparar a constante.
Depois do corte PSD e da reposição do bloco B, não basta aumentar V(_cons).
O vetor cruzado c tem de pertencer à imagem de B e a variância da constante
tem de ser pelo menos c'B⁺c. A correção aplica estas duas condições nos dois
caminhos de montagem, conservando exatamente o bloco dos declives.
`intercept_psd_contract.py` testa o helper real contra uma decomposição
independente: singularidade, bloco nulo, direção originalmente negativa,
reescalamento e as 40 matrizes sintéticas capturadas. O teste inicial passou
48 casos. Essa prova algébrica não substitui a validação das interfaces.

## B3: convergência e momentos necessários não certificam b/V

Classificação: **demonstrated estimator error** nos estados antes da recusa
de directors e pf_difficult. Em directors, a própria referência 2.26.2 viola
os limites independentes de b/V; removê-la da recusa devolveria estimativas
erradas. A continuação CUDA podia parar após uma iteração pela pequena mudança
do iterado. A correção consulta os momentos necessários durante essa única
continuação, sem mudar o método, a tolerância ou o budget; inclui as saídas do
acelerador e do CG. `n1_continuation_context.cpp` verifica isolamento entre
threads, aninhamento e restauração após exceção.

O âmbito novo fica limitado a dois FE, onde directors passou o oráculo.
Em pf com três FE, a tentativa mais completa passou b, FE e normal, mas V
continuou fora do contrato (2,3825e-7 contra 1e-8). Essa via foi excluída da
correção e conserva a recusa anterior. PF permanece aberto; não se apertam
alvos nem se acrescentam provas de covariância ao fit para fazer passar o caso.

O primeiro oráculo Schur de PF era insuficientemente validado: as ancoragens
deixavam uma direção nula exata livre. Classificação: **evaluation-harness
error**. A referência corrigida elimina três redundâncias exatas, compara
duas escolhas de ancoragem, verifica a reconstrução e os momentos. O teste
permanente `schur_gauge_contract.py` confronta o oráculo offline com MQO denso
em dados sintéticos. Os resultados anteriores ficaram preservados e qualificados.

## Falhas de alocação não podem atravessar uma região OpenMP

Classificação: **demonstrated exception-safety defect**. A injeção sintética
de malloc falhado no scan de singletons e no solve MLSMR por lanes confirmou
SIGABRT. As exceções passam a ser capturadas em slots prealocados, com todas
as threads a chegar à sincronização; o chamador relança depois da região,
antes de consumir resultados parciais. `worker_allocation_contract.py`
verifica o local da falha, ausência de abort, invalidação do estado e um novo
fit no mesmo objeto. Na primeira build CPU corrigida, os seis casos passaram
e a recuperação reproduziu os bits da execução sem injeção.

## CUDA strict: continuar a correção mantendo a parte dentro das células

Classificação inicial: **missing coverage**, com recusa strict reproduzida em
native e Stata. Em laplacian_sum, a primária e a repetição existente deixavam
uma média FE de cerca de 2,5e-11 para o pedido de 1e-12. Após essas duas vias
falharem, a continuação agora separa a parte já ortogonal dentro de padrões
de incidência idênticos, resolve a correção pequena com a mesma tolerância e
orçamento restante, recompõe o estado e certifica os inputs originais.
As vias default/Fast e as strict já bem-sucedidas não entram nessa correção.

Na composição limitada de 11out, native e Stata passaram o oráculo strict.
`strict_cuda_continuation_contract.py` acrescenta os budgets 1 e 1128, que
devem recusar com estado vazio; o segundo verifica também a limpeza de um
fit anterior e a recuperação no mesmo objeto. O caso positivo termina em
1945 iterações com CUDA efetiva. Um teste inicial tentou preparar esse estado
com OLS sem FE e depois com apenas uma iteração: ambos eram pré-requisitos
inválidos para a API CUDA. Foram corrigidos no harness, sem alterar o solver.

O controlo adicional com budget 1500 encontrou um erro de diagnóstico no
novo ramo: uma correção auxiliar incompleta substituía o último estado
completo, exibindo os seus critérios como se fossem do modelo original.
A correção preserva os diagnósticos do estado completo e contabiliza todo o
trabalho gasto. O teste permanente verifica agora os três budgets e os
contadores na mensagem. A build CPU resultante ficou byte a byte idêntica;
CUDA e Stata receberam nova validação dos bytes alterados.

## IV: o corte PSD pode contaminar uma constante bem calculada

Classificação: **demonstrated estimator error**, herdado e reproduzido também
na composição congelada de 9out. Nos dados Walsh com direção fraca 2^-27,
o cálculo IV e a covariância antes do corte concordavam com os oráculos
MP90/MP120; o corte completo introduzia erros de 8,64e-4 a 2,50e-3 na
covariância escalada. Os três casos usavam partições de clusters equivalentes,
sem pesos ou com pesos analíticos/de frequência.

Se todas as interseções não vazias têm a mesma cardinalidade, as partições
são iguais salvo a rotulagem. A inclusão-exclusão CGM reduz-se então a um
sandwich de uma só partição, que é PSD. A correção IV usa essa identidade
exata para conservar a linha da constante antes do corte; mantém o tratamento
anterior dos declives. Não introduz tolerâncias nem uma passagem pelos dados.

`iv_intercept_covariance_contract.py` e o equivalente CUDA passaram 46 casos
em cada combinação de modo/backend (184), com coeficientes e blocos dos
declives idênticos aos anteriores. `equivalent_cluster_partition_contract.py`
acrescenta 12 casos por backend, incluindo partições distintas com igual
número de clusters: cardinalidades marginais iguais não bastam para ativar
a identidade. Dois desses controlos por backend têm CGM bruto indefinido;
avaliam a política PSD, não um oráculo positivo inaplicável. O primeiro
harness não fazia essa distinção e foi corrigido, preservando os logs.

`stata_iv_intercept_covariance_contract.py` passou as 12 combinações de pesos,
modo e backend contra o mesmo oráculo independente. Os 80 casos Stata B1
ordinários/agrupados voltaram a passar, incluindo `lincom` e `margins`.

## Âmbito das campanhas integradas

Nos bytes scoped CPU/CUDA e no stage_stata_scoped, os mesmos 984 IDs foram
executados: xhdfe tem 619 PASS e 223 recusas esperadas; T01 tem 533 PASS e
219 recusas esperadas. Exatamente 22 classificações B2 e duas strict melhoraram;
a custódia não divergiu. B1 passou ainda 80 matrizes Stata (normal e agrupado),
com zero indefinidas, declives/b/DoF idênticos e lincom/margins coerentes.
Estes resultados não fecham PF, desempenho global, instalação ou plataformas
nativas indisponíveis.
