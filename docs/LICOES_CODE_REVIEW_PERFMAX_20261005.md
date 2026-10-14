# Lições do code review da candidata de performance — 05out2026

Estado: patch não commitado no worktree `perf/max-performance-20260929`.
Estas lições são evidência de desenvolvimento, não uma certificação de
release. Os caminhos dos reprodutores e recibos abaixo são relativos a
`Verifications/perf_max_20260929/implementacao_code_review_20261005/`.

## CR-1 e CR-3(ii): a memória e os workers fazem parte do cenário suportado

Classe: cobertura em falta, regressão de recursos demonstrada. No caso
sintético `wide`, o lockstep original usou 2698 MB contra 1272 MB nas lanes
da própria candidata e 1050 MB na 2.28.1; falhou com limites virtuais de
2 000 000 e 2 600 000 KB onde as lanes terminaram. O número de RHS e os
blocos privados têm de entrar no despacho antes de reservar o workspace.

Correção candidata: critério só em linhas, RHS, níveis e contiguidade, sem
consultar a equipa; `row_chunks<8` também envia às lanes o desenho
`sparse_levels_6m`. O `wide` corrigido usa 1050–1051 MB, passa ambos os
limites, converge em 14 iterações e reproduz b, V, RSS, amostra e hash dos
resíduos das lanes. O desenho de 6 milhões de linhas terminou em 19
iterações, com os mesmos bits e cerca de 6,88 GB em ambos os braços de
lanes. Reprodutores: `cr1/memory_grid.py`, `cr1/receipt.py`, os logs
`cr1/*_after.*` e `cr1/sparse6m_*`. A grelha completa de 1/5/20/60/250 RHS
× 1/2/8/16 threads está em `cr1/grid_*`.

## CR-2: o pico da arena não é inferível da identidade do fator

Classe: cobertura em falta. O caso sintético `dense` usava 2759 MB na arena
contra 1287 MB na construção de referência, com o mesmo hash de resíduos
`95f07fa76fdbc120`. Uma primeira tentativa de reciclar fatias manteve o
hash mas usou 2764 MB; foi retirada. A correção candidata escolhe a
construção de referência para grafos abaixo de 262 144 vértices, mantendo a
arena no grafo grande H1. O caso `dense` corrigido usou 1287–1291 MB e
terminou com limites virtuais de 2 400 000 e 3 000 000 KB. Reprodutor e
primeira tentativa refutada: `cr2/`.

O corte de vértices protege os desenhos medidos; não prova que a arena de
grafos maiores nunca exceda a memória da referência. A comparação de tempo
em toda a matriz continua obrigatória.

## CR-6 e CR-9: um interruptor de auditoria não pode falhar aberto

Classe: erro do harness. `XHDFE_MLSMR_BUILD=reference` escolhia a construção
antiga mas mantinha o solve lockstep. No caso `small`, devolvia o hash
`4c79fc2a2197465a` em vez de `4a28d8f46eef2fa5` da 2.28.1. Valores
inválidos e espaços em interruptores podiam cair silenciosamente no default.

Correção candidata: `BUILD=reference` ou `legacy` e opções que requerem a
semântica das lanes excluem o lockstep; as 13 variáveis novas são lidas e
validadas por fit, incluindo no certificado CUDA. O teste permanente
`tests/audit_environment_contract.py` passou 104/104 combinações. O caso
`small` com `BUILD=reference` passou a reproduzir o hash da 2.28.1. Logs e
nota de desenho: `cr6_cr9/`.

## CR-7: uma exceção em OpenMP tem de regressar ao chamador

Classe: cobertura em falta. Injetar uma falha de alocação na agregação de
pares MLSMR terminava Python com `rc=134`. A correção guarda uma exceção por
tarefa e relança-a depois da região OpenMP; scratch pequeno passou para o
chamador. Injeções nas regiões de pares, Schur e `apply_lanes` devolveram
`PYTHON_MemoryError std::bad_alloc` com `rc=0` no reprodutor; fits normais
de 5/20/60 RHS mantiveram b, V, RSS, amostra e resíduos bit a bit.
Reprodutores: `cr7/before/`, `cr7/after/`, `cr7/survey_400.err` e
`cr7/fail_apply_56.*`. Os sítios já presentes na 2.28.1 ficam fora desta
correção.

## CR-15: um oráculo tem de entrar no ramo que pretende cobrir

Classe: cobertura em falta. O oráculo antigo nunca chegava a 65 536 níveis
num domínio. `lockstep_team_domain_oracle.py` acrescenta domínio conectado,
componentes desconectados com pesos, lane nula e scan forçado. O trace
mostrou `team_domains=1`; os três primeiros casos deram 48 fits PASS no
binário congelado e no corrigido, com bits invariantes a 1/2/8/16 threads.
O scan opt-in contou 10–13 passagens e manteve b, V, resíduos e iterações
idênticos com a variável desligada. Recibos em `cr15/`.

## CR-4 e CR-5/8: contratos de threads e de compilador precisam de gates próprios

CR-4 permanece parcial. O lookup de DoF passou a respeitar pedidos
explícitos abaixo de 1 048 576 linhas, mantendo o limiar no modo automático.
Dez pares intercalados a 1 040 000 linhas mediram medianas de 1,602 ms antes
e 0,093 ms depois nessa fase; o tempo do fit completo foi ruidoso. Os
desenhos A/B mantiveram os campos numéricos e inteiros; o teste funcional
passou para 1/2/8/16/24/48 workers. Outros limiares listados em CR-4 não
foram alterados.

Em CR-8, a sonda passou a testar também a contração implícita dentro de
`XHDFE_EXACT_FP_BODY`. A grelha GCC/clang × quatro alvos × quatro políticas
de contração tem 32 resultados em `cr8_cr5/FMA_MATRIX.tsv`. Clang com FMA
precisa de `-ffp-contract=off`; GCC Linux mantém as flags e os bits normais.
Em CR-5, CMake, os dois scripts de plugin e os Makevars R executam uma sonda
nas builds nativas. Compilação cruzada é `NOT_RUN`, e uma build marcada como
release não a pode aceitar como PASS. A instalação R em biblioteca isolada,
os dois plugins CPU e os dois plugins CUDA `sm_90` passaram os portões neste
servidor. Windows e macOS não foram executados.

## Limite de certificação encontrado nesta execução

A bateria analítica nativa CPU planeou e executou 224/224 jobs. A graduação
histórica contou 185 PASS e 39 recusas esperadas, mas a adjudicação T01
marcou cinco `REAL_ESTIMATOR_FAILURE`: momentos de FE acima do piso de 1e-9
em `group_sum`, `group_uniform_sum` e `group_fweights_sum` com GS/SGS. Os
mesmos cinco IDs e as métricas T01 completas são numericamente idênticos no
relatório congelado
`Verifications/perf_max_20260929/auditoria_codex_20260930/campaign_cand/report_001.json`;
também estão no relatório da referência. São falhas herdadas do contrato T01,
não foram ocultadas nem corrigidas por alterações de tolerância nesta tranche.
O relatório novo está em
`implementacao_code_review_20261005/analytic_native_cpu/report_001.json`.

## Erro do harness V08 detetado no smoke real

Classe: erro do harness, ainda aberto. Com o plugin CUDA de staging, o V08
de uma repetição deu convergência e o log textual mostrou 23 iterações em
CPU e 77 em CUDA, mas o CSV agregado deixou `iterations` e
`absorption_method` vazios. Reprodutor:
`implementacao_code_review_20261005/cr8_cr5/v08_stage_cuda/` (log, CSV e
metadados com SHA no relatório). Não se substituiu o campo ausente por zero
nem se usou esse recibo como prova da matriz C2. A correção do script e um
teste permanente do parser ficam para trabalho próprio; nenhuma linha do
estimador foi alterada para acomodar a falha.
