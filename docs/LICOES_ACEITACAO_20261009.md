# Lições da aceitação N1/N6/N9 — 09out2026

Base privada 2.29.0.20261006; núcleo e binários e560f80a/b9726de5/a342951c
preservados. A revisão independente do diff ficou bloqueada. Este ficheiro
é um suplemento às lições anteriores, não uma certificação do produto.

## D-A precisa do comparador atual

AKM Comparable CPU T1, dois processos frios atuais: candidata 34 599 932 KiB,
2.28.1 29 970 024 KiB; rácio 1,1544846277. Ambos convergiram em 28 iterações;
vinte campos da candidata exatos face a TLS. O gate de 1,15 falha por
134 405 KiB (131,3 MiB) de pico. Não se arredonda o resultado para PASS,
não se muda a margem e não se inicia adjudicação de tempo sem a decisão
reservada ao mantenedor. Classificação: gate de recursos, não erro b/V.

## Testar a partição além da escolha lockstep/lanes

O controlo conjunto de scatter, carries e parciais pode diminuir os chunks
mesmo quando lockstep continua elegível. A grelha inicial não cobria esse
limite. Reprodutor sintético: n=266000, L=250, FE contíguo 10000 e FE
disperso 11184, seed 20261009. TLS usa 48 chunks; candidata usa 47;
ambos mantêm lockstep e 19 iterações. b/V/resíduo mudam de bits; as
diferenças de regressão b/V são aproximadamente 1,8e-16/4,2e-16.

Classificação: cobertura/declaração de contrato em falta, sem erro de
precisão demonstrado. Uma promessa de bits iguais precisa da identidade
da partição e da via. Não acrescentar precisão ou trabalho ao solver para
recuperar bits de um grafo aritmético diferente. Tratar esta classe no
desenho e adjudicá-la independentemente antes de aceitar o despacho.

## Equipa efetiva também no teste do observer

O teste offline exigia a equipa pedida, mesmo quando OMP_THREAD_LIMIT=2
produzia legitimamente dois trabalhadores. Reproduzido: o observer real
reportava corretamente 2 e o teste falhava. Classificação: erro de harness.

Corrigido em tests/analytic_precision/mlsmr_phase_trace_contract.py:
equipas medidas independentemente, slots de trabalho útil, uma passagem
de uma unidade que distingue equipa de trabalhador útil, e NOT_RUN se a
mutação não for distinguível num runtime estritamente serial. Normal e
limite 2 passam; limite 1 conserva a lacuna explícita, sem PASS falso.
Nada mudou no núcleo, binários, tolerâncias ou contratos numéricos.

## Modelar pedidos reais e nomear os limites

O termo Pmax da candidata é um pior caso sobre partições admissíveis,
não o scratch efetivamente alocado no host atual. A documentação submetida
tem de o dizer e justificar o conservadorismo. O revisor apontou essa
diferença; a nota prévia já a continha, mas não estava no diff enviado.

O probe novo extrai as expressões reais de allocate/assign/row_partials e
mede pedidos de elementos. Doze combinações de geometria e capacidades
48/64/96 coincidem com o modelo; omitir carries é rejeitado. Os blocos
grandes são interceptados: não é prova de RSS, OOM ou solver completo.

Uma afirmação do revisor sobre inelegibilidade universal abaixo de 512 MiB
foi refutada com um contraexemplo compilado. Isso não levanta o bloqueio de
desenho nem autoriza a aceitar contrapartidas de memória/desempenho.

## Evidência

Em Verifications/perf_max_20260929/implementacao_bloqueios_20261007/
tranche_dispatch_acceptance_20261009/: AKM_T1_FINAL_GATE.json,
STATA_REAL_ROUTES_GATE.json (dez pares reais PASS), CLAUDE_REVIEW.md e
CLAUDE_REVIEW_RECEIPT.json (diff autorizado exato, zero ferramentas),
REVIEW_TRIAGE.md, PARTITION_BOUNDARY_GATE.json, ALLOCATION_SIZE_CONTRACT.json,
HARNESS_FIX_GATE.json e PLANO_FECHO_D_A_AKM_T1.md.

Fontes de dados lidas no sítio; temporários Stata apenas na pasta privada
autorizada, vinte subpastas vazias no fim. Nenhum ficheiro individual real
foi guardado no repositório ou enviado ao Claude. A revisão recebeu só o
diff autorizado e instruções genéricas; não houve segunda consulta.

Aceitação/performance/publicação continuam NO-GO. Não repetir otimizações
por ajuste de constantes, nem reinterpretar a revisão como aprovação.
