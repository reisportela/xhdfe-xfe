# Lições — despacho MLSMR com TLS tardio, 09out2026

Aplica-se à candidata privada **2.29.0.20261006**, no worktree
`build-worktrees/perf-max-20260929`. A publicação continua **NO-GO**.

## N1: o número de RHS não dispensa o teste de memória

O reprodutor de 31 RHS da revisão de 06out falhava sob limites de espaço
virtual de 1 300 000 e 1 600 000 KiB. A regra anterior dispensava o teste
`private_bytes + level_bytes <= 3*row_bytes` abaixo de 32 RHS. Isto é um
problema de recursos; não demonstra erro nos coeficientes nem autoriza
alterar o solver, a tolerância ou a paragem.

A primeira candidata de despacho de 08out foi refutada em quatro células
de memória e foi revertida. Essa rejeição mantém-se. A investigação
posterior isolou uma reserva TLS comum sem uso nas lanes; a sua alocação
tardia foi validada separadamente. A tentativa de 09out combina essa
correção com o teste conservador para todos os RHS, usando as mesmas
constantes. Não ajusta as margens para aceitar resultados observados.

O novo header `src/lockstep_memory_plan.hpp` calcula a decisão a partir de
linhas, RHS, níveis e contiguidade. Scatter, carries contíguos e parciais
partilham o orçamento. Overflow produz uma decisão não elegível. O modelo
não pretende limitar toda a memória do processo: os picos físicos são uma
verificação separada.

Na grelha sintética de 64 células e três braços, todos os 192 fits
convergiram. Nenhuma célula excedeu a margem de 15% face à 2.28.1; o maior
rácio foi 1,0850601. O reprodutor de 31 RHS completou com 14 iterações,
uma e duas threads, incluindo os dois limites de espaço virtual. O MQO
explícito com dummies independentes passou em Python e Stata.

Há uma contrapartida: três células contíguas com muitas threads usam
45–48% mais RSS do que a candidata só com TLS, embora fiquem 1–2% acima
da 2.28.1. Uma quarta célula usa 15,01% mais do que TLS. A tabela completa
preserva estas perdas. Este resultado não é aceitação de desempenho.

## N6: distinguir a equipa dos trabalhadores úteis da fase

O trace opt-in regista unidades, equipa e trabalhadores úteis por fase:
linhas, níveis, cinco passagens de precondicionamento e RHS nas lanes.
Cada fase tem um observer próprio. O `threads_used` agregado continua a
representar o máximo observado; não prova que a fase dominante utilizou
toda a equipa.

No reprodutor E2 de seis milhões de linhas, a nova decisão usa lanes e o
trace mostra quatro RHS, equipa de quatro e quatro trabalhadores úteis,
com pedidos globais de oito e dezasseis threads. Mantém 28 iterações.
Esta é uma limitação estrutural explícita, não uma prova de escalabilidade.

O teste permanente extrai a classe real do núcleo e compara os seus
registos com slots independentes de trabalho útil. Um controlo negativo
que reporta a equipa como trabalhadores ativos é rejeitado.

## N9: decisão portátil e execução com capacidade efetiva

A decisão usa a constante histórica de 48; não consulta a equipa,
capacidade, RSS, memória livre ou `ulimit`. A execução pode ter mais de
48 chunks se o orçamento conjunto o permitir. O contrato offline verifica
capacidades simuladas de 48, 64 e 96, a soma dos buffers alocados e overflow.
Não certifica hardware com 64 ou 96 processadores.

As seis células CPU protegidas de ready, AKM e simulated, em fast e
comparable, conservaram método, despacho, iterações e vinte campos do
recibo. Os dez desvios CPU nativos previstos ocorreram e passaram a
regressão de b/V/amostra; os seus dez pares Stata reais estão NOT_RUN,
à espera de autorização específica para a pasta de temporários.

As classes desviadas são **equivalentes, não bit-idênticas**. A grelha
de projeção usa dummies/MQO independentes. Paridade com outro xhdfe é
evidência de regressão, não um oráculo matemático.

## Falhas e limites que permanecem

- D-A AKM T1: a candidata TLS anterior mediu rácio 1,1548745, acima de
  1,15. A nova combinação não fecha esse gate por mudar a margem ou o
  caminho protegido. A aceitação continua reservada ao mantenedor.
- Dez pares de tempo por célula desviada e core24×8: NOT_RUN, após D-A.
- B1/B2/B3 e as duas recusas CUDA estritas já classificadas na tranche 4
  não foram corrigidas por este patch. Não há certificação global.
- Os diagnósticos CUDA suplementares de resíduos continuam FAIL nos
  recibos; os PASS de b/V, uso real do H100 e equipas têm âmbito próprio.
- Windows/macOS nativos, instalação/fórmula/CUDA de R e revisão externa
  deste novo diff continuam pendentes.
- A primeira preparação Stata E1 encontrou um caminho de fixture errado
  e parou antes de escrever dados. Foi corrigido apenas o harness; a
  fixture sintética e o oráculo originais ficaram intactos.

## Evidência e reprodução

Pasta: `Verifications/perf_max_20260929/implementacao_bloqueios_20261007/`
`tranche_dispatch_tls_20261009/`.

- Desenho prévio: `NOTA_DESENHO_DISPATCH_TLS_20261009.md`.
- Memória: `MEMORY_MATRIX_RESULT.json`, `MEMORY_GRID.md` e `runs/`.
- Oráculos: `E1_ORACLE.json`, `STATA_E1_ORACLE_GATE.json`,
  `PROJECTION_GATES.json`; fases E2: `E2_PHASE_GATE.json`.
- Vias protegidas e core: `PROTECTED_CPU_GATE.json`,
  `CORE_NATIVE_ROUTES_GATE.json`, `CELL_LIST.md`.
- Patch próprio: `DISPATCH_TLS_FINAL.diff`; hashes antes/depois e
  inventário: `DISPATCH_TLS_SOURCE_MANIFEST.json`, `DELIVERY_CUSTODY.json`.

Testes permanentes: `xhdfe_lockstep_memory_plan_contract` no CTest e
`tests/analytic_precision/mlsmr_phase_trace_contract.py`. O teste negativo
isolado reintroduz a exceção RHS<32 e tem de falhar. Não se transplanta
nenhum oráculo para o fit.

Para reverter apenas esta tentativa, começar por verificar os hashes e
`git apply --reverse --check` do patch próprio. O inverso preserva tranche
1 V2, H2 e TLS; não usar reset/restore nem apagar a árvore de evidência.
