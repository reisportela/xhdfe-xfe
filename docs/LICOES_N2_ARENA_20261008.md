# N2: vida útil da adjacência approx-Cholesky

Estado: alteração experimental na candidata privada 2.29.0.20261006.
A aceitação depende dos gates de memória, tempo e revisão independente do
[plano](../Verifications/perf_max_20260929/PLANO_RESOLUCAO_N2_20261007.md).
Esta lição não levanta o NO-GO global nem certifica uma release.

## Defeito, reprodução e classificação

No desenho sintético N2 com seis milhões de linhas, g1=1 000 000,
g2=300 000, duas regressoras e oito threads, a arena conservava até ao fim
da fatorização as fatias substituídas e as listas eliminadas. A reprodução
selada usou 21 935 MiB, contra 7 906 MiB com a construção de referência.
As estimativas, resíduos e 12 iterações coincidiam. Sob RLIMIT_AS de
14 000 000 KiB, a candidata antiga recusava por memória um caso em que a
2.28.1 terminava. O processo podia sair com rc 0 apesar de MemoryError.

O defeito é de armazenamento/recursos; não foi demonstrado erro numérico do
estimador. No vocabulário do Precision Contract, faltava cobertura do cenário
suportado (*missing coverage*). Um rc 0 ou um hash correto sem verificar
sucesso e convergência não fecha essa cobertura.

O recycler anterior conservava blocos e não reduzia o pico. A hipótese de
teto posterior foi rejeitada por desviar ready para a construção de
referência. Não se alterou esse teto para fazer o gate passar.

## Alteração e invariantes

`src/ac_arena_storage.hpp` divide a reserva inicial em slabs com donos
explícitos. Uma lista crescida possui o seu buffer; uma substituição destrói
o anterior só depois de a cópia e a publicação terem terminado. A eliminação
liberta a propriedade inicial e os buffers próprios com capacidade superior
a 64, conservando a política de buffers pequenos da referência.

`build_arena` conserva entradas, ordenações, índices inversos, pesos, RNG,
amostras e operações FP. O despacho, os métodos, as tolerâncias e os critérios
de paragem permanecem os do ponto de partida. O armazenamento só produz
contadores agregados quando o trace existente é pedido.

Os ensaios iniciais da alteração usaram 8 019 MiB e 12 iterações com os
mesmos hashes, e 8 017 MiB sob o limite virtual. A 2.28.1 terminou nesse
mesmo limite com 8 262 MiB. Estes são recibos de um desenho de recursos;
não provam aceitação da matriz completa ou ganho de tempo.

## Guardas permanentes e controlos negativos

- `tests/ac_arena_storage_contract.cpp`: propriedade de slabs partilhados e
  dedicados, crescimento, libertação, buffers pequenos, overflow e bad_alloc.
- `tests/ac_factor_memory_contract.cpp` e
  `tests/analytic_precision/ac_factor_storage_contract.py`: comparação de
  todos os campos do fator em 441 desenhos/sementes/splits por braço, incluindo
  o despacho real acima de 2^18. Um bit alterado tem de ser rejeitado.
- ASan/UBSan/LSAN: libertação prematura tem de produzir erro; a implementação
  válida tem de terminar sem acessos inválidos ou fugas. Falta de runtime ou
  restrição de ptrace é NOT_RUN nesse contexto, nunca PASS.
- A bateria analítica passa a fixar também o hash do novo header. A omissão
  do produtor anterior foi um *evaluation-harness error*, corrigido apenas
  depois de concluir e reportar as duas campanhas, cujos raws/manifestos
  permanecem imutáveis. Os pins próprios do patch e dos fatores cobrem o
  header nessas campanhas. Um manifesto com o hash do header corrompido é
  rejeitado antes do primeiro fit.

O diagnóstico corrigiu também uma premissa do plano: simulated usa
Gauss–Seidel por omissão e não chama a arena. A classificação foi
*evaluation-harness error*; conservou-se o método do estimador.

## Evidência e limites

Fontes, builds, hashes, diagnósticos, oráculos e recibos ficam em
[n2_h2](../Verifications/perf_max_20260929/implementacao_bloqueios_20261007/n2_h2/).
O [estado de execução](../Verifications/perf_max_20260929/implementacao_bloqueios_20261007/n2_h2/STATUS.md)
regista os gates ainda pendentes. Os excessos de memória ligados a N1,
falhas T01 herdadas, interfaces não ensaiadas e plataformas NOT_RUN mantêm
classificações próprias. Paridade entre fatores não é um oráculo do
estimador completo nem evidência de desempenho.
